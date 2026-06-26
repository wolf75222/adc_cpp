// Source UNIQUE de la disposition aux (X-macro POPS_AUX_FIELDS, adc/core/state.hpp). Ce test
// prouve que la GENERATION couvre TOUS les champs extra et ferme le trou #51 (T_e oublie dans
// le marshaling) : un champ ne peut plus etre lu sur un chemin et muet sur l'autre, car les deux
// derivent de la meme table. On verifie a la compilation ET a l'execution :
//
//   (A) CHEMIN DEVICE (load_aux<NComp>) : pour CHAQUE champ de POPS_AUX_FIELDS, on remplit la
//       SEULE composante a son indice et on verifie que load_aux la depose dans le BON membre
//       de Aux. La boucle est elle-meme generee par POPS_AUX_FIELDS -> un nouveau champ ajoute a
//       la table est teste automatiquement, sans toucher ce fichier.
//   (B) CHEMIN MARSHALING (host) : la disposition composante-majeur AUX[idx*nn + k] reproduite
//       depuis la MEME table donne, apres load via la table, exactement les memes valeurs ->
//       impossible d'avoir un champ lu cote device mais oublie cote host (le bug #51).
//   (C) RETRO-COMPAT : load_aux<kAuxBaseComps> (defaut) n'ecrit AUCUN champ extra (tous a 0),
//       quelle que soit la valeur parasite des composantes >= 3.
//
// Aucun parallelisme ici : on appelle load_aux sur un Fab2D mono-box, c'est le coeur de la
// lecture device, mais le test reste pur hote (CPU).

#include <pops/core/state/state.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/numerics/spatial_operator.hpp>

#include <cstdio>
#include <vector>

using namespace pops;

// Nombre de champs EXTRA et largeur aux totale, DERIVES de la table (et non codes en dur).
static constexpr int kNExtra = [] {
  int n = 0;
#define POPS_AUX_COUNT(name, idx) ++n;
  POPS_AUX_FIELDS(POPS_AUX_COUNT)
#undef POPS_AUX_COUNT
  return n;
}();
// Largeur necessaire pour lire TOUS les champs extra : max(idx)+1 (>= kAuxBaseComps).
static constexpr int kFullWidth = [] {
  int w = kAuxBaseComps;
#define POPS_AUX_WIDTH(name, idx) w = (idx) + 1 > w ? (idx) + 1 : w;
  POPS_AUX_FIELDS(POPS_AUX_WIDTH)
#undef POPS_AUX_WIDTH
  return w;
}();

// Accesseur membre genere : permet de lire a.<name> via une fonction indexee par la table,
// donc de boucler sur les champs sans citer leurs noms a la main dans le corps du test.
static Real aux_member(const Aux& a, int idx) {
  Real v = 0;
#define POPS_AUX_GET(name, ix) \
  if ((idx) == (ix))          \
    v = a.name;
  POPS_AUX_FIELDS(POPS_AUX_GET)
#undef POPS_AUX_GET
  return v;
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  std::printf("  table POPS_AUX_FIELDS : %d champ(s) extra, largeur pleine = %d\n", kNExtra,
              kFullWidth);
  chk(kNExtra >= 1, "table_non_vide");
  chk(kFullWidth >= kAuxBaseComps + 1, "largeur_pleine_coherente");

  // Fab mono-box 1x1, kFullWidth composantes : canal aux complet pour une cellule.
  const Box2D b = Box2D::from_extents(1, 1);
  Fab2D fab(b, kFullWidth, 0);
  const Array4 w = fab.array();
  const ConstArray4 a = fab.const_array();

  // --- (A) chemin device : chaque champ extra est lu au bon membre ---
  // On met une valeur DISTINCTE par composante extra, puis on verifie pour chaque champ que
  // load_aux<kFullWidth> a depose AUX[idx] dans aux_member(.,idx) (i.e. le bon membre nomme).
  for (int c = 0; c < kFullWidth; ++c)
    w(0, 0, c) = Real(100 + c);  // 100,101,... distincts
  {
    const Aux x = load_aux<kFullWidth>(a, 0, 0);
    chk(x.phi == Real(100) && x.grad_x == Real(101) && x.grad_y == Real(102), "base_phi_grad_lus");
#define POPS_AUX_CHECK_READ(name, idx) \
  chk(aux_member(x, idx) == Real(100 + (idx)), "device_lit_" #name "_au_bon_indice");
    POPS_AUX_FIELDS(POPS_AUX_CHECK_READ)
#undef POPS_AUX_CHECK_READ
  }

  // --- (B) chemin marshaling host : meme table -> memes valeurs que load_aux ---
  // Reproduit la disposition composante-majeur AUX[idx*nn + k] (nn=1, k=0) du marshaling de
  // python/system.cpp, GENEREE depuis POPS_AUX_FIELDS, et compare champ par champ a load_aux.
  {
    const std::size_t nn = 1, k = 0;
    std::vector<double> AUX(static_cast<std::size_t>(kFullWidth));
    for (int c = 0; c < kFullWidth; ++c)
      AUX[static_cast<std::size_t>(c) * nn + k] = 100 + c;
    Aux host{};
    host.phi = AUX[k];
    host.grad_x = AUX[nn + k];
    host.grad_y = AUX[2 * nn + k];
#define POPS_AUX_MARSHAL(name, idx)    \
  if (AUX.size() >= ((idx) + 1) * nn) \
    host.name = AUX[(idx) * nn + k];
    POPS_AUX_FIELDS(POPS_AUX_MARSHAL)
#undef POPS_AUX_MARSHAL
    const Aux dev = load_aux<kFullWidth>(a, 0, 0);
    chk(host.phi == dev.phi && host.grad_x == dev.grad_x && host.grad_y == dev.grad_y,
        "host_base_egal_device");
#define POPS_AUX_CHECK_EQ(name, idx) chk(host.name == dev.name, "host_egal_device_" #name);
    POPS_AUX_FIELDS(POPS_AUX_CHECK_EQ)
#undef POPS_AUX_CHECK_EQ
  }

  // --- (C) retro-compat : largeur de base n'ecrit aucun champ extra ---
  for (int c = 0; c < kFullWidth; ++c)
    w(0, 0, c) = Real(999);  // tout parasite
  {
    const Aux x = load_aux<kAuxBaseComps>(a, 0, 0);
    chk(x.phi == Real(999) && x.grad_x == Real(999) && x.grad_y == Real(999), "base_lit_phi_grad");
#define POPS_AUX_CHECK_ZERO(name, idx) chk(aux_member(x, idx) == Real(0), "base_ignore_" #name);
    POPS_AUX_FIELDS(POPS_AUX_CHECK_ZERO)
#undef POPS_AUX_CHECK_ZERO
  }

  if (fails == 0)
    std::printf("OK test_aux_single_source\n");
  return fails == 0 ? 0 : 1;
}
