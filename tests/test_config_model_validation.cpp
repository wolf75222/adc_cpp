// ADC-299 + ADC-290 : la CONFIGURATION et le MODELE sont valides EN AMONT, sans defaut silencieux.
//
// ADC-299 (validation de config avant construction interne) : une SystemConfig / AmrSystemConfig
//   invalide (n <= 0, L <= 0, regrid_every < 0, coarse_max_grid < 0) est REJETEE avant que l'Impl
//   n'alloue quoi que ce soit. Pour System c'est crucial : Impl(c) derive la geometrie, le BoxArray,
//   le DistributionMapping et alloue l'aux MultiFab -- tous dimensionnes par c.n -- AVANT l'ancien
//   check_geometry. Un n=0 / L=0 ne plantait pas : il construisait une grille degeneree silencieuse
//   (boite vide, dx = L/0 = +inf ou dx negatif) qui ne se manifestait que loin en aval. On asserte le
//   refus immediat (pas un etat degenere) ET qu'une config valide construit toujours.
//
// ADC-290 (modele explicite, pas de retombee physique silencieuse) : un ModelSpec dont transport ou
//   elliptic n'est pas pose ECHOUE clairement -- AUCUN fallback vers compressible / charge. On verifie
//   le contrat directement (detail::validate_model_spec), via la surface utilisateur
//   System::add_block / AmrSystem::add_block (le message clair precede le routage par chaine), et que
//   le message NOMME le champ manquant (lisibilite). Un modele COMPLET reste accepte.
//
// Tests de CONTRAT (aucun calcul) : throws cibles + cas valides qui construisent. Compile
// python/system.cpp et python/amr_system.cpp (objets runtime splices, cf. tests/CMakeLists.txt).

#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/factory/model_factory.hpp>  // detail::validate_model_spec (contrat de completude)
#include <pops/runtime/config/model_spec.hpp>
#include <pops/runtime/system.hpp>

#include "test_harness.hpp"  // pops::test::Checker (style verbose) + raises partages

#include <cstdio>
#include <stdexcept>
#include <string>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;
using pops::test::raises;  // true si l'appelable leve std::runtime_error (le refus attendu)

namespace {

// true si @p f leve un std::runtime_error DONT le message contient @p frag : on ne se contente pas du
// refus, on verifie que c'est le BON refus (le champ manquant nomme), donc un message lisible.
template <class F>
bool raises_with(F&& f, const std::string& frag) {
  try {
    f();
  } catch (const std::runtime_error& e) {
    return std::string(e.what()).find(frag) != std::string::npos;
  } catch (...) {
    return false;
  }
  return false;
}

// Modele natif COMPLET (diocotron scalaire) : transport + source + elliptic poses explicitement.
ModelSpec exb_charge() {
  ModelSpec s;
  s.transport = "exb";
  s.source = "none";
  s.elliptic = "charge";
  return s;
}

}  // namespace

int main(int argc, char** argv) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  pops::test::Checker checker{pops::test::Checker::Style::Verbose};  // imprime [OK ]/[XX ] par ligne
  auto chk = [&](bool c, const char* w) { checker(c, w); };

  // ============================================================================================
  // ADC-299 : SystemConfig invalide rejetee AVANT la construction de Impl (allocation geom/ba/dm/aux).
  // ============================================================================================
  chk(raises_with([&] { System s(SystemConfig{0, 1.0, false}); }, "n >= 1"),
      "System(n=0) rejete avant Impl (n >= 1)");
  chk(raises_with([&] { System s(SystemConfig{-4, 1.0, false}); }, "n >= 1"), "System(n<0) rejete");
  chk(raises_with([&] { System s(SystemConfig{16, 0.0, false}); }, "L > 0"),
      "System(L=0) rejete (L > 0)");
  chk(raises_with([&] { System s(SystemConfig{16, -1.0, false}); }, "L > 0"), "System(L<0) rejete");
  // Une config valide CONSTRUIT toujours (le garde-fou ne sur-rejette pas).
  {
    bool ok = false;
    try {
      System s(SystemConfig{16, 1.0, false});
      ok = (s.nx() == 16);
    } catch (...) {
      ok = false;
    }
    chk(ok, "System config valide construit (nx == 16)");
  }

  // ============================================================================================
  // ADC-299 : AmrSystemConfig invalide rejetee AVANT Impl (parite avec System).
  // ============================================================================================
  chk(raises_with([&] { AmrSystem a(AmrSystemConfig{0}); }, "n >= 1"),
      "AmrSystem(n=0) rejete (n >= 1)");
  {
    AmrSystemConfig c;
    c.n = 32;
    c.L = 0.0;
    chk(raises_with([&] { AmrSystem a(c); }, "L > 0"), "AmrSystem(L=0) rejete (L > 0)");
  }
  {
    AmrSystemConfig c;
    c.n = 32;
    c.regrid_every = -1;
    chk(raises_with([&] { AmrSystem a(c); }, "regrid_every"), "AmrSystem(regrid_every<0) rejete");
  }
  {
    AmrSystemConfig c;
    c.n = 32;
    c.coarse_max_grid = -1;
    chk(raises_with([&] { AmrSystem a(c); }, "coarse_max_grid"),
        "AmrSystem(coarse_max_grid<0) rejete");
  }
  {
    bool ok = false;
    try {
      AmrSystem a(AmrSystemConfig{32});
      ok = (a.nx() == 32);
    } catch (...) {
      ok = false;
    }
    chk(ok, "AmrSystem config valide construit (nx == 32)");
  }

  // ============================================================================================
  // ADC-290 : ModelSpec incomplet ECHOUE clairement -- aucune retombee compressible / charge.
  // ============================================================================================
  // (a) Contrat direct : transport non pose, puis elliptic non pose, puis source vide -> message nomme
  //     le champ manquant. Un ModelSpec defaut-construit n'est plus Euler + charge implicite.
  chk(raises_with([&] { detail::validate_model_spec(ModelSpec{}); }, "transport"),
      "validate_model_spec : transport non pose rejete, message nomme 'transport'");
  {
    ModelSpec only_tr;
    only_tr.transport = "exb";  // elliptic encore vide
    chk(raises_with([&] { detail::validate_model_spec(only_tr); }, "elliptic"),
        "validate_model_spec : elliptic non pose rejete, message nomme 'elliptic'");
  }
  {
    ModelSpec no_src;
    no_src.transport = "exb";
    no_src.elliptic = "charge";
    no_src.source = "";  // source explicitement videe
    chk(raises_with([&] { detail::validate_model_spec(no_src); }, "source"),
        "validate_model_spec : source vide rejetee, message nomme 'source'");
  }
  // Un modele COMPLET passe le contrat (le garde-fou ne sur-rejette pas).
  chk(!raises([&] { detail::validate_model_spec(exb_charge()); }),
      "validate_model_spec : modele complet (exb/none/charge) accepte");

  // (b) Surface utilisateur : le contrat s'applique a l'entree de System::add_block, AVANT le routage
  //     par chaine sur model.transport (qui dirait sinon "unknown transport ''"). Le defaut-construit
  //     ne devient JAMAIS un Euler silencieux.
  chk(raises_with(
          [&] {
            System s(SystemConfig{16, 1.0, false});
            s.add_block("m", ModelSpec{});
          },
          "transport"),
      "System::add_block(ModelSpec incomplet) rejete -- pas de transport 'compressible' "
      "silencieux");
  // Un modele complet s'installe (chemin natif ExB scalaire complet, sans lever).
  chk(!raises([&] {
    System s(SystemConfig{16, 1.0, false});
    s.add_block("ne", exb_charge());
  }),
      "System::add_block(modele complet) accepte");

  // (c) Meme contrat a l'entree de AmrSystem::add_block (parite). add_block est paresseux : le refus
  //     tombe au contrat, sans declencher le build de la hierarchie.
  chk(raises_with(
          [&] {
            AmrSystem a(AmrSystemConfig{16});
            a.add_block("m", ModelSpec{});
          },
          "transport"),
      "AmrSystem::add_block(ModelSpec incomplet) rejete -- pas de fallback silencieux");

  // ============================================================================================
  // ADC-331 : completude du routage. Chaque tag builtin de la registry (model_registry.hpp) DOIT
  // etre route par le dispatch -- une ligne de table sans branche if = une derive registry/dispatch
  // (validate_transport / validate_elliptic acceptent le tag, mais la chaine if/else tombe sur le
  // garde "valid in registry but not routed"). Visiteur no-op : on verifie seulement que le dispatch
  // ATTEINT une branche, sans construire le CompositeModel complet. Pendant runtime du static_assert
  // de non-derive n_vars (model_factory.hpp) : ici on verrouille l'EXHAUSTIVITE du routage.
  {
    bool all_tr = true;
    for (const TransportTag& t : kTransports) {
      ModelSpec s;
      s.transport = t.name;
      bool routed = false;
      try {
        detail::dispatch_transport(s, [&](auto) { routed = true; });
      } catch (...) {
        routed = false;
      }
      all_tr = all_tr && routed;
    }
    chk(all_tr, "ADC-331 : tout transport builtin de la registry est route (pas de derive)");

    bool all_el = true;
    for (const EllipticTag& t : kElliptics) {
      ModelSpec s;
      s.elliptic = t.name;
      bool routed = false;
      try {
        detail::dispatch_elliptic(s, [&](auto) { routed = true; });
      } catch (...) {
        routed = false;
      }
      all_el = all_el && routed;
    }
    chk(all_el, "ADC-331 : tout elliptic builtin de la registry est route (pas de derive)");

    // dispatch_source est templatise sur NV et garde les forces fluides derriere `if constexpr
    // (NV >= 3)` ; a NV=4 (Euler) les sept orthographes builtin routent. Une ligne kSources ajoutee
    // sans branche dans dispatch_source ferait echouer ce passage (meme garde de derive que pour
    // transport / elliptic, etendue a l'axe source que le if/else NV-dependant rend moins evident).
    bool all_src = true;
    for (const SourceTag& t : kSources) {
      ModelSpec s;
      s.source = t.name;
      bool routed = false;
      try {
        detail::dispatch_source<4>(s, [&](auto) { routed = true; });
      } catch (...) {
        routed = false;
      }
      all_src = all_src && routed;
    }
    chk(all_src, "ADC-331 : tout source builtin de la registry route a NV=4 (pas de derive)");
  }

  const int rc = checker.failed();  // 0 si aucun echec, 1 sinon
  std::printf(rc == 0 ? "OK test_config_model_validation\n"
                      : "test_config_model_validation : ECHEC\n");
  return rc;
}
