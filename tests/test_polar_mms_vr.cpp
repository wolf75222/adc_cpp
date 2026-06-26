// MMS POLAIRE DEDIE avec VITESSE RADIALE v_r != 0 (Lot A4). Verifie quantitativement l'ORDRE DE
// CONVERGENCE SPATIAL du transport polaire complet (assemble_rhs_polar + SSPRK3) quand le FLUX RADIAL
// traverse reellement les faces radiales avec la metrique 1/r. Les tests polaires existants couvrent : la
// convergence de la DIVERGENCE seule sur un instantane (test_polar_transport_mms, partie A), la
// conservation FV (parties B), et un pas couple non trivial (test_polar_system_step). AUCUN ne fait une
// AVANCE EN TEMPS jusqu'a un t court avec un TERME SOURCE manufacture et n'asserte l'ordre de l'erreur L2
// de la SOLUTION vs une solution exacte sous un champ a v_r != 0. Ce lot comble ce trou : il prouve que le
// transport RADIAL (et pas seulement azimutal) converge a l'ordre attendu.
//
// PROBLEME RESOLU : d_t rho + div(rho v) = S, avec div polaire = (1/r) d_r(r rho v_r) +
//                   (1/r) d_theta(rho v_theta), sur l'anneau r in [r_min, r_max], theta in [0, 2pi).
//
// SOLUTION MANUFACTUREE STATIONNAIRE (lisse, strictement positive, periodique en theta, v_r != 0) :
//   rho(r, theta) = (1 + 0.5 sin(pi (r-r_min)/(r_max-r_min))) * (2 + cos(m theta))
//   VITESSE CONSTANTE (uniforme en espace) :
//     v_r     = a  (constant, NON NUL)  -> exerce le terme radial (1/r) d_r(r rho v_r)
//     v_theta = b  (constant, NON NUL)  -> exerce le terme azimutal
//   Vitesse constante => la reconstruction de la VITESSE a la face est EXACTE (la valeur de cellule
//   coincide avec la valeur de face), donc l'ordre observe reflete l'operateur de transport lui-meme
//   (et non un plafonnement par une vitesse reconstruite a l'ordre 2). v_r != 0 est le point central :
//   le flux radial pondere par r aux faces (et 1/r en cellule) est reellement exerce.
//
// POURQUOI UNE SOLUTION STATIONNAIRE (d_t rho = 0) : c'est l'isolation la plus PROPRE de l'ordre SPATIAL.
//   - Le source S = div(rho_exact v) et les ghosts radiaux (Dirichlet-MMS exact) sont INDEPENDANTS DU
//     TEMPS -> aucun confond temporel a re-evaluer par etage SSPRK3, le RHS d'un etage est EXACTEMENT le
//     bon a tout instant.
//   - rho_exact est un etat stationnaire de la PDE continue (d_t rho = -div(rho v) + S = 0). L'integration
//     en temps amene la solution DISCRETE vers son etat stationnaire DISCRET ; l'erreur vs l'exact CONTINU
//     au temps final est EXACTEMENT la troncature SPATIALE du schema (l'erreur temporelle SSPRK3 ne touche
//     QUE le transitoire, qui s'amortit ; au stationnaire elle est nulle). On mesure donc l'ordre SPATIAL
//     PUR, sans avoir a raffiner dt en dr^2 (cout O(n) au lieu de O(n^2)).
//
// TERME SOURCE MANUFACTURE : S = (1/r) d_r(r rho v_r) + (1/r) d_theta(rho v_theta), evalue sur les formes
// CLOSES par des stencils centraux d'ordre 4 a pas infinitesimal (h ~ 1e-5) -> reference a ~10-12 chiffres,
// tres en dessous de l'erreur de grille (h ~ 1/48..1/192) : l'erreur mesuree reflete la troncature du
// SCHEMA, pas celle du calcul du source. S est injecte dans le solveur via une brique de transport polaire
// LOCALE AU TEST qui expose source(u, aux) en relisant S depuis un canal aux EXTRA (index 3, reutilise
// B_z) : aucun code de PRODUCTION n'est ajoute (la brique vit dans ce fichier ; load_aux charge deja
// l'index 3 si n_aux>=4, cf. POPS_AUX_FIELDS).
//
// CONDITIONS AUX LIMITES : theta PERIODIQUE ; r DIRICHLET-MMS (les ghosts radiaux sont remplis avec la
// solution EXACTE), pratique standard du MMS : le bord n'introduit pas d'erreur propre, l'ordre mesure est
// celui du schema interieur (limite par la quadrature de face cartesienne, cf. ordre formel 2 de la
// divergence FV polaire documente dans test_polar_transport_mms).
//
// ORDRE ATTENDU : la divergence FV polaire est formellement d'ordre 2 (ponderation par le rayon de face
// -> telescopage haut ordre brise ; cf. test_polar_transport_mms). On MESURE l'ordre reel (tableau err vs
// n + pente) et on asserte un seuil CALIBRE un peu en dessous du mesure (factuel). Une metrique radiale
// (1/r en cellule, r aux faces) erronee donnerait un ordre 0 ou une divergence : l'ordre 2 PROUVE que le
// transport RADIAL (v_r != 0) converge correctement, ce que ce lot vise.
//
// Host / Serial-safe (UNE box, n_ranks()==1 dans les 3 jobs CI : non enregistre MPI, comme les autres
// tests polaires mono-box).

#include <pops/core/state/state.hpp>
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/numerics/fv/reconstruction.hpp>
#include <pops/numerics/spatial/operators/polar_operator.hpp>
#include <pops/numerics/time/integrators/time_steppers.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace pops;

static constexpr double kPiL = 3.14159265358979323846;
static constexpr double kRmin = 0.30;
static constexpr double kRmax = 1.00;
static constexpr double kB0 = 1.0;
static constexpr int kMode = 2;  // mode azimutal m de la solution manufacturee
// Vitesse CONSTANTE non nulle. v_r != 0 est l'objet du lot : le flux radial est reellement exerce.
static constexpr double kVr = 0.35;  // vitesse radiale (NON NULLE)
static constexpr double kVth = 0.6;  // vitesse azimutale (NON NULLE)
static constexpr double kTfinal =
    0.30;  // temps d'avance court (amortit le transitoire vers le stationnaire)

// --- Solution exacte STATIONNAIRE (forme close, lisse, strictement positive, periodique en theta) ----
static double f_r(double r) {
  const double rr = (r - kRmin) / (kRmax - kRmin);
  return 1.0 + 0.5 * std::sin(kPiL * rr);
}
static double rho_exact(double r, double th) {
  return f_r(r) * (2.0 + std::cos(kMode * th));
}

// --- Vitesse ExB polaire encodee dans aux : la brique lit v_r = -grad_theta/B, v_theta = grad_r/B.
// Pour imposer (v_r, v_theta) = (kVr, kVth) constants : grad_theta = -B kVr, grad_r = B kVth.
static double aux_grad_r() {
  return kB0 * kVth;
}  // -> v_theta = grad_r / B = kVth
static double aux_grad_theta() {
  return -kB0 * kVr;
}  // -> v_r = -grad_theta / B = kVr

// --- Terme source manufacture S = (1/r) d_r(r rho v_r) + (1/r) d_theta(rho v_theta) (d_t rho = 0,
// solution stationnaire), evalue par stencils centraux d'ordre 4 a pas h ~ 1e-5 sur les formes closes
// (precision ~1e-11, negligeable vs erreur de grille). v_r, v_theta CONSTANTS -> r rho v_r et rho v_theta
// sont des produits simples des formes closes.
static double mms_source(double r, double th) {
  const double h = 1e-5;
  // (1/r) d_r(r rho v_r), v_r constant
  auto rRhoVr = [](double rr, double tt) { return rr * rho_exact(rr, tt) * kVr; };
  const double drr = (-rRhoVr(r + 2 * h, th) + 8 * rRhoVr(r + h, th) - 8 * rRhoVr(r - h, th) +
                      rRhoVr(r - 2 * h, th)) /
                     (12 * h);
  // (1/r) d_theta(rho v_theta), v_theta constant
  const double dth = (-rho_exact(r, th + 2 * h) + 8 * rho_exact(r, th + h) -
                      8 * rho_exact(r, th - h) + rho_exact(r, th - 2 * h)) /
                     (12 * h) * kVth;
  return (drr + dth) / r;
}

// --- Brique de transport polaire LOCALE AU TEST : ExB polaire (vitesse depuis aux[1]/aux[2]) + un terme
// source manufacture relu depuis le canal aux EXTRA d'index 3 (aux.B_z, reutilise). En declarant n_aux =
// 4, load_aux charge a(i,j,3) dans Aux::B_z (cf. POPS_AUX_FIELDS) ; source() le renvoie tel quel. Cela
// injecte le source manufacture S SANS ajouter de code de production : la brique vit ici, et le solveur
// polaire (assemble_rhs_polar) appelle source() via polar_source<> (detecte par concept PolarHasSource).
struct MmsTransportPolar {
  static constexpr int n_vars = 1;
  static constexpr int n_aux = 4;  // lit phi, grad_r, grad_theta (0..2) + S au canal extra 3 (B_z)
  using State = StateVec<1>;
  Real B0 = 1;
  POPS_HD Real velocity(const Aux& a, int dir) const {
    return (dir == 0) ? (-a.grad_y / B0) : (a.grad_x / B0);
  }
  POPS_HD StateVec<1> flux(const StateVec<1>& u, const Aux& a, int dir) const {
    StateVec<1> f{};
    f[0] = u[0] * velocity(a, dir);
    return f;
  }
  POPS_HD Real max_wave_speed(const StateVec<1>&, const Aux& a, int dir) const {
    const Real d = velocity(a, dir);
    return d < 0 ? -d : d;
  }
  // Terme source manufacture : relu depuis le canal aux extra 3 (B_z), prepositionne par le test.
  POPS_HD StateVec<1> source(const StateVec<1>&, const Aux& a) const {
    StateVec<1> s{};
    s[0] = a.B_z;  // S(r, theta) prepositionne dans aux[3]
    return s;
  }
};

// Remplit la densite U avec la solution exacte (cellules valides).
static void set_exact(MultiFab& U, const PolarGeometry& g, const Box2D& dom) {
  sync_host();
  Array4 u = U.fab(0).array();
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
      u(i, j, 0) = rho_exact(g.r_cell(i), g.theta_cell(j));
}

// Remplit les ghosts RADIAUX (BC Dirichlet-MMS) avec la solution exacte. Les ghosts AZIMUTAUX sont laisses
// au remplissage periodique (fill_ghosts). On ecrit toute la bande de ghosts radiaux (i < lo et i > hi)
// sur la hauteur AVEC ghosts azimutaux pour que le stencil radial voie partout l'exact.
static void fill_radial_ghosts_exact(MultiFab& U, const PolarGeometry& g, const Box2D& dom) {
  Array4 u = U.fab(0).array();
  const Box2D gb = U.fab(0).grown_box();
  for (int j = gb.lo[1]; j <= gb.hi[1]; ++j) {
    const double th = g.theta_cell(j);
    for (int i = gb.lo[0]; i <= gb.hi[0]; ++i) {
      if (i >= dom.lo[0] && i <= dom.hi[0])
        continue;                               // cellule valide : non touchee
      u(i, j, 0) = rho_exact(g.r_cell(i), th);  // ghost radial = exact (Dirichlet-MMS)
    }
  }
}

// Pre-positionne le canal aux : vitesse CONSTANTE (grad_r, grad_theta) + source manufacture S au canal
// EXTRA 3 (B_z) sur TOUTE la boite (valides + ghosts). Statique en temps (solution stationnaire).
static void fill_aux(MultiFab& aux, const PolarGeometry& g) {
  Array4 a = aux.fab(0).array();
  const Box2D gb = aux.fab(0).grown_box();
  for (int j = gb.lo[1]; j <= gb.hi[1]; ++j)
    for (int i = gb.lo[0]; i <= gb.hi[0]; ++i) {
      const double r = g.r_cell(i), th = g.theta_cell(j);
      a(i, j, 0) = 0.0;                // phi (inutilise)
      a(i, j, 1) = aux_grad_r();       // grad_r -> v_theta = kVth (constant)
      a(i, j, 2) = aux_grad_theta();   // grad_theta -> v_r = kVr (constant, NON NUL)
      a(i, j, 3) = mms_source(r, th);  // S au canal extra 3 (relu par MmsTransportPolar::source)
    }
}

// Erreur L2 (ponderee par le volume r dr dtheta) entre la densite numerique et la solution exacte, sur les
// cellules valides.
static double l2_error(const MultiFab& U, const PolarGeometry& g, const Box2D& dom) {
  sync_host();
  const ConstArray4 u = U.fab(0).const_array();
  const double dr = g.dr(), dth = g.dtheta();
  double num = 0.0, vol = 0.0;
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
      const double rc = g.r_cell(i), th = g.theta_cell(j);
      const double e = u(i, j, 0) - rho_exact(rc, th);
      const double w = rc * dr * dth;
      num += e * e * w;
      vol += w;
    }
  return std::sqrt(num / vol);
}

// Avance la solution manufacturee jusqu'a kTfinal sur une grille nr x nth et renvoie l'erreur L2 finale.
// dt sous la CFL (dt ~ C dr, lineaire en dr) : cout O(n) en nombre de pas. La solution est STATIONNAIRE
// (source/ghosts independants du temps) -> l'erreur finale est l'erreur SPATIALE pure (le transitoire
// temporel s'amortit, l'erreur SSPRK3 ne subsiste pas au stationnaire). On part de la CI EXACTE : la
// solution discrete derive seulement de la troncature spatiale, mesuree au temps final.
template <class Limiter>
static double run_mms(int nr, int nth) {
  Box2D dom = Box2D::from_extents(nr, nth);
  PolarGeometry g{dom, kRmin, kRmax};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, n_ranks());

  // BC : theta PERIODIQUE (remplissage periodique des ghosts azimutaux), r DIRICHLET-MMS (gere a la main
  // par fill_radial_ghosts_exact). On passe Periodic en theta et Foextrap en r a fill_ghosts pour
  // l'enroulement azimutal ; les ghosts radiaux sont ECRASES ensuite par l'exact.
  BCRec bc;
  bc.xlo = bc.xhi = BCType::Foextrap;  // r : ghosts radiaux ecrases par l'exact ensuite
  bc.ylo = bc.yhi = BCType::Periodic;  // theta periodique

  const int ng = Limiter::n_ghost;
  MultiFab U(ba, dm, 1, ng);
  MultiFab aux(ba, dm, 4, ng);  // 4 composantes : phi, grad_r, grad_theta, S
  U.set_val(0.0);
  aux.set_val(0.0);

  MmsTransportPolar model;
  model.B0 = kB0;

  // Vitesse constante + source manufacture statiques (solution stationnaire) : remplis UNE fois.
  fill_aux(aux, g);
  fill_ghosts(aux, dom,
              bc);  // enroulement azimutal des composantes aux (grad, S) ; radial deja exact

  // Condition initiale = solution exacte (valides + ghosts radiaux).
  set_exact(U, g, dom);

  // Pas de temps CFL : vitesse physique max ~ max(|v_r|, |v_theta|), pas physique min = r_min * dtheta
  // (azimutal) ou dr (radial). On prend une CFL prudente sur le plus petit des deux pas physiques.
  const double dr = g.dr();
  const double ds_min = std::min(dr, kRmin * g.dtheta());
  const double v_max = std::max(std::fabs(kVr), std::fabs(kVth));
  const double dt = 0.3 * ds_min / v_max;
  const int nsteps = static_cast<int>(std::ceil(kTfinal / dt));
  const double dt_eff = kTfinal / nsteps;

  for (int s = 0; s < nsteps; ++s) {
    SSPRK3Step{}.take_step(
        [&](MultiFab& stage, MultiFab& R) {
          fill_ghosts(stage, dom, bc);  // ghosts azimutaux periodiques
          fill_radial_ghosts_exact(stage, g,
                                   dom);  // ghosts radiaux Dirichlet-MMS (exact, stationnaire)
          assemble_rhs_polar<Limiter, RusanovFlux>(model, stage, aux, g, R);
        },
        U, static_cast<Real>(dt_eff));
  }

  return l2_error(U, g, dom);
}

int main() {
  std::printf("=== MMS POLAIRE DEDIE avec vitesse radiale v_r != 0 (Lot A4) ===\n");
  std::printf("Anneau r in [%.2f, %.2f], theta in [0, 2pi), mode m=%d, B0=%.1f\n", kRmin, kRmax,
              kMode, kB0);
  std::printf(
      "Vitesse CONSTANTE : v_r=%.2f (NON NUL), v_theta=%.2f ; t_final=%.2f (stationnaire)\n", kVr,
      kVth, kTfinal);
  std::printf("Solution exacte : rho = (1+0.5 sin(pi (r-rmin)/(rmax-rmin)))*(2+cos(m theta))\n");
  std::printf("Source S = (1/r) d_r(r rho v_r) + (1/r) d_theta(rho v_theta)  (d_t rho = 0)\n");

  bool ok = true;
  const int res[3] = {48, 96, 192};
  const int nthf = 2;  // nth = 2 * nr (anneau, theta plus echantillonne)

  // --- Convergence spatiale : erreur L2 de la solution vs l'exact, v_r != 0 ---
  std::printf("\n--- Convergence MMS de la SOLUTION (transport radial + azimutal, v_r != 0) ---\n");
  double e[3];
  for (int k = 0; k < 3; ++k) {
    e[k] = run_mms<Weno5>(res[k], nthf * res[k]);
    std::printf("  WENO5  nr=%-4d nth=%-4d : L2=%.6e\n", res[k], nthf * res[k], e[k]);
  }
  const double p1 = std::log2(e[0] / e[1]);
  const double p2 = std::log2(e[1] / e[2]);
  std::printf("  ordre observe WENO5 (L2) : %.2f (48->96), %.2f (96->192)\n", p1, p2);

  // L'operateur de divergence FV polaire est formellement d'ordre 2 (ponderation par le rayon de face
  // brise le telescopage haut ordre ; cf. test_polar_transport_mms). On EXIGE donc un ordre >= 1.8 (un peu
  // sous l'ordre 2 mesure, marge pour les effets de bord). Une metrique radiale (1/r, r aux faces) erronee
  // donnerait un ordre 0 ou une divergence : l'ordre 2 PROUVE que le transport RADIAL (v_r != 0) converge
  // correctement, ce que ce lot vise.
  const double kSeuil = 1.8;
  if (!(p1 >= kSeuil) || !(p2 >= kSeuil) || !std::isfinite(e[2])) {
    std::printf(
        "  ECHEC : ordre < %.1f (transport radial v_r != 0 non convergent a l'ordre attendu)\n",
        kSeuil);
    ok = false;
  } else {
    std::printf("  OK : convergence d'ordre >= %.1f (transport radial v_r != 0 correct)\n", kSeuil);
  }

  std::printf("\n=== VERDICT : %s ===\n", ok ? "SUCCESS" : "ECHEC");
  if (ok)
    std::printf("OK test_polar_mms_vr\n");
  return ok ? 0 : 1;
}
