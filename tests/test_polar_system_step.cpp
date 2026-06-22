// Pas COUPLE POLAIRE (chantier "grille polaire diocotron", Phase 2b) : transport -> Poisson -> aux ->
// avance, AU NIVEAU C++, sur un anneau global (r, theta). C'est le pendant C++ (rapide, sans Python) du
// chemin cable dans System::step pour geometry == "polar". Il exerce et valide les MEMES briques que
// System::step branche :
//
//   (1) PolarPoissonSolver : f = q n (charge), resolu sur l'anneau (FFT-en-theta + tridiag-en-r) ;
//   (2) DERIVATION AUX EN BASE LOCALE (e_r, e_theta), exactement comme System::solve_fields_polar :
//         aux[1] = grad_r     = d phi/dr,
//         aux[2] = grad_theta = (1/r) d phi/d theta  (derivee PHYSIQUE, deja divisee par r),
//       d'ou la vitesse ExB polaire de ExBVelocityPolar : v_r = -grad_theta/B, v_theta = grad_r/B ;
//   (3) AVANCE SSPRK3 du transport polaire (assemble_rhs_polar) avec PAROI RADIALE solide (wall_radial)
//       -> flux radial nul a r_min/r_max -> masse Sum_ij n_ij r_i dr dtheta conservee A LA MACHINE.
//
// Deux verifications :
//   (A) PAS COUPLE NON TRIVIAL : un cran de pas couple (Poisson sur la densite courante, aux derive,
//       transport) modifie reellement la densite (le champ n'est pas gele) ET la vitesse ExB est bien
//       a divergence ~nulle dans la metrique (sanity : le pas reste borne, n > 0).
//   (B) CONSERVATION DE MASSE a la machine sur K pas couples (paroi radiale solide). C'est la propriete
//       que System::step doit garantir (test 4 du livrable).
//
// Host / Serial-safe : UNE box couvrant l'anneau, n_ranks()==1 dans les 3 jobs CI (PolarPoissonSolver
// leve proprement sous MPI ; ce test n'est pas enregistre MPI, comme test_polar_poisson_mms).

#include <adc/core/state.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/elliptic/polar/polar_poisson_solver.hpp>
#include <adc/numerics/numerical_flux.hpp>
#include <adc/numerics/reconstruction.hpp>
#include <adc/numerics/spatial_operator_polar.hpp>
#include <adc/numerics/time/time_steppers.hpp>
#include <adc/physics/bricks.hpp>  // ExBVelocityPolar, CompositeModel, NoSource, ChargeDensity
#include <adc/runtime/builders/block_builder_polar.hpp>  // derive_aux_polar : MEME derivation aux que System::solve_fields_polar

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;

static constexpr double kPiL = 3.14159265358979323846;
static constexpr double kRmin = 0.30;
static constexpr double kRmax = 1.00;
static constexpr double kB0 = 1.0;
static constexpr double kQ = 1.0;  // charge (f = q n)

// Modele compose IDENTIQUE a celui que System::add_block bati en polaire pour un bloc ExB scalaire de
// charge : transport ExB polaire + pas de source + second membre elliptique = charge q n.
using PolarModel = CompositeModel<ExBVelocityPolar, NoSource, ChargeDensity>;

// Masse FV polaire Sum_ij n_ij r_i dr dtheta sur les cellules valides (quantite CONSERVEE).
static double total_mass(const MultiFab& U, const PolarGeometry& g, const Box2D& dom) {
  sync_host();
  const ConstArray4 u = U.fab(0).const_array();
  const double dr = g.dr(), dth = g.dtheta();
  double m = 0.0;
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
      m += u(i, j, 0) * g.r_cell(i) * dr * dth;
  return m;
}

// Min de la densite (sanity : reste > 0 pendant le run). PROPAGE le nan : si une cellule est non finie
// (nan/inf, signe d'un blow-up), on RETOURNE nan -> le test echoue (sinon nan < mn est faux et le min
// resterait 1e300 -> faux positif, le piege exact qui faisait passer un run divergent).
static double min_density(const MultiFab& U, const Box2D& dom) {
  sync_host();
  const ConstArray4 u = U.fab(0).const_array();
  double mn = 1e300;
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
      const double val = u(i, j, 0);
      if (!std::isfinite(val))
        return std::nan("");
      if (val < mn)
        mn = val;
    }
  return mn;
}

// UN pas couple POLAIRE, exactement comme System::step (branche polaire) :
//   solve_fields_polar (Poisson + aux en base locale) PUIS avance SSPRK3 du transport (paroi radiale).
static void coupled_step(const PolarModel& model, MultiFab& U, MultiFab& aux,
                         PolarPoissonSolver& solver, const PolarGeometry& g, const Box2D& dom,
                         const BCRec& bc, double dt) {
  // --- solve_fields_polar : f = q n, resolu, puis aux = (phi, grad_r, grad_theta) ---
  {
    MultiFab& rhs = solver.rhs();
    rhs.set_val(0.0);
    Array4 r = rhs.fab(0).array();
    const ConstArray4 u = U.fab(0).const_array();
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        r(i, j, 0) += model.elliptic_rhs(load_state<PolarModel>(u, i, j));  // q n
    solver.solve();
    // Derivation (phi, grad_r, grad_theta) en base locale via le MEME helper que System::solve_fields_polar
    // (radial DECENTRE aux parois, theta ENROULE periodique : phi est sans ghost). Exercer la production.
    derive_aux_polar(solver.phi(), aux, g);
    fill_ghosts(aux, dom, bc);  // theta periodique, r physique (extrapolation)
  }
  // --- avance SSPRK3 du transport polaire avec PAROI RADIALE solide (wall_radial = true) ---
  SSPRK3Step{}.take_step(
      [&](MultiFab& stage, MultiFab& R) {
        fill_ghosts(stage, dom, bc);
        assemble_rhs_polar<Weno5, RusanovFlux>(model, stage, aux, g, R, /*recon_prim=*/false,
                                               /*wall_radial=*/true);
      },
      U, static_cast<Real>(dt));
}

int main() {
  std::printf("=== Pas COUPLE POLAIRE (transport -> Poisson -> aux -> avance), Phase 2b ===\n");
  std::printf("Anneau r in [%.2f, %.2f], theta in [0, 2pi), B0=%.1f, q=%.1f\n", kRmin, kRmax, kB0,
              kQ);

  const int nr = 64, nth = 64;
  Box2D dom = Box2D::from_extents(nr, nth);
  PolarGeometry g{dom, kRmin, kRmax};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, n_ranks());

  // BC : radial Neumann homogene (Foextrap) pour le Poisson (paroi), theta periodique. (La paroi
  // SOLIDE du transport est portee par wall_radial dans coupled_step, independamment de la BC du
  // Poisson : le test verifie precisement que la masse est conservee a la machine grace a wall_radial.)
  BCRec bc;
  bc.xlo = bc.xhi = BCType::Foextrap;
  bc.ylo = bc.yhi = BCType::Periodic;

  PolarModel model{ExBVelocityPolar{Real(kB0)}, NoSource{}, ChargeDensity{Real(kQ)}};

  const int ng = Weno5::n_ghost;
  MultiFab U(ba, dm, 1, ng);
  MultiFab aux(ba, dm, 3, ng);
  U.set_val(0.0);
  aux.set_val(0.0);

  // Profil de densite annulaire lisse, strictement positif, module en theta (asymetrie -> Poisson non
  // trivial -> grad_theta != 0 -> v_r != 0 a l'interieur : le pas couple EXERCE le terme radial).
  {
    Array4 u = U.fab(0).array();
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
        const double r = g.r_cell(i), th = g.theta_cell(j);
        const double rr = (r - kRmin) / (kRmax - kRmin);
        u(i, j, 0) = 1.0 + 0.3 * std::cos(2.0 * th) * std::sin(kPiL * rr);
      }
  }

  PolarPoissonSolver solver(g, ba, bc);

  const double m0 = total_mass(U, g, dom);
  const double minrho0 = min_density(U, dom);

  // dt sous la CFL azimutale (vitesse de derive ~O(1), pas physique min = r_min * dtheta).
  const double ds_min = kRmin * g.dtheta();
  const double dt = 0.3 * ds_min;  // vitesse caracteristique O(1)
  const int nsteps = 40;

  // (A) Un pas couple modifie reellement la densite (champ non gele).
  MultiFab U0(ba, dm, 1, ng);
  U0.set_val(0.0);
  {
    sync_host();
    Array4 u0 = U0.fab(0).array();
    const ConstArray4 u = U.fab(0).const_array();
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        u0(i, j, 0) = u(i, j, 0);
  }
  coupled_step(model, U, aux, solver, g, dom, bc, dt);
  double dmax = 0.0;
  {
    sync_host();
    const ConstArray4 u = U.fab(0).const_array();
    const ConstArray4 u0 = U0.fab(0).const_array();
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        dmax = std::max(dmax, std::fabs(u(i, j, 0) - u0(i, j, 0)));
  }
  std::printf("\n--- (A) Pas couple non trivial ---\n");
  std::printf("  variation max de densite apres 1 pas : %.4e\n", dmax);
  const double minrho_A = min_density(U, dom);  // nan si blow-up

  bool ok = true;
  if (!std::isfinite(dmax) || !std::isfinite(minrho_A)) {
    std::printf(
        "  ECHEC : champ non fini apres 1 pas (blow-up : Poisson/aux/transport instable)\n");
    ok = false;
  } else if (!(dmax > 1e-9)) {
    std::printf(
        "  ECHEC : le pas couple ne modifie pas la densite (Poisson/aux/transport inertes ?)\n");
    ok = false;
  } else if (dmax > 1.0) {
    // Borne de STABILITE : a la CFL choisie (~0.3) un pas WENO5/SSPRK3 ne change une cellule que de
    // O(CFL * variation locale) ~ 0.1 ; une variation > 1.0 (densite initiale ~1) signe une divergence.
    // C'est ce garde qui rattrape le blow-up "fini mais enorme" (135) qui passait avant le fix.
    std::printf("  ECHEC : variation %.4e > 1.0 apres 1 pas = instabilite (gradient/CFL faux ?)\n",
                dmax);
    ok = false;
  } else if (!(minrho_A > 0.0)) {
    std::printf("  ECHEC : densite <= 0 apres 1 pas\n");
    ok = false;
  } else {
    std::printf(
        "  OK : le pas couple advecte reellement la densite (variation finie, bornee, positive)\n");
  }

  // (B) Conservation de masse a la machine sur K pas couples (paroi radiale solide).
  std::printf("\n--- (B) Conservation de masse sur %d pas couples (paroi radiale solide) ---\n",
              nsteps);
  for (int s = 1; s < nsteps; ++s)
    coupled_step(model, U, aux, solver, g, dom, bc, dt);
  const double m1 = total_mass(U, g, dom);
  const double minrho1 = min_density(U, dom);
  const double rel = std::fabs(m1 - m0) / std::fabs(m0);
  std::printf("  masse initiale=%.15e finale=%.15e  ecart relatif=%.3e\n", m0, m1, rel);
  std::printf("  densite min initiale=%.4e finale=%.4e (sanity : reste positive)\n", minrho0,
              minrho1);
  // GARDE anti-faux-positif : nan/inf doit FAIRE ECHOUER (nan > 1e-12 est faux en C++ -> sinon un run
  // divergent passerait silencieusement, exactement le bug attrape ici).
  if (!std::isfinite(m1) || !std::isfinite(rel) || !std::isfinite(minrho1)) {
    std::printf("  ECHEC : masse/densite non finie apres %d pas (blow-up)\n", nsteps);
    ok = false;
  } else if (rel > 1e-12) {
    // Tolerance machine elargie (accumulation sur K pas x 3 etages SSPRK3, derivation aux, solve hote).
    std::printf("  ECHEC : ecart de masse %.3e > 1e-12 (paroi radiale non conservative)\n", rel);
    ok = false;
  } else if (!(minrho1 > 0.0)) {
    std::printf("  ECHEC : densite devenue negative (pas couple instable)\n");
    ok = false;
  } else {
    std::printf("  OK : masse conservee a ~machine (%.3e <= 1e-12) et densite positive\n", rel);
  }

  std::printf("\n=== VERDICT : %s ===\n", ok ? "SUCCESS" : "ECHEC");
  if (ok)
    std::printf("OK test_polar_system_step\n");
  return ok ? 0 : 1;
}
