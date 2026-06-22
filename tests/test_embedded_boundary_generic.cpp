// ADC-327 : CONTRAT GENERIQUE embedded-boundary / level-set domain (non lie au disque).
//
// Le coeur EB est deja generique : cut_fraction / assemble_rhs_eb sont templates sur un level set
// callable Real(Real, Real) device-safe, et assemble_rhs_masked consomme un masque 0/1 quelconque.
// Le disque (detail::DiscDomain) n'est qu'UNE instance du contrat. Ce test PROUVE que le contrat
// admet un domaine NON-disque de bout en bout, via detail::HalfPlaneDomain (level set lineaire
// a*x + b*y - c, actif < 0), et verifie que le contrat lui-meme (level_set / operator() / cell_active,
// convention de signe) tient pour le disque ET le demi-plan.
//
// Assertions (vraies, pas de no-op) :
//
//   (1) CONTRAT : DiscDomain et HalfPlaneDomain satisfont le concept LevelSetDomain (static_assert,
//       diagnostics uniquement) ; operator() == level_set ; cell_active == (level_set < 0) ; convention
//       de signe (interieur < 0, exterieur > 0).
//
//   (2) EB cut-cell sur un demi-plan : le level set NON-disque partitionne la grille en cellules actives
//       ET inactives et produit de VRAIES cellules coupees (kappa < 1 sur la bande de bord).
//
//   (3) BIT-IDENTITE sans coupe : un demi-plan REJETE loin (toute la grille + ghosts active, aucune coupe
//       -> alpha = 1, kappa = 1) rend assemble_rhs_eb STRICTEMENT egal a assemble_rhs (diff = 0). Invariant
//       "EB additif et inerte sans coupe", prouve sur une geometrie NON-disque.
//
//   (4) RESIDU FINI : un demi-plan qui COUPE la boite donne un residu fini partout sur quelques pas
//       explicites (le clamp small-cell borne 1/kappa sur une geometrie generique : pas de NaN).
//
//   (5) MASQUE STAIRCASE generique : un masque 0/1 materialise depuis HalfPlaneDomain rend assemble_rhs_masked
//       a residu EXACTEMENT nul sur les cellules inactives ; un masque tout-actif (demi-plan rejete) est
//       BIT-IDENTIQUE a assemble_rhs (chemin masque inerte, geometrie generique).
//
//   (6) CONSERVATION DE MASSE a la machine sur une geometrie NON-disque BORNEE (bande |y - yc| - h,
//       periodique en x : faces y haut/bas fermees par l'EB, x telescope) : la masse EB coherente avec le
//       schema, Sum_active n kappa_eff dx dy, est conservee a la machine (drift < 1e-12) sur K pas Euler.
//
// Modele jouet INLINE : advection scalaire a vitesse constante (vx, vy), div v = 0.

#include <adc/core/model/physical_model.hpp>
#include <adc/core/state/state.hpp>
#include <adc/core/foundation/types.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/embedded_boundary.hpp>  // ADC-327 : contrat generique (DiscDomain + HalfPlaneDomain + concept)
#include <adc/numerics/numerical_flux.hpp>
#include <adc/numerics/reconstruction.hpp>
#include <adc/numerics/spatial_operator.hpp>     // assemble_rhs, assemble_rhs_masked
#include <adc/numerics/spatial_operator_eb.hpp>  // assemble_rhs_eb

#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace adc;

static constexpr double kL = 1.0;  // boite carree [0, L]^2
static constexpr double kVx = 0.7;
static constexpr double kVy = -0.4;

// Advection scalaire a vitesse (vx, vy). Flux F = (vx u, vy u).
struct Advect {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  Real vx = 0.0, vy = 0.0;
  ADC_HD State flux(const State& u, const Aux&, int dir) const {
    return State{(dir == 0 ? vx : vy) * u[0]};
  }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int dir) const {
    return std::fabs(dir == 0 ? vx : vy);
  }
  ADC_HD State source(const State&, const Aux&) const { return State{Real(0)}; }
  ADC_HD Real elliptic_rhs(const State&) const { return Real(0); }
};

static_assert(PhysicalModel<Advect>, "Advect est un PhysicalModel");

// ADC-327 : le contrat doit reconnaitre les deux instances natives, a la compilation. Diagnostics
// UNIQUEMENT : le concept ne contraint AUCUNE signature hot-path (cf. embedded_boundary.hpp).
static_assert(LevelSetDomain<detail::DiscDomain>, "DiscDomain satisfait le contrat LevelSetDomain");
static_assert(LevelSetDomain<detail::HalfPlaneDomain>,
              "HalfPlaneDomain satisfait le contrat LevelSetDomain");

// Bande NON-disque BORNEE en y : |y - yc| - h, active ssi |y - yc| < h. Periodique en x, la bande
// est fermee par l'EB en haut/bas -> masse conservee (cf. assertion (6)). Fonteur NOMME, device-safe.
struct SlabBand {
  Real yc, h;
  ADC_HD Real operator()(Real, Real y) const {
    const Real ay = (y - yc) < Real(0) ? -(y - yc) : (y - yc);
    return ay - h;
  }
};

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  std::printf("=== ADC-327 : contrat generique embedded-boundary / level-set (non-disque) ===\n");

  // ----------------------------------------------------------------------
  // (1) CONTRAT : convention de signe, operator() == level_set, cell_active == (ls < 0).
  // ----------------------------------------------------------------------
  std::printf("\n--- (1) contrat level-set (disque + demi-plan) ---\n");
  {
    // Demi-plan a*x + b*y - c : actif (ls < 0) du cote a*x + b*y < c. Diagonal (a, b != 0) : geometrie
    // franchement non-axiale, non-disque.
    const detail::HalfPlaneDomain hp{1.0, 1.0, 1.0};  // x + y < 1 actif
    // Convention de signe : (0.1, 0.1) dedans (0.2 < 1), (0.9, 0.9) dehors (1.8 > 1).
    chk(double(hp.level_set(Real(0.1), Real(0.1))) < 0.0, "(1) demi-plan : interieur ls < 0");
    chk(double(hp.level_set(Real(0.9), Real(0.9))) > 0.0, "(1) demi-plan : exterieur ls > 0");
    chk(hp.cell_active(Real(0.1), Real(0.1)) && !hp.cell_active(Real(0.9), Real(0.9)),
        "(1) demi-plan : cell_active == (ls < 0)");
    // operator() (forme callable consommee par les operateurs) == level_set (alias nomme).
    bool callable_ok = true;
    for (double x = 0.05; x < 1.0; x += 0.17)
      for (double y = 0.05; y < 1.0; y += 0.19)
        if (double(hp(Real(x), Real(y))) != double(hp.level_set(Real(x), Real(y))))
          callable_ok = false;
    chk(callable_ok, "(1) demi-plan : operator() == level_set (forme callable)");

    // Le disque (instance historique) respecte le meme contrat (operator() ajoute, level_set inchange).
    const detail::DiscDomain disc = detail::DiscDomain::centered_in_box(kL, 0.3);
    chk(double(disc.level_set(Real(0.5), Real(0.5))) < 0.0, "(1) disque : centre ls < 0");
    chk(double(disc(Real(0.5), Real(0.5))) == double(disc.level_set(Real(0.5), Real(0.5))),
        "(1) disque : operator() == level_set");
  }

  // ----------------------------------------------------------------------
  // (2) EB cut-cell sur un demi-plan : partition active/inactive + vraies cellules coupees.
  // ----------------------------------------------------------------------
  std::printf("\n--- (2) EB sur demi-plan : partition + cellules coupees ---\n");
  {
    const int n = 64;
    const Box2D dom = Box2D::from_extents(n, n);
    const Geometry geom{dom, 0.0, kL, 0.0, kL};
    const detail::HalfPlaneDomain hp{1.0, 0.6, 0.55};  // diagonal : coupe la boite en biais
    const double dx = geom.dx(), dy = geom.dy();
    int n_active = 0, n_inactive = 0, n_cut = 0;
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
        const double x = geom.x_cell(i), y = geom.y_cell(j);
        if (double(hp.level_set(Real(x), Real(y))) >= 0.0) {
          ++n_inactive;
          continue;
        }
        ++n_active;
        const detail::CutFraction cf =
            detail::cut_fraction(hp, Real(x), Real(y), Real(dx), Real(dy));
        if (double(cf.kappa) < 1.0 - 1e-9)
          ++n_cut;
      }
    std::printf("  actives=%d inactives=%d coupees=%d\n", n_active, n_inactive, n_cut);
    chk(n_active > 0 && n_inactive > 0,
        "(2) le demi-plan partitionne la grille (actives ET inactives)");
    chk(n_cut > 0, "(2) le demi-plan produit de vraies cellules coupees (kappa < 1)");
  }

  // ----------------------------------------------------------------------
  // (3) BIT-IDENTITE sans coupe : demi-plan rejete loin -> EB == cartesien (diff = 0).
  // ----------------------------------------------------------------------
  std::printf("\n--- (3) sans coupe : EB(demi-plan) == cartesien (bit-identite) ---\n");
  {
    const int n = 48;
    const Box2D dom = Box2D::from_extents(n, n);
    const Geometry geom{dom, 0.0, kL, 0.0, kL};
    const BoxArray ba(std::vector<Box2D>{dom});
    const DistributionMapping dm(1, n_ranks());
    BCRec bc;
    // Demi-plan x + y - 100 : ls < 0 sur TOUTE la grille (y compris ghosts), aucun voisin franchi
    // -> alpha = 1, kappa = 1 partout -> l'EB doit reproduire le cartesien BIT a BIT.
    const detail::HalfPlaneDomain hp{1.0, 1.0, 100.0};

    MultiFab U(ba, dm, 1, 2), aux(ba, dm, kAuxBaseComps, 2);
    aux.set_val(0.0);
    {
      Array4 a = U.fab(0).array();
      const Box2D g = U.fab(0).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
          const double x = geom.x_cell(i), y = geom.y_cell(j);
          a(i, j, 0) =
              1.0 + 0.5 * std::exp(-(((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) / 0.02));
        }
    }
    const Advect model{kVx, kVy};
    fill_ghosts(U, geom.domain, bc);
    MultiFab R_ref(ba, dm, 1, 0), R_eb(ba, dm, 1, 0);
    assemble_rhs<Minmod, RusanovFlux>(model, U, aux, geom, R_ref);
    assemble_rhs_eb<Minmod, RusanovFlux>(model, U, aux, hp, geom, R_eb);

    sync_host();
    double max_abs_diff = 0.0;
    const ConstArray4 rr = R_ref.fab(0).const_array();
    const ConstArray4 re = R_eb.fab(0).const_array();
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        max_abs_diff = std::max(max_abs_diff, std::fabs(double(rr(i, j, 0)) - double(re(i, j, 0))));
    std::printf("  max|R_eb - R_cartesien| = %.3e (attendu 0)\n", max_abs_diff);
    chk(max_abs_diff == 0.0,
        "(3) demi-plan sans coupe : residu EB BIT-IDENTIQUE au cartesien (alpha=1, kappa=1)");
  }

  // ----------------------------------------------------------------------
  // (4) RESIDU FINI : un demi-plan qui coupe la boite -> residu fini (clamp small-cell, pas de NaN).
  // ----------------------------------------------------------------------
  std::printf("\n--- (4) demi-plan coupant : residu fini (clamp generique) ---\n");
  {
    const int n = 64;
    const Box2D dom = Box2D::from_extents(n, n);
    const Geometry geom{dom, 0.0, kL, 0.0, kL};
    const BoxArray ba(std::vector<Box2D>{dom});
    const DistributionMapping dm(1, n_ranks());
    BCRec bc;
    const detail::HalfPlaneDomain hp{1.0, 0.6, 0.55};

    MultiFab U(ba, dm, 1, 2), aux(ba, dm, kAuxBaseComps, 2);
    aux.set_val(0.0);
    {
      Array4 a = U.fab(0).array();
      const Box2D g = U.fab(0).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i)
          a(i, j, 0) = 1.0 + 0.3 * std::sin(2.0 * M_PI * geom.x_cell(i)) *
                                 std::cos(2.0 * M_PI * geom.y_cell(j));
    }
    const Advect model{kVx, kVy};
    const double v = std::hypot(kVx, kVy);
    const double dt = 0.2 * geom.dx() / v;
    bool any_nan = false;
    for (int s = 0; s < 20; ++s) {
      fill_ghosts(U, geom.domain, bc);
      MultiFab R(ba, dm, 1, 0);
      assemble_rhs_eb<Minmod, RusanovFlux>(model, U, aux, hp, geom, R);
      device_fence();
      const ConstArray4 rr = R.fab(0).const_array();
      for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
        for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
          if (!std::isfinite(double(rr(i, j, 0))))
            any_nan = true;
      saxpy(U, Real(dt), R);
    }
    chk(!any_nan,
        "(4) demi-plan coupant : residu FINI partout (clamp small-cell generique, pas de NaN)");
  }

  // ----------------------------------------------------------------------
  // (5) MASQUE STAIRCASE generique : residu nul sur les inactives ; tout-actif == cartesien.
  // ----------------------------------------------------------------------
  std::printf("\n--- (5) masque staircase materialise depuis un demi-plan ---\n");
  {
    const int n = 48;
    const Box2D dom = Box2D::from_extents(n, n);
    const Geometry geom{dom, 0.0, kL, 0.0, kL};
    const BoxArray ba(std::vector<Box2D>{dom});
    const DistributionMapping dm(1, n_ranks());
    BCRec bc;
    const Advect model{kVx, kVy};

    // Materialise un masque 0/1 (1 ghost, comme System::set_disc_domain) depuis un level set quelconque.
    auto materialize_mask = [&](const detail::HalfPlaneDomain& hp) {
      MultiFab mask(ba, dm, 1, 1);
      Array4 m = mask.fab(0).array();
      const Box2D g = mask.fab(0).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i)
          m(i, j, 0) =
              hp.cell_active(Real(geom.x_cell(i)), Real(geom.y_cell(j))) ? Real(1) : Real(0);
      return mask;
    };

    MultiFab U(ba, dm, 1, 2), aux(ba, dm, kAuxBaseComps, 2);
    aux.set_val(0.0);
    {
      Array4 a = U.fab(0).array();
      const Box2D g = U.fab(0).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
          const double x = geom.x_cell(i), y = geom.y_cell(j);
          a(i, j, 0) =
              1.0 + 0.5 * std::exp(-(((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) / 0.02));
        }
    }
    fill_ghosts(U, geom.domain, bc);

    // (5a) masque coupant : residu EXACTEMENT nul sur les inactives.
    const detail::HalfPlaneDomain hp_cut{1.0, 0.6, 0.55};
    MultiFab mask_cut = materialize_mask(hp_cut);
    MultiFab R_masked(ba, dm, 1, 0);
    assemble_rhs_masked<Minmod, RusanovFlux>(model, U, aux, mask_cut, geom, R_masked);
    sync_host();
    {
      const ConstArray4 rr = R_masked.fab(0).const_array();
      double max_inactive_residual = 0.0;
      int n_inactive = 0;
      for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
        for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
          if (!hp_cut.cell_active(Real(geom.x_cell(i)), Real(geom.y_cell(j)))) {
            ++n_inactive;
            max_inactive_residual = std::max(max_inactive_residual, std::fabs(double(rr(i, j, 0))));
          }
      chk(n_inactive > 0, "(5) le masque demi-plan a de vraies cellules inactives");
      chk(max_inactive_residual == 0.0,
          "(5) residu masque EXACTEMENT nul sur les cellules inactives (demi-plan generique)");
    }

    // (5b) masque tout-actif (demi-plan rejete) : assemble_rhs_masked BIT-IDENTIQUE a assemble_rhs.
    const detail::HalfPlaneDomain hp_far{1.0, 1.0, 100.0};
    MultiFab mask_all = materialize_mask(hp_far);
    MultiFab R_ref(ba, dm, 1, 0), R_all(ba, dm, 1, 0);
    assemble_rhs<Minmod, RusanovFlux>(model, U, aux, geom, R_ref);
    assemble_rhs_masked<Minmod, RusanovFlux>(model, U, aux, mask_all, geom, R_all);
    sync_host();
    {
      const ConstArray4 rr = R_ref.fab(0).const_array();
      const ConstArray4 ra = R_all.fab(0).const_array();
      double max_abs_diff = 0.0;
      for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
        for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
          max_abs_diff =
              std::max(max_abs_diff, std::fabs(double(rr(i, j, 0)) - double(ra(i, j, 0))));
      std::printf("  masque tout-actif : max|R_masked - R_cartesien| = %.3e (attendu 0)\n",
                  max_abs_diff);
      chk(max_abs_diff == 0.0,
          "(5) masque tout-actif : residu masque BIT-IDENTIQUE au cartesien (chemin inerte)");
    }
  }

  // ----------------------------------------------------------------------
  // (6) CONSERVATION de masse a la machine sur une geometrie NON-disque BORNEE (bande |y - yc| - h).
  // ----------------------------------------------------------------------
  std::printf("\n--- (6) conservation EB sur une bande non-disque (periodique x) ---\n");
  {
    const int n = 96;
    const Box2D dom = Box2D::from_extents(n, n);
    const Geometry geom{dom, 0.0, kL, 0.0, kL};
    const BoxArray ba(std::vector<Box2D>{dom});
    const DistributionMapping dm(1, n_ranks());
    BCRec bc;  // periodique : la bande EB ferme haut/bas, x telescope
    const double dx = geom.dx(), dy = geom.dy();
    const SlabBand ls{Real(0.5 * kL),
                      Real(0.22 * kL)};  // bande centree, epaisseur non alignee grille

    MultiFab U(ba, dm, 1, 2), aux(ba, dm, kAuxBaseComps, 2);
    aux.set_val(0.0);
    {
      Array4 a = U.fab(0).array();
      const Box2D g = U.fab(0).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
          const double x = geom.x_cell(i), y = geom.y_cell(j);
          a(i, j, 0) =
              1.0 + 0.5 * std::exp(-(((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) / 0.02));
        }
    }
    const Advect model{kVx, kVy};

    auto eb_mass = [&](const MultiFab& F) {
      device_fence();
      const ConstArray4 f = F.fab(0).const_array();
      double s = 0.0;
      for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
        for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
          const double x = geom.x_cell(i), y = geom.y_cell(j);
          if (double(ls(Real(x), Real(y))) >= 0.0)
            continue;  // inactive : hors masse
          const detail::CutFraction cf =
              detail::cut_fraction(ls, Real(x), Real(y), Real(dx), Real(dy));
          const double kappa_eff = std::max(double(cf.kappa), double(detail::kEbKappaMin));
          s += double(f(i, j, 0)) * kappa_eff * dx * dy;
        }
      return s;
    };

    int n_active = 0, n_cut = 0;
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
        const double x = geom.x_cell(i), y = geom.y_cell(j);
        if (double(ls(Real(x), Real(y))) >= 0.0)
          continue;
        ++n_active;
        const detail::CutFraction cf =
            detail::cut_fraction(ls, Real(x), Real(y), Real(dx), Real(dy));
        if (double(cf.kappa) < 1.0 - 1e-9)
          ++n_cut;
      }
    chk(n_active > 0 && n_cut > 0,
        "(6) la bande produit des cellules actives ET coupees (test EB non vide)");

    const double m0 = eb_mass(U);
    chk(m0 > 0.0, "(6) masse EB initiale strictement positive");
    const double v = std::hypot(kVx, kVy);
    const double dt = 0.2 * geom.dx() / v;
    for (int s = 0; s < 60; ++s) {
      fill_ghosts(U, geom.domain, bc);
      MultiFab R(ba, dm, 1, 0);
      assemble_rhs_eb<Minmod, RusanovFlux>(model, U, aux, ls, geom, R);
      saxpy(U, Real(dt), R);
    }
    const double m1 = eb_mass(U);
    const double rel_drift = std::fabs(m1 - m0) / std::fabs(m0);
    std::printf("  masse EB : m0=%.15e  m1=%.15e  drift relatif=%.3e\n", m0, m1, rel_drift);
    chk(rel_drift < 1e-12,
        "(6) masse EB conservee a la machine sur une geometrie non-disque (drift < 1e-12)");
    {
      device_fence();
      const ConstArray4 u = U.fab(0).const_array();
      double max_dev = 0.0;
      for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
        for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
          const double x = geom.x_cell(i), y = geom.y_cell(j);
          if (double(ls(Real(x), Real(y))) >= 0.0)
            continue;
          max_dev = std::max(max_dev, std::fabs(double(u(i, j, 0)) - 1.0));
        }
      chk(max_dev > 1e-3,
          "(6) le transport EB a effectivement avance l'etat (conservation non triviale)");
    }
  }

  std::printf("\n=== VERDICT : %s ===\n", fails == 0 ? "SUCCESS" : "ECHEC");
  if (fails == 0)
    std::printf("OK test_embedded_boundary_generic\n");
  return fails == 0 ? 0 : 1;
}
