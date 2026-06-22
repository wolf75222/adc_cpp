// Etage source condense par Schur COMPOSITE multi-niveau (AmrCondensedSchurSourceStepper, Phase 3c) :
// le patch fin RAFFINE l'elliptique (operateur tensoriel Schur resolu par FAC sur grossier + fin), puis
// reconstruction des vitesses PAR NIVEAU + cascade average_down. Cf. amr_condensed_schur_source_stepper.hpp.
//
// On construit une hierarchie 2 niveaux (grossier n x n + UN patch fin central) avec un etat fluide
// magnetise (rho0, mom = rho0 v0, B_z = B0) et on applique UN pas de l'etage condense COMPOSITE. On verifie :
//   (A) FINITUDE de l'etat (rho, mom) sur les deux niveaux.
//   (B) la SOURCE a AGI : la quantite de mouvement FINE a change (l'etage a tourne, pas un no-op).
//   (C) rho GELEE : la densite (grossier ET fin) est inchangee (la source ne traite que la vitesse).
//   (D) CASCADE average_down : les cellules grossieres COUVERTES = moyenne 2x2 des cellules fines (la
//       coherence #169 est retablie apres la source par niveau).
//   (E) STABILITE : a grand dt (source raide), l'etat reste fini et borne (theta=1 implicite).
//
// Serie (Kokkos OFF) : grossier mono-box, 1 patch fin mono-box (cadre Phase 3c ; MPI = Phase 4).

#include <adc/coupling/schur/amr_condensed_schur_source_stepper.hpp>

#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

static VariableSet fluid_vars() {
  VariableSet vs;
  vs.kind = VariableKind::Conservative;
  vs.names = {"rho", "mx", "my", "E"};
  vs.roles = {VariableRole::Density, VariableRole::MomentumX, VariableRole::MomentumY,
              VariableRole::Energy};
  vs.size = 4;
  return vs;
}

// Initialise rho0 + mom = rho0 v0(x,y) + E (profils lisses nuls au bord). Memes profils que l'echelle
// grossiere -> a la resolution du niveau (geom du niveau).
struct InitKernel {
  Geometry geom;
  Array4 st;
  Real rho0;
  int c_rho, c_mx, c_my, c_E;
  ADC_HD void operator()(int i, int j) const {
    const Real x = geom.x_cell(i), y = geom.y_cell(j);
    const Real sx = std::sin(Real(kPi) * x), sy = std::sin(Real(kPi) * y);
    const Real vx = Real(0.6) * sx * sy;
    const Real vy = Real(-0.4) * std::sin(Real(2 * kPi) * x) * sy;
    st(i, j, c_rho) = rho0;
    st(i, j, c_mx) = rho0 * vx;
    st(i, j, c_my) = rho0 * vy;
    st(i, j, c_E) = Real(1.0) + Real(0.5) * rho0 * (vx * vx + vy * vy);
  }
};

struct ConstKernel {
  Array4 a;
  Real v;
  ADC_HD void operator()(int i, int j) const { a(i, j, 0) = v; }
};

static void init_state(MultiFab& U, const Geometry& g, Real rho0, int c_rho, int c_mx, int c_my,
                       int c_E) {
  for (int li = 0; li < U.local_size(); ++li)
    for_each_cell(U.box(li), InitKernel{g, U.fab(li).array(), rho0, c_rho, c_mx, c_my, c_E});
}

// max |state - ref| (toutes composantes valides).
static double max_diff(const MultiFab& a, const MultiFab& b, int nc) {
  sync_host();
  double d = 0;
  for (int li = 0; li < a.local_size(); ++li) {
    const ConstArray4 ua = a.fab(li).const_array(), ub = b.fab(li).const_array();
    const Box2D bx = a.box(li);
    for (int j = bx.lo[1]; j <= bx.hi[1]; ++j)
      for (int i = bx.lo[0]; i <= bx.hi[0]; ++i)
        for (int c = 0; c < nc; ++c)
          d = std::fmax(d, std::fabs(ua(i, j, c) - ub(i, j, c)));
  }
  return all_reduce_max(d);
}

static bool all_finite(const MultiFab& m, int nc) {
  sync_host();
  double bad = 0;
  for (int li = 0; li < m.local_size(); ++li) {
    const ConstArray4 u = m.fab(li).const_array();
    const Box2D b = m.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        for (int c = 0; c < nc; ++c)
          if (!std::isfinite(u(i, j, c)))
            bad = 1;
  }
  return all_reduce_max(bad) == 0.0;  // collectif : aucun NaN sur AUCUN rang.
}

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank();
  long fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      if (me == 0)
        std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int n = 32;
  const Real rho0 = Real(1.5), B0 = Real(4.0), alpha = Real(3.0), theta = Real(1.0);
  const VariableSet vars = fluid_vars();
  const int c_rho = vars.index_of(VariableRole::Density);
  const int c_mx = vars.index_of(VariableRole::MomentumX);
  const int c_my = vars.index_of(VariableRole::MomentumY);
  const int c_E = vars.index_of(VariableRole::Energy);

  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom_c{dom, 0.0, 1.0, 0.0, 1.0};
  Geometry geom_f = geom_c.refine(2);
  BoxArray ba_c = BoxArray::from_domain(dom, n);
  DistributionMapping dm_c(ba_c.size(), n_ranks());
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;

  // patch fin central : grossier [n/4, 3n/4) -> box fine [n/2, 3n/2-1].
  const int Ic0 = n / 4, Ic1 = 3 * n / 4 - 1;
  Box2D fb{{2 * Ic0, 2 * Ic0}, {2 * Ic1 + 1, 2 * Ic1 + 1}};
  BoxArray ba_f(std::vector<Box2D>{fb});
  DistributionMapping dm_f(1, n_ranks());

  auto run_step = [&](Real dt, MultiFab& Uc_out, MultiFab& Uf_out) {
    MultiFab Uc(ba_c, dm_c, vars.size, 1), Uf(ba_f, dm_f, vars.size, 1);
    init_state(Uc, geom_c, rho0, c_rho, c_mx, c_my, c_E);
    init_state(Uf, geom_f, rho0, c_rho, c_mx, c_my, c_E);
    MultiFab coarse_phi(ba_c, dm_c, 1, 1), coarse_bz(ba_c, dm_c, 1, 1), fine_aux(ba_f, dm_f, 1, 1);
    coarse_phi.set_val(0.0);  // phi^n grossier
    fine_aux.set_val(0.0);    // phi^n fin (aux comp 0)
    for (int li = 0; li < coarse_bz.local_size(); ++li)
      for_each_cell(coarse_bz.box(li), ConstKernel{coarse_bz.fab(li).array(), B0});
    std::vector<AmrLevelMP> levels;
    levels.push_back(AmrLevelMP{std::move(Uc), &coarse_phi, geom_c.dx(), geom_c.dy()});
    levels.push_back(AmrLevelMP{std::move(Uf), &fine_aux, geom_f.dx(), geom_f.dy()});
    AmrCondensedSchurSourceStepper amr(vars, geom_c, ba_c, bc, alpha);
    amr.step(levels, coarse_phi, coarse_bz, /*c_bz=*/0, theta, dt);
    Uc_out = std::move(levels[0].U);
    Uf_out = std::move(levels[1].U);
  };

  // etats initiaux pour comparaison (rho gelee, source agit).
  MultiFab Uc0(ba_c, dm_c, vars.size, 1), Uf0(ba_f, dm_f, vars.size, 1);
  init_state(Uc0, geom_c, rho0, c_rho, c_mx, c_my, c_E);
  init_state(Uf0, geom_f, rho0, c_rho, c_mx, c_my, c_E);

  const Real dt = Real(0.05);
  MultiFab Uc(ba_c, dm_c, vars.size, 1), Uf(ba_f, dm_f, vars.size, 1);
  run_step(dt, Uc, Uf);

  // (A) finitude.
  chk(all_finite(Uc, vars.size) && all_finite(Uf, vars.size), "(A) etat fini (grossier + fin)");

  // (B) la source a agi sur le FIN (mom change).
  chk(max_diff(Uf, Uf0, vars.size) > 1e-6,
      "(B) la source modifie l'etat fin (etage composite a tourne)");

  // (C) rho gelee (grossier ET fin) : la densite est inchangee.
  {
    sync_host();
    double drho_c = 0, drho_f = 0;
    for (int li = 0; li < Uc.local_size(); ++li) {
      const ConstArray4 u = Uc.fab(li).const_array(), u0 = Uc0.fab(li).const_array();
      const Box2D b = Uc.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          drho_c = std::fmax(drho_c, std::fabs(u(i, j, c_rho) - u0(i, j, c_rho)));
    }
    for (int li = 0; li < Uf.local_size(); ++li) {
      const ConstArray4 u = Uf.fab(li).const_array(), u0 = Uf0.fab(li).const_array();
      const Box2D b = Uf.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          drho_f = std::fmax(drho_f, std::fabs(u(i, j, c_rho) - u0(i, j, c_rho)));
    }
    // NB : rho grossier COUVERT = average_down de rho fin (= rho0, gele) -> inchange a la precision machine.
    chk(all_reduce_max(drho_c) < 1e-12 && all_reduce_max(drho_f) < 1e-12,
        "(C) rho gelee par la source (grossier + fin inchanges)");
  }

  // (D) cascade average_down : cellule grossiere COUVERTE = moyenne 2x2 des cellules fines (mom_x).
  {
    sync_host();
    const ConstArray4 UC = Uc.fab(0).const_array();
    const ConstArray4 UF = Uf.fab(0).const_array();
    double dcov = 0;
    for (int J = Ic0; J <= Ic1; ++J)
      for (int I = Ic0; I <= Ic1; ++I)
        for (int c = c_mx; c <= c_my; ++c) {
          const double avg = 0.25 * (UF(2 * I, 2 * J, c) + UF(2 * I + 1, 2 * J, c) +
                                     UF(2 * I, 2 * J + 1, c) + UF(2 * I + 1, 2 * J + 1, c));
          dcov = std::fmax(dcov, std::fabs(UC(I, J, c) - avg));
        }
    chk(all_reduce_max(dcov) < 1e-12,
        "(D) cascade average_down : grossier couvert = moyenne 2x2 fine");
  }

  // (E) stabilite a grand dt (source raide, theta=1 implicite) : etat fini et borne.
  {
    MultiFab Uc2(ba_c, dm_c, vars.size, 1), Uf2(ba_f, dm_f, vars.size, 1);
    run_step(Real(0.5), Uc2, Uf2);  // dt 10x plus grand
    const bool fin = all_finite(Uc2, vars.size) && all_finite(Uf2, vars.size);
    // borne : la vitesse fine reste du meme ordre (pas d'explosion). max |mom| fin < 20 * rho0.
    sync_host();
    double mmax = 0;
    for (int li = 0; li < Uf2.local_size(); ++li) {
      const ConstArray4 u = Uf2.fab(li).const_array();
      const Box2D b = Uf2.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          mmax = std::fmax(mmax, std::fmax(std::fabs(u(i, j, c_mx)), std::fabs(u(i, j, c_my))));
    }
    chk(fin && all_reduce_max(mmax) < 20.0 * rho0,
        "(E) stable et borne a grand dt (theta=1 implicite)");
  }

  fails = static_cast<long>(all_reduce_max(static_cast<double>(fails)));
  if (me == 0 && fails == 0)
    std::printf("OK test_amr_condensed_schur_composite\n");
  comm_finalize();
  return fails == 0 ? 0 : 1;
}
