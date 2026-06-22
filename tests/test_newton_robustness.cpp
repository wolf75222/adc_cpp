// Newton de la source implicite GENERALISE (audit 2026-06, vague 2) : options (tolerances,
// damping, fail_policy) et diagnostics (cellule fautive / composante) -- preuves :
//  (1) NON-EULER MULTI-VARIABLES : un systeme de relaxation NON LINEAIRE 3 variables (aucun layout
//      rho/m/E, aucune pression) converge sous tolerance -- le solveur n'est pas hardcode Euler.
//      La solution verifie l'equation BE W = Un + dt*S(W) au residu pres.
//  (2) DAMPING : newton amorti (damping < 1) converge vers la MEME racine (plus d'iterations).
//  (3) PATHOLOGIE PROPRE : une source qui produit NaN sur UNE cellule -> fail_policy=throw leve
//      une erreur claire, le rapport identifie LA cellule fautive (i, j) et la composante.
//  (4) OBSERVATEUR PUR : avec defauts + diagnostics, W est BIT-IDENTIQUE au chemin historique.
#include <adc/core/state/state.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/layout/distribution_mapping.hpp>
#include <adc/mesh/storage/multifab.hpp>
#include <adc/numerics/time/implicit_stepper.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <stdexcept>

using adc::Aux;
using adc::Real;

// Relaxation NON LINEAIRE 3 variables, sans aucun layout fluide (ni densite, ni pression) :
//   S0 = -k (u0 - u1 u2) ; S1 = -k (u1 - u0/2) ; S2 = -k u2^3.
struct StiffModel {
  using State = adc::StateVec<3>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 3;
  Real k = 200.0;
  ADC_HD State flux(const State&, const Aux&, int) const { return State{}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return 0; }
  ADC_HD State source(const State& u, const Aux&) const {
    State s{};
    s[0] = -k * (u[0] - u[1] * u[2]);
    s[1] = -k * (u[1] - Real(0.5) * u[0]);
    s[2] = -k * u[2] * u[2] * u[2];
    return s;
  }
  ADC_HD Real elliptic_rhs(const State&) const { return 0; }
};

// StiffModel + JACOBIEN ANALYTIQUE exact (trait HasSourceJacobian, vague 3) : le Newton doit
// converger vers la MEME racine que les differences finies (l'equation BE est identique).
struct JacStiffModel : StiffModel {
  ADC_HD void source_jacobian(const State& u, const Aux&, Real (&J)[3][3]) const {
    J[0][0] = -k;
    J[0][1] = k * u[2];
    J[0][2] = k * u[1];
    J[1][0] = k * Real(0.5);
    J[1][1] = -k;
    J[1][2] = 0;
    J[2][0] = 0;
    J[2][1] = 0;
    J[2][2] = -Real(3) * k * u[2] * u[2];
  }
};

// Source PATHOLOGIQUE : sqrt(u0 - 10) -> NaN des que u0 < 10 (toutes nos cellules), sur la
// composante 1 SEULEMENT quand u0 < seuil bas (pour viser UNE cellule fautive).
struct NanModel {
  using State = adc::StateVec<3>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 3;
  ADC_HD State flux(const State&, const Aux&, int) const { return State{}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return 0; }
  ADC_HD State source(const State& u, const Aux&) const {
    State s{};
    s[0] = -u[0];
    s[1] = u[0] < Real(0) ? std::sqrt(u[0]) : -u[1];  // u0 < 0 -> NaN sur la composante 1
    s[2] = -u[2];
    return s;
  }
  ADC_HD Real elliptic_rhs(const State&) const { return 0; }
};

static adc::MultiFab make_mf(const adc::BoxArray& ba, const adc::DistributionMapping& dm, int nc) {
  adc::MultiFab m(ba, dm, nc, 0);
  m.set_val(Real(0));
  return m;
}

int main() {
  const adc::Box2D dom = adc::Box2D::from_extents(4, 4);
  const adc::BoxArray ba(std::vector<adc::Box2D>{dom});
  const adc::DistributionMapping dm(1, adc::n_ranks());
  adc::MultiFab aux = make_mf(ba, dm, adc::kAuxBaseComps);

  // --- (1) convergence non-Euler multi-variables sous tolerance -------------------------------
  StiffModel m;
  adc::MultiFab U = make_mf(ba, dm, 3);
  for (int li = 0; li < U.local_size(); ++li) {
    adc::Array4 u = U.fab(li).array();
    const adc::Box2D b = U.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        u(i, j, 0) = 1.0 + 0.1 * i;
        u(i, j, 1) = -0.5 + 0.05 * j;
        u(i, j, 2) = 0.3;
      }
  }
  adc::MultiFab U0 = make_mf(ba, dm, 3);
  for (int li = 0; li < U.local_size(); ++li) {  // copie de l'etat d'entree (verification BE)
    adc::Array4 d = U0.fab(li).array();
    const adc::ConstArray4 s = U.fab(li).const_array();
    const adc::Box2D b = U.box(li);
    for (int c = 0; c < 3; ++c)
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          d(i, j, c) = s(i, j, c);
  }
  const Real dt = 0.05;  // k*dt = 10 : raide (un point-fixe explicite divergerait)
  adc::NewtonOptions opts;
  opts.max_iters = 25;
  opts.rel_tol = 1e-12;
  opts.abs_tol = 1e-13;
  adc::NewtonReport rep;
  adc::backward_euler_source(m, aux, U, dt, opts, {}, &rep);
  if (!rep.converged || rep.n_failed != 0) {
    std::printf("FAIL (1) : non converge (n_failed=%.0f, res=%.3e)\n", rep.n_failed,
                static_cast<double>(rep.max_residual));
    return 1;
  }
  // verification BE : W - Un - dt S(W) ~ 0 sur chaque cellule.
  double worst = 0;
  for (int li = 0; li < U.local_size(); ++li) {
    const adc::ConstArray4 w = U.fab(li).const_array();
    const adc::ConstArray4 un = U0.fab(li).const_array();
    const adc::Box2D b = U.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        StiffModel::State W{};
        for (int c = 0; c < 3; ++c)
          W[c] = w(i, j, c);
        const StiffModel::State S = m.source(W, Aux{});
        for (int c = 0; c < 3; ++c)
          worst = std::fmax(worst, std::fabs(w(i, j, c) - un(i, j, c) - dt * S[c]));
      }
  }
  if (worst > 1e-10) {
    std::printf("FAIL (1) : residu BE %.3e > 1e-10\n", worst);
    return 1;
  }
  std::printf(
      "OK  (1) relaxation non lineaire 3-var NON Euler : converge (res BE %.1e, iters max "
      "%.0f/25)\n",
      worst, static_cast<double>(rep.max_iters_used));

  // --- (2) damping : meme racine, plus d'iterations --------------------------------------------
  adc::MultiFab Ud = make_mf(ba, dm, 3);
  for (int li = 0; li < Ud.local_size(); ++li) {
    adc::Array4 d = Ud.fab(li).array();
    const adc::ConstArray4 s = U0.fab(li).const_array();
    const adc::Box2D b = Ud.box(li);
    for (int c = 0; c < 3; ++c)
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          d(i, j, c) = s(i, j, c);
  }
  adc::NewtonOptions od = opts;
  od.damping = 0.5;
  od.max_iters = 80;
  adc::NewtonReport repd;
  adc::backward_euler_source(m, aux, Ud, dt, od, {}, &repd);
  double dmax = 0;
  for (int li = 0; li < U.local_size(); ++li) {
    const adc::ConstArray4 a4 = U.fab(li).const_array();
    const adc::ConstArray4 b4 = Ud.fab(li).const_array();
    const adc::Box2D b = U.box(li);
    for (int c = 0; c < 3; ++c)
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          dmax = std::fmax(dmax, std::fabs(a4(i, j, c) - b4(i, j, c)));
  }
  if (!repd.converged || dmax > 1e-8) {
    std::printf("FAIL (2) : damping (converged=%d, ecart racine %.3e)\n", int(repd.converged),
                dmax);
    return 1;
  }
  std::printf("OK  (2) Newton amorti (damping=0.5) : meme racine (ecart %.1e), iters %.0f\n", dmax,
              static_cast<double>(repd.max_iters_used));

  // --- (3) pathologie : NaN sur UNE cellule -> throw + cellule fautive -------------------------
  NanModel nm;
  adc::MultiFab Un2 = make_mf(ba, dm, 3);
  for (int li = 0; li < Un2.local_size(); ++li) {
    adc::Array4 u = Un2.fab(li).array();
    const adc::Box2D b = Un2.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        u(i, j, 0) = 1.0;  // sain partout...
        u(i, j, 1) = 0.2;
        u(i, j, 2) = 0.1;
      }
  }
  Un2.fab(0).array()(2, 3, 0) = -4.0;  // ...sauf la cellule (2, 3) : sqrt(-4) -> NaN composante 1
  adc::NewtonOptions opf;
  opf.fail_policy = adc::NewtonOptions::kFailThrow;
  adc::NewtonReport repf;
  bool threw = false;
  try {
    adc::backward_euler_source(nm, aux, Un2, 0.1, opf, {}, &repf);
  } catch (const std::runtime_error& e) {
    threw = true;
    std::printf("OK  (3) fail_policy=throw : %s\n", e.what());
  }
  if (!threw || repf.n_failed < 1) {
    std::printf("FAIL (3) : pas de throw / pas d'echec rapporte (n_failed=%.0f)\n", repf.n_failed);
    return 1;
  }
  if (repf.failed_i != 2 || repf.failed_j != 3) {
    std::printf("FAIL (3) : cellule fautive (%g, %g) != (2, 3)\n", repf.failed_i, repf.failed_j);
    return 1;
  }
  std::printf("OK  (3) cellule fautive identifiee (%g, %g), composante %g\n", repf.failed_i,
              repf.failed_j, repf.failed_comp);

  // --- (4) observateur pur : defauts + diagnostics == defauts sans diagnostics (bit-identique) --
  adc::MultiFab Ua = make_mf(ba, dm, 3), Ub = make_mf(ba, dm, 3);
  for (int li = 0; li < Ua.local_size(); ++li) {
    adc::Array4 a4 = Ua.fab(li).array();
    adc::Array4 b4 = Ub.fab(li).array();
    const adc::ConstArray4 s = U0.fab(li).const_array();
    const adc::Box2D b = Ua.box(li);
    for (int c = 0; c < 3; ++c)
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
          a4(i, j, c) = s(i, j, c);
          b4(i, j, c) = s(i, j, c);
        }
  }
  adc::backward_euler_source(m, aux, Ua, dt, 2);  // chemin historique (surcharge iters)
  adc::NewtonOptions odef;                        // defauts stricts
  adc::NewtonReport repo;
  adc::backward_euler_source(m, aux, Ub, dt, odef, {}, &repo);  // instrumente, defauts
  for (int li = 0; li < Ua.local_size(); ++li) {
    const adc::ConstArray4 a4 = Ua.fab(li).const_array();
    const adc::ConstArray4 b4 = Ub.fab(li).const_array();
    const adc::Box2D b = Ua.box(li);
    for (int c = 0; c < 3; ++c)
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          if (a4(i, j, c) != b4(i, j, c)) {
            std::printf("FAIL (4) : diagnostics non observateur pur en (%d,%d,c%d)\n", i, j, c);
            return 1;
          }
  }
  std::printf("OK  (4) diagnostics = observateur pur (W bit-identique au chemin historique)\n");

  // --- (5) JACOBIEN ANALYTIQUE (vague 3) : meme racine que les differences finies ---------------
  static_assert(!adc::HasSourceJacobian<StiffModel>, "StiffModel sans jacobien : FD historiques");
  static_assert(adc::HasSourceJacobian<JacStiffModel>, "JacStiffModel doit declarer le trait");
  JacStiffModel jm;
  adc::MultiFab Uj = make_mf(ba, dm, 3);
  for (int li = 0; li < Uj.local_size(); ++li) {
    adc::Array4 d = Uj.fab(li).array();
    const adc::ConstArray4 s = U0.fab(li).const_array();
    const adc::Box2D b = Uj.box(li);
    for (int c = 0; c < 3; ++c)
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          d(i, j, c) = s(i, j, c);
  }
  adc::NewtonReport repj;
  adc::backward_euler_source(jm, aux, Uj, dt, opts, {}, &repj);
  double jdiff = 0;
  for (int li = 0; li < U.local_size(); ++li) {
    const adc::ConstArray4 a4 = U.fab(li).const_array();
    const adc::ConstArray4 b4 = Uj.fab(li).const_array();
    const adc::Box2D b = U.box(li);
    for (int c = 0; c < 3; ++c)
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          jdiff = std::fmax(jdiff, std::fabs(a4(i, j, c) - b4(i, j, c)));
  }
  if (!repj.converged || jdiff > 1e-9) {
    std::printf("FAIL (5) : jacobien analytique (converged=%d, ecart racine %.3e)\n",
                int(repj.converged), jdiff);
    return 1;
  }
  std::printf("OK  (5) jacobien analytique : meme racine que les FD (ecart %.1e), iters %.0f\n",
              jdiff, static_cast<double>(repj.max_iters_used));

  std::printf("OK  test_newton_robustness : tout est vert\n");
  return 0;
}
