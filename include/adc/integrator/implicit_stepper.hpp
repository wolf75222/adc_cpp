#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/operator/spatial_operator.hpp>  // load_state, load_aux

#include <concepts>

// Pas implicite / IMEX d'un bloc : le CONTRAT (au lieu du callback nu).
//
// SystemCoupler::step avance les blocs explicites SSPRK lui-meme et DELEGUE les
// blocs implicites / IMEX a un callable `(coupler&, block&, dt, substep, nsub)`.
// Ce header transforme ce callback anonyme en contrat nomme (ImplicitBlockStepper)
// et fournit UN DEFAUT pret a l'emploi pour le cas plasma raide le plus courant :
// une source de relaxation rigide. Objectif tuteur : "un IMEX par defaut, sans que
// l'utilisateur ecrive Newton".
//
//   - backward_euler_source : resout EN PLACE le pas implicite sur la source
//     LOCALE du modele,  W = U + dt * S(W, aux), par NEWTON local avec jacobienne
//     par differences finies. Inconditionnellement stable pour une relaxation
//     lineaire (la ou un simple point-fixe de Picard DIVERGERAIT des que dt*raideur
//     > 1, justement le regime raide) ; EXACT en une iteration si S est lineaire en
//     U ; convergence quadratique sinon. Aucune jacobienne a fournir cote modele.
//   - ImplicitSourceStepper : l'objet-stepper qui branche backward_euler_source sur
//     l'interface du SystemCoupler. C'est l'analogue "source seule" de
//     imex_euler_step (integrator/imex.hpp), le pas implicite d'un schema IMEX dont
//     le transport reste explicite (avance par le coeur sur les blocs explicites).
//
// La separation transport-explicite / source-implicite-par-variable (IMEX partiel)
// reste un raffinement (cf. TODO 2.2) ; ce defaut traite toute la source en
// implicite, ce qui suffit aux blocs raides ou la raideur est dans la source.

namespace adc {

// Contrat d'un stepper implicite/IMEX de bloc. Tout objet (ou lambda) qui sait
// avancer un bloc sur dt en lisant le coupleur (pour aux / phi a jour) et le modele.
template <class Stepper, class Coupler, class Block>
concept ImplicitBlockStepper =
    requires(const Stepper st, Coupler& c, Block& b, Real dt, int s, int n) {
      st(c, b, dt, s, n);
    };

namespace detail {
// Resolution dense J x = b, N petit (= n_vars), pivot partiel. J et b detruits.
// N est constexpr (Model::n_vars), donc Real[N][N] est un tableau fixe ; pas
// d'allocation, device-callable.
template <int N>
ADC_HD inline void solve_dense(Real J[N][N], Real b[N], Real x[N]) {
  for (int p = 0; p < N; ++p) {
    int piv = p;
    Real best = J[p][p] < 0 ? -J[p][p] : J[p][p];
    for (int r = p + 1; r < N; ++r) {
      const Real v = J[r][p] < 0 ? -J[r][p] : J[r][p];
      if (v > best) { best = v; piv = r; }
    }
    if (piv != p) {
      for (int c = 0; c < N; ++c) { const Real t = J[p][c]; J[p][c] = J[piv][c]; J[piv][c] = t; }
      const Real t = b[p]; b[p] = b[piv]; b[piv] = t;
    }
    const Real d = J[p][p];
    for (int r = 0; r < N; ++r) {
      if (r == p) continue;
      const Real f = J[r][p] / d;
      for (int c = p; c < N; ++c) J[r][c] -= f * J[p][c];
      b[r] -= f * b[p];
    }
  }
  for (int p = 0; p < N; ++p) x[p] = b[p] / J[p][p];
}

// Un pas de Newton ponctuel : resout W = Un + dt*S(W,a) pour W, en partant de Un.
// F(W) = W - Un - dt*S(W) ; J = I - dt*(dS/dW), colonnes de dS/dW par differences
// finies (une evaluation de source par colonne). N = Model::n_vars.
template <class Model>
ADC_HD inline typename Model::State newton_source_solve(
    const Model& m, const typename Model::State& Un, const Aux& a, Real dt,
    int iters) {
  constexpr int N = Model::n_vars;
  typename Model::State W = Un;
  for (int k = 0; k < iters; ++k) {
    const typename Model::State S0 = m.source(W, a);
    Real F[N];
    for (int r = 0; r < N; ++r) F[r] = W[r] - Un[r] - dt * S0[r];
    Real J[N][N];
    for (int col = 0; col < N; ++col) {
      const Real wc = W[col] < 0 ? -W[col] : W[col];
      const Real h = Real(1e-7) * wc + Real(1e-7);
      typename Model::State Wp = W;
      Wp[col] += h;
      const typename Model::State Sp = m.source(Wp, a);
      for (int row = 0; row < N; ++row) {
        const Real dSdW = (Sp[row] - S0[row]) / h;
        J[row][col] = (row == col ? Real(1) : Real(0)) - dt * dSdW;
      }
    }
    Real delta[N];
    solve_dense<N>(J, F, delta);
    for (int r = 0; r < N; ++r) W[r] -= delta[r];
  }
  return W;
}
}  // namespace detail

// W = U + dt * model.source(W, aux), resolu EN PLACE par Newton local (jacobienne
// par differences finies). Voir l'en-tete du fichier pour la stabilite.
template <class Model>
void backward_euler_source(const Model& model, const MultiFab& aux, MultiFab& U,
                           Real dt, int iters = 2) {
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 u = U.fab(li).array();
    const ConstArray4 uc = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    const Box2D b = U.box(li);
    const Model m = model;
    const int it = iters;
    for_each_cell(b, [=] ADC_HD(int i, int j) {
      const typename Model::State Un = load_state<Model>(uc, i, j);
      const Aux a = load_aux(ax, i, j);
      const typename Model::State W =
          detail::newton_source_solve<Model>(m, Un, a, dt, it);
      for (int c = 0; c < Model::n_vars; ++c) u(i, j, c) = W[c];
    });
  }
}

// Stepper implicite par defaut : backward-Euler (Newton) sur la source du modele.
// Modele ImplicitBlockStepper ; passe tel quel a SystemCoupler::step comme callback
// d'avancee implicite. L'utilisateur n'ecrit aucun solveur.
struct ImplicitSourceStepper {
  int iters = 2;

  template <class Coupler, class Block>
  void operator()(Coupler& coupler, Block& block, Real dt, int /*substep*/,
                  int /*nsub*/) const {
    backward_euler_source(block.model, coupler.aux(), block.U(), dt, iters);
  }
};

}  // namespace adc
