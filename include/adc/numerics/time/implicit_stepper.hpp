#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/numerics/spatial_operator.hpp>  // load_state, load_aux

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
// IMEX PARTIEL (TODO 2.2) : un modele peut declarer, variable par variable, lesquelles
// sont raides (implicites). backward_euler_source traite alors la source en
// forward-backward Euler : les variables EXPLICITES avancent en Euler avant (a l'etat
// d'entree), les variables IMPLICITES par Newton sur le SOUS-systeme reduit (les
// explicites figees a leur valeur avancee, comme donnee connue). Sans le trait, tout
// reste implicite -> comportement strictement identique a avant.

namespace adc {

// Contrat d'un stepper implicite/IMEX de bloc. Tout objet (ou lambda) qui sait
// avancer un bloc sur dt en lisant le coupleur (pour aux / phi a jour) et le modele.
template <class Stepper, class Coupler, class Block>
concept ImplicitBlockStepper =
    requires(const Stepper st, Coupler& c, Block& b, Real dt, int s, int n) {
      st(c, b, dt, s, n);
    };

// Trait OPTIONNEL : un modele peut declarer quelles variables conservees sont traitees
// en implicite (les raides). is_implicit(c) -> bool. Un modele SANS ce trait est traite
// entierement en implicite (defaut historique).
template <class M>
concept PartiallyImplicitModel = requires(int c) {
  { M::is_implicit(c) } -> std::convertible_to<bool>;
};

// La composante c du modele est-elle implicite ? Defaut (pas de trait) : toutes le sont.
template <class Model>
ADC_HD inline bool model_is_implicit(int c) {
  if constexpr (PartiallyImplicitModel<Model>)
    return Model::is_implicit(c);
  else
    return true;
}

// Masque implicite PORTE PAR LE BLOC / la politique temporelle (et NON par le modele) : carrier POD
// device-clean (tableau fixe N, passe PAR VALEUR dans le kernel, aucun pointeur hote deref. sur device).
// Quand actif (active == true), il OVERRIDE le defaut modele (model_is_implicit) : seules les composantes
// flag[c] == true sont avancees en implicite, les autres en explicite (Euler avant). C'est ce qui permet
// de REUTILISER le MEME modele avec des traitements implicites differents selon le bloc. Inactif (defaut :
// active == false) -> retombe sur model_is_implicit -> comportement bit-identique a l'historique.
template <int N>
struct ImplicitMask {
  bool active = false;
  bool flag[N] = {};
};

// La composante c est-elle implicite, masque de bloc PRIORITAIRE sur le defaut modele ? Le masque inactif
// (defaut) delegue a model_is_implicit<Model> -> strictement identique a avant ce chantier.
template <class Model, int N>
ADC_HD inline bool is_implicit_component(const ImplicitMask<N>& mask, int c) {
  if (mask.active) return mask.flag[c];
  return model_is_implicit<Model>(c);
}

namespace detail {
// Resolution dense J x = b sur le bloc de tete n x n (n <= N), pivot partiel. J et b
// detruits. N est constexpr (= Model::n_vars) -> tableau fixe, pas d'allocation,
// device-callable ; n (<= N) est le nombre de variables implicites (IMEX partiel).
template <int N>
ADC_HD inline void solve_dense(Real J[N][N], Real b[N], Real x[N], int n) {
  for (int p = 0; p < n; ++p) {
    int piv = p;
    Real best = J[p][p] < 0 ? -J[p][p] : J[p][p];
    for (int r = p + 1; r < n; ++r) {
      const Real v = J[r][p] < 0 ? -J[r][p] : J[r][p];
      if (v > best) { best = v; piv = r; }
    }
    if (piv != p) {
      for (int c = 0; c < n; ++c) { const Real t = J[p][c]; J[p][c] = J[piv][c]; J[piv][c] = t; }
      const Real t = b[p]; b[p] = b[piv]; b[piv] = t;
    }
    const Real d = J[p][p];
    for (int r = 0; r < n; ++r) {
      if (r == p) continue;
      const Real f = J[r][p] / d;
      for (int c = p; c < n; ++c) J[r][c] -= f * J[p][c];
      b[r] -= f * b[p];
    }
  }
  for (int p = 0; p < n; ++p) x[p] = b[p] / J[p][p];
}

// Resout W tel que W = Un + dt*S(W,a) en forward-backward Euler (IMEX partiel) :
//   - composantes EXPLICITES : Euler avant a l'etat d'entree, W_e = Un_e + dt*S_e(Un) ;
//   - composantes IMPLICITES : Newton sur le sous-systeme reduit, F_i = W_i - Un_i -
//     dt*S_i(W), jacobienne I - dt*(dS/dW) restreinte aux implicites (colonnes par
//     differences finies), les explicites figees a leur valeur avancee (donnee connue).
// QUI est implicite : un masque PORTE PAR LE BLOC (@p mask) prioritaire sur le defaut modele
// (is_implicit_component). Masque inactif (defaut) + modele sans trait is_implicit : toutes les
// composantes sont implicites -> backward-Euler plein, strictement identique au comportement d'origine.
template <class Model>
ADC_HD inline typename Model::State newton_source_solve(
    const Model& m, const typename Model::State& Un, const Aux& a, Real dt,
    int iters, const ImplicitMask<Model::n_vars>& mask = {}) {
  constexpr int N = Model::n_vars;
  int impl[N];  // indices des composantes implicites (les m_impl premieres slots utiles)
  int m_impl = 0;
  for (int c = 0; c < N; ++c)
    if (is_implicit_component<Model>(mask, c)) impl[m_impl++] = c;

  typename Model::State W = Un;
  // (1) explicite : Euler avant sur les composantes non implicites (source a l'entree).
  if (m_impl < N) {
    const typename Model::State S_in = m.source(Un, a);
    for (int c = 0; c < N; ++c)
      if (!is_implicit_component<Model>(mask, c)) W[c] = Un[c] + dt * S_in[c];
  }
  // (2) implicite : Newton sur le sous-systeme des composantes implicites.
  for (int it = 0; it < iters; ++it) {
    const typename Model::State S0 = m.source(W, a);
    Real F[N];
    for (int r = 0; r < m_impl; ++r) {
      const int c = impl[r];
      F[r] = W[c] - Un[c] - dt * S0[c];
    }
    Real J[N][N];
    for (int cc = 0; cc < m_impl; ++cc) {
      const int col = impl[cc];
      const Real wc = W[col] < 0 ? -W[col] : W[col];
      const Real h = Real(1e-7) * wc + Real(1e-7);
      typename Model::State Wp = W;
      Wp[col] += h;
      const typename Model::State Sp = m.source(Wp, a);
      for (int rr = 0; rr < m_impl; ++rr) {
        const int row = impl[rr];
        const Real dSdW = (Sp[row] - S0[row]) / h;
        J[rr][cc] = (row == col ? Real(1) : Real(0)) - dt * dSdW;
      }
    }
    Real delta[N];
    solve_dense<N>(J, F, delta, m_impl);
    for (int r = 0; r < m_impl; ++r) W[impl[r]] -= delta[r];
  }
  return W;
}
}  // namespace detail

namespace detail {
// Noyau device du pas implicite sur la source (Newton local en place). FONCTEUR NOMME (et non lambda
// etendue) : emission device ROBUSTE quand le noyau Model-template est instancie depuis une TU EXTERNE
// (chemin IMEX d'un bloc add_compiled_model, via la std::function d'avance de block_builder). Corps
// identique a l'ancienne lambda -> resultat bit-identique sur CPU.
template <class Model>
struct BackwardEulerSourceKernel {
  Model m;
  ConstArray4 uc, ax;
  Array4 u;
  Real dt;
  int it;
  ImplicitMask<Model::n_vars> mask;  // masque de bloc (POD, par valeur) ; inactif = defaut modele
  ADC_HD void operator()(int i, int j) const {
    const typename Model::State Un = load_state<Model>(uc, i, j);
    const Aux a = load_aux<aux_comps<Model>()>(ax, i, j);
    const typename Model::State W = newton_source_solve<Model>(m, Un, a, dt, it, mask);
    for (int c = 0; c < Model::n_vars; ++c) u(i, j, c) = W[c];
  }
};
}  // namespace detail

// W = U + dt * model.source(W, aux), resolu EN PLACE par Newton local (jacobienne
// par differences finies). Voir l'en-tete du fichier pour la stabilite. @p mask : masque implicite
// PORTE PAR LE BLOC (override du defaut modele) ; inactif (defaut) -> comportement bit-identique.
template <class Model>
void backward_euler_source(const Model& model, const MultiFab& aux, MultiFab& U,
                           Real dt, int iters = 2,
                           const ImplicitMask<Model::n_vars>& mask = {}) {
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 u = U.fab(li).array();
    const ConstArray4 uc = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    const Box2D b = U.box(li);
    for_each_cell(b, detail::BackwardEulerSourceKernel<Model>{model, uc, ax, u, dt, iters, mask});
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
