#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/numerics/spatial_operator.hpp>  // load_state, load_aux

#include <algorithm>  // std::max (agregation du rapport Newton, hote)
#include <concepts>
#include <cstdio>     // std::snprintf / fprintf (fail_policy : message hote, jamais en kernel)
#include <stdexcept>  // std::runtime_error (fail_policy = throw, hote apres reductions)

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

// Trait OPTIONNEL : JACOBIEN ANALYTIQUE de la source (audit vague 3, JacobianPolicy). Quand le
// modele (ou sa brique source, forwardee par CompositeModel) declare
//   source_jacobian(U, aux, J)  avec  J[r][c] = dS_r/dU_c  (matrice COMPLETE n_vars x n_vars),
// le Newton de la source implicite l'utilise A LA PLACE des differences finies : exactitude
// (plus de bruit fd_eps) et n_impl evaluations de source economisees par iteration. Un modele
// SANS le trait garde les differences finies historiques, bit-identique. ADC_HD requis.
template <class M>
concept HasSourceJacobian =
    requires(const M m, const typename M::State u, const Aux a,
             Real (&J)[M::n_vars][M::n_vars]) {
      m.source_jacobian(u, a, J);
    };

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

/// Options du Newton local de la source implicite (backward-Euler). Les DEFAUTS reproduisent
/// EXACTEMENT le comportement historique : 2 iterations FIXES, aucune evaluation de residu
/// (rel_tol == abs_tol == 0 -> pas de test d'arret, pas de cout supplementaire), pas de la
/// jacobienne par differences finies h = fd_eps*|w| + fd_eps avec fd_eps = 1e-7 (la constante
/// historiquement codee en dur). POD device-clean (passe PAR VALEUR dans le kernel).
///  - max_iters : budget d'iterations Newton (historique : 2).
///  - rel_tol / abs_tol : critere d'arret ||F||_inf <= abs_tol + rel_tol*||F0||_inf, evalue par
///    CELLULE en tete d'iteration. 0/0 (defaut) = desactive -> boucle historique bit-identique.
///  - fd_eps : pas (relatif ET plancher absolu) de la jacobienne par differences finies.
///  - damping : facteur d'amortissement de la mise a jour W -= damping * delta. 1 (defaut) =
///    Newton plein, bit-identique (multiplication par 1.0 exacte en IEEE). < 1 = Newton amorti
///    (source tres raide / mauvais conditionnement : robustesse au prix de la vitesse).
///  - fail_policy : reaction HOTE (apres reduction) a des cellules en echec --
///    kFailNone (defaut) = enregistrer seulement (historique) ; kFailWarn = un avertissement
///    stderr par avance ; kFailThrow = std::runtime_error avec la cellule fautive. != kFailNone
///    force le chemin instrumente (detection necessaire) -- pur observateur, W inchange.
struct NewtonOptions {
  static constexpr int kFailNone = 0;
  static constexpr int kFailWarn = 1;
  static constexpr int kFailThrow = 2;
  int max_iters = 2;
  Real rel_tol = Real(0);
  Real abs_tol = Real(0);
  Real fd_eps = Real(1e-7);
  Real damping = Real(1);
  int fail_policy = kFailNone;
};

/// Statistique de SORTIE du Newton d'UNE cellule (POD device, ecrit dans le scratch diagnostics) :
/// res = ||F||_inf a la sortie ; iters = iterations consommees ; failed = 1 si la cellule a echoue
/// (residu non fini, pivot degenere/non fini, ou tolerance active non atteinte au budget), 0 sinon ;
/// comp = indice de la COMPOSANTE conservee portant le residu max a la sortie (-1 si rien d'implicite).
struct NewtonCellStat {
  Real res = Real(0);
  Real iters = Real(0);
  Real failed = Real(0);
  Real comp = Real(-1);
};

/// Rapport AGREGE (bloc entier, tous sous-pas d'une avance) du Newton de la source implicite.
/// Rempli par backward_euler_source quand un rapport est demande (diagnostics OPT-IN) ; reductions
/// max/somme sur les cellules + all_reduce MPI. reset() en tete d'avance par l'appelant.
/// CELLULE FAUTIVE : (failed_i, failed_j, failed_comp) designent UNE cellule en echec -- celle
/// d'index encode MAXIMAL (j puis i), suffisante pour aller voir l'etat ; -1 si aucune. failed_comp
/// est la composante conservee portant le pire residu de CETTE cellule.
struct NewtonReport {
  bool enabled = false;        ///< un rapport a ete calcule (au moins un sous-pas instrumente)
  bool converged = true;       ///< aucune cellule en echec sur l'avance
  Real max_residual = Real(0); ///< max cellules/sous-pas de ||F||_inf a la sortie
  Real max_iters_used = Real(0);  ///< max cellules/sous-pas des iterations consommees
  double n_failed = 0;         ///< nb (cellules x sous-pas) en echec (non-fini / pivot / non-convergence)
  double failed_i = -1, failed_j = -1, failed_comp = -1;  ///< une cellule fautive (-1 si aucune)
  void reset() { *this = NewtonReport{}; }
};

/// Fini ? (device-safe, sans <cmath> : NaN echoue x == x ; +-inf echoue les bornes). Utilise par le
/// chemin Newton INSTRUMENTE seulement (le chemin par defaut ne teste rien, bit-identique).
ADC_HD inline bool newton_finite(Real x) {
  return x == x && x < Real(1e300) && x > Real(-1e300);
}

namespace detail {
// Resolution dense J x = b sur le bloc de tete n x n (n <= N), pivot partiel. J et b
// detruits. N est constexpr (= Model::n_vars) -> tableau fixe, pas d'allocation,
// device-callable ; n (<= N) est le nombre de variables implicites (IMEX partiel).
// @return true si tous les pivots sont finis et non nuls ; false sinon (pivot degenere : la
// numerique reste STRICTEMENT celle d'origine -- la division produit inf/NaN comme avant, seul le
// drapeau est nouveau ; les appelants historiques ignorent le retour, bit-identique).
template <int N>
ADC_HD inline bool solve_dense(Real J[N][N], Real b[N], Real x[N], int n) {
  bool ok = true;
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
    if (d == Real(0) || !newton_finite(d)) ok = false;
    for (int r = 0; r < n; ++r) {
      if (r == p) continue;
      const Real f = J[r][p] / d;
      for (int c = p; c < n; ++c) J[r][c] -= f * J[p][c];
      b[r] -= f * b[p];
    }
  }
  for (int p = 0; p < n; ++p) x[p] = b[p] / J[p][p];
  return ok;
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
    const NewtonOptions& opts, const ImplicitMask<Model::n_vars>& mask = {},
    NewtonCellStat* stat = nullptr) {
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
  const bool tol_active = opts.rel_tol > Real(0) || opts.abs_tol > Real(0);
  if (!tol_active && stat == nullptr) {
    // (2a) CHEMIN HISTORIQUE (defaut) : iterations FIXES, aucun test, aucune evaluation de residu
    // supplementaire -> BIT-IDENTIQUE a l'historique pour les defauts (max_iters=2, fd_eps=1e-7).
    for (int it = 0; it < opts.max_iters; ++it) {
      const typename Model::State S0 = m.source(W, a);
      Real F[N];
      for (int r = 0; r < m_impl; ++r) {
        const int c = impl[r];
        F[r] = W[c] - Un[c] - dt * S0[c];
      }
      Real J[N][N];
      if constexpr (HasSourceJacobian<Model>) {
        // JACOBIEN ANALYTIQUE (trait, vague 3) : J = I - dt * dS/dU restreint aux implicites.
        Real dS[N][N];
        m.source_jacobian(W, a, dS);
        for (int cc = 0; cc < m_impl; ++cc)
          for (int rr = 0; rr < m_impl; ++rr) {
            const int row = impl[rr], col = impl[cc];
            J[rr][cc] = (row == col ? Real(1) : Real(0)) - dt * dS[row][col];
          }
      } else {
      for (int cc = 0; cc < m_impl; ++cc) {
        const int col = impl[cc];
        const Real wc = W[col] < 0 ? -W[col] : W[col];
        const Real h = opts.fd_eps * wc + opts.fd_eps;
        typename Model::State Wp = W;
        Wp[col] += h;
        const typename Model::State Sp = m.source(Wp, a);
        for (int rr = 0; rr < m_impl; ++rr) {
          const int row = impl[rr];
          const Real dSdW = (Sp[row] - S0[row]) / h;
          J[rr][cc] = (row == col ? Real(1) : Real(0)) - dt * dSdW;
        }
      }
      }
      Real delta[N];
      solve_dense<N>(J, F, delta, m_impl);
      for (int r = 0; r < m_impl; ++r) W[impl[r]] -= opts.damping * delta[r];
    }
    return W;
  }
  // (2b) CHEMIN INSTRUMENTE (tolerances actives et/ou stat demande) : meme Newton, plus le critere
  // d'arret ||F||_inf <= abs_tol + rel_tol*||F0||_inf en tete d'iteration, la detection de residu
  // non fini / pivot degenere, et la statistique de sortie. Une evaluation de source SUPPLEMENTAIRE
  // peut avoir lieu a la sortie (residu honnete apres la derniere mise a jour).
  //
  // INVARIANT OBSERVATEUR PUR (revue adverse) : tolerances INACTIVES + stat demande (le mode
  // newton_diagnostics) -> l'ETAT W est STRICTEMENT identique au chemin (2a), Y COMPRIS sur une
  // cellule degeneree : la detection (residu non fini, pivot degenere) MARQUE failed pour le
  // rapport mais ne change PAS le flux de controle (pas de break, mise a jour appliquee comme
  // (2a), propagation inf/NaN identique). Un diagnostic qui modifierait la trajectoire ne serait
  // pas representatif du run reel. Le SEUL arret anticipe est la CONVERGENCE sous tolerance
  // (tol_active), comportement opt-in explicitement non historique.
  Real res = Real(0), res0 = Real(0);
  int used = 0;
  int worst_comp = -1;  // composante conservee portant le residu max a la sortie (diagnostic)
  bool failed = false;
  bool converged = (m_impl == 0);  // rien d'implicite : trivialement converge
  for (int it = 0; it < opts.max_iters; ++it) {
    const typename Model::State S0 = m.source(W, a);
    Real F[N];
    res = Real(0);
    for (int r = 0; r < m_impl; ++r) {
      const int c = impl[r];
      F[r] = W[c] - Un[c] - dt * S0[c];
      const Real av = F[r] < 0 ? -F[r] : F[r];
      if (av > res) { res = av; worst_comp = c; }
    }
    if (!newton_finite(res)) failed = true;  // marque SANS break : trajectoire (2a) preservee
    if (tol_active) {
      if (it == 0) res0 = res;
      if (res <= opts.abs_tol + opts.rel_tol * res0) { converged = true; break; }
    }
    Real J[N][N];
    if constexpr (HasSourceJacobian<Model>) {
      // JACOBIEN ANALYTIQUE (trait, vague 3) : meme construction que le chemin (2a).
      Real dS[N][N];
      m.source_jacobian(W, a, dS);
      for (int cc = 0; cc < m_impl; ++cc)
        for (int rr = 0; rr < m_impl; ++rr) {
          const int row = impl[rr], col = impl[cc];
          J[rr][cc] = (row == col ? Real(1) : Real(0)) - dt * dS[row][col];
        }
    } else {
    for (int cc = 0; cc < m_impl; ++cc) {
      const int col = impl[cc];
      const Real wc = W[col] < 0 ? -W[col] : W[col];
      const Real h = opts.fd_eps * wc + opts.fd_eps;
      typename Model::State Wp = W;
      Wp[col] += h;
      const typename Model::State Sp = m.source(Wp, a);
      for (int rr = 0; rr < m_impl; ++rr) {
        const int row = impl[rr];
        const Real dSdW = (Sp[row] - S0[row]) / h;
        J[rr][cc] = (row == col ? Real(1) : Real(0)) - dt * dSdW;
      }
    }
    }
    Real delta[N];
    const bool ok = solve_dense<N>(J, F, delta, m_impl);
    if (!ok) failed = true;  // pivot degenere : marque SANS break, division inf/NaN comme (2a)
    for (int r = 0; r < m_impl; ++r) W[impl[r]] -= opts.damping * delta[r];
    used = it + 1;
  }
  // Sortie par epuisement du budget : recalculer le residu APRES la derniere mise a jour (rapport
  // honnete ; le residu de boucle precede la mise a jour). Une evaluation de source en plus,
  // uniquement sur ce chemin instrumente.
  if (!failed && used == opts.max_iters && m_impl > 0) {
    const typename Model::State S0 = m.source(W, a);
    res = Real(0);
    for (int r = 0; r < m_impl; ++r) {
      const int c = impl[r];
      const Real fr = W[c] - Un[c] - dt * S0[c];
      const Real av = fr < 0 ? -fr : fr;
      if (av > res) { res = av; worst_comp = c; }
    }
    if (!newton_finite(res)) failed = true;
    else if (tol_active) converged = res <= opts.abs_tol + opts.rel_tol * res0;
  }
  if (stat) {
    stat->res = res;
    stat->iters = Real(used);
    stat->failed = (failed || (tol_active && !converged)) ? Real(1) : Real(0);
    stat->comp = Real(worst_comp);
  }
  return W;
}

/// COMPATIBILITE : ancienne signature a budget d'iterations nu (iters). Equivaut a NewtonOptions
/// {max_iters = iters} (tolerances inactives, fd_eps historique) -> chemin (2a), bit-identique.
template <class Model>
ADC_HD inline typename Model::State newton_source_solve(
    const Model& m, const typename Model::State& Un, const Aux& a, Real dt,
    int iters, const ImplicitMask<Model::n_vars>& mask = {}) {
  NewtonOptions opts;
  opts.max_iters = iters;
  return newton_source_solve(m, Un, a, dt, opts, mask, nullptr);
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
  NewtonOptions opts;  // options Newton (POD, par valeur) ; defauts = historique bit-identique
  ImplicitMask<Model::n_vars> mask;  // masque de bloc (POD, par valeur) ; inactif = defaut modele
  ADC_HD void operator()(int i, int j) const {
    const typename Model::State Un = load_state<Model>(uc, i, j);
    const Aux a = load_aux<aux_comps<Model>()>(ax, i, j);
    const typename Model::State W = newton_source_solve<Model>(m, Un, a, dt, opts, mask, nullptr);
    for (int c = 0; c < Model::n_vars; ++c) u(i, j, c) = W[c];
  }
};

// Variante INSTRUMENTEE : meme Newton, mais ecrit la statistique de sortie de CHAQUE cellule dans
// le scratch st (comp 0 = ||F||_inf, 1 = iterations, 2 = echec 0/1, 3 = CELLULE FAUTIVE ENCODEE).
// Encodage comp 3 : -1 si la cellule n'a pas echoue ; sinon (j*2^20 + i)*16 + (comp_fautive + 1) --
// entier exact en double jusqu'a ~2^44 (i, j < 2^20), de sorte qu'une reduction MAX rend UNE cellule
// fautive (la plus grande en index) decodable cote hote sans reduction arg-max dediee. FONCTEUR
// NOMME (meme contrat device cross-TU que BackwardEulerSourceKernel). Utilisee uniquement quand un
// rapport ou une fail_policy est demande.
template <class Model>
struct BackwardEulerSourceStatKernel {
  Model m;
  ConstArray4 uc, ax;
  Array4 u, st;
  Real dt;
  NewtonOptions opts;
  ImplicitMask<Model::n_vars> mask;
  ADC_HD void operator()(int i, int j) const {
    const typename Model::State Un = load_state<Model>(uc, i, j);
    const Aux a = load_aux<aux_comps<Model>()>(ax, i, j);
    NewtonCellStat s{};
    const typename Model::State W = newton_source_solve<Model>(m, Un, a, dt, opts, mask, &s);
    for (int c = 0; c < Model::n_vars; ++c) u(i, j, c) = W[c];
    st(i, j, 0) = s.res;
    st(i, j, 1) = s.iters;
    st(i, j, 2) = s.failed;
    st(i, j, 3) = s.failed > Real(0)
                      ? (Real(j) * Real(1048576) + Real(i)) * Real(16) + (s.comp + Real(1))
                      : Real(-1);
  }
};

/// Noyaux de REDUCTION du scratch diagnostics (max / somme d'une composante). FONCTEURS NOMMES
/// passes directement a reduce_max_cell / reduce_sum_cell (chemin device-clean, cf. for_each.hpp).
struct NewtonStatMaxKernel {
  ConstArray4 st;
  int comp;
  ADC_HD void operator()(int i, int j, Real& acc) const {
    const Real v = st(i, j, comp);
    if (v > acc) acc = v;
  }
};
struct NewtonStatSumKernel {
  ConstArray4 st;
  int comp;
  ADC_HD void operator()(int i, int j, Real& acc) const { acc += st(i, j, comp); }
};
}  // namespace detail

// W = U + dt * model.source(W, aux), resolu EN PLACE par Newton local (jacobienne par differences
// finies), pilote par une politique NewtonOptions (tolerances / fd_eps / budget d'iterations).
// @p mask : masque implicite PORTE PAR LE BLOC (override du defaut modele) ; inactif (defaut) ->
// comportement bit-identique. @p report : diagnostics OPT-IN -- s'il est non nul, le Newton passe
// par le chemin instrumente (statistique par cellule dans un scratch, reductions max/somme +
// all_reduce MPI) et AGREGE dans *report (max residu, max iterations, nb d'echecs ; reset() a la
// charge de l'appelant en tete d'avance). report == nullptr ET tolerances inactives -> chemin
// historique strictement bit-identique, zero allocation, zero evaluation supplementaire.
template <class Model>
void backward_euler_source(const Model& model, const MultiFab& aux, MultiFab& U,
                           Real dt, const NewtonOptions& opts,
                           const ImplicitMask<Model::n_vars>& mask = {},
                           NewtonReport* report = nullptr) {
  // Chemin RAPIDE (historique) : ni rapport demande ni fail_policy active -> aucun scratch, aucune
  // reduction, kernel historique bit-identique. Une fail_policy != kFailNone EXIGE la detection,
  // donc le chemin instrumente (qui reste un OBSERVATEUR PUR de W).
  if (report == nullptr && opts.fail_policy == NewtonOptions::kFailNone) {
    for (int li = 0; li < U.local_size(); ++li) {
      Array4 u = U.fab(li).array();
      const ConstArray4 uc = U.fab(li).const_array();
      const ConstArray4 ax = aux.fab(li).const_array();
      const Box2D b = U.box(li);
      for_each_cell(b, detail::BackwardEulerSourceKernel<Model>{model, uc, ax, u, dt, opts, mask});
    }
    return;
  }
  // Chemin DIAGNOSTICS : scratch par-cellule (res, iters, failed, cellule encodee) puis reductions.
  // L'allocation du scratch est locale a l'appel (diagnostics/fail_policy opt-in).
  MultiFab stats(U.box_array(), U.dmap(), 4, 0);
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 u = U.fab(li).array();
    Array4 st = stats.fab(li).array();
    const ConstArray4 uc = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    const Box2D b = U.box(li);
    for_each_cell(b, detail::BackwardEulerSourceStatKernel<Model>{model, uc, ax, u, st, dt, opts,
                                                                  mask});
  }
  Real rmax = Real(0), imax = Real(0), nfail = Real(0), enc = Real(-1);
  for (int li = 0; li < stats.local_size(); ++li) {
    const ConstArray4 st = stats.fab(li).const_array();
    const Box2D b = stats.box(li);
    rmax = std::max(rmax, reduce_max_cell(b, detail::NewtonStatMaxKernel{st, 0}));
    imax = std::max(imax, reduce_max_cell(b, detail::NewtonStatMaxKernel{st, 1}));
    nfail += reduce_sum_cell(b, detail::NewtonStatSumKernel{st, 2});
    enc = std::max(enc, reduce_max_cell(b, detail::NewtonStatMaxKernel{st, 3}));
  }
  rmax = static_cast<Real>(all_reduce_max(static_cast<double>(rmax)));
  imax = static_cast<Real>(all_reduce_max(static_cast<double>(imax)));
  const double nfail_g = all_reduce_sum(static_cast<double>(nfail));
  const double enc_g = all_reduce_max(static_cast<double>(enc));
  double fi = -1, fj = -1, fc = -1;
  if (nfail_g > 0 && enc_g >= 0) {  // decode la cellule fautive d'index maximal (cf. StatKernel)
    const long long k = static_cast<long long>(enc_g);
    fc = static_cast<double>(k % 16) - 1.0;  // -1 = composante inconnue (rien d'implicite)
    const long long cell = k / 16;
    fi = static_cast<double>(cell % 1048576);
    fj = static_cast<double>(cell / 1048576);
  }
  if (report) {
    report->enabled = true;
    report->max_residual = std::max(report->max_residual, rmax);
    report->max_iters_used = std::max(report->max_iters_used, imax);
    report->n_failed += nfail_g;
    if (nfail_g > 0) { report->failed_i = fi; report->failed_j = fj; report->failed_comp = fc; }
    if (nfail_g > 0 || !newton_finite(rmax)) report->converged = false;
  }
  // FAIL_POLICY (hote, apres reductions -- les kernels device ne levent jamais) : reaction aux
  // cellules en echec. kFailWarn : un avertissement stderr (rang 0). kFailThrow : erreur dure avec
  // la cellule fautive (le pas est ABANDONNE en l'etat ; a l'appelant de decider quoi en faire).
  if (nfail_g > 0 && opts.fail_policy != NewtonOptions::kFailNone) {
    char msg[256];
    std::snprintf(msg, sizeof(msg),
                  "Newton source implicite : %.0f cellule(s) en echec (residu max %.3e ; cellule "
                  "(%g, %g), composante %g)",
                  nfail_g, static_cast<double>(rmax), fi, fj, fc);
    if (opts.fail_policy == NewtonOptions::kFailThrow) throw std::runtime_error(msg);
    if (my_rank() == 0) std::fprintf(stderr, "[adc] AVERTISSEMENT %s\n", msg);
  }
}

/// COMPATIBILITE : ancienne signature a budget d'iterations nu (iters = 2 historique). Equivaut a
/// NewtonOptions{max_iters = iters} sans rapport -> chemin historique bit-identique. Conservee pour
/// les appelants existants (coupleurs AMR, ImplicitSourceStepper, tests).
template <class Model>
void backward_euler_source(const Model& model, const MultiFab& aux, MultiFab& U,
                           Real dt, int iters = 2,
                           const ImplicitMask<Model::n_vars>& mask = {}) {
  NewtonOptions opts;
  opts.max_iters = iters;
  backward_euler_source(model, aux, U, dt, opts, mask, nullptr);
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
