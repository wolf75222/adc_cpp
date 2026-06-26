#pragma once

#include <pops/core/state/state.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/spatial_operator.hpp>  // load_state, load_aux

#include <algorithm>  // std::max (Newton report aggregation, host)
#include <concepts>
#include <cstdio>     // std::snprintf / fprintf (fail_policy: host message, never in kernel)
#include <stdexcept>  // std::runtime_error (fail_policy = throw, host after reductions)
#include <string>     // std::string (validate_newton_options message prefix)

/// @file
/// @brief Implicit / IMEX block step as a named CONTRACT. Concept ImplicitBlockStepper,
///        ready-to-use default backward_euler_source (local Newton on the model stiff source)
///        and the ImplicitSourceStepper object that wires it into SystemCoupler::step.
///        Includes the partial-IMEX mask (ImplicitMask), the analytic Jacobian trait
///        (HasSourceJacobian) and the Newton options (NewtonOptions / NewtonReport).
///
/// Layer: `include/pops/numerics/time`.
/// Role: provide "an IMEX by default without the user writing Newton". backward_euler_source
///        solves IN PLACE W = U + dt S(W, aux) by local Newton (finite-difference Jacobian,
///        or analytic if the model declares source_jacobian); exact in one iteration if S is
///        linear, quadratic convergence otherwise, unconditionally stable for a stiff relaxation
///        (where Picard would diverge as soon as dt*stiffness > 1).
///
/// Invariants:
/// - the source is LOCAL (cell by cell): no flux coupling, no reflux;
/// - PARTIAL IMEX: a model can declare variable by variable which ones are stiff
///   (PartiallyImplicitModel::is_implicit), or an ImplicitMask carried by the BLOCK overrides the
///   model default. Inactive mask (default) -> all implicit -> bit-identical to before;
/// - default NewtonOptions {} = 2 FIXED iterations, no stop test, fd_eps=1e-7,
///   damping=1, fail_policy=kFailNone -> reproduces EXACTLY the historical behavior;
/// - device kernels = NAMED functors (BackwardEulerSourceKernel, NewtonStat*Kernel);
///   fail_policy != kFailNone and Newton report aggregation act HOST side, after the
///   reductions (no snprintf/throw in kernel).

namespace pops {

// Contract of an implicit/IMEX block stepper. Any object (or lambda) that knows how to
// advance a block over dt by reading the coupler (for up-to-date aux / phi) and the model.
template <class Stepper, class Coupler, class Block>
concept ImplicitBlockStepper =
    requires(const Stepper st, Coupler& c, Block& b, Real dt, int s, int n) { st(c, b, dt, s, n); };

// OPTIONAL trait: a model can declare which conserved variables are treated
// implicitly (the stiff ones). is_implicit(c) -> bool. A model WITHOUT this trait is treated
// fully implicitly (historical default).
template <class M>
concept PartiallyImplicitModel = requires(int c) {
  { M::is_implicit(c) } -> std::convertible_to<bool>;
};

// Is component c of the model implicit? Default (no trait): all of them are.
template <class Model>
POPS_HD inline bool model_is_implicit(int c) {
  if constexpr (PartiallyImplicitModel<Model>)
    return Model::is_implicit(c);
  else
    return true;
}

// OPTIONAL trait: ANALYTIC JACOBIAN of the source (review wave 3, JacobianPolicy). When the
// model (or its source brick, forwarded by CompositeModel) declares
//   source_jacobian(U, aux, J)  with  J[r][c] = dS_r/dU_c  (FULL n_vars x n_vars matrix),
// the implicit-source Newton uses it INSTEAD of finite differences: exactness
// (no more fd_eps noise) and n_impl source evaluations saved per iteration. A model
// WITHOUT the trait keeps the historical finite differences, bit-identical. POPS_HD required.
template <class M>
concept HasSourceJacobian =
    requires(const M m, const typename M::State u, const Aux a, Real (&J)[M::n_vars][M::n_vars]) {
      m.source_jacobian(u, a, J);
    };

// Implicit mask CARRIED BY THE BLOCK / time policy (and NOT by the model): device-clean POD carrier
// (fixed array N, passed BY VALUE into the kernel, no host pointer dereference on device).
// When active (active == true), it OVERRIDES the model default (model_is_implicit): only the
// components with flag[c] == true are advanced implicitly, the others explicitly (forward Euler). This is what
// lets you REUSE the SAME model with different implicit treatments depending on the block. Inactive (default:
// active == false) -> falls back to model_is_implicit -> behavior bit-identical to the historical one.
template <int N>
struct ImplicitMask {
  bool active = false;
  bool flag[N] = {};
};

// Is component c implicit, with the block mask TAKING PRIORITY over the model default? The inactive mask
// (default) delegates to model_is_implicit<Model> -> strictly identical to before this change.
template <class Model, int N>
POPS_HD inline bool is_implicit_component(const ImplicitMask<N>& mask, int c) {
  if (mask.active)
    return mask.flag[c];
  return model_is_implicit<Model>(c);
}

/// Options of the local Newton of the implicit source (backward-Euler). The DEFAULTS reproduce
/// EXACTLY the historical behavior: 2 FIXED iterations, no residual evaluation
/// (rel_tol == abs_tol == 0 -> no stop test, no extra cost), finite-difference
/// Jacobian step h = fd_eps*|w| + fd_eps with fd_eps = 1e-7 (the constant
/// historically hard-coded). Device-clean POD (passed BY VALUE into the kernel).
///  - max_iters: Newton iteration budget (historical: 2).
///  - rel_tol / abs_tol: stop criterion ||F||_inf <= abs_tol + rel_tol*||F0||_inf, evaluated per
///    CELL at the start of an iteration. 0/0 (default) = disabled -> bit-identical historical loop.
///  - fd_eps: step (relative AND absolute floor) of the finite-difference Jacobian.
///  - damping: damping factor of the update W -= damping * delta. 1 (default) =
///    full Newton, bit-identical (multiplication by 1.0 exact in IEEE). < 1 = damped Newton
///    (very stiff source / poor conditioning: robustness at the cost of speed).
///  - fail_policy: HOST reaction (after reduction) to failed cells --
///    kFailNone (default) = record only (historical); kFailWarn = one stderr warning
///    per advance; kFailThrow = std::runtime_error with the offending cell. != kFailNone
///    forces the instrumented path (detection required) -- a pure observer, W unchanged.
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

/// Range-validate a NewtonOptions POD; shared by System::add_block and AmrSystem::add_block, which
/// carried this defensive check verbatim. @p where prefixes each message ("System::add_block" /
/// "AmrSystem::add_block"). fail_policy is already a valid integer (the bindings resolve it from the
/// string "none"/"warn"/"throw"); the range stays defensive. This does NOT decide whether non-default
/// options are ALLOWED -- the time='imex' gate (and System's extra newton_diagnostics term in the
/// non-default test) differ between the two callers and remain at each call site.
inline void validate_newton_options(const NewtonOptions& newton, const char* where) {
  const std::string ctx = std::string(where) + " : ";
  if (newton.max_iters < 1)
    throw std::runtime_error(ctx + "newton_max_iters >= 1");
  if (newton.rel_tol < 0.0 || newton.abs_tol < 0.0 || newton.fd_eps <= 0.0)
    throw std::runtime_error(ctx + "newton_rel_tol/abs_tol >= 0 and newton_fd_eps > 0");
  if (!(newton.damping > 0.0 && newton.damping <= 1.0))
    throw std::runtime_error(ctx + "newton_damping in (0, 1]");
  if (newton.fail_policy != NewtonOptions::kFailNone &&
      newton.fail_policy != NewtonOptions::kFailWarn &&
      newton.fail_policy != NewtonOptions::kFailThrow)
    throw std::runtime_error(ctx +
                             "newton_fail_policy invalid "
                             "(NewtonOptions::kFailNone|kFailWarn|kFailThrow)");
}

/// OUTPUT statistic of the Newton of ONE cell (device POD, written into the diagnostics scratch):
/// res = ||F||_inf at exit; iters = iterations consumed; failed = 1 if the cell failed
/// (non-finite residual, degenerate/non-finite pivot, or active tolerance not reached within budget), 0 otherwise;
/// comp = index of the conserved COMPONENT carrying the max residual at exit (-1 if nothing implicit).
struct NewtonCellStat {
  Real res = Real(0);
  Real iters = Real(0);
  Real failed = Real(0);
  Real comp = Real(-1);
};

/// AGGREGATED report (whole block, all substeps of one advance) of the implicit-source Newton.
/// Filled by backward_euler_source when a report is requested (OPT-IN diagnostics); max/sum
/// reductions over the cells + MPI all_reduce. reset() at the start of the advance by the caller.
/// OFFENDING CELL: (failed_i, failed_j, failed_comp) designate ONE failed cell -- the one
/// with MAXIMAL encoded index (j then i), enough to go inspect the state; -1 if none. failed_comp
/// is the conserved component carrying the worst residual of THAT cell.
struct NewtonReport {
  bool enabled = false;           ///< a report was computed (at least one instrumented substep)
  bool converged = true;          ///< no failed cell over the advance
  Real max_residual = Real(0);    ///< max over cells/substeps of ||F||_inf at exit
  Real max_iters_used = Real(0);  ///< max over cells/substeps of iterations consumed
  double n_failed =
      0;  ///< number of (cells x substeps) failed (non-finite / pivot / non-convergence)
  double failed_i = -1, failed_j = -1, failed_comp = -1;  ///< one offending cell (-1 if none)
  void reset() { *this = NewtonReport{}; }
};

/// Finite? (device-safe, without <cmath>: NaN fails x == x; +-inf fails the bounds). Used by the
/// INSTRUMENTED Newton path only (the default path tests nothing, bit-identical).
POPS_HD inline bool newton_finite(Real x) {
  return x == x && x < Real(1e300) && x > Real(-1e300);
}

namespace detail {
// Dense solve J x = b on the leading n x n block (n <= N), partial pivoting. J and b
// destroyed. N is constexpr (= Model::n_vars) -> fixed array, no allocation,
// device-callable; n (<= N) is the number of implicit variables (partial IMEX).
// @return true if all pivots are finite and non-zero; false otherwise (degenerate pivot: the
// numerics stays STRICTLY the original one -- the division produces inf/NaN as before, only the
// flag is new; historical callers ignore the return, bit-identical).
template <int N>
POPS_HD inline bool solve_dense(Real J[N][N], Real b[N], Real x[N], int n) {
  bool ok = true;
  for (int p = 0; p < n; ++p) {
    int piv = p;
    Real best = J[p][p] < 0 ? -J[p][p] : J[p][p];
    for (int r = p + 1; r < n; ++r) {
      const Real v = J[r][p] < 0 ? -J[r][p] : J[r][p];
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (piv != p) {
      for (int c = 0; c < n; ++c) {
        const Real t = J[p][c];
        J[p][c] = J[piv][c];
        J[piv][c] = t;
      }
      const Real t = b[p];
      b[p] = b[piv];
      b[piv] = t;
    }
    const Real d = J[p][p];
    if (d == Real(0) || !newton_finite(d))
      ok = false;
    for (int r = 0; r < n; ++r) {
      if (r == p)
        continue;
      const Real f = J[r][p] / d;
      for (int c = p; c < n; ++c)
        J[r][c] -= f * J[p][c];
      b[r] -= f * b[p];
    }
  }
  for (int p = 0; p < n; ++p)
    x[p] = b[p] / J[p][p];
  return ok;
}

// Assemble the Newton Jacobian of the subsystem reduced to the implicit components:
//   J[rr][cc] = (row == col ? 1: 0) - dt * dS_row/dW_col,   row = impl[rr], col = impl[cc].
// HasSourceJacobian trait present => ANALYTIC Jacobian (m.source_jacobian); otherwise finite
// differences (step h = fd_eps*|W| + fd_eps, source perturbed column by column around S0 = S(W)).
// Body EXTRACTED word-for-word from the two paths (2a defaults, 2b instrumented) -> bit-identical;
// POPS_HD because called from newton_source_solve (device-callable). S0 = m.source(W, a) already computed
// by the caller (reused as is by the finite differences).
template <class Model, int N>
POPS_HD inline void assemble_newton_jacobian(const Model& m, const typename Model::State& W,
                                            const Aux& a, Real dt, const NewtonOptions& opts,
                                            const int impl[N], int m_impl,
                                            const typename Model::State& S0, Real J[N][N]) {
  if constexpr (HasSourceJacobian<Model>) {
    // ANALYTIC JACOBIAN (trait, wave 3): J = I - dt * dS/dU restricted to the implicit ones.
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
}

// Solve W such that W = Un + dt*S(W,a) in forward-backward Euler (partial IMEX):
//   - EXPLICIT components: forward Euler at the input state, W_e = Un_e + dt*S_e(Un);
//   - IMPLICIT components: Newton on the reduced subsystem, F_i = W_i - Un_i -
//     dt*S_i(W), Jacobian I - dt*(dS/dW) restricted to the implicit ones (columns by
//     finite differences), the explicit ones frozen at their advanced value (known data).
// WHO is implicit: a mask CARRIED BY THE BLOCK (@p mask) taking priority over the model default
// (is_implicit_component). Inactive mask (default) + model without is_implicit trait: all
// components are implicit -> full backward-Euler, strictly identical to the original behavior.
template <class Model>
POPS_HD inline typename Model::State newton_source_solve(
    const Model& m, const typename Model::State& Un, const Aux& a, Real dt,
    const NewtonOptions& opts, const ImplicitMask<Model::n_vars>& mask = {},
    NewtonCellStat* stat = nullptr) {
  constexpr int N = Model::n_vars;
  int impl[N];  // indices of the implicit components (the first m_impl useful slots)
  int m_impl = 0;
  for (int c = 0; c < N; ++c)
    if (is_implicit_component<Model>(mask, c))
      impl[m_impl++] = c;

  typename Model::State W = Un;
  // (1) explicit: forward Euler on the non-implicit components (source at the input).
  if (m_impl < N) {
    const typename Model::State S_in = m.source(Un, a);
    for (int c = 0; c < N; ++c)
      if (!is_implicit_component<Model>(mask, c))
        W[c] = Un[c] + dt * S_in[c];
  }
  const bool tol_active = opts.rel_tol > Real(0) || opts.abs_tol > Real(0);
  if (!tol_active && stat == nullptr) {
    // (2a) HISTORICAL PATH (default): FIXED iterations, no test, no extra residual evaluation
    // -> BIT-IDENTICAL to the historical one for the defaults (max_iters=2, fd_eps=1e-7).
    for (int it = 0; it < opts.max_iters; ++it) {
      const typename Model::State S0 = m.source(W, a);
      Real F[N];
      for (int r = 0; r < m_impl; ++r) {
        const int c = impl[r];
        F[r] = W[c] - Un[c] - dt * S0[c];
      }
      Real J[N][N];
      assemble_newton_jacobian<Model, N>(m, W, a, dt, opts, impl, m_impl, S0, J);
      Real delta[N];
      solve_dense<N>(J, F, delta, m_impl);
      for (int r = 0; r < m_impl; ++r)
        W[impl[r]] -= opts.damping * delta[r];
    }
    return W;
  }
  // (2b) INSTRUMENTED PATH (active tolerances and/or stat requested): same Newton, plus the stop
  // criterion ||F||_inf <= abs_tol + rel_tol*||F0||_inf at the start of an iteration, the detection of
  // non-finite residual / degenerate pivot, and the exit statistic. One ADDITIONAL source evaluation
  // may happen at exit (honest residual after the last update).
  //
  // PURE-OBSERVER INVARIANT (adversarial review): tolerances INACTIVE + stat requested (the
  // newton_diagnostics mode) -> the STATE W is STRICTLY identical to path (2a), INCLUDING on a
  // degenerate cell: the detection (non-finite residual, degenerate pivot) MARKS failed for the
  // report but does NOT change the control flow (no break, update applied as in
  // (2a), identical inf/NaN propagation). A diagnostic that would alter the trajectory would not be
  // representative of the real run. The ONLY early exit is CONVERGENCE under tolerance
  // (tol_active), explicitly non-historical opt-in behavior.
  Real res = Real(0), res0 = Real(0);
  int used = 0;
  int worst_comp = -1;  // conserved component carrying the max residual at exit (diagnostic)
  bool failed = false;
  bool converged = (m_impl == 0);  // nothing implicit: trivially converged
  for (int it = 0; it < opts.max_iters; ++it) {
    const typename Model::State S0 = m.source(W, a);
    Real F[N];
    res = Real(0);
    for (int r = 0; r < m_impl; ++r) {
      const int c = impl[r];
      F[r] = W[c] - Un[c] - dt * S0[c];
      const Real av = F[r] < 0 ? -F[r] : F[r];
      if (av > res) {
        res = av;
        worst_comp = c;
      }
    }
    if (!newton_finite(res))
      failed = true;  // marks WITHOUT break: trajectory (2a) preserved
    if (tol_active) {
      if (it == 0)
        res0 = res;
      if (res <= opts.abs_tol + opts.rel_tol * res0) {
        converged = true;
        break;
      }
    }
    Real J[N][N];
    assemble_newton_jacobian<Model, N>(m, W, a, dt, opts, impl, m_impl, S0, J);
    Real delta[N];
    const bool ok = solve_dense<N>(J, F, delta, m_impl);
    if (!ok)
      failed = true;  // degenerate pivot: marks WITHOUT break, inf/NaN division as in (2a)
    for (int r = 0; r < m_impl; ++r)
      W[impl[r]] -= opts.damping * delta[r];
    used = it + 1;
  }
  // Exit by budget exhaustion: recompute the residual AFTER the last update (honest
  // report; the loop residual precedes the update). One extra source evaluation,
  // only on this instrumented path.
  if (!failed && used == opts.max_iters && m_impl > 0) {
    const typename Model::State S0 = m.source(W, a);
    res = Real(0);
    for (int r = 0; r < m_impl; ++r) {
      const int c = impl[r];
      const Real fr = W[c] - Un[c] - dt * S0[c];
      const Real av = fr < 0 ? -fr : fr;
      if (av > res) {
        res = av;
        worst_comp = c;
      }
    }
    if (!newton_finite(res))
      failed = true;
    else if (tol_active)
      converged = res <= opts.abs_tol + opts.rel_tol * res0;
  }
  if (stat) {
    stat->res = res;
    stat->iters = Real(used);
    stat->failed = (failed || (tol_active && !converged)) ? Real(1) : Real(0);
    stat->comp = Real(worst_comp);
  }
  return W;
}

/// COMPATIBILITY: old signature with a bare iteration budget (iters). Equivalent to NewtonOptions
/// {max_iters = iters} (tolerances inactive, historical fd_eps) -> path (2a), bit-identical.
template <class Model>
POPS_HD inline typename Model::State newton_source_solve(
    const Model& m, const typename Model::State& Un, const Aux& a, Real dt, int iters,
    const ImplicitMask<Model::n_vars>& mask = {}) {
  NewtonOptions opts;
  opts.max_iters = iters;
  return newton_source_solve(m, Un, a, dt, opts, mask, nullptr);
}
}  // namespace detail

namespace detail {
// Device kernel of the implicit step on the source (local Newton in place). NAMED FUNCTOR (and not an
// extended lambda): ROBUST device emission when the Model-template kernel is instantiated from an EXTERNAL TU
// (IMEX path of an add_compiled_model block, via the advance std::function of block_builder). Body
// identical to the old lambda -> bit-identical result on CPU.
template <class Model>
struct BackwardEulerSourceKernel {
  Model m;
  ConstArray4 uc, ax;
  Array4 u;
  Real dt;
  NewtonOptions opts;  // Newton options (POD, by value); defaults = historical bit-identical
  ImplicitMask<Model::n_vars> mask;  // block mask (POD, by value); inactive = model default
  POPS_HD void operator()(int i, int j) const {
    const typename Model::State Un = load_state<Model>(uc, i, j);
    const Aux a = load_aux<aux_comps<Model>()>(ax, i, j);
    const typename Model::State W = newton_source_solve<Model>(m, Un, a, dt, opts, mask, nullptr);
    for (int c = 0; c < Model::n_vars; ++c)
      u(i, j, c) = W[c];
  }
};

// INSTRUMENTED variant: same Newton, but writes the exit statistic of EACH cell into
// the scratch st (comp 0 = ||F||_inf, 1 = iterations, 2 = failure 0/1, 3 = ENCODED OFFENDING CELL).
// Encoding comp 3: -1 if the cell did not fail; otherwise (j*2^20 + i)*16 + (offending_comp + 1) --
// exact integer in double up to ~2^44 (i, j < 2^20), so that a MAX reduction yields ONE offending
// cell (the largest in index) decodable host side without a dedicated arg-max reduction. NAMED
// FUNCTOR (same cross-TU device contract as BackwardEulerSourceKernel). Used only when a
// report or a fail_policy is requested.
template <class Model>
struct BackwardEulerSourceStatKernel {
  Model m;
  ConstArray4 uc, ax;
  Array4 u, st;
  Real dt;
  NewtonOptions opts;
  ImplicitMask<Model::n_vars> mask;
  POPS_HD void operator()(int i, int j) const {
    const typename Model::State Un = load_state<Model>(uc, i, j);
    const Aux a = load_aux<aux_comps<Model>()>(ax, i, j);
    NewtonCellStat s{};
    const typename Model::State W = newton_source_solve<Model>(m, Un, a, dt, opts, mask, &s);
    for (int c = 0; c < Model::n_vars; ++c)
      u(i, j, c) = W[c];
    st(i, j, 0) = s.res;
    st(i, j, 1) = s.iters;
    st(i, j, 2) = s.failed;
    st(i, j, 3) = s.failed > Real(0)
                      ? (Real(j) * Real(1048576) + Real(i)) * Real(16) + (s.comp + Real(1))
                      : Real(-1);
  }
};

/// REDUCTION kernels of the diagnostics scratch (max / sum of one component). NAMED FUNCTORS
/// passed directly to reduce_max_cell / reduce_sum_cell (device-clean path, cf. for_each.hpp).
struct NewtonStatMaxKernel {
  ConstArray4 st;
  int comp;
  POPS_HD void operator()(int i, int j, Real& acc) const {
    const Real v = st(i, j, comp);
    if (v > acc)
      acc = v;
  }
};
struct NewtonStatSumKernel {
  ConstArray4 st;
  int comp;
  POPS_HD void operator()(int i, int j, Real& acc) const { acc += st(i, j, comp); }
};
}  // namespace detail

// W = U + dt * model.source(W, aux), solved IN PLACE by local Newton (finite-difference
// Jacobian), driven by a NewtonOptions policy (tolerances / fd_eps / iteration budget).
// @p mask: implicit mask CARRIED BY THE BLOCK (override of the model default); inactive (default) ->
// bit-identical behavior. @p report: OPT-IN diagnostics -- if non-null, the Newton goes
// through the instrumented path (per-cell statistic in a scratch, max/sum reductions +
// MPI all_reduce) and AGGREGATES into *report (max residual, max iterations, number of failures; reset() is
// the caller's responsibility at the start of the advance). report == nullptr AND tolerances inactive -> historical
// path strictly bit-identical, zero allocation, zero extra evaluation.
template <class Model>
void backward_euler_source(const Model& model, const MultiFab& aux, MultiFab& U, Real dt,
                           const NewtonOptions& opts, const ImplicitMask<Model::n_vars>& mask = {},
                           NewtonReport* report = nullptr) {
  // FAST path (historical): neither a report requested nor an active fail_policy -> no scratch, no
  // reduction, historical kernel bit-identical. A fail_policy != kFailNone REQUIRES the detection,
  // hence the instrumented path (which remains a PURE OBSERVER of W).
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
  // DIAGNOSTICS path: per-cell scratch (res, iters, failed, encoded cell) then reductions.
  // The scratch allocation is local to the call (opt-in diagnostics/fail_policy).
  MultiFab stats(U.box_array(), U.dmap(), 4, 0);
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 u = U.fab(li).array();
    Array4 st = stats.fab(li).array();
    const ConstArray4 uc = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    const Box2D b = U.box(li);
    for_each_cell(
        b, detail::BackwardEulerSourceStatKernel<Model>{model, uc, ax, u, st, dt, opts, mask});
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
  if (nfail_g > 0 && enc_g >= 0) {  // decode the offending cell with maximal index (cf. StatKernel)
    const long long k = static_cast<long long>(enc_g);
    fc = static_cast<double>(k % 16) - 1.0;  // -1 = unknown component (nothing implicit)
    const long long cell = k / 16;
    fi = static_cast<double>(cell % 1048576);
    fj = static_cast<double>(cell / 1048576);
  }
  if (report) {
    report->enabled = true;
    report->max_residual = std::max(report->max_residual, rmax);
    report->max_iters_used = std::max(report->max_iters_used, imax);
    report->n_failed += nfail_g;
    if (nfail_g > 0) {
      report->failed_i = fi;
      report->failed_j = fj;
      report->failed_comp = fc;
    }
    if (nfail_g > 0 || !newton_finite(rmax))
      report->converged = false;
  }
  // FAIL_POLICY (host, after reductions -- the device kernels never throw): reaction to the
  // failed cells. kFailWarn: one stderr warning (rank 0). kFailThrow: hard error with
  // the offending cell (the step is ABANDONED as is; up to the caller to decide what to do with it).
  if (nfail_g > 0 && opts.fail_policy != NewtonOptions::kFailNone) {
    char msg[256];
    std::snprintf(msg, sizeof(msg),
                  "Implicit source Newton: %.0f cell(s) failed (max residual %.3e; cell "
                  "(%g, %g), component %g)",
                  nfail_g, static_cast<double>(rmax), fi, fj, fc);
    if (opts.fail_policy == NewtonOptions::kFailThrow)
      throw std::runtime_error(msg);
    if (my_rank() == 0)
      std::fprintf(stderr, "[adc] WARNING %s\n", msg);
  }
}

/// COMPATIBILITY: old signature with a bare iteration budget (iters = 2 historical). Equivalent to
/// NewtonOptions{max_iters = iters} without report -> historical path bit-identical. Kept for
/// existing callers (AMR couplers, ImplicitSourceStepper, tests).
template <class Model>
void backward_euler_source(const Model& model, const MultiFab& aux, MultiFab& U, Real dt,
                           int iters = 2, const ImplicitMask<Model::n_vars>& mask = {}) {
  NewtonOptions opts;
  opts.max_iters = iters;
  backward_euler_source(model, aux, U, dt, opts, mask, nullptr);
}

// Default implicit stepper: backward-Euler (Newton) on the model source.
// Models ImplicitBlockStepper; passed as is to SystemCoupler::step as the implicit
// advance callback. The user writes no solver.
struct ImplicitSourceStepper {
  int iters = 2;

  template <class Coupler, class Block>
  void operator()(Coupler& coupler, Block& block, Real dt, int /*substep*/, int /*nsub*/) const {
    backward_euler_source(block.model, coupler.aux(), block.U(), dt, iters);
  }
};

}  // namespace pops
