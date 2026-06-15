/// @file
/// @brief Cartesian spatial operator: assembles R(U, aux) = -div F + S over the cells of a level.
///
/// This is the "PDE -> ODE system" arrow of the method of lines. The time integrator
/// (time/) only knows R; it is unaware of the geometry and the reconstruction scheme.
///
/// Exposed functions and types:
///   - DiffusiveModel: optional concept (model.diffusivity() -> nu); the Fickian flux
///                             F = -nu grad U is added to the hyperbolic flux when present.
///   - SourceFreeModel<M>: adapter that zeroes the source (explicit IMEX half-step).
///   - load_state<Model>: reads the conservative state from an Array4 (ADC_HD).
///   - load_aux<NComp>: reads the auxiliary (phi, grad, extra fields) (ADC_HD).
///   - max_wave_speed_mf: max CFL over the whole MultiFab (reduction + MPI all_reduce).
///   - rusanov_flux: free compat, delegates to RusanovFlux{}.
///   - reconstruct<>: face value from the MUSCL or WENO5 stencil (ADC_HD).
///   - compute_face_fluxes<>: face fluxes (for AMR reflux).
///   - assemble_rhs<>: residual R = -div Fhat + S (+ diffusion) over the box.
///
/// INVARIANT: the Cartesian operator is STRICTLY UNTOUCHED by the polar operator
/// (spatial_operator_polar.hpp); a run on a Cartesian mesh is bit-identical.

#pragma once

#include <adc/core/state.hpp>
#include <adc/core/physical_model.hpp>  // HasPrimitiveVars: optional primitive reconstruction
#include <adc/core/types.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/numerics/numerical_flux.hpp>
#include <adc/numerics/reconstruction.hpp>

#include <algorithm>
#include <concepts>
#include <stdexcept>  // positivity_comp: model without Density role -> clear error

namespace adc {

// aux_comps<Model>() (aux channel width of a model) now lives in the contract header
// adc/core/physical_model.hpp (included above) so that CompositeModel can propagate it.

/// DiffusiveModel: optional concept for models with isotropic scalar diffusion.
///
/// A model satisfies DiffusiveModel if and only if m.diffusivity() -> Real (nu >= 0).
/// The Fickian flux F = -nu grad U is ADDED to the hyperbolic flux in assemble_rhs /
/// compute_face_fluxes. The divergence yields exactly +nu Lap(U).
/// INVARIANT: a model without diffusivity() does not change by a single bit (the hyperbolic
/// path is strictly unchanged -- the if constexpr is false, zero extra codegen).
template <class M>
concept DiffusiveModel = requires(const M m) {
  { m.diffusivity() } -> std::convertible_to<Real>;
};

/// SourceFreeModel<M>: adapter that cancels the source of M (explicit IMEX half-step).
///
/// Same flux and max_wave_speed as M, but source() always returns the zero state.
/// Used for the EXPLICIT half-step of an IMEX scheme (transport only, -div F); the stiff source
/// is treated implicitly by backward_euler_source. Does not expose diffusivity() so as not to
/// break non-diffusive models. Transparent to the HLL/HLLC contract: forwards pressure and
/// wave_speeds only if M exposes them (requires clause).
template <class M>
struct SourceFreeModel {
  using State = typename M::State;
  using Aux = typename M::Aux;
  static constexpr int n_vars = M::n_vars;
  static constexpr int n_aux = aux_comps<M>();  // transparent to the wrapped model's aux width
  M m;
  ADC_HD State flux(const State& u, const Aux& a, int dir) const { return m.flux(u, a, dir); }
  ADC_HD Real max_wave_speed(const State& u, const Aux& a, int dir) const {
    return m.max_wave_speed(u, a, dir);
  }
  ADC_HD State source(const State&, const Aux&) const { return State{}; }
  ADC_HD Real elliptic_rhs(const State& u) const { return m.elliptic_rhs(u); }
  // SourceFreeModel does not expose the primitive variables: the explicit IMEX half-step that
  // uses it therefore reconstructs in conservative variables (the direct explicit path itself
  // has the composite model's conversions and can reconstruct in primitive variables).
  // Transparent to the HLL/HLLC contract: forwards pressure and signed wave speeds ONLY if M
  // exposes them (requires clause), so an IMEX half-step can stay on the HLLC flux.
  ADC_HD Real pressure(const State& u) const
    requires requires(const M& mm, const State& s) { mm.pressure(s); }
  {
    return m.pressure(u);
  }
  ADC_HD void wave_speeds(const State& u, const Aux& a, int dir, Real& smin, Real& smax) const
    requires requires(const M& mm, const State& s, const Aux& aa, int d, Real& lo, Real& hi) {
      mm.wave_speeds(s, aa, d, lo, hi);
    }
  {
    m.wave_speeds(u, a, dir, smin, smax);
  }
  // Forward the VariableSet introspection (HOST): lets positivity_comp resolve the Density role
  // through the explicit IMEX half-step. Conditional (requires), like pressure / wave_speeds.
  static VariableSet conservative_vars()
    requires requires { M::conservative_vars(); }
  {
    return M::conservative_vars();
  }
};

/// load_state<Model>: reads Model::n_vars scalars at (i,j) from an Array4.
///
/// Returns a StateVec<n_vars> initialized from components 0..n_vars-1 of the channel.
/// ADC_HD, zero allocation. Does NOT read components beyond n_vars.
template <class Model>
ADC_HD inline typename Model::State load_state(const ConstArray4& a, int i,
                                              int j) {
  typename Model::State u;
  for (int c = 0; c < Model::n_vars; ++c) u[c] = a(i, j, c);
  return u;
}

/// load_aux<NComp>: reads NComp components of the auxiliary from an Array4 at (i,j).
///
/// The first 3 components (phi, grad_x, grad_y) are the base contract.
/// Components >= 3 (B_z, T_e...) are read only if NComp > their canonical index
/// (if constexpr guard -> zero codegen for NComp = kAuxBaseComps = 3: bit-identical).
/// The extra fields are governed by ADC_AUX_FIELDS (state.hpp): adding a field =>
/// 1 line in ADC_AUX_FIELDS, not in this path. ADC_HD.
//
// The extra fields are loaded from the SINGLE SOURCE ADC_AUX_FIELDS (state.hpp): each
// X(name, idx) generates `if constexpr (NComp > idx) x.name = a(i,j,idx);`, exactly the
// sequence written by hand before. Adding an extra field => 1 line in ADC_AUX_FIELDS is
// enough for this device read path to cover it (and the host marshaling, generated from the
// same table). NComp = kAuxBaseComps: all guards are false -> bit-identical.
template <int NComp = kAuxBaseComps>
ADC_HD inline Aux load_aux(const ConstArray4& a, int i, int j) {
  Aux x{a(i, j, 0), a(i, j, 1), a(i, j, 2)};
#define ADC_AUX_LOAD(name, idx) \
  if constexpr (NComp > (idx)) x.name = a(i, j, idx);
  ADC_AUX_FIELDS(ADC_AUX_LOAD)
#undef ADC_AUX_LOAD
  // NAMED aux fields (ADC-70 phase 1): components from kAuxNamedBase (= 5). Loaded
  // ONLY if the model declares n_aux > kAuxNamedBase (otherwise if constexpr false -> no codegen,
  // NComp = kAuxBaseComps stays strictly bit-identical). The bound n_extra is known at
  // compile time (NComp template): the loop is unrolled and clamped to kAuxMaxExtra (size of
  // x.extra) -- never an out-of-bounds access on the C array, device-clean.
  if constexpr (NComp > kAuxNamedBase) {
    constexpr int n_extra =
        (NComp - kAuxNamedBase) < kAuxMaxExtra ? (NComp - kAuxNamedBase) : kAuxMaxExtra;
    for (int k = 0; k < n_extra; ++k) x.extra[k] = a(i, j, kAuxNamedBase + k);
  }
  return x;
}

namespace detail {
/// MaxWaveSpeedKernel<Model>: device reduction functor for max_wave_speed_mf.
///
/// Accumulates the max of the wave speeds in both directions at cell (i,j).
/// Named functor (and not an extended lambda): robust device emission from an external TU
/// (add_compiled_model). Body bit-identical to the former lambda. ADC_HD.
template <class Model>
struct MaxWaveSpeedKernel {
  Model model;
  ConstArray4 u, a;
  ADC_HD void operator()(int i, int j, Real& acc) const {
    const auto s = load_state<Model>(u, i, j);
    const Aux ax = load_aux<aux_comps<Model>()>(a, i, j);
    const Real wx = model.max_wave_speed(s, ax, 0);
    const Real wy = model.max_wave_speed(s, ax, 1);
    const Real w = wx > wy ? wx : wy;
    if (w > acc) acc = w;
  }
};
}  // namespace detail

/// max_wave_speed_mf: global max of the wave speed over the whole MultiFab (CFL).
///
/// Reduce over all local boxes then all_reduce_max over all MPI ranks.
/// Without the all_reduce, each rank only sees its boxes and step_cfl computes a different
/// dt per rank (desynchronization / divergence). In serial all_reduce_max is the identity.
/// For a model without transport (max_wave_speed = 0 everywhere) -> returns 0 (step unconstrained).
//
// COLLECTIVE UNDER MPI: we aggregate via all_reduce_max over ALL ranks (same convention as
// AmrCouplerMp::max_wave_speed and GeometricMG::current_residual). Without this all-reduce, each
// rank only sees the max of ITS boxes: step_cfl / step_adaptive then choose a DIFFERENT dt per
// rank (the rank whose local max is lower takes too large a step) and the simulation diverges or
// desynchronizes the ranks. In serial all_reduce_max is the identity (behavior unchanged).
template <class Model>
inline Real max_wave_speed_mf(const Model& model, const MultiFab& U,
                              const MultiFab& aux) {
  Real m = 0;
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    m = std::max(m, reduce_max_cell(U.box(li), detail::MaxWaveSpeedKernel<Model>{model, u, a}));
  }
  return static_cast<Real>(all_reduce_max(static_cast<double>(m)));
}

namespace detail {
/// Locates the cell DOMINATING the CFL (dt_hotspot diagnostic, ADC-182): EQUALITY scan
/// of the recomputed w -- same functor and same data as MaxWaveSpeedKernel, hence bit-equal
/// to the max returned by max_wave_speed_mf -- which encodes the GLOBAL index j*nx + i as
/// Real (exact as long as nx*ny < 2^53) and reduces to the MIN (first cell in lexicographic
/// order: deterministic). NAMED functor (cross-TU instantiation under nvcc).
template <class Model>
struct WaveSpeedMatchKernel {
  Model model;
  ConstArray4 u, a;
  Real target;
  Real nx;  // encoding stride (nx of the DOMAIN, global indices)
  ADC_HD void operator()(int i, int j, Real& acc) const {
    const auto s = load_state<Model>(u, i, j);
    const Aux ax = load_aux<aux_comps<Model>()>(a, i, j);
    const Real wx = model.max_wave_speed(s, ax, 0);
    const Real wy = model.max_wave_speed(s, ax, 1);
    const Real w = wx > wy ? wx : wy;
    if (w == target) {
      const Real idx = static_cast<Real>(j) * nx + static_cast<Real>(i);
      if (idx < acc) acc = idx;
    }
  }
};
}  // namespace detail

/// dt_hotspot diagnostic (ADC-182): the cell (GLOBAL indices) that dominates the block's transport
/// CFL bound, and its speed w = max(wx, wy). ON DEMAND only -- two full passes (max then location
/// by bit-exact equality), step_cfl does not touch it (bit-identical). MPI: all_reduce of the max
/// then all_reduce_min of the encoded index (+inf on the non-holder ranks). @p nx: domain width
/// (encoding j*nx + i).
template <class Model>
inline void max_wave_speed_hotspot_mf(const Model& model, const MultiFab& U,
                                      const MultiFab& aux, int nx,
                                      Real& w_out, int& i_out, int& j_out) {
  const Real w = max_wave_speed_mf(model, U, aux);
  Real best = std::numeric_limits<Real>::infinity();
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    best = std::min(best, reduce_min_cell(U.box(li), detail::WaveSpeedMatchKernel<Model>{
                                              model, u, a, w, static_cast<Real>(nx)}));
  }
  best = static_cast<Real>(all_reduce_min(static_cast<double>(best)));
  w_out = w;
  // identity of Kokkos::Min = max_real (finite): a rank/box without a cell equaling the max
  // leaves this value -> we only decode if a REAL index was encoded.
  if (best >= Real(0) && best < std::numeric_limits<Real>::max() * Real(0.5)) {
    const long long idx = static_cast<long long>(best);
    i_out = static_cast<int>(idx % nx);
    j_out = static_cast<int>(idx / nx);
  } else {  // empty domain / degenerate state: no cell (w may be 0)
    i_out = -1;
    j_out = -1;
  }
}

// ============================================================================
// OPTIONAL STEP-BOUND REDUCTIONS (audit 2026-06, step_cfl effort).
// Counterparts of max_wave_speed_mf for the HasStabilitySpeed / HasSourceFrequency /
// HasStabilityDt traits (cf. core/physical_model.hpp). Same conventions: reduction via the seam
// (device under Kokkos), MPI all_reduce (without which each rank would choose a different dt).
// Instantiated ONLY for a model declaring the trait (if constexpr on the block_builder side):
// zero codegen, zero cost for a legacy model.
// ============================================================================

namespace detail {
/// StabilitySpeedKernel: max over cells/directions of model.stability_speed (replaces
/// MaxWaveSpeedKernel when the trait is declared). Named functor (device-clean cross-TU).
template <class Model>
struct StabilitySpeedKernel {
  Model model;
  ConstArray4 u, a;
  ADC_HD void operator()(int i, int j, Real& acc) const {
    const auto s = load_state<Model>(u, i, j);
    const Aux ax = load_aux<aux_comps<Model>()>(a, i, j);
    const Real wx = model.stability_speed(s, ax, 0);
    const Real wy = model.stability_speed(s, ax, 1);
    const Real w = wx > wy ? wx : wy;
    if (w > acc) acc = w;
  }
};

/// SourceFrequencyKernel: max over cells of model.source_frequency (mu >= 0, 1/s).
template <class Model>
struct SourceFrequencyKernel {
  Model model;
  ConstArray4 u, a;
  ADC_HD void operator()(int i, int j, Real& acc) const {
    const auto s = load_state<Model>(u, i, j);
    const Aux ax = load_aux<aux_comps<Model>()>(a, i, j);
    const Real mu = model.source_frequency(s, ax);
    if (mu > acc) acc = mu;
  }
};

/// InvStabilityDtKernel: max over cells of 1/model.stability_dt. We reduce the INVERSE (a
/// frequency) because the seam only provides a MAX reduction initialized to 0 (reduce_max_cell):
/// min(dt) == 1/max(1/dt) for dt > 0. A stability_dt <= 0 or non-finite is ignored (does not
/// constrain) -- the model signals "no bound here" by returning +inf.
template <class Model>
struct InvStabilityDtKernel {
  Model model;
  ConstArray4 u, a;
  ADC_HD void operator()(int i, int j, Real& acc) const {
    const auto s = load_state<Model>(u, i, j);
    const Aux ax = load_aux<aux_comps<Model>()>(a, i, j);
    const Real db = model.stability_dt(s, ax);
    if (db > Real(0)) {
      const Real inv = Real(1) / db;
      if (inv > acc) acc = inv;
    }
  }
};
}  // namespace detail

/// Global max of the STABILITY speed (HasStabilitySpeed trait) -- counterpart of max_wave_speed_mf.
template <class Model>
inline Real max_stability_speed_mf(const Model& model, const MultiFab& U, const MultiFab& aux) {
  Real m = 0;
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    m = std::max(m, reduce_max_cell(U.box(li), detail::StabilitySpeedKernel<Model>{model, u, a}));
  }
  return static_cast<Real>(all_reduce_max(static_cast<double>(m)));
}

/// Global max of the source frequency (HasSourceFrequency trait). 0 if the source does not constrain.
template <class Model>
inline Real max_source_frequency_mf(const Model& model, const MultiFab& U, const MultiFab& aux) {
  Real m = 0;
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    m = std::max(m, reduce_max_cell(U.box(li), detail::SourceFrequencyKernel<Model>{model, u, a}));
  }
  return static_cast<Real>(all_reduce_max(static_cast<double>(m)));
}

/// Global min of the declared admissible step (HasStabilityDt trait), via max(1/dt) (cf.
/// InvStabilityDtKernel). @return 0 if NO cell constrains (the block imposes no bound).
template <class Model>
inline Real min_stability_dt_mf(const Model& model, const MultiFab& U, const MultiFab& aux) {
  Real inv = 0;
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    inv = std::max(inv, reduce_max_cell(U.box(li), detail::InvStabilityDtKernel<Model>{model, u, a}));
  }
  inv = static_cast<Real>(all_reduce_max(static_cast<double>(inv)));
  return inv > Real(0) ? Real(1) / inv : Real(0);
}

/// rusanov_flux: free compat, delegates to RusanovFlux{} (policy of numerical_flux.hpp).
///
/// Kept for the serial references (GPU demos, unit tests) that call rusanov_flux directly.
/// Prefer RusanovFlux{} passed as a template for new calls. ADC_HD.
template <class Model>
ADC_HD inline typename Model::State rusanov_flux(const Model& m,
                                          const typename Model::State& UL,
                                          const Aux& AL,
                                          const typename Model::State& UR,
                                          const Aux& AR, int dir) {
  return RusanovFlux{}(m, UL, AL, UR, AR, dir);
}

/// reconstruct<Model,Limiter>: face value at (i,j) extrapolated in direction dir.
///
/// sgn = +1 -> +dir face of (i,j); sgn = -1 -> -dir face. Reconstructs in PRIMITIVE
/// variables if prim == true AND if Model exposes HasPrimitiveVars (positivity of rho and p
/// for Euler); otherwise in conservative variables. The returned state is ALWAYS conservative.
/// NoSlope (n_ghost == 1): zero slope, prim has no effect -- pure conservative path.
/// INVARIANT: POINTWISE function, does NOT loop over the grid. ADC_HD.
template <class Model, class Limiter>
ADC_HD inline typename Model::State reconstruct(const Model& model, const ConstArray4& u,
                                                int i, int j, int dir, Real sgn,
                                                const Limiter& lim, bool prim) {
  if constexpr (HasPrimitiveVars<Model> && Limiter::n_ghost >= 2) {
    if (prim) {  // convert the stencil U->P, limit on P, convert back P->U
      using Prim = typename Model::Prim;
      const Prim P0 = model.to_primitive(load_state<Model>(u, i, j));
      Prim Pf{};
      if constexpr (Limiter::n_ghost == 2) {
        const Prim Pm = model.to_primitive(
            load_state<Model>(u, dir == 0 ? i - 1 : i, dir == 0 ? j : j - 1));
        const Prim Pp = model.to_primitive(
            load_state<Model>(u, dir == 0 ? i + 1 : i, dir == 0 ? j : j + 1));
        for (int c = 0; c < Model::n_vars; ++c)
          Pf[c] = P0[c] + sgn * Real(0.5) * lim(P0[c] - Pm[c], Pp[c] - P0[c]);
      } else {  // WENO5 on the 5-point stencil in primitive variables
        const int d = (sgn > Real(0)) ? 1 : -1;
        const Prim Pm2 = model.to_primitive(
            load_state<Model>(u, dir == 0 ? i - 2 * d : i, dir == 0 ? j : j - 2 * d));
        const Prim Pm1 = model.to_primitive(
            load_state<Model>(u, dir == 0 ? i - d : i, dir == 0 ? j : j - d));
        const Prim Pp1 = model.to_primitive(
            load_state<Model>(u, dir == 0 ? i + d : i, dir == 0 ? j : j + d));
        const Prim Pp2 = model.to_primitive(
            load_state<Model>(u, dir == 0 ? i + 2 * d : i, dir == 0 ? j : j + 2 * d));
        for (int c = 0; c < Model::n_vars; ++c)
          Pf[c] = weno5z(Pm2[c], Pm1[c], P0[c], Pp1[c], Pp2[c]);
      }
      return model.to_conservative(Pf);
    }
  }
  (void)model;
  (void)prim;
  typename Model::State s = load_state<Model>(u, i, j);
  if constexpr (Limiter::n_ghost == 2) {
    // MUSCL: per-component limited slope (order 2).
    for (int c = 0; c < Model::n_vars; ++c) {
      const Real am = (dir == 0) ? u(i, j, c) - u(i - 1, j, c)
                                 : u(i, j, c) - u(i, j - 1, c);
      const Real ap = (dir == 0) ? u(i + 1, j, c) - u(i, j, c)
                                 : u(i, j + 1, c) - u(i, j, c);
      s[c] += sgn * Real(0.5) * lim(am, ap);
    }
  } else if constexpr (Limiter::n_ghost >= 3) {
    // WENO5 (order 5): face value from a 5-point stencil oriented by sgn
    // (sgn>0 -> +dir face; sgn<0 -> -dir face, reversed stencil). lim unused.
    (void)lim;
    const int d = (sgn > Real(0)) ? 1 : -1;
    for (int c = 0; c < Model::n_vars; ++c) {
      if (dir == 0)
        s[c] = weno5z(u(i - 2 * d, j, c), u(i - d, j, c), u(i, j, c),
                      u(i + d, j, c), u(i + 2 * d, j, c));
      else
        s[c] = weno5z(u(i, j - 2 * d, c), u(i, j - d, c), u(i, j, c),
                      u(i, j + d, c), u(i, j + 2 * d, c));
    }
  }
  return s;
}

/// zhang_shu_scale: POSITIVITY limiter on a reconstructed face state -- LOCAL ORDER-1 FALLBACK
/// (vacuum-robust variant of the Zhang & Shu scaling, JCP 2010).
///
/// If component @p pos_comp (Density role) of the face state @p s falls below @p floor, the WHOLE
/// face state is replaced by the average of its SOURCE cell u(i,j,.) (locally zero slope).
/// WHY not the paper's colinear theta-scaling (s <- ubar + theta (s - ubar), theta such that
/// rho_face = floor): in CONSERVATIVE variables at the edge of the QUASI-VACUUM (1e-6 background of
/// the Hoffart diocotron), it sets rho_face = floor while leaving a face momentum O(average) -> the
/// face VELOCITY v = m/rho diverges (~1e6) -> the Rusanov wave speed blows up whereas dt was chosen
/// BEFORE on the cell velocities -> immediate blow-up (measured: NaN at step 2 of the Hoffart case,
/// whatever the floor). The paper couples its limiter to the recomputed CFL bound; here the fallback
/// to the average bounds the face velocity by CONSTRUCTION (v_face = v_cell), stays conservative
/// (the average is not touched), positive as soon as the average is, and degrades the order only on
/// the offending faces (WENO5 intact everywhere else).
/// Inactive if floor <= 0 (bit-identical path) or if the face is already >= floor. Motivation: WENO5
/// undershoots at the top-hat jump with 1e6 contrast -> negative face rho -> 1/rho and the Lorentz
/// source detonate -> NaN (adc_cases ADC-62/ADC-74, ticket ADC-76). POINTWISE device-clean function. ADC_HD.
template <class Model>
ADC_HD inline void zhang_shu_scale(typename Model::State& s, const ConstArray4& u,
                                   int i, int j, Real floor, int pos_comp) {
  if (!(floor > Real(0))) return;            // strict opt-in: floor <= 0 -> no effect
  if (!(s[pos_comp] < floor)) return;        // face already above the floor
  for (int c = 0; c < Model::n_vars; ++c) s[c] = u(i, j, c);  // order-1 fallback: face = average
}

/// reconstruct_pp: reconstruct + zhang_shu_scale positivity limiter on the returned state.
///
/// (i, j) is the SOURCE cell of the reconstruction: it is to ITS average that the face state is
/// brought back. pos_floor <= 0 -> strictly identical to reconstruct (short-circuit). ADC_HD.
template <class Model, class Limiter>
ADC_HD inline typename Model::State reconstruct_pp(const Model& model, const ConstArray4& u,
                                                   int i, int j, int dir, Real sgn,
                                                   const Limiter& lim, bool prim,
                                                   Real pos_floor, int pos_comp) {
  typename Model::State s = reconstruct<Model>(model, u, i, j, dir, sgn, lim, prim);
  zhang_shu_scale<Model>(s, u, i, j, pos_floor, pos_comp);
  return s;
}

namespace detail {
/// Component of the Density role for the positivity limiter (HOST, resolved once per spatial
/// operator call, never per cell). pos_floor <= 0 -> 0 (never read, the scaling is short-circuited
/// in zhang_shu_scale). A model without VariableSet introspection or without a Density role cannot
/// request positivity: clear error rather than a silent scaling of an arbitrary component.
template <class Model>
inline int positivity_comp(Real pos_floor) {
  if (!(pos_floor > Real(0))) return 0;
  if constexpr (requires { Model::conservative_vars(); }) {
    const int c = Model::conservative_vars().index_of(VariableRole::Density);
    if (c >= 0) return c;
    throw std::runtime_error(
        "positivity_floor > 0: the model does not expose the Density role (scaling target)");
  } else {
    throw std::runtime_error(
        "positivity_floor > 0: model without VariableSet introspection (conservative_vars)");
  }
}

/// require_reconstruction_ghosts<Limiter>: STRUCTURAL ENTRY GUARD of the FV spatial operators.
/// A limiter's reconstruction stencil reads up to Limiter::n_ghost cells BEYOND the valid box: we
/// reconstruct the NEIGHBOR cells i+-1 of each valid cell, which reads i+-2 for a 2-ghost MUSCL
/// (Minmod / VanLeer) and i+-3 for WENO5. If the state does not carry this ghost width, the read
/// runs off the Fab buffer (heap-buffer-overflow, silent UB: negative linear index). We REQUIRE the
/// contract at entry -- CLEAR error rather than an out-of-bounds read -- exactly the rule already
/// applied to ALLOCATION (Limiter::n_ghost) on the AMR side and block_builder (cf. python/system.cpp
/// and PR #22). aux / mask are only read at i+-1 (1 ghost), strictly smaller width: it is the STATE
/// ghosts that size the stencil.
template <class Limiter>
inline void require_reconstruction_ghosts(const MultiFab& U) {
  if (U.n_grow() < Limiter::n_ghost)
    throw std::runtime_error(
        "spatial operator: the state must carry at least Limiter::n_ghost ghost layers "
        "(the reconstruction stencil reads i+-Limiter::n_ghost at the edge of the valid box); "
        "allocate the state MultiFab with this number of ghosts.");
}
}  // namespace detail

/// xface_box / yface_box: face boxes normal to x (resp. y) associated with a cell box.
///
/// xface_box(v): nx+1 x ny (i in [lo..hi+1], j in [lo..hi]).
/// yface_box(v): nx x ny+1 (i in [lo..hi], j in [lo..hi+1]).
/// Used to size the MultiFab Fx, Fy received by compute_face_fluxes.
inline Box2D xface_box(const Box2D& v) {
  return Box2D{{v.lo[0], v.lo[1]}, {v.hi[0] + 1, v.hi[1]}};
}
inline Box2D yface_box(const Box2D& v) {
  return Box2D{{v.lo[0], v.lo[1]}, {v.hi[0], v.hi[1] + 1}};
}

namespace detail {
/// FaceFluxXKernel: device kernel for the flux at the radial x face (between i-1 and i).
///
/// Reconstructs the L (cell i-1, +x face) and R (cell i, -x face) states, computes the
/// numerical flux, writes into fx(i,j). Adds the Fickian flux if DiffusiveModel.
/// Named functor (device-clean cross-TU). ADC_HD.
template <class Limiter, class NumericalFlux, class Model>
struct FaceFluxXKernel {
  Model model;
  ConstArray4 u, ax;
  Array4 fx;
  Real dx;
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  int pos_comp = 0;          ///< component of the Density role (resolved by the host caller)
  ADC_HD void operator()(int i, int j) const {
    const auto L = reconstruct_pp<Model>(model, u, i - 1, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rr = reconstruct_pp<Model>(model, u, i, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto F = nflux(model, L, load_aux<aux_comps<Model>()>(ax, i - 1, j), Rr,
                         load_aux<aux_comps<Model>()>(ax, i, j), 0);
    for (int c = 0; c < Model::n_vars; ++c) fx(i, j, c) = F[c];
    if constexpr (DiffusiveModel<Model>) {
      const Real nu = model.diffusivity();
      for (int c = 0; c < Model::n_vars; ++c)
        fx(i, j, c) += -nu * (u(i, j, c) - u(i - 1, j, c)) / dx;
    }
  }
};
/// FaceFluxYKernel: device kernel for the flux at the y face (between j-1 and j).
///
/// Analogue of FaceFluxXKernel in the j direction. Named functor. ADC_HD.
template <class Limiter, class NumericalFlux, class Model>
struct FaceFluxYKernel {
  Model model;
  ConstArray4 u, ax;
  Array4 fy;
  Real dy;
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  int pos_comp = 0;          ///< component of the Density role (resolved by the host caller)
  ADC_HD void operator()(int i, int j) const {
    const auto L = reconstruct_pp<Model>(model, u, i, j - 1, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rr = reconstruct_pp<Model>(model, u, i, j, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto F = nflux(model, L, load_aux<aux_comps<Model>()>(ax, i, j - 1), Rr,
                         load_aux<aux_comps<Model>()>(ax, i, j), 1);
    for (int c = 0; c < Model::n_vars; ++c) fy(i, j, c) = F[c];
    if constexpr (DiffusiveModel<Model>) {
      const Real nu = model.diffusivity();
      for (int c = 0; c < Model::n_vars; ++c)
        fy(i, j, c) += -nu * (u(i, j, c) - u(i, j - 1, c)) / dy;
    }
  }
};
}  // namespace detail

/// compute_face_fluxes<Limiter,NumericalFlux>: writes the face fluxes BEFORE divergence.
///
/// Fx(i,j) = flux at the face between (i-1,j) and (i,j), i in [lo..hi+1].
/// Fy(i,j) = flux between (i,j-1) and (i,j), j in [lo..hi+1].
/// Brick required by the AMR reflux: assemble_rhs computes -div F directly and discards the face
/// fluxes, but the reflux must see them to correct the coarse-fine interfaces.
/// For a DiffusiveModel, the Fickian flux F_diff = -nu (u_R-u_L)/h is added (its divergence
/// reproduces EXACTLY +nu Lap(u) of assemble_rhs, and stays visible to the reflux).
/// dx=0, dy=0 by default: not read for a non-diffusive model (hyperbolic bit-identical).
//
// compute_face_fluxes: writes the numerical fluxes at the FACES (Fx at faces normal to x,
// Fy at y), BEFORE divergence. This is the brick the AMR reflux needs (it accumulates the
// fine fluxes and subtracts the coarse flux at the coarse-fine interfaces; assemble_rhs
// itself computes -div F directly and discards the face fluxes).
//
// Conventions: Fx(i,j) = flux at the face between cells (i-1,j) and (i,j), i in [lo..hi+1].
// Fy(i,j) = flux between (i,j-1) and (i,j), j in [lo..hi+1]. Same reconstruction (Limiter)
// and numerical flux (NumericalFlux) as assemble_rhs, so
//   r(i,j) = S - (Fx(i+1,j)-Fx(i,j))/dx - (Fy(i,j+1)-Fy(i,j))/dy
// gives back EXACTLY the assemble_rhs residual. Fx, Fy sized by the caller (xface_box/yface_box
// boxes, ncomp = Model::n_vars, 0 ghost). Device-callable.
//
// DIFFUSION on AMR (milestone 4): for a DiffusiveModel, we add the FACE Fickian flux
// F_diff = -nu (u_R - u_L)/h (centered gradient at the face, cell values). Its divergence
// -(Fx(i+1)-Fx(i))/dx gives back EXACTLY +nu Lap(u) of assemble_rhs, but treated as a FLUX:
// the AMR reflux therefore sees it, and the diffusion stays conservative at the coarse-fine
// interfaces (otherwise a direct Laplacian would be ignored by the reflux). dx/dy = step of
// the LEVEL (passed by the caller; 0 by default, not read for a non-diffusive model -> the
// hyperbolic path is strictly bit-identical).
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void compute_face_fluxes(const Model& model, const MultiFab& U, const MultiFab& aux,
                         MultiFab& Fx, MultiFab& Fy, Real dx = 0, Real dy = 0,
                         bool recon_prim = false, Real pos_floor = Real(0)) {
  detail::require_reconstruction_ghosts<Limiter>(U);  // state ghosts >= stencil (otherwise OOB)
  const Limiter lim{};
  const NumericalFlux nflux{};
  const int pos_comp = detail::positivity_comp<Model>(pos_floor);
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    Array4 fx = Fx.fab(li).array();
    Array4 fy = Fy.fab(li).array();
    const Box2D v = U.box(li);
    for_each_cell(xface_box(v), detail::FaceFluxXKernel<Limiter, NumericalFlux, Model>{
                                    model, u, ax, fx, dx, lim, nflux, recon_prim, pos_floor, pos_comp});
    for_each_cell(yface_box(v), detail::FaceFluxYKernel<Limiter, NumericalFlux, Model>{
                                    model, u, ax, fy, dy, lim, nflux, recon_prim, pos_floor, pos_comp});
  }
}

namespace detail {
/// AssembleRhsKernel<Limiter,NumericalFlux,Model>: device kernel of the central residual of
/// assemble_rhs.
///
/// Computes R(i,j) = S - (Fxp-Fxm)/dx - (Fyp-Fym)/dy (+ Fickian term if DiffusiveModel).
/// Named functor: key point of the AOT native parity (add_compiled_model via external TU).
/// Body bit-identical to the former lambda. ADC_HD.
//
// nvcc does not reliably emit the device kernel of a Model-template extended lambda first
// instantiated from an EXTERNAL TU through the std::function / host-lambda nesting of block_builder:
// the test passes on Serial and under compute-sanitizer but segfaults at runtime on Cuda (Heisenbug).
// A device-callable class does not have these instantiation-context restrictions. Body IDENTICAL to
// the former lambda -> residual BIT-IDENTICAL to add_block on CPU (and, targeted, on device).
template <class Limiter, class NumericalFlux, class Model>
struct AssembleRhsKernel {
  Model model;
  ConstArray4 u, ax;
  Array4 r;
  Real dx, dy;
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  int pos_comp = 0;          ///< component of the Density role (resolved by the host caller)
  ADC_HD void operator()(int i, int j) const {
    const Aux Ac = load_aux<aux_comps<Model>()>(ax, i, j);
    const Aux Axm = load_aux<aux_comps<Model>()>(ax, i - 1, j);
    const Aux Axp = load_aux<aux_comps<Model>()>(ax, i + 1, j);
    const Aux Aym = load_aux<aux_comps<Model>()>(ax, i, j - 1);
    const Aux Ayp = load_aux<aux_comps<Model>()>(ax, i, j + 1);

    // x faces: reconstruction of the states on either side of each face
    const auto Lxm = reconstruct_pp<Model>(model, u, i - 1, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rxm = reconstruct_pp<Model>(model, u, i, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto Lxp = reconstruct_pp<Model>(model, u, i, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rxp = reconstruct_pp<Model>(model, u, i + 1, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto Fxm = nflux(model, Lxm, Axm, Rxm, Ac, 0);
    const auto Fxp = nflux(model, Lxp, Ac, Rxp, Axp, 0);

    // y faces
    const auto Lym = reconstruct_pp<Model>(model, u, i, j - 1, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rym = reconstruct_pp<Model>(model, u, i, j, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto Lyp = reconstruct_pp<Model>(model, u, i, j, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Ryp = reconstruct_pp<Model>(model, u, i, j + 1, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto Fym = nflux(model, Lym, Aym, Rym, Ac, 1);
    const auto Fyp = nflux(model, Lyp, Ac, Ryp, Ayp, 1);

    const auto S = model.source(load_state<Model>(u, i, j), Ac);
    for (int c = 0; c < Model::n_vars; ++c)
      r(i, j, c) = S[c] - (Fxp[c] - Fxm[c]) / dx - (Fyp[c] - Fym[c]) / dy;

    // Parabolic (Fickian) term: +nu Lap(U), 5-point centered differences.
    // Guarded by DiffusiveModel: no effect (nor codegen) for a non-diffusive model.
    if constexpr (DiffusiveModel<Model>) {
      const Real nu = model.diffusivity();
      const Real idx2 = Real(1) / (dx * dx), idy2 = Real(1) / (dy * dy);
      for (int c = 0; c < Model::n_vars; ++c)
        r(i, j, c) += nu * ((u(i + 1, j, c) - 2 * u(i, j, c) + u(i - 1, j, c)) * idx2 +
                            (u(i, j + 1, c) - 2 * u(i, j, c) + u(i, j - 1, c)) * idy2);
    }
  }
};
}  // namespace detail

/// assemble_rhs<Limiter,NumericalFlux>: residual R = -div Fhat + S over all boxes.
///
/// Main entry point of the Cartesian spatial operator. The limiter (reconstruction) AND the
/// numerical flux are template parameters chosen at compile time (default: NoSlope + RusanovFlux).
/// recon_prim = true enables reconstruction in primitive variables if the model exposes
/// HasPrimitiveVars. For the diffusive term, see DiffusiveModel.
/// INVARIANT: the operator does not modify U, aux -- it only writes R. No ghost fill.
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void assemble_rhs(const Model& model, const MultiFab& U, const MultiFab& aux,
                  const Geometry& geom, MultiFab& R, bool recon_prim = false,
                  Real pos_floor = Real(0)) {
  detail::require_reconstruction_ghosts<Limiter>(U);  // state ghosts >= stencil (otherwise OOB)
  const Real dx = geom.dx(), dy = geom.dy();
  const Limiter lim{};
  const NumericalFlux nflux{};
  const int pos_comp = detail::positivity_comp<Model>(pos_floor);
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    Array4 r = R.fab(li).array();
    const Box2D v = R.box(li);
    for_each_cell(v, detail::AssembleRhsKernel<Limiter, NumericalFlux, Model>{
                         model, u, ax, r, dx, dy, lim, nflux, recon_prim, pos_floor, pos_comp});
  }
}

// ============================================================================
// DOMAIN MASK (T2 effort, conservative, OPT-IN -- default path untouched)
// ============================================================================
// The mask makes the FV transport aware of an ACTIVE sub-domain (e.g. the paper's disk).
// Convention: mask(i, j) >= 0.5 -> ACTIVE cell, otherwise INACTIVE. A face is OPEN (normal flux
// computed) if BOTH adjacent cells are active; it is CLOSED (normal flux set to ZERO) if at least
// one is inactive. Zeroing the normal flux at active/inactive faces makes the step CONSERVATIVE
// over the active sub-domain: no mass crosses the boundary, so the total mass over the active cells
// is conserved to machine precision (telescoping internal fluxes, zero boundary fluxes). This is the
// FV counterpart of the conducting wall (which only acts on the elliptic part).
//
// The residual is written ONLY on the active cells; an inactive cell keeps its residual at 0
// (the caller does not advance it). This header does NOT wire this path into System::step: it
// provides the mask-aware brick, exercised directly by the tests and, eventually, behind the disk
// opt-in.

namespace detail {
/// Activity indicator of a cell from a 0/1 cell-centered mask (>= 0.5 -> active).
ADC_HD inline bool mask_active(const ConstArray4& mask, int i, int j) {
  return mask(i, j, 0) >= Real(0.5);
}

/// AssembleRhsMaskedKernel: variant of AssembleRhsKernel AWARE of a domain mask.
///
/// Inactive cell -> residual 0 (not advanced by the caller). Active cell -> R = -div Fhat + S,
/// BUT the normal flux of a face whose neighbor cell is INACTIVE is set to ZERO (FV wall:
/// zero normal flux at the active/inactive boundary) -> mass conservation over the active
/// sub-domain. Named functor (same device contract as AssembleRhsKernel). ADC_HD.
///
/// NB: without a diffusive term (transport-only models targeted by the disk); a DiffusiveModel keeps
/// its UNmasked Laplacian here (separate refinement -- the conservative mask targets the hyperbolic
/// flux, cf. the "ring edges" effort).
template <class Limiter, class NumericalFlux, class Model>
struct AssembleRhsMaskedKernel {
  Model model;
  ConstArray4 u, ax, mask;
  Array4 r;
  Real dx, dy;
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  int pos_comp = 0;          ///< component of the Density role (resolved by the host caller)
  ADC_HD void operator()(int i, int j) const {
    if (!mask_active(mask, i, j)) {  // cell outside the active sub-domain: zero residual, not advanced
      for (int c = 0; c < Model::n_vars; ++c) r(i, j, c) = Real(0);
      return;
    }
    const Aux Ac = load_aux<aux_comps<Model>()>(ax, i, j);
    const Aux Axm = load_aux<aux_comps<Model>()>(ax, i - 1, j);
    const Aux Axp = load_aux<aux_comps<Model>()>(ax, i + 1, j);
    const Aux Aym = load_aux<aux_comps<Model>()>(ax, i, j - 1);
    const Aux Ayp = load_aux<aux_comps<Model>()>(ax, i, j + 1);

    // x faces: reconstruction on either side, numerical flux, THEN mask gate (closed face
    // -> zero normal flux) -- an inactive neighbor cell closes the face between it and (i, j).
    const auto Lxm = reconstruct_pp<Model>(model, u, i - 1, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rxm = reconstruct_pp<Model>(model, u, i, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto Lxp = reconstruct_pp<Model>(model, u, i, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rxp = reconstruct_pp<Model>(model, u, i + 1, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    auto Fxm = nflux(model, Lxm, Axm, Rxm, Ac, 0);
    auto Fxp = nflux(model, Lxp, Ac, Rxp, Axp, 0);
    if (!mask_active(mask, i - 1, j)) Fxm = typename Model::State{};
    if (!mask_active(mask, i + 1, j)) Fxp = typename Model::State{};

    // y faces
    const auto Lym = reconstruct_pp<Model>(model, u, i, j - 1, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rym = reconstruct_pp<Model>(model, u, i, j, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto Lyp = reconstruct_pp<Model>(model, u, i, j, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Ryp = reconstruct_pp<Model>(model, u, i, j + 1, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    auto Fym = nflux(model, Lym, Aym, Rym, Ac, 1);
    auto Fyp = nflux(model, Lyp, Ac, Ryp, Ayp, 1);
    if (!mask_active(mask, i, j - 1)) Fym = typename Model::State{};
    if (!mask_active(mask, i, j + 1)) Fyp = typename Model::State{};

    const auto S = model.source(load_state<Model>(u, i, j), Ac);
    for (int c = 0; c < Model::n_vars; ++c)
      r(i, j, c) = S[c] - (Fxp[c] - Fxm[c]) / dx - (Fyp[c] - Fym[c]) / dy;
  }
};
}  // namespace detail

/// assemble_rhs_masked<Limiter,NumericalFlux>: residual R = -div Fhat + S RESTRICTED to a 0/1
/// cell-centered domain mask (OPT-IN, T2 effort). On an inactive cell R = 0 (not advanced); on an
/// active cell, the normal flux of a face whose neighbor is inactive is set to zero (FV wall).
/// Result: the mass over the active sub-domain is CONSERVED to machine precision (no flux crosses
/// the boundary) -- property validated by the conservation test of the disk effort.
///
/// @p mask must have the SAME layout as @p U (same BoxArray / DistributionMapping) and carry at
/// least 1 ghost (reading the neighbors i-1/i+1/j-1/j+1 up to the edge). This entry point is
/// SEPARATE from assemble_rhs: the default path (System::step) stays strictly bit-identical as long
/// as it does NOT call this overload.
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void assemble_rhs_masked(const Model& model, const MultiFab& U, const MultiFab& aux,
                         const MultiFab& mask, const Geometry& geom, MultiFab& R,
                         bool recon_prim = false, Real pos_floor = Real(0)) {
  detail::require_reconstruction_ghosts<Limiter>(U);  // state ghosts >= stencil (otherwise OOB)
  const Real dx = geom.dx(), dy = geom.dy();
  const Limiter lim{};
  const NumericalFlux nflux{};
  const int pos_comp = detail::positivity_comp<Model>(pos_floor);
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    const ConstArray4 mk = mask.fab(li).const_array();
    Array4 r = R.fab(li).array();
    const Box2D v = R.box(li);
    for_each_cell(v, detail::AssembleRhsMaskedKernel<Limiter, NumericalFlux, Model>{
                         model, u, ax, mk, r, dx, dy, lim, nflux, recon_prim, pos_floor, pos_comp});
  }
}

}  // namespace adc
