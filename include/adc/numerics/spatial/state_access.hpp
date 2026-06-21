/// @file
/// @brief Model/state/aux access layer of the Cartesian spatial operator.
///
/// CONTRACT: how the spatial operator reads a model and its data.
///   - DiffusiveModel: optional concept (model.diffusivity() -> nu); the Fickian flux
///     F = -nu grad U is added to the hyperbolic flux when present (face_flux.hpp,
///     cartesian_operator.hpp).
///   - SourceFreeModel<M>: adapter that zeroes the source (explicit IMEX half-step).
///   - load_state<Model>: reads the conservative state from an Array4 (ADC_HD).
///   - load_aux<NComp>: reads the auxiliary (phi, grad, extra fields) (ADC_HD).
///
/// This module carries no grid loop: every entry is POINTWISE (ADC_HD) or a compile-time
/// model adapter. It is the bottom of the spatial/ dependency DAG (depends only on core/).

#pragma once

#include <adc/core/physical_model.hpp>  // aux_comps, HasPrimitiveVars: optional primitive reconstruction
#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/core/variables.hpp>  // VariableSet: SourceFreeModel::conservative_vars forwarding
#include <adc/mesh/fab2d.hpp>      // ConstArray4: load_state / load_aux read path

#include <concepts>

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
  // Roe / HLLC CAPABILITIES (HasRoeDissipation / HasHLLCStructure): forwarded ONLY if M exposes
  // them (requires clause), exactly like pressure / wave_speeds above and like composite.hpp.
  // WITHOUT these, an IMEX explicit half-step on riemann='roe' / 'hllc' loses the model's GENERIC
  // hooks, so RoeFlux / HLLCFlux silently fall back to the canonical Euler-4var path -- which fails
  // to even COMPILE for a non-Euler model (e.g. a moment hierarchy: n_vars != 4, no pressure). The
  // 4-var Euler models compiled before only because that fallback happens to fit them.
  ADC_HD Real contact_speed(const State& ul, const State& ur, Real pl, Real pr, Real sl, Real sr,
                            int dir) const
    requires requires(const M& mm, const State a_, const State b_, Real p, Real q, Real x, Real y,
                      int d) { mm.contact_speed(a_, b_, p, q, x, y, d); }
  {
    return m.contact_speed(ul, ur, pl, pr, sl, sr, dir);
  }
  ADC_HD State hllc_star_state(const State& u, Real p, Real s, Real sStar, int dir) const
    requires requires(const M& mm, const State a_, Real p_, Real s_, Real ss_, int d) {
      mm.hllc_star_state(a_, p_, s_, ss_, d);
    }
  {
    return m.hllc_star_state(u, p, s, sStar, dir);
  }
  ADC_HD State roe_dissipation(const State& ul, const Aux& al, const State& ur, const Aux& ar,
                               int dir) const
    requires requires(const M& mm, const State a_, const Aux x_, const State b_, const Aux y_,
                      int d) { mm.roe_dissipation(a_, x_, b_, y_, d); }
  {
    return m.roe_dissipation(ul, al, ur, ar, dir);
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
ADC_HD inline typename Model::State load_state(const ConstArray4& a, int i, int j) {
  typename Model::State u;
  for (int c = 0; c < Model::n_vars; ++c)
    u[c] = a(i, j, c);
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
  if constexpr (NComp > (idx))  \
    x.name = a(i, j, idx);
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
    for (int k = 0; k < n_extra; ++k)
      x.extra[k] = a(i, j, kAuxNamedBase + k);
  }
  return x;
}

}  // namespace adc
