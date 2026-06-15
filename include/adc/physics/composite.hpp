#pragma once

#include <adc/core/physical_model.hpp>  // HyperbolicPhysicalModel: contract of the hyperbolic brick
#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/core/variables.hpp>

/// @file
/// @brief CompositeModel: assembles (hyperbolic, source, elliptic) into one compiled PhysicalModel.
///
/// The HYPERBOLIC brick carries Vars + flux + wave speeds + conversions (inseparable); source
/// and elliptic are SEPARATE bricks (physics/source.hpp, physics/elliptic.hpp), freely
/// composable. A named scenario is a COMPOSITION chosen from the application (adc_cases).

namespace adc {

/// Composite physical model: one HYPERBOLIC brick + one source + one elliptic right-hand side.
/// Satisfies the PhysicalModel concept; the Vars (conversions + descriptor), the flux and the
/// wave speeds come from the hyperbolic; pressure / wave_speeds are exposed when the
/// hyperbolic provides them (required by the HLLC / Roe flux).
///
/// CONTRACT: CompositeModel is a pure function of its 3 bricks, device-callable (ADC_HD).
/// No MultiFab, no allocation, no global access. Every delegated method inherits the
/// device-clean invariant of the underlying brick.
///
/// n_aux propagation: n_aux = max(aux_comps<Hyperbolic>, aux_comps<Source>, aux_comps<Elliptic>).
/// The system sizes the aux channel accordingly. A brick without n_aux (default kAuxBaseComps=3)
/// does not change the bit-identical history.
template <class Hyperbolic, class Source, class Elliptic>
struct CompositeModel {
  static_assert(HyperbolicPhysicalModel<Hyperbolic>,
                "CompositeModel : the 1st brick must be a HYPERBOLIC model (Vars + "
                "cons<->prim conversions + flux + max_wave_speed), see HyperbolicPhysicalModel");
  using State = StateVec<Hyperbolic::n_vars>;
  using Prim = typename Hyperbolic::Prim;
  using Aux = adc::Aux;
  static constexpr int n_vars = Hyperbolic::n_vars;
  // Aux channel width of the composite model = MAX of the widths of its bricks: if a brick (flux
  // or source) reads an extra auxiliary field (e.g. a magnetized source declaring n_aux=4 to read
  // B_z), the composite exposes it to the system (which then sizes the aux channel).
  // Without any extra-field brick, n_aux = kAuxBaseComps (3) -> strictly identical to the history.
  static constexpr int n_aux = [] {
    int w = aux_comps<Hyperbolic>();
    if (aux_comps<Source>() > w) w = aux_comps<Source>();
    if (aux_comps<Elliptic>() > w) w = aux_comps<Elliptic>();
    return w;
  }();

  Hyperbolic hyp{};
  Source src{};
  Elliptic ell{};

  ADC_HD State flux(const State& u, const Aux& a, int dir) const { return hyp.flux(u, a, dir); }
  ADC_HD Real max_wave_speed(const State& u, const Aux& a, int dir) const {
    return hyp.max_wave_speed(u, a, dir);
  }
  ADC_HD State source(const State& u, const Aux& a) const { return src.apply(u, a); }
  ADC_HD Real elliptic_rhs(const State& u) const { return ell.rhs(u); }
  ADC_HD Prim to_primitive(const State& u) const { return hyp.to_primitive(u); }
  ADC_HD State to_conservative(const Prim& p) const { return hyp.to_conservative(p); }
  static VariableSet conservative_vars() { return Hyperbolic::conservative_vars(); }
  static VariableSet primitive_vars() { return Hyperbolic::primitive_vars(); }

  ADC_HD Real pressure(const State& u) const
    requires requires(const Hyperbolic h, const State s) { h.pressure(s); }
  {
    return hyp.pressure(u);
  }
  ADC_HD void wave_speeds(const State& u, const Aux& a, int dir, Real& smin, Real& smax) const
    requires requires(const Hyperbolic h, const State s, const Aux aa, int d, Real& lo,
                      Real& hi) { h.wave_speeds(s, aa, d, lo, hi); }
  {
    hyp.wave_speeds(u, a, dir, smin, smax);
  }

  /// Riemann CAPABILITIES (audit wave 3): HLLC hooks (contact_speed + hllc_star_state) and Roe
  /// (roe_dissipation) forwarded from the HYPERBOLIC brick when it declares them (the DSL emits
  /// them via enable_hllc; a C++ model can write them by hand). Concept-gated like
  /// pressure / wave_speeds: without hooks, the composite does not expose them (canonical paths /
  /// explicit rejections unchanged).
  ADC_HD Real contact_speed(const State& ul, const State& ur, Real pl, Real pr, Real sl, Real sr,
                            int dir) const
    requires requires(const Hyperbolic h, const State a_, const State b_, Real p, Real q, Real x,
                      Real y, int d) { h.contact_speed(a_, b_, p, q, x, y, d); }
  {
    return hyp.contact_speed(ul, ur, pl, pr, sl, sr, dir);
  }
  ADC_HD State hllc_star_state(const State& u, Real p, Real s, Real sStar, int dir) const
    requires requires(const Hyperbolic h, const State a_, Real p_, Real s_, Real ss_, int d) {
      h.hllc_star_state(a_, p_, s_, ss_, d);
    }
  {
    return hyp.hllc_star_state(u, p, s, sStar, dir);
  }
  ADC_HD State roe_dissipation(const State& ul, const Aux& al, const State& ur, const Aux& ar,
                               int dir) const
    requires requires(const Hyperbolic h, const State a_, const Aux x_, const State b_,
                      const Aux y_, int d) { h.roe_dissipation(a_, x_, b_, y_, d); }
  {
    return hyp.roe_dissipation(ul, al, ur, ar, dir);
  }

  /// GEOMETRIC source term of polar curvature, delegated to the hyperbolic brick when it
  /// exposes it (polar fluid: IsothermalFluxPolar). Concept-gated like pressure / wave_speeds:
  /// if the hyperbolic does not provide it (polar ExB scalar transport), CompositeModel does not
  /// expose it -> assemble_rhs_polar falls back to 0 (bit-identical). Does NOT touch the cartesian
  /// (assemble_rhs never calls it).
  ADC_HD State polar_geom_source(const State& u, Real r) const
    requires requires(const Hyperbolic h, const State s, Real rr) { h.polar_geom_source(s, rr); }
  {
    return hyp.polar_geom_source(u, r);
  }

  /// Optional STEP BOUNDS (audit 2026-06, see core/physical_model.hpp): forwarded
  /// conditionally like pressure / wave_speeds, otherwise the composite does not expose them and the
  /// step policy stays the history. stability_speed / stability_dt come from the HYPERBOLIC
  /// brick (it is the one the DSL emits); source_frequency comes from the SOURCE brick (it is
  /// the source that knows its relaxation/collision frequency).
  ADC_HD Real stability_speed(const State& u, const Aux& a, int dir) const
    requires requires(const Hyperbolic h, const State s, const Aux aa, int d) {
      h.stability_speed(s, aa, d);
    }
  {
    return hyp.stability_speed(u, a, dir);
  }
  ADC_HD Real stability_dt(const State& u, const Aux& a) const
    requires requires(const Hyperbolic h, const State s, const Aux aa) { h.stability_dt(s, aa); }
  {
    return hyp.stability_dt(u, a);
  }
  ADC_HD Real source_frequency(const State& u, const Aux& a) const
    requires requires(const Source sc, const State s, const Aux aa) { sc.frequency(s, aa); }
  {
    return src.frequency(u, a);
  }

  /// ANALYTIC JACOBIAN of the source (audit wave 3): forwarded from the SOURCE brick when
  /// it declares jacobian(U, aux, J) (J[r][c] = dS_r/dU_c). The Newton of the implicit source
  /// uses it instead of finite differences (HasSourceJacobian trait); without the method,
  /// nothing is exposed and the Newton keeps the historical finite differences.
  ADC_HD void source_jacobian(const State& u, const Aux& a, Real (&J)[n_vars][n_vars]) const
    requires requires(const Source sc, const State s, const Aux aa,
                      Real (&JJ)[n_vars][n_vars]) { sc.jacobian(s, aa, JJ); }
  {
    src.jacobian(u, a, J);
  }
};

}  // namespace adc
