#pragma once

#include <adc/core/foundation/types.hpp>
#include <adc/coupling/base/aux_fill.hpp>  // detail::derive_aux_bc + detail::fill_bz_box (shared)
#include <adc/coupling/base/coupling_policy.hpp>
#include <adc/coupling/base/elliptic_rhs.hpp>
#include <adc/numerics/elliptic/interface/elliptic_problem.hpp>
#include <adc/numerics/elliptic/interface/elliptic_solver.hpp>
#include <adc/numerics/elliptic/mg/geometric_mg.hpp>
#include <adc/numerics/time/time_integrator.hpp>
#include <adc/numerics/time/time_steppers.hpp>  // SSPRK2Step / SSPRK3Step (shared scheme)
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/reconstruction.hpp>
#include <adc/numerics/spatial_discretisation.hpp>
#include <adc/numerics/spatial_operator.hpp>
#include <adc/parallel/comm.hpp>

#include <functional>
#include <type_traits>
#include <utility>

/// @file
/// @brief Coupler: single-block hyperbolic-elliptic coupler (Poisson -> aux -> advance loop).
///
/// At each integrator stage (stage-by-stage coupling): (1) RHS f = elliptic_rhs(model, U);
/// (2) solve lap(phi) = f with the elliptic backend (warm start); (3) aux = (phi, grad phi) by
/// centered differences; (4) assemble the hyperbolic residual with this aux. For drift transport
/// aux enters through the FLUX (E x B); for a self-gravitating fluid through the SOURCE. Three
/// orthogonal axes, all template parameters: Limiter (reconstruction), Policy (PerStage vs
/// OncePerStep), NumericalFlux (Rusanov by default). Compatible with a SINGLE model; multi-species
/// goes through SystemCoupler. The detail:: helpers are at namespace scope (an ADC_HD extended
/// lambda cannot live in a private method, an nvcc restriction).

namespace adc {

namespace detail {
// Namespace-scope helpers: an extended __host__ __device__ lambda CANNOT be
// defined in a private/protected method (nvcc restriction), hence the
// extraction out of the Coupler class.

// Single-model compatibility: f = model.elliptic_rhs(U) on valid cells,
// delegated to a named assembler so this responsibility is not buried in Coupler.
/// Assemble the single-model elliptic RHS: rhs = model.elliptic_rhs(U) on valid cells
/// (delegated to SingleModelEllipticRhs). Shared by Coupler and AmrCouplerMP.
template <class Model>
inline void coupler_eval_rhs(const MultiFab& state, MultiFab& rhs, const Model& model) {
  SingleModelEllipticRhs<Model>{model}(state, rhs);
}

// aux = (phi, d phi/dx, d phi/dy) by centered differences. Delegates to the
// named FieldPostProcess convention with GradSign::Plus and store_phi=true: the
// coupler stores +grad phi (the physical sign E = -grad phi is carried by the
// transport drift velocity). Multiplicative form *cx / *cy kept identical
// -> bit-identical.
/// Set aux = (phi, d phi/dx, d phi/dy) by centered differences (factors cx, cy = 1/(2 dx),
/// 1/(2 dy)). Stores +grad phi (the physical sign E = -grad phi is carried by the drift velocity).
inline void coupler_grad_phi(const MultiFab& phi, MultiFab& aux, Real cx, Real cy) {
  field_postprocess(phi, aux, cx, cy, FieldPostProcess{FieldPostProcess::GradSign::Plus, true});
}
}  // namespace detail

/// Single-block hyperbolic-elliptic coupler. @tparam Model: PhysicalModel (flux, source,
/// elliptic_rhs, max_wave_speed, aux channel). @tparam Elliptic: elliptic backend (concept
/// EllipticSolver, default GeometricMG). Owns the aux and the solver; advance/step close the
/// Poisson -> aux -> residual loop at each step. PRECONDITION: U carries at least Limiter::n_ghost ghosts.
template <class Model, class Elliptic = GeometricMG>
class Coupler {
  static_assert(EllipticSolver<Elliptic>, "the Coupler elliptic backend must model EllipticSolver");

 public:
  // active: optional "inside the conductor" predicate (embedded wall for
  // the Poisson solver). Empty => no internal wall.
  // bz: out-of-plane magnetic field B_z(x, y) PROVIDED by the user (constant or
  // field). Only has effect if the model declares the B_z aux component (aux_comps>3);
  // then fills aux component 3 once and for all (B_z static, external to the
  // elliptic solve: derive_aux does not touch it). Empty => no B_z. The aux channel is
  // allocated to the MODEL width: a base model (3) stays bit-identical.
  Coupler(const Model& model, const Geometry& geom, const BoxArray& ba, const BCRec& bcU,
          const BCRec& bcPhi, std::function<bool(Real, Real)> active = {},
          std::function<Real(Real, Real)> bz = {})
      : model_(model),
        geom_(geom),
        ba_(ba),
        dm_(ba.size(), n_ranks()),
        bcU_(bcU),
        bcPhi_(bcPhi),
        aux_bc_(detail::derive_aux_bc(bcPhi)),
        mg_(geom, ba, bcPhi, std::move(active)),
        aux_(ba, dm_, aux_comps<Model>(), 1),
        bz_(std::move(bz)) {
    fill_bz();  // fills the B_z component (no-op if base model or empty bz)
  }

  // Coupled SSPRK2. Three orthogonal axes, all template parameters:
  //   - Limiter: reconstruction (NoSlope / Minmod / VanLeer ...)
  //   - Policy: time coupling (PerStage = phi at each stage; OncePerStep
  //                    = a single solve per step, aux frozen)
  //   - NumericalFlux: Riemann flux (Rusanov by default, HLL, HLLC ...)
  // U must carry at least Limiter::n_ghost ghosts. The historical signature
  // advance<Limiter, Policy> stays valid (NumericalFlux default = Rusanov).
  template <class Limiter = NoSlope, class Policy = PerStageCoupling,
            class NumericalFlux = RusanovFlux>
  void advance(MultiFab& U, Real dt) {
    static_assert(
        std::is_same_v<Policy, PerStageCoupling> || std::is_same_v<Policy, OncePerStepCoupling>,
        "Policy must be PerStageCoupling or OncePerStepCoupling");
    constexpr bool per = std::is_same_v<Policy, PerStageCoupling>;
    // DELEGATES the scheme to the core SSPRK2Step object (dedup, sec.8.2 A4). The residual
    // evaluator counts the stages: recompute_aux=true at stage 0, =per afterward (PerStage:
    // phi recomputed for the intermediate state; OncePerStep: aux frozen). Bit-identical.
    int stage = 0;
    SSPRK2Step{}.take_step(
        [&](MultiFab& s, MultiFab& R) {
          stage_rhs<Limiter, NumericalFlux>(s, R, (stage++ == 0) ? true : per);
        },
        U, dt);
  }

  // Coupled SSPRK3 (Shu-Osher, 3 stages). Same axes as advance.
  template <class Limiter = NoSlope, class Policy = PerStageCoupling,
            class NumericalFlux = RusanovFlux>
  void advance_ssprk3(MultiFab& U, Real dt) {
    static_assert(
        std::is_same_v<Policy, PerStageCoupling> || std::is_same_v<Policy, OncePerStepCoupling>,
        "Policy must be PerStageCoupling or OncePerStepCoupling");
    constexpr bool per = std::is_same_v<Policy, PerStageCoupling>;
    // Same as advance: delegates to SSPRK3Step, recompute_aux=true at stage 0, =per afterward.
    int stage = 0;
    SSPRK3Step{}.take_step(
        [&](MultiFab& s, MultiFab& R) {
          stage_rhs<Limiter, NumericalFlux>(s, R, (stage++ == 0) ? true : per);
        },
        U, dt);
  }

  // Unified entry point: give the coupler a SPATIAL DISCRETISATION
  // (limiter + flux) and an explicit time policy. The old SSPRK2/SSPRK3 tags
  // stay valid; the new form also enables sub-cycling:
  //   sim.step<MusclVanLeerHLLC, ExplicitTime<SSPRK3, 4>>(U, dt);
  template <class Disc = FirstOrder, class TimeInteg = SSPRK2, class Policy = PerStageCoupling>
  void step(MultiFab& U, Real dt) {
    using L = typename Disc::Limiter;
    using F = typename Disc::NumericalFlux;
    using T = typename TimePolicyTraits<TimeInteg>::Method;
    static_assert(TimePolicyTraits<TimeInteg>::treatment == TimeTreatment::Explicit,
                  "Coupler::step can only run explicit policies; "
                  "use a scheduler/system for IMEX or implicit");
    static_assert(std::is_same_v<T, SSPRK2> || std::is_same_v<T, SSPRK3>,
                  "Coupler::step supports SSPRK2 or SSPRK3");
    constexpr int n = TimePolicyTraits<TimeInteg>::substeps;
    const Real h = dt / static_cast<Real>(n);
    for (int s = 0; s < n; ++s) {
      if constexpr (std::is_same_v<T, SSPRK3>)
        advance_ssprk3<L, Policy, F>(U, h);
      else
        advance<L, Policy, F>(U, h);
    }
  }

  /// Solve phi and derive aux = (phi, grad phi) for @p U WITHOUT advancing in time (useful to estimate
  /// the E x B velocity before fixing dt). aux() is up to date on return.
  void solve_fields(const MultiFab& U) { update_aux(U); }

  MultiFab& phi() { return mg_.phi(); }
  const MultiFab& aux() const { return aux_; }

 private:
  void update_aux(const MultiFab& state) {
    detail::coupler_eval_rhs(state, mg_.rhs(), model_);
    mg_.solve();  // EllipticSolver concept interface (backend-agnostic)
    derive_aux();
  }

  // One stage: (optional) elliptic solve, halos, hyperbolic residual into R.
  // Shared by advance (SSPRK2) and advance_ssprk3; order of operations kept
  // to stay bit-identical to the old advance.
  template <class Limiter, class NumericalFlux>
  void stage_rhs(MultiFab& s, MultiFab& R, bool recompute_aux) {
    if (recompute_aux)
      update_aux(s);
    fill_ghosts(s, geom_.domain, bcU_);
    assemble_rhs<Limiter, NumericalFlux>(model_, s, aux_, geom_, R);
  }

  void derive_aux() {
    fill_ghosts(mg_.phi(), geom_.domain, bcPhi_);
    const Real cx = Real(1) / (2 * geom_.dx());
    const Real cy = Real(1) / (2 * geom_.dy());
    detail::coupler_grad_phi(mg_.phi(), aux_, cx, cy);
    fill_ghosts(aux_, geom_.domain, aux_bc_);
  }

  // Fills the B_z aux component (index kAuxBaseComps) on valid cells from
  // bz_(x, y), once only (B_z static). Compile-time guard: without a B_z field in the
  // model (aux_comps == 3) the component does not exist -> no code, no out-of-bound access.
  // The B_z halos are then maintained by derive_aux (Foextrap/periodic of aux_bc_,
  // cf. grad); field_postprocess only writes phi/grad (components 0..2), B_z is preserved.
  void fill_bz() {
    if constexpr (aux_comps<Model>() > kAuxBaseComps) {
      if (!bz_)
        return;
      for (int li = 0; li < aux_.local_size(); ++li)
        detail::fill_bz_box(aux_.fab(li), aux_.box(li), geom_, bz_);  // valid box
      fill_ghosts(aux_, geom_.domain, aux_bc_);  // B_z halos before the 1st solve
    }
  }

  Model model_;
  Geometry geom_;
  BoxArray ba_;
  DistributionMapping dm_;
  BCRec bcU_, bcPhi_, aux_bc_;
  Elliptic mg_;
  MultiFab aux_;
  std::function<Real(Real, Real)> bz_;  // external B_z(x, y) (empty if not provided)
};

// The coupler elliptic backend honors the common contract: swapping
// GeometricMG for another conforming solver (FFT wrapper, PETSc) will only
// require changing the member type, not the coupling logic.
static_assert(EllipticSolver<GeometricMG>, "GeometricMG must model the EllipticSolver concept");

}  // namespace adc
