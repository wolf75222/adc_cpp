#pragma once

#include <adc/core/coupled_system.hpp>
#include <adc/core/types.hpp>
#include <adc/coupling/base/aux_fill.hpp>  // detail::derive_aux_bc + detail::fill_bz_box (shared)
#include <adc/coupling/source/coupled_source.hpp>
#include <adc/coupling/base/elliptic_rhs.hpp>
#include <adc/numerics/elliptic/elliptic_problem.hpp>
#include <adc/numerics/elliptic/elliptic_solver.hpp>
#include <adc/numerics/elliptic/geometric_mg.hpp>
#include <adc/numerics/time/implicit_stepper.hpp>
#include <adc/numerics/time/scheduler.hpp>
#include <adc/numerics/time/time_steppers.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/spatial_operator.hpp>
#include <adc/parallel/comm.hpp>

#include <algorithm>
#include <functional>
#include <type_traits>
#include <utility>

/// @file
/// @brief Single-level multi-species coupled system: SystemAssembler (assembles) + SystemDriver (advances).
///
/// Two responsibilities, two classes (advisor feedback 8.2 B). SystemAssembler ASSEMBLES: system RHS
/// (f = Sum_s q_s n_s), Poisson, aux = (phi, grad phi), and a block residual evaluator
/// R = -div F + S; it does NO time stepping. SystemDriver ADVANCES: carries the schedule
/// (per-species subcycling, adaptive multirate, implicit/IMEX delegated), OWNS an Assembler and
/// calls a TimeStepper. SystemCoupler stays an ALIAS of SystemDriver (test compat / adc_cases
/// facade). The aux channel is SHARED by all blocks, allocated at the MAXIMUM requested width
/// (aux_comps): a block reading B_z (n_aux=4) sees it, a base block (3) ignores the extra component.
/// Neither replaces PhysicalModel / assemble_rhs / GeometricMG: they CONNECT them.

namespace adc {

namespace detail {
template <class>
inline constexpr bool always_false_v = false;

template <class Block>
struct ScopedBlockState {
  Block& block;
  MultiFab* old_state;

  ScopedBlockState(Block& b, MultiFab& stage_state)
      : block(b), old_state(b.state) {
    block.state = &stage_state;
  }

  // RULE OF FIVE (C.21): scope-guard with a side effect in the dtor (restores block.state). Copy/move
  // BY DEFAULT -> double restoration or restoration from a dead copy. Never copied nor moved
  // (always a block-scoped local variable): delete the four operations.
  ScopedBlockState(const ScopedBlockState&) = delete;
  ScopedBlockState& operator=(const ScopedBlockState&) = delete;
  ScopedBlockState(ScopedBlockState&&) = delete;
  ScopedBlockState& operator=(ScopedBlockState&&) = delete;

  ~ScopedBlockState() { block.state = old_state; }
};
}  // namespace detail

// === ASSEMBLER: fields (system Poisson + aux) + block residual. No stepping. ======
/// ASSEMBLES the fields (system Poisson + shared aux) and a block residual evaluator. No time
/// stepping. @tparam System: CoupledSystem. @tparam RhsAssembler: Poisson RHS assembler.
/// @tparam Elliptic: elliptic backend (EllipticSolver concept, default GeometricMG).
template <CoupledSystemLike System, class RhsAssembler,
          class Elliptic = GeometricMG>
class SystemAssembler {
  static_assert(EllipticSolver<Elliptic>,
                "the elliptic backend must model EllipticSolver");

 public:
  // bz: out-of-plane magnetic field B_z(x, y) supplied by the user (constant or field),
  // shared by ALL blocks. The SHARED aux channel is allocated at the MAXIMUM width requested
  // by the blocks (aux_comps): a block reading B_z (n_aux=4) sees it, a base block (3)
  // ignores the component. Without an extra-field block the width stays 3 -> allocation and numerics
  // strictly bit-identical to history.
  SystemAssembler(System system, const Geometry& geom, const BoxArray& ba,
                  const BCRec& bcPhi, RhsAssembler rhs_assembler,
                  std::function<bool(Real, Real)> active = {},
                  std::function<Real(Real, Real)> bz = {})
      : system_(std::move(system)),
        rhs_assembler_(std::move(rhs_assembler)),
        geom_(geom),
        ba_(ba),
        dm_(ba.size(), n_ranks()),
        bcPhi_(bcPhi),
        aux_bc_(detail::derive_aux_bc(bcPhi)),
        mg_(geom, ba, bcPhi, std::move(active)),
        aux_ncomp_(system_aux_comps(system_)),
        aux_(ba, dm_, aux_ncomp_, 1),
        bz_(std::move(bz)) {
    fill_bz();  // populates B_z (no-op if no block requests it or if bz is empty)
  }

  System& system() { return system_; }
  const System& system() const { return system_; }
  MultiFab& phi() { return mg_.phi(); }
  MultiFab& aux() { return aux_; }
  const MultiFab& aux() const { return aux_; }
  const Geometry& geom() const { return geom_; }
  const BoxArray& ba() const { return ba_; }
  const DistributionMapping& dm() const { return dm_; }

  /// Solves the system RHS (Sum_s q_s n_s), the Poisson, then derives aux = (phi, grad phi). aux()
  /// is up to date on return.
  void solve_fields() {
    rhs_assembler_(system_, mg_.rhs());
    mg_.solve();
    derive_aux();
  }

  /// Residual R = -div F + S of a block at a stage (with field re-solve if @p recompute_aux).
  /// This is the method-of-lines arrow the Driver passes to the TimeStepper. Fills the ghosts of
  /// @p state per block.bc before assembly.
  template <class Limiter, class NumericalFlux, class Block>
  void block_residual(Block& block, MultiFab& state, MultiFab& R,
                      bool recompute_aux) {
    if (recompute_aux) {
      detail::ScopedBlockState<Block> swap(block, state);
      solve_fields();
    }
    fill_ghosts(state, geom_.domain, block.bc);
    assemble_rhs<Limiter, NumericalFlux>(block.model, state, aux_, geom_, R);
  }

 private:
  void derive_aux() {
    fill_ghosts(mg_.phi(), geom_.domain, bcPhi_);
    const Real cx = Real(1) / (2 * geom_.dx());
    const Real cy = Real(1) / (2 * geom_.dy());
    field_postprocess(mg_.phi(), aux_, cx, cy,
                      FieldPostProcess{FieldPostProcess::GradSign::Plus, true});
    fill_ghosts(aux_, geom_.domain, aux_bc_);
  }

  // Width of the SHARED aux channel: maximum of aux_comps<Model> over all blocks (at least
  // kAuxBaseComps). The shared channel must be at least as wide as the most demanding block
  // so that load_aux<aux_comps<Model>> never reads out of bounds; a less demanding block
  // simply ignores the extra components.
  static int system_aux_comps(const System& sys) {
    int w = kAuxBaseComps;
    sys.for_each_block([&](const auto& b) {
      using Model = std::decay_t<decltype(b.model)>;
      const int c = aux_comps<Model>();
      if (c > w) w = c;
    });
    return w;
  }

  // Populates the aux B_z component (index kAuxBaseComps) of the shared channel from bz_(x, y), a
  // single time (static B_z). No-op if no block declares B_z (width 3) or if bz_ is empty:
  // RUNTIME guard on aux_ncomp_ (the width is only known at construction). Halos are then
  // maintained by derive_aux (aux_bc_); field_postprocess only writes phi/grad (comp 0..2).
  void fill_bz() {
    if (!bz_ || aux_ncomp_ <= kAuxBaseComps) return;
    for (int li = 0; li < aux_.local_size(); ++li)
      detail::fill_bz_box(aux_.fab(li), aux_.box(li), geom_, bz_);  // valid box
    fill_ghosts(aux_, geom_.domain, aux_bc_);  // B_z halos before the 1st solve
  }

  System system_;
  RhsAssembler rhs_assembler_;
  Geometry geom_;
  BoxArray ba_;
  DistributionMapping dm_;
  BCRec bcPhi_, aux_bc_;
  Elliptic mg_;
  int aux_ncomp_;  // width of the shared aux channel (max over blocks); init before aux_
  MultiFab aux_;
  std::function<Real(Real, Real)> bz_;  // external B_z(x, y) (empty if not supplied)
};

// === DRIVER: advances the system. Owns an Assembler, delegates the fields to it. =========
/// ADVANCES the system: carries the schedule (per-species subcycling, adaptive multirate,
/// implicit/IMEX delegated) and calls a TimeStepper. OWNS a SystemAssembler and delegates the
/// fields to it. Same template parameters as SystemAssembler.
template <CoupledSystemLike System, class RhsAssembler,
          class Elliptic = GeometricMG>
class SystemDriver {
 public:
  /// Builds the driver (which builds the underlying assembler). @p active: optional wall predicate
  /// passed to the MG; @p bz: optional B_z(x, y) field shared by the blocks.
  SystemDriver(System system, const Geometry& geom, const BoxArray& ba,
               const BCRec& bcPhi, RhsAssembler rhs_assembler,
               std::function<bool(Real, Real)> active = {},
               std::function<Real(Real, Real)> bz = {})
      : asm_(std::move(system), geom, ba, bcPhi, std::move(rhs_assembler),
             std::move(active), std::move(bz)) {}

  // Accessors delegated to the assembler (compat with the old SystemCoupler API).
  System& system() { return asm_.system(); }
  const System& system() const { return asm_.system(); }
  MultiFab& phi() { return asm_.phi(); }
  const MultiFab& aux() const { return asm_.aux(); }
  void solve_fields() { asm_.solve_fields(); }
  SystemAssembler<System, RhsAssembler, Elliptic>& assembler() { return asm_; }

  /// Advances the blocks by a macro-step dt per their TimePolicy: explicit ones via TimeStepper,
  /// implicit/IMEX delegated to @p implicit_advance (block, h, s, n). Per-block stride cadence.
  template <class ImplicitAdvance>
  void step(Real dt, ImplicitAdvance&& implicit_advance) {
    ImplicitAdvance& advance_implicit = implicit_advance;
    // macro_step_: a block of cadence `stride` only advances 1 macro-step out of stride
    // (then by an effective step stride*dt). stride=1 -> every step (history).
    advance_subcycled(asm_.system(), dt, macro_step_, [&](auto& block, Real h, int s, int n) {
      advance_block_dispatch(block, h, s, n, advance_implicit);
    });
    ++macro_step_;
  }

  /// FULLY ADAPTIVE multirate: macro-step fixed by the fastest species (CFL @p cfl), stride
  /// of each species derived at RUNTIME from the wave-speed ratio (slow species advanced less
  /// often, larger step). Returns the macro-step. @p implicit_advance handles implicit/IMEX blocks.
  template <class ImplicitAdvance>
  Real step_adaptive(Real cfl, ImplicitAdvance&& implicit_advance) {
    ImplicitAdvance& advance_implicit = implicit_advance;
    asm_.solve_fields();  // aux up to date for the wave speeds
    const Real h = std::min(asm_.geom().dx(), asm_.geom().dy());
    const Real wmax = system_max_wave_speed();
    const Real macro_dt = cfl * h / std::max(wmax, Real(1e-30));
    asm_.system().for_each_block([&](auto& block) {
      using Block = std::decay_t<decltype(block)>;
      if constexpr (block_time_treatment_v<Block> != TimeTreatment::Prescribed) {
        const Real w_s = max_wave_speed_mf(block.model, block.U(), asm_.aux());
        const int stride = (w_s <= Real(0))
                               ? 1
                               : std::max(1, static_cast<int>(wmax / w_s));
        if (macro_step_ % stride == 0) {
          constexpr int n = block_substeps_v<Block>;
          const Real hh = (macro_dt * static_cast<Real>(stride)) / static_cast<Real>(n);
          for (int s = 0; s < n; ++s)
            advance_block_dispatch(block, hh, s, n, advance_implicit);
        }
      }
    });
    ++macro_step_;
    return macro_dt;
  }
  Real step_adaptive(Real cfl) {
    return step_adaptive(cfl, [](auto&, auto& block, Real, int, int) {
      using Block = std::decay_t<decltype(block)>;
      static_assert(detail::always_false_v<Block>,
                    "SystemDriver::step_adaptive(cfl) cannot advance an "
                    "implicit/IMEX block without a callback");
    });
  }

  // Convenience overload for a fully explicit system.
  void step(Real dt) {
    step(dt, [](auto&, auto& block, Real, int, int) {
      using Block = std::decay_t<decltype(block)>;
      static_assert(detail::always_false_v<Block>,
                    "SystemDriver::step(dt) cannot advance an "
                    "implicit/IMEX block without a callback");
    });
  }

  /// Macro-step chosen by multi-species CFL: dt = cfl * min(dx, dy) / w_max (w_max = largest
  /// wave speed over ALL species). Refreshes aux before the measurement.
  Real cfl_dt(Real cfl) {
    asm_.solve_fields();
    const Real h = std::min(asm_.geom().dx(), asm_.geom().dy());
    return cfl * h / std::max(system_max_wave_speed(), Real(1e-30));
  }
  template <class ImplicitAdvance>
  Real step_cfl(Real cfl, ImplicitAdvance&& implicit_advance) {
    const Real dt = cfl_dt(cfl);
    step(dt, std::forward<ImplicitAdvance>(implicit_advance));
    return dt;
  }
  Real step_cfl(Real cfl) {
    const Real dt = cfl_dt(cfl);
    step(dt);
    return dt;
  }

  /// Applies an inter-species COUPLING source (forward-Euler splitting): refreshes phi (aux)
  /// then calls src.apply(system, aux, dt). Distinct from model.source (block-local).
  template <class CoupledSource>
  void coupled_source_step(CoupledSource&& src, Real dt) {
    static_assert(CoupledSourceFor<std::decay_t<CoupledSource>, System>,
                  "coupled_source_step expects a CoupledSource: "
                  "apply(system, aux, dt)");
    asm_.solve_fields();
    src.apply(asm_.system(), asm_.aux(), dt);
  }

 private:
  // Largest wave speed over ALL species (aux assumed up to date). Fixes the CFL step.
  Real system_max_wave_speed() {
    Real wmax = 0;
    asm_.system().for_each_block([&](auto& block) {
      wmax = std::max(wmax, max_wave_speed_mf(block.model, block.U(), asm_.aux()));
    });
    return wmax;
  }

  // Dispatch of a (sub-)step for ONE block, per its treatment. Shared by step (compile-time
  // cadence) and step_adaptive (runtime CFL cadence): no duplication.
  //   Explicit: advance via TimeStepper. Implicit/IMEX: re-solve the fields, (IMEX)
  //   explicit transport, then implicit source via the callback.
  template <class Block, class ImplicitAdvance>
  void advance_block_dispatch(Block& block, Real h, int s, int n,
                              ImplicitAdvance& advance_implicit) {
    constexpr TimeTreatment treatment = block_time_treatment_v<Block>;
    if constexpr (treatment == TimeTreatment::Explicit) {
      advance_explicit_block(block, h);
    } else if constexpr (treatment == TimeTreatment::Implicit ||
                         treatment == TimeTreatment::IMEX) {
      asm_.solve_fields();
      if constexpr (treatment == TimeTreatment::IMEX) explicit_transport(block, h);
      advance_implicit(*this, block, h, s, n);
    }
  }

  // Explicit advance of a block: DELEGATES the scheme to a TimeStepper object (core SSPRK2/3
  // or user integrator), passing it the assembler's residual evaluator.
  template <class Block>
  void advance_explicit_block(Block& block, Real dt) {
    using Time = TimePolicyTraits<typename Block::Time>;
    using Method = typename Time::Method;
    using Limiter = typename Block::Spatial::Limiter;
    using NumericalFlux = typename Block::Spatial::NumericalFlux;
    static_assert(Time::treatment == TimeTreatment::Explicit,
                  "advance_explicit_block expects an explicit block");

    auto rhs_eval = [&](MultiFab& stage, MultiFab& R) {
      asm_.template block_residual<Limiter, NumericalFlux>(block, stage, R,
                                                           /*recompute_aux=*/true);
    };
    if constexpr (std::is_same_v<Method, SSPRK3>)
      SSPRK3Step{}.take_step(rhs_eval, block.U(), dt);
    else if constexpr (std::is_same_v<Method, SSPRK2>)
      SSPRK2Step{}.take_step(rhs_eval, block.U(), dt);
    else if constexpr (TimeStepper<Method>)
      Method{}.take_step(rhs_eval, block.U(), dt);
    else
      static_assert(detail::always_false_v<Method>,
                    "explicit Method must be SSPRK2, SSPRK3, or a TimeStepper "
                    "(object with take_step(rhs_eval, U, dt)) supplied by the user");
  }

  // EXPLICIT half-step of an IMEX block: transport only (-div F, source-free), forward Euler.
  // The stiff source is handled separately in implicit form by the callback. aux assumed up to date.
  template <class Block>
  void explicit_transport(Block& block, Real dt) {
    using Model = typename Block::Model;
    using Limiter = typename Block::Spatial::Limiter;
    using NumericalFlux = typename Block::Spatial::NumericalFlux;
    const SourceFreeModel<Model> sf{block.model};
    MultiFab R(asm_.ba(), asm_.dm(), Model::n_vars, 0);
    fill_ghosts(block.U(), asm_.geom().domain, block.bc);
    assemble_rhs<Limiter, NumericalFlux>(sf, block.U(), asm_.aux(), asm_.geom(), R);
    saxpy(block.U(), dt, R);
  }

  SystemAssembler<System, RhsAssembler, Elliptic> asm_;
  int macro_step_ = 0;  // macro-step counter (per-block stride cadence)
};

// Compat / historical naming: SystemCoupler == the Driver (which advances). We keep the alias
// SystemCoupler (tests, MultiSpeciesSolver facade) AND SystemDriver (the "advances" name).
template <CoupledSystemLike System, class RhsAssembler, class Elliptic = GeometricMG>
using SystemCoupler = SystemDriver<System, RhsAssembler, Elliptic>;

// Friendly builder for the SystemCoupler alias. CTAD written directly on an alias template
// (`SystemCoupler sim(...)`) is accepted by GCC but REJECTED by clang -- alias-template argument
// deduction (P1814) is not implemented the same way, so `SystemCoupler sim(...)` fails to compile
// under clang ("alias template requires template arguments"). This factory deduces the parameters
// through the underlying class template (CTAD on SystemDriver, which every compiler supports) and
// forwards to its constructor. Use `auto sim = make_system_coupler(system, geom, ba, bc, rhs);`.
template <class... Args>
auto make_system_coupler(Args&&... args) {
  return SystemDriver(std::forward<Args>(args)...);
}

}  // namespace adc
