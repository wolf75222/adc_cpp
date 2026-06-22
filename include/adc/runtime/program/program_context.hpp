#pragma once

#include <functional>
#include <utility>

#include <adc/core/foundation/types.hpp>  // Real
#include <adc/mesh/storage/mf_arith.hpp>  // saxpy (linear combine over a MultiFab)
#include <adc/mesh/storage/multifab.hpp>  // MultiFab
#include <adc/runtime/system.hpp>         // System (the runtime this facade forwards to)

/// @file
/// @brief ProgramContext -- the C++-side facade a generated problem.so calls to run a compiled time
///        Program during sim.step(dt) (epic ADC-399, ADC-401 Phase 2b).
///
/// It REIMPLEMENTS NOTHING. Each method forwards to an existing adc::System primitive:
///   install(fn)          -> System::install_program_step(fn)   (registers the macro-step body)
///   solve_fields()       -> System::solve_fields()             (elliptic solve + aux at current U)
///   n_blocks()           -> System::n_blocks()
///   state(b)             -> System::block_state(b)             (the block's live MultiFab, zero-copy)
///   rhs_into(b, U, R)    -> System::block_rhs_into(b, U, R)    (R <- -div F + S, Poisson frozen)
///   axpy(U, a, R)        -> adc::saxpy(U, a, R)                (U <- U + a R, device-dispatched)
///
/// The Program composes the chain (e.g. Forward Euler = solve_fields(); for each block:
/// rhs_into(b, U, R); axpy(U, dt, R)) and installs it via install(...). The .so NEVER touches
/// System::Impl / Array4 / fill_boundary / the elliptic solver / Kokkos / MPI / CFL / substeps.
///
/// IDIOM: ProgramContext is a plain (non-template) class holding a System*. A generated .so receives
/// the System as a flat void* across the dlopen boundary (like the native loader's `void* self`) and
/// wraps it here; it reaches per-block storage through the System's public accessors because
/// System::Impl is private to the _adc translation unit.
namespace adc {
namespace runtime {
namespace program {

class ProgramContext {
 public:
  explicit ProgramContext(System* sys) : sys_(sys) {}
  /// Wraps a System passed as a flat void* (what adc_install_program(void* sys) receives).
  explicit ProgramContext(void* sys) : sys_(static_cast<System*>(sys)) {}

  /// Register the macro-step body. @p step advances ONE macro-step over dt (it owns solve_fields,
  /// the RHS, the linear combine and the commit). Empty std::function clears it.
  void install(std::function<void(double)> step) const {
    sys_->install_program_step(std::move(step));
  }

  void solve_fields() const { sys_->solve_fields(); }
  int n_blocks() const { return sys_->n_blocks(); }
  MultiFab& state(int b) const { return sys_->block_state(b); }
  void rhs_into(int b, MultiFab& u, MultiFab& r) const { sys_->block_rhs_into(b, u, r); }

  /// A zero-initialized RHS scratch with the SAME layout (box array / distribution / ghosts) as @p u,
  /// so the subsequent axpy(u, ., r) combines identical layouts.
  MultiFab rhs_scratch_like(const MultiFab& u) const {
    return MultiFab(u.box_array(), u.dmap(), u.ncomp(), u.n_grow());
  }

  /// A zero-initialized scratch STATE with the same layout as @p u: an intermediate stage state of a
  /// multi-stage scheme (SSPRK/RK). Same allocation as rhs_scratch_like; named for the codegen's
  /// intent. Starts at zero, so a stage `sum_i c_i V_i` is built by axpy-ing each term onto it.
  MultiFab scratch_state_like(const MultiFab& u) const { return rhs_scratch_like(u); }

  /// u <- u + a r over the valid cells (linear combine; forwards to adc::saxpy).
  void axpy(MultiFab& u, Real a, const MultiFab& r) const { adc::saxpy(u, a, r); }

  /// z <- a x + b y over the valid cells (assignment, not accumulation; z may alias x or y).
  /// Forwards to adc::lincomb. The codegen uses it for the committed stage: the block state becomes
  /// z = c_base * z + 1 * acc, where acc holds the non-base terms (self-alias z==x is safe).
  void lincomb(MultiFab& z, Real a, const MultiFab& x, Real b, const MultiFab& y) const {
    adc::lincomb(z, a, x, b, y);
  }

 private:
  System* sys_;
};

}  // namespace program
}  // namespace runtime
}  // namespace adc
