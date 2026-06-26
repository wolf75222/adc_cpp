#pragma once

#include <pops/core/model/coupled_system.hpp>
#include <pops/core/foundation/types.hpp>

#include <type_traits>
#include <utility>

/// @file
/// @brief Minimal scheduler for coupled systems: advance_subcycled reads each block's time policy
///        (traits block_substeps_v / block_stride_v / block_time_treatment_v) and calls a
///        user callable on the substeps.
///
/// Layer: `include/pops/numerics/time`.
/// Role: physics-agnostic skeleton. Enables per-block cadences (e.g. electrons 10 implicit substeps,
///        ions 1 explicit step) without hard-coding any logic in a single-model Coupler. Prescribed
///        blocks are skipped.
///
/// Invariants:
/// - substeps (split: n steps of dt_eff/n) and stride (cadence: advance 1 macro-step out of stride,
///   then by an EFFECTIVE step stride*dt) are ORTHOGONAL; over M macro-steps the total stays M*dt;
/// - a strided block advances only at macro-steps that are multiples of stride (otherwise return);
/// - the overload without macro_step forces macro_step=0 (all blocks advance, stride ignored) ->
///   bit-identical to the old advance_subcycled; stride=1 = historical behavior.

namespace pops {

template <class Block>
constexpr int block_substeps_v = TimePolicyTraits<typename std::decay_t<Block>::Time>::substeps;

template <class Block>
constexpr TimeTreatment block_time_treatment_v =
    TimePolicyTraits<typename std::decay_t<Block>::Time>::treatment;

// Block cadence (tutor return): it advances only 1 macro-step out of `stride`.
template <class Block>
constexpr int block_stride_v = TimePolicyTraits<typename std::decay_t<Block>::Time>::stride;

// Variant with a macro-step counter: a block of cadence `stride` advances only at macro-steps
// that are multiples of stride, and then by an EFFECTIVE step stride*dt (it catches up on
// time). Total after M macro-steps = M*dt like the others, but computed M/stride times (the
// "gas not resolved every step"). substeps stays orthogonal (splits the effective step).
// stride=1 -> every macro-step, effective step dt: strictly the historical behavior.
template <CoupledSystemLike System, class AdvanceBlock>
void advance_subcycled(System& system, Real dt, int macro_step, AdvanceBlock&& advance_block) {
  system.for_each_block([&](auto& block) {
    using Block = std::decay_t<decltype(block)>;
    if constexpr (block_time_treatment_v<Block> != TimeTreatment::Prescribed) {
      constexpr int stride = block_stride_v<Block>;
      if (macro_step % stride != 0)
        return;  // slow block: skip this macro-step
      constexpr int n = block_substeps_v<Block>;
      const Real h = (dt * static_cast<Real>(stride)) / static_cast<Real>(n);
      for (int s = 0; s < n; ++s)
        advance_block(block, h, s, n);
    }
  });
}

// Historical overload (no cadence): macro_step = 0 -> all blocks advance every step (stride
// ignored). Bit-identical to the old advance_subcycled.
template <CoupledSystemLike System, class AdvanceBlock>
void advance_subcycled(System& system, Real dt, AdvanceBlock&& advance_block) {
  advance_subcycled(system, dt, 0, std::forward<AdvanceBlock>(advance_block));
}

}  // namespace pops
