#pragma once

#include <adc/core/coupled_system.hpp>
#include <adc/core/types.hpp>

#include <type_traits>

// Scheduler minimal des systemes couples.
//
// Le scheduler ne connait pas la physique. Il lit la politique temporelle de chaque
// EquationBlock et appelle un callable utilisateur sur des sous-pas. C'est le
// squelette qui permet :
//   electrons : 10 sous-pas implicites
//   ions      : 1 pas explicite
// sans hard-coder cette logique dans un Coupler mono-modele.

namespace adc {

template <class Block>
constexpr int block_substeps_v =
    TimePolicyTraits<typename std::decay_t<Block>::Time>::substeps;

template <class Block>
constexpr TimeTreatment block_time_treatment_v =
    TimePolicyTraits<typename std::decay_t<Block>::Time>::treatment;

template <CoupledSystemLike System, class AdvanceBlock>
void advance_subcycled(System& system, Real dt, AdvanceBlock&& advance_block) {
  system.for_each_block([&](auto& block) {
    using Block = std::decay_t<decltype(block)>;
    constexpr int n = block_substeps_v<Block>;
    if constexpr (block_time_treatment_v<Block> != TimeTreatment::Prescribed) {
      const Real h = dt / static_cast<Real>(n);
      for (int s = 0; s < n; ++s) advance_block(block, h, s, n);
    }
  });
}

}  // namespace adc
