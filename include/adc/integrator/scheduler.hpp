#pragma once

#include <adc/core/coupled_system.hpp>
#include <adc/core/types.hpp>

#include <type_traits>
#include <utility>

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

// Cadence du bloc (retour tuteur) : il n'avance qu'1 macro-pas sur `stride`.
template <class Block>
constexpr int block_stride_v =
    TimePolicyTraits<typename std::decay_t<Block>::Time>::stride;

// Variante avec compteur de macro-pas : un bloc de cadence `stride` n'avance qu'aux
// macro-pas multiples de stride, et alors d'un pas EFFECTIF stride*dt (il rattrape le
// temps). Total au bout de M macro-pas = M*dt comme les autres, mais calcule M/stride
// fois (le « gaz pas resolu tous les pas »). substeps reste orthogonal (decoupe le pas
// effectif). stride=1 -> chaque macro-pas, pas effectif dt : strictement l'historique.
template <CoupledSystemLike System, class AdvanceBlock>
void advance_subcycled(System& system, Real dt, int macro_step,
                       AdvanceBlock&& advance_block) {
  system.for_each_block([&](auto& block) {
    using Block = std::decay_t<decltype(block)>;
    if constexpr (block_time_treatment_v<Block> != TimeTreatment::Prescribed) {
      constexpr int stride = block_stride_v<Block>;
      if (macro_step % stride != 0) return;  // bloc lent : saute ce macro-pas
      constexpr int n = block_substeps_v<Block>;
      const Real h = (dt * static_cast<Real>(stride)) / static_cast<Real>(n);
      for (int s = 0; s < n; ++s) advance_block(block, h, s, n);
    }
  });
}

// Surcharge historique (sans cadence) : macro_step = 0 -> tous les blocs avancent chaque
// pas (stride ignore). Bit-identique a l'ancien advance_subcycled.
template <CoupledSystemLike System, class AdvanceBlock>
void advance_subcycled(System& system, Real dt, AdvanceBlock&& advance_block) {
  advance_subcycled(system, dt, 0, std::forward<AdvanceBlock>(advance_block));
}

}  // namespace adc
