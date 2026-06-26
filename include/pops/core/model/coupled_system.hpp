/// @file
/// @brief CoupledSystem: heterogeneous collection of equation blocks (multi-species, multi-scheme).
///
/// CoupledSystem<Blocks...> is a statically typed tuple of blocks. It solves nothing
/// on its own: it provides a named object to the assembler/scheduler layer to iterate over N blocks
/// (for_each_block) and the couplings (Poisson, collisions, cross sources).
///
/// `CoupledSystemLike`: concept used by the AmrSystemCoupler facade and the schedulers.
/// Avoids passing the concrete CoupledSystem<...> into generic code that only needs
/// for_each_block and n_blocks.
///
/// nvcc/EDG INVARIANT: for_each_block does not accept a generic lambda in a
/// requires-clause (extended lambda in an unevaluated context -> EDG error). The concept
/// therefore uses the named functor detail::ForEachBlockProbe.

#pragma once

#include <pops/core/model/equation_block.hpp>

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace pops {

/// Heterogeneous collection of equation blocks, parameterized by their exact types.
///
/// Iteration over all blocks: for_each_block(f) calls f(block) for each block.
/// Indexed access: block<I>() (compile time). n_blocks = sizeof...(Blocks).
///
/// INVARIANT: the blocks are owned by value in the tuple; the MultiFab they
/// reference (via EquationBlock::state) must outlive the CoupledSystem.
/// Construction by deduction: CoupledSystem{b1, b2} deduces the types via the deduction
/// guide provided at the end of the file.
template <EquationBlockLike... Blocks>
struct CoupledSystem {
  static constexpr std::size_t n_blocks = sizeof...(Blocks);

  std::tuple<Blocks...> blocks;

  explicit CoupledSystem(Blocks... bs) : blocks(std::move(bs)...) {}

  template <std::size_t I>
  decltype(auto) block() {
    return std::get<I>(blocks);
  }

  template <std::size_t I>
  decltype(auto) block() const {
    return std::get<I>(blocks);
  }

  template <class F>
  void for_each_block(F&& f) {
    std::apply([&](auto&... bs) { (std::forward<F>(f)(bs), ...); }, blocks);
  }

  template <class F>
  void for_each_block(F&& f) const {
    std::apply([&](const auto&... bs) { (std::forward<F>(f)(bs), ...); }, blocks);
  }
};

namespace detail {
/// NAMED no-op probe for the CoupledSystemLike concept. A generic lambda in the
// requires-clause (s.for_each_block([](auto&){})) trips the nvcc/EDG frontend: an
// extended lambda in an UNEVALUATED context is not instantiable on the device side -> the
// AmrSystemCoupler<CoupledSystemLike System> facade does not instantiate under Cuda (the B_z-AMR
// device path then went through advance_amr directly). A NAMED FUNCTOR works around this (same recipe
// as the named functors in block_builder.hpp, #64) without changing the host-side semantics.
struct ForEachBlockProbe {
  template <class B>
  void operator()(B&) const {}
};
}  // namespace detail

/// Minimal concept for coupled systems: n_blocks and for_each_block with a named functor.
/// Satisfied by CoupledSystem<...>. Used by AmrSystemCoupler and the schedulers to avoid
/// parameterizing on the concrete CoupledSystem<...> type.
/// nvcc NOTE: do not pass a generic lambda here (EDG: extended lambda not evaluable);
/// use ForEachBlockProbe or a named functor.
template <class S>
concept CoupledSystemLike = requires(S s) {
  { S::n_blocks } -> std::convertible_to<std::size_t>;
  s.for_each_block(detail::ForEachBlockProbe{});
};

template <EquationBlockLike... Blocks>
CoupledSystem(Blocks...) -> CoupledSystem<Blocks...>;

}  // namespace pops
