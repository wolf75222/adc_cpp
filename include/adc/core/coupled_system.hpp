#pragma once

#include <adc/core/equation_block.hpp>

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

// CoupledSystem : collection de blocs d'equations.
//
// Il ne resout rien tout seul. Il donne un objet nomme a la couche
// assembler/scheduler : plusieurs U, plusieurs modeles, plusieurs methodes.
// Les politiques de couplage (Poisson, collisions, sources croisees) peuvent
// ensuite lire tous les blocs au lieu d'etre enfermees dans un seul PhysicalModel.

namespace adc {

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
    std::apply(
        [&](auto&... bs) { (std::forward<F>(f)(bs), ...); },
        blocks);
  }

  template <class F>
  void for_each_block(F&& f) const {
    std::apply(
        [&](const auto&... bs) { (std::forward<F>(f)(bs), ...); },
        blocks);
  }
};

namespace detail {
// Sonde no-op NOMMEE pour le concept CoupledSystemLike. Une lambda generique dans le
// requires-clause (s.for_each_block([](auto&){})) fait buter le frontend nvcc/EDG : une
// lambda etendue en contexte NON EVALUE n'est pas instanciable cote device -> la facade
// AmrSystemCoupler<CoupledSystemLike System> ne s'instancie pas sous Cuda (le chemin B_z-AMR
// device passait alors par advance_amr en direct). Un FONCTEUR NOMME contourne (meme recette
// que les foncteurs nommes de block_builder.hpp, #64) sans changer la semantique cote hote.
struct ForEachBlockProbe {
  template <class B>
  void operator()(B&) const {}
};
}  // namespace detail

template <class S>
concept CoupledSystemLike = requires(S s) {
  { S::n_blocks } -> std::convertible_to<std::size_t>;
  s.for_each_block(detail::ForEachBlockProbe{});
};

template <EquationBlockLike... Blocks>
CoupledSystem(Blocks...) -> CoupledSystem<Blocks...>;

}  // namespace adc
