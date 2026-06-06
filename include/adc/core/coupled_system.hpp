/// @file
/// @brief CoupledSystem : collection heterogene de blocs d'equations (multi-especes, multi-schemas).
///
/// CoupledSystem<Blocks...> est un tuple de blocs typables statiquement. Il ne resout rien
/// seul : il fournit un objet nomme a la couche assembler/scheduler pour iterer sur N blocs
/// (for_each_block) et les couplages (Poisson, collisions, sources croisees).
///
/// `CoupledSystemLike` : concept utilise par la facade AmrSystemCoupler et les schedulers.
/// Evite de passer CoupledSystem<...> concret dans du code generique qui n'a besoin que de
/// for_each_block et n_blocks.
///
/// INVARIANT nvcc/EDG : for_each_block n'accepte pas de lambda generique dans un
/// requires-clause (lambda etendue en contexte non evalue -> erreur EDG). Le concept
/// utilise donc le foncteur nomme detail::ForEachBlockProbe.

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

/// Collection heterogene de blocs d'equations, parametree par leurs types exacts.
///
/// Iteration sur tous les blocs : for_each_block(f) appelle f(block) pour chaque bloc.
/// Acces indexe : block<I>() (temps de compilation). n_blocks = sizeof...(Blocks).
///
/// INVARIANT : les blocs sont possedes par valeur dans le tuple; les MultiFab qu'ils
/// referenceent (via EquationBlock::state) doivent avoir une duree de vie superieure
/// a celle du CoupledSystem.
/// Construction par deduction : CoupledSystem{b1, b2} deduit les types via le guide de
/// deduction fourni en fin de fichier.
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
/// Sonde no-op NOMMEE pour le concept CoupledSystemLike. Une lambda generique dans le
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

/// Concept minimal pour les systemes couples : n_blocks et for_each_block avec un foncteur nomme.
/// Satisfait par CoupledSystem<...>. Utilise par AmrSystemCoupler et les schedulers pour eviter
/// de parametriser sur le type CoupledSystem<...> concret.
/// NOTE nvcc : ne pas passer de lambda generique ici (EDG : lambda etendue non evaluable);
/// utiliser ForEachBlockProbe ou un foncteur nomme.
template <class S>
concept CoupledSystemLike = requires(S s) {
  { S::n_blocks } -> std::convertible_to<std::size_t>;
  s.for_each_block(detail::ForEachBlockProbe{});
};

template <EquationBlockLike... Blocks>
CoupledSystem(Blocks...) -> CoupledSystem<Blocks...>;

}  // namespace adc
