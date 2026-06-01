#pragma once

#include <adc/core/physical_model.hpp>
#include <adc/integrator/time_integrator.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/operator/spatial_discretisation.hpp>

#include <concepts>
#include <string_view>

// Niveau d'abstraction au-dessus de PhysicalModel.
//
// Un PhysicalModel decrit une loi locale. Un EquationBlock dit comment cette loi
// est portee par la simulation : quel champ U, quelle discretisation spatiale,
// quelle politique temporelle, quelles conditions aux limites.
//
// C'est la brique manquante pour exprimer "electrons implicites en 10 sous-pas,
// ions explicites en 1 sous-pas, schemas spatiaux differents" sans melanger ces
// choix dans le modele physique.

namespace adc {

template <class ModelT, class SpatialT = FirstOrder,
          class TimeT = ExplicitTime<SSPRK2>>
struct EquationBlock {
  static_assert(PhysicalModel<ModelT>,
                "EquationBlock attend un ModelT qui modele PhysicalModel");
  static_assert(SpatialDiscretisationLike<SpatialT>,
                "EquationBlock attend une discretisation spatiale nommee");

  using Model = ModelT;
  using Spatial = SpatialT;
  using Time = TimeT;

  std::string_view name{};
  Model model;
  MultiFab* state = nullptr;
  BCRec bc{};

  EquationBlock(std::string_view block_name, const Model& block_model,
                MultiFab& block_state, const BCRec& block_bc = {})
      : name(block_name), model(block_model), state(&block_state), bc(block_bc) {}

  MultiFab& U() { return *state; }
  const MultiFab& U() const { return *state; }
};

template <class B>
concept EquationBlockLike = requires(B b) {
  typename B::Model;
  typename B::Spatial;
  typename B::Time;
  { b.name } -> std::convertible_to<std::string_view>;
  { b.state } -> std::convertible_to<MultiFab*>;
  { b.U() } -> std::same_as<MultiFab&>;
};

}  // namespace adc
