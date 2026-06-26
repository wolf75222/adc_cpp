/// @file
/// @brief EquationBlock: association (model, field U, spatial scheme, time policy, BC).
///        Abstraction level above PhysicalModel.
///
/// A PhysicalModel describes a local pointwise law. An EquationBlock says how this law
/// is carried by the simulation: which MultiFab U, which spatial discretisation (SpatialT),
/// which time policy (TimeT), which boundary conditions (BCRec).
///
/// INVARIANT: `state` is never null after construction (points to the MultiFab passed to the
/// constructor). The EquationBlock does not own the MultiFab; the MultiFab lifetime must
/// exceed that of the block.
///
/// `EquationBlockLike`: minimal concept letting the CoupledSystem and the scheduler
/// manipulate a block without knowing its concrete types (ModelT, SpatialT, TimeT).

#pragma once

#include <pops/core/model/physical_model.hpp>
#include <pops/numerics/time/integrators/time_integrator.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/fv/spatial_discretisation.hpp>

#include <concepts>
#include <string_view>

namespace pops {

/// Association of a PhysicalModel with its field U (MultiFab), its spatial scheme, its
/// time policy and its boundary conditions.
///
/// Template parameters:
///   ModelT: must satisfy PhysicalModel.
///   SpatialT: must satisfy SpatialDiscretisationLike (default FirstOrder).
///   TimeT: time policy (default ExplicitTime<SSPRK2>).
///
/// INVARIANT: `state != nullptr` after construction. The EquationBlock does NOT own
/// the MultiFab; the MultiFab lifetime must exceed that of the block.
/// Do not store in a container by value if the MultiFab is dynamically allocated
/// and could be moved (the pointer would become invalid).
template <class ModelT, class SpatialT = FirstOrder, class TimeT = ExplicitTime<SSPRK2>>
struct EquationBlock {
  static_assert(PhysicalModel<ModelT>, "EquationBlock expects a ModelT that models PhysicalModel");
  static_assert(SpatialDiscretisationLike<SpatialT>,
                "EquationBlock expects a named spatial discretisation");

  using Model = ModelT;
  using Spatial = SpatialT;
  using Time = TimeT;

  std::string_view name{};
  Model model;
  MultiFab* state = nullptr;
  BCRec bc{};

  EquationBlock(std::string_view block_name, const Model& block_model, MultiFab& block_state,
                const BCRec& block_bc = {})
      : name(block_name), model(block_model), state(&block_state), bc(block_bc) {}

  MultiFab& U() { return *state; }
  const MultiFab& U() const { return *state; }
};

/// Minimal concept for equation blocks: Model, Spatial, Time, name, state, U().
/// Lets the CoupledSystem and the scheduler manipulate a block without knowing its
/// concrete types. Checks that `state` is convertible to `MultiFab*` and that U() returns MultiFab&.
template <class B>
concept EquationBlockLike = requires(B b) {
  typename B::Model;
  typename B::Spatial;
  typename B::Time;
  { b.name } -> std::convertible_to<std::string_view>;
  { b.state } -> std::convertible_to<MultiFab*>;
  { b.U() } -> std::same_as<MultiFab&>;
};

}  // namespace pops
