/// @file
/// @brief EquationBlock : association (modele, champ U, schema spatial, politique temporelle, BC).
///        Niveau d'abstraction au-dessus de PhysicalModel.
///
/// Un PhysicalModel decrit une loi locale ponctuelle. Un EquationBlock dit comment cette loi
/// est portee par la simulation : quel MultiFab U, quelle discretisation spatiale (SpatialT),
/// quelle politique temporelle (TimeT), quelles conditions aux limites (BCRec).
///
/// INVARIANT : `state` n'est jamais null apres construction (pointe vers le MultiFab passe au
/// constructeur). L'EquationBlock ne possede pas le MultiFab; la duree de vie du MultiFab doit
/// exceder celle du bloc.
///
/// `EquationBlockLike` : concept minimal permettant au CoupledSystem et au scheduler de
/// manipuler un bloc sans connaitre ses types concrets (ModelT, SpatialT, TimeT).

#pragma once

#include <adc/core/physical_model.hpp>
#include <adc/numerics/time/time_integrator.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/spatial_discretisation.hpp>

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

/// Association d'un PhysicalModel avec son champ U (MultiFab), son schema spatial, sa
/// politique temporelle et ses conditions aux limites.
///
/// Parametres templates :
///   ModelT  : doit satisfaire PhysicalModel.
///   SpatialT: doit satisfaire SpatialDiscretisationLike (defaut FirstOrder).
///   TimeT   : politique temporelle (defaut ExplicitTime<SSPRK2>).
///
/// INVARIANT : `state != nullptr` apres construction. L'EquationBlock ne possede PAS
/// le MultiFab; la duree de vie du MultiFab doit exceder celle du bloc.
/// Ne pas stocker dans un conteneur par valeur si le MultiFab est alloue dynamiquement
/// et pourrait etre deplace (le pointeur deviendrait invalide).
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

/// Concept minimal pour les blocs d'equation : State, Spatial, Time, name, state, U().
/// Permet au CoupledSystem et au scheduler de manipuler un bloc sans connaitre ses types
/// concrets. Verifie que `state` est convertible en `MultiFab*` et que U() retourne MultiFab&.
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
