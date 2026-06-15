/// @file
/// @brief Agregat "methode spatiale" : reconstruction (limiteur) + flux numerique en un type nomme.
///
/// SpatialDiscretisation<Limiter, NumericalFlux> regroupe les deux choix de discretisation spatiale
/// en un seul type de template passe en bloc a assemble_rhs / compute_face_fluxes. Cela permet de
/// choisir la methode PAR sous-modele (Rusanov pour les ions, HLLC pour les electrons) sans toucher
/// au modele ni au coupleur. Choix compile a la compilation, zero surcout de branche.
///
/// Aliases fournis : FirstOrder, MusclMinmod, MusclVanLeer, MusclVanLeerHLLC.

#pragma once

#include <adc/numerics/numerical_flux.hpp>
#include <adc/numerics/reconstruction.hpp>

#include <concepts>

// Discretisation spatiale : une reconstruction (limiteur) + un flux numerique,
// regroupes en un type nomme passe en bloc au coupleur. C'est la "methode spatiale"
// (a distinguer de l'integration en temps, cf. time/time_integrator.hpp).
//
// Le tuteur : separer la discretisation spatiale de l'integration temporelle, et
// pouvoir la choisir PAR sous-modele (Rusanov pour les ions, HLLC pour les electrons)
// sans toucher au modele ni au coupleur. Ici le choix est un parametre de template,
// donc resolu a la compilation, sans surcout de branche.

namespace adc {

/// SpatialDiscretisation<LimiterT, NumericalFluxT> : tag-type regroupant la politique de
/// reconstruction et la politique de flux numerique en un seul parametre de template.
///
/// Satisfait SpatialDiscretisationLike. Expose deux alias Limiter / NumericalFlux
/// consommes par assemble_rhs et compute_face_fluxes.
template <class LimiterT, class NumericalFluxT = RusanovFlux>
struct SpatialDiscretisation {
  using Limiter = LimiterT;
  using NumericalFlux = NumericalFluxT;
};

/// SpatialDiscretisationLike : concept de validation d'un SpatialDiscretisation.
///
/// Verifie la presence des alias Limiter et NumericalFlux. Utilisable comme contrainte
/// sur un parametre de template pour documenter l'interface attendue.
template <class D>
concept SpatialDiscretisationLike = requires {
  typename D::Limiter;
  typename D::NumericalFlux;
};

// Bundles usuels.
using FirstOrder       = SpatialDiscretisation<NoSlope, RusanovFlux>;   // ordre 1 robuste
using MusclMinmod      = SpatialDiscretisation<Minmod,  RusanovFlux>;   // ordre 2 TVD
using MusclVanLeer     = SpatialDiscretisation<VanLeer, RusanovFlux>;
using MusclVanLeerHLLC = SpatialDiscretisation<VanLeer, HLLCFlux>;      // ordre 2 + HLLC

}  // namespace adc
