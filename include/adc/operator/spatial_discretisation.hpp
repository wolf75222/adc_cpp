#pragma once

#include <adc/operator/numerical_flux.hpp>
#include <adc/operator/reconstruction.hpp>

#include <concepts>

// Discretisation spatiale : une reconstruction (limiteur) + un flux numerique,
// regroupes en un type nomme passe en bloc au coupleur. C'est la "methode spatiale"
// (a distinguer de l'integration en temps, cf. integrator/time_integrator.hpp).
//
// Le tuteur : separer la discretisation spatiale de l'integration temporelle, et
// pouvoir la choisir PAR sous-modele (Rusanov pour les ions, HLLC pour les electrons)
// sans toucher au modele ni au coupleur. Ici le choix est un parametre de template,
// donc resolu a la compilation, sans surcout de branche.

namespace adc {

template <class LimiterT, class NumericalFluxT = RusanovFlux>
struct SpatialDiscretisation {
  using Limiter = LimiterT;
  using NumericalFlux = NumericalFluxT;
};

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
