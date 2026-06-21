/// @file
/// @brief "Spatial method" aggregate: reconstruction (limiter) + numerical flux in one named type.
///
/// SpatialDiscretisation<Limiter, NumericalFlux> bundles the two spatial discretisation choices
/// into a single template type passed as a block to assemble_rhs / compute_face_fluxes. This lets
/// you choose the method PER sub-model (Rusanov for the ions, HLLC for the electrons) without
/// touching the model or the coupler. The choice is resolved at compile time, with zero branch
/// overhead.
///
/// Provided aliases: FirstOrder, MusclMinmod, MusclVanLeer, MusclVanLeerHLLC.

#pragma once

#include <adc/numerics/numerical_flux.hpp>
#include <adc/numerics/reconstruction.hpp>

#include <concepts>

namespace adc {

/// SpatialDiscretisation<LimiterT, NumericalFluxT>: tag-type bundling the reconstruction policy
/// and the numerical flux policy into a single template parameter.
///
/// Satisfies SpatialDiscretisationLike. Exposes two aliases Limiter / NumericalFlux
/// consumed by assemble_rhs and compute_face_fluxes.
template <class LimiterT, class NumericalFluxT = RusanovFlux>
struct SpatialDiscretisation {
  using Limiter = LimiterT;
  using NumericalFlux = NumericalFluxT;
};

/// SpatialDiscretisationLike: concept validating a SpatialDiscretisation.
///
/// Checks for the presence of the Limiter and NumericalFlux aliases. Usable as a constraint
/// on a template parameter to document the expected interface.
template <class D>
concept SpatialDiscretisationLike = requires {
  typename D::Limiter;
  typename D::NumericalFlux;
};

// Usual bundles.
using FirstOrder = SpatialDiscretisation<NoSlope, RusanovFlux>;  // robust 1st order
using MusclMinmod = SpatialDiscretisation<Minmod, RusanovFlux>;  // 2nd order TVD
using MusclVanLeer = SpatialDiscretisation<VanLeer, RusanovFlux>;
using MusclVanLeerHLLC = SpatialDiscretisation<VanLeer, HLLCFlux>;  // 2nd order + HLLC

}  // namespace adc
