#pragma once

/// @file
/// @brief Compat forwarder (DEPRECATED location). The AdvectionDiffusion validation/reference brick
///        moved to adc/validation/physics/advection_diffusion.hpp under namespace adc::validation
///        (ADC-329); it is no longer part of the production brick surface.
///
/// Prefer including <adc/validation/physics/advection_diffusion.hpp> and naming the type
/// adc::validation::AdvectionDiffusion. This header keeps the old include path and the
/// adc::AdvectionDiffusion name working for existing/external callers; it may be removed later.

#include <adc/validation/physics/advection_diffusion.hpp>

namespace adc {
using validation::AdvectionDiffusion;  ///< deprecated alias; prefer adc::validation::AdvectionDiffusion
}  // namespace adc
