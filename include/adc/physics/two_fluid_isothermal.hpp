#pragma once

/// @file
/// @brief Compat forwarder (DEPRECATED location). The TwoFluidLinear validation/reference brick
///        moved to adc/validation/physics/two_fluid_isothermal.hpp under namespace adc::validation
///        (ADC-329); it is no longer part of the production brick surface.
///
/// Prefer including <adc/validation/physics/two_fluid_isothermal.hpp> and naming the type
/// adc::validation::TwoFluidLinear. This header keeps the old include path and the
/// adc::TwoFluidLinear name working for existing/external callers; it may be removed later.

#include <adc/validation/physics/two_fluid_isothermal.hpp>

namespace adc {
using validation::TwoFluidLinear;  ///< deprecated alias; prefer adc::validation::TwoFluidLinear
}  // namespace adc
