#pragma once

/// @file
/// @brief Compat forwarder (DEPRECATED location). The LangmuirMode validation/reference brick moved
///        to adc/validation/physics/langmuir.hpp under namespace adc::validation (ADC-329); it is no
///        longer part of the production brick surface.
///
/// Prefer including <adc/validation/physics/langmuir.hpp> and naming the type
/// adc::validation::LangmuirMode. This header keeps the old include path and the adc::LangmuirMode
/// name working for existing/external callers; it may be removed later.

#include <adc/validation/physics/langmuir.hpp>

namespace adc {
using validation::LangmuirMode;  ///< deprecated alias; prefer adc::validation::LangmuirMode
}  // namespace adc
