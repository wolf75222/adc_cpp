#pragma once

/// @file
/// @brief Compatibility forwarding header (ADC-334). The implementation moved to
///        the `mg/` solver-family directory; this shim keeps the historical
///        include path `<adc/numerics/elliptic/composite_fac_poisson.hpp>` valid. Prefer the new path
///        `<adc/numerics/elliptic/mg/composite_fac_poisson.hpp>` in new code.

#include <adc/numerics/elliptic/mg/composite_fac_poisson.hpp>
