#pragma once

/// @file
/// @brief Compatibility forwarding header (ADC-334). The implementation moved to
///        the `poisson/` solver-family directory; this shim keeps the historical
///        include path `<adc/numerics/elliptic/poisson_operator.hpp>` valid. Prefer the new path
///        `<adc/numerics/elliptic/poisson/poisson_operator.hpp>` in new code.

#include <adc/numerics/elliptic/poisson/poisson_operator.hpp>
