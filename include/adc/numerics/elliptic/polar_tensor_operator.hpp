#pragma once

/// @file
/// @brief Compatibility forwarding header (ADC-334). The implementation moved to
///        the `polar/` solver-family directory; this shim keeps the historical
///        include path `<adc/numerics/elliptic/polar_tensor_operator.hpp>` valid. Prefer the new path
///        `<adc/numerics/elliptic/polar/polar_tensor_operator.hpp>` in new code.

#include <adc/numerics/elliptic/polar/polar_tensor_operator.hpp>
