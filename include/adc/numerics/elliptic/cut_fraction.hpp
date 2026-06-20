#pragma once

/// @file
/// @brief Compatibility forwarding header (ADC-334). The implementation moved to
///        the `eb/` solver-family directory; this shim keeps the historical
///        include path `<adc/numerics/elliptic/cut_fraction.hpp>` valid. Prefer the new path
///        `<adc/numerics/elliptic/eb/cut_fraction.hpp>` in new code.

#include <adc/numerics/elliptic/eb/cut_fraction.hpp>
