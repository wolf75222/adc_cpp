#pragma once

/// @file
/// @brief Compatibility forwarding header (ADC-334). The implementation moved to
///        the `interface/` solver-family directory; this shim keeps the historical
///        include path `<adc/numerics/elliptic/elliptic_interface.hpp>` valid. Prefer the new path
///        `<adc/numerics/elliptic/interface/elliptic_interface.hpp>` in new code.

#include <adc/numerics/elliptic/interface/elliptic_interface.hpp>
