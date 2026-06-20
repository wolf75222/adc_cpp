#pragma once

/// @file
/// @brief Compatibility forwarding header (ADC-334). The implementation moved to
///        the `interface/` solver-family directory; this shim keeps the historical
///        include path `<adc/numerics/elliptic/elliptic_solver.hpp>` valid. Prefer the new path
///        `<adc/numerics/elliptic/interface/elliptic_solver.hpp>` in new code.

#include <adc/numerics/elliptic/interface/elliptic_solver.hpp>
