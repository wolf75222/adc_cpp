#pragma once

/// @file
/// @brief Compatibility forwarding header (ADC-334). The implementation moved to
///        the `linear/` solver-family directory; this shim keeps the historical
///        include path `<adc/numerics/elliptic/krylov_solver.hpp>` valid. Prefer the new path
///        `<adc/numerics/elliptic/linear/krylov_solver.hpp>` in new code.

#include <adc/numerics/elliptic/linear/krylov_solver.hpp>
