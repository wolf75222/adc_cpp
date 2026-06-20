#pragma once

/// @file
/// @brief Compatibility forwarding header (ADC-330). The implementation moved to
///        the `detail/` runtime layer; this shim keeps the historical include path
///        `<adc/runtime/grid_context.hpp>` valid. Prefer `<adc/runtime/detail/grid_context.hpp>` in new code.

#include <adc/runtime/detail/grid_context.hpp>
