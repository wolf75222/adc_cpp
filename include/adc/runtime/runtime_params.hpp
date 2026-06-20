#pragma once

/// @file
/// @brief Compatibility forwarding header (ADC-330). The implementation moved to
///        the `detail/` runtime layer; this shim keeps the historical include path
///        `<adc/runtime/runtime_params.hpp>` valid. Prefer `<adc/runtime/detail/runtime_params.hpp>` in new code.

#include <adc/runtime/detail/runtime_params.hpp>
