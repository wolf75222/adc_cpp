#pragma once

/// @file
/// @brief Compatibility forwarding header (ADC-330). The implementation moved to
///        the `detail/` runtime layer; this shim keeps the historical include path
///        `<adc/runtime/dynamic_model.hpp>` valid. Prefer `<adc/runtime/detail/dynamic_model.hpp>` in new code.

#include <adc/runtime/detail/dynamic_model.hpp>
