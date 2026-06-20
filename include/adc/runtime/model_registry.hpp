#pragma once

/// @file
/// @brief Compatibility forwarding header (ADC-330). The implementation moved to
///        the `detail/` runtime layer; this shim keeps the historical include path
///        `<adc/runtime/model_registry.hpp>` valid. Prefer `<adc/runtime/detail/model_registry.hpp>` in new code.

#include <adc/runtime/detail/model_registry.hpp>
