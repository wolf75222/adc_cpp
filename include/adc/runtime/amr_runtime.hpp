#pragma once

/// @file
/// @brief Compatibility forwarding header (ADC-330). The implementation moved to
///        the `amr/` runtime layer; this shim keeps the historical include path
///        `<adc/runtime/amr_runtime.hpp>` valid. Prefer `<adc/runtime/amr/amr_runtime.hpp>` in new code.

#include <adc/runtime/amr/amr_runtime.hpp>
