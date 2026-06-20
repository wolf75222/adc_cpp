#pragma once

/// @file
/// @brief Compatibility forwarding header (ADC-330). The implementation moved to
///        the `detail/` runtime layer; this shim keeps the historical include path
///        `<adc/runtime/dispatch_tags.hpp>` valid. Prefer `<adc/runtime/detail/dispatch_tags.hpp>` in new code.

#include <adc/runtime/detail/dispatch_tags.hpp>
