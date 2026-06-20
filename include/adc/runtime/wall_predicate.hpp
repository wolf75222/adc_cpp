#pragma once

/// @file
/// @brief Compatibility forwarding header (ADC-330). The implementation moved to
///        the `detail/` runtime layer; this shim keeps the historical include path
///        `<adc/runtime/wall_predicate.hpp>` valid. Prefer `<adc/runtime/detail/wall_predicate.hpp>` in new code.

#include <adc/runtime/detail/wall_predicate.hpp>
