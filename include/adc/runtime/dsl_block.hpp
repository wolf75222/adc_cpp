#pragma once

/// @file
/// @brief Compatibility forwarding header (ADC-330). The implementation moved to
///        the `builders/` runtime layer; this shim keeps the historical include path
///        `<adc/runtime/dsl_block.hpp>` valid. Prefer `<adc/runtime/builders/dsl_block.hpp>` in new code.

#include <adc/runtime/builders/dsl_block.hpp>
