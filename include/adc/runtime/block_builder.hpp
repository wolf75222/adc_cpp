#pragma once

/// @file
/// @brief Compatibility forwarding header (ADC-330). The implementation moved to
///        the `builders/` runtime layer; this shim keeps the historical include path
///        `<adc/runtime/block_builder.hpp>` valid. Prefer `<adc/runtime/builders/block_builder.hpp>` in new code.

#include <adc/runtime/builders/block_builder.hpp>
