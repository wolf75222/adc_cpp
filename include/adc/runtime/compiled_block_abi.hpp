#pragma once

/// @file
/// @brief Compatibility forwarding header (ADC-330). The implementation moved to
///        the `builders/` runtime layer; this shim keeps the historical include path
///        `<adc/runtime/compiled_block_abi.hpp>` valid. Prefer `<adc/runtime/builders/compiled_block_abi.hpp>` in new code.

#include <adc/runtime/builders/compiled_block_abi.hpp>
