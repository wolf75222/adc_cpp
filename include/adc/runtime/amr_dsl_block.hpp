#pragma once

/// @file
/// @brief Compatibility forwarding header (ADC-330). The implementation moved to
///        the `builders/` runtime layer; this shim keeps the historical include path
///        `<adc/runtime/amr_dsl_block.hpp>` valid. Prefer `<adc/runtime/builders/amr_dsl_block.hpp>` in new code.

#include <adc/runtime/builders/amr_dsl_block.hpp>
