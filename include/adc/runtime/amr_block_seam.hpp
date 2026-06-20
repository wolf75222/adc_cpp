#pragma once

/// @file
/// @brief Compatibility forwarding header (ADC-330). The implementation moved to
///        the `builders/` runtime layer; this shim keeps the historical include path
///        `<adc/runtime/amr_block_seam.hpp>` valid. Prefer `<adc/runtime/builders/amr_block_seam.hpp>` in new code.

#include <adc/runtime/builders/amr_block_seam.hpp>
