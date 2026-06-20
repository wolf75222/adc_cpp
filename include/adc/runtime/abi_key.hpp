#pragma once

/// @file
/// @brief Compatibility forwarding header (ADC-330). The implementation moved to
///        the `detail/` runtime layer; this shim keeps the historical include path
///        `<adc/runtime/abi_key.hpp>` valid. Prefer `<adc/runtime/detail/abi_key.hpp>` in new code.

#include <adc/runtime/detail/abi_key.hpp>
