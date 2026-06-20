#pragma once

/// @file
/// @brief Compatibility forwarding header (ADC-330). The implementation moved to
///        the `builders/` runtime layer; this shim keeps the historical include path
///        `<adc/runtime/native_loader.hpp>` valid. Prefer `<adc/runtime/builders/native_loader.hpp>` in new code.

#include <adc/runtime/builders/native_loader.hpp>
