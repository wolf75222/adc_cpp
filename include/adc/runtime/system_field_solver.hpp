#pragma once

/// @file
/// @brief Compatibility forwarding header (ADC-330). The implementation moved to
///        the `system/` runtime layer; this shim keeps the historical include path
///        `<adc/runtime/system_field_solver.hpp>` valid. Prefer `<adc/runtime/system/system_field_solver.hpp>` in new code.

#include <adc/runtime/system/system_field_solver.hpp>
