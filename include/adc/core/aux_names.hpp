/// @file
/// @brief HOST-ONLY canonical aux name<->component table: the C++ mirror of AUX_CANONICAL
///        (python/adc/dsl.py). Generated from the SAME single source as adc::Aux -- the base
///        contract (phi/grad_x/grad_y, components 0..2) plus the ADC_AUX_FIELDS X-macro
///        (B_z=3, T_e=4) -- so it cannot drift from the device layout. Lets a C++ caller resolve a
///        CANONICAL aux field by name without going through the Python facade (ADC-291), and lets a
///        test pin the C++<->Python coherence. NOT included by device kernels: it uses
///        std::string_view (host-only). The model-NAMED fields (extra[k] = component
///        kAuxNamedBase + k) are intentionally NOT in this table: they carry no canonical meaning
///        and are resolved per block by name on the facade side.

#pragma once

#include <string_view>
#include <utility>

#include <adc/core/state.hpp>

namespace adc {

/// CANONICAL aux name -> component table (mirror of AUX_CANONICAL on the DSL side). The base-contract
/// names are wired explicitly (components 0..2, NOT part of ADC_AUX_FIELDS); the EXTRA fields come
/// from the X-macro (single source). Adding a canonical extra field = 1 line in ADC_AUX_FIELDS, this
/// table follows automatically.
inline constexpr std::pair<std::string_view, int> kAuxCanonicalNames[] = {
    {"phi", 0},
    {"grad_x", 1},
    {"grad_y", 2},
#define ADC_AUX_NAME_ENTRY(name, idx) {#name, idx},
    ADC_AUX_FIELDS(ADC_AUX_NAME_ENTRY)
#undef ADC_AUX_NAME_ENTRY
};

/// Component of the CANONICAL aux field @p name, or -1 if @p name is not a canonical field (it may
/// then be a model-NAMED field, resolved per block by the facade). HOST-only constexpr.
constexpr int aux_canonical_index(std::string_view name) {
  for (const auto& [n, c] : kAuxCanonicalNames)
    if (n == name)
      return c;
  return -1;
}

/// Inverse: CANONICAL name of component @p comp, or an empty view if @p comp is not a canonical
/// component (e.g. a model-named field at kAuxNamedBase + k). HOST-only constexpr.
constexpr std::string_view aux_canonical_name(int comp) {
  for (const auto& [n, c] : kAuxCanonicalNames)
    if (c == comp)
      return n;
  return {};
}

}  // namespace adc
