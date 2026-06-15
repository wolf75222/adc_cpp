#pragma once

#include <adc/core/types.hpp>  // Real, ADC_HD (device-callable)

#include <cstddef>

/// @file
/// @brief RuntimeParams: carrier for the RUNTIME PARAMETERS of a DSL model (P7-b). A runtime parameter
///        is declared on the Python side via adc.dsl.Param(..., kind="runtime"); its value can be CHANGED
///        at run time WITHOUT recompiling the .so. CONST parameters (kind="const") stay inlined HARD into
///        the .so at codegen (bit-identical to history: this work does NOT touch their path).
///
/// MECHANICS. The codegen (python/adc/dsl.py) assigns each runtime parameter a STABLE INDEX
/// (sorted order of names) and emits, wherever the formula reads that parameter, `params.get(<index>)`
/// instead of a literal constant. Each generated block (hyperbolic / source / elliptic) that READS at
/// least one runtime parameter then carries a member `adc::RuntimeParams params{}` initialized to the
/// DECLARATION value (so, without a runtime set call, the block behaves EXACTLY as with a const param:
/// the declaration value is baked into the member default). At run time, the ABI of the AOT .so
/// (compiled_block_abi.hpp) transports a flat block of doubles; each call rebuilds the model then
/// OVERWRITES `params.values[k]` with the supplied value -> the behavior changes without recompilation.
///
/// DEVICE-CLEAN. RuntimeParams is a trivially copyable aggregate (array of Real by value, no allocation,
/// no std::), so copyable on device and readable in a kernel: get() is ADC_HD. The size is FIXED
/// (kMaxRuntimeParams) to stay allocation-free; a model exceeding this bound is rejected on the Python
/// side (codegen), never here.

namespace adc {

/// Maximum number of runtime parameters per DSL block. Bound deliberately large (a reasonable physical
/// model has a few); overflow is diagnosed on the Python side at codegen. Keeps the structure
/// fixed-size (no allocation -> device-copyable by value).
inline constexpr int kMaxRuntimeParams = 32;

/// FLAT carrier (fixed size, by value) of the runtime parameter values of a block. `count` = number of
/// parameters actually declared by the model; `values[k]` = current value of the parameter at index k
/// (index assigned by the codegen, sorted order of names). Indices >= count are zero and never read by
/// the generated blocks. Trivial aggregate: copyable on device at no cost.
struct RuntimeParams {
  int count = 0;
  Real values[kMaxRuntimeParams] = {};

  /// Value of the runtime parameter at index @p k. No dynamic bound (the index is emitted by the
  /// codegen, thus statically < count): direct read, device-callable.
  ADC_HD Real get(int k) const { return values[k]; }
};

}  // namespace adc

// (P7-b) DSL runtime params: see test_dsl_runtime_params.py
