// ADC-335 (P0-B): per-transport block-build seam for the isothermal (3-var fluid) transport.
// Instantiates ONLY the IsothermalFlux leaves of the System dispatch product (rusanov + hll, 5 sources)
// in its own translation unit. The TR construction matches dispatch_transport's "isothermal" branch.
#include <adc/runtime/block_seam.hpp>

namespace adc::detail {

BuiltBlock build_block_isothermal(const ModelSpec& model, const BlockBuildArgs& a) {
  return build_block_for(IsothermalFlux{Real(model.cs2)}, model, a);
}

}  // namespace adc::detail
