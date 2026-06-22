// ADC-335 (P0-B): per-transport block-build seam for the ExB scalar transport. Instantiates ONLY the
// ExBVelocity leaves of the System dispatch product (see block_seam.hpp); compiled in parallel with the
// other transports' translation units. The TR construction matches dispatch_transport's "exb" branch.
#include <adc/runtime/builders/block_seam.hpp>

namespace adc::detail {

BuiltBlock build_block_exb(const ModelSpec& model, const BlockBuildArgs& a) {
  return build_block_for(ExBVelocity{Real(model.B0)}, model, a);
}

}  // namespace adc::detail
