// ADC-335 (P0-B): per-transport block-build seam for the compressible (Euler) transport. Instantiates
// ONLY the CompressibleFlux leaves of the System dispatch product (4 fluxes x 4 limiters x sources x
// integrators -- the heaviest transport) so they compile in their own translation unit. The TR
// construction matches dispatch_transport's "compressible" branch.
#include <adc/runtime/block_seam.hpp>

namespace adc::detail {

BuiltBlock build_block_compressible(const ModelSpec& model, const BlockBuildArgs& a) {
  return build_block_for(CompressibleFlux{Real(model.gamma)}, model, a);
}

}  // namespace adc::detail
