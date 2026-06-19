// ADC-335 (P0-B): multi-block AMR seam for the compressible (Euler 4-var) transport -- the heaviest AMR
// leaf (all fluxes + the SourceFreeModel IMEX doubling). See amr_block_seam.hpp.
#include <adc/runtime/amr_block_seam.hpp>

namespace adc::detail {

AmrRuntimeBlock build_amr_block_compressible(const AmrBlockBuildArgs& a, const SharedAmrLayout& S) {
  return build_amr_block_for(CompressibleFlux{Real(a.spec.gamma)}, a, S);
}

}  // namespace adc::detail
