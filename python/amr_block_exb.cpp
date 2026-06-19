// ADC-335 (P0-B): multi-block AMR seam for the ExB scalar transport. Instantiates only the ExBVelocity
// build_amr_block leaves; compiled in parallel with the other transports. See amr_block_seam.hpp.
#include <adc/runtime/amr_block_seam.hpp>

namespace adc::detail {

AmrRuntimeBlock build_amr_block_exb(const AmrBlockBuildArgs& a, const SharedAmrLayout& S) {
  return build_amr_block_for(ExBVelocity{Real(a.spec.B0)}, a, S);
}

}  // namespace adc::detail
