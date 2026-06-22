// ADC-335 (P0-B): single-block AMR seam (AmrCouplerMP) for the ExB scalar transport. Instantiates only
// the ExBVelocity build_amr_compiled leaves. See amr_block_seam.hpp.
#include <adc/runtime/builders/amr_block_seam.hpp>

namespace adc::detail {

AmrCompiledHooks build_amr_compiled_exb(const ModelSpec& spec, const std::string& limiter,
                                        const std::string& riemann, const AmrBuildParams& bp) {
  return build_amr_compiled_for(ExBVelocity{Real(spec.B0)}, spec, limiter, riemann, bp);
}

}  // namespace adc::detail
