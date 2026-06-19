// ADC-335 (P0-B): single-block AMR seam (AmrCouplerMP) for the compressible (Euler) transport. See
// amr_block_seam.hpp.
#include <adc/runtime/amr_block_seam.hpp>

namespace adc::detail {

AmrCompiledHooks build_amr_compiled_compressible(const ModelSpec& spec, const std::string& limiter,
                                                 const std::string& riemann, const AmrBuildParams& bp) {
  return build_amr_compiled_for(CompressibleFlux{Real(spec.gamma)}, spec, limiter, riemann, bp);
}

}  // namespace adc::detail
