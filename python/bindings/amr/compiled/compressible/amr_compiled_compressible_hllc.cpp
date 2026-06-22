// ADC-359 flux subdivision of the compressible (Euler) single-block compiled AMR seam: only the hllc
// flux's build_amr_compiled leaves. See amr_compiled_compressible.cpp (the riemann dispatcher).
#include <adc/runtime/builders/block/amr_block_seam.hpp>

namespace adc::detail {

AmrCompiledHooks build_amr_compiled_compressible_hllc(const ModelSpec& spec,
                                                      const std::string& limiter,
                                                      const AmrBuildParams& bp) {
  return build_amr_compiled_for_flux(CompressibleFlux{Real(spec.gamma)}, spec, limiter, bp,
                                     [](auto m, const std::string& lim, const AmrBuildParams& b) {
                                       return dispatch_amr_compiled_hllc(m, lim, b);
                                     });
}

}  // namespace adc::detail
