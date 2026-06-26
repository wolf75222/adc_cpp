// ADC-359 flux subdivision of the compressible (Euler) single-block compiled AMR seam: only the rusanov
// flux's build_amr_compiled leaves. See amr_compiled_compressible.cpp (the riemann dispatcher).
#include <pops/runtime/builders/block/amr_block_seam.hpp>

namespace pops::detail {

AmrCompiledHooks build_amr_compiled_compressible_rusanov(const ModelSpec& spec,
                                                         const std::string& limiter,
                                                         const AmrBuildParams& bp) {
  return build_amr_compiled_for_flux(CompressibleFlux{Real(spec.gamma)}, spec, limiter, bp,
                                     [](auto m, const std::string& lim, const AmrBuildParams& b) {
                                       return dispatch_amr_compiled_rusanov(m, lim, b);
                                     });
}

}  // namespace pops::detail
