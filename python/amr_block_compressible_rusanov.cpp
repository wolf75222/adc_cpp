// ADC-359 flux subdivision of the compressible (Euler) multi-block AMR seam: this TU instantiates ONLY
// the rusanov flux's build_amr_block leaves, so it compiles in parallel with the other flux TUs. See
// amr_block_seam.hpp / amr_block_compressible.cpp (the riemann dispatcher).
#include <adc/runtime/builders/block/amr_block_seam.hpp>

namespace adc::detail {

AmrRuntimeBlock build_amr_block_compressible_rusanov(const AmrBlockBuildArgs& a,
                                                     const SharedAmrLayout& S) {
  return build_amr_block_for_flux(CompressibleFlux{Real(a.spec.gamma)}, a, S,
                                  [](auto m, const AmrBlockBuildArgs& aa, const SharedAmrLayout& SS,
                                     const std::vector<int>& impl, AmrTimeMethod tm) {
                                    return dispatch_amr_block_rusanov(
                                        m, aa.limiter, SS, aa.name, aa.density, aa.has_density,
                                        aa.gamma, aa.substeps, aa.recon_prim, aa.imex, aa.stride,
                                        impl, aa.newton, aa.state, aa.newton_diagnostics, tm,
                                        aa.pos_floor);
                                  });
}

}  // namespace adc::detail
