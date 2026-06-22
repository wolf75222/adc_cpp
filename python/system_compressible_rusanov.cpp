// ADC-335 (P0-B flux subdivision): the compressible (Euler) transport carries all four fluxes and is the
// heaviest TU, so it is split one .cpp per flux. This TU instantiates ONLY the RusanovFlux build_block
// leaves of the compressible models (via make_block_rusanov), so they compile in parallel with the other
// flux TUs. The flux is dispatched by System (riemann string); validation lives there (shared validate_*).
#include <adc/runtime/builders/block/block_seam.hpp>

namespace adc::detail {

BuiltBlock build_block_compressible_rusanov(const ModelSpec& model, const BlockBuildArgs& a) {
  return build_block_for_make(CompressibleFlux{Real(model.gamma)}, model, a,
                              [](auto m, const std::vector<int>& impl, const BlockBuildArgs& aa) {
                                return make_block_rusanov(m, aa.limiter, aa.ctx, aa.imex,
                                                          aa.recon_prim, aa.method, impl, aa.nopts,
                                                          aa.nreport, aa.positivity_floor);
                              });
}

}  // namespace adc::detail
