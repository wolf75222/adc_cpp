// ADC-342 (P0-B flux subdivision, follow-up to ADC-335): the isothermal (3-var fluid) transport carries
// its two reachable fluxes (rusanov + hll) x 4 limiters x 15 models -- the post-split long pole TU -- so it
// is split one .cpp per flux like compressible. This TU instantiates ONLY the RusanovFlux build_block
// leaves of the isothermal models (via make_block_rusanov), so they compile in parallel with the hll TU.
// The flux is dispatched by System (riemann string); validation lives there (shared validate_*).
#include <adc/runtime/builders/block/block_seam.hpp>

namespace adc::detail {

BuiltBlock build_block_isothermal_rusanov(const ModelSpec& model, const BlockBuildArgs& a) {
  return build_block_for_make(IsothermalFlux{Real(model.cs2), Real(model.vacuum_floor)}, model, a,
                              [](auto m, const std::vector<int>& impl, const BlockBuildArgs& aa) {
                                return make_block_rusanov(m, aa.limiter, aa.ctx, aa.imex,
                                                          aa.recon_prim, aa.method, impl, aa.nopts,
                                                          aa.nreport, aa.positivity_floor);
                              });
}

}  // namespace adc::detail
