// ADC-342 (P0-B flux subdivision): isothermal x HLL flux only. make_block_hll forwards wave_speed_cache
// (the only flux that engages it). See system_isothermal_rusanov.cpp.
#include <adc/runtime/builders/block/block_seam.hpp>

namespace adc::detail {

BuiltBlock build_block_isothermal_hll(const ModelSpec& model, const BlockBuildArgs& a) {
  return build_block_for_make(IsothermalFlux{Real(model.cs2), Real(model.vacuum_floor)}, model, a,
                              [](auto m, const std::vector<int>& impl, const BlockBuildArgs& aa) {
                                return make_block_hll(m, aa.limiter, aa.ctx, aa.imex, aa.recon_prim,
                                                      aa.method, impl, aa.nopts, aa.nreport,
                                                      aa.positivity_floor, aa.wave_speed_cache);
                              });
}

}  // namespace adc::detail
