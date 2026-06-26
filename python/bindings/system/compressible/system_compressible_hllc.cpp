// ADC-335 (P0-B flux subdivision): compressible (Euler) x HLLC flux only (contact-resolving algebra,
// one of the two heaviest leaves). See system_compressible_rusanov.cpp.
#include <pops/runtime/builders/block/block_seam.hpp>

namespace pops::detail {

BuiltBlock build_block_compressible_hllc(const ModelSpec& model, const BlockBuildArgs& a) {
  return build_block_for_make(CompressibleFlux{Real(model.gamma)}, model, a,
                              [](auto m, const std::vector<int>& impl, const BlockBuildArgs& aa) {
                                return make_block_hllc(m, aa.limiter, aa.ctx, aa.imex,
                                                       aa.recon_prim, aa.method, impl, aa.nopts,
                                                       aa.nreport, aa.positivity_floor);
                              });
}

}  // namespace pops::detail
