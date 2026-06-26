// ADC-335 (P0-B): single-block AMR seam (AmrCouplerMP) for the isothermal transport. See amr_block_seam.hpp.
#include <pops/runtime/builders/block/amr_block_seam.hpp>

namespace pops::detail {

AmrCompiledHooks build_amr_compiled_isothermal(const ModelSpec& spec, const std::string& limiter,
                                               const std::string& riemann,
                                               const AmrBuildParams& bp) {
  return build_amr_compiled_for(IsothermalFlux{Real(spec.cs2), Real(spec.vacuum_floor)}, spec,
                                limiter, riemann, bp);
}

}  // namespace pops::detail
