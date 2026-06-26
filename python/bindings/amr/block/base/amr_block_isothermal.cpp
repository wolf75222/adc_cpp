// ADC-335 (P0-B): multi-block AMR seam for the isothermal (3-var fluid) transport. See amr_block_seam.hpp.
#include <pops/runtime/builders/block/amr_block_seam.hpp>

namespace pops::detail {

AmrRuntimeBlock build_amr_block_isothermal(const AmrBlockBuildArgs& a, const SharedAmrLayout& S) {
  return build_amr_block_for(IsothermalFlux{Real(a.spec.cs2), Real(a.spec.vacuum_floor)}, a, S);
}

}  // namespace pops::detail
