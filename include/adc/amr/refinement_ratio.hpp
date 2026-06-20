#pragma once

#include <stdexcept>
#include <string>

/// @file
/// @brief Single source of truth for the native AMR refinement ratio.
///
/// Layer: `include/adc/amr`.
/// Role: the in-house (non-SAMRAI) AMR hierarchy refines and coarsens by a
/// FIXED integer ratio between consecutive levels. That ratio is 2, and several
/// coarse/fine paths (aux/density injection, reflux, Berger-Oliger subcycling,
/// the condensed-Schur composite FAC) are written for ratio 2. Rather than
/// scatter the literal `2` across those files, the ratio lives here as one named
/// constant, and `require_supported_ref_ratio` rejects any other value at the
/// hierarchy boundary. A future non-2 ratio therefore has a single entry point
/// instead of a literal hunt: bump `kAmrRefRatio`, then the `static_assert`s
/// guarding the ratio-2-structural kernels (e.g. the 2x2 average-down unroll in
/// `numerics/time/amr_reflux.hpp`) fail loudly at exactly the sites that need
/// generalizing.
///
/// Scope note: this is the AMR INTER-LEVEL ratio. The geometric-multigrid
/// V-cycle coarsens its OWN domain by 2 internally; that is the linear solver's
/// coarsening factor, a separate concern, and is deliberately NOT governed by
/// this constant.
namespace adc {

/// The native AMR refinement ratio between two consecutive levels. Only the
/// value 2 is supported today; see `require_supported_ref_ratio`.
inline constexpr int kAmrRefRatio = 2;

/// Validates a requested AMR refinement ratio at the hierarchy boundary.
/// @param ratio refinement ratio requested for an AMR hierarchy.
/// @throws std::invalid_argument if @p ratio is not the supported value.
inline void require_supported_ref_ratio(int ratio) {
  if (ratio != kAmrRefRatio) {
    throw std::invalid_argument("adc: AMR refinement ratio " + std::to_string(ratio) +
                                " is not supported; the native AMR hierarchy only supports ratio " +
                                std::to_string(kAmrRefRatio) +
                                " (centralized in include/adc/amr/refinement_ratio.hpp)");
  }
}

}  // namespace adc
