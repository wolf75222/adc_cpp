#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/fv/reconstruction.hpp>
#include <pops/numerics/spatial_operator.hpp>
#include <pops/numerics/time/integrators/time_steppers.hpp>  // SSPRK2Step (shared scheme)

/// @file
/// @brief Free convenience function advance_ssprk2: one single-level SSP-RK2 step that assembles
///        the residual evaluator (fill_ghosts + assemble_rhs) and delegates to SSPRK2Step.
///
/// Layer: `include/pops/numerics/time`.
/// Role: call sugar. The RK scheme lives in time_steppers.hpp (SSPRK2Step object); this header
///        only provides the residual evaluator and delegates, to avoid duplicating the combination.
/// Contract: aux is assumed fixed during the step.

namespace pops {

template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void advance_ssprk2(const Model& model, MultiFab& U, const MultiFab& aux, const Geometry& geom,
                    const BCRec& bc, Real dt) {
  SSPRK2Step{}.take_step(
      [&](MultiFab& stage, MultiFab& R) {
        fill_ghosts(stage, geom.domain, bc);
        assemble_rhs<Limiter, NumericalFlux>(model, stage, aux, geom, R);
      },
      U, dt);
}

}  // namespace pops
