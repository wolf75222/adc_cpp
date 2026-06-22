#pragma once

#include <adc/core/foundation/types.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/reconstruction.hpp>
#include <adc/numerics/spatial_operator.hpp>
#include <adc/numerics/time/time_steppers.hpp>  // SSPRK2Step (shared scheme)

/// @file
/// @brief Free convenience function advance_ssprk2: one single-level SSP-RK2 step that assembles
///        the residual evaluator (fill_ghosts + assemble_rhs) and delegates to SSPRK2Step.
///
/// Layer: `include/adc/numerics/time`.
/// Role: call sugar. The RK scheme lives in time_steppers.hpp (SSPRK2Step object); this header
///        only provides the residual evaluator and delegates, to avoid duplicating the combination.
/// Contract: aux is assumed fixed during the step.

namespace adc {

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

}  // namespace adc
