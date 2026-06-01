#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/operator/reconstruction.hpp>
#include <adc/operator/spatial_operator.hpp>
#include <adc/integrator/time_steppers.hpp>  // SSPRK2Step (schema partage)

// Integrateur SSP-RK2 mono-niveau, fonction libre de commodite. Le SCHEMA vit
// desormais dans integrator/time_steppers.hpp (objet SSPRK2Step) ; ici on ne fait
// que fournir l'evaluateur de residu (fill_ghosts + assemble_rhs) et deleguer, pour
// ne pas dupliquer la combinaison RK. aux est suppose fixe pendant le pas.

namespace adc {

template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void advance_ssprk2(const Model& model, MultiFab& U, const MultiFab& aux,
                    const Geometry& geom, const BCRec& bc, Real dt) {
  SSPRK2Step{}.take_step(
      [&](MultiFab& stage, MultiFab& R) {
        fill_ghosts(stage, geom.domain, bc);
        assemble_rhs<Limiter, NumericalFlux>(model, stage, aux, geom, R);
      },
      U, dt);
}

}  // namespace adc
