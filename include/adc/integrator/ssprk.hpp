#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/operator/reconstruction.hpp>
#include <adc/operator/spatial_operator.hpp>

// Integrateurs SSP Runge-Kutta mono-niveau. L'integrateur est agnostique du
// modele et de la discretisation : il ne connait que assemble_rhs (la fleche
// methode-des-lignes). Les ghosts sont remplis a chaque etage.
//
// aux est suppose fixe pendant le pas (ses ghosts deja remplis par l'appelant).
// Le couplage stade par stade avec la resolution elliptique viendra avec le
// coupleur.

namespace adc {

template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void advance_ssprk2(const Model& model, MultiFab& U, const MultiFab& aux,
                    const Geometry& geom, const BCRec& bc, Real dt) {
  MultiFab R(U.box_array(), U.dmap(), U.ncomp(), 0);

  fill_ghosts(U, geom.domain, bc);
  assemble_rhs<Limiter, NumericalFlux>(model, U, aux, geom, R);
  MultiFab U1 = U;
  saxpy(U1, dt, R);  // U1 = U + dt R(U)

  fill_ghosts(U1, geom.domain, bc);
  assemble_rhs<Limiter, NumericalFlux>(model, U1, aux, geom, R);
  saxpy(U1, dt, R);                       // U1 = U + dt R(U) + dt R(U1)
  lincomb(U, Real(0.5), U, Real(0.5), U1);  // U = 1/2 U + 1/2 U1
}

}  // namespace adc
