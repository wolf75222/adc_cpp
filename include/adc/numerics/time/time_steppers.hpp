#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/mf_arith.hpp>  // saxpy, lincomb
#include <adc/mesh/multifab.hpp>

#include <utility>

/// @file
/// @brief Time integrators as first-class OBJECTS with a take_step method: TimeStepper
///        concept, and the ForwardEuler, SSPRK2Step, SSPRK3Step (Shu-Osher) schemes.
///
/// Layer: `include/adc/numerics/time`.
/// Role: keep the mathematical scheme alive in the core, with the coupler CALLING it instead
///        of inlining SSPRK everywhere. The "give a TimeIntegrator to the coupler like you give
///        a PhysicalModel" contract: the user can provide their own (same signature
///        take_step(rhs_eval, U, dt)).
/// Contract: an integrator is agnostic of the model and the discretization -- it only sees
///           rhs_eval(U_stage, R) (the method-of-lines arrow R = -div F + S) and the MultiFab
///           operations (saxpy / lincomb).
///
/// Invariants:
/// - no carried state: the scratch (R, stages U1/U2/U3) is allocated from the layout of U;
/// - SSPRK2Step / SSPRK3Step exactly reproduce the old copies in SystemCoupler and
///   ssprk.hpp (bit-identical).

namespace adc {

// Contract: an integrator knows how to advance U by dt via a residual evaluator.
template <class I>
concept TimeStepper = requires(const I integ, MultiFab& U, Real dt) {
  integ.take_step([](MultiFab&, MultiFab&) {}, U, dt);
};

// Forward Euler (order 1): U <- U + dt R(U).
struct ForwardEuler {
  template <class RhsEval>
  void take_step(RhsEval&& rhs, MultiFab& U, Real dt) const {
    MultiFab R(U.box_array(), U.dmap(), U.ncomp(), 0);
    rhs(U, R);
    saxpy(U, dt, R);
  }
};

// SSP-RK2 (Shu-Osher, 2 stages, order 2). Same operations as the old
// SystemCoupler::advance_explicit_ssprk2 / ssprk.hpp::advance_ssprk2 (bit-identical).
struct SSPRK2Step {
  template <class RhsEval>
  void take_step(RhsEval&& rhs, MultiFab& U, Real dt) const {
    MultiFab R(U.box_array(), U.dmap(), U.ncomp(), 0);
    rhs(U, R);
    MultiFab U1 = U;
    saxpy(U1, dt, R);
    rhs(U1, R);
    saxpy(U1, dt, R);
    lincomb(U, Real(0.5), U, Real(0.5), U1);
  }
};

// SSP-RK3 (Shu-Osher, 3 stages, order 3). Same operations as the old
// SystemCoupler::advance_explicit_ssprk3 (bit-identical).
struct SSPRK3Step {
  template <class RhsEval>
  void take_step(RhsEval&& rhs, MultiFab& U, Real dt) const {
    MultiFab R(U.box_array(), U.dmap(), U.ncomp(), 0);
    rhs(U, R);
    MultiFab U1 = U;
    saxpy(U1, dt, R);

    rhs(U1, R);
    MultiFab U2 = U1;
    saxpy(U2, dt, R);
    lincomb(U2, Real(3) / 4, U, Real(1) / 4, U2);

    rhs(U2, R);
    MultiFab U3 = U2;
    saxpy(U3, dt, R);
    lincomb(U, Real(1) / 3, U, Real(2) / 3, U3);
  }
};

}  // namespace adc
