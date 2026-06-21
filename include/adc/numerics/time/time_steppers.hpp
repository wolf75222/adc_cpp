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
/// - no carried state: the scratch (R, stages U1/U2/U3) is sized from the layout of U. The
///   one-arg take_step allocates it per call; the scratch-taking overload reuses a caller-owned
///   buffer (see run_explicit_substeps) to hoist the allocation out of a substep loop -- both
///   paths are bit-identical;
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
  // Reusable residual buffer, sized from a U layout. Hoist it out of a substep loop (see
  // run_explicit_substeps) to avoid re-allocating per substep; rhs() overwrites R every call,
  // so reuse is bit-identical.
  struct Scratch {
    MultiFab R;
    explicit Scratch(const MultiFab& U) : R(U.box_array(), U.dmap(), U.ncomp(), 0) {}
  };
  template <class RhsEval>
  void take_step(RhsEval&& rhs, MultiFab& U, Real dt, Scratch& s) const {
    rhs(U, s.R);
    saxpy(U, dt, s.R);
  }
  template <class RhsEval>
  void take_step(RhsEval&& rhs, MultiFab& U, Real dt) const {
    Scratch s(U);
    take_step(std::forward<RhsEval>(rhs), U, dt, s);
  }
};

// SSP-RK2 (Shu-Osher, 2 stages, order 2). Same operations as the old
// SystemCoupler::advance_explicit_ssprk2 / ssprk.hpp::advance_ssprk2 (bit-identical).
struct SSPRK2Step {
  // Reusable stage buffers (residual R with 0 ghosts; stage U1 with U's ghosts, since rhs reads
  // its ghosts). Hoist out of a substep loop via run_explicit_substeps. Reuse is bit-identical:
  // R is overwritten by rhs and U1's valid cells are overwritten by the lincomb copy each substep,
  // while U1's ghosts are re-derived by rhs's internal fill_ghosts before any ghost read.
  struct Scratch {
    MultiFab R, U1;
    explicit Scratch(const MultiFab& U)
        : R(U.box_array(), U.dmap(), U.ncomp(), 0),
          U1(U.box_array(), U.dmap(), U.ncomp(), U.n_grow()) {}
  };
  template <class RhsEval>
  void take_step(RhsEval&& rhs, MultiFab& U, Real dt, Scratch& s) const {
    rhs(U, s.R);
    lincomb(s.U1, Real(1), U, Real(0), U);  // U1 = U (valid cells; ghosts re-derived by rhs)
    saxpy(s.U1, dt, s.R);
    rhs(s.U1, s.R);
    saxpy(s.U1, dt, s.R);
    lincomb(U, Real(0.5), U, Real(0.5), s.U1);
  }
  template <class RhsEval>
  void take_step(RhsEval&& rhs, MultiFab& U, Real dt) const {
    Scratch s(U);
    take_step(std::forward<RhsEval>(rhs), U, dt, s);
  }
};

// SSP-RK3 (Shu-Osher, 3 stages, order 3). Same operations as the old
// SystemCoupler::advance_explicit_ssprk3 (bit-identical).
struct SSPRK3Step {
  // Reusable stage buffers (residual R with 0 ghosts; stages U1/U2/U3 with U's ghosts, since the
  // first two are passed back to rhs which reads their ghosts). Hoist out of a substep loop via
  // run_explicit_substeps. Reuse is bit-identical: every buffer is fully overwritten each substep
  // (R by rhs, U1/U2/U3 valid cells by the lincomb copy), and stage ghosts are re-derived by rhs.
  struct Scratch {
    MultiFab R, U1, U2, U3;
    explicit Scratch(const MultiFab& U)
        : R(U.box_array(), U.dmap(), U.ncomp(), 0),
          U1(U.box_array(), U.dmap(), U.ncomp(), U.n_grow()),
          U2(U.box_array(), U.dmap(), U.ncomp(), U.n_grow()),
          U3(U.box_array(), U.dmap(), U.ncomp(), U.n_grow()) {}
  };
  template <class RhsEval>
  void take_step(RhsEval&& rhs, MultiFab& U, Real dt, Scratch& s) const {
    rhs(U, s.R);
    lincomb(s.U1, Real(1), U, Real(0), U);  // U1 = U
    saxpy(s.U1, dt, s.R);

    rhs(s.U1, s.R);
    lincomb(s.U2, Real(1), s.U1, Real(0), s.U1);  // U2 = U1
    saxpy(s.U2, dt, s.R);
    lincomb(s.U2, Real(3) / 4, U, Real(1) / 4, s.U2);

    rhs(s.U2, s.R);
    lincomb(s.U3, Real(1), s.U2, Real(0), s.U2);  // U3 = U2
    saxpy(s.U3, dt, s.R);
    lincomb(U, Real(1) / 3, U, Real(2) / 3, s.U3);
  }
  template <class RhsEval>
  void take_step(RhsEval&& rhs, MultiFab& U, Real dt) const {
    Scratch s(U);
    take_step(std::forward<RhsEval>(rhs), U, dt, s);
  }
};

// Runs @p n explicit RK substeps of @c Stepper on @p U with step @p h. When the stepper exposes a
// reusable Scratch (the built-in ForwardEuler / SSPRK2Step / SSPRK3Step), the scratch is allocated
// ONCE here and reused across substeps via the scratch-taking take_step overload -- removing the
// per-substep alloc/zero/free churn (ADC-261). A custom TimeStepper without a Scratch falls back to
// the one-shot take_step. The result is bit-identical either way (same saxpy/lincomb sequence on
// freshly-overwritten buffers); the substep loop never changes U's layout, so one Scratch suffices.
template <class Stepper, class RhsEval>
inline void run_explicit_substeps(RhsEval&& rhs, MultiFab& U, Real h, int n) {
  // Probe the actual scratch-taking overload (not just a nested type named Scratch): a custom
  // TimeStepper that happens to expose an unrelated Scratch type but no four-arg take_step still
  // takes the one-shot fallback instead of hitting a hard error.
  if constexpr (requires(Stepper st, RhsEval r, MultiFab& u, Real dt,
                         typename Stepper::Scratch& sc) { st.take_step(r, u, dt, sc); }) {
    typename Stepper::Scratch scratch(U);
    for (int s = 0; s < n; ++s)
      Stepper{}.take_step(rhs, U, h, scratch);
  } else {
    for (int s = 0; s < n; ++s)
      Stepper{}.take_step(rhs, U, h);
  }
}

}  // namespace adc
