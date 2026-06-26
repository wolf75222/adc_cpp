# Coupling surface: PUBLIC, INTERNAL, DEPRECATED

This file is the source of truth for classifying the classes in
`include/pops/coupling/`. It answers the question: "Which class is
public, internal or deprecated?"

---

## User entry point (PUBLIC Python)

The user entry point is NOT a C++ coupling class: it is the Python pair
`pops.System` / `pops.AmrSystem` plus the DSL (`pops.dsl.HyperbolicModel`,
`m.compile(...)`). The classes below are internal C++ facades; they
are not part of the documented public API.

---

## Internal C++ facades (INTERNAL)

### `coupler.hpp` -- `pops::Coupler<Model, Elliptic>`

Single-model, single-level facade: closes the Poisson -> aux -> advance loop
for a single `PhysicalModel` on one uniform level. Used in the unit
tests and single-species tutorials; not the user entry point.
**Classification: INTERNAL**

### `system_coupler.hpp` -- `pops::SystemAssembler`, `pops::SystemDriver`, alias `pops::SystemCoupler`

`SystemAssembler` assembles the multi-species RHS (system Poisson, aux, per-block
residuals) without taking a time step. `SystemDriver` advances (per-species
subcycling, IMEX). `SystemCoupler` is the compat alias to `SystemDriver`.
Carried by `pops.System` on the Python side; internal to the core C++.
**Classification: INTERNAL**

### `amr_coupler_mp.hpp` -- `pops::AmrCouplerMP<Model, Elliptic>`

Multi-patch ExB AMR coupler: multi-box hierarchy per level (engine
`advance_amr`, coverage-aware reflux, Berger-Rigoutsos regrid). Single-model
AMR production path (the old single-box `AmrCoupler` was removed, #164).
**Classification: INTERNAL**

### `amr_system_coupler.hpp` -- `pops::AmrSystemCoupler<System, RhsAssembler, Elliptic>`, alias `pops::AmrSystemDriver`

System coupler over AMR: carries a `CoupledSystem` on an AMR hierarchy
shared by all species. Wired through `pops.AmrSystem` Python (#92/#105).
`AmrSystemDriver` is the "advancing" alias (advisor feedback ss8.2 B).
**Classification: INTERNAL**

### `coupling_policy.hpp` -- `pops::PerStageCoupling`, `pops::OncePerStepCoupling`

Tag types for the hyperbolic-elliptic temporal coupling policy
(elliptic solve frequency). Selected at the call site by template.
**Classification: INTERNAL**

### `elliptic_rhs.hpp` -- `pops::SingleModelEllipticRhs`, `pops::SystemEllipticRhs`

Elliptic right-hand side assembly: isolates the responsibility of the `Coupler`.
`SingleModelEllipticRhs` for a single model, `SystemEllipticRhs` for several
species.
**Classification: INTERNAL**

### `coupled_source.hpp` -- concept `pops::CoupledSourceFor`

C++20 concept defining the contract of an inter-species coupling source
(collisions, thermal exchange). The concrete sources live in `adc_cases`.
**Classification: INTERNAL**

### `coupled_source_program.hpp` -- bytecode `CoupledSourceProgram`

Generic coupled-source evaluator by postfix bytecode (device-clean POD).
Allows running symbolic Python sources (`pops.dsl.CoupledSource`) inside
a `for_each_cell` device without a per-cell Python callback.
**Classification: INTERNAL**

### `aux_fill.hpp` -- `pops::detail::derive_aux_bc`, `pops::detail::fill_bz_box`

Helpers shared by the three couplers (Coupler, SystemAssembler,
AmrSystemCoupler) for the aux channel (boundary conditions of phi -> aux,
populating B_z). Pure extraction, bit-identical bodies.
**Classification: INTERNAL**

### `amr_level_storage.hpp` -- `pops::AmrLevelStack<Level>`

Storage of the AMR hierarchy (level stack + aux). Extracted from the AMR
couplers to separate storage and orchestration.
**Classification: INTERNAL**

### `amr_diagnostics.hpp` -- `pops::amr_mass`, `pops::amr_max_drift_speed`

Diagnostics extracted from the AMR couplers (integrated mass, max drift speed).
Namespace free functions, GPU seam.
**Classification: INTERNAL**

### `amr_regrid_coupler.hpp` -- `pops::amr_regrid_finest`

Berger-Rigoutsos regrid extracted from `AmrCouplerMP` (responsibility b). Rebuilds
the fine level on the fly from a refinement criterion provided by the caller.
**Classification: INTERNAL**

### `schur_condensation.hpp` -- `pops::ElectrostaticLorentzCondensation`

Builder of the Schur-condensed source stage (potential/velocity/
Lorentz coupling, Hoffart et al. arXiv:2510.11808). Assembles the tensor
elliptic operator A_op and the right-hand side; does not solve.
**Classification: INTERNAL**

### `condensed_schur_source_stepper.hpp` -- `pops::CondensedSchurSourceStepper`

Full Schur-condensed source stage: composes `ElectrostaticLorentzCondensation`
and `TensorKrylovSolver`. Standalone source stage (frozen transport); facade
wiring to `System::step` planned in PR5.
**Classification: INTERNAL**

---

## Deprecated class (DEPRECATED)

> Note: the old `amr_coupler.hpp` / `pops::AmrCoupler<Model, Elliptic>`
> (single-box ExB AMR coupler) was **removed (#164)**. Its role is entirely
> taken over by `AmrCouplerMP` (`amr_coupler_mp.hpp`), whose single-box is the
> bit-identical degenerate case.

### `spectral_coupler.hpp` -- `pops::SpectralCoupler<Model>`

**DEPRECATED.** Distributed periodic coupler (slab-decomposed FFT). No `#include`
in the core, the tests or the Python bindings. The role is taken over by
`Coupler<Model, DistributedFFTSolver>` (the distributed FFT solver became a
standalone `EllipticSolver`, cf. `poisson_fft_solver.hpp`). Kept for
historical reference (documented in `docs/ARCHITECTURE.md`).
**Recommended replacement: `pops::Coupler<Model, DistributedFFTSolver>` (`coupler.hpp`)**

---

*Last updated: 2026-06-06 (P0.3 of CODEBASE_AUDIT.md).*
