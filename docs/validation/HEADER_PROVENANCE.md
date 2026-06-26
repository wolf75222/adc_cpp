# Header provenance (relocated by ADC-333)

The generic solver core under `include/pops/**` keeps the mathematical contract,
invariants, preconditions, and maintainer warnings. The case/paper/hardware
narration that used to color those headers (the diocotron reproduction of the
reference paper, the ROMEO/GH200 runs, cross-repo `adc_cases` tickets) lives here
and in the validation docs linked below, so it stays discoverable without making
the core read like the implementation of a single scenario.

Reference scenario: Hoffart, Maier, Shadid, Tomas, "structure-preserving FE for
magnetic Euler-Poisson" (arXiv:2510.11808, Sec 5.3 diocotron test). Full
bibliography: `docs/BIBLIOGRAPHY.md`.

## Topic map

| Generalized header phrase | Detail lives in |
| --- | --- |
| Schur-condensed source coupling (potential / velocity / Lorentz) | `docs/SCHUR_CONDENSATION_DESIGN.md` (+ arXiv:2510.11808) |
| Disc domain / "Cartesian-ring-edge lock" | `docs/HOFFART_GEOMETRY_VERDICT.md`, `docs/HOFFART_FIDELITY.md` |
| Diocotron growth-mode reconstruction (VanLeer vs Minmod, EB scheme) | `docs/archive/DIOCOTRON_GROWTH_RATE.md` |
| Polar phi_bc azimuthal-seam drift | `docs/HOFFART_GEOMETRY_VERDICT.md` (ROUTE 1, frozen equilibrium) |
| Positivity average-fallback (Zhang-Shu) | "Positivity fallback" below |
| Device codegen (named-functor nvcc fixes) | "Device codegen" below |

## Positivity fallback (Zhang-Shu average fallback)

`numerics/spatial_operator.hpp::zhang_shu_scale` replaces a sub-floor reconstructed
face state by its source-cell average (local order-1 fallback) instead of the
colinear theta-scaling of Zhang & Shu (JCP 2010). Why the average and not the
theta-scaling: in conservative variables at a quasi-vacuum edge (a ~1e-6 background
under a ~1e6 top-hat contrast), the theta-scaling sets rho_face = floor while
leaving a face momentum O(average), so the face velocity v = m/rho diverges (~1e6)
and the Rusanov wave speed blows past the dt chosen on the cell velocities.
Measured symptom: NaN within a couple of steps, independent of the floor value. The
average fallback bounds the face velocity by construction (v_face = v_cell), stays
conservative, and degrades the order only on the offending faces.

Provenance: diagnosed on the diocotron reproduction; tracked in `adc_cases`
ADC-62 / ADC-74, ticket ADC-76. The generic positivity blow-up at the 1e6 ring
edge is also discussed in `docs/HOFFART_GEOMETRY_VERDICT.md`.

## Polar phi_bc azimuthal-seam drift

`coupling/polar_condensed_schur_source_stepper.hpp::phi_bc` and
`numerics/elliptic/polar_tensor_operator.hpp::force_theta_periodic` force the
azimuthal (theta) boundary periodic with zero seam values. If the phi ghosts are
filled with a raw Dirichlet-in-y BC instead, the theta = 0 / 2pi seam fills by odd
reflection (ghost = -phi) rather than the periodic wrap; the centered azimuthal
gradient there reads an error ~2 phi / (2 r dtheta), an anti-symmetric
radial-momentum dipole at the two seam columns. Measured drift ||R_eq||_inf ~ 83.6
(n = 256), growing as O(1/h), diverging a perturbed run near t ~ 0.01.

Provenance: `adc_cases` ADC-62; the frozen-equilibrium campaign is in
`docs/HOFFART_GEOMETRY_VERDICT.md` (ROUTE 1).

## Device codegen (named-functor nvcc fixes)

The transport and elliptic/mesh kernels are NAMED functors (not inline extended
lambdas) so nvcc reliably emits the nested device kernel. Provenance: the A==B
transport-residual parity (dres = 0) was first obtained on a CUDA device under
`adc_cases` #64; a related Release-Cuda-without-`-g` segfault in solve_fields (a
Heisenbug: passes on Serial, with compute-sanitizer, and with `-g`) was fixed the
same way under `adc_cases` #93. Full-production-path CUDA parity (np = 1) is gated
on closing that segfault; MPI multi-rank parity is a separate follow-up.

## Runtime/system surface (ADC-370)

ADC-333 deferred the runtime/system headers because they were then in flight with
ADC-369 (per-field aux halos, merged PR #192). ADC-370 applies the same wording on
the now-merged surface:

- `runtime/system.hpp` `set_transport_domain` disc mask: FV transport on the true
  disc instead of the full cartesian square (the "cartesian ring edges" lock) is the
  same topic as "Disc domain" above; detail in `HOFFART_GEOMETRY_VERDICT.md` and
  `HOFFART_FIDELITY.md`.
- Gauss-law policy `restart` (default) vs `evolve` (`runtime/system.hpp` and
  `runtime/system/system_field_solver.hpp`): `evolve` derives the aux from the
  Schur-evolved phi instead of re-solving -Delta phi each step, so the Gauss
  constraint is imposed only at t = 0. Originated as work item "R0" of the magnetic
  Euler-Poisson reproduction (arXiv:2510.11808); the mechanism stays in the header,
  only the case label was dropped.
- `runtime/system/system_field_solver.hpp` env-gated `POPS_TRACE_SOLVE_FIELDS` stderr trace: a CUDA
  device-crash diagnostic (the `adc_cases` #93 solve_fields segfault, see "Device
  codegen" above).
- `runtime/amr_system.hpp::set_conservative_state` + `coupling/amr_coupler_mp.hpp`
  drift seed: starts the AMR from a full drift state (rho, rho*u, rho*v) instead of
  m = 0 (the reference paper's "Problem 2" drift initial condition); only the case
  label was dropped, the mechanism stays in the header.

`runtime/system_stepper.hpp` and `runtime/system.hpp::set_time_scheme` were reviewed
and left as-is: the prose is generic (Lie/Strang splitting, phi consistency) and the
only case-named token is the `cf. docs/HOFFART_STEP_SEQUENCE.md` pointer, which is a
sanctioned validation-doc reference (same convention as the `HOFFART_FIDELITY.md`
pointers kept in ADC-333).
