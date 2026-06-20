# `adc/coupling` -- hyperbolic/elliptic coupling headers

The coupling layer wires the finite-volume transport (`numerics/`) to the elliptic
solve (`numerics/elliptic/`): it fills the auxiliary channel from the field, builds the
elliptic right-hand side from the state, and steps the coupled source. The headers are
grouped by **abstraction family** so the API surface is legible at a glance; every
historical flat path `<adc/coupling/<name>.hpp>` still resolves through a forwarding
stub (ADC-326), so the move is source-compatible.

## Families and stability

| Family | Path | What it is | Surface |
| --- | --- | --- | --- |
| base | `base/` | Coupling policy, aux fill, elliptic RHS contracts shared by every coupler. | Stable building blocks. |
| source | `source/` | Coupled-source state and its DSL program. | Stable. |
| single | `single/` | Single-block `Coupler`. | Stable. |
| static_system | `static_system/` | Compile-time `SystemCoupler` / `AmrSystemCoupler`. | Stable reference / static-typed C++ entry; used by tests and the numerical reference. |
| amr | `amr/` | Multipatch AMR coupler (`AmrCouplerMp`) and its storage, regrid, and diagnostics. | AMR production path. |
| schur | `schur/` | Schur condensation, the shared geometry-independent source kernels, and the condensed-source steppers (cartesian, polar, AMR). | Schur path. |
| deprecated | `deprecated/` | `spectral_coupler.hpp` -- superseded; kept for reference only. | Deprecated, do not use in new code. |

## Layout

```text
adc/coupling/
  base/            coupling_policy.hpp  aux_fill.hpp  elliptic_rhs.hpp
  source/          coupled_source.hpp  coupled_source_program.hpp
  single/          coupler.hpp
  static_system/   system_coupler.hpp  amr_system_coupler.hpp
  amr/             amr_coupler_mp.hpp  amr_level_storage.hpp  amr_regrid_coupler.hpp  amr_diagnostics.hpp
  schur/           schur_condensation.hpp  schur_source_kernels.hpp  condensed_schur_source_stepper.hpp
                   polar_condensed_schur_source_stepper.hpp  amr_condensed_schur_source_stepper.hpp
  deprecated/      spectral_coupler.hpp
```

New code should include the canonical family path (for example
`<adc/coupling/static_system/system_coupler.hpp>`). The flat forwarding stubs at
`<adc/coupling/<name>.hpp>` remain only for migration and may be retired once all
callers move to the family paths.
