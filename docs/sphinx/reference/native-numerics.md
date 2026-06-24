# Native numerics: Riemann solvers and reconstruction

The hot-path finite-volume bricks are C++ native in `include/adc/numerics/fv`. Python never
computes a Riemann flux or a WENO reconstruction; it selects a native brick (a descriptor,
see {doc}`typed-bricks`) and supplies the model-dependent quantities the brick needs.

## Native Riemann solvers (`include/adc/numerics/fv/numerical_flux.hpp`)

| Brick | C++ type | Needs from the model |
| --- | --- | --- |
| Rusanov (local Lax-Friedrichs) | `adc::RusanovFlux` | `max_wave_speed` |
| HLL | `adc::HLLFlux` | `physical_flux`, `wave_speeds` |
| HLLC | `adc::HLLCFlux` | `physical_flux`, `pressure`, `wave_speeds`, `contact_speed`, `hllc_star_state` |
| Roe | `adc::RoeFlux` | `physical_flux`, `roe_average` (or a `roe_dissipation` hook) |

HLLC and Roe are generic over the model via capability traits (`HasHLLCStructure`,
`HasRoeDissipation`); 2D Euler is only the fallback when no hooks are provided. A model that
lacks a required capability is rejected with a clear message, e.g.
`riemann HLLC requires model capability 'hllc_star_state' for state U`.

## Native reconstruction / limiters (`include/adc/numerics/fv/reconstruction.hpp`)

| Brick | C++ type |
| --- | --- |
| First order (no slope) | `adc::NoSlope` |
| MUSCL minmod | `adc::Minmod` |
| MUSCL van Leer | `adc::VanLeer` |
| WENO5-Z | `adc::Weno5` (this IS the WENO5-Z reconstruction) |

## Selecting them from Python

Today, a `dsl.Model` selects the solver/reconstruction by string
(`adc.FiniteVolume(riemann="hllc", reconstruction="weno5z")`) and generates the model hooks
from physical roles via `m.enable_hllc()` / `m.enable_roe()` (the hooks become `ADC_HD` C++
functions the native solver calls statically; no Python callback, no per-cell string lookup).

The Spec 3 descriptors name these native bricks without computing anything:

```python
import adc.lib as lib
lib.riemann.HLLC().native_id        # 'adc::HLLCFlux'
lib.reconstruction.WENO5Z().native_id  # 'adc::Weno5'
lib.riemann.HLLC().requirements     # {'capabilities': ['physical_flux', 'pressure', ...]}
```

## Capability validation (board)

`m.riemann("hllc"/"roe"/"hll"/"rusanov")` and `m.finite_volume_rate(riemann=...)` validate the
model's capabilities for the chosen solver before anything is generated, and canonicalize the
board roles (`density` -> `Density`, `momentum_x` -> `MomentumX`, ...) so the role lookup finds
them. HLLC/Roe require a pressure (a primitive `p` or a `pressure=` formula) and the fluid roles
Density/MomentumX/MomentumY; HLL requires wave speeds; Rusanov requires only a max wave speed.
Missing capabilities are rejected with a clear message, e.g.
`riemann HLLC requires model capability 'pressure' for state 'U'`. When valid, `m.riemann` drives
the dsl `enable_hllc()` / `enable_roe()` that generate the `ADC_HD` `contact_speed` /
`hllc_star_state` / `roe_dissipation` hooks from the roles.

## Status

The native solvers, the reconstruction bricks, the descriptor catalog, the board capability
validation and the role-derived hook generation are in place. Generating the hooks from
ARBITRARY board formulas (a non-canonical `contact_speed=` / `star_state=` written in `adc.math`,
beyond the role-derived Toro/Roe forms) and the end-to-end board-model compile path are the
remaining part of ADC-456.
