# System


`pops.System` builds the coupler. The configuration passed as keywords describes only the
mesh (`n`, `L`, `periodic`): the domain is a `[0, L]^2` square of `n x n` cells.

```python
import numpy as np
import pops

sim = pops.System(n=128, L=1.0, periodic=False)
```

You then add one block per model. A model is a brick composition
(`pops.Model(state, transport, source, elliptic)`); the block additionally receives a spatial scheme
(`pops.Spatial` / `pops.FiniteVolume`) and a time policy (`pops.Explicit`, `pops.IMEX`...).

```python
model = pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0),
                  source=pops.NoSource(), elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0))
sim.add_block("ne", model=model, spatial=pops.Spatial(minmod=True), time=pops.Explicit())
```

The system Poisson is shared by all blocks: its right-hand side is the sum of the elliptic
contributions of each block (`f = sum_s elliptic_rhs_s(u_s)`). You configure it once,
after the blocks.

```python
sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet",
                wall="circle", wall_radius=0.40)
```

`rhs` is `"charge_density"` (usual case: all blocks carry a charge density
`f = sum_s q_s n_s`) or `"composite"` (generic sum of the elliptic bricks per block);
`solver` is `"geometric_mg"` (multigrid, any case) or `"fft"` (periodic, `n = 2^k`); `bc`
handles the boundary condition (`"auto"`, `"dirichlet"`, `"periodic"`), `wall`/`wall_radius`
materialize a circular conducting wall (cut-cell). `set_poisson` is the shortcut for
`add_elliptic_model` (the Poisson is an instance of a composable elliptic model).

You set the initial condition then you advance:

```python
sim.set_density("ne", ne0)          # (n, n) array
sim.step_cfl(0.4)                   # one step at CFL 0.4 (returns the effective dt)
sim.advance(0.01, 50)               # 50 steps of fixed dt = 0.01
```

`add_block`, `add_equation`, `set_poisson`, `set_density`, `step`, `step_cfl`, `advance` and the
diagnostics are forwarded to the compiled facade. On the C++ side the coupler lives in
`runtime/system.hpp` (`System`, multi-block single-level, shared Poisson) and is exposed to Python
by `python/bindings/core/bindings.cpp`. The backend (serial / OpenMP / Kokkos GPU / MPI) is the one with which
`libadc` was compiled; the physics never sees the backend.

> AMR variant. `pops.AmrSystem(n=, L=, periodic=)` is the refined counterpart of `System`:
> same `add_block` / `add_equation` / `set_poisson` / `set_density` / `step_cfl` signatures,
> plus `set_refinement(threshold)` (refines where the density exceeds a threshold) and
> `set_phi_refinement(grad_threshold)` (refines on `|grad phi|`). The regrid cadence is set
> via `AmrSystemConfig.regrid_every` (0 = frozen hierarchy). Detail:
> [AMR_MULTIBLOCK_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/AMR_MULTIBLOCK_DESIGN.md).

## Multi-block and multi-species


Several blocks co-exist in the same `System`, coupled only by the right-hand side of the
shared Poisson (`f = sum_s q_s n_s`) and, optionally, by inter-species sources; never
by the flux. Each block keeps its own model, its own spatial scheme and its own
time policy. In multi-block, the block name indexes `set_density(name)` / `density(name)`
/ `mass(name)`.

```python
from pops.numerics.riemann import HLLC

n = 48
electrons = pops.Model(state=pops.FluidState("compressible", gamma=1.4),
                      transport=pops.CompressibleFlux(),
                      source=pops.PotentialForce(charge=-1.0),
                      elliptic=pops.ChargeDensity(charge=-1.0))
ions = pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                 transport=pops.IsothermalFlux(),
                 source=pops.PotentialForce(charge=+1.0),
                 elliptic=pops.ChargeDensity(charge=+1.0))

sim = pops.System(n=n, L=1.0, periodic=True)
sim.add_block("electrons", model=electrons,
              spatial=pops.Spatial(vanleer=True, flux=HLLC()),
              time=pops.IMEX(substeps=10))           # stiff: implicit source, subcycled
sim.add_block("ions", model=ions, spatial=pops.Spatial(minmod=True), time=pops.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
sim.set_density("electrons", ne0)
sim.set_density("ions", np.ones((n, n)))
sim.advance(0.001, 8)
print("blocks:", sim.block_names())
```

Inter-species coupled sources. In addition to the coupling by the field, inter-species sources
(operator-split, applied after the transport) transfer matter, momentum or
energy between blocks. Three fixed forms are added via `sim.add_coupling(...)` (or the direct
methods `add_ionization` / `add_collision` / `add_thermal_exchange`):

- `pops.Ionization(electron, ion, neutral, rate)`: ionization `n_g -> n_i + n_e` (rate
  `k n_e n_g`), mass transferred from the neutral to the ion;
- `pops.Collision(a, b, rate)`: inter-species friction (force `k (u_a - u_b)`), momentum
  conserved (fluid blocks, >= 3 variables);
- `pops.ThermalExchange(a, b, rate)`: thermal exchange `k (T_a - T_b)`, energy conserved
  (Euler blocks with 4 variables).

```python
sim.add_ionization(electron="ne", ion="ni", neutral="ng", rate=0.5)   # n_g decreases, n_i increases
sim.add_coupling(pops.Collision("a", "b", rate=1.0))                    # momentum transfer a -> b
sim.add_thermal_exchange("a", "b", rate=1.0)                           # energy hot -> cold
```

For a generic inter-species source (described in formulas rather than fixed), the DSL
`pops.physics.multispecies.CoupledSource(...).compile(...)` produces a descriptor that `sim.add_coupling(...)`
plugs in too (interpreted bytecode on the C++ side, no per-cell Python callback, MPI-safe). The
detail of the multi-species / plasma case is in
[ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md) (section 18, "composition runtime and multi-species
system") and [ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md). Exhaustive coupling surface:
[COUPLING_SURFACE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/COUPLING_SURFACE.md), [COUPLER_HIERARCHY.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/COUPLER_HIERARCHY.md).
