# First run

The smallest `pops` program: we build a periodic system, we place in it a
diocotron block composed of native bricks, we wire in the system Poisson, we set a numpy
initial condition, we advance a few steps and we read the density back. Copyable as is (it assumes
only that the module is [installed](installation.md) and importable).

```python
import numpy as np
import pops

# 1. A periodic 96 x 96 square system, domain [0, 1]^2.
sim = pops.System(n=96, L=1.0, periodic=True)

# 2. A block "ne": the density, advected by the E x B drift (transport=ExB),
#    coupled to Poisson via a neutralizing background density (elliptic=BackgroundDensity).
sim.add_block(
    "ne",
    model=pops.Model(
        state=pops.Scalar(),                       # one conservative variable: the density
        transport=pops.ExB(B0=1.0),                # drift velocity v = (-d_y phi, d_x phi) / B0
        source=pops.NoSource(),                    # no source term
        elliptic=pops.BackgroundDensity(alpha=1.0, n0=1.0),  # Poisson rhs = alpha (n - n0)
    ),
    spatial=pops.Spatial(minmod=True),             # MUSCL minmod + Rusanov (default)
    time=pops.Explicit(),                          # explicit integration
)

# 3. The system Poisson: rhs = charge density, multigrid solver.
sim.set_poisson(rhs="charge_density", solver="geometric_mg")

# 4. Initial condition: a perturbed charge band along x.
n = 96
xs = (np.arange(n) + 0.5) / n
X, Y = np.meshgrid(xs, xs)                        # indexing 'xy': ne[j, i]
y0 = 0.5 + 0.02 * np.cos(2.0 * np.pi * 2.0 * X)   # azimuthal mode 2
ne0 = 1.0 + np.exp(-((Y - y0) ** 2) / 0.05 ** 2)
sim.set_density("ne", np.ascontiguousarray(ne0))

# 5. A few fixed-CFL steps, then read the state back.
for _ in range(20):
    sim.step_cfl(0.4)

print("t        =", sim.time())
print("mass     =", sim.mass("ne"))             # conserved by the periodic advective transport
print("density  =", sim.density("ne").shape)    # (96, 96)
```

What the key calls do:

- `pops.System(n=, L=, periodic=)` creates the system/coupler (square domain by default).
- `pops.Model(state=, transport=, source=, elliptic=)` composes a model from native
  bricks. Here: `Scalar` + `ExB` + `NoSource` + `BackgroundDensity`. This is exactly the
  reduced diocotron model, but the core does not name it.
- `set_poisson(rhs="charge_density", solver="geometric_mg")`: `rhs` is either `charge_density` or
  `composite`; `solver` is passed by keyword (`geometric_mg` or the FFT).
- `set_density(name, arr2d)` sets a contiguous `(n, n)` array; `step_cfl(cfl)` advances one step
  at the given CFL; `density(name)` / `mass(name)` / `time()` read the state.

To move from the model composed of bricks to the model written in formulas (DSL `pops.dsl.Model`),
to adaptive refinement (`pops.AmrSystem`), to the figures and the GIF, follow the
[A->Z tutorial](tutorial.md).
