# First run

The smallest `adc` program: we build a periodic system, we place in it a
diocotron block composed of native bricks, we wire in the system Poisson, we set a numpy
initial condition, we advance a few steps and we read the density back. Copyable as is (it assumes
only that the module is [installed](installation.md) and importable).

```python
import numpy as np
import adc

# 1. Un systeme carre periodique 96 x 96, domaine [0, 1]^2.
sim = adc.System(n=96, L=1.0, periodic=True)

# 2. Un bloc "ne" : la densite, advectee par la derive E x B (transport=ExB),
#    couplee au Poisson via une densite de fond neutralisante (elliptic=BackgroundDensity).
sim.add_block(
    "ne",
    model=adc.Model(
        state=adc.Scalar(),                       # une variable conservative : la densite
        transport=adc.ExB(B0=1.0),                # vitesse de derive v = (-d_y phi, d_x phi) / B0
        source=adc.NoSource(),                    # pas de terme source
        elliptic=adc.BackgroundDensity(alpha=1.0, n0=1.0),  # rhs Poisson = alpha (n - n0)
    ),
    spatial=adc.Spatial(minmod=True),             # MUSCL minmod + Rusanov (defaut)
    time=adc.Explicit(),                          # integration explicite
)

# 3. Le Poisson de systeme : second membre = densite de charge, solveur multigrille.
sim.set_poisson(rhs="charge_density", solver="geometric_mg")

# 4. Condition initiale : une bande de charge perturbee le long de x.
n = 96
xs = (np.arange(n) + 0.5) / n
X, Y = np.meshgrid(xs, xs)                        # indexing 'xy' : ne[j, i]
y0 = 0.5 + 0.02 * np.cos(2.0 * np.pi * 2.0 * X)   # mode azimutal 2
ne0 = 1.0 + np.exp(-((Y - y0) ** 2) / 0.05 ** 2)
sim.set_density("ne", np.ascontiguousarray(ne0))

# 5. Quelques pas a CFL fixe, puis on relit l'etat.
for _ in range(20):
    sim.step_cfl(0.4)

print("t        =", sim.time())
print("masse    =", sim.mass("ne"))             # conservee par le transport advectif periodique
print("densite  =", sim.density("ne").shape)    # (96, 96)
```

What the key calls do:

- `adc.System(n=, L=, periodic=)` creates the system/coupler (square domain by default).
- `adc.Model(state=, transport=, source=, elliptic=)` composes a model from native
  bricks. Here: `Scalar` + `ExB` + `NoSource` + `BackgroundDensity`. This is exactly the
  reduced diocotron model, but the core does not name it.
- `set_poisson(rhs="charge_density", solver="geometric_mg")`: `rhs` is either `charge_density` or
  `composite`; `solver` is passed by keyword (`geometric_mg` or the FFT).
- `set_density(name, arr2d)` sets a contiguous `(n, n)` array; `step_cfl(cfl)` advances one step
  at the given CFL; `density(name)` / `mass(name)` / `time()` read the state.

To move from the model composed of bricks to the model written in formulas (DSL `adc.dsl.Model`),
to adaptive refinement (`adc.AmrSystem`), to the figures and the GIF, follow the
[A->Z tutorial](tutorial.md).
