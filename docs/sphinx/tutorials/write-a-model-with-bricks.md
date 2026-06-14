# Write a model with native bricks

Compose a model from the four native role bricks (`state`, `transport`, `source`,
`elliptic`), plug it into an `adc.System`, set the system Poisson, and advance a few steps.

A *native brick* is a generic, already-compiled C++ piece of physics that you assemble from
Python. `adc.Model(state, transport, source, elliptic)` glues four of them into a model: the
cell-by-cell math stays compiled C++ (no numpy loop on the hot path), and there is no
just-in-time compilation. This is the most direct way to assemble a model that already exists as
bricks. To write physics that has no native brick, you use the symbolic DSL instead; see
[Models](../models/index.md).

In this tutorial you build the reduced diocotron model: a single electron density `n`, advected by
the E x B drift, coupled to a scalar Poisson with a neutralizing background. You reuse the exact
brick model that the [A->Z tutorial](../getting_started/tutorial.md) validates against the DSL.

## Prerequisites

- A working install of the `adc` Python module. If you do not have one, follow
  [Installation](../getting_started/installation.md). For a first local run, use the Kokkos Serial
  backend (the default of `pip install .`).
- `numpy`.
- Basic familiarity with the model concept; see
  [The PhysicalModel concept](../concepts/physical-model.md) and
  [Hyperbolic-elliptic systems](../concepts/hyperbolic-elliptic-systems.md).

## Step 1: Get the repository and build the module

Clone the repository and enter it.

```bash
git clone https://github.com/wolf75222/adc_cpp.git
```

```bash
cd adc_cpp
```

Build and install the `adc` module into your environment. The core is header-only, so only the
Python module is compiled.

```bash
pip install .
```

The full build, the developer build tree, and the optional environment variables are covered in
[Installation](../getting_started/installation.md).

## Step 2: Start a Python session and import `adc`

Start Python from the repository root and import the module.

```bash
python
```

```python
import numpy as np
import adc
```

## Step 3: Compose the model from four bricks

Build the model with `adc.Model(state, transport, source, elliptic)`. Each argument is one role
brick:

- `state=adc.Scalar()` declares one conservative variable `n` (the transported density);
- `transport=adc.ExB(B0=1.0)` advects `n` by the E x B drift `v = (-d_y phi, d_x phi) / B0`,
  where `B0` is the background magnetic field and `phi` is the potential solved by the Poisson;
- `source=adc.NoSource()` adds no pointwise source term (a scalar density needs none);
- `elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0)` couples the block to the system Poisson with
  the right-hand side `f = alpha (n - n0)`, the neutralizing background.

This is the same brick model that the A->Z tutorial replays. It comes from the tested script
[`diocotron_tutorial.py`](https://github.com/wolf75222/adc_cpp/blob/master/docs/sphinx/tutorials/diocotron_tutorial.py),
included here verbatim:

```{literalinclude} diocotron_tutorial.py
:language: python
:pyobject: native_diocotron_model
```

The script reads `B0` and `ALPHA` from two module constants. Define them in your session, choose
the neutralizing background `n_i0`, and call the function. Replace `N_I0` with the mean of your
initial density (this makes the periodic Poisson solvable); here you set it after building the
density in Step 5, so use a provisional value first.

```python
B0 = 1.0
```

```python
ALPHA = 1.0
```

`adc.Model(...)` validates the four roles as it builds them. The state and the transport must
agree: `Scalar` requires `ExB`. An inconsistent pairing (for example `Scalar` with a fluid flux)
raises a `ValueError` immediately. The full brick catalog, with every signature and constraint, is
the [native bricks reference](../reference/bricks_reference.md).

## Step 4: Create the system

Create a periodic Cartesian system. `adc.System(n=, L=, periodic=)` builds the coupler on a square
domain `[0, L]` by `[0, L]` with `n` by `n` cells. Replace `GRID` with the number of cells per
direction.

```python
n = 96
```

```python
sim = adc.System(n=n, L=1.0, periodic=True)
```

## Step 5: Set the initial density and the background

Build a 2D array for the initial electron density. Use a horizontal band of charge, perturbed
along `x` so the instability has a seed. The array follows the `ne[j, i]` convention.

```python
xs = (np.arange(n) + 0.5) / n
```

```python
X, Y = np.meshgrid(xs, xs)
```

```python
ne0 = 1.0 + np.exp(-((Y - 0.5) ** 2) / 0.05 ** 2)
```

```python
ne0 = ne0 * (1.0 + 0.02 * np.cos(2.0 * np.pi * 2.0 * X))
```

Fix the neutralizing ionic background to the mean of the initial density, then build the model.

```python
n_i0 = float(ne0.mean())
```

```python
model = native_diocotron_model(n_i0)
```

## Step 6: Add the block

Plug the model into the system with `add_block`. You pass the spatial scheme and the time
integrator here, not on the model: the same model is reused with different schemes.

- `spatial=adc.Spatial(minmod=True)` uses MUSCL reconstruction with the minmod limiter;
- `time=adc.Explicit()` uses the explicit SSPRK2 integrator.

```python
sim.add_block("ne", model=model, spatial=adc.Spatial(minmod=True), time=adc.Explicit())
```

`add_block` dispatches on the model type: the `ModelSpec` returned by `adc.Model(...)` goes to the
native block adder. The spatial schemes and time integrators are listed in the
[native bricks reference](../reference/bricks_reference.md).

## Step 7: Set the system Poisson

Wire the system Poisson. `set_poisson` chooses the right-hand side and the solver. Use the charge
density right-hand side (the sum of the blocks' elliptic contributions) and the geometric multigrid
solver.

```python
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
```

## Step 8: Load the density into the block

Copy the initial density into the `ne` block by name.

```python
sim.set_density("ne", ne0)
```

## Step 9: Advance in time

Advance the system one step, bounded by the CFL condition. `step_cfl(cfl)` picks the time step from
the wave speeds and the CFL number, then advances once. Run a few steps in a loop.

```python
for _ in range(50):
    sim.step_cfl(0.4)
```

## Step 10: Read back the state

Read the conserved mass, the potential, and the final density. `mass` returns the total mass of the
block (a conserved invariant under periodic transport); `potential` and `density` return 2D arrays.

```python
print("t        =", sim.time())
```

```python
print("mass(ne) =", sim.mass("ne"))
```

```python
phi = sim.potential()
```

```python
ne = sim.density("ne")
```

## Expected result

The session runs without raising. After the loop:

- `sim.time()` is positive: the system advanced 50 steps.
- `sim.mass("ne")` stays close to its initial value, because periodic advective transport conserves
  mass.
- `sim.potential()` and `sim.density("ne")` are 2D arrays of shape `(n, n)`.
- The perturbation seeded along `x` grows: the band develops the diocotron instability.

The same model object plugs into `adc.AmrSystem` (adaptive refinement) without any change:
`sa.add_block("ne", model=model, ...)`. The A->Z tutorial proves that this brick model produces a
state that is bit-identical to the same physics written as DSL formulas, and compares the uniform
and AMR runs.

## Troubleshooting

- `ValueError` at `adc.Model(...)`: the four roles are inconsistent. `Scalar` must be paired with
  `ExB`; check the pairing against the [native bricks reference](../reference/bricks_reference.md).
- `ImportError` on `import adc`: the module is not built or not on the path. The error message gives
  the cause and the rebuild command; run `python -c "import adc; adc.doctor()"` to check the
  environment, and see [Installation](../getting_started/installation.md).

## Next

- Read the full [native bricks reference](../reference/bricks_reference.md) for every brick, its
  signature, and its constraints.
- Write physics that has no native brick with the [symbolic DSL](../reference/dsl_reference.md).
- Run the complete diocotron simulation, with figures and the uniform/AMR comparison, in the
  [A->Z tutorial](../getting_started/tutorial.md).
