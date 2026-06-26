# Add a native brick

A native model is a composition of four role bricks, each already compiled in C++:
`pops.Model(state=, transport=, source=, elliptic=)`. The `state` brick declares the
conservative variables, the `transport` brick writes the physical flux and the eigenvalues,
the `source` brick adds a pointwise source term, and the `elliptic` brick contributes to the
right-hand side of the system Poisson. This recipe shows how to pick one brick per role and
plug the resulting model into a simulation. If your equation has no native brick for a role,
write that slot as a symbolic formula instead (see
[Write a model with the DSL](../tutorials/write-a-model-with-dsl.md)).

For the full registry of bricks, signatures, parameters and constraints, see the
[native bricks reference](../reference/native-bricks.md).

## Before you begin

Build and import `pops`. For a first local run, use the Kokkos Serial backend (see
[Installation](../getting-started/installation.md)). The bricks in this recipe run on the
native `add_block` path, which preserves MPI, AMR and GPU.

## Steps

1. Choose the `state` brick. It fixes the number of variables and the compatible
   transport. Use `pops.Scalar()` for one transported density, or
   `pops.FluidState(kind="compressible", gamma=1.4)` for a four-variable Euler fluid.

2. Choose the matching `transport` brick. `pops.Model(...)` enforces the pairing:
   `pops.Scalar()` requires `pops.ExB(B0=1.0)`; `pops.FluidState(kind="compressible")`
   requires `pops.CompressibleFlux()`. An inconsistent pairing raises a `ValueError`. Replace
   `B0` with your background magnetic field.

3. Choose the `source` brick. Use `pops.NoSource()` for no pointwise source, or
   `pops.PotentialForce(charge=1.0)` to add the force `(q/m) rho E` on the momentum. Replace
   `charge` with the charge-to-mass ratio `q/m`.

4. Choose the `elliptic` brick. Use `pops.BackgroundDensity(alpha=1.0, n0=0.0)` for the
   neutralizing background `f = alpha (n - n0)`, or `pops.ChargeDensity(charge=1.0)` for the
   charge density `f = q n`.

5. Compose the model. `pops.Model(...)` returns a `ModelSpec`.

   ```python
   model = pops.Model(
       state=pops.Scalar(),
       transport=pops.ExB(B0=1.0),
       source=pops.NoSource(),
       elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0),
   )
   ```

6. Plug the model into a block, with a spatial scheme and a time integrator.

   ```python
   sim.add_block("ne", model=model, spatial=pops.FiniteVolume(limiter="minmod", riemann="rusanov"), time=pops.Explicit())
   ```

The same `ModelSpec` plugs into `pops.AmrSystem` for adaptive refinement without changing the
model.

## What to do next

- To declare a new flux that no native transport brick covers, write the brick as formulas in
  [Write a model with the DSL](../tutorials/write-a-model-with-dsl.md).
- To keep some native bricks and write only one slot as formulas, see
  [Write a model with bricks](../tutorials/write-a-model-with-bricks.md).
- For every brick parameter and constraint, see the
  [native bricks reference](../reference/native-bricks.md).
