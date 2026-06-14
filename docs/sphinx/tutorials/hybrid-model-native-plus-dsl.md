# Write a hybrid native plus DSL model

Mix one native brick and one symbolic DSL brick inside a single `adc.CompositeModel`, then
attach it to an `adc.System` and run.

A native brick is a generic piece of physics already compiled into the core (`adc.ExB`,
`adc.PotentialForce`, `adc.ChargeDensity`...). A symbolic DSL brick is physics you write as
formulas with `adc.dsl` and compile into a `.so`. A hybrid model fills the middle ground: you
reuse a native brick for one slot and write the other slot as formulas. Use it when part of the
physics already exists as a native brick and part is best expressed symbolically.

## Before you start

- A built `adc` Python module and the DSL toolchain (adc headers plus a C++ compiler). See the
  [installation guide](../getting-started/installation.md).
- The three model-writing fronts (native bricks, DSL, hybrid) from the
  [models overview](../models/index.md).
- The hybrid composition rule below uses the brick names from the
  [brick reference](../reference/native-bricks.md) and the DSL bricks from the
  [DSL reference](../reference/symbolic-dsl.md).

## How a CompositeModel is built

`adc.CompositeModel(transport, source, elliptic)` takes three slots. Each slot accepts either a
native brick or a partial compiled DSL brick (`adc.dsl.HyperbolicBrick`, `adc.dsl.SourceBrick`,
`adc.dsl.EllipticBrick`, each followed by `.compile()`). At least one slot must be a DSL brick. An
all-native composition is written with `adc.Model(...)` instead; otherwise `CompositeModel` raises
a `ValueError`.

The transport slot fixes the layout: the number of conservative variables `n_vars`, their names,
the primitives, and `gamma`. A DSL source or elliptic brick must declare the same `n_vars`. The mix
is compiled into a single composite `.so`; the native numerics are reused identically, with the
native parameters (`qom`, `q`, `cs2`...) baked into a derived struct.

## Steps

This example builds an isothermal transport as a DSL brick, then keeps the source and the elliptic
right-hand side as native bricks. The excerpt comes from `python/tests/test_dsl_hybrid.py`. Replace
`CS2` with the isothermal sound speed squared, `QOM` with the charge-to-mass ratio of the source
force, and `Q` with the charge of the elliptic coupling.

1. Import `adc` and the DSL, and fix the physical constants.

   ```python
   import adc
   from adc import dsl

   CS2, QOM, Q = 0.7, -1.0, -1.0
   ```

2. Write the transport as a DSL hyperbolic brick. `adc.dsl.HyperbolicBrick` replicates
   `adc::IsothermalFlux{cs2}` over three conservative variables: it declares the conservatives, the
   primitives, the physical flux, the eigenvalues, the primitive layout, and the inverse
   `conservative_from`.

   ```python
   def build_iso_transport(cs2):
       b = dsl.HyperbolicBrick("iso")
       rho, rho_u, rho_v = b.conservative_vars("rho", "rho_u", "rho_v")
       u = b.primitive("u", rho_u / rho)
       v = b.primitive("v", rho_v / rho)
       c = dsl.sqrt(cs2)
       b.flux(x=[rho_u, rho_u * u + cs2 * rho, rho_v * u],
              y=[rho_v, rho_u * v, rho_v * v + cs2 * rho])
       b.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
       b.primitive_vars(rho, u, v)
       b.conservative_from([rho, rho * u, rho * v])
       return b
   ```

3. Compose the model: the DSL transport (compiled) in the transport slot, a native
   `adc.PotentialForce` in the source slot, and a native `adc.ChargeDensity` in the elliptic slot.

   ```python
   m = adc.CompositeModel(
       transport=build_iso_transport(CS2).compile(),
       source=adc.PotentialForce(charge=QOM),
       elliptic=adc.ChargeDensity(charge=Q),
   )
   ```

4. Compile the composite model. The hybrid path uses the `aot` backend; `m.compile` returns a
   `CompiledModel` whose adder is `add_compiled_block`.

   ```python
   compiled = m.compile(backend="aot")
   ```

5. Build the system and attach the model. `add_equation` routes the `CompiledModel` to its backend
   adder. The `names=` list gives the three conservative variables, in the same order as
   `conservative_vars` in step 2.

   ```python
   sim = adc.System(n=48, L=1.0, periodic=True)
   sim.add_equation("gas", compiled,
                    spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"),
                    names=["rho", "rho_u", "rho_v"])
   ```

## Expected result

`build_iso_transport(CS2).compile()` produces a partial DSL brick, and `m.compile(backend="aot")`
produces one composite `.so` cached by its model hash (a rerun with the same formulas and
parameters reuses the cached `.so` instead of recompiling). `add_equation` attaches the model to
the system through `add_compiled_block`, with the three named conservative variables. The composite
brick advances the same way as a fully native or fully DSL model: the native source and elliptic
numerics are reused without re-derivation. The mix also works the other way around, with a native
transport and a DSL source or elliptic brick.

## Common pitfalls

- At least one slot must be a DSL brick. An all-native composition raises a `ValueError`; write it
  with `adc.Model(...)` instead.
- A DSL source or elliptic brick must declare the same `n_vars` as the transport slot, which fixes
  the layout.
- The hybrid path targets the `aot` backend. For the backend matrix (CPU, MPI, AMR) and the
  difference between `prototype`, `aot`, and `production`, see the
  [DSL reference](../reference/symbolic-dsl.md).

## Next

To write a complete model as formulas from scratch, read the
[DSL reference](../reference/symbolic-dsl.md). For the full list of native bricks you can place in
a composite slot, read the [brick reference](../reference/native-bricks.md).
