# Add a new case

A *case* is a named scenario that builds a model, wires it into an `pops.System`, sets the
initial data, and runs the time loop. Cases are not part of `adc_cpp`: they live in the
companion `adc_cases` repository, one folder per case, each importing the installed `adc`
Python package. This guide assumes you can already build and import `adc`; if not, start
with the [installation guide](../getting-started/installation.md).

The convention is one folder per case under `adc_cases`. The folder holds a runnable
Python script that imports `adc`, composes a model, plugs it into a system, and steps it.

## Steps

1. Build and install `adc` so the case can import it. Use the `python` preset, which builds the
   importable `adc` module. The `serial` preset leaves `POPS_BUILD_PYTHON=OFF`, so `import adc`
   would later fail. Use the `python-parallel` preset instead for the multi-thread (Kokkos
   OpenMP) variant.

   ```bash
   cmake --preset python && cmake --build --preset python
   ```

2. In your clone of `adc_cases`, create one folder for the case. Replace `CASE_NAME` with a
   short name for the scenario.

   ```bash
   mkdir adc_cases/CASE_NAME
   ```

3. Write the case script in that folder. Compose a model the same way the tutorial does:
   `pops.Model(...)` for a native composition, `pops.dsl.Model(...)` for a formula model, or
   `pops.CompositeModel(...)` for a hybrid. See the [models overview](../models/index.md) for
   the three paths.

   ```python
   import pops

   model = pops.Model(
       state=pops.Scalar(),
       transport=pops.ExB(B0=1.0),
       source=pops.NoSource(),
       elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0),
   )

   sim = pops.System(n=96, L=1.0, periodic=True)
   sim.add_block("ne", model=model, spatial=pops.Spatial(minmod=True), time=pops.Explicit())
   sim.set_poisson(rhs="charge_density", solver="geometric_mg")
   sim.set_density("ne", ne0)
   sim.step_cfl(0.4)
   ```

   Replace `ne0` with a 2D array holding the initial density. The same `model` plugs into
   `pops.AmrSystem` for adaptive refinement without any change.

4. Run the case from its folder.

   ```bash
   python adc_cases/CASE_NAME/run.py
   ```

## Next steps

- Follow the [tutorial](../getting-started/tutorial.md) for the full reduced-diocotron walkthrough.
- Read the [models overview](../models/index.md) to choose between native bricks, the DSL, and a hybrid model.
