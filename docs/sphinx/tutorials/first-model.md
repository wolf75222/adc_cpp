# Run your first model

Go from nothing to a real run: build the `adc` Python module, run the smallest model (the reduced
diocotron, a scalar density advected by the E x B drift), and see it produce output.

This is the shortest path. The model is the
[reduced diocotron from the presentation](../getting-started/overview.md): one density `n`
transported by the E x B drift, with the potential `phi` supplied at each step by a system Poisson
`-lap phi = alpha (n - n_i0)`. The drift is `v = (-d_y phi / B0, d_x phi / B0)`. This is the
diocotron normalization benchmark, not the full Euler-Poisson system.

## Prerequisites

- A C++23 compiler (AppleClang 16+, GCC 13+, or Clang 17+), CMake 3.21 or newer, Ninja, and
  Python 3.10 or newer with `numpy` and `matplotlib`. The simplest way to get all of these is the
  repository conda env, set up in [Installation](../getting-started/installation.md).
- A clone of the repository. If you do not have one, run the clone step below.

## Steps

1. Clone the repository and enter it.

   ```bash
   git clone https://github.com/wolf75222/adc_cpp.git
   ```

   ```bash
   cd adc_cpp
   ```

2. Create and activate the conda env. This pins the right toolchain for your platform and provides
   CMake, Ninja, Python, `numpy`, `matplotlib`, Kokkos, and pybind11.

   ```bash
   bash scripts/setup_env.sh
   ```

   ```bash
   conda activate adc
   ```

3. Build and install the `adc` module. `pip install .` drives the build through scikit-build-core
   and installs the package into `site-packages`, so `import adc` works without setting
   `PYTHONPATH`. The build uses the Kokkos Serial backend, the standard path for a first local run.

   ```bash
   pip install .
   ```

   The core is header-only; only the Python module is compiled, which takes a few minutes the first
   time.

4. Check that the install is healthy. `adc.doctor()` checks each link in the environment
   (interpreter, `numpy`, Kokkos, the DSL compiler, header and module sync) and prints a remedy on
   any failure.

   ```bash
   python -c "import adc; adc.doctor()"
   ```

5. Run the model. The tested example script `diocotron_tutorial.py` is self-contained; it depends
   only on `adc`, `numpy`, and `matplotlib`. The `--quick` flag reduces the resolution and the step
   count for a fast first run.

   ```bash
   python docs/sphinx/tutorials/diocotron_tutorial.py --quick
   ```

   Without `--quick`, the script runs at its defaults (`--n 96 --steps 60`):

   ```bash
   python docs/sphinx/tutorials/diocotron_tutorial.py
   ```

## What the model is

The script writes the model symbolically with the symbolic DSL (`adc.dsl.Model`): a single
conservative variable `n`, the auxiliary fields `phi`, `grad_x`, `grad_y` supplied by the solver,
the E x B advection flux, and the elliptic right-hand side `alpha (n - n_i0)`. Here is the model
definition from the tested script:

```{literalinclude} ./diocotron_tutorial.py
:language: python
:pyobject: diocotron_model
```

The two physical constants that drive the reduced model, `B0` (the background magnetic field) and
`ALPHA` (the factor of the elliptic right-hand side), must stay consistent between the flux formulas
and the Poisson right-hand side. They are set near the top of the script:

```{literalinclude} ./diocotron_tutorial.py
:language: python
:lines: 55-57
```

The same physics can also be composed from native bricks with `adc.Model(state, transport, source,
elliptic)`. The DSL formulas and the native bricks are two ways to describe the same physics and
produce an identical numerical kernel. The brick catalog is in the
[native brick reference](../reference/native-bricks.md), and the formula syntax is in the
[symbolic DSL reference](../reference/symbolic-dsl.md).

## Expected result

The script prints its progress as it runs: where `adc` was imported from, the parallelism backend
in use (serial for the Python module), the time taken to compile and wire in the DSL model, the
growth of the perturbation amplitude over the run, and the mass drift. It then reports the
binary-level equivalence between the native bricks and the DSL formulas, and a uniform versus AMR
consistency check. It ends with a confirmation that the figures and a provenance record were
written, followed by an `OK` line.

The script writes its figures and a provenance record into the `_assets` directory next to the
script:

- `diocotron_growth.png`: the L2 amplitude of the perturbation (log scale) against time, next to
  the final density.
- `diocotron.gif` and `diocotron_cover.png`: the time evolution of the density, animated and as a
  static cover image.
- `diocotron_native_vs_dsl.png`: the final density from native bricks against the DSL formulas; the
  maximum difference is zero to binary precision.
- `diocotron_uniform_vs_amr.png`: the final density on a uniform grid against a refined AMR
  hierarchy.
- `provenance.json`: the `adc_cpp` commit, backend, resolution, and command, for reproducibility.

If those files appear and the script ends on its `OK` line, you have run your first model.

## Troubleshooting

If `import adc` raises an `ImportError` on `adc._adc`, the extension is pinned to the interpreter
that built it (suffix `cpython-312`): build and import with the same Python, the one from the conda
env. Whatever the symptom, run `python -c "import adc; adc.doctor()"` first; each line names the
failing link and its remedy.

## Next

To understand the two coupled parts of this model, read
[hyperbolic-elliptic systems](../concepts/hyperbolic-elliptic-systems.md) and the
[physical model](../concepts/physical-model.md). To see the full set of bricks you can compose, see
the [models overview](../models/index.md).
