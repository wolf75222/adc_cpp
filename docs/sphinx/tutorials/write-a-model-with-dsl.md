# Write a model with the symbolic DSL

Write the reduced diocotron model as symbolic formulas with `adc.dsl.Model`, compile it to a `.so`,
run it, and confirm it produces a result bit-identical to the same model built from native bricks.

## Prerequisites

- A built `adc` Python module. If you have not built it yet, follow
  [Installation](../getting-started/installation.md). For this tutorial, use the build from
  `cmake --preset python` or `pip install .`; the Kokkos Serial backend is enough.
- `numpy` and `matplotlib` in the same Python environment.
- The repository headers on disk, because the DSL compiles a `.so` against them. The `production`
  backend needs `_adc` and the generated `.so` built with the same adc headers (the ABI guard).
- Background, if you want it: the [models overview](../models/index.md) explains the three ways to
  write a model, and the [physical model concept](../concepts/physical-model.md) explains the
  `flux` / `max_wave_speed` / `source` / `elliptic_rhs` contract that the DSL fills in.

The model is the reduced diocotron benchmark: a single density `n`, advected by the E x B drift
$v = (-\partial_y \phi / B_0,\ \partial_x \phi / B_0)$, where the potential $\phi$ solves the system
Poisson with right-hand side $\alpha (n - n_{i0})$. A *symbolic DSL* (domain-specific language) is a
way to describe these formulas as an expression tree that the toolchain translates to C++ and
compiles. See the [hyperbolic-elliptic systems concept](../concepts/hyperbolic-elliptic-systems.md)
for what the coupling means.

This tutorial reuses the tested script `diocotron_tutorial.py`, which lives next to this page. The
code blocks below include parts of it verbatim through MyST `literalinclude`; you do not copy the
code by hand.

## Step 1: Open a Python session next to the script

Change into the repository and start the same Python interpreter that built `adc`.

```bash
cd adc_cpp
```

If you used the developer path (`cmake --preset python`), point the interpreter at the in-tree build
and the headers first. Skip this if you installed with `pip install .`.

```bash
export PYTHONPATH=$PWD/build-py/python
```

```bash
export ADC_INCLUDE=$PWD/include
```

`ADC_INCLUDE` tells the DSL where the headers are when it compiles the `.so`. Replace `$PWD` only if
you run from another directory; it must point at the repository root.

## Step 2: Import `adc` and the DSL

Import the module and the `dsl` submodule, then print the running parallelism backend (a Python
module built in CI runs Kokkos Serial).

```{literalinclude} diocotron_tutorial.py
:language: python
:lines: 50-53
```

```{literalinclude} diocotron_tutorial.py
:language: python
:pyobject: detect_backend_runtime
```

## Step 3: Write the model as formulas

Declare the physics with `adc.dsl.Model`. You declare the conservative variable `n`, the auxiliary
fields `phi` / `grad_x` / `grad_y` that the solver fills in (the `adc::Aux` channel), the E x B
advection flux, the eigenvalues (here a single wave, the drift speed), and the elliptic right-hand
side $\alpha (n - n_{i0})$ that couples the block to the system Poisson. The trailing `m.check()`
verifies that every referenced variable is declared.

```{literalinclude} diocotron_tutorial.py
:language: python
:pyobject: diocotron_model
```

`B0` and `ALPHA` are the two physical constants of the reduced model; the script sets them next to
the imports. The argument `n_i0` is the neutralizing ionic background; the script fixes it to the
mean of the initial density so the periodic Poisson is solvable. The full list of declarators
(`conservative_vars`, `aux`, `flux`, `eigenvalues`, `primitive_vars`, `conservative_from`,
`elliptic_rhs`, `param`, `check`) is in the [symbolic DSL reference](../reference/symbolic-dsl.md).

## Step 4: Compile the model to a `.so` and build the System

Compile the symbolic model and wire the resulting `CompiledModel` into a periodic `adc.System`. The
script first tries the `production` backend (the native zero-copy path, preferred under MPI and AMR),
then falls back to `aot` (numerically identical, marshaled host-side). The default of `m.compile(...)`
is `aot`; you ask for `"production"` explicitly. This is also where you choose the spatial scheme
(finite volume, `minmod` limiter, `rusanov` flux), the time integrator (explicit), and the system
Poisson (right-hand side = charge density, geometric multigrid).

```{literalinclude} diocotron_tutorial.py
:language: python
:pyobject: compile_and_build
```

The three backends (`prototype`, `aot`, `production`) and their MPI / AMR / GPU coverage are
described in the [symbolic DSL reference](../reference/symbolic-dsl.md). For a first local run, the
fallback to `aot` is expected and fine; both backends produce the same numerics.

## Step 5: Run the script end to end

Run the script. It builds the initial band of charge, compiles and attaches the DSL model, integrates
60 steps at fixed CFL, writes the figures, and runs the bricks-vs-DSL comparison.

```bash
python docs/sphinx/tutorials/diocotron_tutorial.py
```

For a faster smoke pass at lower resolution and fewer steps, use the `--quick` flag.

```bash
python docs/sphinx/tutorials/diocotron_tutorial.py --quick
```

### Expected result

The script prints the path `adc` was imported from and the parallelism backend, then the compiled
backend it retained (`production` or, after fallback, `aot`). It reports that the perturbation
amplitude grew over the run and that mass stays conserved by the periodic advective transport; it
asserts both, so a failed assertion means one of these did not hold. It then writes the figures and a
provenance record into the `_assets` directory next to the script, and prints `OK tutoriel
diocotron`.

The growth figure plots the L2 perturbation amplitude on a log scale against time, beside the final
density.


## Step 6: Confirm bricks and DSL are bit-identical

The same physics can also be composed from native bricks with `adc.Model(state, transport, source,
elliptic)`. The two writing fronts are interchangeable: they describe the same physics and produce an
identical numerical kernel. The script writes the brick version and replays the same grid, the same
scheme, and the same number of steps.

```{literalinclude} diocotron_tutorial.py
:language: python
:pyobject: native_diocotron_model
```

```{literalinclude} diocotron_tutorial.py
:language: python
:pyobject: native_vs_dsl
```

The difference between the two final states is zero to binary precision: the script computes
`max|native - DSL|` and asserts it equals `0.0`. A nonzero difference, even at the level of rounding,
would betray a wrong formula (the sign of the drift, a wave bound, the right-hand side). The DSL
formulas reproduce exactly the conventions of the native `ExB` and `BackgroundDensity` bricks.


## Troubleshooting

- Import error on `import adc`: the extension is pinned to the interpreter that built it. Import with
  the same Python, and run `python -c "import adc; adc.doctor()"` to check the environment.
- `RuntimeError` about headers when compiling: set `ADC_INCLUDE` to the repository `include`
  directory (Step 1). The DSL validates it by checking that `adc/mesh/storage/multifab.hpp` exists there.
- The script reports that the `production` backend is unavailable and continues on `aot`: this is the
  documented fallback when `_adc` and the `.so` were not built with the same headers. The `aot` run is
  numerically identical; nothing further is required for this tutorial.

## Next

- Compare a uniform grid against an adaptive hierarchy by reading the `uniform_vs_amr` part of the
  [A->Z tutorial](../getting-started/tutorial.md), which replays this same model on `adc.AmrSystem`.
- Browse the [native brick catalog](../reference/native-bricks.md) to see which models you can
  compose without writing formulas.
- Read the [symbolic DSL reference](../reference/symbolic-dsl.md) for the complete declarator and
  backend tables, including runtime parameters and hybrid native-plus-DSL models.
