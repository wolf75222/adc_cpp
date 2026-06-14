# Repository organization

The project is split into two repositories with a clear boundary: the generic library
on one side, the named scenarios on the other. Understanding this separation avoids looking for
"the diocotron" in the wrong place.

## `adc_cpp`: the library (the core)

<https://github.com/wolf75222/adc_cpp>

This is that repository. It contains:

- the header-only C++ core (`include/adc/`): physics bricks, numerical schemes, the AMR
  mesh stack, Poisson solvers, parallelism seams;
- the library's Python bindings (`python/`): the `adc` module (pybind11), that is, the
  composition facades `System` / `AmrSystem` and the `adc.dsl` DSL;
- the test suite (`tests/`, `python/tests/`) and the design docs (`docs/`).

Structuring rule: the core is model-agnostic. It names no scenario. It only
knows generic bricks (`Scalar` / `FluidState`, `ExB` / `CompressibleFlux`,
`NoSource` / `PotentialForce`, `BackgroundDensity` / `ChargeDensity`...) that are composed into a
model via `adc.Model(state, transport, source, elliptic)`, or written as formulas via
`adc.dsl.Model`. There is neither `diocotron` nor `euler_poisson` in `adc_cpp`. See
[`ARCHITECTURE.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md) ("The core is model-agnostic").

This rule also holds for custom integrators: the asymptotic-preserving two-fluid scheme,
which is not a block-by-block composable brick but a scenario, has left
the core. It lives on the applications side, compiled on the fly against the generic headers.

## `adc_cases`: the application layer (the named scenarios)

<https://github.com/wolf75222/adc_cases>

Sibling repository. It contains only Python code that imports the `adc` module and composes the
bricks into named scenarios: `diocotron`, `euler_poisson`, two species, magnetic isothermal,
two-fluid AP... Each case is a composition recipe (which bricks, which scheme, which
Poisson, which initial condition), not new core.

A few reference demonstrators (run in CI on the `adc_cases` side):

| Scenario | File |
|---|---|
| ExB single-species (DSL) | `diocotron_dsl/run.py` |
| Two species (DSL) | `two_species_dsl/run.py` |
| Magnetic isothermal (DSL) | `magnetic_isothermal_dsl/run.py` |

## Who depends on whom

```
adc_cpp  (coeur generique + bindings adc)   <-- agnostique, ne nomme aucun scenario
   ^
   | import adc  /  FetchContent adc::adc
   |
adc_cases  (scenarios Python nommes)         --> diocotron, euler_poisson, ...
```

The dependency is strictly downward: `adc_cases` consumes `adc_cpp` (the Python module, or
the `adc::adc` headers via `FetchContent` for scenarios compiled on the fly). There is
no reverse dependency: the core does not know the cases.

In practice: a new "well-known" model (a diocotron, an Euler-Poisson) is coded in
`adc_cases` by composing bricks. A new generic brick (a new flux, a
new elliptic closure) is coded in `adc_cpp`. The [A->Z tutorial](tutorial.md), for its part, remains
deliberately self-contained: it imports only `adc`, `numpy` and `matplotlib`, and rebuilds the
diocotron model by hand (with no dependency on `adc_cases`) so that it stays reproducible from this single
repository.
