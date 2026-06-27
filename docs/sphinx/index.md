# PoPS - Plasma-Oriented PDE Solver

PoPS is a model-free C++23 core for coupled hyperbolic-elliptic systems on adaptive
(AMR) meshes. It names no scenario: it advances a hyperbolic part `U` (finite-volume
transport) coupled to an elliptic part `phi` (a system Poisson solved each step), and a
model is a composition of generic bricks (`pops.Model(state, transport, source, elliptic)`).

The mesh and execution stack is written *from scratch*: a single dispatch seam
(Serial / OpenMP / Kokkos GPU GH200 / MPI), a `MultiFab` + `BoxArray` + `Geometry` stack,
block-structured multi-level and multi-patch AMR (Berger-Rigoutsos, coverage-aware reflux),
multigrid and spectral-FFT Poisson, and Python bindings via pybind11.

Named physical scenarios -- diocotron (E x B drift), self-gravitating Euler-Poisson,
asymptotic-preserving isothermal two-fluid -- are how the core is validated, not what it is.
Their readable, reproducible form (models, facades, figures) lives in
[`adc_cases`](https://github.com/wolf75222/adc_cases); the runs behind them are tracked in
[VALIDATION](https://github.com/wolf75222/adc_cpp/blob/master/docs/VALIDATION.md).

This documentation is the user guide. The detailed design documents
([ARCHITECTURE](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md),
[ALGORITHMS](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md),
[BACKEND_COVERAGE](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md)...)
remain the contributor reference and are linked where useful.

```{toctree}
:maxdepth: 2
:caption: Getting started

getting-started/index
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

tutorials/first-model
tutorials/write-a-model-with-bricks
tutorials/write-a-model-with-dsl
tutorials/moment-model-hyqmom15
tutorials/hybrid-model-native-plus-dsl
tutorials/euler-poisson-case
tutorials/hoffart-diocotron-case
tutorials/self-gravitating-euler-poisson
tutorials/isothermal-two-fluid-ap
```

```{toctree}
:maxdepth: 1
:caption: Concepts

concepts/hyperbolic-elliptic-systems
concepts/physical-model
concepts/conservative-primitive-variables
concepts/fluxes-sources-eigenvalues
concepts/moments-and-closures
concepts/elliptic-rhs
concepts/poisson
concepts/time-integration
concepts/strang-schur-imex
concepts/amr
concepts/multi-block-multi-species
concepts/polar-disc-geometry
```

```{toctree}
:maxdepth: 1
:caption: How-to guides

how-to/add-a-new-model
how-to/add-a-native-brick
how-to/add-a-dsl-model
how-to/add-a-new-flux
how-to/add-a-new-case
how-to/define-initial-conditions
how-to/configure-outputs-diagnostics
how-to/visualize-with-paraview
how-to/run-with-openmp
how-to/run-with-mpi
how-to/run-with-kokkos
how-to/run-on-gh200
how-to/profile-performance
```

```{toctree}
:maxdepth: 2
:caption: Writing a model

models/index
```

```{toctree}
:maxdepth: 2
:caption: Simulation

simulation/index
```

```{toctree}
:maxdepth: 2
:caption: AMR

amr/index
```

```{toctree}
:maxdepth: 2
:caption: Running

backends/index
```

```{toctree}
:maxdepth: 2
:caption: Advanced topics

advanced/index
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/index
```

```{toctree}
:maxdepth: 1
:caption: Development

development/index
```

```{toctree}
:maxdepth: 1
:caption: C++ reference

Embedded C++ API <doxygen/index>
C++ API (Doxygen) <https://wolf75222.github.io/adc_cpp/cpp/>
```

## In brief

Three orthogonal axes (concept `PhysicalModel`, policy `NumericalFlux`, concept
`EllipticSolver`) and a single parallelism seam:

- generic `System` composition: one block per model, where a model is a composition of
  generic bricks (`pops.Model(state, transport, source, elliptic)`); the core names no
  scenario (the names diocotron, euler_poisson... live on the `adc_cases` side). Shared
  system Poisson; on the Python side via `pops.System`. Three ways to write a model: native
  brick composition, symbolic `pops.physics.facade.Model` model, or hybrid composition (see
  [Models](models/index.md)).
- `RusanovFlux` / `HLLCFlux` / `RoeFlux` flux, MUSCL reconstruction (Minmod / VanLeer) + WENO5-Z;
- `GeometricMG` (multigrid V-cycle red-black GS) / `PoissonFFTSolver` (direct spectral);
- AMR: `AmrSystem` single- and multi-block, multi-patch N levels, coverage-aware reflux,
  `AmrCouplerMP` (Berger-Rigoutsos regrid), see [AMR](amr/index.md);
- `for_each_cell`: serial / OpenMP / Kokkos; `comm.hpp`: MPI collectives, see
  [Parallel backends](backends/index.md).

New here? Start with the [overview](getting-started/overview.md),
[install](getting-started/installation.md), then follow the
[A->Z tutorial](getting-started/tutorial.md).

## Links

- Source code: <https://github.com/wolf75222/adc_cpp>
- C++ Doxygen reference: [/cpp/](https://wolf75222.github.io/adc_cpp/cpp/)
- Sister solvers on the [pde_core_cpp](https://github.com/wolf75222/pde_core_cpp) base:
  euler_cpp and advection_cpp (private repositories)
- Application scenarios: [adc_cases](https://github.com/wolf75222/adc_cases)
