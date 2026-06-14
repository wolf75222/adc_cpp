<div align="center">

# adc_cpp

**A model-free C++23 core for coupled hyperbolic-elliptic systems on adaptive (AMR) meshes, with MPI and GPU backends.**

![C++23](https://img.shields.io/badge/C%2B%2B-23-blue?logo=cplusplus)
![Build](https://img.shields.io/badge/build-CMake%203.21%2B%20(presets)-064F8C?logo=cmake)
![Backends](https://img.shields.io/badge/backends-MPI%20%7C%20Kokkos%20(CPU%2FGPU)-orange)
![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey)
![License](https://img.shields.io/badge/license-BSD--3-green)

</div>

<p align="center">
  <img src="docs/anim_romeo_diocotron_amr3.gif" alt="Diocotron instability, 3-level AMR, on ROMEO" width="640">
</p>

<div align="center">
<sub>
Diocotron instability (E x B drift) on a 3-level nested AMR hierarchy, ROMEO (96 cores).
Reproducible local version (Python facade):
<a href="https://github.com/wolf75222/adc_cases/tree/master/diocotron_amr"><code>adc_cases/diocotron_amr</code></a>.
</sub>
</div>

---

`adc_cpp` is the **core**: a model-free engine, a library of generic physics bricks
(`include/adc/physics/`) and Python bindings (`adc`). It names no scenario; it provides generic
bricks composed into a `CompositeModel`. Named scenarios (diocotron, Euler-Poisson, two-fluid)
live in [`adc_cases`](https://github.com/wolf75222/adc_cases).

On a Cartesian adaptive mesh, the core advances a hyperbolic part `U` coupled to an elliptic
part `phi`:

```
d U / d t  +  div F(U, aux)  =  S(U, aux)
D phi      =  f(U)
```

The coupling flows through the `aux` channel at each step. The base contract is
`(phi, grad_x, grad_y)`; a model may declare `n_aux` to read extra fields (`B_z`, `T_e`).

## Table of contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Versioning](#versioning)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- **C++20** compiler: AppleClang 16+, GCC 13+, Clang 17+ (`nvcc_wrapper` for the CUDA target).
- **CMake >= 3.21**: the build is driven by presets ([CMakePresets.json](CMakePresets.json)).
- **[Kokkos](https://kokkos.org) 4.2+**: the only on-node backend, required. No need to
  pre-install it; if it is not found, CMake fetches and builds it (FetchContent).
- **MPI** *(optional, `-DADC_USE_MPI=ON`: halos and distributed FFT)*.
- **HDF5** parallel *(optional, `-DADC_USE_HDF5=ON`: DataWriter)*.
- **Python 3.12 + numpy** *(optional, the `adc` bindings; conda env via `scripts/setup_env.sh`)*.

Per-platform backend coverage and known pitfalls (macOS, CUDA, conda, CI runners):
[docs/BACKEND_COVERAGE.md](docs/BACKEND_COVERAGE.md).

## Installation

Three ways. Build-from-source details live in the
[installation guide](docs/sphinx/getting_started/installation.md) rather than inline here.

C++ core, via CMake presets:

```bash
git clone https://github.com/wolf75222/adc_cpp.git && cd adc_cpp
cmake --preset serial && cmake --build --preset serial && ctest --preset serial
```

Python module (`adc`): `scripts/setup_env.sh` creates the conda env and pins the platform
toolchain, then `pip install .` (scikit-build-core) drives the build. Backends are selected by
environment variables (`ADC_USE_MPI`, `Kokkos_ROOT`, ...).

```bash
bash scripts/setup_env.sh    # conda env + toolchain
pip install .                # see the installation guide for backends
```

Released versions and binaries: the
[Releases page](https://github.com/wolf75222/adc_cpp/releases).

## Usage

### From a C++ project

The core is header-only and consumed via `find_package(adc)` or FetchContent:

```cmake
include(FetchContent)
FetchContent_Declare(adc_cpp GIT_REPOSITORY https://github.com/wolf75222/adc_cpp.git)
FetchContent_MakeAvailable(adc_cpp)   # adc_cpp's own tests are not built for the consumer
target_link_libraries(my_app PRIVATE adc::adc)
```

Define a type that satisfies the `PhysicalModel` concept, wrap it in a
`Coupler<Model, Elliptic>` (or `AmrCouplerMP` for AMR), and advance in time.

### From Python

A model is written in either of two equivalent ways, plugged the same way into `adc.System` /
`adc.AmrSystem`: composed **bricks** (`adc.Model`, no just-in-time compilation), or symbolic
**formulas** (`adc.dsl.Model`, translated to C++ and compiled to a `.so`). Both produce
bit-identical results. Minimal example, the reduced diocotron as bricks (scalar density advected
by the E x B drift, neutralizing background):

```python
import adc
model = adc.Model(state=adc.Scalar(),
                  transport=adc.ExB(B0=1.0),
                  source=adc.NoSource(),
                  elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0))
sim = adc.System(n=96, L=1.0, periodic=True)
sim.add_block("ne", model=model, spatial=adc.Spatial(minmod=True), time=adc.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
sim.set_density("ne", ne0)          # ne0: initial density (2D array)
sim.step_cfl(0.4)
```

`adc.AmrSystem` composes one or more blocks on a refined hierarchy (`set_refinement`, regrid,
conservative reflux, composite FAC elliptic and a Schur-condensed source stage). Step-by-step
tutorial (bricks and formulas): [getting_started/tutorial](docs/sphinx/getting_started/tutorial.md).
Reference: [bricks_reference](docs/sphinx/reference/bricks_reference.md),
[dsl_reference](docs/sphinx/reference/dsl_reference.md).

## Documentation

- User guide (Sphinx): <https://wolf75222.github.io/adc_cpp/>
- C++ reference (Doxygen): <https://wolf75222.github.io/adc_cpp/cpp/>
- Canonical guides: [ARCHITECTURE](docs/ARCHITECTURE.md) (layers, modules, AMR),
  [ALGORITHMS](docs/ALGORITHMS.md) (methods, formulas), [CHOICES](docs/CHOICES.md) (design),
  [BACKEND_COVERAGE](docs/BACKEND_COVERAGE.md) (backend / test matrix),
  [VALIDATION](docs/VALIDATION.md).
- Documentation policy (taxonomy, tooling, update guide): [DOC_QUALITY](docs/DOC_QUALITY.md).

### Core layers

| Layer | Role | Entry point |
|---|---|---|
| `core/` | types, state, `PhysicalModel`, `EquationBlock`, `CoupledSystem` | [physical_model.hpp](include/adc/core/physical_model.hpp) |
| `physics/` | generic bricks composed into a `CompositeModel` | [composite.hpp](include/adc/physics/composite.hpp) |
| `numerics/` | reconstruction (Minmod / VanLeer / WENO5), flux (Rusanov / HLL / HLLC / Roe) | [reconstruction.hpp](include/adc/numerics/reconstruction.hpp) |
| `numerics/elliptic/` | `EllipticSolver` concept, geometric multigrid, FFT, composite FAC | [elliptic_solver.hpp](include/adc/numerics/elliptic/elliptic_solver.hpp) |
| `numerics/time/` | SSP-RK, multirate scheduler, IMEX, splitting, AMR engine | [numerics/time/](include/adc/numerics/time) |
| `coupling/` | `Coupler`, `SystemCoupler`, `AmrSystemCoupler`, `AmrCouplerMP` | [coupler.hpp](include/adc/coupling/coupler.hpp) |
| `amr/`, `mesh/`, `parallel/` | Berger-Rigoutsos clustering, regrid, MultiFab, MPI comm seam | [amr/](include/adc/amr) |
| `runtime/` | `System` / `AmrSystem` facades, `model_factory`, DSL, aux channel | [system.hpp](include/adc/runtime/system.hpp) |

### Ecosystem

| Repo | Role |
|---|---|
| `adc_cpp` (this repo) | hyperbolic-elliptic core on AMR, with GPU / MPI / Kokkos |
| [`adc_cases`](https://github.com/wolf75222/adc_cases) | applications: named models, facades, examples, Python |
| [`poisson_cpp`](https://github.com/wolf75222/poisson_cpp) | Poisson solvers (Thomas, SOR, CG, DST, multigrid) |
| [`advection_cpp`](https://github.com/wolf75222/advection_cpp) | advection, Burgers, Chorin Navier-Stokes |
| [`euler_cpp`](https://github.com/wolf75222/euler_cpp) | 2D Euler, viscous Navier-Stokes, plasma sources |

## Versioning

`adc_cpp` follows [Semantic Versioning](https://semver.org). The public API under guarantee and
the bump rules are declared in [docs/VERSIONING.md](docs/VERSIONING.md). Available versions and
their change logs: the [Releases page](https://github.com/wolf75222/adc_cpp/releases) and
[CHANGELOG.md](CHANGELOG.md). The project is in `0.y.z` initial development: the public API may
still change until `1.0.0`.

## Contributing

Build, test and workflow conventions: [CONTRIBUTING.md](CONTRIBUTING.md).

## License

BSD-3-Clause. See [LICENSE](LICENSE).
