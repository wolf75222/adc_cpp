<div align="center">

# PoPS - Plasma-Oriented PDE Solver

**A model-free C++23 core for coupled hyperbolic-elliptic systems on adaptive (AMR) meshes.**

![C++23](https://img.shields.io/badge/C%2B%2B-23-blue?logo=cplusplus)
![CMake](https://img.shields.io/badge/CMake-3.21%2B-064F8C?logo=cmake)
![Backends](https://img.shields.io/badge/backends-MPI%20%7C%20Kokkos-orange)
![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python)
![License](https://img.shields.io/badge/license-BSD--3-green)

</div>

<p align="center">
  <img src="docs/anim_romeo_diocotron_amr3.gif" alt="Diocotron instability, 3-level AMR, on ROMEO" width="480">
</p>

<div align="center">
<sub>
Validation scenario: diocotron instability (E x B drift) on a 3-level nested AMR hierarchy, ROMEO (96 cores).
Reproducible local version (Python facade):
<a href="https://github.com/wolf75222/adc_cases/tree/master/diocotron_amr"><code>adc_cases/diocotron_amr</code></a>.
</sub>
</div>

---

PoPS is a model-free engine with a library of generic physics bricks
(`include/pops/physics/`) and Python bindings (`pops`). It names no scenario; it provides generic
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
- **Python 3.12 + numpy** *(optional, the `pops` bindings; conda env via `scripts/setup_env.sh`)*.

Per-platform backend coverage and known pitfalls (macOS, CUDA, conda, CI runners):
[docs/BACKEND_COVERAGE.md](docs/BACKEND_COVERAGE.md).

## Installation

Three ways. Build-from-source details live in the
[installation guide](docs/sphinx/getting-started/installation.md) rather than inline here.

C++ core, via CMake presets:

```bash
git clone https://github.com/wolf75222/adc_cpp.git && cd adc_cpp
cmake --preset serial && cmake --build --preset serial && ctest --preset serial
```

The Ninja build already uses every core; pin it to fewer jobs on a constrained machine with
`cmake --build --preset serial -j<N>`. The serial test preset runs tests one at a time;
parallelize with `ctest --preset serial -j<N>` (`-j$(nproc)` on Linux, `-j$(sysctl -n
hw.ncpu)` on macOS), and add `--output-on-failure` for logs. Two other presets build a
parallel backend instead of the serial one (both read `$CONDA_PREFIX`, so the conda env must
be active):

```bash
cmake --preset parallel && cmake --build --preset parallel && ctest --preset parallel  # threaded, Kokkos OpenMP
cmake --preset mpi      && cmake --build --preset mpi      && ctest --preset mpi        # distributed, MPI
```

Each preset writes into its own folder (`build`, `build-kokkos`, `build-mpi`). Backends and
runtime thread control (`pops.set_threads()`) are covered in the
[installation guide](docs/sphinx/getting-started/installation.md).

Python module (`pops`): `scripts/setup_env.sh` creates the conda env and pins the platform
toolchain, then `scripts/build_python.sh` builds and installs the module in one command (it sizes
the heavy-TU pool, exports the discovery vars, and ends on `pops.doctor()`); `pip install .`
(scikit-build-core) drives the build directly if you prefer. Backends are selected by environment
variables (`POPS_USE_MPI`, `Kokkos_ROOT`, ...).

```bash
bash scripts/setup_env.sh      # conda env + toolchain
bash scripts/build_python.sh   # build + install, then pops.doctor()
# or, by hand:  pip install .  # see the installation guide for backends
```

Released versions and binaries: the
[Releases page](https://github.com/wolf75222/adc_cpp/releases).

## Usage

### From a C++ project

The core is header-only and consumed via `find_package(pops)` or FetchContent:

```cmake
include(FetchContent)
FetchContent_Declare(adc_cpp GIT_REPOSITORY https://github.com/wolf75222/adc_cpp.git)
FetchContent_MakeAvailable(adc_cpp)   # adc_cpp's own tests are not built for the consumer
target_link_libraries(my_app PRIVATE pops::pops)
```

Define a type that satisfies the `PhysicalModel` concept, wrap it in a
`Coupler<Model, Elliptic>` (or `AmrCouplerMP` for AMR), and advance in time.

### From Python

A model is written in either of two equivalent ways, plugged the same way into `pops.System` /
`pops.AmrSystem`: composed **bricks** (`pops.Model`, no just-in-time compilation), or symbolic
**formulas** (`pops.dsl.Model`, translated to C++ and compiled to a `.so`). Both produce
bit-identical results. Minimal example, the reduced diocotron as bricks (scalar density advected
by the E x B drift, neutralizing background):

```python
import pops
model = pops.Model(state=pops.Scalar(),
                  transport=pops.ExB(B0=1.0),
                  source=pops.NoSource(),
                  elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0))
sim = pops.System(n=96, L=1.0, periodic=True)
sim.add_block("ne", model=model, spatial=pops.Spatial(minmod=True), time=pops.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
sim.set_density("ne", ne0)          # ne0: initial density (2D array)
sim.step_cfl(0.4)
sim.write("ne.npz", format="npz")   # save the block states (npz; "vtk" also available)
```

A model written as **formulas** (`pops.dsl.Model`, translated to C++ and compiled to a `.so`,
plugged in the same way with `add_equation`). Here an isothermal fluid:

```python
from pops import dsl
m = dsl.Model("flow")
rho, mx, my = m.conservative_vars("rho", "mx", "my",
                                  roles=["Density", "MomentumX", "MomentumY"])
u, v = m.primitive("u", mx / rho), m.primitive("v", my / rho)
c = dsl.sqrt(0.5)                                   # isothermal sound speed
m.flux(x=[mx, mx * u + 0.5 * rho, mx * v],
       y=[my, my * u, my * v + 0.5 * rho])
m.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
m.primitive_vars(rho, u, v)
m.conservative_from([rho, rho * u, rho * v])
m.elliptic_rhs(0.0 * rho)                          # no elliptic coupling here
compiled = m.compile("flow.so")                    # codegen + C++ compile (headers auto-located)
sim.add_equation("flow", model=compiled,
                 spatial=pops.FiniteVolume(limiter="minmod"), time=pops.Explicit())
```

`pops.AmrSystem` composes one or more blocks on a refined hierarchy (`set_refinement`, regrid,
conservative reflux, composite FAC elliptic and a Schur-condensed source stage). Step-by-step
tutorial (bricks and formulas): [getting-started/tutorial](docs/sphinx/getting-started/tutorial.md).
Reference: [native-bricks](docs/sphinx/reference/native-bricks.md),
[symbolic-dsl](docs/sphinx/reference/symbolic-dsl.md).

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
| `core/` | types, state, `PhysicalModel`, `EquationBlock`, `CoupledSystem` | [physical_model.hpp](include/pops/core/model/physical_model.hpp) |
| `physics/` | generic bricks composed into a `CompositeModel` | [composite.hpp](include/pops/physics/composition/composite.hpp) |
| `numerics/` | reconstruction (Minmod / VanLeer / WENO5), flux (Rusanov / HLL / HLLC / Roe) | [reconstruction.hpp](include/pops/numerics/fv/reconstruction.hpp) |
| `numerics/elliptic/` | `EllipticSolver` concept, geometric multigrid, FFT, composite FAC | [elliptic_solver.hpp](include/pops/numerics/elliptic/interface/elliptic_solver.hpp) |
| `numerics/time/` | SSP-RK, multirate scheduler, IMEX, splitting, AMR engine | [numerics/time/](include/pops/numerics/time) |
| `coupling/` | `Coupler`, `SystemCoupler`, `AmrSystemCoupler`, `AmrCouplerMP` | [coupler.hpp](include/pops/coupling/single/coupler.hpp) |
| `amr/`, `mesh/`, `parallel/` | Berger-Rigoutsos clustering, regrid, MultiFab, MPI comm seam | [amr/](include/pops/amr) |
| `runtime/` | `System` / `AmrSystem` facades, `model_factory`, DSL, aux channel | [system.hpp](include/pops/runtime/system.hpp) |

### Ecosystem

| Repo | Role |
|---|---|
| `adc_cpp` (this repo) | hyperbolic-elliptic core on AMR, with GPU / MPI / Kokkos |
| [`adc_cases`](https://github.com/wolf75222/adc_cases) | applications: named models, facades, examples, Python |
| [`poisson_cpp`](https://github.com/wolf75222/poisson_cpp) | Poisson solvers (Thomas, SOR, CG, DST, multigrid) |
| [`advection_cpp`](https://github.com/wolf75222/advection_cpp) | advection, Burgers, Chorin Navier-Stokes |
| [`euler_cpp`](https://github.com/wolf75222/euler_cpp) | 2D Euler, viscous Navier-Stokes, plasma sources |

## Versioning

PoPS follows [Semantic Versioning](https://semver.org). The public API under guarantee and
the bump rules are declared in [docs/VERSIONING.md](docs/VERSIONING.md). Available versions and
their change logs: the [Releases page](https://github.com/wolf75222/adc_cpp/releases) and
[CHANGELOG.md](CHANGELOG.md). The project is in `0.y.z` initial development: the public API may
still change until `1.0.0`.

## Contributing

Build, test and workflow conventions: [CONTRIBUTING.md](CONTRIBUTING.md).

## License

BSD-3-Clause. See [LICENSE](LICENSE).

<p align="center">
  <img src="docs/banner_pops.png" alt="PoPS - Plasma-Oriented PDE Solver" width="100%">
</p>
