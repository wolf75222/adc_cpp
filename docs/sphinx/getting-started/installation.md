# Installation

The core of `adc_cpp` is header-only: nothing to compile to consume it in C++. What gets
built is the test suite (C++) and the Python module `adc` (pybind11), either through
`pip install .` or through CMake directly.

## conda environment

A single command creates the `adc` env (full tooling: CMake, Ninja, ccache, Python 3.12, NumPy,
pybind11, Kokkos, OpenMPI, libomp) **and pins the best toolchain for the platform in it**:

```bash
bash scripts/setup_env.sh
conda activate adc
```

On macOS, the script sets `CC`/`CXX` to AppleClang inside the env itself (exported on each
activation, taking priority over the PATH): a vanilla LLVM clang sitting at the head of PATH
compiles the large units of the module more than 15 times slower, with no message. On Linux,
it installs `cxx-compiler` (gcc 14, C++23), a full toolchain without root rights. A `CC`/`CXX`
set by hand before a build keeps priority.

Manual equivalent, without the toolchain choice: `conda env create -f environment.yml`.
Update: re-run the script (or `conda env update -f environment.yml --prune`).
The env interpreter builds and imports the module: no cpython ABI divergence.
Standard: C++20 (Kokkos, the only on-node backend, is compiled under nvcc for the Cuda target).

## Python module

### User: `pip install .`

`pip install .` drives the CMakeLists through scikit-build-core (`pyproject.toml`) and installs the
package into `site-packages`: `import adc` then works without `PYTHONPATH`. Backends are
chosen through environment variables, mapped onto the CMake options:

```bash
conda activate adc
pip install .                                  # Kokkos Serial (FetchContent si non installe)
Kokkos_ROOT=$CONDA_PREFIX pip install . -v     # reutilise le Kokkos de l'env (OpenMP si dispo)
ADC_USE_MPI=ON pip install . -v                                # MPI
```

Then, in Python:

```python
import adc
print(adc.__version__)
adc.doctor()           # diagnostic de l'environnement (OK/FAIL + remede par ligne)
adc.set_threads()      # tous les coeurs -- ou set_threads(8) ; AVANT le 1er System
sim = adc.System(n=256)
```

`pip install -e .` (editable) suits dev; the build cache persists under `build/`, and
reinstalls are incremental.

**Slow build?** Three checks, in order:

1. the compiler. On macOS, an LLVM clang (Homebrew) at the head of PATH compiles the large
   units of the module 4 to 5 times slower than AppleClang (measured: over an hour instead
   of ~8 min). The configure step reports it; to force AppleClang:
   `CXX=/usr/bin/clang++ pip install .`;
2. ccache. Provided by the conda env and detected automatically: recompiling a file
   already seen becomes nearly instant;
3. reinstalls. The cache under `build/` makes `pip install .` incremental: only what
   changed recompiles. The first full build stays long (two large units at `-O3`), which is
   the cost of an optimized module, paid once.

### Developer: presets + PYTHONPATH

To iterate on the C++ without reinstalling, the presets build the module in the tree:

```bash
cmake --preset python          && cmake --build --preset python            # Kokkos Serial
cmake --preset python-parallel && cmake --build --preset python-parallel   # Kokkos conda (OpenMP)
export PYTHONPATH=$PWD/build-py/python        # ou build-py-kokkos/python
```

A single path is enough: the configuration copies the package sources next to the extension.
Equivalent without preset:

```bash
cmake -S . -B build-py -G Ninja -DADC_BUILD_PYTHON=ON -DADC_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$(which python3.12)
cmake --build build-py --target _adc -j
```

### Threads

The number of threads is not a model argument: you need a module built against a Kokkos
OpenMP (preset `python-parallel`), then a setting BEFORE the first allocation,
Kokkos initializes at that moment and reads the environment only once:

```python
import adc
adc.set_threads(8)       # = OMP_NUM_THREADS + KOKKOS_NUM_THREADS, sans toucher au shell
sim = adc.System(n=256)
```

Kokkos Serial module or a call made too late: a warning signals it and the setting is ignored.
`adc.parallel_info()` gives the current state. For the DSL `backend="production"` to scale too,
export `ADC_KOKKOS_ROOT` (same Kokkos root as the module build).

## C++ core and tests

```bash
cmake --preset serial   && cmake --build --preset serial   && ctest --preset serial   # Kokkos Serial
cmake --preset parallel && cmake --build --preset parallel && ctest --preset parallel  # Kokkos conda
cmake --preset mpi      && cmake --build --preset mpi      && ctest --preset mpi
```

Each preset writes into its own folder (`build`, `build-kokkos`, `build-mpi`); the
parallel/MPI presets require the conda env active (they read `$CONDA_PREFIX` and refuse to
configure without it). `ctest -L mpi` or `-L core` selects a subset. Manual equivalent:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The per-backend test count is kept in
[`BACKEND_COVERAGE.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md).

## Backends

| CMake option | Default | Role |
|---|---|---|
| `ADC_BUILD_TESTS` | `ON` at top-level, `OFF` in a subproject | test suite (`tests/`) |
| `ADC_BUILD_PYTHON` | `OFF` | pybind11 module `adc` |
| `ADC_USE_KOKKOS` | `ON` | only on-node backend, **required** (`OFF` = fatal error); FetchContent if not installed |
| `ADC_USE_MPI` | `OFF` | distributed `comm` seam |
| `ADC_USE_HDF5` | `OFF` | HDF5 `DataWriter` |
| `ADC_BUILD_BENCH` | `OFF` | profiling harness (`bench/`) |
| `ADC_INSTALL` | `ON` at top-level | `cmake --install` rules + `find_package(adc)` |
| `ADC_PY_LTO` | `OFF` | ThinLTO of the module (`OFF` = fast build) |
| `ADC_USE_CCACHE` | `ON` | ccache if present (ignored under nvcc) |

Each option is also readable from the environment (`Kokkos_ROOT=... pip install .`);
an explicit `-D` keeps priority. The backend is a property of the `adc` target: everything
that links it inherits it, no flag in the code. adc_cpp is **Kokkos-only**: there is no longer a
standalone OpenMP backend nor a non-Kokkos build; Serial, OpenMP and Cuda are Kokkos
execution spaces chosen at install (or fetch) of Kokkos, not separate adc flags.
**Kokkos does not need to be pre-installed**: if not found, it is fetched + built
automatically (FetchContent, release tarball verified by SHA256).

**Kokkos via conda**: the `kokkos` package from conda-forge is generally compiled with the
Serial backend only. The build passes, but does not scale with threads, check the message
`Kokkos found ... = (...)` at configure. For a Kokkos Serial+OpenMP in the env (~2 min,
same compiler as the project):

```bash
bash scripts/kokkos_openmp_conda.sh
cmake --preset python-parallel && cmake --build --preset python-parallel
export ADC_KOKKOS_ROOT="$CONDA_PREFIX"
```

**GPU**: `nvcc_wrapper` as the compiler, validated on ROMEO (not via conda):

```bash
cmake -S . -B build-gpu -DADC_USE_KOKKOS=ON \
      -DCMAKE_CXX_COMPILER=$KOKKOS/bin/nvcc_wrapper -DKokkos_ROOT=$KOKKOS
```

## Cluster (Spack, without root)

On ROMEO and similar systems, the tooling comes from modules/Spack, not conda (`conda env create`
requires the network, often absent from the nodes).

**ROMEO**: a versioned machine profile does the whole setup (site Spack env,
CC/CXX, Kokkos, DSL variables, cache in the scratch):

```bash
cp Tools/machines/romeo/romeo_adc.profile.example ~/romeo_adc.profile
# editer les lignes '# A ADAPTER' (chemin Kokkos), puis a chaque session/job :
source ~/romeo_adc.profile                       # ADC_ROMEO_ARCH=armgpu pour le GPU
```

**Other cluster**: the generic guide
[HPC_SPACK_GUIDE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/HPC_SPACK_GUIDE.md)
covers installing the stack (site Spack or your own bootstrap), microarchitecture-targeted
compilation and SLURM launch with thread/GPU placement. Minimal schema:

```bash
spack load cmake ninja kokkos openmpi            # ou module load <env du site>
cmake -S . -B build-kokkos -G Ninja -DADC_USE_KOKKOS=ON \
      -DKokkos_ROOT=$(spack location -i kokkos)
# configurer sur le LOGIN ; dans l'allocation, relancer seulement : ninja -C build-kokkos
```

Grace-Hopper GPU: [GPU_ROMEO.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/GPU_ROMEO.md).
For the DSL `backend="production"`, export `ADC_KOKKOS_ROOT=<prefix Kokkos>` (the ROMEO profile
does it); the build compiler is baked into `_adc`, the DSL finds it on its own as long as it
exists on the nodes.

## Troubleshooting

First reflex, whatever the symptom:

```bash
python -c "import adc; adc.doctor()"
```

Each line checks a link (interpreter/ABI, numpy, Kokkos, DSL compiler and its
standard, header/module sync, threads) and gives the remedy on failure.

**`ImportError` on `adc._adc`**: the extension is pinned to the interpreter that built it
(suffix `cpython-312`). The error message now indicates the exact cause (extension
absent, or wrong interpreter) and the rebuild command. Simple rule: build
and import with the same python, the one from the conda env.

**`error: invalid value 'c++23'` or `ABI incompatible` (DSL production)**: the `.so` loader
compiled at runtime must share the module toolchain. Three protections cover this case:
the build compiler is baked into the module and preferred over the PATH (`cxx=` explicit >
`$ADC_CXX` > build compiler > PATH), the standard is tested before compilation (fallback
`c++2b`), and the remaining error explains what to do. A module stale with respect to the headers
(after a `git pull`) is detected before loading, with the rebuild command.

## Verification

```python
import numpy as np
import adc

sim = adc.System(n=64, periodic=True)
sim.add_block("ne", model=adc.Model(
    state=adc.Scalar(), transport=adc.ExB(B0=1.0),
    source=adc.NoSource(), elliptic=adc.BackgroundDensity(alpha=1.0, n0=1.0)))
sim.set_poisson()
sim.set_density("ne", np.ones((64, 64)))
sim.step_cfl(0.4)
print(sim.density("ne").shape)   # (64, 64)
```

If these lines run, the install is good. Next: [first run](first-run.md) then
[A->Z tutorial](tutorial.md).
