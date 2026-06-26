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
it installs `cxx-compiler` (gcc 14.2, C++23), a full toolchain without root rights. A `CC`/`CXX`
set by hand before a build keeps priority.

Manual equivalent, without the toolchain choice: `conda env create -f environment.yml`.
Update: re-run the script (or `conda env update -f environment.yml --prune`).
The env interpreter builds and imports the module: no cpython ABI divergence.
Standard: C++20 (Kokkos, the only on-node backend, is compiled under nvcc for the Cuda target).

(macos-fresh-install)=

## macOS: fresh install

On macOS the `adc` conda env carries Python, NumPy, Kokkos and the build tooling; the C++
compiler is the system AppleClang (`setup_env.sh` pins `CC`/`CXX` to `/usr/bin/clang` in the
env, far faster than a Homebrew LLVM at the head of PATH).

**1. Xcode command line tools** (provide AppleClang; `setup_env.sh` stops with a message if
they are missing).

```bash
xcode-select --install
```

**2. Miniforge** (conda-forge installer). Skip if conda is already on the PATH.

```bash
curl -L -o Miniforge3.sh \
  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda init "$(basename "$SHELL")"        # then open a new shell
```

**3. Clone, create the env, build, check.**

```bash
git clone https://github.com/wolf75222/adc_cpp.git ~/adc_cpp
cd ~/adc_cpp
bash scripts/setup_env.sh             # CPU Kokkos; pins AppleClang in the env
conda activate adc
bash scripts/build_python.sh          # one-command build + install, ends on pops.doctor()
python docs/sphinx/tutorials/diocotron_tutorial.py --quick
```

For CPU multithreading, the conda-forge `kokkos` is Serial-only; build a Serial+OpenMP Kokkos
into the env with `bash scripts/kokkos_openmp_conda.sh` (see [Threads](#threads)).

(linux-ubuntu-fresh-install)=

## Linux and Ubuntu: fresh install

From a clean machine with no Python, no compiler and no conda, one path reaches a healthy
environment. `bash scripts/setup_env.sh` automates steps 3-6; the manual steps are shown for
transparency.

**1. System prerequisites** (Debian/Ubuntu). Install only git + curl; do NOT `apt install` Python or
g++ for this workflow -- the `adc` conda env provides the interpreter AND the C++ compiler, and mixing
a system toolchain in is a common source of ABI surprises.

```bash
sudo apt update
sudo apt install -y git curl ca-certificates
```

**2. Miniforge** (conda-forge installer). Skip if conda is already on the PATH.

```bash
cd /tmp
curl -L -o Miniforge3.sh \
  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda init "$(basename "$SHELL")"        # then open a new shell
```

**3. Create the env** (CPU Kokkos by default). The script forces a CPU Kokkos
(`CONDA_OVERRIDE_CUDA=""`, so conda does not pull the CUDA variant on a host with an NVIDIA driver) and
configures conda-forge to survive HTTP 429. Pass `--cuda` only if you really want the CUDA variant.

```bash
git clone https://github.com/wolf75222/adc_cpp.git ~/adc_cpp   # if not already cloned
cd ~/adc_cpp
bash scripts/setup_env.sh            # CPU; or: bash scripts/setup_env.sh --cuda
conda activate adc
```

**4. Build the module.**

```bash
pip install . -v
```

**5. Persisted environment variables.** `setup_env.sh` pins `POPS_INCLUDE`, `POPS_KOKKOS_ROOT`,
`Kokkos_ROOT` and `POPS_CACHE_DIR` inside the env (exported on each activation) so the DSL
`production`/`aot` backend finds its Kokkos and its headers. They take effect on the next activation:

```bash
conda deactivate && conda activate adc
```

**6. Check, then first run.**

```bash
python -c "import adc; pops.doctor()"     # expect: => healthy environment
python docs/sphinx/tutorials/diocotron_tutorial.py --quick
```

If a step fails, `pops.doctor()` names the broken link and prints the copy-paste fix; the
[troubleshooting table](#troubleshooting) below maps each observed error to its remedy.

(update-clean-rebuild)=

## Update or clean rebuild

After a `git pull`, or to rebuild from clean caches, the same two scripts re-sync the env and
the module (works the same on macOS and Linux). `pops.doctor()` reports a module left stale by
the pull (its baked header signature no longer matches the tree) before you hit it at runtime.

```bash
cd ~/adc_cpp
git pull

conda activate adc
python -m pip uninstall -y adc-cpp || true   # drop any pip-installed copy
unset PYTHONPATH                             # so an old build tree cannot mask the install

rm -rf build/cp3* build-py build-py-kokkos build-py-conda .pops_cache   # wheel + DSL caches

bash scripts/setup_env.sh                     # refresh the env + toolchain pins
conda activate adc
bash scripts/build_python.sh                  # rebuild + reinstall, ends on pops.doctor()

python -c "import adc; print(pops.__file__)"
python -c "import adc; pops.doctor()"          # expect: => healthy environment
python docs/sphinx/tutorials/diocotron_tutorial.py --quick
```

`build_python.sh --fresh` does the cache wipe for you (drops the scikit-build wheel cache and
clears ccache for a truly cold build); the explicit `rm -rf` above also removes the preset
build trees (`build-py*`) and the DSL cache (`.pops_cache`).

## Python module

### User: `pip install .`

The quickest path, after `setup_env.sh`, is one command:

```bash
bash scripts/build_python.sh            # --clean to drop the wheel cache, --fresh for a cold build
```

It activates the env, sizes the heavy-TU pool from cores and RAM, exports the discovery vars and a
shared ccache, installs with `--no-build-isolation`, then runs `pops.doctor()`. The rest of this
section is what that does, step by step.

`pip install .` drives the CMakeLists through scikit-build-core (`pyproject.toml`) and installs the
package into `site-packages`: `import adc` then works without `PYTHONPATH`. Backends are
chosen through environment variables, mapped onto the CMake options:

```bash
conda activate adc
pip install . -v                               # builds the Kokkos module (Kokkos is ON, mandatory)
Kokkos_ROOT=$CONDA_PREFIX pip install . -v     # reuse the env Kokkos (OpenMP if available)
POPS_USE_MPI=ON pip install . -v                # MPI
```

`POPS_USE_KOKKOS` is ON by default and mandatory: `pip install .` always builds with Kokkos. With the
`adc` env active, `find_package(Kokkos)` finds the env Kokkos first (it is only fetched when none is
installed). On a host with an NVIDIA driver, conda may have resolved the **CUDA** Kokkos variant; the
build then fails with `Could not find nvcc`. Use `bash scripts/setup_env.sh` (it forces a CPU Kokkos),
or see the [Linux / Ubuntu fresh install](#linux-ubuntu-fresh-install) section below.

Then, in Python:

```python
import pops
print(pops.__version__)
pops.doctor()           # environment diagnosis (OK/FAIL + a remedy per line)
pops.set_threads()      # all cores -- or set_threads(8); BEFORE the 1st System
sim = pops.System(n=256)
```

`pip install -e .` (editable) suits dev; the build cache persists under `build/`, and
reinstalls are incremental.

**Slow build?** Three checks, in order:

1. the compiler. On macOS, an LLVM clang (Homebrew) at the head of PATH compiles the large
   units of the module more than 15 times slower than AppleClang (measured: ~1h24 vs
   ~5min). The configure step reports it; to force AppleClang:
   `CXX=/usr/bin/clang++ pip install .`;
2. ccache. Provided by the conda env and detected automatically: recompiling a file
   already seen becomes nearly instant;
3. parallelism. The heavy runtime dispatch is split into ~16 small translation units (so
   `-j` compiles them in parallel, no longer two giant units); a size-1 Ninja pool
   `POPS_HEAVY_TU_POOL` (default 1) still serializes them as an out-of-memory guard. On a
   high-RAM machine, widen it for a faster first build:
   `pip install . -C cmake.define.POPS_HEAVY_TU_POOL=$(nproc)` (leave it at 1 on a
   memory-constrained host or in CI);
4. reinstalls. The cache under `build/` makes `pip install .` incremental: only what
   changed recompiles, so only the first full build pays the optimized `-O3` cost.

### Developer: presets + PYTHONPATH

To iterate on the C++ without reinstalling, the presets build the module in the tree:

```bash
cmake --preset python          && cmake --build --preset python            # Kokkos Serial
cmake --preset python-parallel && cmake --build --preset python-parallel   # Kokkos conda (OpenMP)
export PYTHONPATH=$PWD/build-py/python        # or build-py-kokkos/python
```

A single path is enough: the configuration copies the package sources next to the extension.
Equivalent without preset:

```bash
cmake -S . -B build-py -G Ninja -DADC_BUILD_PYTHON=ON -DADC_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$(which python3.12)
cmake --build build-py --target _pops -j
```

(threads)=

### Threads

The number of threads is not a model argument: you need a module built against a Kokkos
OpenMP (preset `python-parallel`), then a setting BEFORE the first allocation,
Kokkos initializes at that moment and reads the environment only once:

```python
import pops
pops.set_threads(8)       # = OMP_NUM_THREADS + KOKKOS_NUM_THREADS, without touching the shell
sim = pops.System(n=256)
```

Kokkos Serial module or a call made too late: a warning signals it and the setting is ignored.
`pops.parallel_info()` gives the current state. For the DSL `backend="production"` to scale too,
export `POPS_KOKKOS_ROOT` (same Kokkos root as the module build).

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
| `POPS_BUILD_TESTS` | `ON` at top-level, `OFF` in a subproject | test suite (`tests/`) |
| `POPS_BUILD_PYTHON` | `OFF` | pybind11 module `adc` |
| `POPS_USE_KOKKOS` | `ON` | only on-node backend, **required** (`OFF` = fatal error); FetchContent if not installed |
| `POPS_USE_MPI` | `OFF` | distributed `comm` seam |
| `POPS_USE_HDF5` | `OFF` | links HDF5 + defines `POPS_HAS_HDF5`; HDF5 *output* itself goes through the Python `h5py` facade (`sim.write(format="hdf5")`), not a C++ writer |
| `POPS_BUILD_BENCH` | `OFF` | profiling harness (`bench/`) |
| `POPS_INSTALL` | `ON` at top-level | `cmake --install` rules + `find_package(adc)` |
| `POPS_PY_LTO` | `OFF` | ThinLTO of the module (`OFF` = fast build) |
| `POPS_USE_CCACHE` | `ON` | ccache if present (ignored under nvcc) |

Each option is also readable from the environment (`Kokkos_ROOT=... pip install .`);
an explicit `-D` keeps priority. The backend is a property of the `adc` target: everything
that links it inherits it, no flag in the code. adc_cpp is **Kokkos-only**: there is no longer a
standalone OpenMP backend nor a non-Kokkos build; Serial, OpenMP and Cuda are Kokkos
execution spaces chosen at install (or fetch) of Kokkos, not separate adc flags.
**Kokkos does not need to be pre-installed**: if not found, it is fetched + built
automatically (FetchContent, release tarball verified by SHA256).

**Kokkos via conda**: the `kokkos` package from conda-forge is generally compiled with the
Serial backend only. The build passes, but does not scale with threads, check the message
`Kokkos found ... = (...)` at configure. On a host with an NVIDIA driver, conda may instead resolve
the **CUDA** Kokkos variant (it drags in `cuda-cudart`); `pip install .` then fails with
`Could not find nvcc`. Force a CPU Kokkos with `bash scripts/setup_env.sh` (it sets
`CONDA_OVERRIDE_CUDA=""`) or `CONDA_OVERRIDE_CUDA="" conda env update -n adc -f environment.yml --prune`.
For a Kokkos Serial+OpenMP in the env (~2 min, same compiler as the project):

```bash
bash scripts/kokkos_openmp_conda.sh
cmake --preset python-parallel && cmake --build --preset python-parallel
export POPS_KOKKOS_ROOT="$CONDA_PREFIX"
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
# edit the '# A ADAPTER' lines (Kokkos path), then on each session/job:
source ~/romeo_adc.profile                       # POPS_ROMEO_ARCH=armgpu for the GPU
```

**Other cluster**: the generic guide
[HPC_SPACK_GUIDE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/HPC_SPACK_GUIDE.md)
covers installing the stack (site Spack or your own bootstrap), microarchitecture-targeted
compilation and SLURM launch with thread/GPU placement. Minimal schema:

```bash
spack load cmake ninja kokkos openmpi            # or module load <site env>
cmake -S . -B build-kokkos -G Ninja -DADC_USE_KOKKOS=ON \
      -DKokkos_ROOT=$(spack location -i kokkos)
# configure on the LOGIN node; inside the allocation, only re-run: ninja -C build-kokkos
```

Grace-Hopper GPU: [GPU_ROMEO.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/GPU_ROMEO.md).
For the DSL `backend="production"`, export `POPS_KOKKOS_ROOT=<prefix Kokkos>` (the ROMEO profile
does it); the build compiler is baked into `_pops`, the DSL finds it on its own as long as it
exists on the nodes.

(troubleshooting)=

## Troubleshooting

First reflex, whatever the symptom:

```bash
python -c "import adc; pops.doctor()"
```

Each line checks a link (interpreter/ABI, numpy, Kokkos, DSL compiler and its standard, the Kokkos
root for the production backend, a CUDA-Kokkos-without-nvcc trap, header/module sync, threads) and
gives the remedy on failure.

Common fresh-install errors and their fix:

| Error | Cause | Fix |
|---|---|---|
| `conda: command not found` | Miniforge not installed or not sourced | install Miniforge, or `source "$HOME/miniforge3/etc/profile.d/conda.sh"` |
| `python: command not found` | outside the conda env | `conda activate adc` |
| `EnvironmentNameNotFound: adc` | env not created | `bash scripts/setup_env.sh` |
| `CondaHTTPError: HTTP 429` | conda-forge rate limiting | `setup_env.sh` sets retries + libmamba; otherwise wait and retry |
| `Could not find nvcc` | a CUDA Kokkos was selected on a CPU host | `CONDA_OVERRIDE_CUDA="" bash scripts/setup_env.sh` (forces CPU Kokkos) |
| `[FAIL] include` (adc headers not found) | `POPS_INCLUDE` not set | `conda env config vars set POPS_INCLUDE="$HOME/adc_cpp/include"`, then reactivate |
| `[FAIL] kokkos_root` / `POPS_KOKKOS_ROOT is not defined` | DSL backend has no Kokkos root | `conda env config vars set POPS_KOKKOS_ROOT="$CONDA_PREFIX"` and `Kokkos_ROOT="$CONDA_PREFIX"`, then reactivate |
| `no DSL backend could be wired` | DSL backend compile failed | run `python -c "import adc; pops.doctor()"` and apply the fixes it names |
| `ModuleNotFoundError: matplotlib` | tutorial plotting deps missing | `conda install -n adc -c conda-forge matplotlib pillow` |

`setup_env.sh` already handles the first six on a fresh install; the table is the manual fallback.

**`ImportError` on `pops._pops`**: the extension is pinned to the interpreter that built it
(suffix `cpython-312`). The error message now indicates the exact cause (extension
absent, or wrong interpreter) and the rebuild command. Simple rule: build
and import with the same python, the one from the conda env.

**`error: invalid value 'c++23'` or `ABI incompatible` (DSL production)**: the `.so` loader
compiled at runtime must share the module toolchain. Three protections cover this case:
the build compiler is baked into the module and preferred over the PATH (`cxx=` explicit >
`$POPS_CXX` > build compiler > PATH), the standard is tested before compilation (fallback
`c++2b`), and the remaining error explains what to do. A module stale with respect to the headers
(after a `git pull`) is detected before loading, with the rebuild command.

## Verification

```python
import numpy as np
import pops

sim = pops.System(n=64, periodic=True)
sim.add_block("ne", model=pops.Model(
    state=pops.Scalar(), transport=pops.ExB(B0=1.0),
    source=pops.NoSource(), elliptic=pops.BackgroundDensity(alpha=1.0, n0=1.0)))
sim.set_poisson()
sim.set_density("ne", np.ones((64, 64)))
sim.step_cfl(0.4)
print(sim.density("ne").shape)   # (64, 64)
```

If these lines run, the install is good. Next: [first run](first-run.md) then
[A->Z tutorial](tutorial.md).
