# Windows (WSL2)

This guide starts from scratch on a **Windows 11** PC and leads up to a running
`adc_cases` case, with no prior knowledge of the project. The recommended path is **WSL2 Ubuntu**:
all the Linux tooling already documented ([Installation](installation.md)) applies as is
once inside WSL2. Native Windows is not a goal (the core needs a Linux
toolchain; see epic Port Windows).

The differences from the Linux/macOS/ROMEO paths are marked **(Windows difference)**.

## 1. Prerequisites

### WSL2 + Ubuntu

```powershell
wsl --install            # depuis PowerShell admin ; installe Ubuntu, redemarre
wsl -l -v                # verifier : Ubuntu, VERSION 2
```

Everything else happens **in the Ubuntu terminal** (WSL2), not in PowerShell.

### NVIDIA GPU (optional, for CUDA)

No driver to install on the WSL side: the NVIDIA **Windows** driver exposes the GPU to WSL2. Check
from Ubuntu:

```bash
nvidia-smi               # doit lister le GPU (ex. RTX 3090, driver 591.86, CUDA 13.x)
```

If `nvidia-smi` responds, GPU passthrough works. The CUDA toolkit is installed **on the WSL side**
via conda (GPU section), no need to install it globally.

### conda (Miniforge)

```bash
curl -fsSL -o /tmp/mf.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/mf.sh -b -p "$HOME/miniforge3"
"$HOME/miniforge3/bin/conda" init bash && exec bash
```

### Clone the repositories into the WSL FS

**(Windows difference)** Clone into the **WSL** file system (`~/dev/...`), **not** into
`/mnt/c/...`: I/O on the mounted Windows disk is much slower and significantly slows down
the C++ builds.

```bash
mkdir -p ~/dev/Stage_Romain && cd ~/dev/Stage_Romain
git clone <url>/adc_cpp.git
git clone <url>/adc_cases.git
```

## 2. Build environment

Same as the documented Linux: from `adc_cpp/`,

```bash
bash scripts/setup_env.sh     # cree l'env `adc` + gcc 14 conda (CC/CXX figes)
conda activate adc
```

### (Windows difference) Kokkos conda = CUDA variant by default

conda-forge now serves `kokkos` as a **CUDA build** by default. Its `KokkosConfig.cmake`
requires `CUDAToolkit`: without a CUDA toolkit installed, **all** presets fail at configure
with `Could NOT find CUDAToolkit (missing CUDA_CUDART)` (even `serial`, because `find_package(Kokkos)`
picks up the conda kokkos via `CMAKE_PREFIX_PATH`).

For **CPU dev**, pin the CPU variant of Kokkos (it embeds OpenMP + Serial):

```bash
conda install -n adc -c conda-forge "kokkos=*=*hbbfbac7*"   # build sans 'cuda' dans le hash
# (verifier : `conda list kokkos` ne doit PAS afficher un build 'cuda12...')
```

The GPU is handled in a dedicated env (section 5), so as not to mix CPU and CUDA.

## 3. C++ core and tests

```bash
conda activate adc
cmake --preset serial
cmake --build --preset serial -j 6      # (ecart Windows) borner -j !
ctest --preset serial                   # attendu : 100% tests passed
```

**(Windows difference)** WSL2 caps the RAM (often ~50% of the host, e.g. 15 GB). The large
`system.cpp` / `amr_system.cpp` units at -O3 can cause **OOM `cc1plus`** at full
parallelism. Bounding to `-j 6` (~2.5 GB/job) avoids the OOM. ccache (provided by the env) speeds up
later rebuilds.

CPU multi-thread (Kokkos OpenMP):

```bash
cmake --preset parallel
cmake --build --preset parallel -j 6
OMP_NUM_THREADS=4 ctest --preset parallel
```

MPI (CPU, optional):

```bash
cmake --preset mpi && cmake --build --preset mpi -j 6
OMPI_MCA_btl_smcuda_use_cuda_ipc=0 ctest --preset mpi
```

## 4. Python module and `adc_cases` cases

```bash
conda activate adc
cd ~/dev/Stage_Romain/adc_cpp
cmake --build --preset python-parallel -j 6        # module _pops (Kokkos)
export PYTHONPATH="$PWD/build-py-kokkos/python:$PWD/python"
python -c "import adc; pops.doctor()"               # doit etre tout vert
```

```bash
cd ~/dev/Stage_Romain/adc_cases
pip install -e '.[figures]'
python check_cases.py                              # garde-fous docs/manifeste
export POPS_KOKKOS_ROOT="$CONDA_PREFIX"             # voir encadre ci-dessous
python tutorial/equivalence.py
python tutorial/run.py
```

### (Windows difference) DSL + Kokkos module: `POPS_KOKKOS_ROOT`

When `_pops` is compiled **with Kokkos**, the `production` DSL backend (which compiles a `.so` of the
model at runtime) must be too, otherwise the ABI guard rejects it (`kokkos=0` != `1`) and
no DSL backend compiles. Export `POPS_KOKKOS_ROOT=$CONDA_PREFIX` before running a case
that uses the DSL. With a **serial** module (preset `python`, without Kokkos), this is not required.

## 5. CUDA GPU (WSL2)

The GPU is validated in a **dedicated conda env** `adc-gpu` (kokkos CUDA + CUDA toolkit), separate from
the `adc` (CPU) env so as not to mix the two Kokkos.

```bash
mamba create -y -n adc-gpu -c conda-forge \
  "kokkos=*=*cuda12*" cuda-toolkit "cuda-version=12" cmake ninja cxx-compiler
conda activate adc-gpu
```

**(Windows difference)** The conda-forge CUDA `kokkos` is compiled for `sm_80`; it runs on
an RTX 3090 (`sm_86`) in compatibility mode (Kokkos emits a perf warning, the computation is
correct). For a native `sm_86` binary, recompile Kokkos CUDA by hand.

**(gotcha)** The conda activation scripts of `cuda-nvcc` reference `NVCC_PREPEND_FLAGS`
without a value; under `set -u`/`nounset` they kill the shell. Export before any activation:
`export NVCC_PREPEND_FLAGS="" NVCC_APPEND_FLAGS=""`.

The adc GPU build uses `nvcc_wrapper` as the C++ compiler:

```bash
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/nvcc_wrapper" \
  -DADC_USE_KOKKOS=ON -DKokkos_ROOT="$CONDA_PREFIX" -DADC_BUILD_TESTS=ON
```

### Verified GPU status (WSL2, RTX 3090)

- **Minimal Kokkos CUDA harness: OK.** A Kokkos `parallel_reduce` runs on the GPU
  (`DefaultExecutionSpace = Cuda`, correct result).
- **Dedicated adc GPU case: OK.** A small program using the **adc compute seam**
  (`Fab2D` / `Array4` / `for_each_cell` / `for_each_cell_reduce_sum`) with a **top-level**
  lambda compiles under `nvcc_wrapper` and runs on the GPU; validated against a **CPU oracle**
  (each cell == analytical formula, GPU reduction == host sum). The adc core is therefore
  device-clean on the local GPU.

```bash
# exemple : compiler un programme adc GPU dedie
cmake -S <prog> -B build -G Ninja -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/nvcc_wrapper" \
  -DADC_INC=<adc_cpp>/include   # + target_compile_definitions(... POPS_HAS_KOKKOS)
```

- **(limitation)** Compiling the **full C++ `ctest` suite under nvcc is still KO**: some test
  files use device lambdas **nested inside other lambdas**, which nvcc refuses
  (`__wrapper__device_stub_..._ParallelFor<...> does not match any template declaration`). This is
  NOT the adc core (the dedicated case above proves it) nor the conda Kokkos (a minimal
  `MDRangePolicy<Rank<2>>` harness passes); it is the pattern on the test side. The `ctest` suite under nvcc
  is not green on any platform: **validate the GPU through dedicated programs** (or the
  `_pops` module / Python cases), as on ROMEO.
- **(perf)** The conda CUDA `kokkos` is compiled for `sm_80`; it runs on an RTX 3090
  (`sm_86`) in compatibility mode (Kokkos warning). For a native `sm_86` binary, recompile
  Kokkos CUDA with `Kokkos_ARCH_AMPERE86=ON`.

## 6. Summary of Windows differences

| Topic | Linux/Mac/ROMEO | Windows (WSL2) |
|---|---|---|
| Build OS | native | WSL2 Ubuntu (no native Windows) |
| Repo location | anywhere | WSL FS (`~/dev`), **not** `/mnt/c` |
| Kokkos conda | assumes Serial-only | **CUDA by default** -> pin CPU variant `hbbfbac7` |
| Build parallelism | machine-dependent | bound `-j 6` (WSL RAM ~15 GB, OOM otherwise) |
| GPU | ROMEO (Spack, GH200) | local WSL2 (env `adc-gpu`, kokkos `sm_80` on `sm_86`) |
| DSL + Kokkos module | same | export `POPS_KOKKOS_ROOT=$CONDA_PREFIX` |
| MPI | same | `OMPI_MCA_btl_smcuda_use_cuda_ipc=0` |
