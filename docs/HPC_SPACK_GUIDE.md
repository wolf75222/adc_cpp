# Running adc_cpp on a supercomputer (Spack)

Generic guide to compile and run adc_cpp (header-only C++23 lib +
Python module `_pops`) on any cluster via Spack, at maximum
performance. Optional backends: Kokkos (CPU OpenMP / GPU CUDA),
MPI (OpenMPI), parallel HDF5.

The ROMEO-specific counterpart is `Tools/machines/romeo/romeo_adc.profile.example`:
this guide is its generic version for other clusters, and can serve
as a template to copy into a site profile.

## 0. Prerequisites

- CMake >= 3.21 and Ninja (can come from Spack, see section 2).
- A C++23 compiler (gcc >= 13 recommended, see section 3).
- Access to a fast scratch for the DSL `.so` cache (section 5).

## 1. Choose: site Spack or personal bootstrap

Many centers already provide Spack. Check first:

```sh
module avail spack          # ou : which spack
module load spack           # si un module existe
spack --version
```

If nothing is available, personal bootstrap without root rights:

```sh
git clone --depth=2 https://github.com/spack/spack.git ~/spack
. ~/spack/share/spack/setup-env.sh     # a mettre dans ~/.bashrc
spack bootstrap now                     # construit les dependances internes
```

Prefer the site Spack when it exists: it already knows the network
fabrics, the GPU drivers and the vendor compilers. The personal bootstrap is the
fallback when the site does not provide one or the version is too old.

## 2. Install the stack

Choose the compiler first (section 3), then:

```sh
# CPU (OpenMP)
spack install kokkos +openmp

# GPU (CUDA) : nvcc_wrapper de Kokkos comme compilateur device
#   cuda_arch=80 -> A100, 90 -> H100 / GH200
spack install kokkos +cuda +wrapper cuda_arch=90

# MPI : prendre les fabrics du site si possible (cf. section 8)
spack install openmpi                  # fabrics=auto par defaut
# exemple oriente reseau du site :
# spack install openmpi fabrics=ucx +cuda    # a verifier sur votre site

# Outils de build
spack install cmake ninja

# HDF5 parallele (optionnel, sorties paralleles)
spack install hdf5 +mpi
```

Find the installed prefixes (useful for CMake / the env vars):

```sh
spack location -i kokkos
spack location -i openmpi
spack location -i hdf5
```

`spack location -i <spec>` prints the install prefix of an already
installed package; it is used for `-DKokkos_ROOT`, `POPS_KOKKOS_ROOT`, etc.

## 3. Target the microarchitecture (perf)

List what Spack recognizes:

```sh
spack arch                       # arch courante detectee
spack arch --known-targets       # toutes les cibles connues
```

Target the micro-arch of the compute nodes (not the login one if they differ):

```sh
spack install kokkos +openmp target=zen4            # AMD Genoa
spack install kokkos +openmp target=sapphirerapids  # Intel SPR
spack install kokkos +openmp target=neoverse_v2     # ARM Grace (GH200)
```

Recent compiler (gcc >= 13 for full C++23):

```sh
spack compiler find                 # enregistre les compilos deja presents
spack install gcc@13                # si le site n'a pas de gcc recent
spack load gcc@13
spack compiler find                 # re-detecte gcc@13 fraichement installe
spack compiler list
```

Then pin the compiler on each spec with `%`:

```sh
spack install kokkos +openmp target=zen4 %gcc@13
```

## 4. Compile adc_cpp

Configure on the login node if CMake/Ninja are missing on the compute nodes;
the binaries will still be launched by sbatch. The POPS_* options are also
accepted as environment variables.

Core + tests, CPU:

```sh
export KOKKOS_ROOT=$(spack location -i kokkos)
cmake -S . -B build-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DADC_USE_KOKKOS=ON -DKokkos_ROOT=$KOKKOS_ROOT \
  -DADC_USE_MPI=ON \
  -DADC_USE_HDF5=ON
cmake --build build-cpu -j
ctest --test-dir build-cpu
```

GPU: compile with Kokkos's `nvcc_wrapper` as the C++ compiler:

```sh
export KOKKOS_ROOT=$(spack location -i kokkos)   # build +cuda +wrapper
cmake -S . -B build-gpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=$KOKKOS_ROOT/bin/nvcc_wrapper \
  -DADC_USE_KOKKOS=ON -DKokkos_ROOT=$KOKKOS_ROOT \
  -DADC_USE_MPI=ON
cmake --build build-gpu -j
```

Python module `_pops` (scikit-build-core via pip):

```sh
export POPS_USE_KOKKOS=ON
export Kokkos_ROOT=$(spack location -i kokkos)
export POPS_USE_MPI=ON
# GPU : ajouter
# export CMAKE_CXX_COMPILER=$Kokkos_ROOT/bin/nvcc_wrapper
pip install .                       # dans un venv ou conda env
```

The build compiler is baked into `_pops` (the DSL calls it again at runtime,
see section 5): it must remain accessible on the nodes. Keep the compiler's
`spack load` (or its module) in the sbatch script.

## 5. DSL runtime variables

The module compiles `.so` files at runtime. To export on the nodes:

```sh
# parite loader / module : meme Kokkos que celui du build
export POPS_KOKKOS_ROOT=$(spack location -i kokkos)

# cache des .so sur le scratch (rapide, partage entre jobs si meme micro-archi)
export POPS_CACHE_DIR=$SCRATCH/pops_dsl_cache

# flags d'optimisation des .so DSL (defaut "-O3 -DNDEBUG")
export POPS_DSL_OPTFLAGS="-O3 -DNDEBUG"
```

`-march=native` is possible in `POPS_DSL_OPTFLAGS` to gain a few %,
but the `.so` cache is then NOT shareable across microarchitectures:
use a separate `POPS_CACHE_DIR` per node type, or do without it if the
nodes are heterogeneous.

## 6. Launch at max perf (sbatch)

### CPU (OpenMP, 1 MPI rank per NUMA socket)

```sh
#!/bin/sh
#SBATCH --job-name=adc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2          # 1 rang par socket NUMA
#SBATCH --cpus-per-task=48           # coeurs par socket
#SBATCH --time=01:00:00

. ~/spack/share/spack/setup-env.sh   # ou module load spack
spack load gcc@13 kokkos openmpi     # meme stack que le build

export POPS_KOKKOS_ROOT=$(spack location -i kokkos)
export POPS_CACHE_DIR=$SCRATCH/pops_dsl_cache
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

srun --cpu-bind=cores python run_case.py
```

In Python, `pops.set_threads(n)` sets `OMP_NUM_THREADS` and `KOKKOS_NUM_THREADS`
consistently; the NUMA placement (`OMP_PROC_BIND` / `OMP_PLACES`)
still has to be set in the environment as above.

### GPU (CUDA, 1 MPI rank per GPU)

```sh
#!/bin/sh
#SBATCH --job-name=adc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # 1 rang par GPU
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00

. ~/spack/share/spack/setup-env.sh
spack load gcc@13 kokkos openmpi     # kokkos +cuda +wrapper

export POPS_KOKKOS_ROOT=$(spack location -i kokkos)
export POPS_CACHE_DIR=$SCRATCH/pops_dsl_cache
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Piege OpenMPI + memoire UVM/managed : le transport CUDA-IPC (btl smcuda)
# deadlock entre GPU isoles par cgroup (--gpus-per-task=1). Quick-fix :
export OMPI_MCA_btl_smcuda_use_cuda_ipc=0

srun --cpu-bind=cores python run_case.py
```

1 MPI rank per GPU is the rule; do not oversubscribe. If the job hangs in
the multi-box halos at np >= 2 on GPU, the quick-fix above works around the
CUDA-IPC routing that is impossible between cgroup-isolated GPUs (the proper fix is in
the code since PR #254; the quick-fix is still useful on an old module).

## 7. Reproducibility: a Spack environment per machine

Freeze the stack in a versioned Spack environment (one folder per cluster):

```sh
spack env create adc-clusterX        # cree l'env
spack env activate adc-clusterX
spack add kokkos +cuda +wrapper cuda_arch=90 target=neoverse_v2 %gcc@13
spack add openmpi cmake ninja hdf5 +mpi
spack concretize                     # ecrit spack.lock
spack install
```

Or write a `spack.yaml` directly then `spack env create adc-clusterX spack.yaml`:

```yaml
spack:
  specs:
    - kokkos +cuda +wrapper cuda_arch=90 target=neoverse_v2
    - openmpi
    - cmake
    - ninja
    - hdf5 +mpi
  packages:
    all:
      compiler: [gcc@13]
      providers:
        mpi: [openmpi]
  view: true            # vue filesystem, ou un chemin : view: /chemin/vue
```

Commit `spack.yaml` AND `spack.lock` per machine (e.g. under
`Tools/machines/<cluster>/`): `spack.lock` is the exact concretized state,
replayable identically.

## 8. Adapt to your cluster

Inspect the site before fixing the placement:

```sh
sinfo -o "%P %N %c %G %m"      # partitions, coeurs, GPU, RAM par noeud
module avail                    # modules du site (compilateurs, MPI, CUDA)
spack arch --known-targets      # cibles micro-archi reconnues
```

- Check `--cpus-per-task` against the number of cores per socket from `sinfo`,
  and `--gpus-per-task` against the GRES column (`%G`).
- Prefer the site MPI / CUDA (via `module load` + Spack `externals`)
  when the center optimizes them for the network fabric: it is often faster
  and more stable than rebuilding everything. Declare an external package
  in `packages:` of the `spack.yaml` if needed.   # a verifier sur votre site
- When in doubt about a site-specific Spack command, consult the center's
  documentation: the syntax of variants and targets may be constrained by
  the external packages already declared.
