# Run with MPI

This guide builds `adc_cpp` with the distributed comm seam and runs the tests across
several MPI ranks. It assumes you can already build the reference Kokkos Serial backend
and that MPI and a Kokkos install are available on your machine. For the full backend
model, the validation status of each configuration and the coverage matrix, see the
[parallel backends page](../backends/index.md).

MPI distributes subdomains across ranks through the `comm.hpp` seam; the local compute
stays on the Kokkos Serial execution space, one process per rank. Enable it with the
`POPS_USE_MPI` CMake option.

## Build with MPI

This uses the same Kokkos Serial install as the reference build, plus the MPI seam.

1. Configure the build with both `POPS_USE_KOKKOS` and `POPS_USE_MPI` turned on.
   Replace `KOKKOS_PREFIX` with the path to your Kokkos Serial install.

   ```bash
   cmake -S . -B build-mpi -DCMAKE_BUILD_TYPE=Release -DADC_USE_KOKKOS=ON -DADC_USE_MPI=ON -DKokkos_ROOT=KOKKOS_PREFIX
   ```

2. Compile.

   ```bash
   cmake --build build-mpi -j
   ```

## Run the distributed tests

The MPI test targets each replay the run at np=1, np=2 and np=4 under `mpirun`, and
check that the observables (parity, AMR, Krylov, mass) are bit-identical across the
number of ranks.

1. Run the test suite. On a small machine, allow oversubscribe so that np=4 fits on
   fewer than four physical cores.

   ```bash
   OMPI_MCA_rmaps_base_oversubscribe=true ctest --test-dir build-mpi --output-on-failure
   ```

The non-MPI tests run at np=1 in this build (MPI-linked, single process); the MPI
block checks rank invariance at np=1, np=2 and np=4.

## Run on multiple CPU threads per rank

To use all cores of every rank, build the MPI plus Kokkos OpenMP hybrid against a
Kokkos OpenMP install. Replace `KOKKOS_OPENMP_PREFIX` with the path to that install.

```bash
cmake -S . -B build-mpi-omp -DCMAKE_BUILD_TYPE=Release -DADC_USE_MPI=ON -DADC_USE_KOKKOS=ON -DKokkos_ROOT=KOKKOS_OPENMP_PREFIX
```

Bound the thread count per rank with `OMP_NUM_THREADS` when you run the tests.

```bash
OMP_NUM_THREADS=4 OMPI_MCA_rmaps_base_oversubscribe=true ctest --test-dir build-mpi-omp --output-on-failure
```

## Multi-rank from Python

The `adc` Python module is MPI-aware when `_pops` is built with `POPS_USE_MPI=ON`
(the `mpi` CMake preset, which writes to `build-mpi` and relies on `POPS_USE_KOKKOS`
defaulting to ON). Run a script across ranks the same way as a C++ binary, by
launching the interpreter under `mpirun`.

```bash
cmake --preset mpi && cmake --build --preset mpi   # or the ad-hoc cmake above, conda-free
mpirun -np 4 python your_script.py
```

`pops.my_rank()` and `pops.n_ranks()` report the rank and the rank count. The global
accessors (`sim.state_global(name)`, `sim.density_global(name)`,
`sim.potential_global()`) and `sim.write(...)` are collective: they `all_reduce`/gather
across ranks, so **every** rank must call them or the run deadlocks. `sim.write` gathers
the field collectively and then writes ONE file on rank 0 (the Cartesian `System` is
mono-box); the other ranks return the path without I/O.

```python
import numpy as np
import pops

# Built and stepped identically on every rank.
sim = pops.System(n=96, L=1.0, periodic=True)
sim.add_block(
    "ne",
    model=pops.Model(
        state=pops.Scalar(),
        transport=pops.ExB(B0=1.0),
        source=pops.NoSource(),
        elliptic=pops.BackgroundDensity(alpha=1.0, n0=1.0),
    ),
)
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
sim.set_density("ne", np.ascontiguousarray(np.ones((96, 96))))

for _ in range(20):
    sim.step_cfl(0.4)

# Collective: every rank calls it; rank 0 writes the single .vti file.
sim.write("out/state", format="vtk")
if pops.my_rank() == 0:
    print("ranks =", pops.n_ranks(), "t =", sim.time())
```

For an advanced multi-box output, `sim.write(format="hdf5", parallel=True)` writes
per-rank hyperslabs into one file (needs h5py built with MPI plus mpi4py); on the
mono-box Cartesian `System` it falls back to a rank-0 write.

## Plot the result

Open the `.vti` output in ParaView following
[visualize with ParaView](visualize-with-paraview.md).

## Next steps

- To run multi-GPU on ROMEO, see the [parallel backends page](../backends/index.md).
- A multi-thread (Kokkos OpenMP) Python module is available via the `python-parallel`
  CMake preset, with the thread count set by `pops.set_threads(n)` right after `import adc`;
  there is no nvcc/CUDA Python module (GPU is C++-only). See the
  [installation page](../getting-started/installation.md).
- The multi-rank Python API above is real and works under `mpirun`. What is missing is
  Python MPI *test* coverage: no Python test runs under MPI in CI, so the distributed
  Python surface is exercised by hand, while the distributed numerics are validated through
  the C++/ctest path. See
  [BACKEND_COVERAGE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md).
