# MPI + Kokkos Serial


**What it is.** Distributed build with the Kokkos Serial execution space: `comm.hpp` goes
through `MPI_Comm_rank/size` + collectives on `MPI_COMM_WORLD`. The domain is split into boxes
distributed across ranks; the halos exchange via cross-rank `fill_boundary`, the reflux/mass
via `all_reduce_*`. Single-thread CPU per rank.

**Build.** Kokkos installed with `Kokkos_ENABLE_SERIAL=ON`, plus the MPI seam:

```bash
cmake -S . -B build-mpi -DCMAKE_BUILD_TYPE=Release \
  -DADC_USE_KOKKOS=ON -DADC_USE_MPI=ON -DKokkos_ROOT="$KOKKOS_PREFIX"
cmake --build build-mpi -j
```

**Run.** The MPI targets each replay np=1/2/4 under `mpirun`. For `-np 4` on a small
machine, allow oversubscribe:

```bash
OMPI_MCA_rmaps_base_oversubscribe=true \
  ctest --test-dir build-mpi --output-on-failure
```

**Validation: CI (`ci-full` job).** Job `mpi` (`ubuntu-latest / MPI`, OpenMPI). Full mode
only. Checks invariance to the number of ranks: the observables (parity, AMR,
Krylov, mass) are bit-identical at np=1/2/4. Status `ci-full` on the ~21 entries of the
`POPS_HAS_MPI` block; the non-MPI tests run at np=1 in this build (MPI-linked, single-process).
