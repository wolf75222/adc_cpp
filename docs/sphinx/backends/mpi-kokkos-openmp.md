# MPI + Kokkos OpenMP


**What it is.** Distributed CPU hybrid: MPI between nodes/ranks, Kokkos OpenMP for the
intra-rank multi-thread. This is the "full" CPU mode (all cores of all ranks).

**Build.** Both options at once, on a Kokkos OpenMP:

```bash
cmake -S . -B build-mpi-omp -DCMAKE_BUILD_TYPE=Release \
  -DPOPS_USE_MPI=ON -DPOPS_USE_KOKKOS=ON -DKokkos_ROOT="$KOKKOS_OPENMP_PREFIX"
cmake --build build-mpi-omp -j
```

**Run.**

```bash
OMP_NUM_THREADS=4 OMPI_MCA_rmaps_base_oversubscribe=true \
  ctest --test-dir build-mpi-omp --output-on-failure
```

**Validation: ROMEO-manual (`x64cpu` node).** This combination is not in CI (CI
joins MPI with Kokkos Serial, but never MPI with Kokkos OpenMP). Validated by hand on the `x64cpu` node of
ROMEO: 52/57 rank-invariant runs (bit-identical np=1/2/4, dmax=0 on the
parity/AMR/Krylov observables). Caveat: 3 heavy distributed-MG tests (`mpi_cutcell_multibox`,
`mpi_amr_distributed_coarse`, `condensed_schur_source_stepper`) are too slow at np>1
(exceed 600 s); performance pathology (small tiles + MPI halos, ~5-7x
slowdown), not a deadlock nor a correctness bug. All pass at np=1.
