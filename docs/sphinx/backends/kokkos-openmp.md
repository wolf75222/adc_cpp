# Kokkos OpenMP


**What it is.** Same Kokkos backend, `OpenMP` execution space: multi-thread parallelism
on CPU. `for_each_cell` becomes a multi-thread `parallel_for` (with a serial-switch threshold
for the small V-cycle grids, cf. `POPS_FOREACH_SERIAL_THRESHOLD`).

**Build.** Kokkos installed with `Kokkos_ENABLE_OPENMP=ON`. The `-DPOPS_USE_KOKKOS=ON` on the
`adc_cpp` side is identical to config 1; it is the Kokkos install that changes:

```bash
cmake -S . -B build-kokkos-omp -DCMAKE_BUILD_TYPE=Release \
  -DPOPS_USE_KOKKOS=ON -DKokkos_ROOT="$KOKKOS_OPENMP_PREFIX"
cmake --build build-kokkos-omp -j
```

**Run.** Bound the number of threads on small machines:

```bash
OMP_NUM_THREADS=4 ctest --test-dir build-kokkos-omp --output-on-failure
```

**Validation: CI (`ci-full` job).** Job `kokkos-openmp`
(`ubuntu-latest / Kokkos (OpenMP)`), Kokkos 4.4.01 OpenMP, `OMP_NUM_THREADS=2`. Full mode
only (unlike the Serial gate, which runs on every PR). Status `ci-full`,
91/91 ctest, 0 failure.

> **FP note.** The sum reduction (`Kokkos::Sum`) reassociates floating-point addition per tile:
> deterministic/idempotent (same data + same Kokkos space -> same bits) but not
> bit-identical to a hand-written lexicographic sum. Since there is only one
> Kokkos path, this holds for all spaces (Serial, OpenMP, Cuda). The max reduction
> (`Kokkos::Max`) is exact everywhere. Detail in the header of
> [`for_each.hpp`](https://github.com/wolf75222/adc_cpp/blob/master/include/pops/mesh/execution/for_each.hpp).
