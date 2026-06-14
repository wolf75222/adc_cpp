# Run with OpenMP

Build the `adc_cpp` C++ facade against a Kokkos OpenMP execution space and run the
compiled tests multi-threaded on CPU. Use this when the Kokkos Serial backend is too slow on a
single thread and you want to use the cores of one machine.

The backend is fixed at compile time, not chosen at runtime, so changing it means reconfiguring
and recompiling. This recipe assumes you already build `adc_cpp` with the Kokkos Serial backend;
see [Installation](../getting-started/installation.md) if you do not. For the full list of
parallel configurations, see the [parallel backends](../backends/index.md) page.

The Python `adc` module stays serial. The `_adc` extension is built in Kokkos Serial only, so a
script that does `import adc` runs sequentially whatever the hardware. Multi-thread runs are driven
from the C++ facade (the tests and executables), not from Python.

## Steps

1. Install a Kokkos with `Kokkos_ENABLE_OPENMP=ON`. Point a path variable at that install. Replace
   `KOKKOS_OPENMP_PREFIX` with the absolute path to the Kokkos OpenMP install prefix.

   ```bash
   export KOKKOS_OPENMP_PREFIX=/path/to/kokkos-openmp
   ```

2. Configure the build against that Kokkos. The `-DADC_USE_KOKKOS=ON` flag is the same as for the
   Serial build; only the Kokkos install pointed at by `-DKokkos_ROOT` changes.

   ```bash
   cmake -S . -B build-kokkos-omp -DCMAKE_BUILD_TYPE=Release -DADC_USE_KOKKOS=ON -DKokkos_ROOT="$KOKKOS_OPENMP_PREFIX"
   ```

3. Compile the facade and the tests.

   ```bash
   cmake --build build-kokkos-omp -j
   ```

4. Run the tests, bounding the thread count with `OMP_NUM_THREADS`. Set it to the number of cores you
   want to use.

   ```bash
   OMP_NUM_THREADS=4 ctest --test-dir build-kokkos-omp --output-on-failure
   ```

With the OpenMP execution space, `for_each_cell` becomes a multi-thread `parallel_for`. The sum
reduction reassociates floating-point addition per tile, so a multi-thread total is deterministic
for a fixed thread count but not bit-identical to a lexicographic sum; the max reduction stays exact.

## Next steps

- To add distributed ranks on top of OpenMP, combine `-DADC_USE_MPI=ON` with the same OpenMP Kokkos
  install, described on the [parallel backends](../backends/index.md) page.
- To check which execution space a build actually runs, read
  [Checking your backend](../getting-started/backend.md).
