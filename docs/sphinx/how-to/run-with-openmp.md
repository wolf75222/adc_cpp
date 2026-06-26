# Run with OpenMP

Build the `adc_cpp` C++ facade against a Kokkos OpenMP execution space and run the
compiled tests multi-threaded on CPU. Use this when the Kokkos Serial backend is too slow on a
single thread and you want to use the cores of one machine.

The backend is fixed at compile time, not chosen at runtime, so changing it means reconfiguring
and recompiling. This recipe assumes you already build `adc_cpp` with the Kokkos Serial backend;
see [Installation](../getting-started/installation.md) if you do not. For the full list of
parallel configurations, see the [parallel backends](../backends/index.md) page.

A multi-thread (Kokkos OpenMP) Python module is available too: build it with the
`python-parallel` CMake preset, then set the thread count from Python with `pops.set_threads(n)`
called right after `import pops`. The remaining caveats are real: there is no nvcc/CUDA Python
module (GPU runs are C++-only), and the Python `pops` module does not exercise the MPI code paths
(MPI is validated through the C++/ctest path).

## Steps

1. Install a Kokkos with `Kokkos_ENABLE_OPENMP=ON`. Point a path variable at that install. Replace
   `KOKKOS_OPENMP_PREFIX` with the absolute path to the Kokkos OpenMP install prefix.

   ```bash
   export KOKKOS_OPENMP_PREFIX=/path/to/kokkos-openmp
   ```

2. Configure the build against that Kokkos. The `-DPOPS_USE_KOKKOS=ON` flag is the same as for the
   Serial build; only the Kokkos install pointed at by `-DKokkos_ROOT` changes.

   ```bash
   cmake -S . -B build-kokkos-omp -DCMAKE_BUILD_TYPE=Release -DPOPS_USE_KOKKOS=ON -DKokkos_ROOT="$KOKKOS_OPENMP_PREFIX"
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

## Threading from Python (set_threads)

The Python module sets its thread count with `pops.set_threads(n)`, which writes `OMP_NUM_THREADS`
and `KOKKOS_NUM_THREADS` for you instead of exporting them in the shell. It is pure Python: no
C++ call, so the value must be in place before Kokkos initializes.

```python
import pops

pops.set_threads(8)          # or pops.set_threads() for all cores (os.cpu_count())
print(pops.parallel_info())  # {'has_kokkos': True, 'omp_num_threads': '8', 'first_system_built': False}

sim = pops.System(n=256)     # Kokkos initializes here and reads the thread count once
```

Three rules:

1. Use a module built against a Kokkos OpenMP execution space (`python-parallel` preset). The
   conda-forge Kokkos is often Serial-only; see [Kokkos OpenMP](../backends/kokkos-openmp.md) and
   the Installation [Threads](#threads) section.
2. Call `set_threads` right after `import pops` and before the first `System` or `AmrSystem`. Kokkos
   reads the thread count once at that first object, so a later call cannot change it.
3. A Serial-only module or a late call only emits a `RuntimeWarning` and is ignored; it never
   raises. Confirm the state with `pops.has_kokkos()`, `pops.parallel_info()`, or `pops.doctor()`.

The floating-point note above applies to the Python path too: the per-tile sum reduction is
deterministic for a fixed thread count but not bit-identical to the serial sum; the max is exact.
`POPS_FOREACH_SERIAL_THRESHOLD` keeps small grids on a serial loop, so small problems may not speed
up regardless of the thread count.

## Next steps

- To add distributed ranks on top of OpenMP, combine `-DPOPS_USE_MPI=ON` with the same OpenMP Kokkos
  install, described on the [parallel backends](../backends/index.md) page.
- To check which execution space a build actually runs, read
  [Checking your backend](../getting-started/backend.md).
