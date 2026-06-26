# Kokkos Serial


**What it is.** The reference build: Kokkos Serial (single-thread CPU via
`Kokkos::parallel_for`), without MPI. `comm.hpp` answers rank 0 / 1 rank. It is the oracle: all the
other backends are validated bit-for-bit (or to rounding) against it. Serial goes through
Kokkos Serial, not through a hand-written C++ loop. This is the job that had caught an
init regression (allocating a `Fab` before the lazy Kokkos init); the Kokkos
Serial gate now plays that role on EVERY PR.

**Build.** You need a Kokkos installed with `Kokkos_ENABLE_SERIAL=ON`, then:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DADC_USE_KOKKOS=ON -DKokkos_ROOT="$KOKKOS_PREFIX"
cmake --build build -j
```

**Run.**

```bash
ctest --test-dir build --output-on-failure
```

**Validation: CI (required gate of every PR).** This is the `build-and-test` job
(`ubuntu-latest`, Kokkos Serial), triggered on every `pull_request`. Covers the 109 non-MPI
ctest targets plus the 60 Python tests (built via `-DADC_BUILD_PYTHON=ON`, `_pops`
Kokkos Serial module). Status `ci-fast` in the matrix.
