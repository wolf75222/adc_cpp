# Backend seams


There are no "three layers" stacked. The architecture is MPI + Kokkos:

- **MPI** distributes subdomains across ranks (one GPU per rank in GPU mode). Everything goes
  through `my_rank()` / `n_ranks()` / `all_reduce_*` from
  [`include/adc/parallel/comm.hpp`](https://github.com/wolf75222/adc_cpp/blob/master/include/adc/parallel/comm.hpp). Without
  `ADC_HAS_MPI`, these functions return rank 0 / 1 rank: the code is serial by
  construction.
- **Kokkos** parallelizes the local compute and abstracts the hardware through its `ExecutionSpace`:
  `Cuda` backend for NVIDIA GPUs, `Serial`/`OpenMP` for CPU. Everything goes through `for_each_cell`
  (and `for_each_cell_reduce_*`) from
  [`include/adc/mesh/execution/for_each.hpp`](https://github.com/wolf75222/adc_cpp/blob/master/include/adc/mesh/execution/for_each.hpp), which switches
  CPU <-> GPU at compile time without touching the call sites.

No CUDA kernel is written by hand: the same `.cpp` targets CPU and GPU depending on the Kokkos
backend active at compile time. `nvcc_wrapper` is only the compiler required by the Kokkos
Cuda backend.

> **The Python `adc` module is serial by default.** The `_adc` extension (pybind11) is
> built in CI only in Kokkos Serial (without MPI). No Python test exercises the Kokkos
> OpenMP, Cuda or MPI paths. Multi-thread, GPU and distributed are driven from the
> C++ facade (`System` / `AmrSystem`), not from Python.
