# Overview


`adc_cpp` targets a single compute stack (mesh + transport + Poisson + AMR), but
this stack can run on six parallel configurations: from single-thread serial
up to multi-GPU Grace-Hopper distributed by MPI. The key design point is that no
operator changes from one configuration to another: all parallelism is confined to
two seams (dispatch seams). The backend is chosen at compile time, through
CMake options.

This page describes each configuration: what it is, the build command, how to
run it, and its validation status (tested in CI or validated manually on
ROMEO). For the test-by-test coverage matrix, see
[BACKEND_COVERAGE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md) (single source of truth); for the
phase-by-phase GPU port, see [GPU_RUNTIME_PORT.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/GPU_RUNTIME_PORT.md).
