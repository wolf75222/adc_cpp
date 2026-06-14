# Getting started

This section is the user guide for `adc_cpp`: it starts from zero (what it is,
how to build it) and goes all the way to a complete diocotron simulation,
reproducible end to end. It assumes no knowledge of the C++ core or the DSL.

The recommended path, in order:

- [Overview](overview.md): what `adc_cpp` is, its honest scope, its layers.
- [Installation](installation.md): CMake build of the core, of the Python module, and the pitfall of
  the interpreter (the compiled `.so` is tied to a Python version).
- [Quickstart](quickstart.md): the shortest path from a built module to a running case.
- [First run](first-run.md): the smallest Python example, copyable as is.
- [Repository layout](repository-layout.md): `adc_cpp` (the library, agnostic) vs `adc_cases`
  (the named scenarios: diocotron, euler_poisson...).
- [Checking your backend](backend.md): knowing which parallel backend is running
  (serial by default; Kokkos / MPI are build configs; GPU = ROMEO only).
- [Tutorial A->Z](tutorial.md): the canonical tutorial in 18 steps, from `git clone` to the
  uniform/AMR comparison, with generated figures and GIF.
- [Windows (WSL2)](windows-wsl2.md): a from-scratch guide for a Windows 11 PC, from `wsl --install`
  to an `adc_cases` case, with the differences from the Linux path.
- [Native Windows](windows-native.md): status and roadmap of the native port (without WSL2);
  v1/v2 scope and actual state.

For the design reference (contributors), each page points to the `docs/*.md` documents
of the repository; this section summarizes them for a new user without duplicating them.

```{toctree}
:maxdepth: 1
:hidden:

overview
installation
quickstart
first-run
repository-layout
backend
tutorial
windows-wsl2
windows-native
```
