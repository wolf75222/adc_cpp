# Advanced topics

This page gathers the features that go beyond the basic diocotron loop
(E x B transport + Poisson): generalized elliptic solvers, coupled inter-species
sources, Schur-condensed source stage, polar / disc geometry, extending the
core in C++, and performance profiling.

Each section summarizes the essentials for a user and points to the contributor
reference (`docs/*.md`) for the algorithmic detail and the validation proofs.
The APIs shown here are checked against the repository code (bindings, tests, headers).

```{toctree}
:maxdepth: 1

poisson-solvers
multigrid-poisson
spectral-fft-poisson
coupled-inter-species-sources
condensed-schur-source-stage
compilation-production-aot-prototype
performance-and-profiling
polar-disc-geometry
cpp-extension-native-brick
```
