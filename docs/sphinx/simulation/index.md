# Simulation

`pops.System` is the composition facade of the solver: you declare one block per physical model
(one species), you share one system Poisson across all blocks, you set initial conditions
in numpy, then you advance the whole thing step by step. The core names no scenario (diocotron,
Euler-Poisson... live on the `adc_cases` side); here you assemble generic bricks.

This page walks through the simulation mechanics on the Python side: composing a system, coupling
several species, choosing the spatial scheme and the time policy, handling the multirate,
initializing and reading the fields. The theoretical detail of the numerical methods is in
[ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md); the layered architecture (dispatch seam, frontier
lib/application) in [ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md).

```{toctree}
:maxdepth: 1

system
spatial-schemes
time-schemes
substeps-stride-multirate
initial-conditions
outputs-diagnostics
```
