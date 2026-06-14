# Poisson solvers


The elliptic stage solves `lap(phi) = f` (or a generalization) at each step, and it is
the core of the coupling: `f` depends on the density, and `phi` (via `grad phi`) drives the
drift. The solver is chosen by keyword in `set_poisson`:

```python
import adc

sim = adc.System(n=128, L=1.0, periodic=True)
# ... add_block / add_equation ...
sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="auto")
```

The `solver=` accepts `"geometric_mg"` (default) or `"fft"`. The `rhs=` is
`"charge_density"` (right-hand side `q n`) or `"composite"` (sum of block
contributions). `bc=` is `"auto"`, `"periodic"`, `"dirichlet"`.

## Going further


- Elliptic algorithms (multigrid, FFT, eps/Helmholtz/anisotropic, cut-cell):
  [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md), sections 9 to 12.
- The headers: `include/adc/numerics/elliptic/geometric_mg.hpp`,
  `poisson_fft_solver.hpp`, `poisson_operator.hpp`.
- Conservation properties of the coupled scheme (exact FV mass, momentum, energy, values
  measured by the tests): [CONSERVATION_SUMMARY.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/CONSERVATION_SUMMARY.md).
