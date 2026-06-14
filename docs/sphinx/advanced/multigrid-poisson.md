# Multigrid Poisson


`GeometricMG` is the default solver: a multigrid V-cycle with a red-black
Gauss-Seidel smoother (the coloring makes each sweep independent of the data,
so parallelizable and device-clean). The cycle smooths `nu1` times, restricts the residual onto
a grid twice as coarse (`average_down`), recurses, prolongs the correction,
smooths again `nu2` times. Cost O(N), convergence nearly independent of the mesh. The coarsening
stops cleanly as soon as a box no longer divides exactly (safeguard
`coarsen(2).refine(2) == b`), which avoids degenerate hierarchies under AMR / multi-box.

The same multigrid operator covers three generalizations of the Laplacian, all opt-in
and bit-identical to the historical path as long as they are not activated. On the C++ side
(`GeometricMG`, `numerics/elliptic/geometric_mg.hpp`):

- `set_epsilon(eps)`, variable permittivity `div(eps(x) grad phi) = f` (harmonic
  mean at faces);
- `set_reaction(kappa)`, screened / Helmholtz operator `div(eps grad phi) - kappa phi = f`
  (Debye screening, `kappa = 1 / lambda_D^2`);
- `set_epsilon_anisotropic(eps_x, eps_y)`, diagonal tensor medium `div(diag(eps_x, eps_y) grad phi) = f`.

These three settings are composable; `eps_x == eps_y` gives back the isotropic case, not calling
`set_reaction` gives back pure Poisson. On the Python side, they are exposed by NumPy field
at the `System` level (the coefficients live in the device `for_each_cell` of the smoother):

```python
import numpy as np

eps   = np.ones((n, n))            # permittivite variable (set_epsilon C++)
kappa = np.zeros((n, n))           # terme de reaction kappa >= 0 (Helmholtz/ecrante)

sim.set_epsilon_field(eps)                       # div(eps grad phi) = f
sim.set_reaction_field(kappa)                    # - kappa phi
sim.set_epsilon_anisotropic_field(eps_x, eps_y)  # diag(eps_x, eps_y)
```
