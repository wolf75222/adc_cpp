# Quickstart

```python
import numpy as np
import adc
```

Chaque solveur est piloté par une config, avance par pas, et rend ses champs en numpy
`(n, n)`. Le backend (série / OpenMP / Kokkos) est celui avec lequel `libadc` a été
compilée.

## Diocotron (dérive E x B)

```python
cfg = adc.DiocotronConfig()
cfg.n = 128
cfg.ic = adc.DiocotronIC.Band       # Smooth | Band | Ring
sim = adc.DiocotronSolver(cfg)

m0 = sim.mass()
for _ in range(500):
    sim.step_cfl(0.4)               # CFL sur la dérive E x B
rho = sim.density()                 # ndarray (128, 128)
phi = sim.potential()
print("dérive masse :", abs(sim.mass() - m0))   # ~ arrondi machine
```

## Euler-Poisson auto-gravitant

```python
cfg = adc.EulerPoissonConfig(); cfg.n = 64; cfg.use_fft = True
es = adc.EulerPoissonSolver(cfg)
for _ in range(5): es.step(0.004)
print(es.mass(), es.energy(), es.total_momentum(0))
```

## Deux-fluides isotherme AP (régime raide)

```python
cfg = adc.TwoFluidAPConfig()
cfg.n = 64; cfg.omega_pe = 1e3; cfg.omega_pi = 20.0
cfg.stabilize = True                # AP : False explose
ts = adc.TwoFluidAPSolver(cfg)
ts.advance(5.0 / 1e3, 200)          # dt*omega_pe = 5 : l'explicite exploserait
print("borne AP :", ts.max_dev(), " quasi-neutre :", ts.max_charge())
```

## Invariant à vérifier

```python
m0 = sim.mass()
for _ in range(100): sim.step_cfl(0.4)
assert abs(sim.mass() - m0) < 1e-9   # conservation à l'arrondi
```

Référence complète de l'API : [api](api.md). Recettes complètes (figures, AMR) :
[examples](examples.md). Détail théorique :
[ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md).
