# Quickstart

```python
import numpy as np
import adc
```

On compose un système bloc par bloc avec `adc.System` : un modèle par bloc, un Poisson de
système partagé, des conditions initiales en numpy. Le système avance par pas et rend ses
champs en numpy `(n, n)`. Le backend (série / OpenMP / Kokkos) est celui avec lequel
`libadc` a été compilée.

## Diocotron (dérive E x B)

Un bloc `diocotron` dans un domaine non périodique avec paroi conductrice circulaire.

```python
def ring_density(n, L=1.0):
    coord = (np.arange(n) + 0.5) / n * L
    xx, yy = np.meshgrid(coord, coord, indexing="xy")
    r = np.hypot(xx - 0.5 * L, yy - 0.5 * L)
    ne = np.full((n, n), 1e-3)
    ne[(r > 0.15) & (r < 0.20)] = 1.0
    return ne

sim = adc.System(n=128, L=1.0, B0=1.0, alpha=1.0, n_i0=0.0, periodic=False)
sim.add_block("ne", model="diocotron", charge=1.0,
              spatial=adc.Spatial(minmod=True), time=adc.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet",
                wall="circle", wall_radius=0.40)
sim.set_density("ne", ring_density(128))

m0 = sim.mass("ne")
for _ in range(500):
    sim.step_cfl(0.4)               # CFL sur la dérive E x B
rho = sim.density("ne")             # ndarray (128, 128)
phi = sim.potential()
print("dérive masse :", abs(sim.mass("ne") - m0))   # ~ arrondi machine
```

## Euler-Poisson auto-gravitant

Un bloc `euler_poisson` (4 variables) en domaine périodique. La charge encode le signe du
couplage : `+1.0` attractif (auto-gravité), `-1.0` répulsif (Langmuir).

```python
n = 64
x = (np.arange(n) + 0.5) / n
rho = 1.0 + 0.01 * np.cos(2.0 * np.pi * x)[:, None] * np.ones((n, n))

sim = adc.System(n=n, gamma=1.4, four_pi_G=1.0, rho0=1.0, periodic=True)
sim.add_block("gas", model="euler_poisson", charge=+1.0,
              spatial=adc.Spatial(vanleer=True, flux="hllc"), time=adc.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
sim.set_density("gas", rho)

m0 = sim.mass("gas")
sim.advance(0.004, 5)
U = sim.get_state("gas")            # (4, n, n) = [rho, rho*u, rho*v, E]
print(sim.mass("gas"), U[3].sum(), U[1].sum())   # masse, énergie, impulsion p_x
```

## Composition multi-espèces

Un schéma numérique différent par bloc, couplés par un seul Poisson de système.

```python
n = 48
x = (np.arange(n) + 0.5) / n
ne = 1.0 + 0.02 * np.cos(2.0 * np.pi * x)[None, :] * np.ones((n, n))

sim = adc.System(n=n, gamma=1.4, cs2=0.5, periodic=True)
sim.add_block("electrons", model="electron_euler", charge=-1.0,
              spatial=adc.Spatial(vanleer=True, flux="hllc"),
              time=adc.IMEX(substeps=10))     # raide -> source implicite, sous-cyclé
sim.add_block("ions", model="ion_isothermal", charge=+1.0,
              spatial=adc.Spatial(minmod=True), time=adc.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
sim.set_density("electrons", ne)
sim.set_density("ions", np.ones((n, n)))

sim.advance(0.001, 8)
print("blocs :", sim.block_names(), "| |phi|_max :", np.abs(sim.potential()).max())
```

## Deux-fluides isotherme AP (régime raide)

Solveur spécialisé `adc.TwoFluidAP` : un schéma asymptotic-preserving stable quand
`dt*omega_pe >> 1`, là où un explicite exploserait.

```python
cfg = adc.TwoFluidAPConfig()
cfg.n = 64; cfg.omega_pe = 1e3; cfg.omega_pi = 20.0
ts = adc.TwoFluidAP(cfg)
ts.advance(5.0 / 1e3, 200)          # dt*omega_pe = 5 : l'explicite exploserait
print("borne AP :", ts.max_dev(), " quasi-neutre :", ts.max_charge())
```

## Invariant à vérifier

```python
m0 = sim.mass("ions")
for _ in range(100):
    sim.step_cfl(0.4)
assert abs(sim.mass("ions") - m0) < 1e-9   # conservation à l'arrondi
```

Référence complète de l'API : [api](api.md). Recettes complètes (figures, AMR) :
[examples](examples.md). Détail théorique :
[ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md).
