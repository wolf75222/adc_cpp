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

Une dérive E x B (briques `ExB` + `BackgroundDensity`) dans un domaine non périodique avec
paroi conductrice circulaire.

```python
def ring_density(n, L=1.0):
    coord = (np.arange(n) + 0.5) / n * L
    xx, yy = np.meshgrid(coord, coord, indexing="xy")
    r = np.hypot(xx - 0.5 * L, yy - 0.5 * L)
    ne = np.full((n, n), 1e-3)
    ne[(r > 0.15) & (r < 0.20)] = 1.0
    return ne

sim = adc.System(n=128, L=1.0, periodic=False)     # config = maillage seul
model = adc.Model(state=adc.Scalar(), transport=adc.ExB(B0=1.0),
                  source=adc.NoSource(), elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0))
sim.add_block("ne", model=model, spatial=adc.Spatial(minmod=True), time=adc.Explicit())
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

## Euler compressible couplé à un champ auto-consistant

Un fluide compressible (4 variables) en domaine périodique. Le signe du couplage
(`GravityCoupling(sign=...)`) : `+1.0` attractif (auto-gravité), `-1.0` répulsif (Langmuir).

```python
n = 64
x = (np.arange(n) + 0.5) / n
rho = 1.0 + 0.01 * np.cos(2.0 * np.pi * x)[:, None] * np.ones((n, n))

sim = adc.System(n=n, L=1.0, periodic=True)
model = adc.Model(state=adc.FluidState("compressible", gamma=1.4),
                  transport=adc.CompressibleFlux(), source=adc.GravityForce(),
                  elliptic=adc.GravityCoupling(sign=+1.0, four_pi_G=1.0, rho0=1.0))
sim.add_block("gas", model=model,
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

sim = adc.System(n=n, L=1.0, periodic=True)
electrons = adc.Model(state=adc.FluidState("compressible", gamma=1.4),
                      transport=adc.CompressibleFlux(),
                      source=adc.PotentialForce(charge=-1.0), elliptic=adc.ChargeDensity(charge=-1.0))
ions = adc.Model(state=adc.FluidState("isothermal", cs2=0.5),
                 transport=adc.IsothermalFlux(),
                 source=adc.PotentialForce(charge=+1.0), elliptic=adc.ChargeDensity(charge=+1.0))
sim.add_block("electrons", model=electrons,
              spatial=adc.Spatial(vanleer=True, flux="hllc"),
              time=adc.IMEX(substeps=10))     # raide -> source implicite, sous-cyclé
sim.add_block("ions", model=ions,
              spatial=adc.Spatial(minmod=True), time=adc.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
sim.set_density("electrons", ne)
sim.set_density("ions", np.ones((n, n)))

sim.advance(0.001, 8)
print("blocs :", sim.block_names(), "| |phi|_max :", np.abs(sim.potential()).max())
```

## Deux-fluides isotherme AP (régime raide)

Intégrateur AP **sur mesure** : un schéma asymptotic-preserving stable quand `dt*omega_pe >> 1`,
là où un explicite exploserait. Non composable bloc à bloc (la stabilisation AP couple la
raideur au pas de temps dans l'elliptique), ce n'est **pas** une brique générique mais un
**scénario** : il a quitté le cœur et vit désormais dans `adc_cases/two_fluid_ap/`, où sa
physique C++ (`two_fluid_ap.hpp`) est compilée à la volée contre les en-têtes génériques
d'`adc_cpp` puis pilotée depuis Python (cf. `adc_cases/two_fluid_ap/run.py`). Le module `_adc`
ne l'expose plus.

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
