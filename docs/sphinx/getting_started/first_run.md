# Premier run

Le plus petit programme `adc` : on construit un systeme periodique, on y pose un bloc
diocotron compose de briques natives, on branche le Poisson de systeme, on pose une condition
initiale numpy, on avance de quelques pas et on relit la densite. Copiable tel quel (il suppose
seulement que le module est [installe](installation.md) et importable).

```python
import numpy as np
import adc

# 1. Un systeme carre periodique 96 x 96, domaine [0, 1]^2.
sim = adc.System(n=96, L=1.0, periodic=True)

# 2. Un bloc "ne" : la densite, advectee par la derive E x B (transport=ExB),
#    couplee au Poisson via une densite de fond neutralisante (elliptic=BackgroundDensity).
sim.add_block(
    "ne",
    model=adc.Model(
        state=adc.Scalar(),                       # une variable conservative : la densite
        transport=adc.ExB(B0=1.0),                # vitesse de derive v = (-d_y phi, d_x phi) / B0
        source=adc.NoSource(),                    # pas de terme source
        elliptic=adc.BackgroundDensity(alpha=1.0, n0=1.0),  # rhs Poisson = alpha (n - n0)
    ),
    spatial=adc.Spatial(minmod=True),             # MUSCL minmod + Rusanov (defaut)
    time=adc.Explicit(),                          # integration explicite
)

# 3. Le Poisson de systeme : second membre = densite de charge, solveur multigrille.
sim.set_poisson(rhs="charge_density", solver="geometric_mg")

# 4. Condition initiale : une bande de charge perturbee le long de x.
n = 96
xs = (np.arange(n) + 0.5) / n
X, Y = np.meshgrid(xs, xs)                        # indexing 'xy' : ne[j, i]
y0 = 0.5 + 0.02 * np.cos(2.0 * np.pi * 2.0 * X)   # mode azimutal 2
ne0 = 1.0 + np.exp(-((Y - y0) ** 2) / 0.05 ** 2)
sim.set_density("ne", np.ascontiguousarray(ne0))

# 5. Quelques pas a CFL fixe, puis on relit l'etat.
for _ in range(20):
    sim.step_cfl(0.4)

print("t        =", sim.time())
print("masse    =", sim.mass("ne"))             # conservee par le transport advectif periodique
print("densite  =", sim.density("ne").shape)    # (96, 96)
```

Ce que font les appels cles :

- `adc.System(n=, L=, periodic=)` cree le systeme/coupleur (domaine carre par defaut).
- `adc.Model(state=, transport=, source=, elliptic=)` compose un modele a partir de briques
  natives. Ici : `Scalar` + `ExB` + `NoSource` + `BackgroundDensity`. C'est exactement le
  modele diocotron reduit, mais le coeur ne le nomme pas.
- `set_poisson(rhs="charge_density", solver="geometric_mg")` : `rhs` vaut `charge_density` ou
  `composite` ; `solver` se passe par mot-cle (`geometric_mg` ou la FFT).
- `set_density(name, arr2d)` pose un tableau `(n, n)` contigu ; `step_cfl(cfl)` avance d'un pas
  au CFL donne ; `density(name)` / `mass(name)` / `time()` lisent l'etat.

Pour passer du modele compose en briques au modele ecrit en formules (DSL `adc.dsl.Model`),
au raffinement adaptatif (`adc.AmrSystem`), aux figures et au GIF, suivez le
[tutoriel A->Z](tutorial.md).
