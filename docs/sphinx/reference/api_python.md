# API Python

Reference curee du module `adc` (bindings pybind11 de `libadc` + le sucre objet du paquet
`adc/`). Python COMPOSE un systeme bloc par bloc ; tout le calcul cellule par cellule reste
dans la lib C++ compilee (pas de boucle numpy sur le hot path, GPU/MPI conserves).

Seule la surface PUBLIQUE est documentee ici (les symboles internes ne sont pas listes). Pour
des parcours annotes, voir le [quickstart](../getting_started/first_run.md) ; les compositions nommees
(scenarios) vivent dans le depot [`adc_cases`](https://github.com/wolf75222/adc_cases).

```{note}
Les blocs `autoclass` / `autofunction` ci-dessous ne se rendent que si le module `adc` a ete
construit (`-DADC_BUILD_PYTHON=ON`) et est importable au build de la doc. Voir
[installation](../getting_started/installation.md) et le [quickstart](../getting_started/first_run.md) ; attention au footgun
d'interpreteur (le `.so` est lie a un cpython precis), detaille dans [limitations](limitations.md).
```

## Systeme : composer, configurer, avancer

`adc.System` est le coupleur : on ajoute des blocs (un modele par bloc), on configure un
Poisson de systeme partage, on fixe les conditions initiales en numpy, on avance. `add_block`
prend un modele compose `adc.Model(...)` ; `add_equation` aiguille sur le TYPE du modele
(`ModelSpec` natif ou `CompiledModel` issu du DSL). `set_poisson(rhs=..., solver=..., bc=...)`
configure l'elliptique de systeme ; `set_density` / `step_cfl` / `advance` / `run` pilotent
l'avance ; `density` / `mass` / `time` lisent l'etat.

```{eval-rst}
.. autoclass:: adc.System
   :members:
```

## AMR : composition sur hierarchie raffinee

`adc.AmrSystem` est le pendant raffine de `System` : un ou plusieurs blocs portes sur une
hierarchie AMR block-structured (regrid Berger-Rigoutsos, reflux conservatif). En multi-blocs,
la hierarchie est re-grillee sur l'UNION des tags (densite par bloc et/ou `|grad phi|`). Memes
signatures `add_block` / `add_equation` que `System` ; la cadence du regrid est portee par
`AmrSystemConfig.regrid_every`.

```{eval-rst}
.. autoclass:: adc.AmrSystem
   :members:
```

## Geometrie et maillage

`adc.System` tourne en geometrie cartesienne par defaut ; la geometrie polaire / disque se
choisit via les champs `geometry`, `nr`, `ntheta`, `r_min`, `r_max` de `SystemConfig` (cf.
[Sujets avances](../advanced/index.md), section geometrie). Les classes de maillage exposees :

```{eval-rst}
.. autoclass:: adc.CartesianMesh
.. autoclass:: adc.PolarMesh
```

## Modeles natifs : composition de briques

Un modele NATIF est assemble par `adc.Model(state, transport, source, elliptic)` a partir de
briques generiques. Le coeur C++ ne connait que ces briques (aucun nom de scenario). La
fonction valide la coherence etat <-> transport (Scalar avec ExB ; FluidState compressible
avec CompressibleFlux ; isotherme avec IsothermalFlux).

```{eval-rst}
.. autofunction:: adc.Model

.. autoclass:: adc.Scalar
.. autoclass:: adc.FluidState
.. autoclass:: adc.ExB
.. autoclass:: adc.CompressibleFlux
.. autoclass:: adc.IsothermalFlux
.. autoclass:: adc.NoSource
.. autoclass:: adc.PotentialForce
.. autoclass:: adc.GravityForce
.. autoclass:: adc.ChargeDensity
.. autoclass:: adc.BackgroundDensity
.. autoclass:: adc.GravityCoupling
```

## Schema spatial par bloc

Chaque bloc choisit independamment sa reconstruction (limiteur), son flux numerique de
Riemann et les variables reconstruites. `adc.FiniteVolume(...)` est un raccourci (le flux de
Riemann s'y appelle `riemann`, pour ne pas collisionner avec le flux PHYSIQUE `m.flux` du DSL)
qui remappe sur l'objet `adc.Spatial(...)`.

```{eval-rst}
.. autoclass:: adc.Spatial
   :members:

.. autofunction:: adc.FiniteVolume
```

## Traitement temporel par bloc

Le traitement temporel est porte par le bloc (et non le modele) : le meme modele se reutilise
avec des politiques distinctes. `adc.Explicit` (SSPRK2/3, substeps, stride) est le defaut ;
`adc.IMEX` / `adc.SourceImplicit` traitent la SOURCE raide en implicite (backward-Euler,
Newton local a la cellule) tandis que le transport reste explicite -- ce n'est PAS un solveur
implicite global. `adc.Split` / `adc.Strang` sont l'opt-in du splitting explicite/implicite et
prennent un etage source `adc.CondensedSchur` (condensation de Schur du couplage Lorentz
electrostatique). `adc.Role` adresse une composante par son sens physique.

```{eval-rst}
.. autoclass:: adc.Explicit
   :members:

.. autoclass:: adc.IMEX
   :members:

.. autoclass:: adc.SourceImplicit
   :members:

.. autoclass:: adc.Split
   :members:

.. autoclass:: adc.Strang
   :members:

.. autoclass:: adc.CondensedSchur
   :members:

.. autoclass:: adc.Role
   :members:
```

```{note}
`adc.Implicit(...)` existe encore comme alias d'`adc.IMEX` mais est OBSOLETE (le nom suggere a
tort un solveur implicite global) et emet un `DeprecationWarning` : utiliser
`adc.SourceImplicit(...)` ou `adc.IMEX(...)`.
```

## Couplages inter-especes

Couplages operator-split appliques apres le transport, passes a `sim.add_coupling(...)` :
ionisation, friction inter-especes, echange thermique.

```{eval-rst}
.. autoclass:: adc.Ionization
.. autoclass:: adc.Collision
.. autoclass:: adc.ThermalExchange
```

## DSL symbolique de modeles

Le sous-module `adc.dsl` decrit un modele en formules symboliques (variables conservatives,
auxiliaires, flux, valeurs propres, primitives, second membre elliptique), le verifie, puis le
compile en un `.so` branchable via `System.add_equation`. `dsl.Model` est la facade ;
`dsl.CompiledModel` est le resultat de `m.compile(...)` (il porte le `.so` et les metadonnees
de dispatch) ; `dsl.HybridModel` melange briques natives et briques DSL partielles dans UN
modele (produit par `adc.CompositeModel(...)`).

```{eval-rst}
.. automodule:: adc.dsl
   :members: Model, CompiledModel, HybridModel
```

```{note}
Backends de compilation (`m.compile(..., backend=...)`, defaut `aot`) : `prototype` et `aot`
sont CPU-only (pas de MPI/AMR/GPU) ; `production` est CPU + MPI + AMR. Voir
[matrice des backends](backend_matrix.md) et [limitations](limitations.md). Conception du DSL :
[DSL_API.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/DSL_API.md), [DSL_MODEL_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/DSL_MODEL_DESIGN.md).
```
