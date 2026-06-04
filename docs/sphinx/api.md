# API Python

Référence du module `adc` (bindings pybind11 de `libadc`, plus le sucre objet du paquet
`adc/`). Python compose un système bloc par bloc ; tout le calcul cellule par cellule
reste dans la lib C++ compilée. Pour des exemples annotés, voir le [quickstart](quickstart.md) ;
les parcours guidés applicatifs vivent dans le dépôt
[`adc_cases`](https://github.com/wolf75222/adc_cases).

> Les blocs `autoclass` ci-dessous ne se rendent que si le module `adc` a été construit
> (`-DADC_BUILD_PYTHON=ON`) et est importable au build de la doc.

## Composition générique : `adc.System`

On ajoute des blocs (un modèle par bloc), on configure un Poisson de système partagé, on
fixe les conditions initiales en numpy, on avance. Un modèle est une **composition de briques
génériques** `adc.Model(state, transport, source, elliptic)` ; le cœur ne nomme aucun scénario
(les compositions nommées vivent côté application, cf. `adc_cases/models.py`).

```{eval-rst}
.. autoclass:: adc.System
   :members:

.. autoclass:: adc.SystemConfig
   :members:
```

### Modèle composé par briques

Un modèle est assemblé par `adc.Model(state, transport, source, elliptic)` à partir de briques
génériques. Le cœur C++ ne connaît que ces briques (aucun nom de scénario) ; une composition
nommée (diocotron, electron_euler, ...) vit côté application (`adc_cases/models.py`).

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

### Schéma par bloc

Chaque bloc choisit indépendamment sa reconstruction spatiale, son flux et son traitement
temporel.

```{eval-rst}
.. autoclass:: adc.Spatial
   :members:

.. autoclass:: adc.Explicit
   :members:

.. autoclass:: adc.IMEX
   :members:

.. autofunction:: adc.Implicit
```

### Intégrateur temporel écrit en Python

`System` expose les primitives `solve_fields()`, `eval_rhs(name)`, `get_state(name)` et
`set_state(name, U)`. Le résidu et le Poisson restent calculés en C++ (par cellule) ;
seul l'assemblage des étages RK est en Python (par pas).

```{eval-rst}
.. automodule:: adc.integrate
   :members:
```

## Intégrateurs sur mesure (hors API publique)

Intégrateurs sur mesure, non composables bloc à bloc. Ils ne font **pas** partie de l'API
publique : aucun scénario nommé n'est exposé par le paquet `adc`.

### Deux-fluides isotherme asymptotic-preserving

L'intégrateur AP deux-fluides a quitté le cœur : ce n'est pas une brique générique mais un
**scénario**. La stabilisation AP couple la raideur au pas de temps dans l'elliptique, ce que
la composition `System` ne reproduit pas ; d'où son statut d'intégrateur sur mesure plutôt que
de brique composable. Sa physique C++ vit désormais dans `adc_cases/two_fluid_ap/`
(`two_fluid_ap.hpp`), compilée à la volée contre les en-têtes génériques d'`adc_cpp` puis
pilotée depuis Python ; le module `_adc` ne l'expose plus. Voir `adc_cases/two_fluid_ap/run.py`
(note de méthode archivée :
[archive/two_fluid_ap.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/archive/two_fluid_ap.md)).

### Composition sur AMR multi-patch

```{eval-rst}
.. autoclass:: adc.AmrSystem
   :members:

.. autoclass:: adc.AmrSystemConfig
   :members:
```
