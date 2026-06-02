# API Python

Référence du module `adc` (bindings pybind11 de `libadc`, plus le sucre objet du paquet
`adc/`). Python compose un système bloc par bloc ; tout le calcul cellule par cellule
reste dans la lib C++ compilée. Voir le tutoriel
[03, API Python](https://github.com/wolf75222/adc_cpp/blob/master/tutorials/03_python_api.md)
pour des exemples annotés.

> Les blocs `autoclass` ci-dessous ne se rendent que si le module `adc` a été construit
> (`-DADC_BUILD_PYTHON=ON`) et est importable au build de la doc.

## Composition générique : `adc.System`

On ajoute des blocs (un modèle par bloc), on configure un Poisson de système partagé, on
fixe les conditions initiales en numpy, on avance. Modèles disponibles : `diocotron`,
`electron_euler`, `ion_isothermal`, `euler_poisson`.

```{eval-rst}
.. autoclass:: adc.System
   :members:

.. autoclass:: adc.SystemConfig
   :members:
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

## Solveurs spécialisés

Intégrateurs sur mesure exposés comme façades, non composables bloc à bloc.

### Deux-fluides isotherme asymptotic-preserving

```{eval-rst}
.. autoclass:: adc.TwoFluidAP
   :members:

.. autoclass:: adc.TwoFluidAPConfig
   :members:
```

### Diocotron sur AMR multi-patch

```{eval-rst}
.. autoclass:: adc.DiocotronAmr
   :members:

.. autoclass:: adc.DiocotronAmrConfig
   :members:
```
