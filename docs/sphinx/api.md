# API Python

Référence du module compilé `adc` (façade `libadc`, bindings pybind11). Miroir 1:1 de la
surface publique C++ : voir aussi le tutoriel
[03, API Python](https://github.com/wolf75222/adc_cpp/blob/master/tutorials/03_python_api.md)
pour des exemples annotés.

> Les blocs `autoclass` ci-dessous ne se rendent que si le module `adc` a été construit
> (`-DADC_BUILD_PYTHON=ON`) et est importable au build de la doc.

## Diocotron (dérive E x B)

```{eval-rst}
.. autoclass:: adc.DiocotronConfig
   :members:

.. autoclass:: adc.DiocotronSolver
   :members:

.. autoclass:: adc.DiocotronIC
   :members:
```

## Euler-Poisson auto-gravitant

```{eval-rst}
.. autoclass:: adc.EulerPoissonConfig
   :members:

.. autoclass:: adc.EulerPoissonSolver
   :members:
```

## Deux-fluides isotherme AP

```{eval-rst}
.. autoclass:: adc.TwoFluidAPConfig
   :members:

.. autoclass:: adc.TwoFluidAPSolver
   :members:
```
