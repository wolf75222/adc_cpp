# Installation

## Prérequis

- Compilateur C++23 (AppleClang 16+, GCC 13+, Clang 17+) ; C++20 sous Kokkos/CUDA
- CMake >= 3.20
- Python >= 3.10 pour le module
- Catch2 / pybind11 récupérés automatiquement si absents

## Build C++

```bash
git clone https://github.com/wolf75222/adc_cpp.git
cd adc_cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build          # suite coeur (decompte a jour : docs/BACKEND_COVERAGE.md)
```

## Module Python

Le module `adc` se construit avec `-DADC_BUILD_PYTHON=ON` et s'utilise via `PYTHONPATH`
(pas de paquet pip pour l'instant) :

```bash
cmake -S . -B build-py -DADC_BUILD_PYTHON=ON
cmake --build build-py -j
export PYTHONPATH=$PWD/build-py/python
python3 -c "import adc; print(adc.__doc__)"
```

## Backends optionnels

```bash
cmake -S . -B build-omp -DADC_USE_OPENMP=ON    # OpenMP autonome (déprécié -> Kokkos)
cmake -S . -B build-mpi -DADC_USE_MPI=ON       # MPI (+21 entrées ctest via mpirun, np=1/2/4)
cmake -S . -B build-gpu -DADC_USE_KOKKOS=ON    # Kokkos (CPU Serial/OpenMP, GPU Cuda)
```

Le backend Kokkos est recommandé (CPU multi-thread ET GPU avec un seul code) ; le backend
OpenMP autonome est déprécié. Le backend est une propriété de la cible `adc` (héritée par tout
ce qui la lie) ; aucun drapeau à rajouter dans le code. La CI joue trois jobs : Release, MPI et
Kokkos (Serial). Tutoriels et exemples vivent dans le dépôt applicatif `adc_cases`.

## Vérification

```python
import numpy as np
import adc

sim = adc.System(n=64, periodic=True)
sim.add_block("ne", model=adc.Model(
    state=adc.Scalar(), transport=adc.ExB(B0=1.0),
    source=adc.NoSource(), elliptic=adc.BackgroundDensity(alpha=1.0, n0=1.0)))
sim.set_poisson()
sim.set_density("ne", np.ones((64, 64)))
sim.step_cfl(0.4)
print(sim.density("ne").shape)   # (64, 64)
```
