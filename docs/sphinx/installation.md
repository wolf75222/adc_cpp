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
ctest --test-dir build          # 47/47
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
cmake -S . -B build-omp -DADC_USE_OPENMP=ON    # OpenMP
cmake -S . -B build-mpi -DADC_USE_MPI=ON       # MPI (+8 tests via mpirun -np 4)
cmake -S . -B build-gpu -DADC_USE_KOKKOS=ON    # Kokkos (CPU/GPU GH200)
```

Le backend est une propriété de la cible `adc` (héritée par tout ce qui la lie) ; aucun
drapeau à rajouter dans le code. Voir le tutoriel
[08, Backends](https://github.com/wolf75222/adc_cpp/blob/master/tutorials/08_backends.md).

## Vérification

```python
import adc
cfg = adc.DiocotronConfig(); cfg.n = 64
sim = adc.DiocotronSolver(cfg)
sim.step_cfl(0.4)
print(sim.density().shape)      # (64, 64)
```
