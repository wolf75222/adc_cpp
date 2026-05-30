# 00, Installer et compiler

## Prérequis

- Compilateur C++23 (Apple Clang 16+, GCC 13+). Sous Kokkos/CUDA, C++20 (nvcc 12.x).
- CMake >= 3.20
- Python >= 3.10 pour le module
- Optionnel : `matplotlib` + `numpy` pour les scripts de tracé ; Eigen pour les outils
  d'analyse hôte (théorie diocotron)

## C++ seul

```bash
git clone https://github.com/wolf75222/adc_cpp.git
cd adc_cpp
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

**53/53** tests doivent passer (transport, flux de face, Riemann, Poisson, AMR reflux
2-niveaux / N-niveaux / multi-patch, deux-fluides AP, cyclotron, coupleurs).

## Options CMake

| Option | Défaut | Effet |
|---|---|---|
| `ADC_USE_OPENMP` | OFF | dispatch OpenMP de `for_each_cell` |
| `ADC_USE_KOKKOS` | OFF | dispatch Kokkos (CPU/GPU portable, GH200) |
| `ADC_USE_MPI` | OFF | seam distribué (`comm.hpp`) + tests `mpirun` |
| `ADC_USE_HDF5` | OFF | DataWriter HDF5 parallèle |
| `ADC_USE_EIGEN` | ON | outils d'analyse hôte (théorie diocotron) |
| `ADC_BUILD_PYTHON` | OFF | module pybind11 `adc` |

Le backend est une **propriété de la bibliothèque** : il est attaché à la cible `adc`
(INTERFACE), pas rajouté dans chaque solveur. `-DADC_USE_OPENMP=ON` suffit ; rien d'autre
à changer dans le code.

## Backends

```bash
cmake -S . -B build-omp -DADC_USE_OPENMP=ON   # OpenMP
cmake -S . -B build-mpi -DADC_USE_MPI=ON      # MPI (+8 tests via mpirun -np 4)
ctest --test-dir build-mpi                    # 55/55 (47 + 8 MPI)
```

Sous OpenMP les résultats sont **identiques à la série** (déterminisme thread-count) ;
sous MPI ils sont **bit-identiques** à np=1 (invariance au nombre de rangs).

## Module Python

```bash
cmake -S . -B build-py -DADC_BUILD_PYTHON=ON
cmake --build build-py -j
PYTHONPATH=build-py/python python3 python/test_bindings.py   # "OK test_bindings"
```

Le `.so` est sous `build-py/python/adc.cpython-3xx-*.so`. Voir
[03_python_api.md](03_python_api.md).

## Artefacts

- `build/bin/` : exécutables d'exemple (`diocotron`, `diocotron_amr`, `diocotron_amr3`,
  `diocotron_multipatch`, `diocotron_mpi`, `bench_amr`, ...).
- `docs/` : animations (`anim_diocotron*.gif`) et figures, pré-générées par les scripts
  `scripts/make_*.py` et `scripts/plot_*.py`.
