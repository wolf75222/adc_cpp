# Installation

`adc_cpp` se construit avec CMake. Le coeur est header-only ; ce que l'on compile, c'est la
suite de tests (C++) et, optionnellement, le module Python (pybind11). Il n'y a pas de paquet
pip : le module s'utilise via `PYTHONPATH`.

## Prerequis

- Compilateur C++23 (AppleClang 16+, GCC 13+, Clang 17+). Sous Kokkos/CUDA on retombe a
  C++20 (CUDA 12.x ne propose pas `-std=c++23` ; tout le coeur compile en C++20).
- CMake >= 3.20.
- Python >= 3.10 pour le module (3.12 recommande, voir le piege interpreteur ci-dessous).
- Catch2 / pybind11 sont recuperes automatiquement par `FetchContent` s'ils sont absents.

## Build C++ (coeur + tests)

```bash
git clone https://github.com/wolf75222/adc_cpp.git
cd adc_cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

C'est le build par defaut : serie, sans Kokkos ni MPI (`comm.hpp` = rang unique). Pour le
decompte exact des tests par backend, voir la matrice de couverture
[`BACKEND_COVERAGE.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md) (source de verite unique, mise a jour a la
main) plutot qu'un nombre fige ici.

## Module Python

Le module `adc` (les bindings pybind11 de la lib) se construit avec `-DADC_BUILD_PYTHON=ON` et
s'utilise via `PYTHONPATH` :

```bash
cmake -S . -B build-py -DADC_BUILD_PYTHON=ON
cmake --build build-py -j
export PYTHONPATH=$PWD/python:$PWD/build-py/python
python3 -c "import adc; print(adc.__doc__.splitlines()[0])"
```

Deux chemins doivent etre sur `PYTHONPATH` : `python/` (le paquet pur `adc` : `__init__.py`,
`dsl.py`) et `build-py/python/` (l'extension compilee `_adc`). Le paquet `adc` importe `_adc`
depuis le second.

### Le piege de l'interpreteur (a lire avant de debugger un `ImportError`)

L'extension compilee porte le suffixe ABI de l'interpreteur qui l'a construite, par exemple
`_adc.cpython-312-darwin.so`. Elle ne s'importe que depuis la version de Python qui correspond
a ce suffixe (ici CPython 3.12), avec `numpy` installe dans ce meme interpreteur, et le
repertoire `python/` sur `PYTHONPATH`. Tenter `import adc` depuis un `python3` systeme different
echoue par :

```
ModuleNotFoundError: No module named 'adc._adc'
```

Le module pur `adc` est bien trouve, mais il ne trouve pas son extension `_adc` (suffixe ABI
incompatible). Pour eviter cela :

- pinnez l'interpreteur au moment du build :
  `cmake -S . -B build-py -DADC_BUILD_PYTHON=ON -DPython_EXECUTABLE=$(which python3.12)` ;
- importez `adc` avec exactement ce meme `python3.12` ;
- assurez-vous que `numpy` est present dans cet interpreteur.

Sur macOS, plusieurs Python coexistent souvent (framework Python.org, conda, brew) ; le
`CMakeLists.txt` deprioritise les frameworks (`Python_FIND_FRAMEWORK LAST`) pour que
l'interpreteur et l'extension concordent, mais le pin explicite reste le plus sur. Le message de
configuration `adc Python module: interpreteur <chemin> (<version>)` confirme lequel a ete
retenu.

## Backends optionnels (C++)

Le backend est une propriete de la cible `adc` : tout ce qui la lie (la facade compilee, les
tests) en herite. Il n'y a aucun drapeau a ajouter dans le code.

```bash
cmake -S . -B build-mpi    -DADC_USE_MPI=ON       # backend MPI (seam comm distribue)
cmake -S . -B build-kokkos -DADC_USE_KOKKOS=ON    # Kokkos : CPU Serial/OpenMP + GPU CUDA/HIP
cmake -S . -B build-omp    -DADC_USE_OPENMP=ON     # OpenMP autonome (DEPRECIE -> Kokkos)
cmake -S . -B build-hdf5   -DADC_USE_HDF5=ON      # DataWriter HDF5 (sortie parallele)
```

Options (cf. `CMakeLists.txt`) :

| Option CMake | Defaut | Role |
|---|---|---|
| `ADC_BUILD_TESTS` | `ON` | construit la suite de tests (`tests/`) |
| `ADC_BUILD_PYTHON` | `OFF` | construit le module pybind11 `adc` |
| `ADC_USE_KOKKOS` | `OFF` | backend de dispatch Kokkos (CPU OpenMP + GPU), recommande |
| `ADC_USE_OPENMP` | `OFF` | backend OpenMP autonome, deprecie (-> Kokkos) |
| `ADC_USE_MPI` | `OFF` | backend MPI (seam `comm` distribue) |
| `ADC_USE_HDF5` | `OFF` | `DataWriter` HDF5 (ecriture parallele) |
| `ADC_BUILD_BENCH` | `OFF` | harnais de profilage (`bench/`, jamais en CI) |

Choisir un backend de dispatch : `ADC_USE_KOKKOS` ou `ADC_USE_OPENMP`, pas les deux (erreur de
configuration sinon). Kokkos est le chemin recommande : il couvre OpenMP CPU et ouvre le GPU
CUDA/HIP avec un seul code portable. La configuration Kokkos GPU exige `nvcc_wrapper` comme
compilateur, par exemple :

```bash
cmake -S . -B build-gpu -DADC_USE_KOKKOS=ON \
      -DCMAKE_CXX_COMPILER=$KOKKOS/bin/nvcc_wrapper -DKokkos_ROOT=$KOKKOS
```

La CI joue les jobs CPU (Release, MPI, Kokkos Serial, Kokkos OpenMP) ; le GPU est valide
manuellement sur ROMEO. Detail : [Verifier son backend](backend.md).

## Verification

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

Si ces lignes s'executent, le module est correctement construit et importe. Pour aller plus
loin, voir le [premier run](first_run.md) puis le [tutoriel A->Z](tutorial.md).
