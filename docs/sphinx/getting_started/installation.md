# Installation

Le coeur d'`adc_cpp` est header-only : rien a compiler pour le consommer en C++. Ce qui se
construit, c'est la suite de tests (C++) et le module Python `adc` (pybind11) -- soit par
`pip install .`, soit par CMake directement.

## Environnement conda

Une commande cree l'env `adc` (outillage complet : CMake, Ninja, ccache, Python 3.12, NumPy,
pybind11, Kokkos, OpenMPI, libomp) **et y fige la meilleure toolchain de la plateforme** :

```bash
bash scripts/setup_env.sh
conda activate adc
```

Sur macOS, le script fixe `CC`/`CXX` sur AppleClang dans l'env meme (exportes a chaque
activation, prioritaires sur le PATH) : un clang LLVM vanilla qui trainerait en tete de PATH
compile les grosses unites du module plus de 15 fois plus lentement, sans message. Sur Linux,
il installe `cxx-compiler` (gcc 14, C++23) -- toolchain complete sans droits root. Un `CC`/`CXX`
pose a la main avant un build garde la priorite.

Equivalent manuel, sans le choix de toolchain : `conda env create -f environment.yml`.
Mise a jour : relancer le script (ou `conda env update -f environment.yml --prune`).
L'interpreteur de l'env construit et importe le module : pas de divergence d'ABI cpython.
Norme : C++20 (Kokkos, seul backend on-node, est compile sous nvcc pour la cible Cuda).

## Module Python

### Utilisateur : `pip install .`

`pip install .` pilote le CMakeLists via scikit-build-core (`pyproject.toml`) et installe le
paquet dans `site-packages` : `import adc` marche ensuite sans `PYTHONPATH`. Les backends se
choisissent par variables d'environnement, mappees sur les options CMake :

```bash
conda activate adc
pip install .                                  # Kokkos Serial (FetchContent si non installe)
Kokkos_ROOT=$CONDA_PREFIX pip install . -v     # reutilise le Kokkos de l'env (OpenMP si dispo)
ADC_USE_MPI=ON pip install . -v                                # MPI
```

Puis, en Python :

```python
import adc
print(adc.__version__)
adc.doctor()           # diagnostic de l'environnement (OK/FAIL + remede par ligne)
adc.set_threads()      # tous les coeurs -- ou set_threads(8) ; AVANT le 1er System
sim = adc.System(n=256)
```

`pip install -e .` (editable) convient au dev ; le cache de build persiste sous `build/`, les
reinstallations sont incrementales.

**Build lent ?** Trois verifications, dans l'ordre :

1. le compilateur. Sur macOS, un clang LLVM (Homebrew) en tete de PATH compile les grosses
   unites du module 4 a 5 fois plus lentement qu'AppleClang (mesure : plus d'une heure au lieu
   de ~8 min). Le configure l'indique ; pour forcer AppleClang :
   `CXX=/usr/bin/clang++ pip install .` ;
2. ccache. Fourni par l'env conda et detecte automatiquement : les recompilations d'un fichier
   deja vu deviennent quasi instantanees ;
3. les reinstallations. Le cache sous `build/` rend `pip install .` incremental : seul ce qui a
   change recompile. Le premier build complet reste long (deux grosses unites a `-O3`) -- c'est
   le cout d'un module optimise, paye une fois.

### Developpeur : presets + PYTHONPATH

Pour iterer sur le C++ sans reinstaller, les presets construisent le module dans l'arbre :

```bash
cmake --preset python          && cmake --build --preset python            # Kokkos Serial
cmake --preset python-parallel && cmake --build --preset python-parallel   # Kokkos conda (OpenMP)
export PYTHONPATH=$PWD/build-py/python        # ou build-py-kokkos/python
```

Un seul chemin suffit : la configuration copie les sources du paquet a cote de l'extension.
Equivalent sans preset :

```bash
cmake -S . -B build-py -G Ninja -DADC_BUILD_PYTHON=ON -DADC_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$(which python3.12)
cmake --build build-py --target _adc -j
```

### Threads

Le nombre de threads n'est pas un argument du modele : il faut un module construit contre un Kokkos
OpenMP (preset `python-parallel`), puis un reglage AVANT la premiere allocation --
Kokkos s'initialise a ce moment-la et lit l'environnement une seule fois :

```python
import adc
adc.set_threads(8)       # = OMP_NUM_THREADS + KOKKOS_NUM_THREADS, sans toucher au shell
sim = adc.System(n=256)
```

Module Kokkos Serial ou appel trop tardif : un avertissement le signale et le reglage est ignore.
`adc.parallel_info()` donne l'etat courant. Pour que le DSL `backend="production"` scale aussi,
exporter `ADC_KOKKOS_ROOT` (meme racine Kokkos que le build du module).

## Coeur C++ et tests

```bash
cmake --preset serial   && cmake --build --preset serial   && ctest --preset serial   # Kokkos Serial
cmake --preset parallel && cmake --build --preset parallel && ctest --preset parallel  # Kokkos conda
cmake --preset mpi      && cmake --build --preset mpi      && ctest --preset mpi
```

Chaque preset ecrit dans son dossier (`build`, `build-kokkos`, `build-mpi`) ; les presets
parallele/MPI exigent l'env conda actif (ils lisent `$CONDA_PREFIX` et refusent de se
configurer sans). `ctest -L mpi` ou `-L core` selectionne un sous-ensemble. Equivalent manuel :

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Le decompte des tests par backend est tenu dans
[`BACKEND_COVERAGE.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md).

## Backends

| Option CMake | Defaut | Role |
|---|---|---|
| `ADC_BUILD_TESTS` | `ON` en top-level, `OFF` en sous-projet | suite de tests (`tests/`) |
| `ADC_BUILD_PYTHON` | `OFF` | module pybind11 `adc` |
| `ADC_USE_KOKKOS` | `ON` | seul backend on-node, **obligatoire** (`OFF` = erreur fatale) ; FetchContent si non installe |
| `ADC_USE_MPI` | `OFF` | seam `comm` distribue |
| `ADC_USE_HDF5` | `OFF` | `DataWriter` HDF5 |
| `ADC_BUILD_BENCH` | `OFF` | harnais de profilage (`bench/`) |
| `ADC_INSTALL` | `ON` en top-level | regles `cmake --install` + `find_package(adc)` |
| `ADC_PY_LTO` | `OFF` | ThinLTO du module (`OFF` = build rapide) |
| `ADC_USE_CCACHE` | `ON` | ccache si present (ignore sous nvcc) |

Chaque option est aussi lisible depuis l'environnement (`Kokkos_ROOT=... pip install .`) ;
un `-D` explicite garde la priorite. Le backend est une propriete de la cible `adc` : tout ce
qui la lie en herite, aucun drapeau dans le code. adc_cpp est **Kokkos-only** : il n'y a plus de
backend OpenMP autonome ni de build non-Kokkos ; Serial, OpenMP et Cuda sont des espaces
d'execution Kokkos choisis a l'install (ou au fetch) de Kokkos, pas des drapeaux adc distincts.
**Kokkos n'a pas besoin d'etre pre-installe** : introuvable, il est recupere + construit
automatiquement (FetchContent, tarball de release verifie par SHA256).

**Kokkos via conda** : le paquet `kokkos` de conda-forge est generalement compile avec le seul
backend Serial. Le build passe, mais ne scale pas en threads -- verifier le message
`Kokkos found ... = (...)` au configure. Pour un Kokkos Serial+OpenMP dans l'env (~2 min,
meme compilateur que le projet) :

```bash
bash scripts/kokkos_openmp_conda.sh
cmake --preset python-parallel && cmake --build --preset python-parallel
export ADC_KOKKOS_ROOT="$CONDA_PREFIX"
```

**GPU** : `nvcc_wrapper` comme compilateur, valide sur ROMEO (pas via conda) :

```bash
cmake -S . -B build-gpu -DADC_USE_KOKKOS=ON \
      -DCMAKE_CXX_COMPILER=$KOKKOS/bin/nvcc_wrapper -DKokkos_ROOT=$KOKKOS
```

## Cluster (Spack, sans root)

Sur ROMEO et assimiles, l'outillage vient des modules/Spack -- pas de conda (`conda env create`
exige le reseau, souvent absent des noeuds).

**ROMEO** : un profil machine versionne fait toute la mise en place (env Spack du site,
CC/CXX, Kokkos, variables du DSL, cache dans le scratch) :

```bash
cp Tools/machines/romeo/romeo_adc.profile.example ~/romeo_adc.profile
# editer les lignes '# A ADAPTER' (chemin Kokkos), puis a chaque session/job :
source ~/romeo_adc.profile                       # ADC_ROMEO_ARCH=armgpu pour le GPU
```

**Autre cluster** : le guide generique
[HPC_SPACK_GUIDE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/HPC_SPACK_GUIDE.md)
couvre l'installation de la pile (Spack du site ou bootstrap perso), la compilation ciblee
microarchitecture et le lancement SLURM avec placement des threads/GPU. Schema minimal :

```bash
spack load cmake ninja kokkos openmpi            # ou module load <env du site>
cmake -S . -B build-kokkos -G Ninja -DADC_USE_KOKKOS=ON \
      -DKokkos_ROOT=$(spack location -i kokkos)
# configurer sur le LOGIN ; dans l'allocation, relancer seulement : ninja -C build-kokkos
```

GPU Grace-Hopper : [GPU_ROMEO.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/GPU_ROMEO.md).
Pour le DSL `backend="production"`, exporter `ADC_KOKKOS_ROOT=<prefix Kokkos>` (le profil ROMEO
le fait) ; le compilateur du build est bake dans `_adc`, le DSL le retrouve seul tant qu'il
existe sur les noeuds.

## Depannage

Premier reflexe, quel que soit le symptome :

```bash
python -c "import adc; adc.doctor()"
```

Chaque ligne verifie un maillon (interpreteur/ABI, numpy, Kokkos, compilateur du DSL et sa
norme, synchronisation en-tetes/module, threads) et donne le remede en cas d'echec.

**`ImportError` sur `adc._adc`** : l'extension est epinglee a l'interpreteur qui l'a construite
(suffixe `cpython-312`). Le message d'erreur indique desormais la cause exacte (extension
absente, ou mauvais interpreteur) et la commande de reconstruction. Regle simple : construire
et importer avec le meme python -- celui de l'env conda.

**`error: invalid value 'c++23'` ou `ABI incompatible` (DSL production)** : le loader `.so`
compile a l'execution doit partager la toolchain du module. Trois protections couvrent ce cas :
le compilateur du build est bake dans le module et prefere au PATH (`cxx=` explicite >
`$ADC_CXX` > compilateur du build > PATH), la norme est testee avant compilation (repli
`c++2b`), et l'erreur restante explique quoi faire. Un module perime vis-a-vis des en-tetes
(apres un `git pull`) est detecte avant le chargement, avec la commande de rebuild.

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

Si ces lignes s'executent, l'installation est bonne. Suite : [premier run](first_run.md) puis
[tutoriel A->Z](tutorial.md).
