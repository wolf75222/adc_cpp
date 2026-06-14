<div align="center">

# ADC CPP

**Coeur C++23 d'un solveur AMR / MPI / GPU pour systemes hyperbolique-elliptique couples.**

![C++23](https://img.shields.io/badge/C%2B%2B-23-blue?logo=cplusplus)
![Build](https://img.shields.io/badge/build-CMake%203.20%2B%20(presets)-064F8C?logo=cmake)
![Backends](https://img.shields.io/badge/backends-MPI%20%7C%20Kokkos%20(CPU%2FGPU)-orange)
![CI](https://img.shields.io/badge/CI-Release%20%7C%20MPI%20%7C%20Kokkos%20%7C%20Python-brightgreen)
![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey)
![License](https://img.shields.io/badge/license-BSD--3-green)

</div>

<p align="center">
  <img src="docs/anim_romeo_diocotron_amr3.gif" alt="Instabilite diocotron AMR 3 niveaux sur ROMEO" width="640">
</p>

<div align="center">
<sub>
Instabilite diocotron (derive E x B) sur AMR 3 niveaux emboites, ROMEO (96 coeurs AMD EPYC).
Patchs fins suivis par regrid Berger-Rigoutsos, sous-cyclage Berger-Oliger, reflux conservatif
(derive de masse ~ 1e-15). Version locale reproductible (facade Python) :
<a href="https://github.com/wolf75222/adc_cases/tree/master/diocotron_amr"><code>adc_cases/diocotron_amr</code></a>.
</sub>
</div>

---

`adc_cpp` est le **coeur** : un moteur generique sans modele, une bibliotheque de briques physiques
(`include/adc/physics/`) et les bindings Python (`adc`). Le coeur ne nomme aucun scenario ; il fournit
des briques generiques composees en `CompositeModel`. Les scenarios nommes (diocotron, Euler-Poisson,
deux-fluides) vivent dans [`adc_cases`](https://github.com/wolf75222/adc_cases).

Sur maillage cartesien adaptatif, le coeur avance une partie hyperbolique `U` couplee a une partie
elliptique `phi` :

```
d U / d t  +  div F(U, aux)  =  S(U, aux)
D phi      =  f(U)
```

Le couplage passe a chaque pas par le canal `aux`. Le contrat de base est `(phi, grad_x, grad_y)` ; un
modele peut declarer `n_aux` pour lire des champs supplementaires (`B_z`, `T_e`).

## Table des matieres

- [Ce que fournit le coeur](#ce-que-fournit-le-coeur)
- [Ecosysteme](#ecosysteme)
- [Documentation](#documentation)
- [Plateformes supportees](#plateformes-supportees)
- [Quick start](#quick-start)
  - [Prerequis](#prerequis)
  - [Build et tests](#build-et-tests)
  - [Backends](#backends)
  - [Options CMake](#options-cmake)
- [Utiliser le coeur depuis un projet C++](#utiliser-le-coeur-depuis-un-projet-c)
- [Module Python `adc`](#module-python-adc)
  - [Systemes AMR et multi-especes](#systemes-amr-et-multi-especes)

## Ce que fournit le coeur

| Couche | Role | Point d'entree |
|---|---|---|
| `core/` | types, etat, `PhysicalModel`, `EquationBlock`, `CoupledSystem` | [physical_model.hpp](include/adc/core/physical_model.hpp) |
| `physics/` | briques generiques composees en `CompositeModel` | [composite.hpp](include/adc/physics/composite.hpp) |
| `numerics/` | reconstruction (Minmod / VanLeer / WENO5), flux (Rusanov / HLL / HLLC / Roe), operateur spatial | [reconstruction.hpp](include/adc/numerics/reconstruction.hpp) |
| `numerics/elliptic/` | concept `EllipticSolver`, multigrille geometrique, FFT, composite FAC | [elliptic_solver.hpp](include/adc/numerics/elliptic/elliptic_solver.hpp) |
| `numerics/time/` | SSP-RK, scheduler multirate, IMEX, splitting, moteur AMR | [numerics/time/](include/adc/numerics/time) |
| `coupling/` | `Coupler`, `SystemCoupler`, `AmrSystemCoupler`, `AmrCouplerMP` | [coupler.hpp](include/adc/coupling/coupler.hpp) |
| `amr/` | clustering Berger-Rigoutsos, regrid, hierarchie | [amr/](include/adc/amr) |
| `mesh/`, `parallel/` | MultiFab, BoxArray, Geometry, seam comm MPI, load balance | [mesh/](include/adc/mesh) |
| `runtime/` | facades `System` / `AmrSystem`, `model_factory`, DSL, canal aux | [system.hpp](include/adc/runtime/system.hpp) |

Carte par module et par fichier : [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Algorithmes et formules :
[docs/ALGORITHMS.md](docs/ALGORITHMS.md). Decisions de design : [docs/CHOICES.md](docs/CHOICES.md).

## Ecosysteme

| Repo | Role | Socle maillage |
|---|---|---|
| `adc_cpp` (ce depot) | coeur hyperbolique-elliptique sur AMR + GPU / MPI / Kokkos | propre |
| [`adc_cases`](https://github.com/wolf75222/adc_cases) | applications : modeles nommes, facades, exemples, Python | consomme `adc::adc` |
| [`poisson_cpp`](https://github.com/wolf75222/poisson_cpp) | solveurs Poisson (Thomas, SOR, CG, DST, multigrille) | propre |
| [`pde_core_cpp`](https://github.com/wolf75222/pde_core_cpp) | infra partagee (mesh, fields, AMR) | propre |
| [`advection_cpp`](https://github.com/wolf75222/advection_cpp) | advection + Burgers + Chorin NS | `pde_core_cpp` |
| [`euler_cpp`](https://github.com/wolf75222/euler_cpp) | Euler 2D + NS visqueux + sources plasma | `pde_core_cpp` |

## Documentation

- Guide utilisateur (Sphinx) : <https://wolf75222.github.io/adc_cpp/>
- Reference C++ (Doxygen) : <https://wolf75222.github.io/adc_cpp/cpp/>
- Canoniques : [ARCHITECTURE](docs/ARCHITECTURE.md) (couches, modules, AMR), [ALGORITHMS](docs/ALGORITHMS.md)
  (methodes, formules), [VALIDATION](docs/VALIDATION.md), [BACKEND_COVERAGE](docs/BACKEND_COVERAGE.md)
  (matrice backends / tests).
- Politique de qualite documentaire : [DOC_QUALITY](docs/DOC_QUALITY.md) (taxonomie des docs,
  decisions d'outillage, guide de mise a jour).

## Plateformes supportees

Le coeur vise trois cibles de premier rang -- la CI Linux (gate de chaque PR), le poste de dev macOS
et la production GPU sur ROMEO -- plus une cible Windows en cours via WSL2. Couverture backend detaillee
par cible : [docs/BACKEND_COVERAGE.md](docs/BACKEND_COVERAGE.md).

| Plateforme | Role | Compilateur(s) | Backends valides |
|---|---|---|---|
| `x86_64-linux` (ubuntu-latest) | CI (gate + quality) | gcc (defaut runner), clang (job tidy ; fuzz des la vague 2 qualite) | Kokkos Serial + OpenMP, MPI |
| `aarch64-darwin` (macOS) | dev local | AppleClang (build) ; LLVM Homebrew pour clang-format/clang-tidy/fuzzing | Kokkos Serial + OpenMP, MPI conda |
| `aarch64-linux + CUDA` (GH200 / cluster ROMEO) | production HPC | gcc + nvcc CUDA 12.x | Kokkos CUDA, MPI multi-GPU |
| `WSL2 Ubuntu` (Windows 11) | en cours (epic ADC-90) | comme Linux | port Windows v1 |

### Pieges connus par plateforme

- **macOS** : l'ASan du LLVM Homebrew deadlocke AVANT `main` (initialisation re-entrante via dyld) ;
  `MallocNanoZone=0` ne suffit pas. L'ASan d'AppleClang, lui, fonctionne (preset `ci-asan`, cf.
  [docs/QUALITY_TOOLING.md](docs/QUALITY_TOOLING.md)). Pour le fuzzing local (harnais `fuzz/`,
  livre par la vague 2 qualite), passer `-DADC_FUZZ_SANITIZERS=undefined` au configure.
- **nvcc CUDA 12.x** : pas de `-std=c++23` -> forcer `ADC_CXX_STD=20` sous Kokkos (impacte la cle
  d'ABI du DSL).
- **conda-forge** : le paquet Kokkos est souvent Serial-only -> `scripts/kokkos_openmp_conda.sh`
  installe un Kokkos OpenMP dans l'env conda actif.
- **Runners GitHub** : MPI exige l'oversubscribe (`OMPI_MCA_rmaps_base_oversubscribe=true`,
  `PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe`).

Hors perimetre : pas de 32 bits, pas de big-endian, arithmetique flottante IEEE 754 supposee ; le
portage Windows natif est differe en v2 (epic ADC-90).

## Quick start

### Prerequis

- C++20 : AppleClang 16+, GCC 13+, Clang 17+ (nvcc_wrapper pour la cible Cuda)
- CMake >= 3.21 (le build se pilote par presets, `CMakePresets.json`)
- Kokkos 4.2+ : le SEUL backend on-node, **obligatoire** -- mais **pas besoin de le pre-installer** :
  s'il n'est pas trouve, CMake le recupere + construit automatiquement (FetchContent)
- MPI *(optionnel, `-DADC_USE_MPI=ON` : halos + FFT distribuee)*
- HDF5 parallele *(optionnel, `-DADC_USE_HDF5=ON` : DataWriter)*
- Python 3.12 + numpy *(optionnel, bindings `adc` ; env conda via `scripts/setup_env.sh`)*

### Build et tests

Le coeur C++ seul (sans le module Python), via les presets :

```bash
git clone https://github.com/wolf75222/adc_cpp.git
cd adc_cpp
cmake --preset serial          # configure (binaryDir : build/)
cmake --build --preset serial
ctest --preset serial
```

### Backends

adc_cpp est **Kokkos-only** : le dispatch on-node passe par un seul seam (`for_each_cell`) compile
vers Kokkos, et il n'existe plus de build non-Kokkos (le backend OpenMP autonome `ADC_USE_OPENMP` a
ete retire ; le serie passe par Kokkos Serial). La cible on-node se choisit par les options Kokkos
(`Kokkos_ENABLE_SERIAL` / `_OPENMP` / `_CUDA` / `_HIP`) -- a la config (chemin FetchContent) ou a
l'install du Kokkos pointe par `-DKokkos_ROOT`.

```bash
cmake --preset serial      # Kokkos Serial (FetchContent si non installe) -> build/
cmake --preset mpi         # + distribue (halos + FFT par MPI)            -> build-mpi/
cmake --preset parallel    # Kokkos CPU multi-thread (env conda actif)    -> build-kokkos/
cmake -B build-gpu -DCMAKE_CXX_COMPILER=$K/bin/nvcc_wrapper -DKokkos_ROOT=$K  # GPU GH200 (ROMEO)
```

Chaque option s'accepte aussi en variable d'environnement (`ADC_USE_KOKKOS=ON cmake ...`, ou via
`pip install .`) ; un `-D` explicite garde la priorite.

### Options CMake

| Option | Defaut | Role |
|---|---|---|
| `ADC_BUILD_TESTS` | `ON` | suite CTest du coeur |
| `ADC_BUILD_PYTHON` | `OFF` | module pybind11 `adc` |
| `ADC_USE_MPI` | `OFF` | backend distribue (comm, halos, FFT) |
| `ADC_USE_KOKKOS` | `ON` | backend on-node Kokkos, **obligatoire** (OFF = erreur fatale) |
| `ADC_USE_HDF5` | `OFF` | DataWriter HDF5 parallele |
| `ADC_BUILD_BENCH` | `OFF` | harnais de profilage (`bench/`) |

La CI (presets `ci-*`) couvre Kokkos Serial (gate de chaque PR : C++ + suite Python), MPI + Kokkos
Serial (bit-identique np=1/2/4, en `ci-full`) et Kokkos OpenMP (en `ci-full`), tout en CPU. Les chemins CUDA (Kokkos GPU, MPI + GPU) sont valides a la main sur
ROMEO GH200, jamais en CI ; couverture detaillee : [docs/VALIDATION.md](docs/VALIDATION.md).

## Utiliser le coeur depuis un projet C++

```cmake
include(FetchContent)
FetchContent_Declare(adc_cpp GIT_REPOSITORY https://github.com/wolf75222/adc_cpp.git)
FetchContent_MakeAvailable(adc_cpp)   # les tests d'adc_cpp ne sont pas compiles chez le consommateur
target_link_libraries(mon_appli PRIVATE adc::adc)
```

On definit un type qui satisfait le concept `PhysicalModel`, on l'instancie dans un
`Coupler<Model, Elliptic>` (ou `AmrCouplerMP` pour l'AMR) et on avance en temps.

## Module Python `adc`

`scripts/setup_env.sh` cree l'env conda (CMake, Ninja, NumPy, pybind11, Kokkos) et y fige la toolchain de
la plateforme. L'installation tient alors en une commande : `pip install .` (scikit-build-core) pilote le
CMakeLists, et les backends se choisissent par variables d'environnement.

```bash
bash scripts/setup_env.sh                                       # env conda + toolchain
pip install .                                                   # serie
ADC_USE_KOKKOS=ON Kokkos_ROOT=$CONDA_PREFIX pip install . -v    # parallele (Kokkos)
```

Ensuite, sans `PYTHONPATH` ni variable a exporter :

```python
import adc
adc.set_threads()          # tous les coeurs (ou set_threads(8)), avant le 1er System
sim = adc.System(n=256)    # adc.doctor() diagnostique l'environnement en cas de doute
```

Pour iterer sur le C++ sans reinstaller : `cmake --build --preset python` (serie) ou `python-parallel`
(Kokkos), puis `PYTHONPATH=$PWD/build-py/python` (resp. `build-py-kokkos/python`).

> L'extension est epinglee a l'interpreteur qui l'a construite (`cpython-312`) : construire et importer
> avec le meme Python. En cas d'erreur d'import, le message indique la cause et la commande de
> reconstruction. Detail : [installation](docs/sphinx/getting_started/installation.md).

Un modele Python s'ecrit de deux facons equivalentes, branchees de la meme maniere sur `adc.System` /
`adc.AmrSystem` :

- **briques** : on compose des briques deja compilees (`adc.Model`), sans compilation a la volee ;
- **formules (DSL)** : on ecrit le modele en formules symboliques (`adc.dsl.Model`), traduites en C++
  puis compilees en `.so`. La sortie est bit-identique a la composition de briques.

Exemple minimal, le diocotron reduit en briques (densite scalaire advectee par la derive E x B, fond
neutralisant) :

```python
import adc
model = adc.Model(state=adc.Scalar(),
                  transport=adc.ExB(B0=1.0),
                  source=adc.NoSource(),
                  elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0))
sim = adc.System(n=96, L=1.0, periodic=True)
sim.add_block("ne", model=model, spatial=adc.Spatial(minmod=True), time=adc.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
sim.set_density("ne", ne0)          # ne0 : densite initiale (tableau 2D)
sim.step_cfl(0.4)
```

Tutoriel pas a pas (briques et formules) :
[getting_started/tutorial](docs/sphinx/getting_started/tutorial.md). Reference :
[bricks_reference](docs/sphinx/reference/bricks_reference.md),
[dsl_reference](docs/sphinx/reference/dsl_reference.md). Les quatre chemins de modele (natif compose,
production, aot, prototype) et leurs limites : [docs/DSL_MODEL_DESIGN.md](docs/DSL_MODEL_DESIGN.md).

### Systemes AMR et multi-especes

`adc.AmrSystem` compose un ou plusieurs blocs sur une hierarchie raffinee (API proche de `System`, plus
`set_refinement`). Il gere le mono- et le multi-bloc, la cadence de regrid, le reflux conservatif, la
reconstruction `conservative|primitive`, les flux `rusanov|hllc|roe` et une source IMEX locale. Le chemin
fidelite AMR ajoute un solveur elliptique composite (FAC) et un etage source condense par Schur global
(`set_source_stage`). Detail : [docs/DSL_MODEL_DESIGN.md](docs/DSL_MODEL_DESIGN.md).

Pour coupler N especes (ions, electrons, neutres), chacune a son `PhysicalModel`, son schema spatial et
sa politique temporelle. Les interactions vivent dans le second membre elliptique (`ChargeDensityRhs`) et
dans la source (`CoupledSource`), jamais dans le flux. Le scheduler gere sous-pas, cadence, IMEX partiel
et pas adaptatif multirate (`step_adaptive`). Voir [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

BSD-3-Clause. Voir [LICENSE](LICENSE).
