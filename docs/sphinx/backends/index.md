# Backends paralleles

`adc_cpp` cible une seule pile de calcul (mesh + transport + Poisson + AMR), mais
cette pile peut s'executer sur six configurations paralleles : du sequentiel mono-thread
jusqu'au multi-GPU Grace-Hopper distribue par MPI. Le point cle de conception est qu'aucun
operateur ne change d'une configuration a l'autre : tout le parallelisme est confine a
deux seams (coutures de dispatch). Le backend est choisi a la compilation, par des
options CMake.

Cette page decrit chaque configuration : ce qu'elle est, la commande de build, comment
la lancer, et son statut de validation (teste en CI ou valide manuellement sur
ROMEO). Pour la matrice de couverture test par test, voir
[BACKEND_COVERAGE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md) (source de verite unique) ; pour le portage
GPU phase par phase, voir [GPU_RUNTIME_PORT.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/GPU_RUNTIME_PORT.md).

## Le modele : deux seams, MPI + Kokkos

Il n'y a pas "trois couches" empilees. L'architecture est MPI + Kokkos :

- **MPI** distribue les sous-domaines entre rangs (un GPU par rang en mode GPU). Tout passe
  par `my_rank()` / `n_ranks()` / `all_reduce_*` de
  [`include/adc/parallel/comm.hpp`](https://github.com/wolf75222/adc_cpp/blob/master/include/adc/parallel/comm.hpp). Sans
  `ADC_HAS_MPI`, ces fonctions renvoient rang 0 / 1 rang : le code est serie par
  construction.
- **Kokkos** parallelise le calcul local et abstrait le materiel via son `ExecutionSpace` :
  backend `Cuda` pour GPU NVIDIA, `Serial`/`OpenMP` pour CPU. Tout passe par `for_each_cell`
  (et `for_each_cell_reduce_*`) de
  [`include/adc/mesh/for_each.hpp`](https://github.com/wolf75222/adc_cpp/blob/master/include/adc/mesh/for_each.hpp), qui bascule
  CPU <-> GPU a la compilation sans toucher les sites d'appel.

On n'ecrit aucun kernel CUDA a la main : le meme `.cpp` cible CPU et GPU selon le backend
Kokkos actif a la compilation. `nvcc_wrapper` n'est que le compilateur exige par le backend
Cuda de Kokkos.

> **Le module Python `adc` est serie par defaut.** L'extension `_adc` (pybind11) n'est
> construite en CI qu'en Kokkos Serial (sans MPI). Aucun test Python n'exerce les chemins
> Kokkos OpenMP, Cuda ou MPI. Le multi-thread, le GPU et le distribue se pilotent depuis la
> facade C++ (`System` / `AmrSystem`), pas depuis Python.

## Les options CMake

Verifiees dans [`CMakeLists.txt`](../../../CMakeLists.txt) :

| Option CMake | Effet | Defaut |
|--------------|-------|--------|
| `ADC_USE_KOKKOS` | Seul backend on-node, obligatoire (CPU Serial/OpenMP + GPU Cuda/HIP). Configurer avec `OFF` est une erreur fatale CMake. | `ON` |
| `ADC_USE_MPI` | Seam comm distribue (`comm.hpp` -> collectives MPI). | `OFF` |
| `ADC_BUILD_PYTHON` | Module Python `adc` (pybind11), serie uniquement. | `OFF` |

Le sous-backend Kokkos (Serial / OpenMP / Cuda) n'est pas une option `adc_cpp` : il est
choisi au moment ou l'on installe Kokkos (`Kokkos_ENABLE_SERIAL`, `Kokkos_ENABLE_OPENMP`,
`Kokkos_ENABLE_CUDA` + `Kokkos_ARCH_HOPPER90`), puis pointe par `-DKokkos_ROOT=...`. C'est ce
qui distingue les configurations 1/2/5 ci-dessous.

Notes :

- Kokkos est le seul backend on-node et il est obligatoire : configurer sans lui
  (`-DADC_USE_KOKKOS=OFF`) est une erreur fatale CMake, et le seam
  [`for_each.hpp`](https://github.com/wolf75222/adc_cpp/blob/master/include/adc/mesh/for_each.hpp)
  ne compile pas sans `ADC_HAS_KOKKOS` (`#error`).
- **Kokkos n'a pas besoin d'etre pre-installe** : CMake fait `find_package(Kokkos)` puis, a defaut,
  le recupere + construit via FetchContent (version `ADC_KOKKOS_FETCH_VERSION`, defaut 4.4.01, tarball verifie par SHA256). Les commandes
  `-DKokkos_ROOT=...` ci-dessous reutilisent une install (plus rapide) ; sans elles, Kokkos est fetch.
- La norme C++ est C++20 (nvcc CUDA 12.x ne propose pas `-std=c++23`).

---

## 1. Kokkos Serial

**Ce que c'est.** Le build de reference : Kokkos Serial (CPU mono-thread via
`Kokkos::parallel_for`), sans MPI. `comm.hpp` repond rang 0 / 1 rang. C'est l'oracle : tous les
autres backends sont valides bit-a-bit (ou a l'arrondi pres) contre lui. Le serie passe par
Kokkos Serial, pas par une boucle C++ ecrite a la main. C'est ce job qui avait rattrape une
regression d'init (allocation d'un `Fab` avant l'init paresseuse de Kokkos) ; le gate Kokkos
Serial joue maintenant ce role sur CHAQUE PR.

**Build.** Il faut un Kokkos installe avec `Kokkos_ENABLE_SERIAL=ON`, puis :

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DADC_USE_KOKKOS=ON -DKokkos_ROOT="$KOKKOS_PREFIX"
cmake --build build -j
```

**Run.**

```bash
ctest --test-dir build --output-on-failure
```

**Validation : CI (gate obligatoire de chaque PR).** C'est le job `build-and-test`
(`ubuntu-latest`, Kokkos Serial), declenche sur tout `pull_request`. Couvre les 109 cibles
ctest hors-MPI plus les 60 tests Python (construits via `-DADC_BUILD_PYTHON=ON`, module
`_adc` Kokkos Serial). Statut `ci-fast` dans la matrice.

---

## 2. Kokkos OpenMP

**Ce que c'est.** Meme backend Kokkos, espace d'execution `OpenMP` : parallelisme multi-thread
sur CPU. `for_each_cell` devient un `parallel_for` multi-thread (avec un seuil de bascule serie
pour les petites grilles du V-cycle, cf. `ADC_FOREACH_SERIAL_THRESHOLD`).

**Build.** Kokkos installe avec `Kokkos_ENABLE_OPENMP=ON`. Le `-DADC_USE_KOKKOS=ON` cote
`adc_cpp` est identique a la config 1 ; c'est l'install Kokkos qui change :

```bash
cmake -S . -B build-kokkos-omp -DCMAKE_BUILD_TYPE=Release \
  -DADC_USE_KOKKOS=ON -DKokkos_ROOT="$KOKKOS_OPENMP_PREFIX"
cmake --build build-kokkos-omp -j
```

**Run.** Borner le nombre de threads sur les petites machines :

```bash
OMP_NUM_THREADS=4 ctest --test-dir build-kokkos-omp --output-on-failure
```

**Validation : CI (job `ci-full`).** Job `kokkos-openmp`
(`ubuntu-latest / Kokkos (OpenMP)`), Kokkos 4.4.01 OpenMP, `OMP_NUM_THREADS=2`. Mode plein
uniquement (a la difference du gate Serial, qui tourne sur chaque PR). Statut `ci-full`,
91/91 ctest, 0 echec.

> **Note FP.** La reduction somme (`Kokkos::Sum`) reassocie l'addition flottante par tuile :
> deterministe/idempotent (memes donnees + meme espace Kokkos -> memes bits) mais pas
> bit-identique a une somme lexicographique ecrite a la main. Comme il n'y a qu'un seul chemin
> Kokkos, cela vaut pour tous les espaces (Serial, OpenMP, Cuda). La reduction max
> (`Kokkos::Max`) est exacte partout. Detail dans l'en-tete de
> [`for_each.hpp`](https://github.com/wolf75222/adc_cpp/blob/master/include/adc/mesh/for_each.hpp).

---

## 3. MPI + Kokkos Serial (MPI CPU)

**Ce que c'est.** Build distribue avec l'espace d'execution Kokkos Serial : `comm.hpp` passe
par `MPI_Comm_rank/size` + collectives sur `MPI_COMM_WORLD`. Le domaine est decoupe en boites
reparties sur les rangs ; les halos s'echangent par `fill_boundary` cross-rang, le reflux/masse
par `all_reduce_*`. CPU mono-thread par rang.

**Build.** Kokkos installe avec `Kokkos_ENABLE_SERIAL=ON`, plus le seam MPI :

```bash
cmake -S . -B build-mpi -DCMAKE_BUILD_TYPE=Release \
  -DADC_USE_KOKKOS=ON -DADC_USE_MPI=ON -DKokkos_ROOT="$KOKKOS_PREFIX"
cmake --build build-mpi -j
```

**Run.** Les cibles MPI rejouent chacune np=1/2/4 sous `mpirun`. Pour `-np 4` sur une petite
machine, autoriser l'oversubscribe :

```bash
OMPI_MCA_rmaps_base_oversubscribe=true \
  ctest --test-dir build-mpi --output-on-failure
```

**Validation : CI (job `ci-full`).** Job `mpi` (`ubuntu-latest / MPI`, OpenMPI). Mode plein
uniquement. Verifie l'invariance au nombre de rangs : les observables (parite, AMR,
Krylov, masse) sont bit-identiques a np=1/2/4. Statut `ci-full` sur les ~21 entrees du bloc
`ADC_HAS_MPI` ; les tests non-MPI tournent a np=1 dans ce build (lies MPI, mono-process).

---

## 4. MPI + Kokkos OpenMP

**Ce que c'est.** Hybride distribue CPU : MPI entre les noeuds/rangs, Kokkos OpenMP pour le
multi-thread intra-rang. C'est le mode CPU "plein" (tous les coeurs de tous les rangs).

**Build.** Les deux options a la fois, sur un Kokkos OpenMP :

```bash
cmake -S . -B build-mpi-omp -DCMAKE_BUILD_TYPE=Release \
  -DADC_USE_MPI=ON -DADC_USE_KOKKOS=ON -DKokkos_ROOT="$KOKKOS_OPENMP_PREFIX"
cmake --build build-mpi-omp -j
```

**Run.**

```bash
OMP_NUM_THREADS=4 OMPI_MCA_rmaps_base_oversubscribe=true \
  ctest --test-dir build-mpi-omp --output-on-failure
```

**Validation : ROMEO-manuel (noeud `x64cpu`).** Cette combinaison n'est pas dans la CI (la
CI joint MPI a Kokkos Serial, mais jamais MPI a Kokkos OpenMP). Validee a la main sur le noeud `x64cpu` de
ROMEO : 52/57 runs rank-invariants (bit-identiques np=1/2/4, dmax=0 sur les observables
parite/AMR/Krylov). Reserve : 3 tests distribues-MG lourds (`mpi_cutcell_multibox`,
`mpi_amr_distributed_coarse`, `condensed_schur_source_stepper`) sont trop lents a np>1
(depassent 600 s) ; pathologie de performance (petites tuiles + halos MPI, ~5-7x de
ralentissement), pas un deadlock ni un bug de correction. Tous passent a np=1.

---

## 5. Kokkos CUDA (ROMEO / GH200 uniquement)

**Ce que c'est.** Backend Kokkos avec l'espace d'execution `Cuda` : `for_each_cell` devient un
`Kokkos::parallel_for` qui s'execute sur le GPU. Le meme code que les configs 1/2 ; seul le
backend Kokkos change. Les `Fab` vivent en memoire unifiee (`Kokkos::SharedSpace`), donc
device-accessibles par construction ; `for_each_cell` est async sous Cuda, d'ou un
`device_fence()` (via `sync_host()`) avant toute lecture hote.

**Pas de build local.** Il n'y a pas de `nvcc` sur les postes de dev ; `nvcc` ne tourne que
sur le noeud GPU (aarch64) de ROMEO, pas sur le login (x86). La CI ne construit jamais avec
CUDA : toutes les cellules "Kokkos Cuda" de la matrice sont soit ROMEO-manuel, soit `?`.

**Build (sur ROMEO, noeud `armgpu`).** Kokkos installe avec
`Kokkos_ENABLE_CUDA=ON -DKokkos_ARCH_HOPPER90=ON`, compilateur `nvcc_wrapper` :

```bash
module load cuda/12.6
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DADC_USE_KOKKOS=ON \
  -DCMAKE_CXX_COMPILER="$KOKKOS_PREFIX/bin/nvcc_wrapper" \
  -DKokkos_ROOT="$KOKKOS_PREFIX"
cmake --build build-cuda -j
```

**Run (SLURM, un GPU).**

```bash
srun --account=<compte> -p instant --constraint=armgpu --gres=gpu:1 \
  ./build-cuda/bin/<harness>
```

**Validation : ROMEO-manuel (jamais en CI).** Les harnesses GPU vivent dans
[`python/tests/gpu/*.cpp`](https://github.com/wolf75222/adc_cpp/tree/master/python/tests/gpu) (hors du graphe ctest, lances par
sbatch/`srun`). Chacun compile la meme logique en `exec=Cuda` et en oracle `exec=Serial`,
puis compare cellule par cellule (`dmax = max|cuda - serial|`). Resultats sur GH200
(Kokkos 4.4.01, `Kokkos_ARCH_HOPPER90`) :

- Solveur mono-grille complet (transport + BCs + couplages + Poisson + pas de temps,
  orchestre par le `System`) : bit-identique CPU (phases 1-5, 7).
- Briques elliptiques post-#48 (T_e via `load_aux<5>`, EPM ecrante/Helmholtz, EPM anisotrope,
  B_z par niveau AMR) : `dmax = 0` vs Serial, memes cycles MG.

**Reserves.** Le chemin `System::add_compiled_model` (modele DSL natif zero-copie)
butait sur une limite `nvcc` (lambdas etendues `__host__ __device__` cross-TU), contournee par
des foncteurs nommes (le chemin device `assemble_rhs` / `advance_amr`), mais
`test_compiled_model_parity` lui-meme n'est pas encore porte device. Le capstone AMR
multi-blocs (7 tests) reste `?` sur Cuda (foncteurs nommes, en principe nvcc-compatibles, mais
sans harness ROMEO dedie).

---

## 6. MPI + Kokkos CUDA (ROMEO multi-GPU uniquement)

**Ce que c'est.** Le mode production cible : MPI distribue les sous-domaines (un GPU par rang),
Kokkos Cuda calcule sur chaque GPU, OpenMPI CUDA-aware echange les halos device-to-device
(UCX). C'est la config 5 + la config 3 dans un seul run.

**Pas de build local** (memes contraintes que la config 5 : `nvcc` ROMEO-only, jamais en CI).

**Build (sur ROMEO).** Les deux options, OpenMPI CUDA-aware :

```bash
module load cuda/12.6
cmake -S . -B build-mpicuda -DCMAKE_BUILD_TYPE=Release \
  -DADC_USE_KOKKOS=ON -DADC_USE_MPI=ON \
  -DCMAKE_CXX_COMPILER="$KOKKOS_PREFIX/bin/nvcc_wrapper" \
  -DKokkos_ROOT="$KOKKOS_PREFIX"
cmake --build build-mpicuda -j
```

**Run (SLURM, plusieurs GPU, un par rang).**

```bash
srun -n 4 --gpus-per-task=1 --constraint=armgpu ./build-mpicuda/bin/<harness>
```

**Validation : ROMEO-manuel multi-GPU (jamais en CI).** Sur un noeud a 4x GH200 (OpenMPI 4.1.7
CUDA-aware), valide np=1/2/4 (np=1 = oracle mono-GPU). Acquis :

- 10 tests de la pile elliptique / Schur(stepper) / Poisson / system-solve / AMR
  rank-invariants : `dmax` cross-np = 0 (krylov_solver, mpi_poisson,
  mpi_system_solve_fields, mpi_amr_compiled_parity, mpi_amr_distributed_coarse,
  mpi_coupled_source, mpi_mbox_parity, mpi_cutcell_multibox, condensed_schur_source_stepper,
  test_schur_condensation cote invariance).
- Validation integree AmrSystem + MPI + GPU dans un seul run (phase 10) : densite
  grossiere bit-identique a np=1/2/4 (`dmax = 0`), masse conservee a 0.

**Reserves.** (a) Le run integre ne scale pas : le grossier est replique par
defaut (calcul redondant) ; le mode grossier-reparti (`distribute_coarse`) est correct et
bit-identique mais ~3.7-5x plus lent (le trafic de halos cross-rang domine le compute
economise), resultat negatif chiffre, documente. (b) `test_schur_condensation` echoue cote
backend Cuda des np=1 (defaut d'assemblage device, independant du nombre de rangs) ; il passe
en Serial / Kokkos Serial. (c) Sur grossier reparti, les sommes globales different a l'arrondi
entre np (ordre de reduction FMA, ~9e-13) ; le max reste bit-identique.

---

## Matrice recapitulative

| # | Backend | Options CMake (en plus de `-DCMAKE_BUILD_TYPE=Release`) | Build local ? | CI ? | Valide ou |
|---|---------|---------------------------------------------------------|---------------|------|-----------|
| 1 | Kokkos Serial | `-DADC_USE_KOKKOS=ON` + Kokkos Serial | Oui | Oui (`ci-fast`, gate PR) | CI ubuntu (oracle de reference) |
| 2 | Kokkos OpenMP | `-DADC_USE_KOKKOS=ON` + Kokkos OpenMP | Oui | Oui (`ci-full`) | CI job `kokkos-openmp`, 91/91 ctest |
| 3 | MPI + Kokkos Serial | `-DADC_USE_KOKKOS=ON -DADC_USE_MPI=ON` + Kokkos Serial | Oui | Oui (`ci-full`) | CI job `mpi`, rank-invariant np=1/2/4 |
| 4 | MPI + Kokkos OpenMP | `-DADC_USE_MPI=ON -DADC_USE_KOKKOS=ON` + Kokkos OpenMP | Oui | Non (jamais MPI+Kokkos en CI) | ROMEO `x64cpu` manuel (52/57 rank-invariants) |
| 5 | Kokkos CUDA | `-DADC_USE_KOKKOS=ON` + Kokkos Cuda + `nvcc_wrapper` | Non (pas de `nvcc` local) | Non (jamais CUDA en CI) | ROMEO GH200 manuel (`python/tests/gpu/`, sbatch) |
| 6 | MPI + Kokkos CUDA | `-DADC_USE_MPI=ON -DADC_USE_KOKKOS=ON` + Kokkos Cuda + `nvcc_wrapper` | Non | Non | ROMEO multi-GPU manuel (`srun --gpus-per-task=1`) |

**Lecture rapide :**

- Les configs 1-3 sont couvertes par la CI GitHub (1 sur chaque PR ; 2-3 en mode plein
  `ci-full` : push `master`, cron, dispatch, ou label `ci-full`).
- Les configs 4-6 ne sont jamais en CI : la CI joint MPI a Kokkos Serial mais jamais a Kokkos
  OpenMP, et ne construit jamais CUDA. Elles sont validees manuellement sur ROMEO, par comparaison
  bit-a-bit a l'oracle Serial (`dmax`).
- Les configs 5-6 n'ont pas de build local : `nvcc` ne tourne que sur le noeud GPU
  aarch64 de ROMEO.

Pour la couverture detaillee (chaque test x chaque colonne backend, avec le statut
`ci-fast` / `ci-full` / `ROMEO` / `self-skip` / `?`), la source de verite reste
[BACKEND_COVERAGE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md). Pour les phases du portage GPU et les
resultats de validation detailles, voir [GPU_RUNTIME_PORT.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/GPU_RUNTIME_PORT.md).
