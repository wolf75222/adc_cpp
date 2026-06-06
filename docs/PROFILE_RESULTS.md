# Profil d'un pas de temps representatif (mesure seule)

Date : 2026-06-06. Branche : `feat/profiling-harness`. Auteur de la mesure : harnais
`bench/profile_step` (cf. `bench/`), construit HORS du build par defaut (`-DADC_BUILD_BENCH=ON`).

Ce document RAPPORTE des mesures. Il N'APPLIQUE AUCUNE optimisation et NE recommande aucun
changement de code au-dela d'une piste a investiguer, conformement a la regle du proprietaire :
"aucun refactor de performance sans profil montrant le goulot". Les fichiers historiques
`docs/PERFORMANCE.md` et `docs/BACKEND_COVERAGE.md` ne sont PAS touches.

## 1. Methode

### Ce qui est mesure

Le harnais reconstruit, a partir des SEAMS PUBLICS de la bibliotheque (en-tetes seuls, sans toucher
`python/system.cpp` ni aucun en-tete du hot path), un pas de temps REPRESENTATIF du cas diocotron tel
que l'orchestre `System::step` :

- modele `CompositeModel<ExBVelocity, NoSource, ChargeDensity>` (advection ExB scalaire + densite de
  charge `q n` au second membre du Poisson) -- la composition exacte du diocotron ;
- `solve_fields` : assemblage du second membre `f = q n`, solve elliptique, derivation
  `(phi, grad phi)` vers le canal aux, remplissage des halos de aux ;
- avance SSPRK2 du bloc : 2 etages, chacun `fill_ghosts(U)` + `assemble_rhs<Minmod, Rusanov>` (operateur
  volumes finis) + combinaison lineaire ;
- pas CFL : reduction `max_wave_speed_mf` (max de la vitesse d'onde, `all_reduce_max` sous MPI) ;
- une reduction `dot(U, U)` (brique des produits scalaires Krylov / diagnostics).

Chaque PHASE est chronometree separement (`std::chrono`, encadree de `device_fence()` pour capturer
l'execution device reelle sous Kokkos, qui est asynchrone) :

| phase | contenu |
|-------|---------|
| `transport`  | `assemble_rhs<L,F>` (operateur FV) + combinaisons SSPRK |
| `poisson`    | solve elliptique (`GeometricMG::solve()` = V-cycles, ou `PoissonFFTSolver::solve()`) |
| `halos`      | `fill_boundary` / `fill_ghosts` (echange MPI + ghosts physiques) sur U et aux |
| `aux_derive` | assemblage `f = q n` + derivation `(phi, grad phi)` -> aux (boucles hote par cellule) |
| `reduction`  | `max_wave_speed_mf` (CFL) + `dot` |
| `fence`      | cout d'une `device_fence()` isolee (~0 hors Kokkos) |
| `alloc_tmp`  | (re)allocation des MultiFab temporaires par pas (residu R, etage SSPRK U1) |

### Cas de reference

Grille 256x256, UNE box repartie en round-robin `DistributionMapping(1, n_ranks())` -- C'EST le layout
de `System` (qui ne decoupe pas la box). `bc=periodic`, `solver=geometric_mg`, `limiter=minmod`,
`cfl=0.4`, 5 pas de chauffe + 50 pas mesures (sauf indication contraire). Sous MPI le temps rapporte
par phase est le MAX sur les rangs (chemin critique d'un pas collectif).

### Plateformes exactes

- CPU local : Apple M-series (macOS, AppleClang 21), 8 coeurs logiques. OpenMPI 5.0.9 (Homebrew).
  Kokkos Homebrew (device OpenMP, sans CUDA). NB : c'est un portable, pas un noeud de calcul ; les
  chiffres ABSOLUS y sont indicatifs, les PARTS (%) et TENDANCES sont robustes.
- GPU : ROMEO 2025, noeud `armgpu` `romeo-a057`, NVIDIA GH200 120GB (Grace-Hopper, aarch64),
  CUDA 12.6, Kokkos 4.x (SERIAL;CUDA, ARCH HOPPER90, nvcc_wrapper), OpenMPI 4.1.7 CUDA-aware.

## 2. Tableau de profil (phase x backend, ms par pas et %)

Toutes les valeurs en MILLISECONDES PAR PAS. `(pct)` = part de la phase dans le pas. 256x256.

| phase \\ backend | Serie CPU | Kokkos OMP t=1 | Kokkos OMP t=4 | Kokkos OMP t=8 | Kokkos Cuda GH200 | MPI CPU np=2 | MPI CPU np=4 | MPI+Cuda np=2 | MPI+Cuda np=4 |
|---|---|---|---|---|---|---|---|---|---|
| transport  | 1.29 | 1.29 | 0.49 | 0.57 | 0.57 | 3.62 | 3.88 | 0.40 | 0.40 |
| poisson    | 138.08 | 169.41 | 1301.80 | 3378.23 | 261.14 | 259.91 | 912.04 | 284.66 | 284.31 |
| halos      | 0.01 | 0.02 | 0.59 | 1.74 | 0.39 | 0.02 | 0.09 | 0.30 | 0.29 |
| aux_derive | 0.08 | 0.08 | 0.09 | 0.08 | 1.06 | 0.26 | 0.44 | 0.86 | 0.86 |
| reduction  | 0.22 | 0.22 | 0.13 | 0.17 | 0.14 | 6.08 | 16.53 | 1.61 | 1.60 |
| fence      | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| alloc_tmp  | 0.01 | -- | -- | -- | 0.24 | -- | -- | 0.20 | 0.19 |
| **TOTAL**  | **139.68** | **171.03** | **1303.14** | **3380.82** | **263.53** | **269.92** | **933.02** | **288.03** | **287.67** |
| poisson %  | 98.9% | 99.0% | 99.9% | 99.9% | 99.1% | 96.3% | 97.8% | 98.8% | 98.8% |

(Serie CPU TOTAL 139.68 ms correspond au run direct ; la colonne `MPI CPU np=1` mesuree separement
donne 136.83 ms, identique a la verticale serie a la variance pres.)

### Sensibilite (Serie CPU, 30 pas)

| variation | per_step (ms) | poisson % |
|---|---|---|
| 256, geometric_mg, minmod | 144.6 | 98.9% |
| 256, **fft**, minmod      | 142.9 | 98.5% |
| 256, geometric_mg, **weno5** | 158.3 | 98.9% |
| **128**, geometric_mg, minmod | 40.6 | 99.0% |
| **512**, geometric_mg, minmod | 532.2 | 98.8% |

### Scaling isole du transport vs poisson (Kokkos OMP, 512x512, weno5)

| threads | transport (ms) | poisson (ms) |
|---|---|---|
| 1 | 45.21 | 738.95 |
| 4 | 17.10 (x2.6 plus rapide) | 2229.55 (x3.0 plus LENT) |

## 3. Goulot identifie et recommandation (justifies par la mesure)

### Constat principal : le Poisson elliptique domine TOUS les backends

Sur les six backends mesures, la phase `poisson` represente **96 a 99.9 %** du temps d'un pas. Le
transport (l'operateur volumes finis `assemble_rhs`), les halos, les reductions et les allocations
temporaires sont chacun **< 1 ms par pas** au cas de reference -- ensemble < 1.1 % du pas en serie.
Le verrou de performance est donc, sans ambiguite, le **solve elliptique** (`GeometricMG::solve()`,
appele a chaque `solve_fields`, donc a chaque pas).

Deux faits chiffres aggravent ce constat :

1. **Le Poisson NE PROFITE PAS du parallelisme on-node ; il REGRESSE.** Sous Kokkos OpenMP, la phase
   poisson passe de 169 ms (1 thread) a 1302 ms (4) puis 3378 ms (8) au cas de reference -- elle
   RALENTIT d'un facteur ~20 a 8 threads. Le scaling isole (512x512) confirme la dichotomie : le
   transport accelere proprement (x2.6 sur 4 threads) tandis que le poisson ralentit (x3.0). La cause
   mesuree : le V-cycle multigrille descend jusqu'a des grilles grossieres minuscules (2x2, 4x4, ...)
   et lance un `Kokkos::parallel_for` PAR balayage de lissage sur chacune ; sur une boite de quelques
   cellules, le cout d'ouverture de la region parallele (fork/join OpenMP, ou lancement de kernel)
   ecrase le calcul utile. Le chemin SERIE de `for_each.hpp` a deja un garde-fou pour ca
   (`if (n_cells >= 4096)` avant `#pragma omp parallel for`, avec un commentaire mentionnant une
   "regression x40 mesuree sans le seuil") ; le chemin KOKKOS, lui, n'a PAS de seuil equivalent et
   dispatche un kernel quelle que soit la taille de la boite.

2. **Sur GPU, le Poisson coute encore 261 ms/pas (GH200) et ne profite d'aucun GPU supplementaire.**
   Kokkos Cuda mono-GPU : 263.5 ms/pas, poisson 99.1 %. MPI + Kokkos Cuda np=2 et np=4 : 288.0 et
   287.7 ms/pas -- IDENTIQUES au mono-GPU. Le V-cycle enchaine des dizaines de lancements de kernels
   sur des niveaux grossiers de plus en plus petits ; la LATENCE de lancement (et non la bande passante)
   domine, donc ni un GPU plus large ni des GPU en plus n'aident. La non-amelioration np=2/4 vient
   aussi du layout `System` MONO-BOX (une seule box, donc un seul rang porte le travail ; le solve
   elliptique reste collectif et chaque rang supplementaire n'ajoute que de la latence de collective).

3. **Cote MPI CPU, meme structure.** np=2 et np=4 ralentissent (270 et 933 ms/pas) au lieu d'accelerer,
   et la phase `reduction` enfle (0.22 -> 6.08 -> 16.53 ms) : c'est le surcout des collectives
   (`all_reduce` dans `dot` / `max_wave_speed_mf`) sur le decoupage mono-box, sans travail reparti en
   face. (Les chiffres np=4 CPU local sont amplifies par l'oversubscription sur 8 coeurs : la TENDANCE
   -- pas de speedup, mono-box -- est l'observable solide, pas le facteur exact.)

### Ce qu'il faut investiguer EN PREMIER (sans l'implementer ici)

Le profil pointe une seule cible prioritaire : **le solve elliptique `GeometricMG`**, et plus
precisement le COMPORTEMENT DE DISPATCH DE SON V-CYCLE SUR LES NIVEAUX GROSSIERS sous backend
parallele. Pistes a explorer, par ordre de rapport (gain attendu / risque), a chiffrer chacune par une
nouvelle mesure AVANT tout code :

1. **Seuil de dispatch parallele pour les petites boites (chemin Kokkos), pendant du garde
   `n_cells >= 4096` deja present en serie.** C'est la cause directement mesuree de la regression
   multi-thread/GPU du V-cycle. A valider : un seuil sous lequel le lissage des niveaux grossiers
   s'execute en serie (ou en un seul kernel fusionne) restaure-t-il le scaling du poisson sans changer
   le resultat numerique ? Mesure decisive : re-profiler poisson a t=1/4/8 et sur GH200 avec le seuil.

2. **Cout fixe par `solve()` : tolerance et nombre de V-cycles.** `GeometricMG::solve()` par defaut
   fait `solve(1e-8, 50)` (jusqu'a 50 V-cycles, tolerance serree) a CHAQUE pas, sans warm start explicite
   du `phi` precedent dans ce harnais. A chiffrer : combien de V-cycles sont reellement effectues par
   pas en regime etabli (le diocotron evolue lentement, `phi^n` est un excellent point de depart) ?
   Un warm start + un critere d'arret adapte reduiraient le nombre de cycles -- a MESURER avant de
   conclure.

3. **Reductions collectives sous MPI** (`dot`, `max_wave_speed_mf`) : secondaires tant que le mono-box
   du `System` n'est pas leve, mais leur croissance (reduction 0.22 -> 16.5 ms en CPU np=4) est a
   garder a l'oeil des qu'un decoupage multi-box reel repartira le travail.

Le transport, les halos, les fences et les allocations temporaires N'ONT PAS BESOIN d'optimisation au
vu de ces chiffres (< 1 % du pas). Optimiser quoi que ce soit la serait du gold-plating non justifie
par le profil.

## 4. Reproduire

```sh
# Serie CPU
bench/run_bench.sh serie
# Kokkos OpenMP (Kokkos installe avec le device OpenMP)
bench/run_bench.sh kokkos-omp  /chemin/kokkos
# MPI CPU (np=2)
bench/run_bench.sh mpi 2
# Kokkos Cuda / MPI+Cuda : sur ROMEO (GH200), cf. bench/run_bench.sh {kokkos-cuda,mpi-cuda} <Kroot> [NP]
```

Le harnais accepte `--n --steps --warmup --cfl --solver {geometric_mg|fft} --limiter
{none|minmod|vanleer|weno5} --bc {periodic|dirichlet}`. Il est HORS du build par defaut (option
`ADC_BUILD_BENCH=OFF`) : le CI ne le configure ni ne le compile jamais.

## 5. Garanties

- AUCUNE optimisation, AUCUN refactor du hot path. `python/system.cpp` et les en-tetes du chemin chaud
  ne sont PAS modifies. Seuls ajouts : `bench/` (nouveau) et l'option `ADC_BUILD_BENCH` (OFF par
  defaut) + un `add_subdirectory(bench)` garde dans le `CMakeLists.txt` racine.
- `docs/PERFORMANCE.md` (historique) et `docs/BACKEND_COVERAGE.md` ne sont PAS touches.
