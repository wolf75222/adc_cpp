# Audit perf scaling et frontends ADC - 2026-06-08

Audit redige le 2026-06-08 14:47 CEST.

But : definir une campagne reproductible pour mesurer le scaling CPU/GPU/MPI
d'`adc_cpp`, et estimer ou se perd la performance entre C++ natif, Python
briques natives et Python DSL `production`. Hoffart ne doit pas servir de cas
benchmark principal ici : il reste un cas scientifique delicat, pas un banc de
performance neutre.

Ce document est volontairement separe de `docs/PAPER_ROADMAP.md` et de
`docs/HOFFART_GEOMETRY_VERDICT.md`. Il parle de couts runtime, de frontieres
Python/C++, de scaling et d'instrumentation.

## 1. Etat Git fige

Les checkouts locaux sont en retard au moment de l'audit. Les mesures doivent
donc etre etiquetees par commit exact et ne doivent pas melanger `HEAD` local,
`origin/master` et branches de PR.

| repo | local HEAD | origin/master | statut local |
|---|---:|---:|---|
| `adc_cpp` | `0187329` | `075255b` | local derriere `origin/master` de 18 commits |
| `adc_cases` | `1affec1` | `6483e37` | local derriere `origin/master` de 16 commits |

Branches/PR ouvertes a isoler des mesures `master` :

| repo | PR | branche | role perf/science |
|---|---:|---|---|
| `adc_cpp` | #239 | `feat/isothermal-hll-flux` | flux HLL pour modeles 3 variables ; peut changer cout transport et robustesse |
| `adc_cpp` | #238 | `docs/hoffart-spatial-diagnostics` | documentation Hoffart ; hors benchmark perf neutre |
| `adc_cpp` | #237 | `feat/gauss-policy` | politique de Gauss ; peut changer cout et dynamique des runs couples |
| `adc_cpp` | #232 | `docs/amr-condensed-schur-design` | design Schur AMR ; pas une cible runtime a mesurer comme implementation |
| `adc_cases` | #30 | `docs/hoffart-imre-case-verdict` | diagnostic Hoffart ; hors benchmark perf neutre |

Regle de publication : chaque tableau de perf doit commencer par :

```text
adc_cpp_commit=<sha>
adc_cases_commit=<sha ou n/a>
backend=<serial|kokkos-openmp|kokkos-cuda|mpi+kokkos-openmp|mpi+kokkos-cuda>
compiler=<id/version>
kokkos=<version/devices>
mpi=<implementation/version/cuda-aware?>
machine=<CPU/GPU/node>
case=<euler-periodic|poisson-mms|halo|amr-synthetic|frontend-euler>
```

## 2. Verdict anti-MUFFIN

Le probleme signale par le tuteur sur MUFFIN est reel, mais il faut le nommer
precisement : ce n'est pas seulement "des copies Python". Dans
`alvarezlaguna/MUFFIN_Release`, plusieurs couts se superposent :

- stockage coeur dans des `py::array_t` (`Simulation1D.cpp`, `MeshData.hpp`) ;
- callbacks Python dans des fonctions de modele (`PythonPhysicalModel.cpp`) ;
- creation de vues NumPy par cellule via `CellDataRef::getView()` ;
- callbacks Python de source et de condition limite (`PythonSourceTerm.cpp`,
  `PythonBC.cpp`) ;
- I/O HDF5 via Python (`h5py`) avec allocation NumPy et `memcpy` par sortie
  (`DataWriter1DH5Py.cpp`) ;
- solveur lineaire periodique passant par `numpy.linalg.solve`, avec copie de
  matrice et de second membre (`ThomasAlgorithmPeriodic.cpp`) ;
- `py::print` et retours d'arrays depuis la boucle de simulation
  (`Simulation1D::simulate`, `advance_one`).

Sources amont :

- https://github.com/alvarezlaguna/MUFFIN_Release/blob/main/src/PhysicalModel/PythonPhysicalModel.cpp
- https://github.com/alvarezlaguna/MUFFIN_Release/blob/main/src/MeshData/CellDataRef.hpp
- https://github.com/alvarezlaguna/MUFFIN_Release/blob/main/src/DataWriter/DataWriter1DH5Py.cpp
- https://github.com/alvarezlaguna/MUFFIN_Release/blob/main/src/LinearSolver/ThomasAlgorithmPeriodic.cpp
- https://github.com/alvarezlaguna/MUFFIN_Release/blob/main/src/Simulation1D.cpp

Le risque a auditer dans ADC est donc :

```text
cout catastrophique = callback Python par cellule/face
cout structurel fort = copie full-array par pas ou diagnostic frequent
cout acceptable = appel pybind par macro-pas si le calcul reste C++
cout acceptable = compilation DSL froide, si amortie par cache et run long
```

Etat observe dans ADC au moment de l'audit :

- `include/adc/runtime/system.hpp` documente le contrat : Python compose, le
  calcul cellule par cellule reste C++ compile ; aucun callback Python dans le
  hot path, sauf integrateur custom via `eval_rhs/get_state/set_state`.
- `python/bindings.cpp` expose `step` et `advance` directement. Les copies
  visibles sont aux frontieres `set_*` (`flat(arr)`) et diagnostics
  `get_state`, `density`, `potential` (`to_2d`, `to_3d` avec `memcpy`).
- `python/adc/integrate.py` est le chemin a bannir des mesures production :
  il appelle `eval_rhs`, `get_state`, `set_state` depuis Python et copie donc
  des champs complets a chaque etage.
- `python/adc/dsl.py` distingue `prototype`, `aot` et `production`. Pour les
  mesures de production, il faut exiger `backend="production"` et verifier que
  l'adder est `add_native_block`.

Verdict : ADC n'a pas, sur les chemins briques natives et DSL `production`, le
profil MUFFIN "Python dans la boucle cellule". Les couts Python attendus sont
principalement setup, compilation DSL, appel pybind par `step`, et copies de
diagnostic. Le risque principal est l'usage accidentel d'un mauvais chemin
(`integrate.py`, `eval_rhs` dans une boucle Python, DSL `aot/prototype`, ou
diagnostics trop frequents).

## 3. Modele theorique de cout

Le temps total doit etre modele ainsi :

```text
T_total =
  T_import
+ T_compile_dsl
+ T_setup
+ Nsteps * T_step
+ Ndiag * T_diag
+ Nio * T_io

T_step =
  T_kernel
+ T_halo
+ T_reduction
+ T_poisson
+ T_fence
+ T_py_boundary
```

Interpretation :

- C++ natif : `T_py_boundary = 0`.
- Python briques natives avec `advance(dt, nsteps)` :
  `T_py_boundary ~= un appel pybind pour tout le run`, donc amorti.
- Python briques natives avec boucle `for step(dt)` :
  `T_py_boundary ~= Nsteps * cout_pybind_step`. Le cout reste par pas, pas par
  cellule.
- Python DSL `production` warm cache :
  `T_compile_dsl ~= 0`, puis meme hot path que `add_block`.
- Python DSL `production` cold cache :
  `T_compile_dsl` peut dominer un petit run, mais doit disparaitre dans un run
  long ou cache.
- Python DSL `aot` :
  marshaling de tableaux plats, pas MPI/AMR, pas zero-copie ; utile comme
  contre-exemple, pas comme cible.
- Python `integrate.py` :
  copies full-array par etage. A traiter comme baseline "mauvais usage" et non
  comme performance ADC production.

Ordres de grandeur utiles avant mesure :

- Un callback Python par cellule/face est incompatible avec GPU/MPI performant.
  Meme `1 us` par callback donne deja environ `1 s` pour `10^6` appels.
- Une copie host d'un etat Euler `4*n*n*8` vaut `32 MiB` a `n=1024`. Le minimum
  bande passante peut etre sub-ms sur CPU, mais allocation NumPy, GIL, cache et
  synchronisation GPU peuvent la rendre bien plus chere.
- Une lecture hote sur GPU peut forcer un `Kokkos::fence` ou une migration de
  memoire unifiee. Il faut donc mesurer diagnostics avec et sans extraction.
- Le cout pybind par appel `step` ne doit pas apparaitre dans le profil par
  phase C++ ; il doit etre mesure par comparaison `advance` vs boucle Python.

## 4. Cas de benchmark neutres

Cas principal, frontends :

```text
case=frontend-euler-periodic
modele=Euler compressible 2D lisse, periodique, sans disque, sans Hoffart
schema=minmod ou weno5 selon campagne, flux rusanov puis hllc si stable
source=none pour transport pur, ou gravity+Poisson pour chemin couple controle
dt=fixe pour comparer step/advance, pas step_cfl pour la comparaison frontend
diagnostics=off pendant la boucle, extraction finale separee
```

Cas kernels isoles :

| case | objectif | source actuelle |
|---|---|---|
| `transport-fv` | cout kernel transport pur | a extraire de `tests/test_mpi_mbox_parity.cpp` ou d'un futur bench dedie |
| `poisson-mms` | cout solve elliptique controle | tests Poisson MMS et `GeometricMG` |
| `halo-mpi` | cout halos seuls | `python/tests/gpu/mpi6_fillboundary.cpp` |
| `reduction` | cout reductions globales | `MultiFab` reductions / `profile_step` |
| `amr-synthetic` | scaling AMR hors Hoffart | `python/tests/gpu/amrmpi_integrated.cpp` |

Cas existant a ne pas utiliser comme benchmark principal :

- `bench/profile_step.cpp` : utile pour breakdown phase, mais son modele est
  ExB/diocotron et la decomposition MPI est mono-box dans le style `System`.
  Il ne doit pas etre vendu comme strong/weak scaling distribue general.

## 5. Campagne scaling

### Strong scaling

Definition : taille globale fixe, ressources croissantes.

CPU on-node :

```text
backend=kokkos-openmp
threads=1,2,4,8,16,...
OMP_PROC_BIND=spread
OMP_PLACES=cores
KOKKOS_NUM_THREADS=<threads>
```

MPI + Kokkos CPU :

```text
backend=mpi+kokkos-openmp
ranks=1,2,4,...
threads_per_rank=1,2,4,...
conserver ranks*threads <= coeurs physiques
ne pas utiliser Kokkos Serial comme cible perf
```

GPU :

```text
backend=kokkos-cuda
np=1 pour GPU single-rank
backend=mpi+kokkos-cuda
np=1,2,4,... avec un GPU par rang
srun -n <np> --gpus-per-task=1
```

Metriques :

```text
per_step_ms = temps mur max sur rangs / pas mesures
speedup(np) = per_step_ms(np=1) / per_step_ms(np)
efficiency(np) = speedup(np) / np
```

### Weak scaling

Definition : taille locale fixe, taille globale croissante.

Pour un domaine carre, prendre `n_global = n_local * sqrt(np)` quand `np` est
un carre parfait. Sinon imposer une decomposition rectangulaire documentee.

Metriques :

```text
weak_eff(np) = per_step_ms(np=1) / per_step_ms(np)
communication_pct = (halos + reductions + fences MPI) / total
```

### AMR

Le seul cas AMR a utiliser pour la perf neutre est synthetique, pas Hoffart.
`python/tests/gpu/amrmpi_integrated.cpp` mesure deja un cas quatre bulles
Euler-Poisson AMR et compare grossier replique vs reparti.

Resultat deja documente dans `docs/GPU_RUNTIME_PORT.md` : a petite echelle
GH200, le strong scaling AMR par grossier reparti est negatif. Ce resultat ne
doit pas etre cache : distribuer un grossier trop petit economise peu de calcul
et ajoute beaucoup de latence MPI/MG.

## 6. Campagne frontends C++ vs Python

Executer exactement le meme cas Euler periodique lisse en trois variantes :

1. `cpp-native` : C++ natif, sans Python.
2. `python-bricks` : `adc.Model(FluidState, CompressibleFlux, NoSource, ...)`.
3. `python-dsl-production` : `dsl.Model(...).compile(backend="production")`.

Controles obligatoires :

- `python-dsl-production.compiled.adder == "add_native_block"`.
- `compiled.backend == "production"`.
- `compiled.target == "system"` pour uniforme, `"amr_system"` seulement pour AMR.
- Aucun appel `eval_rhs/get_state/set_state` dans la boucle mesuree.
- Mesurer `advance(dt, nsteps)` et la boucle Python `for step(dt)` separement.
- Mesurer `get_state` final dans une phase `extract_final`, pas dans `step`.
- Vider ou isoler `ADC_CACHE_DIR` pour la mesure cold compile, puis refaire
  warm cache avec le meme modele.

Table cible :

| frontend | setup_ms | compile_ms | advance_ms | python_step_loop_ms | extract_ms | notes |
|---|---:|---:|---:|---:|---:|---|
| C++ natif | | | | n/a | | reference |
| Python briques | | 0 | | | | `ModelSpec` natif |
| Python DSL production cold | | | | | | compile + run |
| Python DSL production warm | | ~0 | | | | cache hit |
| Python DSL aot | | | | | | contre-exemple |
| Python custom integrate | | 0 | n/a | | | mauvais usage controle |

Ratios a publier :

```text
R_bricks_hot = python_bricks.advance_ms / cpp_native.advance_ms
R_dsl_hot = dsl_production_warm.advance_ms / cpp_native.advance_ms
R_step_boundary = python_step_loop_ms / python_advance_ms
R_dsl_cold_warm = dsl_cold_total_ms / dsl_warm_total_ms
R_extract = extract_ms / advance_ms
```

Interpretation attendue :

- `R_bricks_hot` et `R_dsl_hot` proches de 1 si le hot path est bien natif.
- Si `R_step_boundary` est eleve, recommander `advance(dt, nsteps)` pour les
  runs longs depuis Python.
- Si `R_extract` est eleve sur GPU, reduire les diagnostics hote ou les faire
  cote device/reduction.

## 7. Graphes a produire

Les graphes ne doivent etre produits qu'a partir de CSV de campagne, jamais a
partir de valeurs inventees.

Fichiers resultats recommandes :

```text
bench/results/perf_scaling_master_<sha>.csv
bench/results/perf_frontends_master_<sha>.csv
bench/results/perf_phases_master_<sha>.csv
```

Colonnes minimales `perf_scaling` :

```text
commit,repo,case,scaling,backend,machine,np,threads,gpus,n_global,n_local,steps,warmup,
wall_s,per_step_ms,speedup,efficiency,notes
```

Colonnes minimales `perf_frontends` :

```text
commit,case,frontend,backend,n,steps,dt,cache_state,setup_ms,compile_ms,
advance_ms,step_loop_ms,extract_ms,total_ms,ratio_vs_cpp,notes
```

Colonnes minimales `perf_phases` :

```text
commit,case,backend,np,threads,n,steps,phase,total_s,per_step_ms,pct
```

Graphes obligatoires :

- `strong_scaling_speedup.png`
- `strong_scaling_efficiency.png`
- `weak_scaling_efficiency.png`
- `phase_breakdown_stacked.png`
- `frontend_ratios.png`
- `dsl_cold_warm.png`
- `diagnostics_io_impact.png`

Script de tracage ajoute :

```bash
python3 bench/plot_perf_campaign.py --results-dir bench/results --out-dir docs/perf_figures
```

Le script lit `perf_scaling*.csv`, `perf_frontends*.csv` et
`perf_phases*.csv`. Il saute les graphes dont les donnees manquent ; il ne
genere jamais de valeurs synthetiques.

## 8. Recommandations

1. Ne pas utiliser Hoffart pour conclure sur C++ vs Python.
2. Ne pas utiliser `bench/profile_step` comme preuve de strong scaling MPI : il
   profile un pas mono-box representatif de `System`, pas une charge repartie.
3. Pour les frontends, faire de `advance(dt, nsteps)` la mesure primaire et de
   `for step(dt)` une mesure de cout de frontiere pybind.
4. Pour le DSL, publier separement cold compile, warm cache et hot loop.
5. Bannir `python/adc/integrate.py` des chiffres production ; le garder comme
   anti-exemple chiffre.
6. Sur GPU, mettre les diagnostics dans leur propre phase : toute extraction
   hote peut cacher un `fence` ou une migration de memoire unifiee.
7. Pour MPI + GPU, publier le temps max sur rangs, pas la moyenne.
8. Pour AMR, commencer par le cas synthetique quatre bulles et documenter le
   mode grossier replique/reparti.
9. Toute PR ouverte doit etre mesuree comme variante explicite, jamais melee a
   `master`.

## 9. Critere d'acceptation de la campagne

La campagne est exploitable seulement si elle permet de repondre a ces
questions :

- Le hot loop Python briques est-il a parite avec C++ natif ?
- Le hot loop DSL `production` warm cache est-il a parite avec C++ natif ?
- Quelle part du surcout vient de `step` appele depuis Python au lieu de
  `advance` ?
- Quelle part vient des copies `set_state/get_state/density/potential` ?
- Quelle part vient de Poisson, halos, reductions et fences ?
- Le scaling weak degrade-t-il par communication ou par kernel local ?
- Le scaling strong AMR est-il limite par grossier replique, grossier reparti,
  MG multi-box, regrid ou diagnostics ?

Tant que ces questions n'ont pas de tableau chiffre, ne pas annoncer de gain ou
de perte definitive C++ vs Python.

## 10. Resultats ROMEO du 2026-06-08

Cette section ajoute les premiers runs lances sur ROMEO le 2026-06-08. Elle ne
remplace pas le protocole ci-dessus : elle donne un etat mesure, avec ses
limites.

### 10.1 Etat Git et jobs

Les runs scaling ont ete figes sur :

```text
adc_cpp=1f9fb4a
adc_cases=b8bccbe
```

Pendant le rattrapage frontend, `origin/master` de `adc_cpp` a avance vers
`0c3eae1`, puis vers `adde23b` pendant le rattrapage transport multi-box.
Les resultats ci-dessous restent donc volontairement rattaches a
`1f9fb4a` pour ne pas melanger les commits.

Jobs ROMEO :

| job | cible | statut | temps | remarque |
|---:|---|---|---:|---|
| `647780` | CPU `x64cpu`, OpenMP + MPI/OpenMP | `COMPLETED` | `00:17:51` | scaling CPU + C++ frontend |
| `647781` | GPU `armgpu`, CUDA + MPI/CUDA | `COMPLETED` | `00:13:32` | scaling GPU + AMR synthetique |
| `647809` | rattrapage frontend CPU | `FAILED` | `00:00:10` | harness Python absent |
| `647813` | rattrapage frontend CPU | `CANCELLED` | `00:03:23` | annule pour eviter un build C++ inutile |
| `647815` | frontend CPU Python-only, Kokkos PIC | `COMPLETED` | `00:03:55` | Python briques + DSL production |

Resultats locaux :

```text
bench/romeo_results_final_647780_647781_647815/
docs/perf_figures_647780_647781_647815/
```

Resultats ROMEO :

```text
/home/rmdraux/adc_perf_20260608/results/
/home/rmdraux/adc_perf_20260608/logs/
```

Graphes generes :

- `docs/perf_figures_647780_647781_647815/strong_scaling_speedup.png`
- `docs/perf_figures_647780_647781_647815/strong_scaling_efficiency.png`
- `docs/perf_figures_647780_647781_647815/weak_scaling_efficiency.png`
- `docs/perf_figures_647780_647781_647815/phase_breakdown_stacked.png`
- `docs/perf_figures_647780_647781_647815/frontend_ratios.png`
- `docs/perf_figures_647780_647781_647815/dsl_cold_warm.png`
- `docs/perf_figures_647780_647781_647815/diagnostics_io_impact.png`

Graphes finaux apres rattrapages `647836` et `647848` :

- `docs/perf_figures_647780_647781_647815_647836_647848/strong_scaling_speedup.png`
- `docs/perf_figures_647780_647781_647815_647836_647848/strong_scaling_efficiency.png`
- `docs/perf_figures_647780_647781_647815_647836_647848/weak_scaling_efficiency.png`
- `docs/perf_figures_647780_647781_647815_647836_647848/phase_breakdown_stacked.png`
- `docs/perf_figures_647780_647781_647815_647836_647848/frontend_ratios.png`
- `docs/perf_figures_647780_647781_647815_647836_647848/dsl_cold_warm.png`
- `docs/perf_figures_647780_647781_647815_647836_647848/diagnostics_io_impact.png`

### 10.2 Verrou observe : Poisson domine

Sur `bench/profile_step`, le temps est presque entierement dans Poisson :

| backend | ressources | n | per_step_ms | part Poisson |
|---|---:|---:|---:|---:|
| `kokkos-openmp` | 1 rang, 1 thread | 256 | 194.19 | 96.5 % |
| `kokkos-openmp` | 1 rang, 8 threads | 256 | 347.04 | 99.0 % |
| `mpi+kokkos-openmp` | 1 rang, 4 threads | 256 | 183.15 | 98.8 % |
| `mpi+kokkos-openmp` | 8 rangs, 4 threads/rang | 256 | 4313.26 | 95.7 % |
| `kokkos-cuda` | 1 GPU | 256 | 238.38 | 99.1 % |
| `mpi+kokkos-cuda` | 4 GPU | 256 | 292.73 | 98.8 % |

Interpretation : cette campagne mesure surtout le solveur elliptique et les
reductions associees. Elle ne suffit pas a conclure sur le cout du transport FV
pur. Le prochain bench doit isoler transport, Poisson, halos et reductions.

### 10.3 Strong scaling CPU/GPU

CPU OpenMP, taille globale `n=256` :

| backend | ressources | per_step_ms |
|---|---:|---:|
| `kokkos-openmp` | 1 thread | 194.19 |
| `kokkos-openmp` | 2 threads | 855.31 |
| `kokkos-openmp` | 4 threads | 576.57 |
| `kokkos-openmp` | 8 threads | 347.04 |

MPI + OpenMP, taille globale `n=256`, `threads=4` par rang :

| backend | rangs | per_step_ms |
|---|---:|---:|
| `mpi+kokkos-openmp` | 1 | 183.15 |
| `mpi+kokkos-openmp` | 2 | 474.94 |
| `mpi+kokkos-openmp` | 4 | 2722.01 |
| `mpi+kokkos-openmp` | 8 | 4313.26 |

GPU, taille globale `n=256` :

| backend | GPU/rangs | per_step_ms |
|---|---:|---:|
| `kokkos-cuda` | 1 | 238.38 |
| `mpi+kokkos-cuda` | 1 | 234.06 |
| `mpi+kokkos-cuda` | 2 | 284.61 |
| `mpi+kokkos-cuda` | 4 | 292.73 |

Conclusion limitee : `n=256` est trop petit et trop Poisson-dominant pour
montrer un scaling positif. Le resultat utile est negatif mais informatif :
augmenter les rangs ajoute plus de couts de solveur/reductions/coordination
qu'il ne retire de calcul local.

### 10.4 Weak scaling : smoke-test seulement

Les lignes weak lancees dans ce premier job utilisent `n = 128*np`. Ce n'est
pas encore le weak scaling canonique recommande plus haut
(`n_global = n_local*sqrt(np)` pour garder la taille locale constante en 2D).
Elles doivent donc etre lues comme smoke-test, pas comme courbe weak finale.

| backend | ressources | n | per_step_ms |
|---|---:|---:|---:|
| `mpi+kokkos-openmp` | 1 rang, 4 threads | 128 | 59.60 |
| `mpi+kokkos-openmp` | 4 rangs, 4 threads/rang | 512 | 8414.39 |
| `mpi+kokkos-cuda` | 1 GPU | 128 | 219.76 |
| `mpi+kokkos-cuda` | 4 GPU | 512 | 328.74 |

Action : relancer un vrai weak scaling avec decomposition 2D et taille locale
fixe par rang/GPU.

### 10.5 AMR synthetique GPU

`amrmpi_integrated` a ete lance sur GH200 en `np=1,2,4`. Les diagnostics
`cmax_crossrank_spread=0.0` et la conservation de masse montrent que le test
reste coherent numeriquement.

| mode AMR | GPU/rangs | per_step_ms | observation |
|---|---:|---:|---|
| `replique` | 1 | 216.37 | reference |
| `replique` | 2 | 273.15 | scaling negatif |
| `replique` | 4 | 272.66 | plateau |
| `reparti` | 1 | 696.85 | deja plus cher que replique |
| `reparti` | 2 | 1014.36 | degrade |
| `reparti` | 4 | 1373.70 | degrade |

Conclusion : sur ce petit cas AMR, distribuer le grossier ne paie pas. C'est
un resultat coherent avec le risque annonce : le cout de coordination/MG domine
le calcul economise.

### 10.6 Frontends C++ vs Python

Le premier build Python a echoue avec la Kokkos OpenMP existante :

```text
relocation R_X86_64_32 ... libkokkoscore.a ... recompile with -fPIC
```

Correction de campagne : construire une Kokkos OpenMP PIC dediee dans
`/home/rmdraux/adc_perf_20260608/kinstall_omp_pic`, puis relancer seulement le
frontend Python sur le meme commit `1f9fb4a`.

Les lignes C++ natives viennent du job `647780`; les lignes Python viennent du
job `647815`. Les ratios doivent donc etre lus comme premiere indication, pas
comme benchmark definitif meme noeud/meme build. Ils suffisent toutefois a
exclure un profil MUFFIN catastrophique sur `python-bricks`.

| threads | frontend | advance_ms | ratio vs C++ | compile_ms | total_ms |
|---:|---|---:|---:|---:|---:|
| 1 | `cpp-native` | 291.20 | 1.00 | 0.00 | 291.32 |
| 1 | `python-bricks` | 281.33 | 0.97 | 0.00 | 282.20 |
| 1 | `python-dsl-production warm` | 339.12 | 1.16 | 6.75 | 346.10 |
| 1 | `python-dsl-production cold` | 339.04 | 1.16 | 11732.36 | 12071.99 |
| 4 | `cpp-native` | 253.70 | 1.00 | 0.00 | 253.84 |
| 4 | `python-bricks` | 220.19 | 0.87 | 0.00 | 221.04 |
| 4 | `python-dsl-production warm` | 339.62 | 1.34 | 6.71 | 346.57 |
| 4 | `python-dsl-production cold` | 339.62 | 1.34 | 11744.18 | 12084.41 |
| 8 | `cpp-native` | 181.67 | 1.00 | 0.00 | 181.81 |
| 8 | `python-bricks` | 148.03 | 0.81 | 0.00 | 149.07 |
| 8 | `python-dsl-production warm` | 339.35 | 1.87 | 6.66 | 346.24 |
| 8 | `python-dsl-production cold` | 339.54 | 1.87 | 11746.57 | 12086.72 |

Lecture :

- `python-bricks` n'a pas de perte MUFFIN visible : le hot loop reste du C++
  appele par `advance`, pas un callback Python par cellule.
- `step_loop_ms` et `advance_ms` sont presque identiques sur ces mesures, donc
  le cout pybind par `step` est faible face au cout du pas pour `n=128`.
- l'extraction finale `get_state` vaut environ `0.07-0.21 ms`, donc
  ne domine pas ce cas CPU.
- le DSL `production` cold compile vaut environ `11.7 s`; warm cache vaut
  environ `6-7 ms`.
- le hot loop DSL `production` warm est plus lent et quasi invariant avec les
  threads dans ce harness. Il faut isoler si cela vient du chemin
  `add_native_block`, du modele genere, d'une absence de scaling Kokkos dans le
  bloc DSL, ou d'un artefact de comparaison avec les lignes C++ du job `647780`.

### 10.7 Rattrapage transport pur multi-box CPU

Un benchmark dedie a ete ajoute dans `bench/profile_transport_mbox.cpp` pour
isoler un cas Euler 2D periodique lisse, sans Poisson, sans disque, sans Schur
et avec une vraie decomposition multi-box distribuee par MPI. Le script ROMEO
associe est `bench/romeo_perf_transport_mbox_cpu.sbatch`.

Run ROMEO :

| job | commit | cible | statut | temps |
|---:|---|---|---|---:|
| `647836` | `adde23b` | CPU `x64cpu`, transport multi-box | `COMPLETED` | `00:02:34` |

Resultats locaux :

```text
bench/romeo_results_transport_mbox_adde23b_647836/
```

Strong OpenMP, `n=1024`, 1 rang :

| threads | per_step_ms | speedup vs 1 thread | efficacite |
|---:|---:|---:|---:|
| 1 | 524.13 | 1.00 | 1.00 |
| 2 | 434.25 | 1.21 | 0.60 |
| 4 | 284.15 | 1.84 | 0.46 |
| 8 | 157.82 | 3.32 | 0.42 |
| 16 | 89.31 | 5.87 | 0.37 |

Lecture : le transport pur scale positivement en OpenMP, contrairement au
premier `profile_step` Poisson-dominant. L'efficacite baisse quand les threads
augmentent, mais le signal est exploitable.

Strong MPI + OpenMP, `n=1024`, `threads=4` par rang :

| rangs | per_step_ms | phase dominante |
|---:|---:|---|
| 1 | 188.35 | transport 96.9 % |
| 2 | 371.05 | halos 50.3 %, transport 43.0 % |
| 4 | 419.46 | transport 45.2 %, halos 34.3 %, reductions 20.6 % |
| 8 | 594.75 | halos 37.3 %, transport 35.1 %, reductions 27.6 % |

Lecture : le strong scaling MPI reste negatif meme sans Poisson. Le verrou
n'est donc pas seulement `GeometricMG` : le passage multi-rang rend
`fill_boundary` et les reductions globales assez chers pour dominer une partie
du pas.

Weak MPI + OpenMP 2D, `n_global ~= 384*sqrt(np)`, `threads=4` par rang :

| rangs | n_global | per_step_ms | weak_eff vs np=1 | phase dominante |
|---:|---:|---:|---:|---|
| 1 | 384 | 25.23 | 1.00 | transport 97.5 % |
| 2 | 543 | 207.13 | 0.12 | halos 68.4 % |
| 4 | 768 | 350.46 | 0.07 | halos 45.9 %, transport 31.2 %, reductions 22.9 % |
| 8 | 1086 | 659.29 | 0.04 | transport 51.7 %, halos 32.3 %, reductions 16.0 % |

Lecture : ce weak scaling est le resultat negatif le plus actionnable de la
campagne. A taille locale a peu pres constante, le cout par pas explose des
`np=2`, principalement par `fill_boundary`; a `np=4/8`, les reductions
`max_wave_speed_mf` et `dot` deviennent aussi visibles. Il faut donc profiler
la construction des jobs de halo, le nombre de messages, pack/unpack, le choix
SFC des boites, les collectives MPI et l'affinite `ranks x threads`.

Point metodologique : cette mesure remplace le smoke-test weak de la section
10.4 pour le CPU. Elle ne donne pas encore le weak GPU, qui doit rester separe
et ne pas etre lance pendant un autre job GH200 actif.

### 10.8 Rattrapage frontends meme job

Le job `647848` a relance C++ natif, Python briques et Python DSL `production`
dans le meme job ROMEO, sur le meme noeud `x64cpu`, avec la meme Kokkos OpenMP
PIC (`/home/rmdraux/adc_perf_20260608/kinstall_omp_pic`) et le meme commit
`adc_cpp=adde23b`.

| job | commit | cible | statut | temps |
|---:|---|---|---|---:|
| `647848` | `adde23b` | frontends CPU, Kokkos PIC | `COMPLETED` | `00:19:42` |

Resultats locaux :

```text
bench/romeo_results_frontends_adde23b_647848/
```

Le build C++ frontend a ete le cout froid dominant : `frontend_cpp` a passe
environ 15 minutes a compiler `python/system.cpp` en `-O3` avant les mesures.
Conclusion pratique : les futures campagnes doivent separer build et mesure,
ou reutiliser un build prepare, sinon le temps de campagne mesure surtout le
compilateur.

Table hot-loop `advance(dt, 40)` :

| threads | frontend | advance_ms | ratio vs C++ PIC | compile_ms | total_ms |
|---:|---|---:|---:|---:|---:|
| 1 | `cpp-native` | 292.42 | 1.00 | 0.00 | 292.54 |
| 1 | `python-bricks` | 283.85 | 0.97 | 0.00 | 285.22 |
| 1 | `python-dsl-production warm` | 341.10 | 1.17 | 6.78 | 348.14 |
| 1 | `python-dsl-production cold` | 341.54 | 1.17 | 15413.57 | 15755.87 |
| 4 | `cpp-native` | 239.59 | 1.00 | 0.00 | 239.73 |
| 4 | `python-bricks` | 228.26 | 0.95 | 0.00 | 229.36 |
| 4 | `python-dsl-production warm` | 342.05 | 1.43 | 7.61 | 349.90 |
| 4 | `python-dsl-production cold` | 341.67 | 1.43 | 15521.54 | 15863.92 |
| 8 | `cpp-native` | 177.46 | 1.00 | 0.00 | 177.60 |
| 8 | `python-bricks` | 165.29 | 0.93 | 0.00 | 166.40 |
| 8 | `python-dsl-production warm` | 341.09 | 1.92 | 7.49 | 348.81 |
| 8 | `python-dsl-production cold` | 341.29 | 1.92 | 15350.63 | 15692.59 |

Lecture :

- `python-bricks` n'a pas de penalite hot-loop visible par rapport au C++
  natif PIC sur ce cas. L'apparent avantage de quelques pourcents ne doit pas
  etre vendu comme "Python plus rapide" ; il signifie surtout que le cout
  Python n'est pas dans la boucle cellule.
- `step_loop_ms` et `advance_ms` restent tres proches. Le cout pybind par
  `step(dt)` est donc faible face au cout d'un pas a `n=128`.
- `get_state` final reste sub-ms ; les diagnostics hote ne dominent pas ce cas
  CPU. Ce point doit etre recontrole sur GPU, ou une lecture hote peut forcer
  synchronisation et migration.
- DSL `production` warm reste autour de `341 ms`, presque invariant avec les
  threads. Le ratio se degrade quand C++/bricks scalent (`1.17`, `1.43`,
  `1.92`). C'est maintenant un vrai sujet : verifier le code genere, les
  chemins Kokkos effectivement utilises, et le fait que le bloc DSL passe bien
  par le meme noyau/limiteur/flux que le bloc natif.
- DSL cold compile vaut environ `15.3-15.5 s` sur ce setup. Ce cout est
  acceptable seulement s'il est amorti par cache ou par un run long.

### 10.9 Mesures separees branche `feat/perf-campaign-bench`

Des jobs externes ROMEO `647857` et `647858` ont produit des JSONL sur la
branche `feat/perf-campaign-bench`, commit
`0162d5f4a8f2ef559325acce64decc1dede83e40`. Ces resultats sont utiles pour
l'audit, mais ils ne doivent pas etre melanges aux tableaux `adde23b/master`.

Fichiers locaux ajoutes :

```text
bench/romeo_results_matrix_647857_647858/matrix_cpu_0162d5f4a8_647857.jsonl
bench/romeo_results_matrix_647857_647858/matrix_gpu_0162d5f4a8_647858.jsonl
bench/romeo_results_matrix_647857_647858/perf_scaling_matrix_0162d5f4a8_647857_647858.csv
bench/romeo_results_matrix_647857_647858/perf_phases_matrix_0162d5f4a8_647857_647858.csv
docs/perf_figures_matrix_647857_647858/
```

CPU OpenMP transport, `n=4096`, 1 rang :

| threads | per_step_ms | speedup vs 1 thread |
|---:|---:|---:|
| 1 | 5238.64 | 1.00 |
| 2 | 5106.55 | 1.03 |
| 4 | 2735.66 | 1.91 |
| 8 | 1483.22 | 3.53 |
| 16 | 912.96 | 5.74 |

Lecture : a plus grande taille, le transport pur OpenMP scale nettement mieux
que le premier cas `profile_step`, et confirme que le probleme initial venait
du cas Poisson-dominant. Le passage `1 -> 2 threads` reste mauvais, donc
l'affinite et le choix `OMP/Kokkos` doivent rester controles.

CPU weak OpenMP local par thread, taille globale croissante :

| threads | n_global | per_step_ms |
|---:|---:|---:|
| 1 | 512 | 80.84 |
| 4 | 1024 | 194.13 |
| 16 | 2048 | 252.23 |

Lecture : ce weak scaling OpenMP on-node est plus propre que le smoke-test
initial, mais ce n'est pas le weak MPI final. Il mesure l'effet taille/threads
dans un seul rang.

GPU GH200 mono-rang, transport size sweep :

| n_global | per_step_ms | cellules/s |
|---:|---:|---:|
| 1024 | 30.09 | 3.48e7 |
| 2048 | 123.73 | 3.39e7 |
| 4096 | 497.12 | 3.37e7 |

Lecture : le throughput GPU mono-rang est stable quand la taille augmente.
Cette table est un size sweep, pas une preuve de strong scaling multi-GPU.
Elle donne une bonne reference `np=1` pour le futur MPI+CUDA transport.

GPU GH200 Poisson mono-rang :

| n_global | per_step_ms |
|---:|---:|
| 512 | 6.35 |
| 1024 | 19.54 |

AMR synthetique GPU, grossier reparti pour `np>1` :

| n | rangs/GPU | per_step_ms |
|---:|---:|---:|
| 128 | 1 | 215.44 |
| 128 | 2 | 1012.82 |
| 128 | 4 | 1389.79 |
| 256 | 1 | 233.26 |
| 256 | 2 | 1079.72 |
| 256 | 4 | 1485.32 |

Lecture : le resultat AMR reste negatif en multi-GPU, ce qui renforce le
diagnostic deja pose : le grossier distribue et les coordinations AMR/MG
dominent sur ces tailles. En revanche, ces jobs ne ferment pas encore le TODO
MPI+CUDA transport weak/strong : il manque le cas transport pur multi-rang GPU
avec un GPU par rang.

### 10.10 Diagnostic DSL `production` et Kokkos

Le rattrapage `647848` a montre que `python-dsl-production` warm reste autour
de `341 ms` quel que soit le nombre de threads. L'enquete code indique une
cause plausible et actionnable : le loader DSL `production` est zero-copie et
utilise bien `add_native_block`, mais il compilait ses templates header-only
sans propager explicitement `ADC_HAS_KOKKOS`, les includes Kokkos et
`-fopenmp`.

Consequence : sur un module Python `_adc` compile avec Kokkos OpenMP, le bloc
DSL pouvait rester sur le fallback serie dans les portions inline du loader,
tout en ayant l'apparence d'un chemin production natif. Ce n'est pas un
probleme MUFFIN de copies Python ; c'est une incoherence de backend C++ entre
le module `_adc` et le `.so` genere par le DSL.

Correctif local experimental ajoute dans cette worktree :

- `include/adc/runtime/abi_key.hpp` encode maintenant les features ABI
  `kokkos=on/off` et `mpi=on/off`. Un loader compile sans Kokkos ne peut plus
  etre considere ABI-equivalent a un module `_adc` compile avec Kokkos.
- `python/adc/dsl.py` detecte `ADC_KOKKOS_ROOT`/`Kokkos_ROOT`/`KOKKOS_ROOT`,
  utilise `ADC_KOKKOS_CXX` ou `nvcc_wrapper` si disponible, ajoute
  `-DADC_HAS_KOKKOS`, les includes Kokkos, `-fopenmp` en OpenMP, et met ces
  features dans la cle de cache DSL. Dans la version courante de la worktree,
  le loader ne linke plus `libkokkos*` ; il laisse les symboles se resoudre
  contre `_adc`.
- `bench/romeo_perf_frontends_cpu.sbatch` sait superposer ces deux fichiers
  patchés sur un checkout ROMEO explicite, et reutiliser les lignes C++
  natives deja mesurees pour relancer seulement le frontend Python.

Attention : cette variante locale est plus stricte que la branche amont
`origin/feat/dsl-production-optflags`. La branche amont met le backend Kokkos
dans la cle de cache Python et evite de linker `libkokkos*` dans le `.so`
production ; elle ne modifie pas la cle ABI C++ publique
`include/adc/runtime/abi_key.hpp`. La modification ABI locale doit donc etre
lue comme une experience de securite, pas comme un patch a merger sans revue.
La partie `HybridModel.compile()` a aussi ete etendue localement et demande un
test dedie avant integration.

Validation ROMEO ciblee :

```text
jobs=648017, 648031, 648034
base=/home/rmdraux/adc_perf_frontends_dslkokkos_20260608
checkout=adde23b + patch local dsl/abi
reference_cpp=job 647848, lignes cpp-native uniquement
objectif=verifier si DSL warm suit enfin les threads Kokkos OpenMP
```

Deux essais intermediaires (`648017`, `648031`) ont echoue avant mesure DSL car
le patch choisissait automatiquement `bin/nvcc_wrapper` des que le fichier
existait dans le root Kokkos PIC. Sur ROMEO CPU, cette installation OpenMP
fournit bien `nvcc_wrapper`, mais `nvcc` n'est pas charge. Le correctif final
est donc volontairement explicite : `nvcc_wrapper` n'est utilise que si
`ADC_KOKKOS_CXX` le designe ou si `ADC_KOKKOS_USE_NVCC_WRAPPER=1`; OpenMP prend
le compilateur hote `g++` avec `-fopenmp`.

Le job `648034` a alors produit les lignes completes. Le processus Python
termine ensuite avec `rc=134` sur `Kokkos::finalize()` ("Execution space
instance to be removed couldn't be found!") apres ecriture du CSV. La branche
amont `origin/feat/dsl-production-optflags` corrige cette cause : il ne faut
pas linker `libkokkos*` dans le `.so` production, car le module `_adc` a deja
charge le runtime Kokkos. Le loader doit laisser les symboles Kokkos indefinis
et les resoudre au chargement contre `_adc` promu en `RTLD_GLOBAL`; sinon on
cree deux singletons Kokkos et la finalisation abort.

Resultats locaux :

```text
bench/romeo_results_frontends_dslkokkosfix_648034/
docs/perf_figures_frontends_dslkokkosfix_648034/
```

Table `advance(dt, 40)` apres correctif partiel DSL/Kokkos local :

| threads | frontend | advance_ms | ratio vs C++ PIC | compile_ms | total_ms |
|---:|---|---:|---:|---:|---:|
| 1 | `python-bricks` | 280.23 | 0.96 | 0.00 | 281.21 |
| 1 | `python-dsl-production warm` | 345.96 | 1.18 | 6.67 | 352.86 |
| 4 | `python-bricks` | 231.15 | 0.96 | 0.00 | 232.22 |
| 4 | `python-dsl-production warm` | 261.31 | 1.09 | 6.83 | 269.81 |
| 8 | `python-bricks` | 164.26 | 0.93 | 0.00 | 165.49 |
| 8 | `python-dsl-production warm` | 192.72 | 1.09 | 6.80 | 201.24 |

Avant correctif (`647848`), DSL warm valait `341.10`, `342.05`, `341.09 ms`
pour `1,4,8` threads. Apres correctif partiel (`648034`), il vaut `345.96`,
`261.31`, `192.72 ms`. Le verrou principal etait donc bien l'absence de backend
Kokkos correct dans le loader DSL, pas une copie Python. En revanche `648034`
n'est pas une mesure finale publiable du DSL : le processus a ecrit le CSV puis
a abort sur `Kokkos::finalize()` parce que cette variante linkait encore un
runtime Kokkos dans le `.so`. Les ratios `648034` servent donc a diagnostiquer
le retour du scaling OpenMP, pas a donner le cout final du frontend.

La branche amont ajoute aussi `-O3 -DNDEBUG` par defaut pour le `.so` production
(`ADC_DSL_OPTFLAGS` permet de surcharger, par exemple `-march=native`) et ne
linke pas `libkokkos*` dans le loader. D'apres son rapport adc_cases
`origin/feat/perf-campaign-harness`, cette variante ramene le DSL a la parite
serie (`1.04x`) et a la parite threadee OpenMP (`1.02x` a 8 threads) avec sortie
propre. La prochaine optimisation locale consiste donc a s'aligner proprement
sur cette branche, puis a relancer une validation courte, pas a chercher des
copies Python.

### 10.10-b Fix halo MPI multi-box

Apres fetch, `origin/master` contient `f3e1bf9 fix(mesh): halos MPI en memoire
hote epinglee (fin du deadlock multi-box GPU) (#254)` ; au moment de cette
mise a jour, `origin/master=5e0b3f6` inclut toujours ce correctif. Cela vise
directement le blocage signale plus haut pour MPI multi-box GPU : les tampons
de communication MPI doivent passer par une memoire hote epinglee plutot que
par un pointeur UVM/device qui declenche CUDA IPC sous cgroups GPU ROMEO.

Nuance de provenance : la worktree locale de ce rapport est encore sur
`HEAD=0187329` et son `include/adc/mesh/fill_boundary.hpp` montre les anciens
tampons `fab_allocator` en memoire unifiee. Le code audite pour `#254` est
donc `origin/master`/la branche ROMEO, pas le fichier local actuellement
checkout. Cette distinction evite de melanger audit de code local et code
execute sur ROMEO.

Avant relance v2, cela changeait le run utile : il ne fallait plus relancer
l'ancien diagnostic de deadlock, mais tester une vraie campagne
validation/scaling MPI+CUDA transport pur sur `origin/master` recent, ou sur
une branche combinant `#254` et les harnais `feat/perf-campaign-bench`. Les
resultats precedents restaient valides comme diagnostic historique, mais ne
devaient pas etre utilises seuls pour conclure que MPI+CUDA transport etait
toujours bloque.

Relance v2 observee ensuite :

```text
adc_cpp=1d4cd25e25d244cd7c4f6cfd4c0eb815cd997790
base=feat/perf-campaign-bench merge origin/master
jobs=648114 CPU, 648115 GPU
resultats=bench/romeo_results_mpi_v2_648114_648115/
figures=docs/perf_figures_mpi_v2_648114_648115/
```

Table des points qui terminent :

| job | backend | rangs/GPU | workload | n | per_step_ms | statut |
|---:|---|---:|---|---:|---:|---|
| `648114` | `mpi-omp` | 1 rang, 16 threads | transport | 4096 | 764.73 | OK |
| `648115` | `mpi-cuda` | 1 GPU | transport | 4096 | 502.91 | OK |
| `648115` | `mpi-cuda` | 1 GPU | poisson | 1024 | 20.68 | OK |

Les points multi-rang restent en timeout :

| job | cas | resultat |
|---:|---|---|
| `648114` | transport `2 rangs x 8 threads` | timeout `300 s` |
| `648114` | transport `4 rangs x 4 threads` | timeout `300 s` |
| `648115` | transport `2 GPU` | timeout `300 s` |
| `648115` | poisson `2 GPU` | timeout `300 s` |
| `648115` | transport/poisson suivants | timeout dans la meme campagne |

Conclusion : `#254` et la memoire hote epinglee ne suffisent pas a debloquer
ce harnais multi-box `scaling_step` sur ROMEO. Le blocage restant n'est donc
plus a presenter comme "CUDA IPC uniquement" ; il faut ouvrir un chantier code
dedie `fill_boundary`/progress MPI/ordonnancement halos, avec test minimal
multi-box CPU et GPU, avant toute nouvelle campagne weak/strong MPI+CUDA.

Nuance importante : cela ne prouve pas que tout MPI CPU ADC est bloque. Le
bench CPU dedie `profile_transport_mbox` du job `647836` termine bien en
`np=1/2/4/8`, meme avec un scaling negatif. Le timeout v2 concerne le harnais
`scaling_step` a grande taille et les chemins testes dans `648114/648115`.
Il peut donc venir d'un deadlock restant, d'un cout extremement pathologique,
du lanceur ROMEO, ou d'une difference de code path ; il ne faut pas le convertir
en conclusion generale sans test minimal.

### 10.10-c Interpretation theorie vs mesures

Le modele de cout pose au debut du rapport explique bien les observations :

```text
T_step = T_kernel + T_halo + T_reduction + T_poisson + T_fence + T_py_boundary
```

1. `python-bricks` suit le C++ natif parce que `T_py_boundary` est amorti par
   `advance(dt, nsteps)` et que le calcul cellule reste C++ compile. Les
   mesures `647848/648034` montrent des ratios autour de `0.93-0.97x`, a lire
   comme bruit de mesure et non comme "Python plus rapide".
2. Le DSL `production` n'etait pas ralenti par Python mais par le "comment" de
   compilation du loader : sans `ADC_HAS_KOKKOS`, les templates inline
   utilisent le fallback serie ; avec `libkokkos*` linke dans le `.so`, on cree
   un second runtime Kokkos ; avec `-O2` sans `-DNDEBUG`, le hot loop garde des
   checks/optimisations insuffisantes. La theorie dit que `T_py_boundary` doit
   etre quasi nul ; les runs confirment que le verrou etait `T_kernel` cote
   loader, pas la frontiere Python.
3. Le transport FV pur est surtout `T_kernel` memory-bound. En OpenMP il scale
   positivement mais sous-lineaire (`5.7x` a 16 threads sur le grand cas),
   ce qui est coherent avec une bande passante memoire partagee et des
   allocations temporaires visibles.
4. Le bench MPI transport multi-box qui termine (`647836`) fait exploser
   `T_halo` et `T_reduction`. Sur CPU, son weak scaling tombe de `25.23 ms` a
   `659.29 ms` entre `1` et `8` rangs malgre une taille locale comparable :
   ce n'est pas un manque de calcul local, c'est la communication. Les timeouts
   `648114/648115` disent autre chose : le harnais `scaling_step` multi-rang
   ne fournit pas encore de point exploitable apres `#254`.
5. Poisson/MG est domine par `T_poisson`, `T_reduction` et les synchronisations
   de niveaux grossiers. Le premier `profile_step` et les matrices
   `647857/647858` montrent que les petits solveurs elliptique/AMR ne scalent
   pas simplement avec les ressources ; la theorie MG predit ce mur des niveaux
   grossiers et des collectives.
6. Sur GPU, une lecture hote ou un diagnostic peut ajouter `T_fence`, mais les
   runs actuels montrent surtout que le transport mono-GPU a un debit stable
   autour de `3.3-3.5e7 cells/s`. Le multi-GPU ne peut pas etre interprete tant
   que le harnais multi-rang time-out avant de produire un point valide.

Synthese : le probleme observe n'est pas "C++ vs Python" mais "hot path natif
correctement specialise" et "communication/synchronisation". Pour les fronts
Python, la theorie attend une quasi-parite si le calcul reste en C++ ; c'est ce
qui est mesure. Pour MPI/GPU, la theorie attend que halos, reductions et
fences dominent des que le calcul local est trop petit ou que les echanges se
bloquent ; c'est exactement ce que montrent les campagnes.

### 10.10-d Graphes integres et lecture

Cette section embarque les graphes produits par les CSV/JSONL de campagne. Les
figures ne remplacent pas les tableaux : elles servent a voir d'un coup les
regimes dominants.

#### Frontends apres correction partielle DSL/Kokkos locale

![Ratios frontends apres correction partielle DSL/Kokkos locale](perf_figures_frontends_dslkokkosfix_648034/frontend_ratios.png)

Lecture : `python-bricks` reste au niveau C++ dans le bruit de mesure. Le DSL
`production` n'est plus plat a tous les threads : il suit enfin le scaling
OpenMP, mais garde dans ce run local un residu d'environ `+9 %` a `4/8`
threads. Comme `648034` finit en `rc=134`, ce graphe est une preuve de
diagnostic, pas une mesure finale. La branche amont
`feat/dsl-production-optflags` explique ce residu par les flags du `.so` et le
runtime Kokkos unique.

![Impact cold/warm DSL apres correction partielle](perf_figures_frontends_dslkokkosfix_648034/dsl_cold_warm.png)

Lecture : le cout froid du DSL est un cout de compilation, pas un cout de
runtime. Le cache warm ramene `compile_ms` a quelques millisecondes. La theorie
`T_total = T_compile + Nsteps*T_step` est donc confirmee : pour un petit run,
le cold compile domine; pour un run long ou cache warm, il s'amortit. La valeur
cold de ce graphe ne doit pas etre extrapolee au correctif amont final, qui ne
linke plus `libkokkos*`.

![Impact diagnostics frontends apres correction partielle](perf_figures_frontends_dslkokkosfix_648034/diagnostics_io_impact.png)

Lecture : sur CPU, l'extraction finale ne domine pas ce cas. Elle reste a
recontroler sur GPU, ou une lecture hote peut ajouter un `Kokkos::fence` et une
migration memoire.

#### Campagne finale CPU/Poisson/frontends avant correction DSL amont

![Strong scaling speedup campagne finale](perf_figures_647780_647781_647815_647836_647848/strong_scaling_speedup.png)

![Strong scaling efficiency campagne finale](perf_figures_647780_647781_647815_647836_647848/strong_scaling_efficiency.png)

Lecture : les premiers points `profile_step` sont Poisson-dominants et ne
doivent pas etre lus comme transport FV pur. Le rattrapage transport multi-box
montre un scaling OpenMP positif, mais MPI devient negatif des que halos et
reductions prennent le dessus.

![Weak scaling efficiency campagne finale](perf_figures_647780_647781_647815_647836_647848/weak_scaling_efficiency.png)

Lecture : le weak scaling CPU multi-box degrade fortement. A taille locale
comparable, `T_halo` puis `T_reduction` croissent trop vite; ce n'est pas un
probleme de Python ni de Poisson dans le cas transport pur.

![Breakdown phases campagne finale](perf_figures_647780_647781_647815_647836_647848/phase_breakdown_stacked.png)

Lecture : ce graphe justifie les conclusions par phase. Poisson domine le
premier bench system-like; halos/reductions dominent le multi-rang transport.
Il faut donc optimiser/profiler par phase, pas seulement regarder un temps
total.

![Ratios frontends campagne finale](perf_figures_647780_647781_647815_647836_647848/frontend_ratios.png)

![Cold/warm DSL campagne finale](perf_figures_647780_647781_647815_647836_647848/dsl_cold_warm.png)

Lecture : avant la correction amont, le DSL warm restait quasi invariant avec
les threads; c'est le signal qui a conduit au diagnostic "loader compile sans
backend Kokkos". Les graphes apres correction montrent que ce verrou a ete
leve.

#### Matrice CPU/GPU branche `feat/perf-campaign-bench`

![Strong scaling speedup matrice](perf_figures_matrix_647857_647858/strong_scaling_speedup.png)

![Strong scaling efficiency matrice](perf_figures_matrix_647857_647858/strong_scaling_efficiency.png)

![Weak scaling efficiency matrice](perf_figures_matrix_647857_647858/weak_scaling_efficiency.png)

![Breakdown phases matrice](perf_figures_matrix_647857_647858/phase_breakdown_stacked.png)

Lecture : la matrice separe mieux transport, Poisson et AMR. Le transport GPU
mono-rang atteint un debit stable, mais ce graphe ne prouve pas un scaling
multi-GPU. L'AMR multi-GPU reste negatif sur les petites tailles, conforme a
un regime communication/coarse-grid bound.

#### Relance MPI v2 apres merge `#254`

![Strong scaling MPI v2](perf_figures_mpi_v2_648114_648115/strong_scaling_speedup.png)

![Efficiency MPI v2](perf_figures_mpi_v2_648114_648115/strong_scaling_efficiency.png)

![Breakdown MPI v2](perf_figures_mpi_v2_648114_648115/phase_breakdown_stacked.png)

Lecture : seules les lignes mono-rang terminent. Les points `2/4` rangs CPU et
`2` GPU time-out et ne sont pas visibles comme speedup mesurable. Le graphe est
donc surtout une trace de reference `np=1`; le verdict multi-rang vient des
timeouts documentes dans les logs et tableaux, pas d'une courbe absente.

### 10.10-e Audit critique des resultats et de la campagne

Cette section corrige explicitement les risques d'interpretation de la
campagne. Elle ne retire pas les resultats precedents ; elle indique leur
niveau de preuve.

Ce qui est solide :

1. `python-bricks` ne reproduit pas le modele MUFFIN. La boucle chaude reste
   dans `advance(dt, nsteps)` cote C++ ; les ratios `0.93-0.97x` face au C++
   PIC sont du bruit de mesure et ne doivent pas etre lus comme "Python plus
   rapide". La conclusion fiable est la parite pratique quand les diagnostics
   et extractions NumPy sont hors boucle.
2. Le DSL `production` avant correctif avait un vrai probleme de specialisation
   C++ : temps warm presque invariant avec `1/4/8` threads. Le passage de
   `341/342/341 ms` a `346/261/193 ms` apres propagation de Kokkos prouve que
   le cout venait du backend du loader, pas d'une copie Python.
3. La theorie de cout est respectee : quand le calcul reste natif,
   `T_py_boundary` est petit ; quand les halos/reductions ou le Poisson sont
   actifs, `T_halo`, `T_reduction`, `T_poisson` et `T_fence` dominent. Les
   breakdowns par phase sont donc plus importants que les temps totaux seuls.
4. `adc::Real` vaut actuellement `double`, donc les appels MPI en `MPI_DOUBLE`
   ne sont pas une erreur de type aujourd'hui.

Ce qui est seulement diagnostic, pas une conclusion finale :

1. Le job `648034` est utile pour identifier le verrou DSL/Kokkos, mais il
   n'est pas publiable comme performance finale du DSL : il termine par
   `rc=134` apres ecriture du CSV a cause du double runtime Kokkos. Les chiffres
   de `648034` disent "le DSL re-scale avec OpenMP", pas "le DSL final coute
   +9 %".
2. Le cout cold DSL de `648034` inclut une variante locale de build. La branche
   amont qui ne linke pas `libkokkos*` et ajoute `-O3 -DNDEBUG` peut avoir un
   cold compile different ; il faut mesurer cette branche pour figer ce nombre.
3. Les graphes MPI v2 n'affichent que les lignes terminees. Les timeouts
   multi-rangs sont une information experimentale forte, mais ils ne produisent
   pas de points de speedup. Toute courbe v2 doit donc etre lue comme reference
   `np=1`, pas comme scaling.
4. Les JSONL GPU du harnais `scaling_step` contiennent `gpus:0` meme sur des
   runs GPU. Les CSV consolides ont ete corriges quand l'information etait
   connue par le job, mais le champ brut du harnais est un bug de metadonnees.
   Il n'affecte pas les temps, mais il affaiblit la provenance automatique.
5. Les JSONL v2 indiquent `adc_cpp_branch=unknown`; la provenance repose donc
   sur les logs Slurm et le SHA `1d4cd25e25`, pas sur le champ branche du JSON.

Auto-critique de ce que j'ai fait :

1. J'ai d'abord presente trop vite le correctif local DSL comme "apres
   correction". La formulation correcte est "apres correction partielle
   DSL/Kokkos locale". Le resultat utile est le changement de pente avec les
   threads ; la mesure finale doit venir de la branche amont propre.
2. J'ai aussi ete trop direct dans la lecture de `#254`. Le correctif pinned
   host existe bien dans `origin/master`, mais la worktree locale courante ne le
   contient pas encore. Les conclusions v2 concernent le code ROMEO a SHA
   `1d4cd25e25`, pas le fichier local checkout.
3. Les timeouts `648114/648115` ne suffisent pas a declarer "MPI CPU casse".
   Le job `647836` prouve que `profile_transport_mbox` CPU termine en
   `np=1/2/4/8`. Le probleme restant est plus precis : le harnais
   `scaling_step` multi-rang, a grande taille et apres merge `#254`, time-out.
4. Les ratios inferieurs a `1.0` pour `python-bricks` ne prouvent pas que
   Python accelere ADC. Ils signalent la variabilite du banc et la difficulte
   de comparer des executables/processus separes. Pour une publication, il faut
   des repetitions appariees, mediane/IC, et pinning CPU plus strict.
5. Le patch local `include/adc/runtime/abi_key.hpp` est potentiellement trop
   invasif. Il protege contre un mismatch backend header-only, mais il change
   une cle ABI publique. La branche amont resout le meme probleme par la cle de
   cache Python ; c'est probablement le chemin de moindre risque.

Risques code a investiguer avant de relancer une grosse campagne :

1. Dans `origin/master`, `fill_boundary` utilise des tampons
   `comm_allocator<Real>` en `Kokkos::SharedHostPinnedSpace`. Cela evite le
   piege CUDA IPC, mais peut encore couter cher : les kernels pack/unpack
   lisent/ecrivent de la memoire hote epinglee depuis le device. Il faut mesurer
   separement `pack`, `MPI_Waitall`, `unpack` et les fences.
2. Le protocole halo repose sur une enumeration deterministe des jobs et un
   tag MPI unique `0`. Si deux rangs divergent sur l'ordre ou la taille des
   jobs, `MPI_Waitall` peut bloquer. Le prochain test minimal doit imprimer ou
   reduire par voisin `send_bytes`, `recv_bytes`, `n_jobs` et un hash d'ordre
   avant de poster MPI.
3. `MPI_DOUBLE` est correct tant que `Real=double`. Pour rendre le code robuste,
   ajouter un `static_assert(std::is_same_v<Real,double>)` pres des appels MPI,
   ou centraliser un `mpi_datatype<Real>()`.
4. Les diagnostics GPU restent sous-instrumentes : toute extraction hote peut
   cacher un `Kokkos::fence`. Les prochains runs doivent separer diagnostics
   off, diagnostics reduction-only, extraction NumPy, et I/O.

Interpretation revisee :

Le resultat principal n'est pas "C++ bat Python" ni "Python coute cher". Le
resultat principal est que Python production coute peu si la frontiere est au
niveau `advance` et si le loader DSL est compile avec le meme backend que le
module natif. Les pertes de performance observees viennent surtout des mauvais
chemins natifs : loader DSL mal specialise, halos/reductions, Poisson/MG,
fences et metadonnees MPI. La prochaine etape technique prioritaire n'est donc
pas une nouvelle campagne de courbes, mais un micro-benchmark halo MPI qui
prouve la correspondance des buffers et decompose pack/comm/unpack.

### 10.11 Conclusion de cloture du TODO perf

Les runs ROMEO et les branches amont confirment quatre points solides :

1. Le chemin Python briques ADC ne reproduit pas le probleme MUFFIN : pas de
   callback Python par cellule, pas de copie full-array par pas dans `advance`.
2. Le verrou perf observe n'est pas Python. Selon le cas, il est soit
   Poisson/MG/reductions (`profile_step`), soit halos/reductions MPI
   (`profile_transport_mbox`).
3. Le DSL `production` est corrige cote amont par
   `feat/dsl-production-optflags` : parite backend Kokkos, `-O3 -DNDEBUG`,
   runtime Kokkos unique. Le diff restant n'est pas une copie Python.
4. Le multi-rang `scaling_step` reste bloque/time-out meme apres merge de
   `#254` dans `feat/perf-campaign-bench` (`1d4cd25e25`). Ce n'est pas un
   verdict general contre MPI CPU, mais ce n'est plus une tache de campagne
   perf : c'est un chantier code MPI/halo a isoler.

Le TODO perf est donc clos : les mesures disponibles permettent d'interpreter
ou part le temps, et les manques restants sont des blocages techniques
identifies, pas des courbes manquantes a relancer telles quelles.
