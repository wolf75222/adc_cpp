# Matrice des backends

`adc_cpp` compile le meme code numerique vers plusieurs backends via un seam de dispatch
unique. La couverture de tests par backend est tenue a jour dans un document unique, qui est la
source de verite : [BACKEND_COVERAGE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md). Cette page en resume la
structure ; ne pas dupliquer le tableau detaille ici (il evolue a chaque ajout de test).

## Les backends

| Backend | Build | Ou il tourne |
|---------|-------|--------------|
| **Kokkos Serial** | `-DADC_USE_KOKKOS=ON`, `Kokkos_ENABLE_SERIAL=ON` | CI gate `build-and-test` (toute PR), C++ + module Python. |
| **MPI + Kokkos Serial** | `-DADC_USE_MPI=ON -DADC_USE_KOKKOS=ON` | CI job MPI (`ci-full`), `mpirun` np=1/2/4. |
| **Kokkos OpenMP** | `-DADC_USE_KOKKOS=ON`, `Kokkos_ENABLE_OPENMP=ON` | CI job `ci-full` (active depuis #155). |
| **Kokkos Cuda (GH200)** | Kokkos + `Kokkos_ARCH_HOPPER90`, un GPU par rang | ROMEO manuel uniquement (jamais en CI). |
| **MPI + Kokkos Cuda** | build precedent + OpenMPI CUDA-aware | ROMEO manuel (`srun -n {1,2,4}`). |

```{important}
La CI ne construit jamais `-DADC_USE_KOKKOS=ON -DKokkos_ENABLE_CUDA=ON`. Toute validation GPU
(Cuda mono-GPU ou MPI multi-GPU) est faite manuellement sur le supercalculateur ROMEO (noeud
GH200), pas dans la CI. Les harnesses GPU vivent dans `python/tests/gpu/` et sont lances par
SBATCH. Voir [GPU_RUNTIME_PORT.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/GPU_RUNTIME_PORT.md) et la page
[limitations](limitations.md).
```

## Comment lire la matrice

Le tableau de [BACKEND_COVERAGE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md) croise chaque test (C++ ctest et
Python) avec les backends ci-dessus. La legende y distingue notamment :

- **build-and-test** : gate OBLIGATOIRE sur toute PR ; compile en Kokkos Serial (C++ + module
  Python).
- **ci-full** : tourne en mode plein (push `master`, nightly, PR labellisee), ajoute MPI + Kokkos
  Serial et Kokkos OpenMP.
- **ROMEO** : valide manuellement sur GH200 ; le harness exact est cite entre parentheses, avec
  l'evidence chiffree (p.ex. `dmax=0`).
- **self-skip** : le test detecte l'absence du backend et sort proprement (exit 0).
- **?** : non exerce ; ces cellules sont listees dans la section "Lacunes notables" du
  document source.

## Points saillants

- Les tests Python n'exercent que Kokkos Serial : le module `_adc` est construit en CI en
  Kokkos Serial, sans MPI ni Cuda. Aucun test Python ne couvre MPI ou Cuda.
- La voie FFT Poisson (`PoissonFFTSolver`) est refusee sous MPI (single-rank par design) ;
  un test verrouille cette non-regression.
- Les colonnes Kokkos Cuda et MPI + Kokkos Cuda sont soit `ROMEO` (validees a la main),
  soit `?`. Le detail des gaps GPU restants est dans la section "Lacunes notables" du document
  source.

Pour le decompte chiffre (cibles ctest, entrees MPI, tests Python) et la liste exhaustive des
gaps, se referer directement a [BACKEND_COVERAGE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md).
