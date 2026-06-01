<div align="center">

# ADC CPP

**Coeur C++23 d'un solveur AMR / MPI / GPU pour systemes hyperbolique-elliptique couples.**

![Tests](https://img.shields.io/badge/tests-32-brightgreen)

</div>

<p align="center">
  <img src="docs/anim_romeo_diocotron_amr3.gif" alt="Instabilite diocotron AMR 3 niveaux sur ROMEO" width="640">
</p>

<div align="center">
<sub>
Exemple produit avec ce moteur (via le depot applicatif <code>adc_cases</code>) : instabilite diocotron
(derive E x B) sur AMR 3 niveaux emboites, ROMEO (x64cpu, 96 coeurs AMD EPYC). Patchs fins suivis par
regrid Berger-Rigoutsos, sous-cyclage Berger-Oliger + reflux conservatif (derive de masse ~ 1e-15).
</sub>
</div>

---

`adc_cpp` est la **bibliotheque coeur** : le moteur generique, sans aucun modele physique.
Les modeles, les facades compilees, les exemples, les bindings Python et les cas
d'utilisation vivent dans le depot separe **[`adc_cases`](https://github.com/wolf75222/adc_cases)**,
qui consomme ce coeur via la cible CMake `adc::adc`.

Le coeur resout, sur maillage cartesien adaptatif, la partie generique :

```
d U / d t  +  div F(U, aux)  =  S(U, aux)
D phi = f(U)
```

ou la partie hyperbolique (U) et la partie elliptique (phi) sont couplees a chaque pas
via `aux = (phi, grad phi)`. Le coeur ne connait aucun modele : il fournit les contrats
(`PhysicalModel`, `EllipticSolver`), les operateurs, l'elliptique, les integrateurs, l'AMR
et les seams de parallelisme. Un modele concret (diocotron, Euler-Poisson, deux-fluides)
est fourni par l'application.

## Ce que fournit le coeur

| Module | Role |
|---|---|
| [`core/physical_model.hpp`](include/adc/core/physical_model.hpp) | concept `PhysicalModel` (flux, max_wave_speed, source, elliptic_rhs) |
| [`core::{EquationBlock,CoupledSystem}`](include/adc/core/equation_block.hpp) | squelette multi-blocs : state + modele + methode spatiale + politique temps |
| [`operator::{RusanovFlux,HLLFlux,HLLCFlux}`](include/adc/operator/numerical_flux.hpp) | flux numeriques (politiques `ADC_HD`) |
| [`operator::reconstruction`](include/adc/operator/reconstruction.hpp) | MUSCL ordre 2 (NoSlope / Minmod / VanLeer) + WENO5Z |
| [`operator::assemble_rhs` / `compute_face_fluxes`](include/adc/operator/spatial_operator.hpp) | `R = -div F + S`, flux de face pour le reflux ; GPU via `for_each_cell` |
| [`integrator::{TimePolicy,SSPRK2,SSPRK3}`](include/adc/integrator/time_integrator.hpp) | choix explicite / implicite / IMEX + sous-pas par bloc |
| [`integrator::advance_subcycled`](include/adc/integrator/scheduler.hpp) | scheduler generique pour avancer chaque `EquationBlock` avec son nombre de sous-pas |
| [`integrator::imex_euler_step`](include/adc/integrator/imex.hpp) | IMEX asymptotic-preserving |
| [`integrator::{lie_step,strang_step}`](include/adc/integrator/splitting.hpp) | splitting d'operateurs |
| [`integrator::advance_amr`](include/adc/integrator/amr_reflux_mf.hpp) | moteur AMR unifie : multi-patch N-niveaux, reflux coverage-aware, distribue MPI |
| [`elliptic::GeometricMG`](include/adc/elliptic/geometric_mg.hpp) | multigrille geometrique (V-cycle GS rb), AMR-compatible, on-device |
| [`elliptic::PoissonFFTSolver` / `DistributedFFTSolver`](include/adc/elliptic) | Poisson FFT spectral (mono-rang) et distribue (MPI) |
| [`coupling::elliptic_rhs`](include/adc/coupling/elliptic_rhs.hpp) | assembleurs de second membre elliptique mono-modele ou multi-champs |
| [`coupling::Coupler`](include/adc/coupling/coupler.hpp) | couplage hyperbolique-elliptique par etage : `Coupler<Model, Elliptic>` |
| [`coupling::SystemCoupler`](include/adc/coupling/system_coupler.hpp) | execution mono-niveau d'un `CoupledSystem` multi-blocs |
| [`coupling::AmrCouplerMP`](include/adc/coupling) | couplage AMR multi-patch (route par `advance_amr`) |
| [`amr::{cluster,regrid,tag_box}`](include/adc/amr) | tagging + clustering Berger-Rigoutsos + regrid |
| [`mesh::{MultiFab,BoxArray,Geometry}`](include/adc/mesh) | conteneurs distribues, halos, geometrie |
| seams [`for_each_cell`](include/adc/mesh/for_each.hpp), [`comm`](include/adc/parallel/comm.hpp) | dispatch serie/OpenMP/Kokkos, comm MPI |

Concepts et seams : [**docs/ARCHITECTURE.md**](docs/ARCHITECTURE.md). Algorithmes :
[docs/ALGORITHMS.md](docs/ALGORITHMS.md). Profil : [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Backends : configures UNE fois, herites partout

OpenMP, MPI, HDF5 et Kokkos sont attaches a la cible d'interface `adc` ; tout ce qui lie
`adc` en herite. Aucun drapeau rebadge par cible.

```bash
cmake -B build                       # serie
cmake -B build -DADC_USE_OPENMP=ON   # CPU multi-thread
cmake -B build -DADC_USE_MPI=ON      # distribue (halos + FFT par MPI)
cmake -B build -DADC_USE_KOKKOS=ON \ # GPU GH200 (ou CPU portable)
   -DCMAKE_CXX_COMPILER=$K/bin/nvcc_wrapper -DKokkos_ROOT=$K
```

Le seam `for_each_cell` bascule serie -> `#pragma omp` -> `Kokkos::parallel_for` (Cuda)
sans toucher les operateurs.

## Utiliser le coeur

Depuis une application (cf. `adc_cases`), on tire le coeur via FetchContent et on lie
`adc::adc` :

```cmake
include(FetchContent)
FetchContent_Declare(adc_cpp GIT_REPOSITORY https://github.com/wolf75222/adc_cpp.git)
FetchContent_MakeAvailable(adc_cpp)
target_link_libraries(mon_appli PRIVATE adc::adc)
```

On definit alors un type qui satisfait `PhysicalModel`, on l'instancie dans un
`Coupler<Model, Elliptic>` (ou `AmrCouplerMP` pour l'AMR), et on avance en temps.
Modeles prets, facades, exemples et Python : voir **`adc_cases`**.

## Ecosysteme

| Repo | Role | Socle maillage |
|---|---|---|
| **`adc_cpp`** (ce depot) | coeur hyperbolique-elliptique sur **AMR** + GPU/MPI/Kokkos | propre (from scratch) |
| [`adc_cases`](https://github.com/wolf75222/adc_cases) | applications : modeles, facades, exemples, Python | consomme `adc::adc` |
| [`poisson_cpp`](https://github.com/wolf75222/poisson_cpp) | solveurs Poisson (Thomas, SOR, CG, DST, multigrille) | propre |
| [`pde_core_cpp`](https://github.com/wolf75222/pde_core_cpp) | infra partagee (mesh, fields, AMR) | propre |
| [`advection_cpp`](https://github.com/wolf75222/advection_cpp) | advection + Burgers + Chorin NS | `pde_core_cpp` |
| [`euler_cpp`](https://github.com/wolf75222/euler_cpp) | Euler 2D + viscous NS + sources plasma | `pde_core_cpp` |

## Build et tests

```bash
git clone https://github.com/wolf75222/adc_cpp.git
cd adc_cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build                 # 32 tests coeur (maillage, AMR, elliptique, integrateurs)
```

| Option | Defaut | Role |
|---|---|---|
| `ADC_BUILD_TESTS` | `ON` | suite CTest du coeur |
| `ADC_USE_OPENMP` | `OFF` | dispatch OpenMP |
| `ADC_USE_KOKKOS` | `OFF` | dispatch Kokkos (CPU/GPU) |
| `ADC_USE_MPI` | `OFF` | backend distribue (comm, halos, FFT) |
| `ADC_USE_HDF5` | `OFF` | DataWriter HDF5 parallele |
| `ADC_USE_EIGEN` | `ON` | cible d'analyse host `adc_eigen` (utilisee par adc_cases) |

## Organisation du depot

```
include/adc/
  core/        types, etat, PhysicalModel, EquationBlock, CoupledSystem
  mesh/        MultiFab, BoxArray, Geometry, for_each_cell, CL physiques
  parallel/    seam comm (MPI)
  operator/    reconstruction, flux numeriques, operateur spatial
  elliptic/    concept EllipticSolver, multigrille, FFT (mono / distribue)
  coupling/    Coupler, SystemCoupler, AmrCouplerMP, diagnostics
  integrator/  SSP-RK, TimePolicy, scheduler, IMEX, splitting, moteur AMR
  amr/         clustering Berger-Rigoutsos, regrid, hierarchie
tests/         32 tests coeur (+ tests MPI via mpirun)
docs/          ARCHITECTURE.md, ALGORITHMS.md, PERFORMANCE.md, validation diocotron
```

## Validation (coeur)

- **32** tests coeur (serie ; MPI bit-identique np=1/2/4 quand `-DADC_USE_MPI=ON`).
- AMR conservatif : reflux multi-patch a l'arrondi machine (`~1e-15`).
- GPU GH200 : pas couple + AMR bit-identiques au CPU (checksum exact).

Validation bout-en-bout (modeles, taux diocotron, runs ROMEO) : depot
[`adc_cases`](https://github.com/wolf75222/adc_cases). Reference ROMEO : [docs/ROMEO.md](docs/ROMEO.md).
