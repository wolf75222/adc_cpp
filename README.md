<div align="center">

# ADC CPP

**Coeur C++23 d'un solveur AMR / MPI / GPU pour systemes hyperbolique-elliptique couples.**

![Tests](https://img.shields.io/badge/tests-71%20ctests%20C%2B%2B%20%28serie%20%2B%20Kokkos%20Serial%29%20%7C%20%2B21%20MPI%20%7C%2026%20Python-brightgreen)

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

`adc_cpp` est la **bibliotheque** : le moteur generique (coeur sans modele) **plus** une
bibliotheque de briques physiques (`include/adc/physics/`) et **les bindings Python de la lib**,
le module `adc` (composition `System` / `AmrSystem`). Le coeur est **agnostique au modele** :
il ne nomme aucun scenario, il ne fournit que des briques generiques (etat, transport, source,
second membre elliptique) composees en `CompositeModel` ; les scenarios nommes (diocotron,
Euler-Poisson, deux-fluides...) vivent cote application. Le depot separe
**[`adc_cases`](https://github.com/wolf75222/adc_cases)** contient les **cas d'utilisation**
(un dossier par cas) qui importent ce module : essentiellement du Python pilotant les briques
generiques ; un cas SUR MESURE peut porter son propre C++ compile a la volee contre les en-tetes
generiques d'`adc_cpp` (cf. le scenario `two_fluid_ap/`, l'integrateur AP deux-fluides sorti du coeur).

Le coeur resout, sur maillage cartesien adaptatif, la partie generique :

```
d U / d t  +  div F(U, aux)  =  S(U, aux)
D phi = f(U)
```

ou la partie hyperbolique (U) et la partie elliptique (phi) sont couplees a chaque pas
via le canal `aux`. Contrat de base `(phi, grad_x, grad_y)`, desormais EXTENSIBLE : un modele
declare `n_aux` pour lire des champs supplementaires (`B_z` fourni, `T_e` derive p/rho d'un bloc
fluide), avec retro-compat bit-exacte si `n_aux=3`. Le coeur ne connait aucun modele : il fournit les contrats
(`PhysicalModel`, `EllipticSolver`), les operateurs, l'elliptique, les integrateurs, l'AMR
et les seams de parallelisme. Les briques physiques (etat compressible/isotherme, flux d'Euler,
flux Roe, source de potentiel/gravite, second membre de charge) vivent dans
`include/adc/physics/` ; le module Python les compose en `CompositeModel`.

## Ce que fournit le coeur

| Module | Role |
|---|---|
| [`core/physical_model.hpp`](include/adc/core/physical_model.hpp) | concept `PhysicalModel` (flux, max_wave_speed, source, elliptic_rhs) |
| [`core::{EquationBlock,CoupledSystem}`](include/adc/core/equation_block.hpp) | bundle par espece (modele + schema spatial + politique temps + BC) et systeme de N especes |
| [`physics::bricks` / `composite`](include/adc/physics/bricks.hpp) | briques generiques (etat, transport, source, elliptique) composees en `CompositeModel` |
| [`numerics::{RusanovFlux,HLLFlux,HLLCFlux,RoeFlux}`](include/adc/numerics/numerical_flux.hpp) | flux numeriques (politiques `ADC_HD`) |
| [`numerics::reconstruction`](include/adc/numerics/reconstruction.hpp) | MUSCL ordre 2 (NoSlope / Minmod / VanLeer) + WENO5Z |
| [`numerics::assemble_rhs` / `compute_face_fluxes`](include/adc/numerics/spatial_operator.hpp) | `R = -div F + S`, flux de face pour le reflux (diffusion incluse) ; GPU via `for_each_cell` |
| [`numerics::time::{TimePolicy,SSPRK2,SSPRK3}`](include/adc/numerics/time/time_integrator.hpp) | par bloc : explicite / implicite / IMEX, sous-pas (`substeps`) ET cadence (`stride`) |
| [`numerics::time::{ForwardEuler,SSPRK2Step,SSPRK3Step}`](include/adc/numerics/time/time_steppers.hpp) | integrateurs en temps OBJETS (`take_step(rhs, U, dt)`) ; l'utilisateur peut fournir le sien |
| [`numerics::time::{ImplicitSourceStepper,backward_euler_source}`](include/adc/numerics/time/implicit_stepper.hpp) | defaut implicite (Newton local) ; IMEX partiel via `Model::is_implicit(c)` |
| [`numerics::time::advance_subcycled`](include/adc/numerics/time/scheduler.hpp) | scheduler : sous-pas + cadence (macro-pas) par `EquationBlock` |
| [`numerics::time::imex_euler_step`](include/adc/numerics/time/imex.hpp) | IMEX asymptotic-preserving |
| [`numerics::time::{lie_step,strang_step}`](include/adc/numerics/time/splitting.hpp) | splitting d'operateurs |
| [`numerics::time::advance_amr`](include/adc/numerics/time/amr_reflux_mf.hpp) | moteur AMR unifie : multi-patch N-niveaux, reflux coverage-aware, distribue MPI |
| [`numerics::elliptic::GeometricMG`](include/adc/numerics/elliptic/geometric_mg.hpp) | multigrille geometrique (V-cycle GS rb), AMR-compatible, on-device ; eps(x) variable cote coeur (`set_epsilon`) |
| [`numerics::elliptic::PoissonFFTSolver` / `DistributedFFTSolver`](include/adc/numerics/elliptic) | Poisson FFT spectral (mono-rang) et distribue (MPI), correctif `n` non puissance de 2 |
| [`coupling::{ChargeDensityRhs,CoupledSource}`](include/adc/coupling/elliptic_rhs.hpp) | RHS de systeme `f = Σ_s q_s n_s` (N especes) ; source inter-especes `S(U_e,U_i,φ)` |
| [`coupling::Coupler`](include/adc/coupling/coupler.hpp) | couplage hyperbolique-elliptique mono-modele : `Coupler<Model, Elliptic>` |
| [`coupling::{SystemAssembler,SystemDriver}`](include/adc/coupling/system_coupler.hpp) | multi-especes mono-niveau : l'**assembleur** assemble (Poisson de systeme + aux), le **driver** avance (`step`, `step_cfl`, `step_adaptive`) ; `SystemCoupler` = alias du driver |
| [`coupling::AmrSystemCoupler`](include/adc/coupling/amr_system_coupler.hpp) | le systeme multi-especes porte sur **AMR** (Poisson grossier + reflux par bloc) |
| [`coupling::AmrCouplerMP`](include/adc/coupling) | couplage AMR multi-patch mono-modele (route par `advance_amr`) |
| [`amr::{cluster,regrid,tag_box}`](include/adc/amr) | tagging + clustering Berger-Rigoutsos + regrid |
| [`mesh::{MultiFab,BoxArray,Geometry}`](include/adc/mesh) | conteneurs distribues, halos, geometrie |
| [`runtime::{System,AmrSystem}`](include/adc/runtime/system.hpp) | facades runtime de composition (assise des bindings Python) |
| [`runtime::{model_factory,model_spec}`](include/adc/runtime/model_factory.hpp) | assemblage d'un `CompositeModel` a partir d'une spec de briques |
| [`runtime::{dynamic_model,compiled_block_abi,dsl_block}`](include/adc/runtime/dsl_block.hpp) | dispatch JIT (`.so`, `IModel` virtuel) et AOT (bloc compile) d'un modele genere par le DSL |
| seams [`for_each_cell`](include/adc/mesh/for_each.hpp), [`comm`](include/adc/parallel/comm.hpp) | dispatch serie/OpenMP/Kokkos, comm MPI |

Concepts et seams : [**docs/ARCHITECTURE.md**](docs/ARCHITECTURE.md). Algorithmes :
[docs/ALGORITHMS.md](docs/ALGORITHMS.md). Profil : [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## DSL symbolique (prototype)

Un mini-DSL symbolique cote Python decrit un modele en FORMULES et le transforme en brique C++ :
emission de la fonction de flux (`emit_cpp_brick`), de la source et du second membre elliptique,
elimination de sous-expressions communes (CSE). Deux chemins de mise en oeuvre :

- **JIT** (`.so` + `IModel` virtuel) : `System.add_dynamic_block` charge un modele a dispatch
  virtuel, pour le prototypage hote ;
- **AOT** : `compile_or_jit(mode="compile")` produit un bloc compile via `compiled_block_abi.hpp`,
  branche par `System.add_compiled_block` (`.so`, marshaling de tableaux plats cote hote, sans
  AMR ni MPI) ; et `System.add_compiled_model` (`dsl_block.hpp`) branche un `CompositeModel`
  connu a la compilation comme un bloc NATIF, **valide bit-identique a `add_block` sur CPU/Serial**
  (`test_compiled_model_parity`). Le chemin device-clean equivalent passe par les FONCTEURS NOMMES
  de `block_builder.hpp` (la machinerie d'`add_compiled_model`, instanciee depuis une unite de
  traduction externe sans lambda etendue) : il est valide BIT-IDENTIQUE jusqu'au GH200, multi-box et
  MPI multi-GPU (`test_mpi_mbox_parity`, np=1/2/4). En revanche la variante `add_compiled_model` a
  LAMBDAS ETENDUES bute encore sur une limite nvcc (lambda etendue `__host__ __device__` instanciee
  dans une TU externe) : la parite zero-copie sur device par CE chemin n'est pas acquise. Voir
  [docs/GPU_RUNTIME_PORT.md](docs/GPU_RUNTIME_PORT.md).

## Systemes multi-especes

On couple N especes (ions, electrons, neutres...), chacune avec **son** `PhysicalModel`, son
schema spatial, sa politique temporelle. Les especes interagissent dans le **second membre
elliptique** (`f = Σ_s q_s n_s`, `ChargeDensityRhs`) et dans la **source** (`CoupledSource`,
`S` inter-especes), jamais dans le flux. Tout est composable par bloc :

| Capacite | Comment |
|---|---|
| especes heterogenes (Euler 4 var + isotherme 3 var + ...) | `CoupledSystem<Blocks...>` |
| schema spatial different par espece (HLLC electrons, Rusanov ions...) | `EquationBlock<Model, SpatialDiscretisation, Time>` |
| sous-pas plus frequents (10 electrons : 1 ion) | `ExplicitTime<Method, /*substeps*/10>` |
| cadence plus lente (gaz pas resolu tous les pas) | `ExplicitTime<Method, 1, /*stride*/3>` |
| pas adaptatif multirate (stride derive du CFL par espece) | `SystemDriver::step_adaptive(cfl)` |
| electrons implicites + ions explicites | `ImplicitTime` / `ExplicitTime` par bloc |
| implicite sur une PARTIE des variables (coute moins cher) | `Model::is_implicit(c)` (IMEX partiel) |
| espece en profil constant donne | `PrescribedTime` (le scheduler la saute) |
| integrateur en temps fait maison | un objet a `take_step` donne comme `Method` du bloc |

Le `SystemDriver` *avance*, le `SystemAssembler` *assemble* (Poisson de systeme + aux) :
« avancer un assembleur » n'a pas de sens, c'est le driver qui avance. Porte sur AMR par
`AmrSystemCoupler`.

## Backends : configures UNE fois, herites partout

MPI, HDF5 et Kokkos sont attaches a la cible d'interface `adc` ; tout ce qui lie `adc` en
herite. Aucun drapeau rebadge par cible. **Kokkos est le backend de dispatch recommande** :
il couvre le CPU multi-thread (device OpenMP) ET le GPU (CUDA/HIP) avec un seul code. Le
backend OpenMP autonome (`ADC_USE_OPENMP`) est **deprecie** au profit de Kokkos.

```bash
cmake -B build                       # serie
cmake -B build -DADC_USE_MPI=ON      # distribue (halos + FFT par MPI)
cmake -B build -DADC_USE_KOKKOS=ON   # CPU multi-thread (device OpenMP), recommande
cmake -B build -DADC_USE_KOKKOS=ON \ # GPU GH200
   -DCMAKE_CXX_COMPILER=$K/bin/nvcc_wrapper -DKokkos_ROOT=$K
```

Le seam `for_each_cell` bascule serie -> `Kokkos::parallel_for` (OpenMP ou Cuda) sans toucher
les operateurs (Kokkos est initialise paresseusement, aucun `Kokkos::initialize` a ecrire).

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

## Module Python `adc` (bindings de la lib)

Les bindings vivent **ici** (`python/`), pas dans l'application : ce sont les bindings de la
lib. On construit le module avec `-DADC_BUILD_PYTHON=ON` :

```bash
cmake -B build-py -DADC_BUILD_PYTHON=ON && cmake --build build-py --target _adc -j
export PYTHONPATH=$PWD/build-py/python        # contient le paquet adc/
ctest --test-dir build-py                     # lance test_bindings
```

L'extension est liee a une version de Python (suffixe ABI `cpython-3XY`). Construire et
lancer avec le MEME interpreteur ; en cas de Python multiples, pinner explicitement :
`cmake -B build-py -DADC_BUILD_PYTHON=ON -DPython_EXECUTABLE=$(which python3.12)`. La ligne
`adc Python module: interpreteur ...` affichee a la configuration indique l'interpreteur retenu.

Le module expose deux niveaux, dans l'esprit "Python compose QUOI, le C++ calcule" :

```python
import adc
sim = adc.System(n=192, periodic=False)            # config = maillage seul
# un modele = COMPOSITION de briques generiques (le coeur ne nomme aucun scenario)
electrons = adc.Model(state=adc.FluidState("compressible", gamma=1.4),
                      transport=adc.CompressibleFlux(),
                      source=adc.PotentialForce(charge=-1.0),
                      elliptic=adc.ChargeDensity(charge=-1.0))
ions = adc.Model(state=adc.FluidState("isothermal", cs2=0.5),
                 transport=adc.IsothermalFlux(),
                 source=adc.PotentialForce(charge=+1.0),
                 elliptic=adc.ChargeDensity(charge=+1.0))
# un BLOC par espece : modele compose + schema spatial + traitement temporel, au choix
sim.add_block("electrons", model=electrons,
              spatial=adc.Spatial(vanleer=True, flux="hllc"), time=adc.IMEX(substeps=10))
sim.add_block("ions", model=ions,
              spatial=adc.Spatial(minmod=True), time=adc.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet",
                wall="circle", wall_radius=0.40)   # paroi conductrice = embedded boundary
sim.set_density("electrons", ne_numpy)             # CI ecrite en numpy
sim.step_cfl(0.4)
```

- **`adc.System`** (composition multi-blocs) : `add_block(name, model, spatial=adc.Spatial(...),
  time=adc.Explicit()|adc.IMEX(...)|adc.Implicit(...))`, `set_poisson(...)`, `set_density`,
  `step`/`advance`/`step_cfl`, diagnostics. Un `model` est une COMPOSITION de briques generiques
  `adc.Model(state, transport, source, elliptic)` : etat (`Scalar`/`FluidState`), transport
  (`ExB`/`CompressibleFlux`/`IsothermalFlux`), source (`NoSource`/`PotentialForce`/`GravityForce`),
  second membre elliptique (`ChargeDensity`/`BackgroundDensity`/`GravityCoupling`). Le coeur ne
  nomme aucun scenario ; les compositions nommees vivent cote application (`adc_cases/models.py`).
  Le choix implicite/explicite est **par bloc et reversible** ; aucun callback Python dans le hot path.
- **Integrateur temporel ecrit en Python** : primitives `solve_fields()`, `eval_rhs(name)`,
  `get_state`/`set_state` : on ecrit son propre `take_step` cote Python (par PAS), le residu
  et Poisson restant calcules en C++ (par CELLULE). Cf. `adc.integrate.ssprk2_step`.
- **AMR** : `adc.AmrSystem` compose un bloc sur une hierarchie raffinee (API proche de System
  plus `set_refinement`). Il partage desormais l'operateur spatial de `System` : reconstruction
  primitive et flux HLLC/Roe, transmis via `advance_amr` (cf. `test_amr_spatial_parity`) ; il
  reste mono-bloc et explicite. L'integrateur AP deux-fluides (asymptotic-preserving) est un
  integrateur **sur mesure**, non composable bloc a bloc : ce n'est pas une brique generique
  mais un SCENARIO, qui a quitte le coeur et vit dans `adc_cases/two_fluid_ap/` (physique C++
  compilee a la volee contre les en-tetes generiques d'`adc_cpp`). Le module `_adc` ne l'expose plus.

Le test `python/tests/test_bindings.py` exerce ces chemins. Exemples complets : depot
[`adc_cases`](https://github.com/wolf75222/adc_cases) (un dossier Python par cas).
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
ctest --test-dir build                 # 71 tests coeur (maillage, AMR, elliptique, integrateurs)
```

La CI a trois jobs : Release (serie), MPI (build `-DADC_USE_MPI=ON`, qui ajoute `+21` entrees
`ctest` lancees par `mpirun`, bit-identiques np=1/2/4) et Kokkos (backend Serial). Le module
Python ajoute une suite (`26` tests : bindings + DSL). La CI ignore les changements purement
documentaires (`paths-ignore: docs/**`, `**.md`).

| Option | Defaut | Role |
|---|---|---|
| `ADC_BUILD_TESTS` | `ON` | suite CTest du coeur |
| `ADC_USE_KOKKOS` | `OFF` | dispatch Kokkos (CPU OpenMP + GPU), **recommande** |
| `ADC_USE_OPENMP` | `OFF` | dispatch OpenMP autonome, **deprecie** (utiliser Kokkos) |
| `ADC_USE_MPI` | `OFF` | backend distribue (comm, halos, FFT) |
| `ADC_USE_HDF5` | `OFF` | DataWriter HDF5 parallele |
| `ADC_USE_EIGEN` | `ON` | cible d'analyse host `adc_eigen` (utilisee par adc_cases) |

## Organisation du depot

```
include/adc/
  core/         types, etat, PhysicalModel, EquationBlock, CoupledSystem, variables
  mesh/         MultiFab, BoxArray, Geometry, for_each_cell, CL physiques
  parallel/     seam comm (MPI), load balance
  physics/      briques generiques (etat, transport, source, elliptique) -> CompositeModel
  numerics/     reconstruction, flux numeriques (Rusanov/HLL/HLLC/Roe), operateur spatial
  numerics/elliptic/  concept EllipticSolver, multigrille, FFT (mono / distribue)
  numerics/time/      SSP-RK, TimePolicy, scheduler, IMEX, splitting, moteur AMR
  coupling/     Coupler, SystemCoupler, AmrSystemCoupler, AmrCouplerMP, diagnostics
  amr/          clustering Berger-Rigoutsos, regrid, hierarchie
  runtime/      facades System / AmrSystem, model_factory, JIT/AOT du DSL, canal aux extensible
tests/          71 tests coeur (+21 entrees ctest MPI via mpirun quand -DADC_USE_MPI=ON)
docs/           ARCHITECTURE.md, ALGORITHMS.md, GPU_RUNTIME_PORT.md, CHOICES.md, PERFORMANCE.md, BIBLIOGRAPHY.md ; archive/ (planning + notes applicatives)
```

## Validation (coeur)

- **71** ctests coeur, joues sur deux builds en CI : Release (serie) et Kokkos (backend Serial) ;
  **+21** entrees ctest MPI bit-identiques np=1/2/4 (build `-DADC_USE_MPI=ON`). Le module Python
  ajoute **26** tests (bindings + DSL).
- AMR conservatif : reflux multi-patch a l'arrondi machine (`~1e-15`).
- GPU GH200 (backend Kokkos Cuda, validations integrees au depot, hors CI car le materiel n'est
  pas un runner) :
  - composants valides bit-identiques au CPU (System mono-grille, ops de champ AMR, halos MPI
    multi-GPU, chemin compile a foncteurs nommes multi-box + MPI) ;
  - validation INTEGREE AmrSystem + MPI + GPU FAITE (les trois axes dans un seul run) : densite
    grossiere bit-identique au nombre de rangs (np=1/2/4, `dmax=0`), masse conservee a `0` ;
  - features a chemin device fusionnees apres la validation integree confirmees bit-identiques :
    `T_e` lu via le canal aux, EPM ecrante/Helmholtz, EPM anisotrope `eps_x/eps_y`, B_z par niveau AMR.
  - CAVEATS HONNETES : (1) un grossier multi-box DISTRIBUE n'est pas bit-identique sur les sommes
    globales (`mass`/`csum`) entre np, par ordre de reduction FMA (le max reste exact) ; (2) la
    facade `AmrSystemCoupler` complete ne s'instancie pas sous nvcc (concept + lambda generique),
    donc le chemin device de B_z par niveau est valide via `advance_amr`/`assemble_rhs` ; (3)
    `System::add_compiled_model` a lambdas etendues n'est pas zero-copie sur device (limite nvcc) ;
    (4) le strong-scaling AMR par grossier reparti est cable et correct mais NEGATIF a cette echelle
    (la latence des halos multi-box cross-rang domine le compute economise). Detail :
    [docs/GPU_RUNTIME_PORT.md](docs/GPU_RUNTIME_PORT.md).

Validation bout-en-bout (modeles, taux diocotron, runs ROMEO a grande echelle) : depot
[`adc_cases`](https://github.com/wolf75222/adc_cases). Trace des validations device : [docs/GPU_RUNTIME_PORT.md](docs/GPU_RUNTIME_PORT.md).
