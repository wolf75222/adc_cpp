# Architecture de adc_cpp

Solveur C++23 maison (inspire d'AMReX / Parthenon, ecrit from scratch) pour les
systemes **hyperbolique-elliptique couples** sur **AMR**, concu des le depart pour
**OpenMP + MPI + Kokkos**, cible cluster **ROMEO** (GH200). Cas de validation fil
rouge : l'instabilite **diocotron** (derive E x B), puis le **deux-fluides isotherme**
(type Hoffart, arXiv:2510.11808).

Ce document fige l'architecture. Le README porte la narration et les resultats ;
ici on decrit les couches, les seams et les decisions.

## 1. Principe : trois axes orthogonaux

La physique ne voit JAMAIS le parallelisme. Trois axes se composent par
templates / concepts :

```
  PHYSIQUE (point-wise)        DISCRETISATION + PARALLELISME (seams)      TEMPS
  -------------------          ------------------------------------       --------------
  PhysicalModel        --->    for_each_cell  (serial/OpenMP/Kokkos)      ssprk2/3
  NumericalFlux        --->    MultiFab + BoxArray + DistributionMapping   imex (AP)
  EllipticSolver       --->    fill_boundary / fill_ghosts  (halos, MPI)   splitting (Lie/Strang)
  CouplingPolicy       --->    AMR (hierarchy, clustering, regrid)         two_fluid_ap
                               comm (rang / all-reduce, MPI)
                               allocator (Arena memoire unifiee)
```

## 2. Les seams (points de bascule)

| Seam | Fichier | Role | Backends |
|---|---|---|---|
| `for_each_cell(box, f)` | `mesh/for_each.hpp` | boucle sur cellules | serie / `_OPENMP` / Kokkos (Cuda) |
| `Array4` + `ADC_HD` | `mesh/fab2d.hpp`, `core/types.hpp` | vue POD device-callable | identique host/device |
| `device_fence()` | `mesh/for_each.hpp` | barriere avant lecture HOTE apres kernel | `Kokkos::fence` / no-op |
| `comm` | `parallel/comm.hpp` | rang/size, all-reduce, barrier | identite serie / MPI |
| `allocator` | `core/allocator.hpp` | stockage des Fab | `std::allocator` / `cudaMallocManaged` + Arena |

**Regle de fence (discipline halo).** Toute fonction qui fait un `for_each_cell`
(kernel device) puis une boucle HOTE sur la meme memoire, dans le meme appel, doit
`device_fence()` entre les deux (sinon course memoire unifiee sur GPU, invisible en
CI CPU). Couvert : `sum`/`norm_inf`/`set_val`, `gs_rb_sweep`, `fill_physical_bc`,
les pack/unpack MPI de `fill_boundary`/`parallel_copy`, les diagnostics distribues.

## 3. Carte des modules (`include/adc/`)

| Module | Role |
|---|---|
| `core/` | `types` (`ADC_HD`, `Real`), `state` (`StateVec<N>`), `physical_model` (concept), `allocator` (Arena) |
| `mesh/` | `box2d`, `box_array`, `fab2d`/`multifab`, `for_each`, `fill_boundary`, `physical_bc`, `geometry`, `mf_arith`, `refinement`, `box_hash` |
| `operator/` | `numerical_flux` (Rusanov/HLL/HLLC, `ADC_HD`), `reconstruction` (MUSCL), `spatial_operator` (`assemble_rhs`) |
| `elliptic/` | concept `EllipticSolver` ; `geometric_mg` (V-cycle) ; `poisson_fft`(+`_solver`) (spectral) ; `poisson_operator` |
| `integrator/` | `ssprk`, `imex` (AP), `splitting`, `two_fluid_ap`, `amr_multilevel`, `amr_reflux` |
| `coupling/` | `coupler` (`Coupler<Model,Elliptic>`), `coupling_policy`, `amr_coupler` (`AmrCoupler`), `spectral_coupler` (`SpectralCoupler`) |
| `amr/` | `amr_hierarchy`, `cluster` (Berger-Rigoutsos), `regrid`, `tag_box` |
| `parallel/` | `comm` (seam MPI), `load_balance` (Z-order + knapsack) |
| `model/` | `diocotron`, `euler`, `euler_poisson` (PhysicalModel) ; `langmuir`, `two_fluid_isothermal` (noyaux 0D AP) |
| `analysis/` | `diocotron_growth` (Eigen, `#ifdef ADC_HAS_EIGEN`), `hdf5_writer` (`#ifdef ADC_HAS_HDF5`) |
| `solver/` | facades PIMPL : `diocotron_solver`, `euler_poisson_solver`, `two_fluid_ap_solver` |

## 4. Backends : propriete de la bibliotheque, pas un drapeau par cible

OpenMP, MPI, HDF5 et Kokkos sont attaches a la cible d'interface `adc`
(`target_compile_definitions` / `target_link_libraries(adc INTERFACE ...)`). **Tout
ce qui lie `adc` herite du backend** : la facade compilee `src/` (`libadc`), les
tests, les exemples. On configure **une seule fois** :

```
cmake -B build                       # serie
cmake -B build -DADC_USE_OPENMP=ON   # CPU multi-thread (_OPENMP)
cmake -B build -DADC_USE_MPI=ON      # distribue (ADC_HAS_MPI + MPI::MPI_CXX)
cmake -B build -DADC_USE_KOKKOS=ON \ # GPU / CPU portable (ADC_HAS_KOKKOS)
   -DCMAKE_CXX_COMPILER=$K/bin/nvcc_wrapper -DKokkos_ROOT=$K
```

Sous `-DADC_USE_KOKKOS=ON`, la norme retombe a C++20 (nvcc CUDA 12.x) et **`libadc`
elle-meme se compile pour le GPU** (les 3 facades `src/` passent sous nvcc, valide
GH200 bit-identique au CPU). Aucun `ADC_HAS_KOKKOS` rebadge par cible.

Le langage C n'est active (`enable_language` implicite via `project(... C)`) que pour
fabriquer `MPI::MPI_C` exige par le hdf5-config parallele ; `find_package(OpenMP
COMPONENTS CXX)` evite d'exiger un OpenMP C inutile.

## 5. Frontiere bibliotheque / demo

| Couche | Contenu | Lien |
|---|---|---|
| `include/` | coeur generique (concepts, templates, seam GPU). DOIT etre visible a l'instanciation. | header-only `adc::adc` |
| `src/` | facade COMPILEE `libadc` : solveurs concrets non templatises (PIMPL), instancient la pile UNE fois. API stable (apps, pybind11). | `adc::solver` |
| `examples/` (CPU) | pilotes minces. `diocotron`/`diocotron_column` lient `adc::solver` ; `diocotron_amr/mpi/theory` lient `adc::adc` (capacites moteur hors facade). | |
| `examples/gpu/` | demos Kokkos/CUDA (GH200), heritent Kokkos de `adc`. | |
| `tests/` | CTest (+ MPI via `mpirun`). | |
| `python/` | bindings pybind11 de la facade. | |

Regle : besoin du solveur standard -> `adc::solver` ; besoin de toucher AMR/MPI/
champs internes -> `adc::adc`.

## 6. Solveur elliptique : dual MG + FFT

Deux backends, tous deux modelant `EllipticSolver` et resolvant le MEME Laplacien
discret 5 points (memes valeurs propres) :
- **`GeometricMG`** (V-cycle GS rouge-noir) : seul compatible AMR et tout `n`,
  entierement on-device. Le cheval de trait.
- **`PoissonFFTSolver`** / `PoissonFFT` : DIRECT pour le mono-niveau periodique (`n`
  puissance de 2), ~5x quand l'elliptique domine, distribue par bandes
  (`MPI_Alltoall`). Pas de FFT sous AMR.

`Coupler<Model, Elliptic = GeometricMG>` est generique sur le backend.

## 7. AMR : etat (unification MultiFab faite, multi-patch a venir)

**L'integrateur AMR de production tourne desormais sur la pile MultiFab + seam.**
Le chemin (`AmrCoupler` + `diocotron_amr`/`amr3`) utilise :
- `integrator/amr_reflux_mf.hpp` : `amr_step_2level_mf` / `amr_step_multilevel_mf`
  (sous-cyclage Berger-Oliger recursif + reflux), generique
  `<Limiter, NumericalFlux, N-comp>`, bulk via `for_each_cell` (GPU-ready), bati sur
  `operator/spatial_operator.hpp::compute_face_fluxes` (les flux de FACE que le reflux
  exige) ;
- `coupling/amr_coupler.hpp::AmrCoupler<Model, Elliptic=GeometricMG>` : hierarchie
  `std::vector<AmrLevelMF>`, Poisson via le concept `EllipticSolver`, `sync_down` /
  `inject_aux` / aux sur MultiFab.

Chaque brique est prouvee **bit-identique** a la pile Fab2D de reference
(`integrator/amr_reflux.hpp` / `amr_multilevel.hpp`), qui ne sert plus que de
reference testee (`test_amr_reflux`, `test_amr_multilevel`). Tests d'equivalence :
`test_face_fluxes`, `test_amr_reflux_mf`, `test_amr_multilevel_mf` ; conservation
production : `test_amr_coupler` (`5.55e-16`).

Acquis : l'AMR peut utiliser MUSCL / HLL / HLLC / N-composantes et est GPU-ready, la
ou la pile Fab2D etait figee Rusanov 1er ordre scalaire hote-only.

**Reste (le dernier morceau) : le multi-patch.** Chaque niveau est encore une **box
unique** (`AmrLevelMF` = 1 MultiFab a 1 box). Le vrai multi-patch demande : niveaux a
plusieurs boxes, FluxRegister coverage-aware (reflux uniquement aux interfaces fin-
grossier, pas fin-fin), FillPatch inter-patch, regrid **Berger-Rigoutsos** (la Pile A
`amr/cluster`+`regrid`, deja testee mais pas branchee) + `load_balance` SFC sur le
multi-box. Effort distinct et conservation-critique.

## 8. Comparaison AMReX

Correspondances : `MultiFab`, `BoxArray`/`DistributionMapping`, `Geometry`,
`AmrLevel`, FillBoundary, Arena, reflux, MLMG ~ `GeometricMG`. Divergences assumees :
pas de `MFIter` (on itere `for_each_cell` + fab local, GPU-ready) ; `EllipticSolver`
joue `LinOp` mais Laplacien a coefficient constant (EB en escalier) ; AMR a box unique
par niveau (le multi-patch facon AMReX FluxRegister/FillPatch est le morceau restant).

## 9. Validation

- Tests : 37/37 CPU serie ; 37/37 OpenMP ; +7 MPI (`mpirun -np 4`) ; +1 HDF5 ; +1 Eigen.
- GPU : GH200 (CUDA 12.6), advection / MG / pas couple Euler-Poisson / deux-fluides AP
  + `libadc` compilee GPU, tous **bit-identiques au CPU**.
- MPI : advection bit-identique a np=1/2/4/7, halos + FFT distribues.
- Perf : `docs/PERFORMANCE.md` (Poisson 86%, FFT x4.8, OncePerStep x2.6).
