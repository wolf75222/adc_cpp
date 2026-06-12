# Plan technique : backend AMR SAMRAI pour adc_cpp

Date : 2026-06-11

**Statut (2026-06-12).** Plan revu par un panel multi-agent (vérification du code +
> critique + vote). Verdict : **spike-first** -- un gate doit être franchi avant tout codage
> lourd : (1) **faisabilité technique** (verrou device Kokkos `SharedSpace` host==device →
> v1 CPU/MPI-only) ; (2) **preuve de la valeur visée = scaling** (point de croisement chiffré
> natif vs SAMRAI : SAMRAI lève les plafonds coarse-répliqué / `DistributionMapping`
> round-robin / collectives globales ; le natif reste plus capable côté *physique*
> elliptique aniso/Schur). Les 7 décisions ouvertes (§10) sont tranchées à l'unanimité
> (CPU/MPI-first, `distribute_coarse=false`→erreur, interp C/F constante, reflux adc gardé,
> regrid couverture/nesting, restart hors scope). **Conséquence** : l'elliptique distribué
> (HYPRE/FAC scalaire) est *scaling-critique* -- poteau long, pas optionnel ; M8 = palier de
> parité seulement. Suivi : milestone Linear « Backend AMR SAMRAI » -- gate **ADC-126**,
> valeur **ADC-130**, benchmark scaling **ADC-162** ; décision dans
> `SAMRAI_BACKEND_DECISION.md`.

Objectif : brancher SAMRAI comme backend AMR optionnel dans `adc_cpp`, sans casser les API
publiques C++/Python existantes ni les tests de parite du backend natif.

Sources lues :

- Code local `adc_cpp` : `include/adc/amr`, `include/adc/mesh`,
  `include/adc/numerics/time`, `include/adc/coupling`, `include/adc/runtime`,
  `python/`, `tests/`, `docs/ARCHITECTURE.md`,
  `docs/AMR_MULTIBLOCK_DESIGN.md`, `docs/AMR_REGRID_UNION_TAGS_DESIGN.md`,
  `docs/AMR_CONDENSED_SCHUR_DESIGN.md`, `docs/BACKEND_COVERAGE.md`.
- Documentation SAMRAI officielle :
  [GitHub LLNL/SAMRAI](https://github.com/LLNL/SAMRAI),
  [page projet LLNL](https://computing.llnl.gov/projects/samrai),
  [page software LLNL](https://computing.llnl.gov/projects/samrai/software),
  [SAMRAI Concepts](https://computing.llnl.gov/sites/default/files/SAMRAI-Concepts_0.pdf).
- Source SAMRAI verifiee localement : `hier::PatchHierarchy`, `hier::PatchLevel`,
  `hier::Patch`, `hier::VariableDatabase`, `pdat::CellVariable`,
  `pdat::CellData`, `pdat::FaceVariable`, `pdat::SideVariable`,
  `mesh::GriddingAlgorithm`, `mesh::TagAndInitializeStrategy`,
  `mesh::BergerRigoutsos`, `mesh::TreeLoadBalancer`,
  `xfer::RefineAlgorithm`, `xfer::CoarsenAlgorithm`,
  `solv::CellPoissonFACSolver`, `solv::CellPoissonFACOps`,
  `solv::CellPoissonHypreSolver`.

## 1. Etat actuel a preserver

### Entrees C++ publiques

A conserver comme surface de haut niveau :

- `include/adc/runtime/amr_system.hpp`
  - `adc::runtime::AmrSystemConfig`
  - `adc::runtime::AmrSystem`
- `include/adc/runtime/amr_runtime.hpp`
  - moteur multi-blocs runtime, hierarchie partagee, regrid par union des tags.
- `include/adc/coupling/amr_coupler_mp.hpp`
  - chemin mono-bloc historique.
- `include/adc/coupling/amr_system_coupler.hpp`
  - chemin multi-blocs compile-time.
- `include/adc/numerics/time/amr_advance.hpp`
  - `LevelHierarchy`, `advance_amr`.

Ces API doivent continuer a compiler avec le backend natif sans SAMRAI.

### Entrees Python publiques

A conserver sans changement comportemental par defaut :

- `python/bindings.cpp`
  - bindings `AmrSystemConfig`.
- `python/amr_system.cpp`
  - bindings `AmrSystem`.
- `python/adc/__init__.py`
  - facade Python `AmrSystem`, sucre `add_block`, `add_equation`,
    `patch_rectangles`, `write`, `checkpoint`, `restart`.

Methodes Python a garder :

- configuration : `n`, `L`, `periodic`, `regrid_every`, `distribute_coarse`,
  `coarse_max_grid`.
- modeles : `add_block`, `add_native_block`, `add_equation`.
- raffinement : `set_refinement`, `set_phi_refinement`.
- elliptique : `set_poisson`, `set_density`, `set_conservative_state`,
  `potential`.
- couplage : `add_coupled_source`, `set_source_stage`, `set_magnetic_field`,
  `add_dt_bound`, `last_dt_bound`.
- avancement : `step`, `advance`, `step_cfl`, `time`.
- diagnostics : `nx`, `n_blocks`, `block_names`, `n_patches`,
  `patch_boxes`, `mass`, `density`.

Ajout propose : un selecteur optionnel de backend, par exemple
`AmrSystemConfig::amr_backend = "native" | "samrai"`, expose en Python comme
`cfg.amr_backend`. La valeur par defaut reste `"native"`.

## 2. Structures adc_cpp actuelles a remplacer ou adapter

| Besoin | Implementation actuelle | Fichiers actuels |
|---|---|---|
| Niveaux AMR | `AmrHierarchy`, `AmrLevelStack`, `AmrLevelMP` | `include/adc/amr/amr_hierarchy.hpp`, `include/adc/coupling/amr_level_storage.hpp`, `include/adc/numerics/time/amr_subcycling.hpp` |
| Patchs | `Box2D`, `BoxArray`, index local/global de `MultiFab` | `include/adc/mesh/box2d.hpp`, `include/adc/mesh/box_array.hpp`, `include/adc/mesh/multifab.hpp` |
| Hierarchie | vecteurs de niveaux et de `MultiFab`, ref ratio fixe 2 | `amr_hierarchy.hpp`, `amr_runtime.hpp`, `amr_coupler_mp.hpp` |
| Distribution | `DistributionMapping` round-robin/explicite | `include/adc/mesh/distribution_mapping.hpp` |
| Donnees | `MultiFab` -> `Fab2D` contigus, ghosts, `sync_host/device` | `include/adc/mesh/multifab.hpp`, `include/adc/mesh/fab2d.hpp` |
| Fill ghosts intra-niveau | halos MPI faits maison | `include/adc/mesh/fill_boundary.hpp` |
| BC physiques | `BCRec`, `fill_physical_bc`, `fill_ghosts` | `include/adc/mesh/physical_bc.hpp` |
| Interpolation/restriction | `interpolate`, `average_down`, `parallel_copy` | `include/adc/mesh/refinement.hpp` |
| Tags et clustering | `TagBox`, `tag_cells`, `grow_tags`, `berger_rigoutsos` | `include/adc/amr/tag_box.hpp`, `include/adc/amr/regrid.hpp`, `include/adc/amr/cluster.hpp` |
| Regrid production | layout fin impose, union multi-blocs deja implementee | `include/adc/coupling/amr_regrid_coupler.hpp`, `include/adc/runtime/amr_runtime.hpp` |
| Reflux | `FluxRegister`, `CoverageMask`, `CoarseFineInterface` | `include/adc/numerics/time/amr_patch_range.hpp`, `include/adc/numerics/time/amr_reflux*.hpp` |
| Subcycling | Berger-Oliger ratio 2, average-down, reflux | `include/adc/numerics/time/amr_subcycling.hpp` |
| Elliptique | `GeometricMG`, `CompositeFacPoisson` limite, solve coarse + injection par defaut | `include/adc/numerics/elliptic/*`, `include/adc/coupling/amr_coupler_mp.hpp` |
| Physique | modeles locaux, flux, sources, CFL, DSL/native | `include/adc/physics`, `include/adc/numerics`, `include/adc/runtime` |

Conclusion : SAMRAI doit remplacer la gestion de hierarchie, patchs,
transferts AMR et regridding. Les modeles physiques, kernels numeriques,
DSL/Python, sources couplees et diagnostics adc doivent rester au-dessus.

## 3. Ce que SAMRAI apporte

D'apres la documentation LLNL, SAMRAI structure un calcul AMR en niveaux
imbriques, chaque niveau etant couvert par des patchs rectangulaires
logiques avec donnees contigues et cellules fantomes. La page software liste
les packages utiles :

- `hier` : `Patch`, `PatchLevel`, `PatchHierarchy`, `VariableDatabase`,
  calcul de boites et metadonnees distribuees.
- `pdat` : donnees centrees cellule/noeud/face/side, notamment
  `CellData<T>`, `FaceData<T>`, `SideData<T>`.
- `mesh` : construction de hierarchie, regridding, clustering des cellules
  tagguees, load balancing.
- `xfer` : communications intra-niveau, raffinement/coarsening entre niveaux,
  interpolation temps/espace, hooks de BC physiques.
- `algs` : integrateurs de niveau, local time cycling. A ne pas adopter au
  premier jalon pour limiter le risque.
- `solv` : vecteurs de donnees sur hierarchie, interfaces PETSc/Sundials/HYPRE,
  solveur FAC Poisson.
- `geom` : geometrie cartesienne et registres d'operateurs refine/coarsen.

Classes SAMRAI a utiliser directement :

- Hierarchie : `SAMRAI::hier::PatchHierarchy`, `PatchLevel`, `Patch`,
  `Box`, `BoxContainer`, `IntVector`, `Index`.
- Variables : `SAMRAI::hier::VariableDatabase`, `VariableContext`,
  `PatchData`.
- Donnees : `SAMRAI::pdat::CellVariable<double>`,
  `CellData<double>`, `FaceVariable<double>` ou `SideVariable<double>`.
- Geometrie : `SAMRAI::geom::CartesianGridGeometry`.
- Regrid : `SAMRAI::mesh::GriddingAlgorithm`,
  `TagAndInitializeStrategy`, `BergerRigoutsos`, `TreeLoadBalancer`.
- Transferts : `SAMRAI::xfer::RefineAlgorithm`, `RefineSchedule`,
  `RefinePatchStrategy`, `CoarsenAlgorithm`, `CoarsenSchedule`,
  `CoarsenPatchStrategy`.
- Operateurs standard : `CONSTANT_REFINE`, `LINEAR_REFINE`,
  `CONSERVATIVE_LINEAR_REFINE`, `CONSERVATIVE_COARSEN` ou moyenne ponderee
  via `geom::CartesianCellDoubleWeightedAverage`, selon le champ.
- Elliptique : `SAMRAI::solv::CellPoissonFACSolver`,
  `CellPoissonFACOps`, `CellPoissonHypreSolver`,
  `PoissonSpecifications`.

## 4. Architecture cible

Principe : introduire une frontiere backend AMR. Le code applicatif garde la
facade `AmrSystem`; le backend natif reste la reference; SAMRAI devient une
implementation alternative derriere un adaptateur.

```text
Python adc.AmrSystem / C++ adc::runtime::AmrSystem
        |
        v
AmrSystemBackend (nouvelle frontiere)
        |
        +-- NativeAmrBackend
        |     reutilise AmrRuntime, AmrCouplerMP, MultiFab, AmrHierarchy
        |
        +-- SamraiAmrBackend
              possede PatchHierarchy + VariableDatabase + schedules xfer
              expose des vues compatibles avec les kernels adc_cpp
```

Nouveaux composants proposes :

- `include/adc/amr/backend.hpp`
  - enum `AmrBackendKind { Native, Samrai }`.
  - interface minimale commune pour `step`, `advance`, `mass`,
    `density`, `potential`, `patch_boxes`, `n_patches`.
- `include/adc/samrai/samrai_fwd.hpp`
  - inclusions SAMRAI confinees et garde `ADC_USE_SAMRAI`.
- `include/adc/samrai/box_adapter.hpp`
  - conversions `Box2D <-> hier::Box`, `BoxArray <-> BoxContainer`.
- `include/adc/samrai/cell_data_view.hpp`
  - vue `pdat::CellData<double>` vers un acces compatible `Fab2D` /
    `Array4` pour reutiliser les kernels adc.
- `include/adc/samrai/hierarchy_adapter.hpp`
  - proprietaire de `PatchHierarchy`, `CartesianGridGeometry`,
    contexts `CURRENT`, `NEW`, `SCRATCH`, ids de patch data.
- `include/adc/samrai/transfer_adapter.hpp`
  - `RefineAlgorithm`, `CoarsenAlgorithm`, schedule cache, BC physiques.
- `include/adc/samrai/regrid_adapter.hpp`
  - implementation adc de `mesh::TagAndInitializeStrategy`.
- `include/adc/samrai/flux_register_adapter.hpp`
  - phase 1 : reutilise la logique adc sur vues SAMRAI.
  - phase 2 : migration vers flux face/side SAMRAI si rentable.
- `include/adc/samrai/elliptic_adapter.hpp`
  - phase 1 : parite avec le solveur adc actuel.
  - phase 2 : FAC SAMRAI/HYPRE.
- `include/adc/runtime/amr_system_samrai.hpp`
  - backend concret appele par `AmrSystem`.

Responsabilites transferees a SAMRAI :

- creation et ownership de la hierarchie de patchs;
- metadonnees distribuees et voisinages;
- load balancing;
- generation des patchs fins via `GriddingAlgorithm`;
- allocation de donnees patch par variable/context;
- ghost fills et transferts coarse/fine via schedules;
- average-down via `CoarsenAlgorithm`;
- solveur elliptique FAC seulement apres la parite de base.

Responsabilites restant dans adc_cpp :

- API C++/Python;
- definition des modeles physiques, DSL, ABI native;
- kernels de flux, reconstruction, sources, CFL;
- sources couplees multi-blocs;
- choix des criteres de raffinement;
- cadence de regrid exposee a l'utilisateur;
- diagnostics `mass`, `density`, `potential`, `patch_boxes`;
- tests de parite.

## 5. Mapping adc_cpp -> SAMRAI

| adc_cpp actuel | SAMRAI cible | Adaptateur adc_cpp | Note |
|---|---|---|---|
| `Box2D` | `hier::Box`, `hier::Index` | `samrai/box_adapter.hpp` | Indices inclusifs des deux cotes; fixer `tbox::Dimension(2)`. |
| `BoxArray` | boites de `PatchLevel`, `hier::BoxContainer` | `samrai/box_adapter.hpp` | Dans SAMRAI, le layout est produit par `GriddingAlgorithm`; `BoxArray` devient surtout une vue diagnostic. |
| `DistributionMapping` | mapping processeur de `PatchLevel`, `TreeLoadBalancer` | `hierarchy_adapter.hpp` | Ne plus piloter la distribution a la main; exposer seulement le resultat si necessaire. |
| `MultiFab` | `VariableDatabase` + ids de `PatchData` + `CellData<double>` par patch | `samrai_multifab_view.hpp` ou `cell_data_view.hpp` | Adapter uniquement le sous-ensemble utile (`ncomp`, `ngrow`, iteration patch, pointeur). |
| `Fab2D` | `pdat::CellData<double>` | `cell_data_view.hpp` | Verifier layout memoire column-major SAMRAI vs `Fab2D`; encapsuler stride. |
| `AmrHierarchy` | `hier::PatchHierarchy` | `hierarchy_adapter.hpp` | Le backend natif garde `AmrHierarchy`; le backend SAMRAI expose une facade equivalente. |
| `AmrLevelMP` | `{level_number, data_id, aux_id, dx, dy}` | `level_view.hpp` | Ne doit plus posseder `MultiFab`; il reference les patch data SAMRAI. |
| `AmrLevelStack` | `PatchHierarchy` + contexts | `hierarchy_adapter.hpp` | Remplace les vecteurs de niveaux et d'aux. |
| `TagBox` | patch data entier de tags (`CellData<int>`) | `regrid_adapter.hpp` | `tagCellsForRefinement` appelle les predicats adc puis met `tag_index` a 1. |
| `ClusterParams`, `berger_rigoutsos` | `BergerRigoutsos` + input DB `GriddingAlgorithm` | `regrid_adapter.hpp` | Mapper `max_grid`, `grow`, `margin`, proper nesting et efficacite. |
| `fill_boundary` | `xfer::RefineSchedule` intra-niveau | `transfer_adapter.hpp` | Schedules caches et invalides apres regrid. |
| `fill_physical_bc` | `xfer::RefinePatchStrategy` | `transfer_adapter.hpp` | Portage de `BCRec` vers callbacks SAMRAI. |
| `fill_cf_ghost_cell`, `mf_fill_fine_ghosts_*` | `RefineAlgorithm` coarse->fine avec interpolation temps/espace | `transfer_adapter.hpp` | Reproduire d'abord l'interpolation constante adc; ajouter lineaire ensuite. |
| `parallel_copy` | schedule xfer entre data ids/contexts | `transfer_adapter.hpp` | Utile pour regrid et old/new contexts. |
| `average_down` | `CoarsenAlgorithm`, `CoarsenSchedule` | `transfer_adapter.hpp` | Operateur conservatif pour champs conserves. |
| `FluxRegister` | phase 1 : adapter adc; phase 2 : `FaceData`/`SideData` + sync SAMRAI | `flux_register_adapter.hpp` | Risque de changement numerique; garder adc pour la parite initiale. |
| `CoverageMask` | `PatchHierarchy` + regions couvertes par niveaux fins | `coverage_adapter.hpp` | Peut etre derive de la hierarchie SAMRAI; garder tests actuels. |
| `GeometricMG` | phase 1 : solveur adc; phase 2 : `CellPoissonFACSolver` | `elliptic_adapter.hpp` | FAC composite SAMRAI est une decision separee. |
| `CompositeFacPoisson` | `CellPoissonFACSolver`/`CellPoissonFACOps` | `elliptic_adapter.hpp` | Cible long terme pour solve composite multi-niveaux. |
| `AmrRuntime` | orchestration conservee, stockage delegue | `amr_system_samrai.hpp` | Garder union tags et multi-blocs; changer backend data/hierarchy. |
| Modeles physiques | pas de mapping SAMRAI | `cell_data_view.hpp` | Les kernels continuent a lire/ecrire des vues adc. |

## 6. API a preserver et changements acceptables

### Stable

- `adc::runtime::AmrSystemConfig` garde ses champs existants.
- `adc::runtime::AmrSystem` garde ses methodes publiques.
- Python `adc.AmrSystem` garde ses methodes et valeurs de retour.
- `patch_boxes()` retourne encore des `PatchBox`/rectangles dans l'ordre expose.
- `density()` et `potential()` gardent les memes conventions d'evaluation.
- `regrid_every == 0` signifie toujours hierarchie figee.
- Backend natif reste bit-identique et teste.

### Ajouts compatibles

- `AmrSystemConfig::amr_backend`.
- `AmrSystemConfig::poisson_backend`, plus tard :
  `"adc"` par defaut, `"samrai_fac"` opt-in.
- `ADC_USE_SAMRAI` dans CMake; sans cette option, demander `"samrai"` leve une
  erreur explicite.
- Tests de parite qui comparent `"native"` et `"samrai"`.

### Changements internes acceptables

- Les algorithmes AMR deviennent generiques sur une facade de donnees au lieu
  de prendre directement `MultiFab`.
- `DistributionMapping` n'est plus l'objet source de verite en backend SAMRAI.
- `BoxArray` devient une vue/diagnostic en backend SAMRAI.
- Les schedules de communication sont caches et reconstruits dans
  `resetHierarchyConfiguration`.

### Changements a eviter au premier jalon

- Remplacer les kernels physiques par les integrateurs `SAMRAI::algs`.
- Introduire une hierarchie par espece.
- Activer FAC SAMRAI comme solveur par defaut.
- Changer la semantique Python de `add_block`, `set_refinement`, `step`.

## 7. Ordre d'implementation concret

### M0 - Build et dependance SAMRAI

Fichiers :

- `CMakeLists.txt`
- `python/CMakeLists.txt`
- nouveau `cmake/FindSAMRAI.cmake` si `find_package(SAMRAI CONFIG)` n'est pas
  disponible dans l'environnement cible.
- `docs/HPC_SPACK_GUIDE.md`

Travail :

- Ajouter `option(ADC_USE_SAMRAI "Enable SAMRAI AMR backend" OFF)`.
- Lier SAMRAI seulement si l'option est active.
- Verifier les variantes SAMRAI : MPI, HDF5, HYPRE, RAJA, Umpire.
- Documenter une commande Spack/CMake reproductible.

Critere de sortie :

- Build sans SAMRAI inchangé.
- Build avec `ADC_USE_SAMRAI=ON` compile un test vide qui inclut
  `SAMRAI/hier/PatchHierarchy.h`.

### M1 - Selecteur backend sans changement de comportement

Fichiers :

- `include/adc/runtime/amr_system.hpp`
- `python/bindings.cpp`
- `python/amr_system.cpp`
- `python/adc/__init__.py`
- `tests/test_amr_system_contract.cpp`
- `python/tests/test_bindings.py`

Travail :

- Ajouter `amr_backend`, valeur defaut `"native"`.
- Si `"samrai"` est demande sans `ADC_USE_SAMRAI`, lever une erreur claire.
- Brancher `NativeAmrBackend` comme chemin actuel.

Critere de sortie :

- Tous les tests actuels restent verts en backend natif.
- Nouveau test Python : `cfg.amr_backend = "native"` et `"samrai"` sans build
  SAMRAI produisent les comportements attendus.

### M2 - Adaptateurs boites, geometrie et vues de donnees

Fichiers nouveaux :

- `include/adc/samrai/samrai_fwd.hpp`
- `include/adc/samrai/box_adapter.hpp`
- `include/adc/samrai/cell_data_view.hpp`
- `tests/test_samrai_box_adapter.cpp`
- `tests/test_samrai_cell_data_view.cpp`
- `tests/CMakeLists.txt`

Travail :

- Convertir `Box2D` vers `hier::Box` et retour.
- Construire une vue sur `pdat::CellData<double>::getPointer(depth)`.
- Valider les strides, ghosts, bornes inclusives et composantes.

Critere de sortie :

- Tests unitaires serie.
- Comparaison ecriture/lecture cellule par cellule avec `Fab2D`.

### M3 - Hierarchie SAMRAI un niveau

Fichiers :

- `include/adc/samrai/hierarchy_adapter.hpp`
- `include/adc/runtime/amr_system_samrai.hpp`
- `include/adc/runtime/amr_system.hpp`
- `python/amr_system.cpp`

Travail :

- Creer `CartesianGridGeometry`.
- Creer `PatchHierarchy`.
- Enregistrer variables `state`, `aux`, `phi`, `rhs` via
  `VariableDatabase::registerVariableAndContext`.
- Allouer patch data sur le niveau 0.
- Implementer `patch_boxes`, `n_patches`, `density`, `mass` en lecture SAMRAI.

Critere de sortie :

- `AmrSystem` SAMRAI un niveau, sans regrid, sans subcycling.
- Parite avec `test_amr_spatial_parity` ou test minimal equivalent.

### M4 - Fill ghosts intra-niveau et BC physiques

Fichiers :

- `include/adc/samrai/transfer_adapter.hpp`
- `include/adc/samrai/physical_bc_strategy.hpp`
- `tests/test_samrai_fill_boundary.cpp`
- `tests/test_samrai_physical_bc.cpp`

Travail :

- Enregistrer `RefineAlgorithm` pour copie same-level.
- Implementer `RefinePatchStrategy` qui appelle la logique `BCRec`.
- Cacher `RefineSchedule` par niveau/data_id/context.
- Invalider les schedules apres regrid.

Critere de sortie :

- Parite avec `test_fill_boundary`, `test_physical_bc`.
- Variante MPI : parite avec `test_mpi_fillboundary`.

### M5 - Transferts coarse/fine

Fichiers :

- `include/adc/samrai/transfer_adapter.hpp`
- `include/adc/numerics/time/amr_flux_helpers.hpp` si une interface generique
  est necessaire.
- `tests/test_samrai_refinement.cpp`
- `tests/test_samrai_cf_interface.cpp`

Travail :

- Mapper `interpolate` vers `RefineAlgorithm`.
- Mapper `average_down` vers `CoarsenAlgorithm`.
- Reproduire d'abord l'interpolation constante actuelle pour parite.
- Ajouter ensuite `LINEAR_REFINE` / `CONSERVATIVE_LINEAR_REFINE` comme option.

Critere de sortie :

- Parite avec `test_refinement`, `test_cf_interface`.
- Conservation average-down sur champs conserves.

### M6 - Regrid SAMRAI via TagAndInitializeStrategy

Fichiers :

- `include/adc/samrai/regrid_adapter.hpp`
- `include/adc/runtime/amr_system_samrai.hpp`
- `tests/test_samrai_regrid.cpp`
- `tests/test_samrai_multiblock_regrid_union.cpp`

Travail :

- Implementer `SamraiTagAndInitializeStrategy`.
- Dans `tagCellsForRefinement`, appeler les predicats adc par bloc et faire
  l'union des tags.
- Dans `initializeLevelData`, allouer les donnees, interpoler depuis coarse
  et copier l'ancien fine si disponible.
- Dans `resetHierarchyConfiguration`, reconstruire schedules et caches.
- Utiliser `GriddingAlgorithm::makeCoarsestLevel`, `makeFinerLevel`,
  `regridAllFinerLevels`.
- Configurer `BergerRigoutsos` et `TreeLoadBalancer`.

Critere de sortie :

- Parite structurelle avec `test_regrid`.
- Parite multi-blocs avec `test_amr_multiblock_regrid_union`.
- Invariant layout partage verifie apres chaque regrid.

### M7 - Subcycling et reflux

Fichiers :

- `include/adc/samrai/flux_register_adapter.hpp`
- `include/adc/numerics/time/amr_subcycling.hpp`
- `include/adc/numerics/time/amr_patch_range.hpp`
- `include/adc/runtime/amr_system_samrai.hpp`
- `tests/test_samrai_flux_register.cpp`
- `tests/test_samrai_amr_diffusion.cpp`

Travail :

- Phase 1 : garder les registres de flux adc sur vues SAMRAI.
- Adapter `PatchRange`, `CoverageMask`, `CoarseFineInterface`.
- Conserver le ratio 2 et la cadence actuelle.
- Comparer les corrections de reflux contre backend natif.

Critere de sortie :

- Parite avec `test_flux_register`, `test_coverage_mask`,
  `test_patch_range`, `test_amr_diffusion`.
- Conservation de masse sur interfaces coarse/fine.

### M8 - Elliptique, phase parite adc

Fichiers :

- `include/adc/samrai/elliptic_adapter.hpp`
- `include/adc/coupling/amr_coupler_mp.hpp`
- `include/adc/runtime/amr_runtime.hpp`
- `tests/test_samrai_amr_potential.cpp`

Travail :

- Conserver le comportement actuel : solve coarse + injection aux, ou miroir
  natif temporaire pour `GeometricMG`.
- Copier RHS/phi entre patch data SAMRAI et vues/miroirs adc.
- Ne pas activer FAC SAMRAI par defaut.

Critere de sortie :

- Parite avec `test_amr_potential`, `test_poisson_convergence` au niveau
  attendu.
- Tests Python `test_poisson_composite.py`, `test_dsl_elliptic.py` adaptes
  en mode backend parametre.

### M9 - Elliptique, phase FAC SAMRAI/HYPRE

Fichiers :

- `include/adc/samrai/elliptic_adapter.hpp`
- `tests/test_samrai_fac_poisson.cpp`
- `tests/test_samrai_fac_variable_eps.cpp`
- documentation backend.

Travail :

- Ajouter `poisson_backend = "samrai_fac"`.
- Utiliser `CellPoissonHypreSolver`, `CellPoissonFACOps`,
  `CellPoissonFACSolver`.
- Mapper conditions aux limites Poisson vers `SimpleCellRobinBcCoefs` ou
  strategie BC dediee.
- Mapper coefficients `D`, `C`, epsilon scalaire/tensoriel si supporte.

Critere de sortie :

- Manufactured solution 2D.
- Test composite coarse/fine.
- Comparaison avec `CompositeFacPoisson` actuel sur les cas limites deja
  couverts.

### M10 - Python, CI et documentation

Fichiers :

- `python/tests/test_amr_multiblock.py`
- `python/tests/test_dsl_hybrid_amr.py`
- `python/tests/test_dsl_production_amr.py`
- `docs/BACKEND_COVERAGE.md`
- `docs/sphinx/reference/backend_matrix.md`
- `docs/sphinx/amr/index.md`

Travail :

- Parametrer certains tests Python sur `amr_backend`.
- Ajouter colonne SAMRAI dans la matrice backend.
- Documenter limitations : 2D, ratio 2, CPU/MPI initial si Kokkos/SAMRAI
  device memory n'est pas tranche.

Critere de sortie :

- CI native inchangee.
- Job opt-in SAMRAI CPU/MPI.
- Parite documentee.

## 8. Tests de validation

### Tests unitaires nouveaux

- `test_samrai_box_adapter`
- `test_samrai_cell_data_view`
- `test_samrai_hierarchy_adapter`
- `test_samrai_fill_boundary`
- `test_samrai_physical_bc`
- `test_samrai_refinement`
- `test_samrai_cf_interface`
- `test_samrai_regrid`
- `test_samrai_flux_register`
- `test_samrai_fac_poisson`

### Tests adc_cpp existants a rejouer en backend SAMRAI

Maillage et transferts :

- `test_box2d`, `test_box_array` via adaptateurs.
- `test_multifab` via facade `SamraiMultiFabView`.
- `test_fill_boundary`, `test_physical_bc`, `test_refinement`.

AMR primitives :

- `test_amr_hierarchy`
- `test_cluster` comme reference native uniquement, plus comparaison
  structurelle contre `BergerRigoutsos`.
- `test_regrid`
- `test_flux_register`
- `test_coverage_mask`
- `test_patch_range`
- `test_cf_interface`
- `test_load_balance`

Runtime AMR :

- `test_amr_system_coupler`
- `test_amr_source_covered_cells`
- `test_amr_composite_source_conservation`
- `test_amr_stride_cadence`
- `test_amr_layout_guard`
- `test_amr_spatial_parity`
- `test_amr_system_contract`
- `test_amr_system_twoblock`
- `test_amr_multiblock_regrid_union`
- `test_amr_multiblock_compiled`
- `test_amr_regrid_mpi_parity`

Elliptique :

- `test_geometric_mg`
- `test_poisson_convergence`
- `test_amr_potential`
- `test_composite_fac_poisson`
- `test_composite_fac_tensor`
- nouveaux MMS FAC SAMRAI.

MPI :

- `test_mpi_fillboundary`
- `test_mpi_redistribute`
- `test_mpi_coupler_inject`
- `test_mpi_mbox_parity`
- `test_mpi_hybrid_mbox_parity`
- `test_mpi_amr_compiled_parity`
- `test_mpi_amr_twoblock_parity`
- `test_mpi_amr_distributed_coarse`
- `test_mpi_poisson`

Python :

- `python/tests/test_bindings.py`
- `python/tests/test_amr_patch_boxes.py`
- `python/tests/test_amr_multiblock.py`
- `python/tests/test_dsl_hybrid_amr.py`
- `python/tests/test_dsl_production_amr.py`
- `python/tests/test_poisson_composite.py`
- `python/tests/test_dsl_elliptic.py`

### Criteres numeriques

- Backend natif : bit-identique par rapport a l'etat actuel.
- Backend SAMRAI, niveaux/patchs : egalite exacte des boites exposees quand le
  scenario impose le meme layout; sinon invariants structurels et couverture.
- Backend SAMRAI, champs : norme L_inf et L1 relative avec tolerance explicite,
  pas bit-identique obligatoire a cause de l'ordre MPI/schedules.
- Conservation : masse par bloc conservee a tolerance machine sur tests sans
  source; bilan source/reflux explicite sur tests avec source.
- Regrid : aucun bloc ne perd son layout partage; `same_layout_or_throw` ou
  equivalent apres chaque regrid.

## 9. Risques techniques

1. Memoire et execution device
   - adc_cpp s'appuie sur Kokkos; SAMRAI propose RAJA/Umpire en option.
   - Decision a prendre : backend SAMRAI CPU/MPI d'abord, ou integration
     device des le depart.
   - Risque : copies host/device si `pdat::CellData` n'est pas directement
     exploitable par les kernels Kokkos.

2. Layout memoire
   - `CellData` stocke en ordre type Fortran, avec profondeur de composantes.
   - Les vues doivent encapsuler strides et ghosts; aucun kernel ne doit
     supposer le layout `Fab2D` natif.

3. Parite de regrid
   - `BergerRigoutsos` SAMRAI peut produire des boites differentes du
     clustering natif.
   - Les tests doivent separer parite numerique, couverture des tags et
     invariants de nesting.

4. Reflux
   - C'est la zone la plus sensible pour la conservation.
   - Phase 1 doit garder l'arithmetique adc; migration vers `FaceData` /
     `SideData` seulement apres parite.

5. Elliptique composite
   - Le comportement actuel par defaut est surtout coarse solve + injection,
     avec un composite limite.
   - FAC SAMRAI est pertinent mais doit rester un jalon separe, opt-in.

6. Semantique `distribute_coarse`
   - Le backend natif supporte un mode coarse replique.
   - SAMRAI gere naturellement une hierarchie distribuee; il faut decider si
     `distribute_coarse=false` est emule, refuse, ou ignore avec warning en
     backend SAMRAI.

7. Schedules invalides apres regrid
   - `RefineSchedule` et `CoarsenSchedule` ne restent valides que tant que les
     patchs ne changent pas.
   - `resetHierarchyConfiguration` doit tout reconstruire.

8. Multiblocs
   - adc_cpp impose une hierarchie partagee par union des tags.
   - SAMRAI ne doit pas introduire une hierarchie par espece; un seul
     `PatchHierarchy`, plusieurs variables/data ids.

9. Build HPC
   - SAMRAI peut dependre de MPI, HDF5, HYPRE, RAJA, Umpire selon options.
   - Le backend doit rester optionnel pour ne pas alourdir la CI normale.

10. Licence et distribution
    - SAMRAI est distribue sous licence LLNL/LGPL; verifier les implications
      de link statique/dynamique pour les livrables.

## 10. Decisions a prendre avant codage lourd

- Backend SAMRAI v1 CPU/MPI uniquement, ou objectif GPU immediat ?
- `distribute_coarse=false` : emulation, erreur explicite, ou option ignoree ?
- Interpolation coarse/fine v1 : constante pour parite stricte, ou lineaire
  SAMRAI directement avec changement de reference ?
- Flux : conserver `FluxRegister` adc en v1, ou basculer sur `FaceData` /
  `SideData` immediatement ?
- FAC SAMRAI : jalon separe avec `poisson_backend="samrai_fac"` ?
- Regrid tests : exige-t-on les memes boites que le backend natif, ou seulement
  meme couverture/nesting et meme solution a tolerance ?
- Checkpoint/restart : reste hors scope, ou mapping vers HDF5/SAMRAI restart ?

## 11. Synthese

Le chemin le moins risque est :

1. Ajouter un selecteur de backend sans changer le backend natif.
2. Construire une couche d'adaptation SAMRAI qui expose des vues compatibles
   avec les kernels adc_cpp.
3. Porter d'abord hierarchy/data/ghosts/transfers/regrid.
4. Garder reflux et elliptique adc pour la premiere parite.
5. Activer FAC SAMRAI/HYPRE seulement apres validation du backend AMR de base.

Ainsi, SAMRAI prend les responsabilites pour lesquelles il est fait
(`PatchHierarchy`, patch data, schedules, gridding, load balancing), tandis que
adc_cpp conserve sa valeur principale : modeles physiques, DSL/Python,
couplage multi-blocs, kernels numeriques et contrats publics.
