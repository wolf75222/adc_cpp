# Audit de maintenabilite de `adc_cpp`

Date : 2026-06-06.  
Base relue : `origin/master` / `9ba36f5` apres les PR #118-#142, #135 et #141.
PR ouverte impactant l'audit : #140 (cadence stride AMR).
Perimetre lu : `include/adc/**/*.hpp`, `python/*.cpp`, `python/adc/*.py`, docs d'architecture
principales. Les tests GPU sous `python/tests/gpu/` sont classes comme harnais de validation, pas
comme API de production.

Objectif : verifier que chaque classe/fichier a une raison d'etre claire, que les responsabilites
ne se melangent pas, et identifier les morceaux a factoriser, archiver ou renforcer. Ce document
n'est pas une roadmap scientifique : c'est un audit de code.

Convention de commentaires associee : voir
[`CODE_DOCUMENTATION_CONVENTION.md`](CODE_DOCUMENTATION_CONVENTION.md). Chaque fichier doit porter
une entete de responsabilite, chaque classe non triviale doit exposer son contrat et ses contraintes,
et les commentaires internes doivent expliquer les invariants numeriques/MPI/GPU plutot que
paraphraser les lignes.

## 1. Lecture globale

La bibliotheque a maintenant une architecture lisible :

```text
PhysicalModel local
  -> operateurs numeriques
  -> donnees maillage/MultiFab
  -> backends execution/MPI/Kokkos
  -> runtime System/AmrSystem + bindings Python
```

Le sens general est bon : `adc_cpp` ne contient plus de solveur applicatif nomme `DiocotronSolver`
dans l'API publique ; les cas vivent dans `adc_cases`. Le coeur fournit des briques generiques, un
DSL, des chemins natifs compiles, MPI et Kokkos.
Nuance : quelques commentaires de headers generiques citent encore "diocotron" pour expliquer
l'origine d'un chantier polaire/Schur. Ce n'est pas un symbole applicatif public, mais le vocabulaire
devrait etre neutralise progressivement en "anneau polaire", "derive ExB" ou "modele test".

Le risque principal n'est plus "mauvaise abstraction de depart". Le risque est l'accumulation de
plusieurs generations de code qui restent toutes presentes :

- `System` runtime Python moderne.
- `AmrSystem` runtime Python, volontairement plus limite.
- anciens coupleurs header-only (`Coupler`, `SystemCoupler`, `AmrCoupler`, `AmrCouplerMP`).
- chemins DSL multiples : `dynamic`, `aot`, `production/native`.
- moteurs AMR historiques et multipatch dans un meme gros header.

Il faut donc conserver les briques bas niveau, mais clarifier quelles classes sont l'API actuelle,
quelles classes sont des moteurs internes, et quelles classes sont de compatibilite.

Etat recent integre dans cet audit :

- Schur est maintenant une pile C++ complete cote uniforme : operateur tensoriel plein, brique
  `LorentzEliminator`, builder de condensation, solveur Krylov BiCGStab, etage
  `CondensedSchurSourceStepper`, binding Python `adc.Split` / `adc.CondensedSchur`.
- `System` a gagne stride par bloc, `SourceImplicit`, masque `implicit_vars` / `implicit_roles`,
  set/get en variables primitives, source couplee DSL explicite, et une API qui rejette plusieurs
  chemins trompeurs au lieu de les ignorer silencieusement.
- La geometrie polaire a une Phase 1 transport + un solveur de Poisson polaire direct autonome
  (FFT-theta + tridiag-r). Le branchement complet dans `System.step` reste a faire.
- `AmrSystem` a progresse sur WENO5/HLLC/Roe/reconstruction primitive, IMEX local source-only et
  production DSL mono-bloc. Le capstone multi-blocs AMR est documente mais pas encore runtime.
- La CI a ete reorganisee (#136) avec cache Kokkos/ccache/ninja, auto-decouverte Python et split
  fast/full. L'audit doit donc distinguer "test couvert en CI" et "harness GH200 manuel".
- #135 est maintenant merge sur `origin/master` : `Geometry` / `Box2D` sont device-callable et la
  CFL passe par une reduction MPI globale. C'est un correctif de validite device/MPI, pas une raison
  de supprimer les harnais GH200 Schur/polaire.
- #141 est maintenant merge sur `origin/master` : `AmrHierarchyLayout` et `same_layout_or_throw`
  existent comme premier garde-fou de layout AMR partage.

PR ouvertes a ne pas compter comme "fait" tant qu'elles ne sont pas mergees :

- #140 corrige la cadence AMR `stride` en hold-then-catch-up. Tant que #140 n'est pas merge,
  `AmrSystemCoupler` garde la condition `macro_step_ % stride`, qui avance un bloc lent au premier
  macro-pas.

## 2. Invariants d'architecture a conserver

Ces invariants sont les "rites" du code : si un fichier les casse, il devient suspect.

| Invariant | Sens |
|---|---|
| Un `PhysicalModel` est local | Pas de `MultiFab`, pas de MPI, pas d'AMR, pas de stockage global. |
| La geometrie appartient au mesh | `PolarMesh` / `CartesianMesh`, pas `FiniteVolume(geometry=...)`. |
| Le schema spatial ne connait pas le scenario | Il compose reconstruction, Riemann, variables. |
| Le runtime Python compose | Python choisit les briques ; les boucles cellules restent C++. |
| Les chemins GPU utilisent des foncteurs nommes | Eviter les lambdas device cross-TU fragiles sous nvcc. |
| `System` est multi-bloc | Especes, Poisson global, sources couplees, stride/substeps. |
| `AmrSystem` est mono-bloc pour l'instant | Sa surface publique reste plus etroite que celle de `System` tant que le multi-bloc AMR conservatif n'est pas implemente. |
| Futur AMR multi-blocs conservatif | Hierarchie commune, cellules co-localisees pour tous les blocs evolues ; pas d'espece absente localement par patch. |
| Les noms applicatifs vivent hors coeur | `diocotron`, `two_fluid_ap`, validations, presets : `adc_cases`. |
| Backend precise dans les rapports | Dire `MPI CPU`, `Kokkos Cuda`, `MPI + Kokkos Cuda`, pas seulement "MPI". |
| Schur global separe des sources locales | `SourceImplicit` = source locale implicite ; Schur = etage source condense non local. |

## 3. Carte des responsabilites par module

| Module | Responsabilite correcte | Probleme possible |
|---|---|---|
| `core/` | Concepts, etats, variables, blocs abstraits. | Stable. Independant de la numerique lourde. |
| `physics/` | Briques physiques locales et composables. | Garder generique ; pas de scenario nomme. |
| `numerics/` | Reconstruction, flux, RHS FV, temps, elliptique, Schur primitives. | Certains headers sont trop gros ou trop couples. |
| `mesh/` | Box, MultiFab, halos, CL, execution cellulaire. | Bien separe, mais `fill_boundary`/`physical_bc` doivent rester bas niveau. |
| `amr/` | Tagging, clustering, regrid abstrait. | Correct, petit. |
| `coupling/` | Moteurs de couplage header-only historiques et AMR, sources couplees, Schur. | Zone la plus redondante. A classer public/interne/deprecated. |
| `runtime/` | Facades C++ utilisees par Python, ABI DSL, loaders. | Sens fort, mais beaucoup de chemins. Besoin de nommage strict. |
| `python/*.cpp` | Bindings et implementation runtime hors ligne. | `python/system.cpp` est trop gros. |
| `python/adc/*.py` | API utilisateur Python et DSL. | Certaines docstrings restent anciennes. |

## 4. Audit par fichier

### `include/adc/core`

| Fichier | Classes / objets | Sens | Frontiere | Audit |
|---|---|---|---|---|
| `types.hpp` | `Real`, macros host/device | Base minimale partagee. | Importer Kokkos lourdement. | OK. |
| `state.hpp` | `StateVec<N>`, `Aux`, champs aux canoniques | Etat local device-callable. | Devenir un conteneur grille. | OK, mais l'ajout d'aux exige synchronisation avec `dsl.py`. |
| `variables.hpp` | `VariableRole`, `VariableSet`, metadata ABI | Donner du sens aux composantes. | Piloter directement le calcul numerique. | Tres utile pour sources couplees et Schur. |
| `physical_model.hpp` | concepts `PhysicalModel`, `HyperbolicPhysicalModel`, `aux_comps` | Contrat local du modele. | Porter un solveur, un schema ou un mesh. | OK. C'est le noyau conceptuel. |
| `equation_block.hpp` | `EquationBlock` | Associe modele + etat + schema + temps. | Resoudre ou scheduler. | Bon niveau d'abstraction, surtout cote C++ compile. |
| `coupled_system.hpp` | `CoupledSystem`, `CoupledSystemLike` | Groupe heterogene de blocs. | Imposer une physique ou un ordre de temps. | OK. Garder simple. |
| `allocator.hpp` | `ManagedArena`, `ManagedAllocator` | Abstraction memoire unifiee/device. | Cacher les fences dans l'accesseur. | Utile mais a surveiller avec Kokkos/CUDA. |
| `kokkos_env.hpp` | init/finalize Kokkos | Gestion environnement backend. | Porter de la numerique. | OK. |

### `include/adc/physics`

| Fichier | Classes / objets | Sens | Frontiere | Audit |
|---|---|---|---|---|
| `hyperbolic.hpp` | `ExBVelocity`, `ExBVelocityPolar`, `IsothermalFlux`, `CompressibleFlux` | Vars + flux + vitesses. | Se melanger avec source ou Poisson. | Bon. `ExBVelocityPolar` est un choix physique/local compatible mesh polaire. |
| `euler.hpp` | `Euler` | Flux compressible complet. | Devenir un solveur Euler-Poisson. | OK. |
| `source.hpp` | `NoSource`, `PotentialForce`, `GravityForce` | Sources locales par cellule. | Representer Schur/global implicite. | OK. Sources couplees doivent rester hors de ce fichier. |
| `elliptic.hpp` | `ChargeDensity`, `BackgroundDensity`, `GravityCoupling` | RHS elliptique local par bloc. | Modifier l'operateur elliptique. | OK. Important pour clarifier Schur : RHS seulement. |
| `composite.hpp` | `CompositeModel<H,S,E>` | Compose transport/source/elliptic. | Autoriser combinaisons incoherentes Vars/Flux. | Bon grain d'abstraction. |
| `advection_diffusion.hpp` | `AdvectionDiffusion` | Modele test / transport-diffusion. | Devenir API utilisateur principale. | A classer comme exemple/validation si peu utilise. |
| `langmuir.hpp` | `LangmuirMode` | Modele analytique/test plasma. | Polluer API publique si scenario trop specifique. | Potentiellement "physics test brick"; documenter usage. |
| `two_fluid_isothermal.hpp` | `TwoFluidLinear` | Brique lineaire specifique deux-fluides. | Redevenir `TwoFluidAPSolver`. | A verifier : si seulement test, deplacer/documenter comme validation. |
| `bricks.hpp` | include agregateur | Convenience header. | Porter logique. | OK. |

### `include/adc/numerics`

| Fichier | Classes / objets | Sens | Frontiere | Audit |
|---|---|---|---|---|
| `reconstruction.hpp` | `NoSlope`, `Minmod`, `VanLeer`, `Weno5` | Reconstruction point-wise. | Boucler sur les grilles. | OK. |
| `numerical_flux.hpp` | `RusanovFlux`, `HLLFlux`, `HLLCFlux`, `RoeFlux` | Flux de Riemann local. | Connaitre `MultiFab`. | OK. |
| `spatial_discretisation.hpp` | `SpatialDiscretisation` aliases | Nommer limiter + Riemann. | Porter le temps. | OK. |
| `spatial_operator.hpp` | `assemble_rhs`, kernels face/RHS, `load_aux` | Operateur FV cartesien. | Choisir le modele ou l'API Python. | Correct, mais gros et central. Garder comme moteur bas niveau. |
| `spatial_operator_polar.hpp` | `assemble_rhs_polar`, kernels polaires | Divergence FV polaire. | Etre branche silencieusement dans `System`. | OK comme Phase 1. Il faut un chemin runtime explicite avant usage utilisateur. |
| `lorentz_eliminator.hpp` | `LorentzEliminator` | Brique locale pour Schur : `B^{-1}` analytique. | Resoudre l'elliptique. | Bon : petite classe avec responsabilite nette. |

### `include/adc/numerics/elliptic`

| Fichier | Classes / objets | Sens | Frontiere | Audit |
|---|---|---|---|---|
| `elliptic_solver.hpp` | concept `EllipticSolver` | Contrat solveur minimal. | Decrire un probleme complet. | OK. |
| `elliptic_problem.hpp` | `EllipticProblem`, `FieldPostProcess` | Description probleme + postprocess phi/grad. | Remplacer `System::set_poisson`. | Bon, mais encore peu central dans runtime. |
| `poisson_operator.hpp` | kernels laplacien, residual, smoother, BC conducteur | Operateur elliptique discret. | Porter politique de solveur complete. | Gros mais coherent. Avec Schur, extraire interface `OperatorSpec` plus formelle. |
| `geometric_mg.hpp` | `GeometricMG` | Solveur MG + niveaux + coefficients. | Porter toute la famille elliptique future. | Fonctionne, mais melange solver, stockage niveaux, coefficients. A factoriser avant GMRES/Schur. |
| `krylov_solver.hpp` | `TensorKrylovSolver`, `KrylovResult` | Solveur BiCGStab matrice-libre pour operateur tensoriel non symetrique. | Remplacer MG partout. | C++ uniforme OK ; garder validation GPU explicite apres le correctif #135. |
| `polar_poisson_solver.hpp` | `PolarPoissonSolver` | Poisson direct sur anneau polaire : FFT en theta + tridiag en r. | Se presenter comme solveur distribue ou runtime `System` complet. | Bon choix numerique. Mono-rang/box unique ; Phase 2b runtime a faire. |
| `poisson_fft.hpp` | `PoissonFFT` direct | Solveur FFT mono-rang bas niveau. | Pretendre etre MPI `System`. | OK si garde mono-rang claire. |
| `poisson_fft_solver.hpp` | `PoissonFFTSolver`, `DistributedFFTSolver` | Wrappers FFT pour `MultiFab`. | Cacher une incompatibilite de layout. | Garde MPI propre ajoutee. `DistributedFFTSolver` non route dans `System`. |

### `include/adc/numerics/time`

| Fichier | Classes / objets | Sens | Frontiere | Audit |
|---|---|---|---|---|
| `time_integrator.hpp` | tags `SSPRK2`, `SSPRK3`, `TimePolicy`, `TimeTreatment` | Decrire le choix temporel compile. | Executer directement le pas. | OK. |
| `time_steppers.hpp` | `ForwardEuler`, `SSPRK2Step`, `SSPRK3Step` | Integrateurs `take_step(rhs,U,dt)`. | Connaitre Poisson. | OK, bon seam. |
| `scheduler.hpp` | `advance_subcycled`, stride compile | Scheduler C++ par bloc. | Cacher des effets de temps dans le modele. | OK. Runtime Python a sa propre cadence. |
| `ssprk.hpp` | ancien helper SSPRK2 | Convenience/legacy. | Dupliquer le driver moderne. | Potentiellement a archiver si plus utilise. |
| `imex.hpp` | `imex_euler_step` | Petite brique splitting. | Remplacer vraie politique IMEX generale. | OK mais limite. |
| `implicit_stepper.hpp` | Newton source local, `ImplicitSourceStepper` | Source locale implicite. | Vendre le Schur global comme source locale. | Utile mais doit etre renomme/documente "local implicit". |
| `splitting.hpp` | Lie/Strang | Composition d'operateurs. | Connaitre la physique. | OK. |
| `amr_multilevel.hpp` | ancien AMR multi-level | Pas AMR historique simple. | Etre chemin principal si `amr_reflux_mf` le remplace. | A classer legacy ou validation. |
| `amr_reflux.hpp` | AMR Fab2D 2 niveaux | Reflux pedagogique/bas niveau. | Coexister comme API principale. | Probable legacy. |
| `amr_reflux_mf.hpp` | `AmrLevelMF`, `AmrLevelMP`, `PatchRange`, `FluxRegister`, `CoverageMask`, `SubcyclingSchedule`, `CoarseFineInterface`, `advance_amr`, source IMEX locale | Moteur AMR MultiFab multipatch. | Rester un fichier fourre-tout de 1000 lignes. | Plus gros point de factorisation. Gap2 IMEX local merge, mais fichier a decouper. |

### `include/adc/mesh`

| Fichier | Classes / objets | Sens | Frontiere | Audit |
|---|---|---|---|---|
| `box2d.hpp` | `Box2D` | Domaine discret rectangulaire. | Connaitre MPI. | OK. |
| `box_array.hpp` | `BoxArray` | Liste de boxes. | Stocker les donnees. | OK. |
| `distribution_mapping.hpp` | `DistributionMapping` | Mapping boxes -> ranks. | Faire load balancing complexe. | OK. |
| `fab2d.hpp` | `Fab2D`, `Array4`, `ConstArray4` | Tableau local + vues device. | Connaitre AMR global. | OK. |
| `multifab.hpp` | `MultiFab` | Collection distribuee de `Fab2D`. | Porter les schemas numeriques. | OK. |
| `fill_boundary.hpp` | halo local/MPI, periodicite | Echange ghosts. | Appliquer CL physiques complexes. | OK, device-sensitive. |
| `physical_bc.hpp` | `BCRec`, `fill_physical_bc` | CL physiques domaine. | Echanger MPI. | OK. |
| `mf_arith.hpp` | saxpy, lincomb, norm, sum | Operations grille. | Porter logique de solver. | OK. |
| `for_each.hpp` | seams execution/reductions/fence | Boucles backend. | Contenir logique numerique. | OK. |
| `geometry.hpp` | `Geometry`, `PolarGeometry` | Metrique/domaines. | Etre choisi dans `FiniteVolume`. | OK. |
| `refinement.hpp` | refinement/coarsen helpers | Indices coarse/fine. | Orchestrer un pas AMR complet. | OK, mais verifier lambdas device restantes. |
| `box_hash.hpp` | `BoxHash` | Acceleration lookup boxes. | Porter logique AMR. | OK. |

### `include/adc/amr`

| Fichier | Classes / objets | Sens | Frontiere | Audit |
|---|---|---|---|---|
| `tag_box.hpp` | `TagBox` | Masque de cellules taguees. | Construire la hierarchie. | OK. |
| `cluster.hpp` | clustering Berger-Rigoutsos | Boxes a partir de tags. | Avancer en temps. | OK. |
| `regrid.hpp` | `RegridParams`, `regrid_level` | Regridding abstrait. | Gerer Poisson/couplage. | OK. |
| `amr_hierarchy.hpp` | `AmrHierarchy` | Hierarchie de niveaux. | Duplicater `AmrSystem`. | OK mais a relier clairement au moteur principal. |

### `include/adc/coupling`

| Fichier | Classes / objets | Sens | Frontiere | Audit |
|---|---|---|---|---|
| `coupler.hpp` | `Coupler<Model>` | Coupleur mono-bloc historique Poisson + transport. | Etre l'API Python moderne. | Utile en tests/C++ direct, mais API secondaire. |
| `system_coupler.hpp` | `SystemAssembler`, `SystemDriver`, alias `SystemCoupler` | Couplage multi-blocs compile. | Refaire le runtime Python. | Bonne abstraction tuteur, mais coexiste avec `runtime/System`. |
| `amr_system_coupler.hpp` | `AmrSystemCoupler`, `AmrSystemDriver` | CoupledSystem sur AMR compile. | Pretendre couvrir runtime `AmrSystem`. | Utile mais surface a clarifier. |
| ~~`amr_coupler.hpp`~~ | ~~`AmrCoupler`~~ | AMR mono-box RETIRE (C.2, juin 2026). | Remplace par `AmrCouplerMP`. | Aucun client #include ; fichier supprime. |
| `amr_coupler_mp.hpp` | `AmrCouplerMP` + helpers | Moteur AMR multipatch couple. | Porter toute la logique runtime Python. | Important, mais gros. Extraire helpers layout/read/write/inject. |
| `amr_regrid_coupler.hpp` | regrid AMR coupler | Rebuild niveau fin mono-bloc. | Faire avance en temps. | OK. Pour multi-blocs, le regrid par union des tags reste a ecrire. |
| `amr_level_storage.hpp` | `AmrLevelStack` | Stockage niveaux + aux. | Porter equations. | OK. |
| `amr_diagnostics.hpp` | masse, vitesse AMR | Diagnostics. | Modifier etat. | OK. |
| `elliptic_rhs.hpp` | RHS mono/multi-especes | Assemblage RHS depuis blocs. | Resoudre Poisson. | OK. |
| `coupled_source.hpp` | `NoCoupledSource`, concept source couplee | Source lisant plusieurs blocs. | Remplacer couplages runtime Python. | Concept utile, maintenant complete par le bytecode DSL. |
| `coupled_source_program.hpp` | `CoupledSourceKernel`, bytecode source couplee | Source couplee arbitraire DSL, evaluee en C++/device sans callback Python. | Devenir un solveur implicite. | Bon point P5. Test conservation et MPI a renforcer sur AMR composite. |
| `coupling_policy.hpp` | tags cadence Poisson | Policy de couplage. | Porter implementation. | OK. |
| `aux_fill.hpp` | derive aux, fill Bz | Helpers de champs auxiliaires. | Devenir un solveur. | OK. |
| `schur_condensation.hpp` | builder operateur/RHS Schur | Assemble coefficients de l'operateur condense. | Resoudre ou avancer le temps. | Bon decoupage : builder seulement. |
| `condensed_schur_source_stepper.hpp` | `CondensedSchurSourceStepper` | Etage source condense par Schur. | Etre une source locale `model.source`. | C++ uniforme merge ; validation GPU a tracer explicitement apres #135. |
| `spectral_coupler.hpp` | coupleur FFT mono-modele | Variante spectrale. | Etre route MPI sans garde. | Probablement API secondaire. |

### `include/adc/runtime`

| Fichier | Classes / objets | Sens | Frontiere | Audit |
|---|---|---|---|---|
| `system.hpp` | `SystemConfig`, `System` | Facade C++ runtime multi-blocs exposee a Python. | Contenir toute l'implementation. | Interface riche : stride, primitives, source DSL, Schur. Implementation trop lourde dans `python/system.cpp`. |
| `amr_system.hpp` | `AmrSystemConfig`, `AmrSystem`, hooks | Facade runtime AMR mono-bloc. | Promettre multi-bloc runtime. | Mono-bloc mais plus riche : WENO5/HLLC/Roe/recon primitive, production DSL, IMEX local. Multi-bloc capstone non merge. |
| `model_spec.hpp` | `ModelSpec` | Tags de briques natives depuis Python. | Devenir scenario nomme. | OK. |
| `model_factory.hpp` | dispatch `ModelSpec` -> types C++ | Pont tags runtime vers templates. | Grossir sans fin. | OK court terme. A surveiller si briques nombreuses. |
| `block_builder.hpp` | construit closures compilees `advance/rhs` | Unifie `add_block` et native DSL. | Connaitre Python. | Tres important, bon seam. |
| `grid_context.hpp` | `GridContext`, `BlockClosures` | Donnees minimales pour closures. | Ne devient pas proprietaire du mesh ni des auxiliaires. | OK. |
| `dsl_block.hpp` | `add_compiled_model(System&)` | Chemin production zero-copy pour DSL/C++ modele. | Refaire ABI plate. | Bon. |
| `amr_dsl_block.hpp` | `add_compiled_model(AmrSystem&)` | Chemin production AMR. | Cacher les limites AMR. | OK mais mono-bloc explicite ; multi-bloc nomme = futur capstone. |
| `compiled_block_abi.hpp` | ABI plate `.so` AOT | Prototypage runtime sans ABI C++ partagee. | Etre presente comme production principale. | A garder avance/debug. |
| `dynamic_model.hpp` | `IModel`, `ModelAdapter` | Prototype host virtuel. | GPU/MPI hot path. | A marquer clairement experimental. |
| `abi_key.hpp` | cle ABI headers/build | Evite UB loader natif. | Porter logique runtime. | OK. |
| `export.hpp` | macro export symboles | Frontiere loader. | Logique metier. | OK. |
| `wall_predicate.hpp` | predicat paroi commune | Geometrie simple conducteur. | Confondre avec bord d'anneau mobile. | OK. |

### `python/*.cpp`

| Fichier | Classes / objets | Sens | Frontiere | Audit |
|---|---|---|---|---|
| `bindings.cpp` | module pybind11 `_adc` | Exposer `System`, `AmrSystem`, configs. | Implementer la simulation. | OK. |
| `amr_system.cpp` | `AmrSystem::Impl` | Runtime AMR hors ligne. | Devenir multi-bloc sans registre explicite. | Taille raisonnable. Refuse encore le 2e bloc ; capstone doit ajouter un registre type-erased, pas juste retirer le throw. |
| `system.cpp` | `System::Impl`, loaders, Poisson, couplings, I/O, stride, Schur stage | Runtime principal multi-blocs. | Rester une classe dieu. | Plus gros risque de maintenabilite cote runtime ; les ajouts recents renforcent le besoin d'extraction. |

### `python/adc/*.py`

| Fichier | Sens | Audit |
|---|---|---|
| `__init__.py` | API utilisateur Python : briques, `System`, `AmrSystem`, `FiniteVolume`, temps, couplages, Schur. | Bon point d'entree. Les rejets explicites #137 vont dans le bon sens. Verifier que chaque limite AMR/polaire reste claire. |
| `dsl.py` | DSL symbolique, codegen, cache, backends `prototype/aot/production`, couplages DSL. | Fonctionnel mais trop gros. Le docstring de tete dit encore "on ne genere PAS encore de code compile", contredit le statut actuel. A corriger en priorite. |
| `integrate.py` | Integrateurs Python par pas. | Utile pour prototypage, hors hot path. |
| `elliptic.py` | Facade elliptique Python. | Garder comme declaration, pas solveur. |

### `python/tests/gpu`

Ces fichiers sont des harnais, pas des briques API. Leur sens est de figer des invariants device/MPI :

| Fichier | Ce qu'il prouve |
|---|---|
| `gpu_dsl_production_validate.cpp` | chemin DSL production sur GPU. |
| `gpu_epm_validate.cpp` | operateurs elliptiques avances. |
| `gpu_amrsys_facade_validate.cpp` | facade AMR sous nvcc. |
| `gpu_amr_bz_validate.cpp`, `gpu_amr_bz_mpi_validate.cpp` | AMR + Bz + device/MPI. |
| `phase1_transport.cpp` a `phase7_system.cpp` | progression historique du port GPU. |
| `mpi6_fillboundary.cpp` | halos MPI/device. |
| `amrmpi_integrated.cpp` | validation integree AMR MPI. |
| `diff_bin.cpp` | comparaison binaire. |

Audit : les garder, mais separer clairement `tests/gpu/` de l'API. Ne pas en faire une dependance de
runtime.

## 5. Classes/fichiers a factoriser en priorite

### P0 - Ne pas casser, mais clarifier

1. **`python/system.cpp`**
   - Probleme : il porte trop de responsabilites : registry de blocs, allocation, Poisson,
     couplings, native loaders, I/O state, stride, diagnostics, champs auxiliaires.
   - Refactor cible :
     - `SystemBlocks` : ajout/recherche blocs, ghosts, state I/O.
     - `SystemFields` : Poisson, aux, epsilon/kappa/Bz/Te.
     - `SystemCouplingsRuntime` : ionisation/collision/thermal exchange.
     - `NativeLoader` : dlopen, ABI, symboles.
     - `SystemStepper` : `step`, `step_cfl`, stride/substeps.
   - Gain : rendre chaque future feature localisee.

2. **`include/adc/numerics/time/amr_reflux_mf.hpp`**
   - Probleme : 1016 lignes, contient plusieurs niveaux d'abstraction.
   - Decoupage cible :
     - `amr_level.hpp` : `AmrLevelMF`, `AmrLevelMP`, `LevelHierarchy`.
     - `amr_patch_range.hpp` : `PatchRange`, interfaces coarse/fine.
     - `amr_flux_register.hpp` : flux/reflux.
     - `amr_subcycling.hpp` : schedules, reflux et `advance_amr`.
     - `amr_source_step.hpp` : etages source IMEX locaux sur hierarchie.
   - Gain : AMR lisible, plus facile a porter GPU/MPI.

3. **Famille `coupling/`**
   - `amr_coupler.hpp` (AmrCoupler mono-box) est RETIRE (C.2, juin 2026) : aucun client #include detecte, remplace par AmrCouplerMP.
   - `Coupler` et `SpectralCoupler` sont des facades C++ directes utiles, mais secondaires.
   - `SystemCoupler` est conceptuellement propre, mais coexiste avec `runtime/System`.
   - Action : ajouter une page "public vs internal vs deprecated".

### P1 - Renforcer les contrats

4. **DSL et docstrings**
   - `python/adc/dsl.py` commence par une description obsolete de prototype CPU sans codegen compile.
   - Action : aligner l'en-tete avec l'etat actuel : `prototype`, `aot`, `production`, cache, GPU/MPI.

5. **Temps implicite : source locale vs Schur global**
   - `SourceImplicit` est maintenant le bon nom pour la source locale implicite.
   - `adc.Split` / `adc.CondensedSchur` introduisent le vrai etage non local par Schur.
   - Action : garder cette separation dans tous les exemples : ne pas presenter
     `SourceImplicit` comme "implicite total", et ne pas cacher Schur derriere une source locale.

6. **`AmrSystem`**
   - Limites honnetes : runtime encore mono-bloc, meme si le moteur compile-time
     `AmrSystemCoupler` sait deja representer plusieurs blocs co-localises.
   - Action : faire evoluer `AmrSystem` vers un registre multi-blocs explicite, pas seulement
     retirer les `throw` sur le deuxieme bloc.
   - Cadrage : le multi-bloc AMR conservatif doit partager une meme hierarchie et des cellules
     co-localisees pour tous les blocs evolues ; la souplesse est modele/schema/temps, pas
     l'absence spatiale locale d'une espece sur certains patches.
   - Prealable ouvert : #140 pour corriger la cadence `stride` de `AmrSystemCoupler`.

### P2 - Stabiliser Schur et elliptique

7. **Elliptique**
   - `GeometricMG` fait encore solver + stockage + coefficients pour Poisson scalaire.
   - Schur a maintenant les briques C++ : operateur tensoriel plein, Krylov BiCGStab, builder,
     etage condense et binding Python.
   - Action : formaliser apres coup les interfaces communes :
     - `EllipticOperator` : apply/residual/smooth/coefficients.
     - `LinearSolver` : MG, FFT, Krylov.
     - `FieldPostProcess` : phi -> aux.
   - Backend : #135 leve le bug device connu (`Geometry` / `Box2D` + CFL MPI). Continuer a exiger
     un harnais GH200 pour declarer un chemin Schur/polaire device-clean.

8. **Roles variables**
   - `VariableRole` est une bonne base.
   - Action : imposer roles sur les modeles DSL utilises par sources couplees/Schur, lever erreur si
     `Density/MomentumX/MomentumY` manquent.

### P3 - Nettoyage API utilisateur

9. **Chemins DSL**
   - `dynamic`, `aot`, `production` ont tous un sens.
   - Action : dans docs et API, recommander :
     - `production` par defaut.
     - `aot` debug ABI plate.
     - `prototype/dynamic` test CPU hote seulement.

10. **Geometrie polaire**
    - `PolarMesh`, transport polaire et `PolarPoissonSolver` existent.
    - `System.step` avec `mesh=PolarMesh` et le couplage runtime complet restent a brancher.
    - Action : conserver l'erreur explicite tant que Phase 2b n'est pas faite. Ne pas promettre
      reproduction papier avant un run annulaire complet.

## 6. Ce qui semble inutile ou trop specifique

| Element | Diagnostic | Recommandation |
|---|---|---|
| ~~`amr_coupler.hpp`~~ | RETIRE (C.2, juin 2026). | Supprime -- aucun client #include ; remplace par AmrCouplerMP. |
| `amr_multilevel.hpp`, `amr_reflux.hpp` | Moteurs AMR historiques simples. | Classer legacy/test ou fusionner documentation avec `amr_reflux_mf`. |
| `SpectralCoupler` | Variante mono-modele utile mais secondaire. | Garder si tests, sinon documenter comme API C++ avancee. |
| `DynamicModel` | Prototype CPU hote. | Garder, mais ne pas mettre dans chemin principal. |
| `AdvectionDiffusion`, `LangmuirMode`, `TwoFluidLinear` | Physiques potentiellement de test. | Si non utilisees par `adc_cases`, les marquer comme exemples/validation ou les sortir. |

## 7. Ce qui est bien abstrait

- `CompositeModel<Hyperbolic, Source, Elliptic>` : bon grain, pas trop puriste.
- `VariableRole` : indispensable pour sources couplees, Schur, DSL.
- `block_builder.hpp` : bon seam pour partager `add_block` et DSL production.
- `GridContext` : garde les closures compilees decouplees du runtime.
- `FiniteVolume` cote Python avec `riemann`, pas `flux` : bonne separation flux physique/numerique.
- `AmrSystem` documente ses limites au lieu de faire semblant.
- Garde FFT MPI : refuser proprement vaut mieux qu'un segfault.
- `CoupledSourceKernel` : bonne direction pour source couplee arbitraire sans callback Python.
- `Split` / `CondensedSchur` : bonne separation entre politique de splitting et etage source
  condense.
- `PolarPoissonSolver` : bon choix direct pour l'anneau polaire ; evite de forcer MG dans une
  geometrie ou il stagnait.

## 8. Plan de validation et d'optimisation MPI + Kokkos

### 8.1 Vocabulaire a figer

Dans le repo, "MPI" et "Kokkos" sont deux axes differents :

| Nom court | Build | Ce que ca valide |
|---|---|---|
| Serie CPU | `ADC_USE_MPI=OFF`, `ADC_USE_KOKKOS=OFF` | reference hote, boucle sequentielle. |
| MPI CPU | `ADC_USE_MPI=ON`, `ADC_USE_KOKKOS=OFF` | decomposition par rangs/processus, halos, reductions, rangs sans donnees. |
| Kokkos CPU Serial | `ADC_USE_KOKKOS=ON` avec device Serial | meme code Kokkos, mais CPU mono-thread ; bon garde-fou CI. |
| Kokkos CPU OpenMP | `ADC_USE_KOKKOS=ON` avec device OpenMP | parallelisme local CPU multi-thread via Kokkos. |
| Kokkos GPU | `ADC_USE_KOKKOS=ON` avec device Cuda/HIP | kernels cellules sur GPU, souvent `np=1`. |
| MPI + Kokkos CPU | `ADC_USE_MPI=ON`, `ADC_USE_KOKKOS=ON`, device Serial/OpenMP | MPI entre rangs + execution locale Kokkos CPU. |
| MPI + Kokkos GPU | `ADC_USE_MPI=ON`, `ADC_USE_KOKKOS=ON`, device Cuda/HIP | cible production distribuee : MPI entre rangs/noeuds, Kokkos sur GPU local. |

Regle de langage : ne pas ecrire "MPI valide" si la preuve est seulement CPU. Ecrire
explicitement `MPI CPU`, `Kokkos Cuda np=1`, ou `MPI + Kokkos Cuda np=1/2/4`.

### 8.2 Etat actuel a maintenir

Le design de base est bon : `parallel/comm.hpp` isole MPI, `mesh/for_each.hpp` isole Kokkos, et
`MultiFab` / `fill_boundary` font la jonction entre les deux. La cible naturelle est donc :

```text
rang MPI 0 -> kernels Kokkos sur CPU/GPU local
rang MPI 1 -> kernels Kokkos sur CPU/GPU local
...
```

Ce qui semble deja en place :

- options CMake separees : `ADC_USE_MPI` et `ADC_USE_KOKKOS` ;
- `for_each_cell` / reductions Kokkos pour les boucles locales ;
- `comm.hpp` pour rangs, barriers, all-reduce ;
- halos MPI dans `fill_boundary`, avec tampons en memoire unifiee sous Kokkos ;
- harnais GPU sous `python/tests/gpu/`, dont des cas MPI + Kokkos Cuda ;
- CI reguliere couvrant au minimum Release, MPI CPU et Kokkos Serial.

Ce qui reste a rendre systematique :

- un tableau de couverture par sous-systeme, pas seulement par PR ;
- des labels CTest par backend (`serial`, `mpi`, `kokkos-serial`, `kokkos-openmp`,
  `kokkos-cuda`, `mpi-kokkos-cuda`) ;
- une validation Kokkos OpenMP CPU explicite, car elle n'est pas strictement identique a Kokkos
  Serial ni au backend OpenMP historique ;
- une validation device-MPI reguliere sur ROMEO/GH200, au moins nightly ou manuelle standardisee.

### 8.3 Matrice de validation par couche

| Couche | MPI CPU | Kokkos CPU | Kokkos GPU | MPI + Kokkos GPU | Qualite attendue |
|---|---|---|---|---|---|
| `comm.hpp` | obligatoire | n/a | n/a | obligatoire | collectives appelees par tous les rangs, pas de deadlock. |
| `for_each.hpp` | indirect | obligatoire | obligatoire | obligatoire | pas de lambda device fragile cross-TU ; reductions tolerees non bit-exactes. |
| `Fab2D` / `MultiFab` | obligatoire | obligatoire | obligatoire | obligatoire | `local_size()==0` safe ; pas de `fab(0)` hors garde. |
| `fill_boundary` | obligatoire | obligatoire | obligatoire | critique | fence avant MPI ; tampons valides pour MPI CUDA-aware. |
| `spatial_operator` | utile | obligatoire | obligatoire | utile | memes schemas ; differences FP documentees. |
| `GeometricMG` / elliptique | obligatoire | obligatoire | obligatoire | critique | convergence sous tous backends ; refus propre des solveurs incompatibles. |
| `System` runtime | obligatoire | obligatoire | obligatoire | critique | `step`, `step_cfl`, Poisson, sources, DSL production. |
| `AmrSystem` / `advance_amr` | obligatoire | obligatoire | obligatoire | critique | reflux, average_down, regrid, halos fine/coarse. |
| Schur / Krylov | obligatoire | obligatoire | a valider apres #135 | a valider apres #135 | dot/norm collectifs, preconditionneur sans effet affine inattendu. |
| Polaire | utile | obligatoire | a valider apres #135 | plus tard | MMS polaire, Poisson polaire, pas de claim papier avant runtime complet. |

### 8.4 Plan concret

1. **Inventaire tests/backend**
   - Ajouter ou verifier des labels CTest.
   - Produire un tableau `docs/BACKEND_COVERAGE.md` :
     `test -> serial/MPI/Kokkos Serial/Kokkos OpenMP/Kokkos Cuda/MPI+Cuda`.
   - Marquer les tests qui s'auto-skip sous Kokkos ou MPI, pour eviter les faux verts.

2. **Kokkos CPU OpenMP**
   - Ajouter une configuration de build Kokkos OpenMP.
   - Lancer les tests de base : `for_each`, reductions, `MultiFab`, `fill_boundary`, `System`,
     `GeometricMG`, `advance_amr`.
   - Comparer aux resultats serie avec tolerance numerique, pas forcement bit-identique.

3. **MPI + Kokkos CPU**
   - Build combine `ADC_USE_MPI=ON` + `ADC_USE_KOKKOS=ON` avec device Serial puis OpenMP.
   - Verifier que les collectives restent appelees par tous les rangs meme quand un rang n'a pas
     de fab local.
   - Tests prioritaires : `fill_boundary`, `solve_fields`, `mf_arith::dot`, AMR reflux, Schur.

4. **Kokkos GPU np=1**
   - Standardiser les harnais GH200 : un script par groupe (`system`, `amr`, `elliptic`,
     `schur`, `polar`).
   - Exiger `compute-sanitizer` sur les petits cas.
   - Tout kernel nouveau doit avoir un test minimal Cuda ou etre explicitement note "non device".
   - Garder un test de regression pour le correctif #135 : `Geometry` / `Box2D` device-callable et
     reductions CFL collectives en MPI.

5. **MPI + Kokkos GPU np=2/4**
   - Valider d'abord les briques de communication : halos, reductions, `parallel_copy`,
     `refinement`.
   - Puis les chemins complets : `System` production DSL + `geometric_mg`, AMR multi-patch,
     Schur quand il sera device-clean.
   - Mesurer et documenter les limites MPI CUDA-aware : UVM, GPUDirect, fences, cout des
     all-reduce et des exchanges.

6. **Optimisation**
   - Remplacer les chemins host-side qui lisent de grands tableaux par kernels ou reductions
     Kokkos.
   - Eviter les fences globaux repetes ; regrouper les fences avant MPI ou lecture hote.
   - Reduire les allocations temporaires dans les boucles AMR (`FluxRegister`, face `MultiFab`,
     buffers de regrid).
   - Mesurer separement temps kernel, halos MPI, Poisson, regrid, reflux.
   - Pour GPU, surveiller : occupation, taille des kernels, acces stride, UVM page faults,
     synchronisations implicites.

### 8.5 Criteres de qualite

Un chemin "valide" doit preciser :

- backend exact (`MPI CPU`, `Kokkos OpenMP`, `Kokkos Cuda`, `MPI+Kokkos Cuda`) ;
- nombre de rangs (`np=1/2/4`) ;
- solveur elliptique (`geometric_mg`, `fft`, `polar_poisson`, Schur) ;
- tolerance attendue (`bit-identique`, `dmax < tol`, ou convergence MMS) ;
- comportement des rangs sans donnees locales ;
- compatibilite ou refus explicite si le chemin n'est pas supporte.

Qualite minimale pour merger une feature backend :

1. test serie ou CPU de reference ;
2. test Kokkos Serial ;
3. test MPI CPU si le code touche `MultiFab`, halos, reductions ou AMR ;
4. test Cuda ou justification explicite si le code est appele depuis un chemin device ;
5. pas de callback Python cellule par cellule dans le hot path.

## 9. Audit de la documentation

La documentation est devenue une deuxieme codebase. Elle contient de bonnes sources de verite, mais
elle garde aussi des couches historiques. Le risque principal est le meme que dans le code : plusieurs
documents disent presque la meme chose, avec des dates et statuts differents.

### 9.1 Carte des documents

| Document | Role actuel | Etat | Action recommandee |
|---|---|---|---|
| `README.md` | Page publique GitHub, pitch, API courte, build, validation. | Utile mais trop long. Il melange introduction, architecture, DSL, limites, validation GPU, roadmap. Plusieurs statuts sont deja fragiles. | Garder court : promesse, installation, exemple minimal, liens vers docs specialisees. Sortir les longues tables vers docs. |
| `todo.md` | Journal vivant de session. | Tres utile pour suivre les chantiers, mais deja perime : `master = #139`, #135 encore "en vol" alors que merge sur `origin/master`, reference `docs/ROADMAP.md` alors que le fichier vivant est archive. | Le traiter comme backlog vivant, pas source de verite publique. Le mettre a jour apres chaque vague de merge ou le renommer `DEVELOPMENT_STATUS.md`. |
| `docs/ARCHITECTURE.md` | Source principale pour les couches et frontieres. | Bonne base conceptuelle. Quelques compteurs/tests/statuts backend peuvent diverger. | Garder comme source de verite architecture ; ajouter un encart "statut runtime actuel" court et date. |
| `docs/ALGORITHMS.md` | Catalogue numerique : formules, code, validations. | Bon format. Doit rester stable et moins "session". | Ajouter Schur/polaire seulement comme algorithmes, pas comme roadmap. Garder les tests cites a jour. |
| `docs/DSL_MODEL_DESIGN.md` | Spec + historique du DSL. | Tres riche mais trop long ; melange API stable, notes historiques et statuts. Mentionne encore `AmrSystem.potential()` comme en cours alors que le binding existe. | Scinder : `DSL_API.md` court pour l'utilisateur, `DSL_MODEL_DESIGN.md` pour conception, historique en archive. |
| `docs/PAPER_ROADMAP.md` | Etat scientifique Hoffart / reproduction papier. | Utile et prudent, mais certains statuts infra sont stale (`AmrSystem.potential()`, AMR explicite) et doivent suivre les merges. | Garder comme doc scientifique. Ne pas y mettre les details de toutes les PR infra ; lier vers `BACKEND_COVERAGE.md`. |
| `docs/AMR_MULTIBLOCK_DESIGN.md` | Design capstone AMR multi-blocs. | Bonne source pour l'invariant conservatif : hierarchie commune, blocs co-localises. | Garder. Mettre a jour apres #140/#141 ; ne pas dupliquer dans README. |
| `docs/COUPLER_HIERARCHY.md` | Classement des coupleurs. | Tres utile pour clarifier public/interne/deprecated. | Le referencer depuis `ARCHITECTURE.md` et `CODEBASE_AUDIT.md`. |
| `docs/SCHUR_CONDENSATION_DESIGN.md` | Conception Schur. | Bon document de design, mais commence encore comme "cible" alors que plusieurs PR sont livrees. | Ajouter un encart de statut en tete : livre / reste / validation GPU. |
| `docs/GPU_RUNTIME_PORT.md` | Journal de validation GPU/GH200. | Indispensable pour les preuves device, mais tres historique. | Ajouter une synthese en tete par backend et garder le detail comme journal. |
| `docs/GPU_ROMEO.md` | Recette manuelle ROMEO. | Utile. | Garder operationnel, verifier commandes/module load apres chaque campagne. |
| `docs/PERFORMANCE.md` | Mesures M1/OpenMP/FFT/AMR. | Mesures anciennes et liees a des pilotes applicatifs. Ne doit pas etre lu comme perf actuelle globale. | Renommer ou preambule plus fort : "mesures historiques". Refaire une campagne Kokkos/MPI/GPU avant d'en tirer des conclusions. |
| `docs/CHOICES.md` | Decisions de design. | Court et utile. | Garder stable ; y pointer les decisions structurantes, pas les statuts de PR. |
| `docs/BIBLIOGRAPHY.md` | References externes. | Stable. | Garder. Ajouter les references Schur/polaire si necessaire. |
| `docs/CODE_DOCUMENTATION_CONVENTION.md` | Convention de commentaires. | Bonne source locale, alignee Doxygen / Google / C++ Core Guidelines / PEP 257. | Appliquer dossier par dossier ; ne pas faire un patch de commentaires massif. |
| `docs/sphinx/*.md` | Documentation publiee. | Plus courte, mais risque de diverger du README et de l'API courante. | Faire de Sphinx la doc utilisateur courte, generee/revue apres chaque release. |
| `docs/archive/*` | Historique et anciennes roadmaps. | Necessaire pour ne pas perdre le contexte, mais doit rester hors navigation principale. | Ne jamais citer un chemin archive comme source de verite actuelle sans le dire explicitement. |

### 9.2 Incoherences documentaires confirmees

Ces points ne demandent pas de debat architectural ; ce sont des divergences entre texte et code ou
entre documents :

1. **#135 est merge sur `origin/master`, mais `todo.md` le liste encore "en vol".**
   - Corriger l'etat courant et les sections Schur/polaire device.
   - Garder #140 comme seule PR ouverte structurante dans ce bloc ; #141 est deja merge.

2. **`AmrSystem.potential()` est documente "EN COURS" dans plusieurs documents, mais le binding existe.**
   - Code lu : `python/bindings.cpp` expose `.def("potential", ...)` pour `AmrSystem` et
     `python/amr_system.cpp` implemente `AmrSystem::potential()`.
   - Documents a corriger : `README.md`, `docs/DSL_MODEL_DESIGN.md`, `docs/PAPER_ROADMAP.md`,
     probablement `todo.md`.

3. **Le statut AMR "explicite seulement / pas IMEX" est encore present dans des docs publiques.**
   - Le moteur AMR a maintenant l'IMEX source locale (#132).
   - La phrase correcte est : runtime `AmrSystem` reste mono-bloc ; la source implicite locale existe ;
     Schur global et multi-blocs AMR restent des chantiers separes.

4. **`todo.md` reference `docs/ROADMAP.md`, qui n'existe plus comme doc active.**
   - Le fichier est `docs/archive/ROADMAP.md`.
   - Soit corriger le lien, soit assumer que `todo.md` n'est plus derive de cette roadmap.

5. **`README.md` presente parfois `DistributedFFTSolver` comme une capacite generale MPI.**
   - La phrase doit toujours preciser : `DistributedFFTSolver` existe et est teste a part, mais
     n'est pas route dans `System` a cause du layout en bandes.

6. **Les chiffres de tests et claims CI sont disperses.**
   - `README.md`, `docs/sphinx/installation.md`, `docs/ARCHITECTURE.md` et `todo.md` ne parlent pas
     toujours de la meme CI apres #136.
   - Action : creer une source unique `docs/BACKEND_COVERAGE.md` et lier depuis les autres docs.

7. **`docs/DSL_MODEL_DESIGN.md` garde volontairement des sections historiques, mais le lecteur peut
   confondre cible et etat courant.**
   - Le document a un bon `0bis`, mais la taille rend la lecture risquee.
   - Action : extraire une doc stable `docs/DSL_API.md` ou renforcer la doc Sphinx API.

8. **`docs/PERFORMANCE.md` n'est pas a jour avec la pile Kokkos/MPI/GPU actuelle.**
   - Le preambule dit que ce sont des mesures applicatives, mais le nom `PERFORMANCE.md` fait penser
     a un etat de performance courant.
   - Action : ajouter une date de campagne, backend exact, et un statut "historique".

### 9.3 Sources de verite proposees

Pour eviter les divergences :

| Sujet | Source de verite proposee | Autres docs |
|---|---|---|
| API utilisateur Python | `docs/sphinx/api.md` + un futur `docs/DSL_API.md` | README ne montre qu'un exemple minimal. |
| Architecture des couches | `docs/ARCHITECTURE.md` | README renvoie vers ce document. |
| Algorithmes numeriques | `docs/ALGORITHMS.md` | README liste seulement les familles. |
| Schur | `docs/SCHUR_CONDENSATION_DESIGN.md` + tests | PAPER_ROADMAP ne garde que l'impact scientifique. |
| AMR multi-blocs | `docs/AMR_MULTIBLOCK_DESIGN.md` | README ne doit pas decrire le capstone. |
| Backends / CI / GPU | futur `docs/BACKEND_COVERAGE.md` + `docs/GPU_RUNTIME_PORT.md` | README donne seulement le resume. |
| Reproduction papier | `docs/PAPER_ROADMAP.md` | todo ne doit pas etre la source scientifique. |
| Travail en cours | `todo.md` | Jamais cite comme preuve utilisateur sans date/commit. |

### 9.4 Plan de remise en ordre documentaire

1. **Patch court de verite**
   - Corriger `todo.md` pour #135 merge.
   - Corriger `AmrSystem.potential()` dans `README.md`, `DSL_MODEL_DESIGN.md`,
     `PAPER_ROADMAP.md`.
   - Corriger les phrases "AMR pas IMEX" pour tenir compte de #132.

2. **Creer `docs/BACKEND_COVERAGE.md`**
   - Tableau par test/groupe : Serie, MPI CPU, Kokkos Serial, Kokkos OpenMP, Kokkos Cuda,
     MPI + Kokkos Cuda.
   - Noter quels tests sont CI, quels tests sont ROMEO manuels, quels tests self-skip.

3. **Raccourcir `README.md`**
   - Garder : promesse, build, exemple Python minimal, matrice tres courte, liens.
   - Deplacer : longues limites DSL/AMR/GPU vers `DSL_MODEL_DESIGN.md`, `PAPER_ROADMAP.md`,
     `BACKEND_COVERAGE.md`.

4. **Scinder la doc DSL**
   - `DSL_API.md` : comment ecrire un modele, compiler, brancher `System`/`AmrSystem`.
   - `DSL_MODEL_DESIGN.md` : decisions et historique technique.

5. **Mettre Sphinx au meme niveau que la doc Markdown**
   - Corriger les exemples `Spatial` / `FiniteVolume` selon l'API recommandee.
   - Generer ou relire Sphinx apres chaque changement API utilisateur.

6. **Archiver explicitement les documents historiques**
   - `docs/archive/README.md` doit dire : "documents non normatifs".
   - Les docs actives ne doivent pas renvoyer vers `archive/*` sauf pour historique.

## 10. Plan de nettoyage propose

### Lot A - Documentation et verite API

1. Corriger le docstring de tete de `python/adc/dsl.py`.
2. Ajouter dans `ARCHITECTURE.md` un tableau "API actuelle / interne / legacy".
3. Ajouter une note : `SourceImplicit` = source locale implicite, `CondensedSchur` = etage global.
4. Neutraliser les mentions applicatives dans les commentaires de headers generiques
   (`diocotron` -> `anneau polaire` / `derive ExB` quand c'est le sens reel).
5. Appliquer progressivement `CODE_DOCUMENTATION_CONVENTION.md` dossier par dossier : entete de
   fichier, entete de classe, invariants MPI/GPU/conservation, sans churn automatique.

### Lot B - Runtime `System`

1. Extraire `NativeLoader` depuis `python/system.cpp`.
2. Extraire `SystemFieldSolver` pour `ensure_elliptic`, `solve_fields`, `apply_eps/kappa/Bz/Te`.
3. Extraire `SystemBlockStore` pour blocs, ghosts, get/set state.
4. Garder `System::Impl` comme orchestrateur mince.

### Lot C - AMR

1. Decouper `amr_reflux_mf.hpp`.
2. `amr_coupler.hpp` RETIRE (C.2, juin 2026) -- aucun client #include, supprime, remplace par AmrCouplerMP.
3. S'appuyer sur la garde de layout stricte #141 avant toute API multi-blocs.
4. Corriger la cadence `stride` AMR type #140 avant d'exposer des pas lents depuis Python.
5. Decider explicitement : `AmrSystem` reste mono-bloc ou devient runtime multi-bloc.
6. Si runtime multi-bloc : imposer une hierarchie AMR commune, regrid par union des tags
   (`electrons OR ions OR neutres OR phi OR user`), prolongation/restriction et reflux bloc par
   bloc, sources couplees sur cellules co-localisees. Ne pas introduire de hierarchies separees ni
   de blocs absents de certains patches tant que la conservation n'est pas formellement traitee.

### Lot D - Schur et elliptique

1. Extraire une interface `EllipticOperator` commune entre Poisson scalaire, tenseur plein et
   operateur Schur.
2. Garder `TensorKrylovSolver` comme solveur, pas comme proprietaire du probleme physique.
3. Garder `CondensedSchurSourceStepper` comme etage de temps/couplage, pas comme
   `model.source(U,aux)`.
4. Valider Schur sur `Kokkos Cuda` avec un harnais dedie maintenant que #135 est merge.

### Lot E - Validation

1. Tests de non-regression stride : hold-then-catch-up AMR (#140), `step_cfl` avec stride,
   Poisson avec bloc tenu.
2. Tests roles obligatoires pour sources couplees DSL.
3. Tests d'erreur explicite pour `PolarMesh` tant que runtime polaire complet absent.
4. Tests backend avec noms precis : `MPI CPU`, `Kokkos Serial`, `Kokkos Cuda`,
   `MPI + Kokkos Cuda`.
5. Tests de couverture CI auto-decouverte : verifier que l'optimisation #136 ne masque pas un test
   non enregistre.

## 11. Verdict

`adc_cpp` a maintenant un vrai niveau d'abstraction. Le probleme n'est pas que le coeur serait mal
pense ; le probleme est qu'il a grandi tres vite et garde plusieurs generations de chemins valides.
Les merges #118-#142 ont ajoute des briques importantes : Schur utilisable depuis Python,
sources couplees DSL, primitives runtime, cadence substeps-aware, polaire transport/Poisson, et
beaucoup de garde-fous API. Ces ajouts rendent le projet plus puissant, mais ils augmentent aussi la
pression sur `python/system.cpp`, `amr_reflux_mf.hpp` et la matrice backend.

Le code est maintenable si on fait maintenant deux choses :

1. **Classer les surfaces** : public / interne / legacy / test.
2. **Extraire les classes-dieu** : surtout `python/system.cpp` et `amr_reflux_mf.hpp`.
3. **Nommer les preuves backend** : ne jamais remplacer `MPI CPU`, `Kokkos Cuda` ou
   `MPI + Kokkos Cuda` par un simple "MPI".

Le principe a conserver est simple : une classe doit soit decrire une physique locale, soit appliquer
un operateur numerique, soit stocker des donnees, soit orchestrer. Quand elle commence a faire deux de
ces choses a la fois, elle doit etre decoupee.
