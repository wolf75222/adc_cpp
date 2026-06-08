> **ARCHIVE -- document non-normatif.** Plan/spec interne conservee pour l'historique. La source de verite courante est le code + les docs canoniques (ARCHITECTURE.md, ALGORITHMS.md, BACKEND_COVERAGE.md). Ne pas s'y fier comme etat courant.

# Plan d'extraction de python/system.cpp

PLAN UNIQUEMENT. Ce document ne modifie aucun code. Il identifie quatre candidats
d'extraction dans `python/system.cpp` (1805 lignes), precise les plages de code courantes,
les dependances et les risques, puis propose une sequence de PR bit-identiques.

---

## 0. Contexte

`python/system.cpp` est la seule unite de traduction de la facade runtime `System`. Elle
contient en vrac :
- le chargement de `.so` par `dlopen` (trois chemins : JIT, AOT, natif) ;
- le solveur elliptique et son cablage (construction paresseuse, champs eps/kappa) ;
- le registre de blocs (`Impl::Species`, `sp`) et ses fermetures type-erase ;
- le pas de temps (`step`, `step_cfl`, `step_adaptive`).

La taille rend la navigation couteuse et les reviews longs. L'objectif est de decouper sans
aucun changement de comportement observable.

---

## 1. NativeLoader

### Ce qu'il possede

Tout ce qui touche au chargement d'un `.so` genere par le DSL :

- `add_dynamic_block` : dlopen, lecture de `adc_model_nvars`, dispatch vers `push_dynamic<NV>`,
  template `Impl::push_dynamic<NV>` (fermetures hote `host_residual`, `max_speed`, `advance`,
  `add_poisson`) ;
- `add_compiled_block` : dlopen, lecture des symboles AOT (`adc_compiled_residual/advance/
  max_speed/poisson_rhs/naux`), construction des fermetures marshaling, `read_block_meta` ;
- `add_native_block` : dlopen avec promotion RTLD_GLOBAL, garde-fou ABI (`adc_native_abi_key`
  vs `abi_key()`), appel de `adc_install_native` ;
- utilitaires partagees par les trois : `host_residual<NV>`, `limited_slope`, `read_block_meta`,
  `parse_var_set`, `split`, `BlockMeta`.

### Plages actuelles (approximatives)

| Sous-section | Lignes |
|---|---|
| `limited_slope` + `host_residual<NV>` (template hote) | 111-197 |
| `role_index`, `split`, `BlockMeta`, `parse_var_set`, `read_block_meta` | 198-266 |
| `Impl::push_dynamic<NV>` (bloc JIT) | 733-858 |
| `System::add_dynamic_block` | 985-1015 |
| `System::add_compiled_block` | 1017-1153 |
| `System::add_native_block` | 1155-1241 |

Total estime : ~350 lignes.

### Dependances

- `dlopen`/`dlsym`/`dlclose` (POSIX) + `dlfcn.h` ;
- `IModel<NV>` (`runtime/dynamic_model.hpp`) pour le chemin JIT ;
- `GridContext` (`runtime/block_builder.hpp`) pour les fermetures AOT/natif ;
- `System::install_block` (doit rester dans system.cpp ou etre dans un header partageable) ;
- `System::ensure_aux_width` (modifie le canal aux partage, voir SystemBlockStore) ;
- `abi_key()` (ligne 98, symbole exporte).

### Risques

- FAIBLE. Aucun de ces chemins n'est sur le hot path GPU/MPI. Les fermetures capturent `P`
  (pointeur `Impl*`) et les lambdas type-erase de `push_dynamic` contiennent `copy_state`/
  `write_state` qui restent dans `Impl` (ou il faut les exposer).
- Point d'attention : `push_dynamic<NV>` appelle `P->ensure_aux_width` et `P->write_state` /
  `P->copy_state`. Si on extrait NativeLoader hors de `Impl`, ces acces doivent passer par
  une interface ou rester des amis de `Impl`.
- `add_native_block` emet un `dlopen(RTLD_GLOBAL)` qui modifie la visibilite du module
  courant. Ce comportement doit rester atomique et ne pas etre reordonne par rapport au
  garde-fou ABI.

---

## 2. SystemFieldSolver

### Ce qu'il possede

Tout ce qui construit et exploite le solveur elliptique :

- configuration paresseuse (`ensure_elliptic`, `Impl::p_rhs`, `p_solver`, `p_bc`, `p_wall`,
  `p_wall_radius`, `p_eps_`, `has_eps_field_`, `p_eps_field_`, `has_eps_xy_field_`,
  `p_eps_x_field_`, `p_eps_y_field_`, `has_kappa_field_`, `p_kappa_field_`,
  `std::optional<std::variant<GeometricMG, PoissonFFTSolver>> ell_`) ;
- installation des champs operateur : `apply_epsilon_field`, `apply_epsilon_anisotropic_field`,
  `apply_reaction_field` ;
- accesseurs internes : `ell_rhs()`, `ell_phi()`, `ell_solve()` ;
- calcul du champ electrique et peuplement de l'aux : `solve_fields()` (le corps) ;
- API publiques : `System::set_poisson`, `System::set_epsilon_field`,
  `System::set_epsilon_anisotropic_field`, `System::set_reaction_field`,
  `System::solve_fields()`, `System::potential()`.

### Plages actuelles (approximatives)

| Sous-section | Lignes |
|---|---|
| Membres Impl (p_rhs..ell_) | 329-342 |
| `ensure_elliptic` + garde-fous FFT/MPI | 482-534 |
| `apply_epsilon_field` | 538-553 |
| `apply_epsilon_anisotropic_field` | 554-577 |
| `apply_reaction_field` | 578-595 |
| `ell_rhs` / `ell_phi` / `ell_solve` | 596-615 |
| `solve_fields()` (corps Impl) | 640-698 |
| `System::set_poisson` | 1243-1254 |
| `System::set_epsilon_field` | 1256-1266 |
| `System::set_epsilon_anisotropic_field` | 1268-1283 |
| `System::set_reaction_field` | 1285-1296 |
| `System::ensure_aux_width` (delegation) | 1298 |
| `System::solve_fields()` (delegation) | 1640 |
| `System::potential()` | 1793-1803 |

Total estime : ~120 lignes + membres Impl.

### Dependances

- `GeometricMG` et `PoissonFFTSolver` (headers inclus) ;
- `apply_bz()` et `apply_te()` : appeles depuis `ensure_aux_width` et `solve_fields` ->
  dependance avec SystemBlockStore. Le `solve_fields` appelle `apply_te()` apres
  `ell_solve()` : l'ordre est semantiquement important (T_e calculee APRES le nouveau phi,
  B_z preserve) ;
- `comm::n_ranks()` (garde FFT MPI) ;
- `wall_active()` -> `detail::wall_predicate` (`runtime/wall_predicate.hpp`) ;
- `device_fence()` (apres `ell_solve`, avant derivation grad phi) : point chaud pour les
  bugs device. NE PAS reordonner ce fence.
- `fill_boundary` et `fill_ghosts` (remplissage halos aux en fin de `solve_fields`).

### Risques

- MOYEN. `solve_fields` est appele a chaque pas depuis `step`/`step_cfl`/`step_adaptive`.
  C'est un hot path au sens frequence (mais pas au sens kernel GPU : la boucle `for_each_cell`
  y vit dans `GeometricMG`).
- Le `device_fence()` entre `ell_solve()` et la derivation grad phi est un invariant de
  correction sur GPU (sans ce fence, le kernel du V-cycle n'est pas termine quand on lit
  `phi_mf`). Toute extraction doit preserver cet ordre.
- La garde `if (rhs.local_size() == 0) return` dans les boucles MPI est critique (evite
  `fab(0)` sur un rang sans box). A auditer si le corps est deplace.

---

## 3. SystemBlockStore

### Ce qu'il possede

Le registre de blocs et toutes les operations qui portent sur `sp` directement :

- struct `Impl::Species` (membres, commentaires, constructeur) ;
- `Impl::sp` (vecteur, macro_step_, t, stride_due) ;
- `Impl::find`, `Impl::index` ;
- `install_block` (creation de l'espece, push dans sp) ;
- `set_block_ghosts`, `set_block_conversion` ;
- `set_magnetic_field`, `set_electron_temperature_from` (cablage B_z et T_e sur le registre) ;
- `apply_bz()`, `apply_te()` (peuplement du canal aux a partir du registre) ;
- `set_density`, `set_state`, `get_state`, `set_primitive_state`, `get_primitive_state`,
  `eval_rhs`, `density`, `mass` (accesseurs etat) ;
- `variable_names`, `variable_roles`, `block_gamma`, `n_vars`, `block_names`, `n_species` ;
- `add_ionization`, `add_collision`, `add_thermal_exchange`, `add_coupled_source`,
  `add_source_stage` (sources couplees et etage Schur) ;
- `apply_couplings`, `run_source_stage`.

### Plages actuelles (approximatives)

| Sous-section | Lignes |
|---|---|
| `Impl::Species` struct | 272-300 |
| Membres communs Impl (geom, ba, dm, bc_, t, macro_step_, couplings...) | 301-318 |
| `stride_due` | 326 |
| `ensure_aux_width`, `apply_bz`, `set_block_ghosts`, `apply_te` | 357-430 |
| `Impl::find`, `Impl::index` | 438-452 |
| `apply_couplings` | 458-461 |
| `run_source_stage` | 627-630 |
| `System::add_block` (dispatch_model + install_block) | 887-954 |
| `System::install_block` | 960-976 |
| `System::set_block_ghosts` (delegation) | 981-983 |
| `System::add_ionization` / `add_collision` / `add_thermal_exchange` | 1319-1418 |
| `System::add_coupled_source` | 1420-1522 |
| `System::set_source_stage` | 1524-1570 |
| `System::set_density` | 1572-1587 |
| `System::set_block_conversion` | 1589-1594 |
| `System::set_primitive_state` / `get_primitive_state` | 1596-1638 |
| `System::eval_rhs`, `get_state`, `set_state`, `n_vars`, `variable_names/roles/block_gamma` | 1740-1776 |
| `System::nx`, `time`, `n_species`, `block_names`, `mass`, `density` | 1778-1792 |

Total estime : ~550 lignes.

### Dependances

- `dispatch_model` (`runtime/model_factory.hpp`) dans `add_block` ;
- `make_block`, `make_max_speed`, `make_poisson_rhs`, `make_cell_convert`, `block_n_ghost`
  (`runtime/block_builder.hpp`) dans `add_block` ;
- `CondensedSchurSourceStepper` (`coupling/condensed_schur_source_stepper.hpp`) dans
  `set_source_stage` et `run_source_stage` ;
- `CoupledSourceKernel` (`coupling/coupled_source_program.hpp`) dans `add_coupled_source` ;
- `for_each_cell` dans les lambdas de couplage ;
- `sum` (`mesh/mf_arith.hpp`) dans `mass`.
- `ensure_aux_width` + `apply_bz`/`apply_te` : dependances circulaires potentielles avec
  SystemFieldSolver (les deux lisent/ecrivent le canal `aux`). A gerer par interface ou par
  co-localisation dans un meme fichier.

### Risques

- MOYEN-ELEVE. `add_ionization`, `add_collision`, `add_thermal_exchange` construisent des
  lambdas capturant `P->sp[ia].U.fab(0).array()` : le fab est referencement DIRECT. Si
  `sp` est realloue entre l'enregistrement et l'application (push_back), les lambdas
  deviendraient invalides. C'est un risque EXISTANT (pas introduit par l'extraction), mais
  l'extraction ne doit pas le masquer.
- `add_coupled_source` et `add_ionization` utilisent `fab(0)` sans garde `local_size()` pour
  les lambdas de couplage (contrairement a `apply_bz`/`solve_fields` qui iterent sur
  `local_size()`). Ce n'est pas un bug ici (System a une seule box, round-robin, donc le
  proprietaire a toujours `fab(0)`) mais c'est fragile si le layout change.
- Les sources couplees sont appliquees APRES `step`/`step_cfl` via `apply_couplings`. L'ordre
  "transport puis couplages" est une convention semantique a preserver.

---

## 4. SystemStepper

### Ce qu'il possede

La logique de pas de temps :

- `System::step(dt)` ;
- `System::advance(dt, nsteps)` ;
- `System::step_cfl(cfl)` ;
- `System::step_adaptive(cfl)`.

### Plages actuelles (approximatives)

| Sous-section | Lignes |
|---|---|
| `System::step` | 1642-1659 |
| `System::advance` | 1660-1662 |
| `System::step_cfl` | 1663-1701 |
| `System::step_adaptive` | 1702-1738 |

Total : ~100 lignes.

### Dependances

- `solve_fields()` (appel Impl) : appele en tete de chaque pas ;
- `Impl::stride_due` : filtre par bloc ;
- `s.advance(s.U, eff_dt, s.substeps)` : fermeture type-erase capturee dans Species ;
- `run_source_stage(s, eff_dt)` : etage Schur OPT-IN (no-op si nullptr) ;
- `apply_couplings(dt)` : sources couplees post-transport ;
- `s.max_speed(s.U)` dans `step_cfl`/`step_adaptive` pour calculer le dt CFL par bloc.

### Risques

- FAIBLE. Ces quatre fonctions sont des orchestrateurs lisibles et sans acces direct aux fabs.
  Elles appellent les blocs de SystemFieldSolver (via `P->solve_fields()`) et SystemBlockStore
  (via `P->sp`). L'extraction est propre.
- Point d'attention : la formule CFL de `step_cfl` (substeps-aware, post-#121) est documentee
  et commentee ; ne pas la simplifier par erreur lors du deplacemenent.
- `step_adaptive` calcule les sous-pas a la volee (`ceil(stride * w_b / w_min)`) : la semantique
  multirate est subtile (voir commentaire ligne 1726). A verifier que les commentaires suivent.

---

## 5. Sequence de PR recommandee

Principe : chaque PR laisse les tests CI verts (bit-identique) et ne change aucun comportement.
Sequence du moins dependant au plus dependant.

### PR-1 : extraire NativeLoader

WRITE-SET :
- Nouveau fichier `python/system_native_loader.cpp` (ou `python/native_loader.cpp`) ;
- `python/system.cpp` : supprimer les fonctions deplacees, ajouter un include si necessaire ;
- `python/CMakeLists.txt` : ajouter le nouveau fichier a la cible `_adc`.

Contenu deplace : `host_residual`, `limited_slope`, `split`, `BlockMeta`, `parse_var_set`,
`read_block_meta`, `Impl::push_dynamic<NV>`, `add_dynamic_block`, `add_compiled_block`,
`add_native_block`.

DEPENDANCES A EXPOSER : `Impl::copy_state`, `Impl::write_state`, `Impl::ensure_aux_width`
doivent rester accessibles depuis le nouveau fichier (par friend, par header Impl, ou en
les deplacant dans un header partageable). La solution la moins invasive : placer `push_dynamic`
dans un header `python/impl_dynamic.hpp` inclus par les deux TU, ou l'implementer dans le
nouveau fichier en passant `Impl*` explicitement (deja le cas : `push_dynamic(Impl* P, ...)`).

TESTS D'ACCEPTATION :
- `test_dynamic_model` (modele type-erased == Euler statique) ;
- `test_compiled_model_parity` (AOT natif == bloc natif) ;
- `test_dsl_*` (suite Python) ;
- tous les tests CI actuels (Release + MPI + Kokkos).

RISQUE : FAIBLE. Aucun hot path, aucun acces direct aux fabs GPU. Seul point : maintenir les
symboles `ADC_EXPORT` (`install_block`, `grid_context`, `ensure_aux_width`) dans `system.cpp`
pour que les loaders natifs les resolvent.

---

### PR-2 : extraire SystemFieldSolver

WRITE-SET :
- Nouveau fichier `python/system_field_solver.cpp` ;
- `python/system.cpp` : supprimer les methodes et membres Impl deplacees.

Contenu deplace : membres Impl elliptique (`p_rhs`, ..., `ell_`), `ensure_elliptic`,
`apply_epsilon_field`, `apply_epsilon_anisotropic_field`, `apply_reaction_field`, `ell_rhs()`,
`ell_phi()`, `ell_solve()`, `solve_fields()` (corps Impl), `System::set_poisson`,
`System::set_epsilon_field`, `System::set_epsilon_anisotropic_field`,
`System::set_reaction_field`, `System::solve_fields()`, `System::potential()`,
`System::ensure_aux_width` (delegation).

DEPENDANCES CRITIQUES :
- `apply_bz()` et `apply_te()` restent dans SystemBlockStore mais sont appelees depuis
  `ensure_aux_width` et `solve_fields`. Solution : les exposer comme methodes publiques de
  `Impl` (elles le sont deja) et les appeler via `P->apply_bz()`.
- `device_fence()` entre `ell_solve()` et la derivation grad phi : cet ordre doit rester
  atomique. Ne pas le refactorer en deux appels distincts.
- La garde `local_size() == 0` dans les boucles de derivation et de peuplement B_z/T_e :
  verifier qu'elle est preservee.

TESTS D'ACCEPTATION :
- `test_bindings.py::test_poisson_*` (eps variable, ecrante, anisotrope) ;
- `test_mpi_system_solve_fields_np{1,2,4}` (bit-identique np=1/2/4) ;
- `test_system_abstraction`, `test_system_coupler`.

RISQUE : MOYEN. `solve_fields` est sur le hot path (appel par pas) mais le corps ne contient
pas de boucle critique : c'est un orchestrateur. Le risque reel est le reordonnancement
accidentel du `device_fence`.

---

### PR-3 : extraire SystemStepper

WRITE-SET :
- Nouveau fichier `python/system_stepper.cpp` ;
- `python/system.cpp` : supprimer `step`, `advance`, `step_cfl`, `step_adaptive`.

Contenu deplace : `System::step`, `System::advance`, `System::step_cfl`,
`System::step_adaptive`.

DEPENDANCES : toutes les dependances sont des methodes de `Impl` (`solve_fields`, `sp`,
`stride_due`, `apply_couplings`, `run_source_stage`). Aucune nouvelle API a exposer si on
implemente dans une TU qui inclut la definition complete d'`Impl` (via un header partageable).

TESTS D'ACCEPTATION :
- `test_system_two_explicit`, `test_adaptive_multirate`, `test_multirate_stride` ;
- `test_cfl_dt` (formule substeps-aware) ;
- `test_assembler_driver`.

RISQUE : FAIBLE. Fonctions d'orchestration pures, sans acces direct aux fabs.

---

### PR-4 : extraire SystemBlockStore

WRITE-SET :
- Nouveau fichier `python/system_block_store.cpp` ;
- `python/system.cpp` : supprimer tout ce qui reste (struct Species, registre sp,
  add_block, install_block, add_ionization, ..., set_density, ...).

Contenu deplace : voir section 3.

DEPENDANCES :
- `dispatch_model` et les fermetures de `block_builder` restent des includes dans le nouveau
  fichier ;
- `CondensedSchurSourceStepper` et `CoupledSourceKernel` pareil ;
- `for_each_cell` dans les lambdas de couplage.

TESTS D'ACCEPTATION : toute la suite CI (le registre de blocs est utilise par tous les tests).

RISQUE : ELEVE (par la taille et la pervasivite, pas par la complexite). C'est la PR la plus
large. Recommandation : la decouvrir en deux etapes :
- PR-4a : `install_block`, `set_density`, `get_state`/`set_state`, accesseurs lectures
  (variable_names, block_gamma...) ;
- PR-4b : `add_block` + fermetures dispatch_model, sources couplees (add_ionization...,
  add_coupled_source), `set_source_stage`.

---

## 6. Resume des risques par PR

| PR | Contenu | Lignes ~ | Risque | Hot path GPU | MPI local_size() |
|---|---|---|---|---|---|
| PR-1 NativeLoader | dlopen, JIT, AOT, natif | 350 | FAIBLE | non | oui (add_poisson des blocs .so) |
| PR-2 FieldSolver | elliptique, solve_fields | 120 | MOYEN | oui (device_fence) | oui (derivation phi) |
| PR-3 Stepper | step, step_cfl, step_adaptive | 100 | FAIBLE | non | via solve_fields |
| PR-4 BlockStore | Species, sp, add_block, couplages | 550 | ELEVE (taille) | oui (for_each_cell couplages) | oui (lambdas couplage) |

---

## 7. Precondition commune a toutes les PR

Avant PR-1, creer un header `python/impl_fwd.hpp` (ou equivalent) qui expose :
- La definition complete de `Impl` (ou les membres necessaires a la TU externe) ;
- Les methodes `Impl` appelees cross-TU.

Cela evite de placer des definitions de methodes dans un header et de generer des ODR violations.
Alternative plus conservative : utiliser la technique friend + forward declaration dans `system.hpp`,
et laisser `Impl` defini uniquement dans `system.cpp`. Les methodes des TU auxiliaires recoivent
alors `Impl*` et ont acces via `friend` declare dans `System::Impl`.

Verification bit-identique (recette) : chaque PR compare `max|U_avant - U_apres|` sur un cas de
reference (diocotron n=64, 20 pas, np=1) et verifie `dmax == 0.0` avant de merger.
