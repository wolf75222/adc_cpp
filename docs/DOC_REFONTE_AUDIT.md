# Audit documentaire adc_cpp / adc_cases -- matrice de verite (Phase 1)

> Document de synthese assemble a partir de l'inventaire 8-agents (prose adc_cpp, Sphinx, trois docs canoniques verifiees code-en-main, API Python/DSL reelle, cas adc_cases, build/run, tests, assets) plus une passe de verification adverse. Les verdicts marques CONFIRME proviennent de la passe adverse independante ; les autres reposent sur l'inventaire seul (signale la ou la donnee est mince).

---

## 1. Synthese executive

- **L'ossature canonique est fiable, le bookkeeping ne l'est pas.** Les docs de fond (ARCHITECTURE, ALGORITHMS, BACKEND_COVERAGE, CHOICES, DSL_API, DSL_MODEL_DESIGN, CONSERVATION_SUMMARY, BIBLIOGRAPHY, COUPLER_HIERARCHY, COUPLING_SURFACE, PERFORMANCE) decrivent correctement les couches, les concepts C++20 (`PhysicalModel`, `EllipticSolver`), la couture parallele three-way (`for_each.hpp` + `comm.hpp`), la matrice CI 4-jobs CPU-only. Le risque est concentre sur (a) le recit "AmrSystem mono-bloc" et (b) la comptabilite (compteurs de tests, symboles fantomes, fichiers supprimes).

- **Risque #1 (haute severite, CONFIRME) : "AmrSystem mono-bloc" est OBSOLETE.** Repete comme limitation honnete dans README, ARCHITECTURE Sec. 8, DSL_MODEL_DESIGN (Sec. 0bis/Sec. 5/Phase D) et la memoire persistante, alors que `python/amr_system.cpp` livre une vraie facade multi-bloc (`multi_block()`/`build_multi()`/`set_regrid`, plus aucun throw "un seul bloc"), avec 7 tests capstone et regrid union-tags (#195/#199/#205). Une refonte qui fait confiance aux docs canoniques **re-affirmerait une fausse limitation**.

- **Risque #2 (CONFIRME) : fichier supprime encore documente.** `amr_coupler.hpp`/`AmrCoupler` est DELETE (#164) mais decrit comme "deprecie conserve pour reference" dans TROIS docs normatives (ARCHITECTURE:33, COUPLING_SURFACE:118, COUPLER_HIERARCHY Sec. 4). Seul CODEBASE_AUDIT enregistre la suppression. Tout autodoc/reference coupler bati sur ces trois docs listerait un fichier inexistant.

- **Symbole fantome : le limiteur "MC".** Documente deux fois (ALGORITHMS:97, ARCHITECTURE:288/509) mais absent de `reconstruction.hpp` (seuls `NoSlope`/`Minmod`/`VanLeer`/`Weno5` existent). Un utilisateur qui suit la doc et passe `limiter='mc'` echoue.

- **Compteurs de tests : tous perimes et mutuellement incoherents.** Disque reel ~ 90-94 `adc_add_test` (non-MPI), ~51 entrees MPI, 60-61 fichiers Python. Les docs disent 71/84/103 (C++) et 26/55 (Python) selon le fichier ; ARCHITECTURE et BACKEND_COVERAGE se contredisent entre eux. Toutes les valeurs sont des **sous-estimations conservatrices** (la suite est plus grosse qu'annoncee). A regenerer programmatiquement, pas a coder en dur.

- **Sphinx : mecanisme de galerie mort (CONFIRME).** `_copy_tutorials.setup_gallery()` pointe vers un `tutorials/` supprime (commit 194c63f, app layer parti vers adc_cases) -> copie 0 fichier ; et meme rempli, `tutorials_index.md` est un stub sans toctree -> doublement orphelin. Un `sphinx-build` propre produit une section "Tutoriels" vide. `_build/`/`_generated/` locaux sont perimes (mai 30) mais deja gitignores/non-trackes.

- **Honnetement runnable en local (darwin/arm64, sans CUDA) :** coeur C++ Serie + ctest, MPI (OpenMPI 5.0.9), OpenMP standalone, Kokkos device=OpenMP, et le module Python -- **mais uniquement avec l'interpreteur qui l'a compile** (`/opt/homebrew/anaconda3/bin/python3.12`, qui a numpy). Footgun confirme : le `.so` est `cpython-312` ; sous `python3` systeme (3.9.6) -> `ModuleNotFoundError: adc._adc` ; sous un 3.12 sans numpy -> mort sur `import numpy` dans `dsl.py`.

- **Necessite ROMEO/GH200 :** tout chemin CUDA -- Kokkos Cuda mono-GPU, MPI+Kokkos Cuda multi-GPU. La CI ne construit **jamais** `Kokkos_ENABLE_CUDA=ON`. Les harnais `python/tests/gpu/*.cpp` sont SLURM-only (`--constraint=armgpu`, nvcc_wrapper, Kokkos 4.4.01) et exclus du glob CI (`-maxdepth 1`). Toute mention "GPU-valide" doit dire "manuellement sur ROMEO GH200".

- **DSL_API.md est materiellement perime (4 claims faux haute severite, CONFIRMES).** Des snippets crashent tels qu'ecrits (`m.u[0]`/`m.a.grad_x` -> AttributeError ; `set_poisson('geometric_mg')` -> rhs inconnu ; `AmrSystem(max_level=2)` -> attribut inconnu ; "runtime params = NotImplementedError" faux : ils marchent). Le claim runtime-NotImplementedError vit aussi dans le **docstring source** (`dsl.py:1623/1627`) -> l'autodoc le propagerait.

- **adc_cases : labeling diocotron-vs-Hoffart HONNETE.** Le diocotron ExB reduit est partout flagge "benchmark de normalisation modele reduit, PAS reproduction du systeme Euler-Poisson complet" ; le cas complet `hoffart_euler_poisson_dsl` est classe "reproduction-candidate" avec table PENDING/MEASURED montrant -82 a -95% d'erreur. Aucun fichier ne ment. **Mais** le README top-level adc_cases omet 6 cas DSL ; seuls 4/15 cas ont un README ; et la provenance des assets est faible (figures committees sans SHA, produites ailleurs que leur chemin committe).

---

## 2. Matrice des fichiers documentaires

Statuts : **active** (normatif vivant), **historical** (spec/plan partiellement execute), **archive** (sous `docs/archive/`), **generated** (artefact). Decisions : keep / rewrite / merge / archive / delete.

### adc_cpp -- prose & docs/

| Fichier | Role | Public | Statut | Source de verite | Obsolete/Faux | Doublons | Decision |
|---|---|---|---|---|---|---|---|
| README.md | Portail projet (pitch, GIF, build, validation) | new user | active | ARCHITECTURE + GPU_RUNTIME_PORT + DSL_MODEL_DESIGN | "AmrSystem mono-bloc" (l.130-132) FAUX ; `potential()` :272 stale (reel :332) ; 208 l. vs cible 120-150 | ARCHITECTURE Sec. 6/Sec. 13, DSL Sec. 5, GPU_RUNTIME_PORT | **rewrite** |
| docs/ARCHITECTURE.md | Canonique : 5 couches, concepts, module map, AMR Sec. 8, validation Sec. 11 | contributor | active | `include/adc/**` | `amr_coupler.hpp` (l.33) supprime ; Sec. 8 mono-bloc obsolete ; Sec. 11 compteurs perimes ; limiteur MC inexistant | README module-map, COUPLER_HIERARCHY, GPU_RUNTIME_PORT | **rewrite** |
| docs/ALGORITHMS.md | Canonique : catalogue methodes (formule/code/test x20) | contributor | active | `numerics/**` + `tests/` | Limiteur MC (l.97) inexistant ; le reste verifie bon | Sec. 18 DSL compose, Sec. 19 DSL Sec. 0/Sec. 7, Sec. 20 ARCH Sec. 4 | **keep** (purger MC) |
| docs/BACKEND_COVERAGE.md | SoT auto-declaree matrice backends/tests | contributor | active | `tests/CMakeLists.txt` + CI | Totaux d'en-tete perimes (84/19/55 vs disque 90-94/31/60-61) ; lignes per-test OK | remplace README "4 chemins" + DSL Sec. 5 | **rewrite** (totaux) |
| docs/CHOICES.md | Canonique : rationale decisions D-1..D-7 | contributor | active | ARCHITECTURE + oracles | Aucun faux ; seul doc accentue (incoherence encodage) | -- | **keep** |
| docs/DSL_API.md | Reference user courte DSL | new user | active | `adc/dsl.py` + DSL_MODEL_DESIGN | **4 claims faux** (m.u/m.a, set_poisson, max_level, runtime) ; backend GPU sur-vendu | sous-ensemble DSL Sec. 5 | **rewrite** |
| docs/DSL_MODEL_DESIGN.md | Canonique : design DSL/modele + statut par phase | contributor | active | `dsl.py` + `system.cpp` + `runtime/**` | "AmrSystem mono-bloc" (Sec. 0bis/Sec. 5/Phase D) obsolete ; line numbers indicatifs | README, ALGORITHMS Sec. 18/19, PAPER_ROADMAP | **rewrite** |
| docs/CONSERVATION_SUMMARY.md | Canonique : conservation FV/Schur mesuree vs Hoffart FE | contributor | active | `adc_cases/.../test_schur_conservation.py` (#207) | Aucun interne ; autorite cross-repo non verifiable depuis adc_cpp | FULL_MODEL_VALIDATION PR2, HOFFART_FIDELITY | **keep** |
| docs/BIBLIOGRAPHY.md | Canonique : references (Hoffart arXiv:2510.11808, etc.) | new user | active | litterature externe | Aucun | citations ancrees ailleurs | **keep** |
| docs/HOFFART_FIDELITY.md | Cross-check cas hoffart vs papier (par-aspect) | contributor | active | `adc_cases` model.py/run.py + engine | Ancres cross-repo non verifiables depuis adc_cpp ; labels non_verifie honnetes | CONSERVATION_SUMMARY, HOFFART_STEP_SEQUENCE | **keep** |
| docs/CODEBASE_AUDIT.md | Audit maintenabilite par-fichier (2026-06-06) | internal-dev | active | `include/adc/**` + `python/*.cpp` | L'inventaire le plus exact (note bien amr_coupler RETIRE) ; legerement en retard sur AmrSystem multi-bloc | COUPLING_SURFACE, ARCH module map | **keep** |
| docs/DOC_CLEANUP_PLAN.md | Plan interne : overlaps + SoT par doc | internal-dev | active | self (meta) | "BACKEND_COVERAGE a creer" existe ; README dit 373 l. (reel 208) ; partiellement execute | indexe les autres | **archive** |
| docs/COUPLER_HIERARCHY.md | Reference : chaque coupler de `coupling/` | contributor | active | `coupling/*.hpp` | Sec. 4 decrit `amr_coupler.hpp` SUPPRIME comme existant (l.152, table l.424) | ARCH Sec. 5/Sec. 8, COUPLING_SURFACE | **rewrite** |
| docs/COUPLING_SURFACE.md | SoT classification PUBLIC/INTERNAL/DEPRECATED couplers | contributor | active | `coupling/*.hpp` | l.118 liste `amr_coupler.hpp` comme conserve-historique : FAUX (supprime) | COUPLER_HIERARCHY | **rewrite** |
| docs/PERFORMANCE.md | Mesures perf historiques (CS:APP, M1) | contributor | historical | `adc_cases` benches + `coupling_policy.hpp` | Banniere "historique" OK ; cite `examples/bench_amr.cpp`, `scripts/plot_bench_scaling.py` ABSENTS d'adc_cpp | PROFILE_RESULTS, GPU_RUNTIME_PORT ph11 | **archive** |
| docs/PROFILE_RESULTS.md | Profilage mesure (2026-06-06), MG domine 96-99.9% | internal-dev | active | `bench/profile_step` | Aucun ; colonnes GH200/CUDA ROMEO-only (honnete) | PERFORMANCE (ancien), RESEARCH_BACKLOG | **keep** |
| docs/GPU_RUNTIME_PORT.md | Journal validation GH200 phases 1-11 | internal-dev | active | harnais `python/tests/gpu/` (hors ctest) | Aucun materiel | README Validation, BACKEND_COVERAGE Sec. 3, GPU_ROMEO | **keep** |
| docs/GPU_ROMEO.md | Recette verif GPU brique DSL sur GH200 (maxdiff=0) | internal-dev | active | `gen_cuda_harness.py` + `romeo_run.sh` | Aucun ; recette etroite | GPU_RUNTIME_PORT, BACKEND_COVERAGE Sec. 3 | **keep** |
| docs/PAPER_ROADMAP.md | Roadmap science (Hoffart, O5, ring-edge lock) | internal-dev | active | cas diocotron + todo Sec. 6 | Chiffres taux croissance ANCIENS (Minmod-o2 n=192) supersedes | FULL_MODEL_VALIDATION, HOFFART_FIDELITY, todo Sec. 6 | **keep** |
| docs/FULL_MODEL_VALIDATION_ROADMAP.md | Roadmap repro modele complet ; SUPERSEDE PAPER_ROADMAP | internal-dev | active | HOFFART_FIDELITY + HOFFART_STEP_SEQUENCE + run.py | Auto-corrige PAPER_ROADMAP ; PR2 (#208) faite ; doc science la plus a jour | PAPER_ROADMAP, HOFFART_FIDELITY | **keep** |
| docs/HOFFART_STEP_SEQUENCE.md | Ordre exact macro-step master (Lie pas Strang) | contributor | active | `system_stepper.hpp`, `amr_runtime.hpp`, etc. | Aucun apparent ; ancres file:line specifiques | HOFFART_FIDELITY, CONSERVATION_SUMMARY | **keep** |
| docs/CODE_DOCUMENTATION_CONVENTION.md | Convention doc code (Doxygen/PEP257) | internal-dev | active | self (style) | Aucun (style guide) | -- | **keep** |
| docs/SYSTEM_CPP_EXTRACTION_PLAN.md | Plan split `system.cpp` (NativeLoader) | internal-dev | historical | `python/system.cpp` | Partiellement execute (#151, native_loader livre) | CODEBASE_AUDIT | **archive** |
| todo.md | TODO interne vivant (etat session, log PR) | internal-dev | active | git history + master | Partiellement stale (l.579 "6/6" vs header "7/7") ; 65 KB | PAPER_ROADMAP Sec. 6, RESEARCH_BACKLOG | **keep** |
| docs/RESEARCH_BACKLOG.md | Backlog recherche non-auto-completable | internal-dev | active | `schur_condensation.hpp` + ROMEO | Aucun materiel | PAPER_ROADMAP, FULL_MODEL_VALIDATION | **keep** |
| docs/archive/README.md | Index archive ("hors-archive fait foi") | internal-dev | archive | self | Aucun (disclaim correct) | -- | **keep** |
| docs/archive/ROADMAP.md | Roadmap archivee | internal-dev | archive | superseded ARCH/todo | Historique (placement archive) | todo, PAPER_ROADMAP | **archive** |
| docs/archive/TODO.md | TODO archive chantier multi-especes | internal-dev | archive | superseded todo.md | Historique | todo.md | **archive** |
| docs/archive/ETAT_DES_LIEUX.md | Synthese 6 audits archivee | internal-dev | archive | superseded CODEBASE_AUDIT | Historique, file:line stale | CODEBASE_AUDIT | **archive** |
| docs/archive/ARCHITECTURE_CIBLE.md | Vision north-star archivee | internal-dev | archive | superseded ARCHITECTURE | Aspirationnel (cas "desire vs implemente") | ARCHITECTURE | **archive** |
| docs/archive/DESIGN_MULTISPECIES.md | Design multi-especes archive | internal-dev | archive | superseded ARCH Sec. 5 | Multi-especes livre (SystemCoupler) | ARCH Sec. 5 | **archive** |
| docs/archive/PLAN_VARIABLES_EPM.md | Plan Variables + EPM archive | internal-dev | archive | superseded ALGORITHMS Sec. 11 | Plan execute (eps/Helmholtz/aniso) | ALGORITHMS Sec. 11 | **archive** |
| docs/archive/HERO_RUN_AMR.md | Design hero-run AMR distribue | internal-dev | archive | superseded GPU_RUNTIME_PORT ph10/11 | Historique | GPU_RUNTIME_PORT, archive/ROMEO | **archive** |
| docs/archive/ROMEO.md | Journal runs ROMEO archive | internal-dev | archive | superseded GPU_ROMEO/GPU_RUNTIME_PORT | Log historique | GPU_ROMEO, GPU_RUNTIME_PORT | **archive** |
| docs/archive/DIOCOTRON_GROWTH_RATE.md | Note diocotron growth-rate archivee | internal-dev | archive | adc_cases diocotron + HOFFART_FIDELITY | Chiffres pre-O5 ; contenu vit dans adc_cases | HOFFART_FIDELITY, PAPER_ROADMAP | **archive** |
| docs/archive/two_fluid_ap.md | Note methode two-fluid AP archivee | internal-dev | archive | `adc_cases/two_fluid_ap/` | Correctement archivee (AP a quitte le coeur) | BIBLIOGRAPHY, RESEARCH_BACKLOG | **archive** |

### adc_cpp -- Sphinx (docs/sphinx/)

| Fichier | Role | Public | Statut | Source de verite | Obsolete/Faux | Doublons | Decision |
|---|---|---|---|---|---|---|---|
| docs/sphinx/conf.py | Config Sphinx (Furo, autodoc, inject sys.path, setup_gallery) | internal-dev | active | `_copy_tutorials.py`, `docs.yml` | l.21 `setup_gallery()` mort (tutorials/ supprime) ; l.59 exclut `_generated/tutorials/README.md` | -- | **rewrite** |
| docs/sphinx/_copy_tutorials.py | Helper galerie (copie tutorials/ -> _generated) | nobody | active | src `../../tutorials` (is_dir=False) | Module entierement mort (early-return 0) | -- | **delete** |
| docs/sphinx/index.md | master_doc, 3 toctrees | new user | active | installation/quickstart/examples/api.md | Toctree Tutoriels = stub dead-end ; nav plate vs cible 7-sections | -- | **rewrite** |
| docs/sphinx/tutorials_index.md | Stub "tutos partis vers adc_cases" | new user | active | git 194c63f~1 (ancien toctree) | Dead-end navigationnel, pas de toctree | examples.md | **merge** |
| docs/sphinx/api.md | Reference API Python (autodoc, 20 symboles) | new user | active | `build-py/python/adc/__init__.py` | Aucun (20 symboles resolus) | -- | **keep** |
| docs/sphinx/installation.md | Build C++/Python + backends | new user | active | `CMakeLists.txt` | "71 ctests" (l.17) a reverifier | BACKEND_COVERAGE (intro) | **keep** |
| docs/sphinx/quickstart.md | Recettes Python annotees | new user | active | `adc/__init__.py` | Non execute (pas d'env local) ; API plausible | installation, api | **keep** |
| docs/sphinx/examples.md | Redirige vers adc_cases | new user | active | repo (pas d'examples/ ni scripts/) | Liens cross-doc = URLs GitHub, pas pages Sphinx | tutorials_index | **merge** |
| docs/sphinx/requirements.txt | Deps doc (sphinx, furo, myst, autodoc-typehints) | contributor | active | `conf.py` extensions | `sphinx-autodoc-typehints` jamais active ; numpy MANQUANT | -- | **keep** (corriger) |
| docs/sphinx/_generated/tutorials/ | Sortie galerie generee (00..08 + README) | nobody | generated | `.gitignore` | Perime mai 30, non regenere, gitignore/non-tracke | -- | **delete** |
| docs/sphinx/__pycache__/ | Cache bytecode | nobody | generated | `.gitignore` | Stale, gitignore | -- | **delete** |
| docs/_build/ | Sortie HTML Sphinx + Doxygen | nobody | generated | `.gitignore` | Perime mai 30 (galerie ancienne trompeuse), gitignore/non-tracke | -- | **delete** |

### adc_cpp -- build & racine

| Fichier | Role | Public | Statut | Source de verite | Obsolete/Faux | Doublons | Decision |
|---|---|---|---|---|---|---|---|
| CMakeLists.txt | Build root (options, C++23/C++20 Kokkos) | contributor | active | self (autoritatif options) | Ne declare PAS `ADC_USE_EIGEN` malgre README+CI | -- | **keep** |
| python/CMakeLists.txt | Build module pybind11 `_adc` | contributor | active | self + `python/tests/` | Aucun | -- | **keep** |

### adc_cases

| Fichier | Role | Public | Statut | Source de verite | Obsolete/Faux | Doublons | Decision |
|---|---|---|---|---|---|---|---|
| README.md (top-level) | Portail cas + table "Les cas" | new user | active | `cases_manifest.toml` | Table l.109-119 INCOMPLETE : omet 6 cas DSL | cases_manifest.toml | **rewrite** |
| cases_manifest.toml | Manifeste 16 entrees (categorie/ci/needs) | contributor | active | self | A jour ; SoT du scope | README table | **keep** |
| diocotron/README.md | Cas diocotron reduit (honnete "normalisation") | new user | active | run.py + NORMALIZATION.md | Aucun (disclaim correct) | manifest, NORMALIZATION | **keep** |
| hoffart_euler_poisson_dsl/README.md | Cas modele complet (reproduction-candidate) | new user | active | run.py + results.py + measurement_record | Outputs section implique modes 3/4/5 (seul mode_3 sur disque) | HOFFART_FIDELITY | **keep** |
| magnetic_isothermal_dsl/README.md | Cas DSL aux B_z (parite inter-backend) | new user | active | run.py | Caveat plateforme macOS=aot only (honnete) | -- | **keep** |
| schur_magnetized_cartesian/README.md | Cas timing Schur vs explicite | new user | active | run.py | Docstring cite `adc.Split` mais code = hook prive `sim._s.set_source_stage` | -- | **rewrite** |

> Note couverture README : seuls **4/15** dossiers de cas ont un README (diocotron, hoffart_euler_poisson_dsl, magnetic_isothermal_dsl, schur_magnetized_cartesian). Les 11 autres reposent sur des docstrings de module (detailles et exacts). Decision Phase 2 a prendre : generer des README par cas ou canoniser docstrings + table top-level.

---

## 3. Claims faux/obsoletes confirmes

Fusion `claim_findings` + passe adverse. Colonne Verdict : verdict de l'inventaire ; statut adverse en gras (CONFIRME = verifie independamment, NON-CONFIRME = pas re-verifie).

| Claim | Doc source | Verdict | Preuve |
|---|---|---|---|
| AmrSystem est mono-bloc / explicite-only / sans Roe / sans recon primitive / PAS a parite avec System | ARCHITECTURE Sec. 8 (l.26,379-382,547), README:130-132, DSL_MODEL_DESIGN Sec. 0bis/Sec. 5/Phase D | **obsolete -- CONFIRME (haute)** | `amr_system.cpp:202 multi_block()`, `:208 build_multi()`, `:300 if(multi_block())`, parse `recon in {conservative,primitive}` `:343-345`, `riemann in {rusanov,hllc,roe}` `:135`, IMEX `:340-346` ; 7 tests capstone (`test_amr_system_twoblock`, `test_amr_multiblock_*`) ; regrid union #199. Aucun throw "un seul bloc". |
| `m.u[0]`/`m.a.grad_x` pour acceder vars conservatives/aux dans dsl.Model | DSL_API.md Sec. 1 (l.21-22,29) | **faux -- CONFIRME (haute)** | `dsl.Model` n'a ni `.u` ni `.a` ; API reelle = `conservative_vars(...)->Vars` (`dsl.py:1543`), aux via `m.aux('grad_x')` (`:1585`). Crashe en AttributeError. |
| Runtime params : `param(kind='runtime')` leve NotImplementedError (Phase E) | DSL_API.md Sec. 5 (l.104-106) **ET** `dsl.py:1623/1627` (docstring+commentaire) | **faux -- CONFIRME (haute)** | `Param.__init__` (`dsl.py:1420-1432`) implemente kind='runtime' (`RuntimeParamRef`), leve seulement ValueError pour kinds inconnus. `test_dsl_runtime_params.py` compile+run+`set_block_params` end-to-end. **Le docstring source propagerait le faux claim via autodoc.** |
| `sim.set_poisson("geometric_mg")` (positionnel) | DSL_API.md Sec. 3 (l.70) | **faux -- CONFIRME (haute)** | 1er arg positionnel = `rhs` (`bindings.cpp:153`), valide  in  {charge_density,composite} (`system_field_solver.hpp:197-199`) -> throw "rhs inconnu". Correct : `set_poisson(solver='geometric_mg')`. |
| Galerie Tutoriels cablee dans un toctree (pages atteignables) | sphinx/tutorials_index.md, index.md | **faux -- CONFIRME (haute)** | Aucun `.md` ne reference `_generated/tutorials/` ; `tutorials_index.md` = stub prose sans toctree ; `index.md` toctree Tutoriels ne liste que `tutorials_index`. Pre-194c63f avait un vrai toctree. Build propre -> section vide. |
| `setup_gallery()` copie `tutorials/*.md` -> `_generated/tutorials/` | sphinx/_copy_tutorials.py (+conf.py:21) | **obsolete -- CONFIRME (haute)** | `tutorials/` supprime commit 194c63f ; `_copy_tutorials.py:21-22` early-return 0 (is_dir=False). Copie 0 fichier. Build propre -> galerie vide. |
| Construire les docs produit une galerie de tutoriels peuplee | sphinx/index.md | **faux -- CONFIRME (haute)** | Consequence des deux ci-dessus. `_build/` mai 30 montre une galerie qui ne reflete plus les sources. (Adverse rectifie : `_build/`/`_generated/` ne sont PAS committes -- gitignores/non-trackes -- artefacts locaux seulement.) |
| Limiteur de reconstruction "MC" (monotonized-central) existe | ALGORITHMS:97, ARCHITECTURE:288/509 | **faux -- CONFIRME (medium)** | `reconstruction.hpp` ne definit que `NoSlope:35`/`Minmod:46`/`VanLeer:61`/`Weno5:124`. `grep 'struct MC|MonotonizedCentral'` -> aucun. `limiter='mc'` echouerait. |
| `amr_coupler.hpp`/`AmrCoupler` = fichier deprecie conserve | ARCHITECTURE:33, COUPLING_SURFACE:118, COUPLER_HIERARCHY Sec. 4 | **faux -- CONFIRME (medium)** | `ls include/adc/coupling/amr_coupler.hpp` -> absent (supprime #164). Seul `AmrCouplerMP` existe. CODEBASE_AUDIT:210 enregistre RETIRE ; les 3 autres docs sur-decrivent un symbole mort. |
| `AmrSystem(n=128, max_level=2, periodic=True)` | DSL_API.md Sec. 3 (l.76) | **faux -- CONFIRME (medium)** | `AmrSystemConfig` n'a PAS de champ `max_level` (`bindings.cpp:259-266` : n,L,regrid_every,periodic,distribute_coarse,coarse_max_grid). `setattr(config,'max_level',...)` echoue. |
| Pour backend='production'/target='amr_system' la facade Python AMR rejette HLLC/Roe/primitif (mono-bloc) | DSL_API.md Sec. 3 (l.86-88) | **obsolete -- CONFIRME (medium)** | `AmrSystem.add_equation` (`__init__.py:1321-1334`) cable `recon='primitive'` et flux roe/hllc via `dispatch_amr_compiled` ; `test_amr_riemann_native.cpp` existe. |
| Option CMake `ADC_USE_EIGEN` (defaut ON, cible `adc_eigen`) | README:167 (+ CI ci.yml:145, docs.yml:45) | **faux -- CONFIRME (medium)** | Aucun `option(ADC_USE_EIGEN)`, aucune cible `adc_eigen`, aucun `find_package(Eigen)`. `-DADC_USE_EIGEN` = no-op silencieux. |
| `python -c "...import adc..."` (et fallback python3) importe le module | (commande de prompt) | **faux -- CONFIRME (haute)** | `python` introuvable ; `python3`=3.9.6 -> `ModuleNotFoundError: adc._adc` (`.so` cpython-312). Importe propre avec `/opt/homebrew/anaconda3/bin/python3.12` + numpy. |
| ARCHITECTURE Sec. 11 : 71 ctests coeur + 21 MPI + 26 Python | ARCHITECTURE:442-444 | **obsolete -- CONFIRME (medium)** | Disque : ~90-94 `adc_add_test` (non-MPI, dont elliptic_interface), ~51 entrees MPI, 60 fichiers Python. Toutes sous-estimees. |
| BACKEND_COVERAGE base : 84 `adc_add_test` + 19 `add_executable` + 55 Python | BACKEND_COVERAGE:322-324 | **partiellement vrai -- CONFIRME (medium)** | Reel : 90 `adc_add_test` + 19 runtime = 109 non-MPI ; bloc MPI 12 add_executable -> ~51 entrees ; 60 Python. Totaux d'en-tete perimes, lignes per-test OK. |
| FFT Poisson fonctionne sous MPI | ARCHITECTURE Sec. 7 | **faux -- CONFIRME (medium)** | `test_mpi_system_fft.cpp` est un lock de NON-regression : la voie FFT est REFUSEE sous MPI (n_ranks>1) avec erreur collective (fix SIGSEGV #93). FFT = single-rank par design. |
| `schur_magnetized_cartesian` utilise `adc.Split(Explicit, CondensedSchur)` | schur_magnetized_cartesian/run.py:130-146 | **obsolete -- CONFIRME (medium)** | Le code appelle le hook PRIVE `sim._s.set_source_stage(...)` (`run.py:142`) car l'ABI AOT ne transporte pas SSPRK3 (README:68-71). Meme `CondensedSchurSourceStepper` C++ sous-jacent. |
| Suite Python tournee sous pytest | (implicite docs) | **faux -- CONFIRME (medium)** | `ci.yml:149` : `find python/tests -maxdepth 1 -name 'test_*.py' | xargs python3`. Pas de pytest ; tests = scripts standalone avec assert+exit. |
| README top-level adc_cases "Les cas" = set complet | adc_cases/README.md:107-119 | **partiellement vrai -- CONFIRME (medium)** | Table liste 10 cas, omet diocotron_dsl, two_species_dsl, magnetic_isothermal_dsl, dsl_euler, hoffart_euler_poisson_dsl, schur_magnetized_cartesian (presents dans manifest). |
| diocotron figures "produites dans diocotron/figures/" | diocotron/run.py:13, README:36 | **faux -- CONFIRME (haute)** | `run.py:207` ecrit dans `out/diocotron/` (gitignore) ; tous les savefig (`:246,257,276,286`) y vont. Aucun script n'ecrit `figures/`. Les 4 figures committees sont des copies manuelles perimees. |
| Chaque run emet un record capturant adc_cpp SHA + adc_cases SHA | hoffart_euler_poisson_dsl/README:143 | **partiellement vrai -- CONFIRME (haute)** | `run.py:422-423` ecrit les SHA, MAIS les `metadata.json` sur disque n'ont PAS les cles `adc_cpp_sha`/`adc_cases_sha` (artefacts d'un run anterieur) ; les 4 figures diocotron n'ont aucun metadata. |
| "Reproduction du benchmark diocotron de Hoffart" (titre) | diocotron/run.py:2-11 | **partiellement vrai -- CONFIRME (basse)** | Le meme docstring + README:8-10 qualifie "modele reduit ExB, PAS reproduction Euler-Poisson complet". Reproduit l'oracle analytique a 3 chiffres ; taux FV sous-predits -22/-27/-5%. Mot "Reproduction" desambiguise partout. |
| AOT two-fluid AP valide par un test coeur | tests/test_ap_limit.cpp | **partiellement vrai -- CONFIRME (medium)** | `test_ap_limit.cpp`/`test_imex_ap.cpp` valident la PROPRIETE AP sur un modele TOY scalaire (du/dt=(u_eq-u)/eps), pas le vrai two-fluid. Le vrai integrateur vit dans `adc_cases/two_fluid_ap/` (scenario, pas brique). |
| CUDA mono-GPU / multi-GPU valide par test automatique | BACKEND_COVERAGE | **partiellement vrai -- CONFIRME (haute)** | La CI ne build jamais `Kokkos_ENABLE_CUDA=ON`. Validation device = ROMEO-manuel via `python/tests/gpu/*.cpp` (exclus du glob CI `-maxdepth 1`). Dire "manuellement sur GH200". |
| DSL production/hybride valide sur device Kokkos Cuda | GPU_RUNTIME_PORT | **partiellement vrai -- CONFIRME (haute)** | ROMEO-only ; `test_compiled_model_parity` PAS device-valide (segfault nvcc extended-lambda cross-TU). Workaround foncteur nomme non porte sur ce test. |
| Toutes les sorties hoffart ecrites "pour chaque mode" 3/4/5 | hoffart_euler_poisson_dsl/README:192-200 | **partiellement vrai -- CONFIRME (basse)** | Sur disque seul `mode_3/` existe pour les deux engines ; mode_4/5 absents (table marque PENDING). |

**Non-confirmes / a revoir (data mince) :** la verification adverse n'a couvert que ~7 claims. Pour ALGORITHMS ("tous chemins/tests existent"), les concepts EllipticSolver/PhysicalModel, FFT box-unique, WENO5-Z Borges, GeometricMG eps/reaction/aniso, le timing `schur_magnetized_cartesian` (562x/1000x) et la matrice ROMEO multi-GPU "?" : verdicts **vrai/non-verifiable** issus de l'inventaire seul, a confirmer en Phase 2 par execution/grep cible.

---

## 4. API reelle a documenter (autodoc)

Surface publique reelle issue des `api_entries`. Symboles Python = `python/adc/__init__.py` + `dsl.py` + `integrate.py`, bindings dans `python/bindings.cpp` (seul fichier de `.def()` -- `system.cpp`/`amr_system.cpp` sont du C++ pur sans pybind). Concepts/classes C++ = `include/adc/**`. `doc_status` : documented / wrong-docstring / unverified.

### C++ -- concepts & classes coeur (graine Reference C++ / Doxygen)

| Symbole | Module | doc_status | Test preuve |
|---|---|---|---|
| `PhysicalModel` (concept) | `core/physical_model.hpp` | documented (mais ARCH:102 sur-liste `wave_speeds` comme requis) | `test_spatial_discretisation.cpp` |
| `HasPrimitiveVars`/`HyperbolicPhysicalModel` | `core/physical_model.hpp` | documented | `test_primitive_recon.cpp` |
| `EllipticSolver` (concept) | `numerics/elliptic/elliptic_solver.hpp` | documented | `test_elliptic_interface.cpp` |
| `CompositeModel` | `physics/composite.hpp` | documented | `test_amr_composite_source_conservation.cpp` |
| `RusanovFlux`/`HLLFlux`/`HLLCFlux`/`RoeFlux` | `numerics/numerical_flux.hpp` | documented | `test_roe_flux.cpp` |
| `NoSlope`/`Minmod`/`VanLeer`/`Weno5` | `numerics/reconstruction.hpp` | **wrong-docstring** (pas de `MC`) | `test_weno_convergence.cpp` |
| `EquationBlock` | `core/equation_block.hpp` | documented | `test_system_abstraction.cpp` |
| `CoupledSystem` | `core/coupled_system.hpp` | documented | `test_two_species_minimal.cpp` |
| `GeometricMG` (set_epsilon/set_reaction/set_epsilon_anisotropic/solve_robust) | `numerics/elliptic/geometric_mg.hpp` | documented | `test_geometric_mg.cpp` |
| `PoissonFFTSolver` (mono-rank/mono-box, throw si n_ranks!=1) | `numerics/elliptic/poisson_fft_solver.hpp` | documented | `test_poisson_fft.cpp` |
| `DistributedFFTSolver` (band MPI_Alltoall) | `numerics/elliptic/poisson_fft_solver.hpp` | documented | `test_mpi_fft_distributed.cpp` |
| `Coupler<Model,Elliptic>` | `coupling/coupler.hpp` | documented | `test_mpi_coupler_inject.cpp` |
| `SystemCoupler`/`SystemDriver`/`SystemAssembler` | `coupling/system_coupler.hpp` | documented | `test_assembler_driver.cpp` |
| `AmrCouplerMP` (legacy `AmrCoupler` SUPPRIME) | `coupling/amr_coupler_mp.hpp` | documented | `test_mpi_amr_compiled_parity.cpp` |
| `AmrSystemCoupler` | `coupling/amr_system_coupler.hpp` | documented | `test_amr_system_coupler.cpp` |
| `System` (pimpl SystemStepper/FieldSolver/BlockStore) | `runtime/system.hpp` | documented | `python/tests/test_bindings.py` |
| `AmrSystem` (mono- OU multi-bloc, recon/riemann/imex) | `runtime/amr_system.hpp` | **wrong-docstring** (docs disent mono-bloc) | `test_amr_system_twoblock.cpp` |
| `advance_amr` (defini `amr_advance.hpp`, inclus par `amr_reflux_mf.hpp`) | `numerics/time/amr_advance.hpp` | documented | `test_amr_compiled_model.cpp` |
| `for_each_cell` (seam Kokkos/OpenMP/serial) | `mesh/for_each.hpp` | documented | `test_reduce.cpp` |
| `comm` (my_rank/n_ranks/all_reduce_*/barrier) | `parallel/comm.hpp` | documented | `test_mpi_array_reduce.cpp` |
| `ModelSpec`/`model_factory` (dispatch_model) | `runtime/model_factory.hpp` | documented | `python/tests/test_dsl_compose.py` |
| Briques `ExBVelocity`/`IsothermalFlux`/`PotentialForce`/`GravityForce`/`ChargeDensity`/`BackgroundDensity`/`NoSource` | `physics/{hyperbolic,source,elliptic}.hpp` | documented (ARCH dit "ExB" au lieu de `ExBVelocity`) | `test_coupled_source.cpp` |

### Python -- `adc.System` (facade composition mono-niveau)

| Symbole | Signature | doc_status | Test preuve |
|---|---|---|---|
| `System(config, mesh, **cfg_kw)` | constructeur | documented | `test_bindings.py` |
| `add_block` / `add_equation` / `add_background` | composition de blocs (add_equation type-dispatch ModelSpec vs CompiledModel) | documented | `test_bindings.py`, `test_dsl_production.py`, `test_dsl_dynamic.py` |
| `add_coupling` / `add_coupled_source` / `add_ionization` / `add_collision` / `add_thermal_exchange` | couplages inter-especes | documented | `test_dsl_coupled_source.py` |
| `add_elliptic_model` | EPM (operator/rhs/output) | documented | `test_poisson_eps.py` |
| `set_poisson(rhs,solver,bc,wall,wall_radius,epsilon)` | config Poisson | documented | `test_bindings.py` |
| `set_density` / `set_primitive_state` / `get_primitive_state` / `set_block_params` | I/O etat | documented | `test_bindings.py`, `test_primitive_state.py`, `test_dsl_runtime_params.py` |
| `set_magnetic_field` / `set_epsilon_field` / `set_epsilon_anisotropic_field` | champs aux etendus | documented | `test_magnetic_field.py`, `test_poisson_eps_aniso.py` |
| `set_source_stage` / `set_time_scheme` | stage source (Schur) / schema temps | documented | `test_schur_via_system.py`, `test_strang_split.py` |
| `step` / `advance` / `step_cfl` / `step_adaptive` / `run` | avancement | documented | `test_bindings.py`, `test_stride.py` |
| `solve_fields` / `eval_rhs` / `get_state` / `set_state` | champs/RHS | documented | `test_weno5_ssprk3.py` |
| `set_disc_domain` / `disc_mask` | domaine disque | documented | `test_disc_domain_mask.py` |
| `block_names`/`mass`/`density`/`potential`/`nx`/`ny`/`time`/`n_species`/`n_vars`/`variable_names`/`variable_roles`/`block_gamma`/`abi_key` | introspection | documented | `test_bindings.py`, `test_dsl_roles.py`, `test_dsl_abi_metadata.py` |

### Python -- `adc.AmrSystem` (facade AMR ; multi-bloc reel)

| Symbole | doc_status | Test preuve |
|---|---|---|
| `AmrSystem(config, **cfg_kw)`, `add_block`, `add_equation` | documented | `test_amr_multiblock.py`, `test_dsl_production_amr.py` |
| `set_refinement` / `set_phi_refinement` / `set_poisson` | documented | `test_amr_multiblock.py` |
| `AmrSystemConfig` (n,L,regrid_every,periodic,distribute_coarse,coarse_max_grid -- **PAS de max_level**) | documented | `test_amr_multiblock.py` |
| `SystemConfig` (n,L,periodic,geometry,nr,ntheta,r_min,r_max) | documented | `test_polar_system.py` |

### Python -- briques de composition (`adc.Model(...)`)

| Symbole | doc_status | Test preuve |
|---|---|---|
| `Model(state,transport,source,elliptic)->ModelSpec`, `ModelSpec` | documented | `test_bindings.py` |
| `CompositeModel(...)->dsl.HybridModel` | documented | `test_dsl_hybrid.py` |
| `Scalar`/`FluidState`, `ExB`/`CompressibleFlux`/`IsothermalFlux` | documented | `test_bindings.py` |
| `NoSource`/`PotentialForce`/`GravityForce`, `ChargeDensity`/`BackgroundDensity`/`GravityCoupling` | documented | `test_bindings.py` |
| `Spatial(limiter,flux,recon)` / `FiniteVolume(...)` | documented | `test_weno5_compiledmodel.py`, `test_dsl_recon.py` |
| `Explicit` / `IMEX` / `SourceImplicit` / `Implicit` (**OBSOLETE, DeprecationWarning -> IMEX**) | documented | `test_stride.py`, `test_time_policy.py` |
| `Split` / `Strang` / `CondensedSchur` (descripteurs role/field hardcodes C++) | documented | `test_strang_split.py`, `test_schur_via_system.py` |
| `Role`, `CartesianMesh`/`PolarMesh`, `Ionization`/`Collision`/`ThermalExchange` | documented | `test_dsl_roles.py`, `test_polar_system.py`, `test_dsl_coupled.py` |
| EPM : `elliptic`/`div_eps_grad`/`charge_density`/`composite_rhs`/`electric_field_from_potential`, `EllipticSolver`/`EllipticModel`/`DivEpsGrad`/`CompositeRhs`/`ChargeDensitySource` | documented | `test_poisson_eps.py`, `test_poisson_composite.py` |
| `PythonFlux` (backend host interprete) | documented | `test_dsl.py` |

### Python -- `adc.dsl` (DSL symbolique) & `adc.integrate`

| Symbole | doc_status | Test preuve |
|---|---|---|
| `dsl.Model` (conservative_vars/primitive/aux/flux/eigenvalues/source/elliptic_rhs/param/compile) | documented (mais docstring `param` runtime = **wrong**) | `test_dsl_phase_a.py`, `test_dsl_production.py` |
| `dsl.Model.compile(backend='aot'|'production'|'prototype', target='system'|'amr_system', ...)` | documented (defaut reel = `aot`) | `test_dsl_compile_facade.py`, `test_dsl_compile_cache.py` |
| `dsl.CompiledModel` (backend/adder/so_path/caps/abi_key/runtime_param_*) | documented | `test_dsl_compile_facade.py` |
| `dsl.HyperbolicModel`, `dsl.Param`/`RuntimeParam`, `dsl.sqrt`/`Expr` | documented (Param: voir wrong-docstring) | `test_dsl_codegen.py`, `test_dsl_runtime_params.py`, `test_dsl_cse.py` |
| `dsl.HyperbolicBrick`/`SourceBrick`/`EllipticBrick`, `dsl.HybridModel`, `dsl.NativeBrick`/`CompiledBrick` | documented | `test_dsl_brick.py`, `test_dsl_hybrid*.py` |
| `dsl.CoupledSource`/`CompiledCoupledSource` (bytecode couplings) | documented | `test_dsl_coupled_source*.py` |
| `integrate.euler_step` / `ssprk2_step` | documented | `test_dsl.py` |
| `abi_key` (module-level) | documented | `test_dsl_abi_metadata.py` |

**Matrice capacites backend (load-bearing, vit dans `_BACKEND_CAPS` `dsl.py:1382-1393`) :** prototype/aot = CPU-only, no MPI/AMR/GPU ; production = CPU+MPI+AMR, **gpu=False cote Python** (par prudence, malgre device-clean C++). Restrictions appliquees comme `ValueError` explicites dans `add_equation` (stride>1 sur aot ; evolve=False sur prototype/aot ; weno5 sur prototype ; implicit_vars sur .so compile ; HLLC/Roe necessite un primitif 'p'). Toute doc "feature X marche sur backend Y" doit matcher ces guards.

---

## 5. Cas adc_cases -- verite par cas

15 dossiers de cas (le manifest a 16 entrees car diocotron a 2 scripts : `run.py` + `band_instability.py`). Modele : natif (briques C++ composees) / dsl (formules symboliques compilees) / hybride / interprete / bespoke-C++. CI/categorie = `cases_manifest.toml`.

| Cas | README ? | Modele | Schema temps | Schema espace | Poisson | Backends reels | CI/categorie | Assets | Limites honnetes |
|---|---|---|---|---|---|---|---|---|---|
| composition | non | natif (electron_euler + ion_isothermal + diocotron) | IMEX(10) e- / Explicit ions ; Part D = SSPRK2 Python | per-bloc vanleer/hllc + minmod/rusanov | charge_density, geometric_mg | adc compile | true / tutoriel | aucun (prints) | Aucun claim physique ; teste compo heterogene + determinisme bit |
| custom_scheme | non | numpy pur (ExB+upwind) ; adc = oracle Poisson seul | SSPRK2 Python, dt=CFL*dx/v | central + upwind numpy | charge_density, geometric_mg (seul appel lib) | adc (Poisson only) | true / tutoriel | aucun | Scheme entier en Python ; lib = solveur elliptique |
| **diocotron** | **oui** | **natif REDUIT ExB scalaire** (Scalar+ExB+BackgroundDensity) -- **PAS Euler-Poisson complet** | Explicit SSPRK2 (CFL=0.4) | MUSCL minmod + Rusanov | dirichlet, wall=circle R=0.40 | adc + matplotlib | run.py false/**reproduction** ; band_instability true/validation | figures/ committees (dispersion/amplitude/snapshots/diocotron.gif) | **HONNETE** : README+docstring disent "normalisation modele reduit, PAS repro systeme complet". Oracle Petri a 3 chiffres ; FV sous-predit -22/-27/-5% |
| diocotron_amr | non | natif diocotron sur AmrSystem | Explicit (CFL=0.4) | NoSlope + Rusanov o1 | charge_density, geometric_mg periodic | adc (AmrSystem) | true / validation | aucun | Multi-patch Berger-Rigoutsos + reflux ; teste raffinement reel vs controle |
| diocotron_dsl | non | dsl (formules == ExBVelocity+BackgroundDensity) | Explicit SSPRK2 (== natif) | FiniteVolume(minmod, rusanov) | charge_density, geometric_mg | dsl production->aot fallback ; needs cxx | true / validation (needs cxx) | out/*.so | Claim : etat **bit-identique** au natif (np.array_equal) |
| dsl_euler | non | dsl HyperbolicModel Euler 2D (pas de source/Poisson) | forward Euler, cfl_dt | Rusanov via `PythonFlux` (**interprete numpy**) | aucun | host interprete (PythonFlux) | false / **experimental** | aucun | Prototype INTERPRETE, distinct du chemin compile des autres *_dsl |
| euler_poisson | non | natif euler_poisson (GravityForce + GravityCoupling +/-) | Explicit SSPRK2, 20 pas | vanleer/hllc | charge_density, geometric_mg periodic | adc compile | true / validation | aucun | Contraste signe couplage (attractif dE<0 / repulsif dE>0) ; pas de claim papier |
| **hoffart_euler_poisson_dsl** | **oui** | **dsl Euler-Poisson magnetique COMPLET** (continuite + momentum + Lorentz, barotrope) ; variantes schur/local | system-schur SSPRK3 + CondensedSchur(theta=0.5) ; amr-imex backward-Euler | FiniteVolume WENO5-Z + Rusanov | composite, dirichlet, wall=circle R=16 | dsl production (system/amr_system) ; amr-imex needs Kokkos/MPI | check_model true/validation ; run.py false/**reproduction-candidate** | out/.../{amplitude,snapshots,growth_rates,gifs,metadata} | **HONNETE** : banniere "PAS de repro quantitative etablie" ; table -82 a -95% (n=256/384), n=512 PENDING, amr-imex PENDING. Gaps : Lie pas Strang, Poisson once-per-step, geometrie cart+wall suspecte |
| **magnetic_isothermal_dsl** | **oui** | dsl isotherme magnetise (aux B_z index 3) ; **pas d'oracle natif** | Explicit SSPRK2, 40 pas | FiniteVolume(minmod, rusanov) | charge_density, geometric_mg periodic ; set_magnetic_field | dsl production+aot (macOS=aot only) ; needs cxx | true / validation | out/*.so | Valide SANS oracle natif : parite inter-backend + oracle Lorentz analytique (dmax==0) |
| multispecies | non | natif (e- Euler 4-var + ions isotherme 3-var) couples 1 Poisson | Explicit, dt=0.001 | minmod les deux | charge_density, geometric_mg periodic | adc compile | true / validation | aucun | Conservation masse par espece <1e-9 |
| plasma | non | recipe natif 3-especes (e- HLLC+primitif, ions+neutres isotherme) + ionisation+collision | Explicit step_cfl(0.3) | vanleer/hllc/primitif + minmod | f=q_e n_e+q_i n_i | adc compile | true / validation | aucun | Honnete : transfert momentum/energie des particules creees = simplification |
| **schur_magnetized_cartesian** | **oui** | dsl isotherme magnetise ; variantes local/schur (meme equations que magnetic_isothermal) | transport SSPRK2 ; source explicite OU CondensedSchur via `sim._s.set_source_stage` (hook prive) | FiniteVolume(minmod, rusanov, conservative) | charge_density, geometric_mg periodic ; B_z=omega_c | dsl aot (production echoue dlopen macOS arm64) ; needs cxx | false / **experimental** | out/dt_stable.csv | Etude timing : explicite plateau dt*omega_c~0.3, Schur 178-316 (gain 562x/1000x). Utilise hook prive car `adc.Split` pas cable sur ABI AOT |
| two_euler | non | natif euler x2 gaz independants (NON couples) | multirate step_adaptive(0.4) | vanleer/hllc/primitif | f=0 (juste pour solve_fields) | adc compile | true / validation | aucun | "2 Euler meme code" ; blocs independants |
| two_fluid_ap | non | **bespoke C++** (two_fluid_ap.hpp) AP isotherme ; PAS brique composable | IMEX/AP (terme raide implicite) | FV continuite, dans le JIT | elliptic AP reformule dans le scenario C++ | JIT .so via ctypes (build_shared) ; needs cxx C++20 | true / validation (needs cxx) | out/.../_two_fluid_ap.{so,dylib} | Remplace l'echappatoire `_TwoFluidAP` retiree du coeur ; vit dans adc_cases |
| two_species_dsl | non | dsl x2 (e- Euler 4-var + ions isotherme 3-var) + source electrostatique, 1 Poisson | Explicit SSPRK2 (== natif) | FiniteVolume(minmod, rusanov) | charge_density, geometric_mg periodic | dsl production->aot ; needs cxx | true / validation (needs cxx) | out/*.so | Ions bit-identiques ; e- different ~machine-eps (<1e-24, reassociation float dans RHS Poisson partage, documente) |

**Honnetete diocotron-vs-Hoffart (CONFIRMEE) :** deux modeles physiques DIFFERENTS partagent le nom "diocotron" et la ref arXiv:2510.11808 : (a) le REDUIT ExB scalaire (diocotron, custom_scheme, diocotron_amr, diocotron_dsl) et (b) le COMPLET Euler-Poisson magnetique (hoffart_euler_poisson_dsl). Le manifest classe (a)/run.py "reproduction" (de la normalisation analytique uniquement) et (b) "reproduction-candidate" (table PENDING). Grep repo-wide pour "reproduction complete/full" ne renvoie que des disclaimers. `results.py` (verify_paper_windows, engine_label) **refuse mecaniquement** de melanger les taux 2pi/rho du reduit avec les nombres bruts du complet. Aucun fichier ne ment ; le risque residuel est en prose : le mot "Reproduction" dans le titre diocotron, hors contexte, sur-vend (il reproduit l'oracle analytique, pas le systeme complet, et sous-predit jusqu'a -27%). Recommandation : remplacer par "benchmark de normalisation modele reduit".

---

## 6. Assets & provenance

Tous les assets image vivent sous adc_cases. Le submodule adc_cpp n'est PAS present sur disque (pas de `.gitmodules`, pas de gitlink) -> aucune figure sous `adc_cpp/docs/` ne peut etre inspectee localement.

**Compteurs :**

| Metrique | Valeur |
|---|---|
| Total fichiers image | 11 (4 PNG + GIF trackes ; 7 untracked) |
| Trackes (canoniques) | 4 -- `diocotron/figures/{dispersion,amplitude,snapshots}.png` + `diocotron.gif` |
| Orphelins (references seulement par phrase generique, pas par nom) | 3 -- `mode_3/snapshots.png` (x2 engines), `mode_3/diocotron_l3.gif` |
| Sans provenance (aucun SHA/backend/resolution) | 11/11 |
| Liens image casses (embeds `![]()` / `<img>`) | **0** -- le repo entier ne contient AUCUN embed markdown ; figures referencees par nom en prose/table uniquement |

**Embeds casses :** aucun. Le risque de repro n'est pas le dangling `![]()` mais le **mismatch de chemin** et l'**absence de provenance**.

**Assets untracked (out/, gitignores) -> decision regenerate :**

| Asset | Producteur | Reference par | Decision | Note |
|---|---|---|---|---|
| out/hoffart_..._system_schur/growth_rates.png | run.py:469 | README:200 (par nom) | regenerate | metadata.json sans SHA (run anterieur) |
| out/hoffart_..._system_schur/mode_3/amplitude.png | run.py:320 | README:197 | regenerate | seul mode_3 existe (3/4/5 attendus) |
| out/hoffart_..._system_schur/mode_3/snapshots.png | run.py:339 | **orphelin** ("3x3 panel" generique) | regenerate | pas de SHA |
| out/hoffart_..._system_schur/mode_3/diocotron_l3.gif | run.py:360-361 | **orphelin** ("an animated GIF") | regenerate | nom dynamique mode-indexe |
| out/hoffart_..._amr_imex/growth_rates.png | run.py:469 | README:200 | regenerate | engine experimental, pas de SHA |
| out/hoffart_..._amr_imex/mode_3/amplitude.png | run.py:320 | README:197 | regenerate | experimental |
| out/hoffart_..._amr_imex/mode_3/snapshots.png | run.py:339 | **orphelin** | regenerate | experimental |

**Assets trackes (canoniques) -> decision keep, mais provenance a reconstruire :**

| Asset | Producteur reel | Decision | Note |
|---|---|---|---|
| diocotron/figures/dispersion.png | run.py:246 (ecrit dans `out/diocotron/`, PAS figures/) | keep | 806x559 ; aucun SHA/backend ; copie manuelle |
| diocotron/figures/amplitude.png | run.py:257 (-> out/) | keep | idem |
| diocotron/figures/snapshots.png | run.py:286 (-> out/) | keep | 1690x442 ; aucune provenance |
| diocotron/figures/diocotron.gif | run.py:276 (-> out/) | keep | 420x420 ~512KB ; copie manuelle |

**Decisions de provenance Phase 5 :** (1) repointer les `savefig` de `run.py` vers `figures/` OU documenter explicitement l'etape de copie ; (2) regenerer avec le `run.py` courant pour peupler `adc_cpp_sha`/`adc_cases_sha` puis figer ; (3) decider quelles figures hoffart sont promues (committees) vs laissees ephemeres ; (4) figer les modes exacts (3/4/5) + n + dt + engine + SHA ; (5) ajouter des references par nom de fichier ou un manifeste d'assets pour que les renommages soient detectes par un link-check.

---

## 7. Plan d'integration ordonne (Phases 2-6)

Legende execution : **[L]** entierement faisable en local darwin/arm64 ; **[ROMEO]** necessite le cluster (CUDA/MPI multi-GPU) pour regenerer des assets ou valider ; **[TC]** bloque/conditionne par la toolchain de build doc (interpreteur 3.12 + numpy + compilateur C++23).

### Phase 2 -- Hygiene & squelette (prerequis)

- **PR-01 (adc_cpp)** -- *Supprimer mecanisme galerie casse + cruft stale.* Delete `_copy_tutorials.py`, retirer l'appel `setup_gallery()` de `conf.py`, supprimer localement `_build/`/`_generated/`/`__pycache__` (deja gitignores, pas de changement gitignore requis). depends-on : aucun. **[L]**. Prerequis de tout le reste Sphinx.
- **PR-02 (adc_cpp)** -- *Squelette Sphinx cible.* Toctree 7 sections (Getting Started / Models / Simulation / AMR / Parallel Backends / Advanced / Reference) ; merger `tutorials_index.md` + `examples.md` en une page "Examples -> adc_cases" ; integrer les docs canoniques (`ARCHITECTURE`/`ALGORITHMS`/`BACKEND_COVERAGE`/`AMR_*`) comme pages Sphinx au lieu d'URLs GitHub brutes. depends-on : PR-01. **[L]**.

### Phase 3 -- Correction des claims faux (verite)

- **PR-03 (adc_cpp)** -- *Purger "AmrSystem mono-bloc".* README:130-132, ARCHITECTURE:26/379-382/547, DSL_MODEL_DESIGN Sec. 0bis/Sec. 5/Phase D, DSL_API Sec. 3 l.86-88 ; corriger aussi le commentaire stale du header `amr_system.hpp:28-31`. depends-on : aucun (peut paralleliser PR-02). **[L]**. Risque #1.
- **PR-04 (adc_cpp)** -- *Supprimer fichier/symbole mort.* Retirer `amr_coupler.hpp`/`AmrCoupler` de ARCHITECTURE:33, COUPLING_SURFACE:118, COUPLER_HIERARCHY Sec. 4 ; supprimer le limiteur fantome "MC" de ALGORITHMS:97 et ARCHITECTURE:288/509 ; corriger "ExB"->`ExBVelocity` et `wave_speeds` hors concept (ARCH:102). depends-on : aucun. **[L]**.
- **PR-05 (adc_cpp)** -- *Reecrire DSL_API.md + docstring source.* Regenerer les 4 snippets faux (conservative_vars/aux, set_poisson(solver=), pas de max_level, runtime params marchent) ; **corriger le docstring `dsl.py:1623/1627`** (NotImplementedError faux) pour que l'autodoc ne propage pas ; ajuster defaut compile (`aot`), tokens flux ('hll' invalide), GPU production cap=False. depends-on : aucun. **[L]**.
- **PR-06 (adc_cpp)** -- *Regenerer compteurs de tests programmatiquement.* Script qui compte depuis `tests/CMakeLists.txt` + glob `python/tests/test_*.py`, injecte dans BACKEND_COVERAGE (SoT) ; ARCHITECTURE Sec. 11 et installation.md referencent au lieu de coder en dur ; corriger `ADC_USE_EIGEN` (README:167 + CI) -- supprimer la ligne fantome. Supprimer les artefacts Finder-copy locaux (`test_polar_system 2.py`, `test_polar_system_step 2.cpp`). depends-on : aucun. **[L]**.
- **PR-07 (adc_cases)** -- *Reconcilier README top-level avec le manifest.* Ajouter les 6 cas DSL manquants a la table "Les cas" ; corriger docstring `schur_magnetized_cartesian` (hook prive `sim._s.set_source_stage`, pas `adc.Split`). depends-on : aucun. **[L]**.

### Phase 4 -- Autodoc & Reference

- **PR-08 (adc_cpp)** -- *Stabiliser l'env autodoc.* Ajouter `numpy` a `requirements.txt`, retirer `sphinx-autodoc-typehints` inutilise, documenter/pinner l'interpreteur exact (cpython-3.12 + numpy + `.so` sur PYTHONPATH) ; ajouter le caveat ABI/interpreteur au quickstart Python README. depends-on : PR-02. **[L] [TC]** (build doc local OK avec l'anaconda 3.12).
- **PR-09 (adc_cpp)** -- *Section Reference (Python autodoc).* Pages Models/Simulation/AMR depuis la surface Sec. 4 ; expliciter add_block(ModelSpec) vs add_equation(type-dispatch), les 3 chemins d'authoring (natif/dsl/hybride), la matrice `_BACKEND_CAPS` et les guards ValueError. depends-on : PR-05, PR-08. **[L]**.
- **PR-10 (adc_cpp)** -- *Section Reference C++ (Doxygen).* Concepts/classes Sec. 4 ; marquer `AmrSystem`/reconstruction wrong-docstring corriges en amont (PR-03/04). depends-on : PR-04. **[L]**.

### Phase 5 -- Assets & provenance

- **PR-11 (adc_cases)** -- *Fixer le mismatch de chemin diocotron.* Repointer les `savefig` vers `figures/` (ou documenter l'etape de copie) ; ajouter capture SHA aux runs. depends-on : aucun. **[L]** (diocotron tourne sur CPU).
- **PR-12 (adc_cases)** -- *Manifeste d'assets + provenance figee.* Regenerer les figures diocotron avec SHA, geler n/dt/backend ; ajouter references par nom de fichier (tuer les 3 orphelins). depends-on : PR-11. **[L]** pour diocotron.
- **PR-13 (adc_cases)** -- *Figures hoffart system-schur (modes 3/4/5).* Decider promotion committee vs ephemere ; regenerer avec SHA. depends-on : PR-12. **[L]** pour system-schur (cart-square CPU) ; **[ROMEO]** pour la variante **amr-imex** (needs Kokkos/MPI build + `--acknowledge-amr-approximation`).

### Phase 6 -- CI doc

- **PR-14 (adc_cpp)** -- *CI `sphinx-build -W` + doxygen.* Job qui build avec warnings-as-errors (attrape orphan-toctree, broken refs) sur runner avec 3.12+numpy+`.so` ; build Doxygen. depends-on : PR-01..PR-10 (sinon `-W` echoue sur la galerie morte et les refs cassees). **[L] [TC]** (le runner CI doit avoir la toolchain doc ; pas de CUDA requis).

**Synthese local vs cluster :** seul **PR-13 (variante amr-imex)** a une dependance dure ROMEO pour regenerer un asset. Tout le reste -- hygiene, squelette, corrections de verite, autodoc, Doxygen, CI doc, figures diocotron + hoffart system-schur -- est **entierement faisable en local darwin/arm64** avec l'interpreteur anaconda 3.12 + numpy. Aucune correction de claim ne necessite le GPU ; la validation device reste hors-scope refonte (ROMEO-manuel).

---

## 8. Risques & angles morts

**Risques cles consolides (confirmes ou inventaire) :**

- **Le recit "AmrSystem mono-bloc" est le risque d'exactitude #1 et il est ancre jusque dans la memoire persistante.** Multi-bloc/multi-especes/Poisson somme/coupled-source/regrid union-tags ont tous ete livres (#195/#199/#205, 7 tests capstone). Nuance honnete a preserver : "pas a parite COMPLETE" reste partiellement vrai (pas de Schur GLOBAL sur AMR), mais la formulation actuelle est fausse sur 4 axes.
- **References a fichier supprime dans 3 docs normatives** (`amr_coupler.hpp`) + **symbole fantome MC** : tout autodoc/Doxygen bati dessus listerait des entites inexistantes.
- **Compteurs de tests : aucune valeur coherente entre docs et disque.** Choisir le disque comme autorite et generer programmatiquement (PR-06), jamais coder en dur.
- **Specs "DESIGN-ONLY" deja implementees** (`SCHUR_CONDENSATION_DESIGN` "rien livre", `AMR_MULTIBLOCK_DESIGN` "facade reste mono-bloc") decrivent un etat passe : a archiver/estampiller IMPLEMENTED, pas a utiliser comme source normative.
- **Autorite cross-repo non verifiable depuis un seul repo :** CONSERVATION_SUMMARY/HOFFART_FIDELITY/HOFFART_STEP_SEQUENCE ancrent leurs claims a des fichiers adc_cases (`model.py`, `run.py`, `test_schur_conservation.py`) absents d'adc_cpp ; garder ces claims clairement labelles "validation application-side".
- **Line-number rot generalise :** la plupart des docs canoniques embarquent des ancres file:line decalees par l'extraction #151 native_loader et autres refactors (ex. `AmrSystem.potential()` est a `bindings.cpp:332`, pas :272). Disclaimees "indicatives" mais trompeuses si lues comme exactes.
- **Galerie Sphinx morte + ABI Python epinglee :** un `sphinx-build -W` echouera tant que PR-01 n'est pas faite ; l'autodoc degrade silencieusement en stubs sans signature hors de l'interpreteur 3.12+numpy exact.
- **adc_cases :** README top-level incomplet (6 cas DSL omis) ; provenance figures faible (committees sans SHA, produites ailleurs que leur chemin) ; couplage `schur` brittle (hook prive) ; seuls 4/15 cas documentes.

**Ce que l'audit N'A PAS PU verifier (a controler AVANT publication) :**

- **Submodule adc_cpp absent du checkout adc_cases** (pas de `.gitmodules`/gitlink) : tous les liens `../../adc_cpp/docs/*.md` sont irresolubles ici et casseraient pour un clone adc_cases standalone. Aucune figure/doc cote adc_cpp n'a pu etre ouverte depuis ce checkout.
- **Couverture de la passe adverse limitee a ~7 claims.** Les verdicts "vrai/non-verifiable" du reste (ALGORITHMS exhaustivite tests, concepts, FFT box-unique, WENO5-Z Borges, GeometricMG, timing 562x/1000x de schur_magnetized_cartesian, matrice ROMEO multi-GPU "?") reposent sur l'inventaire seul.
- **Aucun code execute pour quickstart.md / les recettes Python** (pas d'env numpy+py3.12 dans la passe d'inventaire) : surface API plausible mais non exercee.
- **Tous les nombres de perf et de fidelite** (PERFORMANCE M1, PROFILE_RESULTS, table hoffart -82/-95%, gains Schur) sont mesures et non reproductibles localement (M1/GH200) ; non re-confirmes.
- **Statut Phase F DSL "merge juin 2026" (memoire)** non verifie dans l'inventaire ; a confirmer avant de retirer toute mention "branche non-mergee".
- **Compteurs exacts** (90 vs 94 `adc_add_test`, 60 vs 61 Python) divergent legerement entre les deux agents d'inventaire selon le traitement des macros/lignes dupliquees : regenerer canoniquement avant de publier un chiffre.
---

## 9. Corrections du coordinateur (post-synthese)

Verifications faites par le coordinateur apres la passe 8-agents, corrigeant des erreurs de perimetre d'un finder.

### 9.1 Sec. 6 ERRONE : adc_cpp EST sur disque et porte 33 assets image trackes

Le finder "assets" a conclu a tort que "le submodule adc_cpp n'est PAS present sur disque" et que "tous les assets vivent sous adc_cases" (cwd trop etroit). **Faux.** `adc_cpp/docs/` contient **33 images trackees** (20 PNG + 13 GIF). Le perimetre Phase 5 est donc ~2x plus large qu'annonce en Sec. 6.

**Carte de reference reelle (adc_cpp/docs/, hors `_build/`) :**

| Reference par | Assets | Statut refonte |
|---|---|---|
| **README live (1)** | `anim_romeo_diocotron_amr3.gif` (hero GIF, README:12) | **keep** -- seul asset du portail vivant ; provenance ROMEO a documenter |
| **Galerie MORTE `_generated/tutorials/*` uniquement (6)** | `fig_diocotron_growth.png`, `fig_diocotron_modes.png`, `anim_diocotron.gif`, `anim_diocotron_column.gif`, `anim_diocotron_amr3.gif`, `anim_diocotron_multipatch.gif` | **orphelins apres PR-01** (la suppression de la galerie morte les rend non references) -> reaffecter a la nouvelle section tutoriels OU document/supprimer |
| **`docs/archive/*` uniquement (8)** | `fig_diocotron_amr_vs_uniforme.png`, `fig_diocotron_conv_modes.png`, `fig_diocotron_highorder.png`, `fig_diocotron_invariants.png`, `fig_diocotron_ml_convergence.png`, `fig_diocotron_reproduction.png`, `romeo_amr_efficiency.png`, `romeo_growth_mode4.png`, `romeo_highorder_convergence.png`, `anim_magnetic_diocotron.gif` | archive-only -> **garder avec l'archive** ou deplacer sous `docs/archive/assets/` |
| **`docs/PERFORMANCE.md` (historical) (1)** | `fig_openmp_scaling.png` (aussi galerie morte) | suit la decision PERFORMANCE.md (archive) |
| **ORPHELINS COMPLETS -- le set `tut_*` (10)** | `tut_diocotron_growth.png`, `tut_diocotron_sequence.png`, `tut_euler_poisson.png`, `tut_plasma.png`, `tut_poisson_backends.png`, `tut_tfap_ap.png`, `tut_diocotron_py.gif`, `tut_diocotron_ring.gif`, `tut_ep_collapse.gif`, `tut_tfap_field.gif` | **deja orphelins** : c'etait le pool d'assets des tutoriels Sphinx partis vers adc_cases (suppression `tutorials/` 194c63f). Decision Phase 5 : regenerer-avec-provenance pour la nouvelle galerie, OU supprimer. AUCUN n'a de provenance. |

**Consequence pour le plan :** PR-12/PR-13 (assets) doivent couvrir **deux** depots, pas seulement adc_cases. La suppression de la galerie morte (PR-01) **augmente** le nombre d'orphelins adc_cpp de 10 -> 16 ; il faut une decision explicite "regenerer pour la nouvelle galerie vs supprimer" sur les 16, avec un manifeste d'assets (script producteur + commande + SHA adc_cpp/adc_cases + backend + resolution) comme l'exige la Phase 5.

### 9.2 Sec. 6 ERRONE : "0 embed casse" sous-estime -- il y a des embeds reels cote adc_cpp

Le repo adc_cpp contient bien des embeds markdown/HTML d'images (`README.md:12` `<img src="docs/...">`, `docs/archive/*` et l'ancienne galerie en `![]()`). Le link-check de Phase 6 (`sphinx-build -W`) doit donc couvrir adc_cpp, pas seulement verifier l'absence d'embeds. Risque reel confirme : apres PR-01, 6 embeds de la galerie morte pointeront vers des pages supprimees -- a nettoyer dans le meme PR.

### 9.3 A reconfirmer avant publication (inchange)

Les angles morts de Sec. 8 restent valides : couverture adverse limitee (~7 claims), aucune execution de quickstart.md, compteurs de tests a regenerer canoniquement, et le statut "AmrSystem mono-bloc" dans la **memoire persistante** du coordinateur doit etre corrige (multi-bloc livre #195/#199/#205).
