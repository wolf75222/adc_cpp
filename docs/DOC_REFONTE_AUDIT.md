# Audit documentaire adc_cpp / adc_cases — matrice de vérité (Phase 1)

> Document de synthèse assemblé à partir de l'inventaire 8-agents (prose adc_cpp, Sphinx, trois docs canoniques vérifiées code-en-main, API Python/DSL réelle, cas adc_cases, build/run, tests, assets) plus une passe de vérification adverse. Les verdicts marqués CONFIRMÉ proviennent de la passe adverse indépendante ; les autres reposent sur l'inventaire seul (signalé là où la donnée est mince).

---

## 1. Synthèse exécutive

- **L'ossature canonique est fiable, le bookkeeping ne l'est pas.** Les docs de fond (ARCHITECTURE, ALGORITHMS, BACKEND_COVERAGE, CHOICES, DSL_API, DSL_MODEL_DESIGN, CONSERVATION_SUMMARY, BIBLIOGRAPHY, COUPLER_HIERARCHY, COUPLING_SURFACE, PERFORMANCE) décrivent correctement les couches, les concepts C++20 (`PhysicalModel`, `EllipticSolver`), la couture parallèle three-way (`for_each.hpp` + `comm.hpp`), la matrice CI 4-jobs CPU-only. Le risque est concentré sur (a) le récit « AmrSystem mono-bloc » et (b) la comptabilité (compteurs de tests, symboles fantômes, fichiers supprimés).

- **Risque #1 (haute sévérité, CONFIRMÉ) : « AmrSystem mono-bloc » est OBSOLÈTE.** Répété comme limitation honnête dans README, ARCHITECTURE §8, DSL_MODEL_DESIGN (§0bis/§5/Phase D) et la mémoire persistante, alors que `python/amr_system.cpp` livre une vraie façade multi-bloc (`multi_block()`/`build_multi()`/`set_regrid`, plus aucun throw « un seul bloc »), avec 7 tests capstone et regrid union-tags (#195/#199/#205). Une refonte qui fait confiance aux docs canoniques **ré-affirmerait une fausse limitation**.

- **Risque #2 (CONFIRMÉ) : fichier supprimé encore documenté.** `amr_coupler.hpp`/`AmrCoupler` est DÉLÉTÉ (#164) mais décrit comme « déprécié conservé pour référence » dans TROIS docs normatives (ARCHITECTURE:33, COUPLING_SURFACE:118, COUPLER_HIERARCHY §4). Seul CODEBASE_AUDIT enregistre la suppression. Tout autodoc/référence coupler bâti sur ces trois docs listerait un fichier inexistant.

- **Symbole fantôme : le limiteur « MC ».** Documenté deux fois (ALGORITHMS:97, ARCHITECTURE:288/509) mais absent de `reconstruction.hpp` (seuls `NoSlope`/`Minmod`/`VanLeer`/`Weno5` existent). Un utilisateur qui suit la doc et passe `limiter='mc'` échoue.

- **Compteurs de tests : tous périmés et mutuellement incohérents.** Disque réel ≈ 90-94 `adc_add_test` (non-MPI), ~51 entrées MPI, 60-61 fichiers Python. Les docs disent 71/84/103 (C++) et 26/55 (Python) selon le fichier ; ARCHITECTURE et BACKEND_COVERAGE se contredisent entre eux. Toutes les valeurs sont des **sous-estimations conservatrices** (la suite est plus grosse qu'annoncée). À régénérer programmatiquement, pas à coder en dur.

- **Sphinx : mécanisme de galerie mort (CONFIRMÉ).** `_copy_tutorials.setup_gallery()` pointe vers un `tutorials/` supprimé (commit 194c63f, app layer parti vers adc_cases) → copie 0 fichier ; et même rempli, `tutorials_index.md` est un stub sans toctree → doublement orphelin. Un `sphinx-build` propre produit une section « Tutoriels » vide. `_build/`/`_generated/` locaux sont périmés (mai 30) mais déjà gitignorés/non-trackés.

- **Honnêtement runnable en local (darwin/arm64, sans CUDA) :** cœur C++ Série + ctest, MPI (OpenMPI 5.0.9), OpenMP standalone, Kokkos device=OpenMP, et le module Python — **mais uniquement avec l'interpréteur qui l'a compilé** (`/opt/homebrew/anaconda3/bin/python3.12`, qui a numpy). Footgun confirmé : le `.so` est `cpython-312` ; sous `python3` système (3.9.6) → `ModuleNotFoundError: adc._adc` ; sous un 3.12 sans numpy → mort sur `import numpy` dans `dsl.py`.

- **Nécessite ROMEO/GH200 :** tout chemin CUDA — Kokkos Cuda mono-GPU, MPI+Kokkos Cuda multi-GPU. La CI ne construit **jamais** `Kokkos_ENABLE_CUDA=ON`. Les harnais `python/tests/gpu/*.cpp` sont SLURM-only (`--constraint=armgpu`, nvcc_wrapper, Kokkos 4.4.01) et exclus du glob CI (`-maxdepth 1`). Toute mention « GPU-validé » doit dire « manuellement sur ROMEO GH200 ».

- **DSL_API.md est matériellement périmé (4 claims faux haute sévérité, CONFIRMÉS).** Des snippets crashent tels qu'écrits (`m.u[0]`/`m.a.grad_x` → AttributeError ; `set_poisson('geometric_mg')` → rhs inconnu ; `AmrSystem(max_level=2)` → attribut inconnu ; « runtime params = NotImplementedError » faux : ils marchent). Le claim runtime-NotImplementedError vit aussi dans le **docstring source** (`dsl.py:1623/1627`) → l'autodoc le propagerait.

- **adc_cases : labeling diocotron-vs-Hoffart HONNÊTE.** Le diocotron ExB réduit est partout flaggé « benchmark de normalisation modèle réduit, PAS reproduction du système Euler-Poisson complet » ; le cas complet `hoffart_euler_poisson_dsl` est classé « reproduction-candidate » avec table PENDING/MEASURED montrant -82 à -95% d'erreur. Aucun fichier ne ment. **Mais** le README top-level adc_cases omet 6 cas DSL ; seuls 4/15 cas ont un README ; et la provenance des assets est faible (figures committées sans SHA, produites ailleurs que leur chemin committé).

---

## 2. Matrice des fichiers documentaires

Statuts : **active** (normatif vivant), **historical** (spec/plan partiellement exécuté), **archive** (sous `docs/archive/`), **generated** (artefact). Décisions : keep / rewrite / merge / archive / delete.

### adc_cpp — prose & docs/

| Fichier | Rôle | Public | Statut | Source de vérité | Obsolète/Faux | Doublons | Décision |
|---|---|---|---|---|---|---|---|
| README.md | Portail projet (pitch, GIF, build, validation) | new user | active | ARCHITECTURE + GPU_RUNTIME_PORT + DSL_MODEL_DESIGN | « AmrSystem mono-bloc » (l.130-132) FAUX ; `potential()` :272 stale (réel :332) ; 208 l. vs cible 120-150 | ARCHITECTURE §6/§13, DSL §5, GPU_RUNTIME_PORT | **rewrite** |
| docs/ARCHITECTURE.md | Canonique : 5 couches, concepts, module map, AMR §8, validation §11 | contributor | active | `include/adc/**` | `amr_coupler.hpp` (l.33) supprimé ; §8 mono-bloc obsolète ; §11 compteurs périmés ; limiteur MC inexistant | README module-map, COUPLER_HIERARCHY, GPU_RUNTIME_PORT | **rewrite** |
| docs/ALGORITHMS.md | Canonique : catalogue méthodes (formule/code/test ×20) | contributor | active | `numerics/**` + `tests/` | Limiteur MC (l.97) inexistant ; le reste vérifié bon | §18 DSL compose, §19 DSL §0/§7, §20 ARCH §4 | **keep** (purger MC) |
| docs/BACKEND_COVERAGE.md | SoT auto-déclarée matrice backends/tests | contributor | active | `tests/CMakeLists.txt` + CI | Totaux d'en-tête périmés (84/19/55 vs disque 90-94/31/60-61) ; lignes per-test OK | remplace README « 4 chemins » + DSL §5 | **rewrite** (totaux) |
| docs/CHOICES.md | Canonique : rationale décisions D-1..D-7 | contributor | active | ARCHITECTURE + oracles | Aucun faux ; seul doc accentué (incohérence encodage) | — | **keep** |
| docs/DSL_API.md | Référence user courte DSL | new user | active | `adc/dsl.py` + DSL_MODEL_DESIGN | **4 claims faux** (m.u/m.a, set_poisson, max_level, runtime) ; backend GPU sur-vendu | sous-ensemble DSL §5 | **rewrite** |
| docs/DSL_MODEL_DESIGN.md | Canonique : design DSL/modèle + statut par phase | contributor | active | `dsl.py` + `system.cpp` + `runtime/**` | « AmrSystem mono-bloc » (§0bis/§5/Phase D) obsolète ; line numbers indicatifs | README, ALGORITHMS §18/19, PAPER_ROADMAP | **rewrite** |
| docs/CONSERVATION_SUMMARY.md | Canonique : conservation FV/Schur mesurée vs Hoffart FE | contributor | active | `adc_cases/.../test_schur_conservation.py` (#207) | Aucun interne ; autorité cross-repo non vérifiable depuis adc_cpp | FULL_MODEL_VALIDATION PR2, HOFFART_FIDELITY | **keep** |
| docs/BIBLIOGRAPHY.md | Canonique : références (Hoffart arXiv:2510.11808, etc.) | new user | active | littérature externe | Aucun | citations ancrées ailleurs | **keep** |
| docs/HOFFART_FIDELITY.md | Cross-check cas hoffart vs papier (par-aspect) | contributor | active | `adc_cases` model.py/run.py + engine | Ancres cross-repo non vérifiables depuis adc_cpp ; labels non_verifie honnêtes | CONSERVATION_SUMMARY, HOFFART_STEP_SEQUENCE | **keep** |
| docs/CODEBASE_AUDIT.md | Audit maintenabilité par-fichier (2026-06-06) | internal-dev | active | `include/adc/**` + `python/*.cpp` | L'inventaire le plus exact (note bien amr_coupler RETIRÉ) ; légèrement en retard sur AmrSystem multi-bloc | COUPLING_SURFACE, ARCH module map | **keep** |
| docs/DOC_CLEANUP_PLAN.md | Plan interne : overlaps + SoT par doc | internal-dev | active | self (méta) | « BACKEND_COVERAGE à créer » existe ; README dit 373 l. (réel 208) ; partiellement exécuté | indexe les autres | **archive** |
| docs/COUPLER_HIERARCHY.md | Référence : chaque coupler de `coupling/` | contributor | active | `coupling/*.hpp` | §4 décrit `amr_coupler.hpp` SUPPRIMÉ comme existant (l.152, table l.424) | ARCH §5/§8, COUPLING_SURFACE | **rewrite** |
| docs/COUPLING_SURFACE.md | SoT classification PUBLIC/INTERNAL/DEPRECATED couplers | contributor | active | `coupling/*.hpp` | l.118 liste `amr_coupler.hpp` comme conservé-historique : FAUX (supprimé) | COUPLER_HIERARCHY | **rewrite** |
| docs/PERFORMANCE.md | Mesures perf historiques (CS:APP, M1) | contributor | historical | `adc_cases` benches + `coupling_policy.hpp` | Bannière « historique » OK ; cite `examples/bench_amr.cpp`, `scripts/plot_bench_scaling.py` ABSENTS d'adc_cpp | PROFILE_RESULTS, GPU_RUNTIME_PORT ph11 | **archive** |
| docs/PROFILE_RESULTS.md | Profilage mesuré (2026-06-06), MG domine 96-99.9% | internal-dev | active | `bench/profile_step` | Aucun ; colonnes GH200/CUDA ROMEO-only (honnête) | PERFORMANCE (ancien), RESEARCH_BACKLOG | **keep** |
| docs/GPU_RUNTIME_PORT.md | Journal validation GH200 phases 1-11 | internal-dev | active | harnais `python/tests/gpu/` (hors ctest) | Aucun matériel | README Validation, BACKEND_COVERAGE §3, GPU_ROMEO | **keep** |
| docs/GPU_ROMEO.md | Recette vérif GPU brique DSL sur GH200 (maxdiff=0) | internal-dev | active | `gen_cuda_harness.py` + `romeo_run.sh` | Aucun ; recette étroite | GPU_RUNTIME_PORT, BACKEND_COVERAGE §3 | **keep** |
| docs/PAPER_ROADMAP.md | Roadmap science (Hoffart, O5, ring-edge lock) | internal-dev | active | cas diocotron + todo §6 | Chiffres taux croissance ANCIENS (Minmod-o2 n=192) supersédés | FULL_MODEL_VALIDATION, HOFFART_FIDELITY, todo §6 | **keep** |
| docs/FULL_MODEL_VALIDATION_ROADMAP.md | Roadmap repro modèle complet ; SUPERSÈDE PAPER_ROADMAP | internal-dev | active | HOFFART_FIDELITY + HOFFART_STEP_SEQUENCE + run.py | Auto-corrige PAPER_ROADMAP ; PR2 (#208) faite ; doc science la plus à jour | PAPER_ROADMAP, HOFFART_FIDELITY | **keep** |
| docs/HOFFART_STEP_SEQUENCE.md | Ordre exact macro-step master (Lie pas Strang) | contributor | active | `system_stepper.hpp`, `amr_runtime.hpp`, etc. | Aucun apparent ; ancres file:line spécifiques | HOFFART_FIDELITY, CONSERVATION_SUMMARY | **keep** |
| docs/CODE_DOCUMENTATION_CONVENTION.md | Convention doc code (Doxygen/PEP257) | internal-dev | active | self (style) | Aucun (style guide) | — | **keep** |
| docs/SYSTEM_CPP_EXTRACTION_PLAN.md | Plan split `system.cpp` (NativeLoader) | internal-dev | historical | `python/system.cpp` | Partiellement exécuté (#151, native_loader livré) | CODEBASE_AUDIT | **archive** |
| todo.md | TODO interne vivant (état session, log PR) | internal-dev | active | git history + master | Partiellement stale (l.579 « 6/6 » vs header « 7/7 ») ; 65 KB | PAPER_ROADMAP §6, RESEARCH_BACKLOG | **keep** |
| docs/RESEARCH_BACKLOG.md | Backlog recherche non-auto-complétable | internal-dev | active | `schur_condensation.hpp` + ROMEO | Aucun matériel | PAPER_ROADMAP, FULL_MODEL_VALIDATION | **keep** |
| docs/archive/README.md | Index archive (« hors-archive fait foi ») | internal-dev | archive | self | Aucun (disclaim correct) | — | **keep** |
| docs/archive/ROADMAP.md | Roadmap archivée | internal-dev | archive | superseded ARCH/todo | Historique (placement archive) | todo, PAPER_ROADMAP | **archive** |
| docs/archive/TODO.md | TODO archivé chantier multi-espèces | internal-dev | archive | superseded todo.md | Historique | todo.md | **archive** |
| docs/archive/ETAT_DES_LIEUX.md | Synthèse 6 audits archivée | internal-dev | archive | superseded CODEBASE_AUDIT | Historique, file:line stale | CODEBASE_AUDIT | **archive** |
| docs/archive/ARCHITECTURE_CIBLE.md | Vision north-star archivée | internal-dev | archive | superseded ARCHITECTURE | Aspirationnel (cas « désiré vs implémenté ») | ARCHITECTURE | **archive** |
| docs/archive/DESIGN_MULTISPECIES.md | Design multi-espèces archivé | internal-dev | archive | superseded ARCH §5 | Multi-espèces livré (SystemCoupler) | ARCH §5 | **archive** |
| docs/archive/PLAN_VARIABLES_EPM.md | Plan Variables + EPM archivé | internal-dev | archive | superseded ALGORITHMS §11 | Plan exécuté (eps/Helmholtz/aniso) | ALGORITHMS §11 | **archive** |
| docs/archive/HERO_RUN_AMR.md | Design hero-run AMR distribué | internal-dev | archive | superseded GPU_RUNTIME_PORT ph10/11 | Historique | GPU_RUNTIME_PORT, archive/ROMEO | **archive** |
| docs/archive/ROMEO.md | Journal runs ROMEO archivé | internal-dev | archive | superseded GPU_ROMEO/GPU_RUNTIME_PORT | Log historique | GPU_ROMEO, GPU_RUNTIME_PORT | **archive** |
| docs/archive/DIOCOTRON_GROWTH_RATE.md | Note diocotron growth-rate archivée | internal-dev | archive | adc_cases diocotron + HOFFART_FIDELITY | Chiffres pré-O5 ; contenu vit dans adc_cases | HOFFART_FIDELITY, PAPER_ROADMAP | **archive** |
| docs/archive/two_fluid_ap.md | Note méthode two-fluid AP archivée | internal-dev | archive | `adc_cases/two_fluid_ap/` | Correctement archivée (AP a quitté le cœur) | BIBLIOGRAPHY, RESEARCH_BACKLOG | **archive** |

### adc_cpp — Sphinx (docs/sphinx/)

| Fichier | Rôle | Public | Statut | Source de vérité | Obsolète/Faux | Doublons | Décision |
|---|---|---|---|---|---|---|---|
| docs/sphinx/conf.py | Config Sphinx (Furo, autodoc, inject sys.path, setup_gallery) | internal-dev | active | `_copy_tutorials.py`, `docs.yml` | l.21 `setup_gallery()` mort (tutorials/ supprimé) ; l.59 exclut `_generated/tutorials/README.md` | — | **rewrite** |
| docs/sphinx/_copy_tutorials.py | Helper galerie (copie tutorials/ → _generated) | nobody | active | src `../../tutorials` (is_dir=False) | Module entièrement mort (early-return 0) | — | **delete** |
| docs/sphinx/index.md | master_doc, 3 toctrees | new user | active | installation/quickstart/examples/api.md | Toctree Tutoriels = stub dead-end ; nav plate vs cible 7-sections | — | **rewrite** |
| docs/sphinx/tutorials_index.md | Stub « tutos partis vers adc_cases » | new user | active | git 194c63f~1 (ancien toctree) | Dead-end navigationnel, pas de toctree | examples.md | **merge** |
| docs/sphinx/api.md | Référence API Python (autodoc, 20 symboles) | new user | active | `build-py/python/adc/__init__.py` | Aucun (20 symboles résolus) | — | **keep** |
| docs/sphinx/installation.md | Build C++/Python + backends | new user | active | `CMakeLists.txt` | « 71 ctests » (l.17) à revérifier | BACKEND_COVERAGE (intro) | **keep** |
| docs/sphinx/quickstart.md | Recettes Python annotées | new user | active | `adc/__init__.py` | Non exécuté (pas d'env local) ; API plausible | installation, api | **keep** |
| docs/sphinx/examples.md | Redirige vers adc_cases | new user | active | repo (pas d'examples/ ni scripts/) | Liens cross-doc = URLs GitHub, pas pages Sphinx | tutorials_index | **merge** |
| docs/sphinx/requirements.txt | Deps doc (sphinx, furo, myst, autodoc-typehints) | contributor | active | `conf.py` extensions | `sphinx-autodoc-typehints` jamais activé ; numpy MANQUANT | — | **keep** (corriger) |
| docs/sphinx/_generated/tutorials/ | Sortie galerie générée (00..08 + README) | nobody | generated | `.gitignore` | Périmé mai 30, non régénéré, gitignoré/non-tracké | — | **delete** |
| docs/sphinx/__pycache__/ | Cache bytecode | nobody | generated | `.gitignore` | Stale, gitignoré | — | **delete** |
| docs/_build/ | Sortie HTML Sphinx + Doxygen | nobody | generated | `.gitignore` | Périmé mai 30 (galerie ancienne trompeuse), gitignoré/non-tracké | — | **delete** |

### adc_cpp — build & racine

| Fichier | Rôle | Public | Statut | Source de vérité | Obsolète/Faux | Doublons | Décision |
|---|---|---|---|---|---|---|---|
| CMakeLists.txt | Build root (options, C++23/C++20 Kokkos) | contributor | active | self (autoritatif options) | Ne déclare PAS `ADC_USE_EIGEN` malgré README+CI | — | **keep** |
| python/CMakeLists.txt | Build module pybind11 `_adc` | contributor | active | self + `python/tests/` | Aucun | — | **keep** |

### adc_cases

| Fichier | Rôle | Public | Statut | Source de vérité | Obsolète/Faux | Doublons | Décision |
|---|---|---|---|---|---|---|---|
| README.md (top-level) | Portail cas + table « Les cas » | new user | active | `cases_manifest.toml` | Table l.109-119 INCOMPLÈTE : omet 6 cas DSL | cases_manifest.toml | **rewrite** |
| cases_manifest.toml | Manifeste 16 entrées (catégorie/ci/needs) | contributor | active | self | À jour ; SoT du scope | README table | **keep** |
| diocotron/README.md | Cas diocotron réduit (honnête « normalisation ») | new user | active | run.py + NORMALIZATION.md | Aucun (disclaim correct) | manifest, NORMALIZATION | **keep** |
| hoffart_euler_poisson_dsl/README.md | Cas modèle complet (reproduction-candidate) | new user | active | run.py + results.py + measurement_record | Outputs section implique modes 3/4/5 (seul mode_3 sur disque) | HOFFART_FIDELITY | **keep** |
| magnetic_isothermal_dsl/README.md | Cas DSL aux B_z (parité inter-backend) | new user | active | run.py | Caveat plateforme macOS=aot only (honnête) | — | **keep** |
| schur_magnetized_cartesian/README.md | Cas timing Schur vs explicite | new user | active | run.py | Docstring cite `adc.Split` mais code = hook privé `sim._s.set_source_stage` | — | **rewrite** |

> Note couverture README : seuls **4/15** dossiers de cas ont un README (diocotron, hoffart_euler_poisson_dsl, magnetic_isothermal_dsl, schur_magnetized_cartesian). Les 11 autres reposent sur des docstrings de module (détaillés et exacts). Décision Phase 2 à prendre : générer des README par cas ou canoniser docstrings + table top-level.

---

## 3. Claims faux/obsolètes confirmés

Fusion `claim_findings` + passe adverse. Colonne Verdict : verdict de l'inventaire ; statut adverse en gras (CONFIRMÉ = vérifié indépendamment, NON-CONFIRMÉ = pas re-vérifié).

| Claim | Doc source | Verdict | Preuve |
|---|---|---|---|
| AmrSystem est mono-bloc / explicite-only / sans Roe / sans recon primitive / PAS à parité avec System | ARCHITECTURE §8 (l.26,379-382,547), README:130-132, DSL_MODEL_DESIGN §0bis/§5/Phase D | **obsolète — CONFIRMÉ (haute)** | `amr_system.cpp:202 multi_block()`, `:208 build_multi()`, `:300 if(multi_block())`, parse `recon∈{conservative,primitive}` `:343-345`, `riemann∈{rusanov,hllc,roe}` `:135`, IMEX `:340-346` ; 7 tests capstone (`test_amr_system_twoblock`, `test_amr_multiblock_*`) ; regrid union #199. Aucun throw « un seul bloc ». |
| `m.u[0]`/`m.a.grad_x` pour accéder vars conservatives/aux dans dsl.Model | DSL_API.md §1 (l.21-22,29) | **faux — CONFIRMÉ (haute)** | `dsl.Model` n'a ni `.u` ni `.a` ; API réelle = `conservative_vars(...)→Vars` (`dsl.py:1543`), aux via `m.aux('grad_x')` (`:1585`). Crashe en AttributeError. |
| Runtime params : `param(kind='runtime')` lève NotImplementedError (Phase E) | DSL_API.md §5 (l.104-106) **ET** `dsl.py:1623/1627` (docstring+commentaire) | **faux — CONFIRMÉ (haute)** | `Param.__init__` (`dsl.py:1420-1432`) implémente kind='runtime' (`RuntimeParamRef`), lève seulement ValueError pour kinds inconnus. `test_dsl_runtime_params.py` compile+run+`set_block_params` end-to-end. **Le docstring source propagerait le faux claim via autodoc.** |
| `sim.set_poisson("geometric_mg")` (positionnel) | DSL_API.md §3 (l.70) | **faux — CONFIRMÉ (haute)** | 1er arg positionnel = `rhs` (`bindings.cpp:153`), validé ∈ {charge_density,composite} (`system_field_solver.hpp:197-199`) → throw « rhs inconnu ». Correct : `set_poisson(solver='geometric_mg')`. |
| Galerie Tutoriels câblée dans un toctree (pages atteignables) | sphinx/tutorials_index.md, index.md | **faux — CONFIRMÉ (haute)** | Aucun `.md` ne référence `_generated/tutorials/` ; `tutorials_index.md` = stub prose sans toctree ; `index.md` toctree Tutoriels ne liste que `tutorials_index`. Pré-194c63f avait un vrai toctree. Build propre → section vide. |
| `setup_gallery()` copie `tutorials/*.md` → `_generated/tutorials/` | sphinx/_copy_tutorials.py (+conf.py:21) | **obsolète — CONFIRMÉ (haute)** | `tutorials/` supprimé commit 194c63f ; `_copy_tutorials.py:21-22` early-return 0 (is_dir=False). Copie 0 fichier. Build propre → galerie vide. |
| Construire les docs produit une galerie de tutoriels peuplée | sphinx/index.md | **faux — CONFIRMÉ (haute)** | Conséquence des deux ci-dessus. `_build/` mai 30 montre une galerie qui ne reflète plus les sources. (Adverse rectifie : `_build/`/`_generated/` ne sont PAS committés — gitignorés/non-trackés — artefacts locaux seulement.) |
| Limiteur de reconstruction « MC » (monotonized-central) existe | ALGORITHMS:97, ARCHITECTURE:288/509 | **faux — CONFIRMÉ (medium)** | `reconstruction.hpp` ne définit que `NoSlope:35`/`Minmod:46`/`VanLeer:61`/`Weno5:124`. `grep 'struct MC|MonotonizedCentral'` → aucun. `limiter='mc'` échouerait. |
| `amr_coupler.hpp`/`AmrCoupler` = fichier déprécié conservé | ARCHITECTURE:33, COUPLING_SURFACE:118, COUPLER_HIERARCHY §4 | **faux — CONFIRMÉ (medium)** | `ls include/adc/coupling/amr_coupler.hpp` → absent (supprimé #164). Seul `AmrCouplerMP` existe. CODEBASE_AUDIT:210 enregistre RETIRÉ ; les 3 autres docs sur-décrivent un symbole mort. |
| `AmrSystem(n=128, max_level=2, periodic=True)` | DSL_API.md §3 (l.76) | **faux — CONFIRMÉ (medium)** | `AmrSystemConfig` n'a PAS de champ `max_level` (`bindings.cpp:259-266` : n,L,regrid_every,periodic,distribute_coarse,coarse_max_grid). `setattr(config,'max_level',...)` échoue. |
| Pour backend='production'/target='amr_system' la façade Python AMR rejette HLLC/Roe/primitif (mono-bloc) | DSL_API.md §3 (l.86-88) | **obsolète — CONFIRMÉ (medium)** | `AmrSystem.add_equation` (`__init__.py:1321-1334`) câble `recon='primitive'` et flux roe/hllc via `dispatch_amr_compiled` ; `test_amr_riemann_native.cpp` existe. |
| Option CMake `ADC_USE_EIGEN` (défaut ON, cible `adc_eigen`) | README:167 (+ CI ci.yml:145, docs.yml:45) | **faux — CONFIRMÉ (medium)** | Aucun `option(ADC_USE_EIGEN)`, aucune cible `adc_eigen`, aucun `find_package(Eigen)`. `-DADC_USE_EIGEN` = no-op silencieux. |
| `python -c "...import adc..."` (et fallback python3) importe le module | (commande de prompt) | **faux — CONFIRMÉ (haute)** | `python` introuvable ; `python3`=3.9.6 → `ModuleNotFoundError: adc._adc` (`.so` cpython-312). Importe propre avec `/opt/homebrew/anaconda3/bin/python3.12` + numpy. |
| ARCHITECTURE §11 : 71 ctests cœur + 21 MPI + 26 Python | ARCHITECTURE:442-444 | **obsolète — CONFIRMÉ (medium)** | Disque : ~90-94 `adc_add_test` (non-MPI, dont elliptic_interface), ~51 entrées MPI, 60 fichiers Python. Toutes sous-estimées. |
| BACKEND_COVERAGE base : 84 `adc_add_test` + 19 `add_executable` + 55 Python | BACKEND_COVERAGE:322-324 | **partiellement vrai — CONFIRMÉ (medium)** | Réel : 90 `adc_add_test` + 19 runtime = 109 non-MPI ; bloc MPI 12 add_executable → ~51 entrées ; 60 Python. Totaux d'en-tête périmés, lignes per-test OK. |
| FFT Poisson fonctionne sous MPI | ARCHITECTURE §7 | **faux — CONFIRMÉ (medium)** | `test_mpi_system_fft.cpp` est un lock de NON-régression : la voie FFT est REFUSÉE sous MPI (n_ranks>1) avec erreur collective (fix SIGSEGV #93). FFT = single-rank par design. |
| `schur_magnetized_cartesian` utilise `adc.Split(Explicit, CondensedSchur)` | schur_magnetized_cartesian/run.py:130-146 | **obsolète — CONFIRMÉ (medium)** | Le code appelle le hook PRIVÉ `sim._s.set_source_stage(...)` (`run.py:142`) car l'ABI AOT ne transporte pas SSPRK3 (README:68-71). Même `CondensedSchurSourceStepper` C++ sous-jacent. |
| Suite Python tournée sous pytest | (implicite docs) | **faux — CONFIRMÉ (medium)** | `ci.yml:149` : `find python/tests -maxdepth 1 -name 'test_*.py' | xargs python3`. Pas de pytest ; tests = scripts standalone avec assert+exit. |
| README top-level adc_cases « Les cas » = set complet | adc_cases/README.md:107-119 | **partiellement vrai — CONFIRMÉ (medium)** | Table liste 10 cas, omet diocotron_dsl, two_species_dsl, magnetic_isothermal_dsl, dsl_euler, hoffart_euler_poisson_dsl, schur_magnetized_cartesian (présents dans manifest). |
| diocotron figures « produites dans diocotron/figures/ » | diocotron/run.py:13, README:36 | **faux — CONFIRMÉ (haute)** | `run.py:207` écrit dans `out/diocotron/` (gitignoré) ; tous les savefig (`:246,257,276,286`) y vont. Aucun script n'écrit `figures/`. Les 4 figures committées sont des copies manuelles périmées. |
| Chaque run émet un record capturant adc_cpp SHA + adc_cases SHA | hoffart_euler_poisson_dsl/README:143 | **partiellement vrai — CONFIRMÉ (haute)** | `run.py:422-423` écrit les SHA, MAIS les `metadata.json` sur disque n'ont PAS les clés `adc_cpp_sha`/`adc_cases_sha` (artefacts d'un run antérieur) ; les 4 figures diocotron n'ont aucun metadata. |
| « Reproduction du benchmark diocotron de Hoffart » (titre) | diocotron/run.py:2-11 | **partiellement vrai — CONFIRMÉ (basse)** | Le même docstring + README:8-10 qualifie « modèle réduit ExB, PAS reproduction Euler-Poisson complet ». Reproduit l'oracle analytique à 3 chiffres ; taux FV sous-prédits -22/-27/-5%. Mot « Reproduction » désambiguïsé partout. |
| AOT two-fluid AP validé par un test cœur | tests/test_ap_limit.cpp | **partiellement vrai — CONFIRMÉ (medium)** | `test_ap_limit.cpp`/`test_imex_ap.cpp` valident la PROPRIÉTÉ AP sur un modèle TOY scalaire (du/dt=(u_eq-u)/eps), pas le vrai two-fluid. Le vrai intégrateur vit dans `adc_cases/two_fluid_ap/` (scénario, pas brique). |
| CUDA mono-GPU / multi-GPU validé par test automatique | BACKEND_COVERAGE | **partiellement vrai — CONFIRMÉ (haute)** | La CI ne build jamais `Kokkos_ENABLE_CUDA=ON`. Validation device = ROMEO-manuel via `python/tests/gpu/*.cpp` (exclus du glob CI `-maxdepth 1`). Dire « manuellement sur GH200 ». |
| DSL production/hybride validé sur device Kokkos Cuda | GPU_RUNTIME_PORT | **partiellement vrai — CONFIRMÉ (haute)** | ROMEO-only ; `test_compiled_model_parity` PAS device-validé (segfault nvcc extended-lambda cross-TU). Workaround foncteur nommé non porté sur ce test. |
| Toutes les sorties hoffart écrites « pour chaque mode » 3/4/5 | hoffart_euler_poisson_dsl/README:192-200 | **partiellement vrai — CONFIRMÉ (basse)** | Sur disque seul `mode_3/` existe pour les deux engines ; mode_4/5 absents (table marque PENDING). |

**Non-confirmés / à revoir (data mince) :** la vérification adverse n'a couvert que ~7 claims. Pour ALGORITHMS (« tous chemins/tests existent »), les concepts EllipticSolver/PhysicalModel, FFT box-unique, WENO5-Z Borges, GeometricMG eps/reaction/aniso, le timing `schur_magnetized_cartesian` (562×/1000×) et la matrice ROMEO multi-GPU « ? » : verdicts **vrai/non-vérifiable** issus de l'inventaire seul, à confirmer en Phase 2 par exécution/grep ciblé.

---

## 4. API réelle à documenter (autodoc)

Surface publique réelle issue des `api_entries`. Symboles Python = `python/adc/__init__.py` + `dsl.py` + `integrate.py`, bindings dans `python/bindings.cpp` (seul fichier de `.def()` — `system.cpp`/`amr_system.cpp` sont du C++ pur sans pybind). Concepts/classes C++ = `include/adc/**`. `doc_status` : documented / wrong-docstring / unverified.

### C++ — concepts & classes cœur (graine Reference C++ / Doxygen)

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
| `PoissonFFTSolver` (mono-rank/mono-box, throw si n_ranks≠1) | `numerics/elliptic/poisson_fft_solver.hpp` | documented | `test_poisson_fft.cpp` |
| `DistributedFFTSolver` (band MPI_Alltoall) | `numerics/elliptic/poisson_fft_solver.hpp` | documented | `test_mpi_fft_distributed.cpp` |
| `Coupler<Model,Elliptic>` | `coupling/coupler.hpp` | documented | `test_mpi_coupler_inject.cpp` |
| `SystemCoupler`/`SystemDriver`/`SystemAssembler` | `coupling/system_coupler.hpp` | documented | `test_assembler_driver.cpp` |
| `AmrCouplerMP` (legacy `AmrCoupler` SUPPRIMÉ) | `coupling/amr_coupler_mp.hpp` | documented | `test_mpi_amr_compiled_parity.cpp` |
| `AmrSystemCoupler` | `coupling/amr_system_coupler.hpp` | documented | `test_amr_system_coupler.cpp` |
| `System` (pimpl SystemStepper/FieldSolver/BlockStore) | `runtime/system.hpp` | documented | `python/tests/test_bindings.py` |
| `AmrSystem` (mono- OU multi-bloc, recon/riemann/imex) | `runtime/amr_system.hpp` | **wrong-docstring** (docs disent mono-bloc) | `test_amr_system_twoblock.cpp` |
| `advance_amr` (défini `amr_advance.hpp`, inclus par `amr_reflux_mf.hpp`) | `numerics/time/amr_advance.hpp` | documented | `test_amr_compiled_model.cpp` |
| `for_each_cell` (seam Kokkos/OpenMP/serial) | `mesh/for_each.hpp` | documented | `test_reduce.cpp` |
| `comm` (my_rank/n_ranks/all_reduce_*/barrier) | `parallel/comm.hpp` | documented | `test_mpi_array_reduce.cpp` |
| `ModelSpec`/`model_factory` (dispatch_model) | `runtime/model_factory.hpp` | documented | `python/tests/test_dsl_compose.py` |
| Briques `ExBVelocity`/`IsothermalFlux`/`PotentialForce`/`GravityForce`/`ChargeDensity`/`BackgroundDensity`/`NoSource` | `physics/{hyperbolic,source,elliptic}.hpp` | documented (ARCH dit « ExB » au lieu de `ExBVelocity`) | `test_coupled_source.cpp` |

### Python — `adc.System` (façade composition mono-niveau)

| Symbole | Signature | doc_status | Test preuve |
|---|---|---|---|
| `System(config, mesh, **cfg_kw)` | constructeur | documented | `test_bindings.py` |
| `add_block` / `add_equation` / `add_background` | composition de blocs (add_equation type-dispatch ModelSpec vs CompiledModel) | documented | `test_bindings.py`, `test_dsl_production.py`, `test_dsl_dynamic.py` |
| `add_coupling` / `add_coupled_source` / `add_ionization` / `add_collision` / `add_thermal_exchange` | couplages inter-espèces | documented | `test_dsl_coupled_source.py` |
| `add_elliptic_model` | EPM (operator/rhs/output) | documented | `test_poisson_eps.py` |
| `set_poisson(rhs,solver,bc,wall,wall_radius,epsilon)` | config Poisson | documented | `test_bindings.py` |
| `set_density` / `set_primitive_state` / `get_primitive_state` / `set_block_params` | I/O état | documented | `test_bindings.py`, `test_primitive_state.py`, `test_dsl_runtime_params.py` |
| `set_magnetic_field` / `set_epsilon_field` / `set_epsilon_anisotropic_field` | champs aux étendus | documented | `test_magnetic_field.py`, `test_poisson_eps_aniso.py` |
| `set_source_stage` / `set_time_scheme` | stage source (Schur) / schéma temps | documented | `test_schur_via_system.py`, `test_strang_split.py` |
| `step` / `advance` / `step_cfl` / `step_adaptive` / `run` | avancement | documented | `test_bindings.py`, `test_stride.py` |
| `solve_fields` / `eval_rhs` / `get_state` / `set_state` | champs/RHS | documented | `test_weno5_ssprk3.py` |
| `set_disc_domain` / `disc_mask` | domaine disque | documented | `test_disc_domain_mask.py` |
| `block_names`/`mass`/`density`/`potential`/`nx`/`ny`/`time`/`n_species`/`n_vars`/`variable_names`/`variable_roles`/`block_gamma`/`abi_key` | introspection | documented | `test_bindings.py`, `test_dsl_roles.py`, `test_dsl_abi_metadata.py` |

### Python — `adc.AmrSystem` (façade AMR ; multi-bloc réel)

| Symbole | doc_status | Test preuve |
|---|---|---|
| `AmrSystem(config, **cfg_kw)`, `add_block`, `add_equation` | documented | `test_amr_multiblock.py`, `test_dsl_production_amr.py` |
| `set_refinement` / `set_phi_refinement` / `set_poisson` | documented | `test_amr_multiblock.py` |
| `AmrSystemConfig` (n,L,regrid_every,periodic,distribute_coarse,coarse_max_grid — **PAS de max_level**) | documented | `test_amr_multiblock.py` |
| `SystemConfig` (n,L,periodic,geometry,nr,ntheta,r_min,r_max) | documented | `test_polar_system.py` |

### Python — briques de composition (`adc.Model(...)`)

| Symbole | doc_status | Test preuve |
|---|---|---|
| `Model(state,transport,source,elliptic)→ModelSpec`, `ModelSpec` | documented | `test_bindings.py` |
| `CompositeModel(...)→dsl.HybridModel` | documented | `test_dsl_hybrid.py` |
| `Scalar`/`FluidState`, `ExB`/`CompressibleFlux`/`IsothermalFlux` | documented | `test_bindings.py` |
| `NoSource`/`PotentialForce`/`GravityForce`, `ChargeDensity`/`BackgroundDensity`/`GravityCoupling` | documented | `test_bindings.py` |
| `Spatial(limiter,flux,recon)` / `FiniteVolume(...)` | documented | `test_weno5_compiledmodel.py`, `test_dsl_recon.py` |
| `Explicit` / `IMEX` / `SourceImplicit` / `Implicit` (**OBSOLETE, DeprecationWarning → IMEX**) | documented | `test_stride.py`, `test_time_policy.py` |
| `Split` / `Strang` / `CondensedSchur` (descripteurs role/field hardcodés C++) | documented | `test_strang_split.py`, `test_schur_via_system.py` |
| `Role`, `CartesianMesh`/`PolarMesh`, `Ionization`/`Collision`/`ThermalExchange` | documented | `test_dsl_roles.py`, `test_polar_system.py`, `test_dsl_coupled.py` |
| EPM : `elliptic`/`div_eps_grad`/`charge_density`/`composite_rhs`/`electric_field_from_potential`, `EllipticSolver`/`EllipticModel`/`DivEpsGrad`/`CompositeRhs`/`ChargeDensitySource` | documented | `test_poisson_eps.py`, `test_poisson_composite.py` |
| `PythonFlux` (backend host interprété) | documented | `test_dsl.py` |

### Python — `adc.dsl` (DSL symbolique) & `adc.integrate`

| Symbole | doc_status | Test preuve |
|---|---|---|
| `dsl.Model` (conservative_vars/primitive/aux/flux/eigenvalues/source/elliptic_rhs/param/compile) | documented (mais docstring `param` runtime = **wrong**) | `test_dsl_phase_a.py`, `test_dsl_production.py` |
| `dsl.Model.compile(backend='aot'|'production'|'prototype', target='system'|'amr_system', ...)` | documented (défaut réel = `aot`) | `test_dsl_compile_facade.py`, `test_dsl_compile_cache.py` |
| `dsl.CompiledModel` (backend/adder/so_path/caps/abi_key/runtime_param_*) | documented | `test_dsl_compile_facade.py` |
| `dsl.HyperbolicModel`, `dsl.Param`/`RuntimeParam`, `dsl.sqrt`/`Expr` | documented (Param: voir wrong-docstring) | `test_dsl_codegen.py`, `test_dsl_runtime_params.py`, `test_dsl_cse.py` |
| `dsl.HyperbolicBrick`/`SourceBrick`/`EllipticBrick`, `dsl.HybridModel`, `dsl.NativeBrick`/`CompiledBrick` | documented | `test_dsl_brick.py`, `test_dsl_hybrid*.py` |
| `dsl.CoupledSource`/`CompiledCoupledSource` (bytecode couplings) | documented | `test_dsl_coupled_source*.py` |
| `integrate.euler_step` / `ssprk2_step` | documented | `test_dsl.py` |
| `abi_key` (module-level) | documented | `test_dsl_abi_metadata.py` |

**Matrice capacités backend (load-bearing, vit dans `_BACKEND_CAPS` `dsl.py:1382-1393`) :** prototype/aot = CPU-only, no MPI/AMR/GPU ; production = CPU+MPI+AMR, **gpu=False côté Python** (par prudence, malgré device-clean C++). Restrictions appliquées comme `ValueError` explicites dans `add_equation` (stride>1 sur aot ; evolve=False sur prototype/aot ; weno5 sur prototype ; implicit_vars sur .so compilé ; HLLC/Roe nécessite un primitif 'p'). Toute doc « feature X marche sur backend Y » doit matcher ces guards.

---

## 5. Cas adc_cases — vérité par cas

15 dossiers de cas (le manifest a 16 entrées car diocotron a 2 scripts : `run.py` + `band_instability.py`). Modèle : natif (briques C++ composées) / dsl (formules symboliques compilées) / hybride / interprété / bespoke-C++. CI/catégorie = `cases_manifest.toml`.

| Cas | README ? | Modèle | Schéma temps | Schéma espace | Poisson | Backends réels | CI/catégorie | Assets | Limites honnêtes |
|---|---|---|---|---|---|---|---|---|---|
| composition | non | natif (electron_euler + ion_isothermal + diocotron) | IMEX(10) e⁻ / Explicit ions ; Part D = SSPRK2 Python | per-bloc vanleer/hllc + minmod/rusanov | charge_density, geometric_mg | adc compilé | true / tutoriel | aucun (prints) | Aucun claim physique ; teste compo hétérogène + déterminisme bit |
| custom_scheme | non | numpy pur (ExB+upwind) ; adc = oracle Poisson seul | SSPRK2 Python, dt=CFL·dx/v | central + upwind numpy | charge_density, geometric_mg (seul appel lib) | adc (Poisson only) | true / tutoriel | aucun | Scheme entier en Python ; lib = solveur elliptique |
| **diocotron** | **oui** | **natif RÉDUIT ExB scalaire** (Scalar+ExB+BackgroundDensity) — **PAS Euler-Poisson complet** | Explicit SSPRK2 (CFL=0.4) | MUSCL minmod + Rusanov | dirichlet, wall=circle R=0.40 | adc + matplotlib | run.py false/**reproduction** ; band_instability true/validation | figures/ committées (dispersion/amplitude/snapshots/diocotron.gif) | **HONNÊTE** : README+docstring disent « normalisation modèle réduit, PAS repro système complet ». Oracle Petri à 3 chiffres ; FV sous-prédit -22/-27/-5% |
| diocotron_amr | non | natif diocotron sur AmrSystem | Explicit (CFL=0.4) | NoSlope + Rusanov o1 | charge_density, geometric_mg periodic | adc (AmrSystem) | true / validation | aucun | Multi-patch Berger-Rigoutsos + reflux ; teste raffinement réel vs contrôle |
| diocotron_dsl | non | dsl (formules ≡ ExBVelocity+BackgroundDensity) | Explicit SSPRK2 (≡ natif) | FiniteVolume(minmod, rusanov) | charge_density, geometric_mg | dsl production→aot fallback ; needs cxx | true / validation (needs cxx) | out/*.so | Claim : état **bit-identique** au natif (np.array_equal) |
| dsl_euler | non | dsl HyperbolicModel Euler 2D (pas de source/Poisson) | forward Euler, cfl_dt | Rusanov via `PythonFlux` (**interprété numpy**) | aucun | host interprété (PythonFlux) | false / **experimental** | aucun | Prototype INTERPRÉTÉ, distinct du chemin compilé des autres *_dsl |
| euler_poisson | non | natif euler_poisson (GravityForce + GravityCoupling ±) | Explicit SSPRK2, 20 pas | vanleer/hllc | charge_density, geometric_mg periodic | adc compilé | true / validation | aucun | Contraste signe couplage (attractif dE<0 / répulsif dE>0) ; pas de claim papier |
| **hoffart_euler_poisson_dsl** | **oui** | **dsl Euler-Poisson magnétique COMPLET** (continuité + momentum + Lorentz, barotrope) ; variantes schur/local | system-schur SSPRK3 + CondensedSchur(θ=0.5) ; amr-imex backward-Euler | FiniteVolume WENO5-Z + Rusanov | composite, dirichlet, wall=circle R=16 | dsl production (system/amr_system) ; amr-imex needs Kokkos/MPI | check_model true/validation ; run.py false/**reproduction-candidate** | out/.../{amplitude,snapshots,growth_rates,gifs,metadata} | **HONNÊTE** : bannière « PAS de repro quantitative établie » ; table -82 à -95% (n=256/384), n=512 PENDING, amr-imex PENDING. Gaps : Lie pas Strang, Poisson once-per-step, géométrie cart+wall suspecte |
| **magnetic_isothermal_dsl** | **oui** | dsl isotherme magnétisé (aux B_z index 3) ; **pas d'oracle natif** | Explicit SSPRK2, 40 pas | FiniteVolume(minmod, rusanov) | charge_density, geometric_mg periodic ; set_magnetic_field | dsl production+aot (macOS=aot only) ; needs cxx | true / validation | out/*.so | Validé SANS oracle natif : parité inter-backend + oracle Lorentz analytique (dmax==0) |
| multispecies | non | natif (e⁻ Euler 4-var + ions isotherme 3-var) couplés 1 Poisson | Explicit, dt=0.001 | minmod les deux | charge_density, geometric_mg periodic | adc compilé | true / validation | aucun | Conservation masse par espèce <1e-9 |
| plasma | non | recipe natif 3-espèces (e⁻ HLLC+primitif, ions+neutres isotherme) + ionisation+collision | Explicit step_cfl(0.3) | vanleer/hllc/primitif + minmod | f=q_e n_e+q_i n_i | adc compilé | true / validation | aucun | Honnête : transfert momentum/énergie des particules créées = simplification |
| **schur_magnetized_cartesian** | **oui** | dsl isotherme magnétisé ; variantes local/schur (même équations que magnetic_isothermal) | transport SSPRK2 ; source explicite OU CondensedSchur via `sim._s.set_source_stage` (hook privé) | FiniteVolume(minmod, rusanov, conservative) | charge_density, geometric_mg periodic ; B_z=ω_c | dsl aot (production échoue dlopen macOS arm64) ; needs cxx | false / **experimental** | out/dt_stable.csv | Étude timing : explicite plateau dt·ω_c~0.3, Schur 178-316 (gain 562×/1000×). Utilise hook privé car `adc.Split` pas câblé sur ABI AOT |
| two_euler | non | natif euler ×2 gaz indépendants (NON couplés) | multirate step_adaptive(0.4) | vanleer/hllc/primitif | f=0 (juste pour solve_fields) | adc compilé | true / validation | aucun | « 2 Euler même code » ; blocs indépendants |
| two_fluid_ap | non | **bespoke C++** (two_fluid_ap.hpp) AP isotherme ; PAS brique composable | IMEX/AP (terme raide implicite) | FV continuité, dans le JIT | elliptic AP reformulé dans le scénario C++ | JIT .so via ctypes (build_shared) ; needs cxx C++20 | true / validation (needs cxx) | out/.../_two_fluid_ap.{so,dylib} | Remplace l'échappatoire `_TwoFluidAP` retirée du cœur ; vit dans adc_cases |
| two_species_dsl | non | dsl ×2 (e⁻ Euler 4-var + ions isotherme 3-var) + source électrostatique, 1 Poisson | Explicit SSPRK2 (≡ natif) | FiniteVolume(minmod, rusanov) | charge_density, geometric_mg periodic | dsl production→aot ; needs cxx | true / validation (needs cxx) | out/*.so | Ions bit-identiques ; e⁻ diffèrent ~machine-eps (<1e-24, réassociation float dans RHS Poisson partagé, documenté) |

**Honnêteté diocotron-vs-Hoffart (CONFIRMÉE) :** deux modèles physiques DIFFÉRENTS partagent le nom « diocotron » et la réf arXiv:2510.11808 : (a) le RÉDUIT ExB scalaire (diocotron, custom_scheme, diocotron_amr, diocotron_dsl) et (b) le COMPLET Euler-Poisson magnétique (hoffart_euler_poisson_dsl). Le manifest classe (a)/run.py « reproduction » (de la normalisation analytique uniquement) et (b) « reproduction-candidate » (table PENDING). Grep repo-wide pour « reproduction complète/full » ne renvoie que des disclaimers. `results.py` (verify_paper_windows, engine_label) **refuse mécaniquement** de mélanger les taux 2π/ρ̄ du réduit avec les nombres bruts du complet. Aucun fichier ne ment ; le risque résiduel est en prose : le mot « Reproduction » dans le titre diocotron, hors contexte, sur-vend (il reproduit l'oracle analytique, pas le système complet, et sous-prédit jusqu'à -27%). Recommandation : remplacer par « benchmark de normalisation modèle réduit ».

---

## 6. Assets & provenance

Tous les assets image vivent sous adc_cases. Le submodule adc_cpp n'est PAS présent sur disque (pas de `.gitmodules`, pas de gitlink) → aucune figure sous `adc_cpp/docs/` ne peut être inspectée localement.

**Compteurs :**

| Métrique | Valeur |
|---|---|
| Total fichiers image | 11 (4 PNG + GIF trackés ; 7 untracked) |
| Trackés (canoniques) | 4 — `diocotron/figures/{dispersion,amplitude,snapshots}.png` + `diocotron.gif` |
| Orphelins (référencés seulement par phrase générique, pas par nom) | 3 — `mode_3/snapshots.png` (×2 engines), `mode_3/diocotron_l3.gif` |
| Sans provenance (aucun SHA/backend/résolution) | 11/11 |
| Liens image cassés (embeds `![]()` / `<img>`) | **0** — le repo entier ne contient AUCUN embed markdown ; figures référencées par nom en prose/table uniquement |

**Embeds cassés :** aucun. Le risque de repro n'est pas le dangling `![]()` mais le **mismatch de chemin** et l'**absence de provenance**.

**Assets untracked (out/, gitignorés) → décision regenerate :**

| Asset | Producteur | Référencé par | Décision | Note |
|---|---|---|---|---|
| out/hoffart_..._system_schur/growth_rates.png | run.py:469 | README:200 (par nom) | regenerate | metadata.json sans SHA (run antérieur) |
| out/hoffart_..._system_schur/mode_3/amplitude.png | run.py:320 | README:197 | regenerate | seul mode_3 existe (3/4/5 attendus) |
| out/hoffart_..._system_schur/mode_3/snapshots.png | run.py:339 | **orphelin** (« 3×3 panel » générique) | regenerate | pas de SHA |
| out/hoffart_..._system_schur/mode_3/diocotron_l3.gif | run.py:360-361 | **orphelin** (« an animated GIF ») | regenerate | nom dynamique mode-indexé |
| out/hoffart_..._amr_imex/growth_rates.png | run.py:469 | README:200 | regenerate | engine expérimental, pas de SHA |
| out/hoffart_..._amr_imex/mode_3/amplitude.png | run.py:320 | README:197 | regenerate | expérimental |
| out/hoffart_..._amr_imex/mode_3/snapshots.png | run.py:339 | **orphelin** | regenerate | expérimental |

**Assets trackés (canoniques) → décision keep, mais provenance à reconstruire :**

| Asset | Producteur réel | Décision | Note |
|---|---|---|---|
| diocotron/figures/dispersion.png | run.py:246 (écrit dans `out/diocotron/`, PAS figures/) | keep | 806×559 ; aucun SHA/backend ; copie manuelle |
| diocotron/figures/amplitude.png | run.py:257 (→ out/) | keep | idem |
| diocotron/figures/snapshots.png | run.py:286 (→ out/) | keep | 1690×442 ; aucune provenance |
| diocotron/figures/diocotron.gif | run.py:276 (→ out/) | keep | 420×420 ~512KB ; copie manuelle |

**Décisions de provenance Phase 5 :** (1) repointer les `savefig` de `run.py` vers `figures/` OU documenter explicitement l'étape de copie ; (2) régénérer avec le `run.py` courant pour peupler `adc_cpp_sha`/`adc_cases_sha` puis figer ; (3) décider quelles figures hoffart sont promues (committées) vs laissées éphémères ; (4) figer les modes exacts (3/4/5) + n + dt + engine + SHA ; (5) ajouter des références par nom de fichier ou un manifeste d'assets pour que les renommages soient détectés par un link-check.

---

## 7. Plan d'intégration ordonné (Phases 2-6)

Légende exécution : **[L]** entièrement faisable en local darwin/arm64 ; **[ROMEO]** nécessite le cluster (CUDA/MPI multi-GPU) pour régénérer des assets ou valider ; **[TC]** bloqué/conditionné par la toolchain de build doc (interpréteur 3.12 + numpy + compilateur C++23).

### Phase 2 — Hygiène & squelette (prérequis)

- **PR-01 (adc_cpp)** — *Supprimer mécanisme galerie cassé + cruft stale.* Delete `_copy_tutorials.py`, retirer l'appel `setup_gallery()` de `conf.py`, supprimer localement `_build/`/`_generated/`/`__pycache__` (déjà gitignorés, pas de changement gitignore requis). depends-on : aucun. **[L]**. Prérequis de tout le reste Sphinx.
- **PR-02 (adc_cpp)** — *Squelette Sphinx cible.* Toctree 7 sections (Getting Started / Models / Simulation / AMR / Parallel Backends / Advanced / Reference) ; merger `tutorials_index.md` + `examples.md` en une page « Examples → adc_cases » ; intégrer les docs canoniques (`ARCHITECTURE`/`ALGORITHMS`/`BACKEND_COVERAGE`/`AMR_*`) comme pages Sphinx au lieu d'URLs GitHub brutes. depends-on : PR-01. **[L]**.

### Phase 3 — Correction des claims faux (vérité)

- **PR-03 (adc_cpp)** — *Purger « AmrSystem mono-bloc ».* README:130-132, ARCHITECTURE:26/379-382/547, DSL_MODEL_DESIGN §0bis/§5/Phase D, DSL_API §3 l.86-88 ; corriger aussi le commentaire stale du header `amr_system.hpp:28-31`. depends-on : aucun (peut paralléliser PR-02). **[L]**. Risque #1.
- **PR-04 (adc_cpp)** — *Supprimer fichier/symbole mort.* Retirer `amr_coupler.hpp`/`AmrCoupler` de ARCHITECTURE:33, COUPLING_SURFACE:118, COUPLER_HIERARCHY §4 ; supprimer le limiteur fantôme « MC » de ALGORITHMS:97 et ARCHITECTURE:288/509 ; corriger « ExB »→`ExBVelocity` et `wave_speeds` hors concept (ARCH:102). depends-on : aucun. **[L]**.
- **PR-05 (adc_cpp)** — *Réécrire DSL_API.md + docstring source.* Régénérer les 4 snippets faux (conservative_vars/aux, set_poisson(solver=), pas de max_level, runtime params marchent) ; **corriger le docstring `dsl.py:1623/1627`** (NotImplementedError faux) pour que l'autodoc ne propage pas ; ajuster défaut compile (`aot`), tokens flux ('hll' invalide), GPU production cap=False. depends-on : aucun. **[L]**.
- **PR-06 (adc_cpp)** — *Régénérer compteurs de tests programmatiquement.* Script qui compte depuis `tests/CMakeLists.txt` + glob `python/tests/test_*.py`, injecté dans BACKEND_COVERAGE (SoT) ; ARCHITECTURE §11 et installation.md référencent au lieu de coder en dur ; corriger `ADC_USE_EIGEN` (README:167 + CI) — supprimer la ligne fantôme. Supprimer les artefacts Finder-copy locaux (`test_polar_system 2.py`, `test_polar_system_step 2.cpp`). depends-on : aucun. **[L]**.
- **PR-07 (adc_cases)** — *Réconcilier README top-level avec le manifest.* Ajouter les 6 cas DSL manquants à la table « Les cas » ; corriger docstring `schur_magnetized_cartesian` (hook privé `sim._s.set_source_stage`, pas `adc.Split`). depends-on : aucun. **[L]**.

### Phase 4 — Autodoc & Reference

- **PR-08 (adc_cpp)** — *Stabiliser l'env autodoc.* Ajouter `numpy` à `requirements.txt`, retirer `sphinx-autodoc-typehints` inutilisé, documenter/pinner l'interpréteur exact (cpython-3.12 + numpy + `.so` sur PYTHONPATH) ; ajouter le caveat ABI/interpréteur au quickstart Python README. depends-on : PR-02. **[L] [TC]** (build doc local OK avec l'anaconda 3.12).
- **PR-09 (adc_cpp)** — *Section Reference (Python autodoc).* Pages Models/Simulation/AMR depuis la surface §4 ; expliciter add_block(ModelSpec) vs add_equation(type-dispatch), les 3 chemins d'authoring (natif/dsl/hybride), la matrice `_BACKEND_CAPS` et les guards ValueError. depends-on : PR-05, PR-08. **[L]**.
- **PR-10 (adc_cpp)** — *Section Reference C++ (Doxygen).* Concepts/classes §4 ; marquer `AmrSystem`/reconstruction wrong-docstring corrigés en amont (PR-03/04). depends-on : PR-04. **[L]**.

### Phase 5 — Assets & provenance

- **PR-11 (adc_cases)** — *Fixer le mismatch de chemin diocotron.* Repointer les `savefig` vers `figures/` (ou documenter l'étape de copie) ; ajouter capture SHA aux runs. depends-on : aucun. **[L]** (diocotron tourne sur CPU).
- **PR-12 (adc_cases)** — *Manifeste d'assets + provenance figée.* Régénérer les figures diocotron avec SHA, geler n/dt/backend ; ajouter références par nom de fichier (tuer les 3 orphelins). depends-on : PR-11. **[L]** pour diocotron.
- **PR-13 (adc_cases)** — *Figures hoffart system-schur (modes 3/4/5).* Décider promotion committée vs éphémère ; régénérer avec SHA. depends-on : PR-12. **[L]** pour system-schur (cart-square CPU) ; **[ROMEO]** pour la variante **amr-imex** (needs Kokkos/MPI build + `--acknowledge-amr-approximation`).

### Phase 6 — CI doc

- **PR-14 (adc_cpp)** — *CI `sphinx-build -W` + doxygen.* Job qui build avec warnings-as-errors (attrape orphan-toctree, broken refs) sur runner avec 3.12+numpy+`.so` ; build Doxygen. depends-on : PR-01..PR-10 (sinon `-W` échoue sur la galerie morte et les refs cassées). **[L] [TC]** (le runner CI doit avoir la toolchain doc ; pas de CUDA requis).

**Synthèse local vs cluster :** seul **PR-13 (variante amr-imex)** a une dépendance dure ROMEO pour régénérer un asset. Tout le reste — hygiène, squelette, corrections de vérité, autodoc, Doxygen, CI doc, figures diocotron + hoffart system-schur — est **entièrement faisable en local darwin/arm64** avec l'interpréteur anaconda 3.12 + numpy. Aucune correction de claim ne nécessite le GPU ; la validation device reste hors-scope refonte (ROMEO-manuel).

---

## 8. Risques & angles morts

**Risques clés consolidés (confirmés ou inventaire) :**

- **Le récit « AmrSystem mono-bloc » est le risque d'exactitude #1 et il est ancré jusque dans la mémoire persistante.** Multi-bloc/multi-espèces/Poisson sommé/coupled-source/regrid union-tags ont tous été livrés (#195/#199/#205, 7 tests capstone). Nuance honnête à préserver : « pas à parité COMPLÈTE » reste partiellement vrai (pas de Schur GLOBAL sur AMR), mais la formulation actuelle est fausse sur 4 axes.
- **Références à fichier supprimé dans 3 docs normatives** (`amr_coupler.hpp`) + **symbole fantôme MC** : tout autodoc/Doxygen bâti dessus listerait des entités inexistantes.
- **Compteurs de tests : aucune valeur cohérente entre docs et disque.** Choisir le disque comme autorité et générer programmatiquement (PR-06), jamais coder en dur.
- **Specs « DESIGN-ONLY » déjà implémentées** (`SCHUR_CONDENSATION_DESIGN` « rien livré », `AMR_MULTIBLOCK_DESIGN` « façade reste mono-bloc ») décrivent un état passé : à archiver/estampiller IMPLEMENTED, pas à utiliser comme source normative.
- **Autorité cross-repo non vérifiable depuis un seul repo :** CONSERVATION_SUMMARY/HOFFART_FIDELITY/HOFFART_STEP_SEQUENCE ancrent leurs claims à des fichiers adc_cases (`model.py`, `run.py`, `test_schur_conservation.py`) absents d'adc_cpp ; garder ces claims clairement labellés « validation application-side ».
- **Line-number rot généralisé :** la plupart des docs canoniques embarquent des ancres file:line décalées par l'extraction #151 native_loader et autres refactors (ex. `AmrSystem.potential()` est à `bindings.cpp:332`, pas :272). Disclaimées « indicatives » mais trompeuses si lues comme exactes.
- **Galerie Sphinx morte + ABI Python épinglée :** un `sphinx-build -W` échouera tant que PR-01 n'est pas faite ; l'autodoc dégrade silencieusement en stubs sans signature hors de l'interpréteur 3.12+numpy exact.
- **adc_cases :** README top-level incomplet (6 cas DSL omis) ; provenance figures faible (committées sans SHA, produites ailleurs que leur chemin) ; couplage `schur` brittle (hook privé) ; seuls 4/15 cas documentés.

**Ce que l'audit N'A PAS PU vérifier (à contrôler AVANT publication) :**

- **Submodule adc_cpp absent du checkout adc_cases** (pas de `.gitmodules`/gitlink) : tous les liens `../../adc_cpp/docs/*.md` sont irrésolubles ici et casseraient pour un clone adc_cases standalone. Aucune figure/doc côté adc_cpp n'a pu être ouverte depuis ce checkout.
- **Couverture de la passe adverse limitée à ~7 claims.** Les verdicts « vrai/non-vérifiable » du reste (ALGORITHMS exhaustivité tests, concepts, FFT box-unique, WENO5-Z Borges, GeometricMG, timing 562×/1000× de schur_magnetized_cartesian, matrice ROMEO multi-GPU « ? ») reposent sur l'inventaire seul.
- **Aucun code exécuté pour quickstart.md / les recettes Python** (pas d'env numpy+py3.12 dans la passe d'inventaire) : surface API plausible mais non exercée.
- **Tous les nombres de perf et de fidélité** (PERFORMANCE M1, PROFILE_RESULTS, table hoffart -82/-95%, gains Schur) sont mesurés et non reproductibles localement (M1/GH200) ; non re-confirmés.
- **Statut Phase F DSL « mergé juin 2026 » (mémoire)** non vérifié dans l'inventaire ; à confirmer avant de retirer toute mention « branche non-mergée ».
- **Compteurs exacts** (90 vs 94 `adc_add_test`, 60 vs 61 Python) divergent légèrement entre les deux agents d'inventaire selon le traitement des macros/lignes dupliquées : régénérer canoniquement avant de publier un chiffre.
---

## 9. Corrections du coordinateur (post-synthèse)

Vérifications faites par le coordinateur après la passe 8-agents, corrigeant des erreurs de périmètre d'un finder.

### 9.1 §6 ERRONÉ : adc_cpp EST sur disque et porte 33 assets image trackés

Le finder « assets » a conclu à tort que « le submodule adc_cpp n'est PAS présent sur disque » et que « tous les assets vivent sous adc_cases » (cwd trop étroit). **Faux.** `adc_cpp/docs/` contient **33 images trackées** (20 PNG + 13 GIF). Le périmètre Phase 5 est donc ~2× plus large qu'annoncé en §6.

**Carte de référence réelle (adc_cpp/docs/, hors `_build/`) :**

| Référencé par | Assets | Statut refonte |
|---|---|---|
| **README live (1)** | `anim_romeo_diocotron_amr3.gif` (hero GIF, README:12) | **keep** — seul asset du portail vivant ; provenance ROMEO à documenter |
| **Galerie MORTE `_generated/tutorials/*` uniquement (6)** | `fig_diocotron_growth.png`, `fig_diocotron_modes.png`, `anim_diocotron.gif`, `anim_diocotron_column.gif`, `anim_diocotron_amr3.gif`, `anim_diocotron_multipatch.gif` | **orphelins après PR-01** (la suppression de la galerie morte les rend non référencés) → réaffecter à la nouvelle section tutoriels OU document/supprimer |
| **`docs/archive/*` uniquement (8)** | `fig_diocotron_amr_vs_uniforme.png`, `fig_diocotron_conv_modes.png`, `fig_diocotron_highorder.png`, `fig_diocotron_invariants.png`, `fig_diocotron_ml_convergence.png`, `fig_diocotron_reproduction.png`, `romeo_amr_efficiency.png`, `romeo_growth_mode4.png`, `romeo_highorder_convergence.png`, `anim_magnetic_diocotron.gif` | archive-only → **garder avec l'archive** ou déplacer sous `docs/archive/assets/` |
| **`docs/PERFORMANCE.md` (historical) (1)** | `fig_openmp_scaling.png` (aussi galerie morte) | suit la décision PERFORMANCE.md (archive) |
| **ORPHELINS COMPLETS — le set `tut_*` (10)** | `tut_diocotron_growth.png`, `tut_diocotron_sequence.png`, `tut_euler_poisson.png`, `tut_plasma.png`, `tut_poisson_backends.png`, `tut_tfap_ap.png`, `tut_diocotron_py.gif`, `tut_diocotron_ring.gif`, `tut_ep_collapse.gif`, `tut_tfap_field.gif` | **déjà orphelins** : c'était le pool d'assets des tutoriels Sphinx partis vers adc_cases (suppression `tutorials/` 194c63f). Décision Phase 5 : régénérer-avec-provenance pour la nouvelle galerie, OU supprimer. AUCUN n'a de provenance. |

**Conséquence pour le plan :** PR-12/PR-13 (assets) doivent couvrir **deux** dépôts, pas seulement adc_cases. La suppression de la galerie morte (PR-01) **augmente** le nombre d'orphelins adc_cpp de 10 → 16 ; il faut une décision explicite « régénérer pour la nouvelle galerie vs supprimer » sur les 16, avec un manifeste d'assets (script producteur + commande + SHA adc_cpp/adc_cases + backend + résolution) comme l'exige la Phase 5.

### 9.2 §6 ERRONÉ : « 0 embed cassé » sous-estime — il y a des embeds réels côté adc_cpp

Le repo adc_cpp contient bien des embeds markdown/HTML d'images (`README.md:12` `<img src="docs/...">`, `docs/archive/*` et l'ancienne galerie en `![]()`). Le link-check de Phase 6 (`sphinx-build -W`) doit donc couvrir adc_cpp, pas seulement vérifier l'absence d'embeds. Risque réel confirmé : après PR-01, 6 embeds de la galerie morte pointeront vers des pages supprimées — à nettoyer dans le même PR.

### 9.3 À reconfirmer avant publication (inchangé)

Les angles morts de §8 restent valides : couverture adverse limitée (~7 claims), aucune exécution de quickstart.md, compteurs de tests à régénérer canoniquement, et le statut « AmrSystem mono-bloc » dans la **mémoire persistante** du coordinateur doit être corrigé (multi-bloc livré #195/#199/#205).
