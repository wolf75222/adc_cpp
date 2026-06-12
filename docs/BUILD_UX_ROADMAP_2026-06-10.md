# Build & UX d'installation -- audit complet et feuille de route (2026-06-10)

Suite de [TOOLCHAIN_ROBUSTESSE_AUDIT_2026-06-10.md](TOOLCHAIN_ROBUSTESSE_AUDIT_2026-06-10.md).
Audit par workflow multi-agents (6 lentilles : projets similaires **via web**, CMake moderne
**via web**, build-system complet, CI, **critique adversariale de nos propres changements**,
parcours utilisateur sur 3 personas) + contre-vérification adversariale des findings critiques.

---

## 1. Ce que font les projets jumeaux (recherche web, sources primaires)

Projets étudiés : **pyAMReX / WarpX** (AMR+GPU+MPI+pybind11 -- les plus proches d'adc_cpp),
openPMD-api, PyKokkos, FEniCSx/dolfinx, OpenMC, PyBaMM, doc officielle pybind11, Scientific
Python Development Guide.

| Sujet | Pattern dominant 2025-2026 | adc_cpp aujourd'hui |
|---|---|---|
| Install du module Python | **`pip install .` via scikit-build-core** (1er choix de la doc pybind11) ou cible CMake **`pip_install`** (pyAMReX/WarpX) → site-packages | PYTHONPATH manuel (béquille dev chez les jumeaux, pas un mode de distribution) |
| Backends optionnels (GPU/MPI) | env vars mappées 1:1 sur des `-D` CMake au moment du `pip install` (`WARPX_COMPUTE=CUDA WARPX_MPI=ON pip install .`) ; variantes conda `=*=mpi_openmpi` ; variants Spack | options CMake + presets (équivalent local, pas pip) |
| Bugs « JIT/runtime avec une autre toolchain » | classe RÉELLE et documentée (cppyy/Cling ABI GCC vs Clang, Numba/JAX PTX vs toolkit). Anti-pattern : **figer/baker la toolchain** | exactement ce qu'on a implémenté (`__cxx_compiler__` baké, probes, clé d'ABI enrichie) ✓ |
| Reproductibilité d'env | `environment.yml` jugé non reproductible (pas de lock) → **pixi** (lockfile auto multi-plateforme, env dans le projet -- HPC-friendly) ou conda-lock | environment.yml (suffisant pour démarrer, lock à considérer) |
| Doc d'install exemplaire | WarpX/openPMD : 3 voies (Users / Developers / **HPC avec profils machine sourçables** `Tools/machines/<cluster>/*.profile.example`) | 1 page installation + tutoriel ; pas de profil ROMEO |

## 2. Corrigé AUJOURD'HUI (2e vague, après l'audit complet)

| Fix | Détail | Validation |
|---|---|---|
| **Clé d'ABI enrichie `kokkos=` + `stdlib=`** (les 2 trous UB confirmés) | [abi_key.hpp](../include/adc/runtime/abi_key.hpp) : `;kokkos=<0|1>` (ADC_HAS_KOKKOS, layouts allocator/types) + `;stdlib=libc++_NNN|libstdc++_NNN` (panachage libc++/libstdc++). Une seule fonction inline → module ET loader cohérents automatiquement | clé imprimée avec/sans `-DADC_HAS_KOKKOS` ; parsers Python (`std=`, `headers=`) insensibles (testé) |
| **Kokkos OpenMP "via conda" en 1 commande** | [scripts/kokkos_openmp_conda.sh](../scripts/kokkos_openmp_conda.sh) : compile Kokkos Serial+OpenMP dans `$CONDA_PREFIX` (~2 min, outillage déjà fourni par l'env). Réponse au paquet conda-forge Serial-only | **testé réellement** : build OK, `KOKKOS_ENABLE_OPENMP` présent, configure adc_cpp → `Kokkos found = (OPENMP;SERIAL)` |
| Hints libomp avant `find_package(Kokkos)` | `KokkosConfig.cmake` fait `find_dependency(OpenMP REQUIRED)` → échouait sur macOS ; macro `adc_apple_libomp_hints()` factorisée (conda puis brew), appelée pour OpenMP ET Kokkos | configure contre le Kokkos OpenMP de test : OK |
| **Consommateur FetchContent ne compile plus les tests** (HIGH confirmé) | `option(ADC_BUILD_TESTS ${PROJECT_IS_TOP_LEVEL})` + bump `cmake_minimum_required(3.21)` (aligné presets) | super-projet de test : **0 cible test** tirée, `app` lie `adc::adc` et tourne ; top-level inchangé (120 cibles) |
| **Version unifiée 0.1.0 + `adc.__version__`** | `project(VERSION 0.1.0)` (Doxygen/Sphinx disaient déjà 0.1.0, CMake disait 0.0.1) → bakée `ADC_VERSION` → `adc.__version__` | bindings compile avec `ADC_VERSION="0.1.0"` |
| Presets durcis | preset caché `conda` avec **`condition: CONDA_PREFIX != ""`** (échec clair au lieu de roots vides silencieux) ; `CMAKE_EXPORT_COMPILE_COMMANDS=ON` (clangd/IDE) ; description `python-parallel` honnête (note Serial-only + remède) | `cmake --list-presets` OK |
| Labels + timeouts ctest | `adc_add_test` → `LABELS core TIMEOUT 600` ; `adc_add_mpi_test` → `LABELS mpi` (un deadlock MPI ne bloque plus le runner) → `ctest -L mpi` possible | configure top-level OK |
| Garde **cache-HIT** (trou trouvé par autocritique) | `check_compiled_matches_module()` au **branchement** (System+AmrSystem `add_equation`) : un `.so` en cache + module périmé → erreur actionnable, plus de dlopen cryptique. Le vérificateur adversarial l'a ensuite **confirmée comme déjà suffisante** | scénario simulé : lève/n-op correctement |
| Divers confirmés | un seul walk+sha256 (réutilisation de la signature) ; `.adc_cache*/` gitignoré ; mention Catch2 fantôme supprimée ; `build-master/python` → `build-py-kokkos/python` ; tutoriel Étape 16 : chemin conda `$CONDA_PREFIX` vs custom `$KOKKOS_ROOT` ; pin pybind11 `<3` levé (le build prend l'env actif, 3.0.4 validé) ; mémoïsation du probe `-std` ; `OMP_PROC_BIND=false` **seulement sur macOS** (NUMA cluster préservé) | py_compile/configure/0 warning |

## 3. Autocritique : verdicts sur notre propre travail

- **Réfuté-en-notre-faveur ×3** : les vérificateurs adversariaux ont tenté de casser nos gardes
  et ont conclu qu'elles couvrent déjà les cas signalés (résolution compilateur, probe c++2b,
  garde cache-HIT).
- **Confirmé contre nous et corrigé** : presets sans condition CONDA_PREFIX ; pin pybind11
  incohérent ; `OMP_PROC_BIND=false` pénalisant cluster ; probe non mémoïsé ; displayName
  `python-parallel` trompeur. → tous corrigés (§2).
- **Connu et assumé** : `set_threads` se base sur un drapeau Python (`_first_system_built`) qui
  ne voit pas une init Kokkos déclenchée par un autre chemin (.so DSL direct). Fix propre =
  exposer `Kokkos::is_initialized()` dans les bindings (roadmap P2).
- **Généreux mais réel** : `generator: Ninja` forcé dans les presets -- assumé (l'env conda
  fournit ninja ; documenté).

## 4. Feuille de route restante (priorisée)

### P1 -- la marche « distribution » (décision de design, PR dédiée)
- **`pyproject.toml` + scikit-build-core** : `pip install .` pilote le CMake existant (une seule
  source de vérité), `pip install -e .` remplace le PYTHONPATH. Backends via env vars → `-D`
  (pattern WarpX : `ADC_KOKKOS=ON ADC_MPI=ON pip install .`). C'est LE standard 2025-2026
  (pybind11, Scientific Python Guide, pyAMReX/WarpX/PyBaMM). Caveat PyBaMM : tester tôt sur
  aarch64 (ROMEO Grace).
- **`install()`/export package-config** (~30 lignes) : `find_package(adc)` pour les consommateurs
  hors FetchContent + prérequis d'un éventuel feedstock conda/Spack. Gardé par `ADC_INSTALL`.

### P2 -- robustesse/confort incrémental
- `_adc.kokkos_is_initialized()` exposé → `set_threads` fiable sur toutes les voies d'init.
- CI : utiliser `cmake --preset` (single source of truth ; la CI réécrit les flags à la main --
  duplication confirmée, chiffrée différemment selon les jobs) ; étendre
  `adc_cap_opt_for_kokkos_ram`/pool aux ~39 cibles qui compilent system/amr_system.cpp.
- Étape configure+build du `bench/` dans ci-full (pourriture silencieuse aujourd'hui).
- Profil machine ROMEO versionné (`Tools/machines/romeo/...profile.example`, pattern WarpX) +
  section « cluster Spack sans root » dans installation.md.
- Doc en une commande (cible `docs` ou script : sphinx+doxygen+deps conda-forge).

### P3 -- opportuniste
- **pixi** (ou conda-lock) pour le lock reproductible multi-plateforme ; pixi stocke l'env dans
  le projet (quotas $HOME cluster).
- cibuildwheel/wheels PyPI le jour où la distribution publique compte ; feedstock conda-forge
  (avec `pybind11-abi` pour la cohérence ABI inter-modules).
- `import numpy` paresseux dans dsl.py (production n'en a pas besoin) ; presets debug/asan ;
  version Kokkos dans la feature-key du cache DSL.

## 5. Réponse nette à « Kokkos parallèle avec conda ? »

Le **paquet** conda-forge `kokkos` : non (Serial-only, CUDA = paquet séparé, pas d'OpenMP).
Mais **oui en pratique** : `bash scripts/kokkos_openmp_conda.sh` compile un Kokkos Serial+OpenMP
**dans l'env conda** en ~2 min avec l'outillage de l'env -- testé de bout en bout sur cette
machine (build + `Kokkos found = (OPENMP;SERIAL)` au configure d'adc_cpp). Le GPU reste
ROMEO/Spack.
