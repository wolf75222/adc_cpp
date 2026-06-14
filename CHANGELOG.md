# Changelog

Format : [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/) ; versionnement
[SemVer](https://semver.org/lang/fr/) (0.y.z tant que l'API publique bouge encore).

## Politique de version

- **Source unique** : `project(VERSION x.y.z)` dans `CMakeLists.txt`. Tout en derive :
  `adc.__version__` (bake `ADC_VERSION` dans `_adc`), la wheel pip (regex du `pyproject.toml`),
  `adcConfigVersion.cmake` (install/export). Ne JAMAIS dupliquer le numero ailleurs --
  `docs/Doxyfile` (`PROJECT_NUMBER`) et `docs/sphinx/conf.py` (`release`) doivent suivre a la
  main lors d'un bump (TODO : les generer ; en attendant, les trois fichiers sont la checklist).
- **Bump** : PATCH = correctifs sans changement d'API ; MINOR = ajouts d'API/briques retrocompatibles ;
  MAJOR (post-1.0) = rupture d'API ou d'ABI du chemin DSL production.
- **Tag** : poser `git tag vX.Y.Z` sur master au merge de la PR qui bumpe, puis `git push --tags`.
- A chaque PR notable : une ligne dans `[Non publie]` ci-dessous ; au bump, la section devient
  `[x.y.z] - AAAA-MM-JJ`.

## [Non publie]

### Ajoute

- **Outillage qualite / analyse statique** (ADC-105) : workflow CI dedie `.github/workflows/quality.yml`,
  hors du chemin critique des PR (cron hebdomadaire dimanche + `workflow_dispatch` + label `quality`).
  Cinq jobs *informatifs* (non bloquants) : `clang-format` (`.clang-format`), warnings stricts
  (`ADC_ENABLE_WARNINGS`), `clang-tidy` (`.clang-tidy`), sanitizers ASan+UBSan (`ADC_ENABLE_SANITIZERS`,
  presets `ci-warnings`/`ci-asan`) et CodeQL. Options CMake OFF par defaut (cible `adc_dev_options`
  vide) -> `ci.yml`, les builds locaux et `adc_cases` sont inchanges. Voir `docs/QUALITY_TOOLING.md`.
- Hook generique de PROJECTION PONCTUELLE post-pas (ADC-177) : `m.projection([...])` cote DSL
  (trait C++ `HasPointwiseProjection`, compile comme flux/source), applique par `System` a la fin
  de chaque macro-pas entier (jamais par etage RK) sur les cellules valides de chaque bloc --
  remplace le callback Python par cellule (backends `aot` et `production` ; `prototype` et
  `target="amr_system"` rejettent explicitement). `dsl.sign(x)` (selections par masques sans
  branche, derivable). Cf. `docs/DSL_API.md` section 5.

## [0.1.0] - 2026-06-10

Premiere version numerotee (auparavant `0.0.1` jamais exposee ; Doxygen/Sphinx annoncaient
deja 0.1.0 -- ce bump aligne la source unique CMake dessus).

### Ajoute
- `pip install .` via scikit-build-core : module dans site-packages, sans PYTHONPATH ;
  backends par variables d'environnement (`ADC_USE_KOKKOS=ON Kokkos_ROOT=... pip install .`).
- `find_package(adc)` : regles d'installation/export du coeur header-only (`ADC_INSTALL`).
- `adc.__version__`, `adc.doctor()` (diagnostic complet), `adc.set_threads()` /
  `adc.parallel_info()` / `adc.has_kokkos()`, `_adc.kokkos_is_initialized()`.
- Presets CMake (`python`, `python-parallel`, `serial`, `parallel`, `mpi` + serie `ci-*`
  utilisee par la CI : source unique des flags).
- Environnement conda (`environment.yml`) + `scripts/setup_env.sh` (toolchain par plateforme
  figee dans l'env) + `pixi.toml` (lockfile reproductible multi-plateforme).
- `scripts/kokkos_openmp_conda.sh` (Kokkos Serial+OpenMP dans l'env conda, ~2 min) ;
  `scripts/build_docs.sh` (lint + Sphinx + Doxygen + site en une commande) ;
  profil machine `Tools/machines/romeo/romeo_adc.profile.example`.
- Cle d'ABI du chemin DSL production etendue : jetons `kokkos=` et `stdlib=` (divergences
  auparavant indetectees), en litteral preprocesseur (insensible a l'interposition ELF).
- Gardes toolchain runtime du DSL : compilateur du build bake (`__cxx_compiler__`) et prefere
  au PATH, probe de norme (`c++23`->`c++2b`), garde pre-dlopen module/en-tetes (y compris sur
  cache HIT), erreurs de compilation remontees avec la sortie compilateur.

### Modifie
- `ADC_BUILD_TESTS` suit `PROJECT_IS_TOP_LEVEL` : un consommateur FetchContent ne compile plus
  la suite de tests.
- `import adc` fonctionne sans numpy (`adc.dsl` paresseux, erreur ciblee a l'usage).
- Cache des `.so` DSL : cle machine-aware (arch + optflags) et empreinte de l'install Kokkos
  (`KokkosCore_config.h`) -- un Kokkos different invalide le cache.
- Tests : labels ctest (`core`/`mpi`) + timeouts ; garde memoire `-O0` + pool ninja etendu
  automatiquement a toute cible compilant `system.cpp`/`amr_system.cpp` (39 objets).
- `pybind11` pris dans l'environnement avant tout FetchContent ; ccache auto-detecte ;
  option `ADC_PY_LTO` (OFF par defaut).

### Corrige
- Trois bugs utilisateurs reels du chemin DSL production : compilateur du PATH conda rejetant
  `-std=c++23`, module perime -> dlopen `symbol not found` cryptique, `CalledProcessError`
  sans la sortie du compilateur.
- `find_package(Kokkos)` echouait sur macOS face a un Kokkos OpenMP (hints libomp poses avant
  le `find_dependency(OpenMP)` de KokkosConfig).
- Doc : contradiction pip/PYTHONPATH, mention Catch2 fantome, `$KOKKOS_ROOT` non defini dans
  le tutoriel, claim numpy perime, versions incoherentes (0.0.1 vs 0.1.0).
