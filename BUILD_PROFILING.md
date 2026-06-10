# Rapport — Profilage de compilation de `adc_cpp`

**Worktree audité :** `/Users/romaindespoulain/.codex/worktrees/8a24/adc_cpp_audit`
**Date :** 2026-06-10 · **Toolchain mesurée :** AppleClang (Xcode), macOS arm64, `-O3 -std=c++2b`, Release
**Objectif :** réduire le temps de compilation **sans changer la numérisation ni le comportement scientifique** (le niveau `-O` ne change pas les résultats IEEE tant que `-ffast-math` n'est pas utilisé — ce qui est le cas ici).

---

## TL;DR

| Constat | Preuve mesurée | Levier |
|---|---|---|
| **`system.cpp` = 469 s, `amr_system.cpp` = 218 s** par compilation (mono-thread) | `/usr/bin/time -p` sur la TU isolée | — |
| Le coût est à **93 % dans le backend LLVM `-O3`**, pas dans le parsing ni le frontend | `-ftime-trace` : Backend 435 s / 468 s ; `-O0` retombe à **41 s** | baisser `-O`, réduire le nb d'instanciations |
| `_adc` ne compile que **3 fichiers** → `-j` plafonne à 3 cœurs, chemin critique = `system.cpp` (~469 s) | `python/CMakeLists.txt:9` | **splitter la TU** |
| `system.cpp`/`amr_system.cpp` recompilés **20×** dans le build de tests (6 + 14) | `grep` sur `build.ninja` | **bibliothèque objet unique** |
| Cause racine = **explosion combinatoire** (≈36 CompositeModels × 4 flux × 4 limiteurs × ~3 intégrateurs ≈ 1700 chemins) toute optimisée à `-O3` dans **une** TU | `model_factory.hpp` + top OptFunction du trace | réduire l'éventail / split |
| Le **linking n'est PAS un problème** (< 1 s par exécutable) | `.ninja_log` : top link = 0,85 s | — |

**Sensibilité au niveau d'optimisation (`system.cpp`, mono-thread, mesuré) :**

| `-O` | temps | gain vs `-O3` |
|------|-------|---------------|
| `-O3` (actuel) | **469 s** | référence |
| `-O2` | 363 s | **−23 %** |
| `-O1` | 284 s | **−39 %** |
| `-O0` | 41 s | **−91 %** |

Les ~41 s à `-O0` sont le coût incompressible (parsing + frontend + codegen non optimisé). Tout le reste (~428 s) est l'optimiseur travaillant sur les ~1700 fonctions feuilles instanciées.

---

## 1. Périmètre & méthode

- **Fichiers de build lus :** [`CMakeLists.txt`](CMakeLists.txt), [`python/CMakeLists.txt`](python/CMakeLists.txt), [`tests/CMakeLists.txt`](tests/CMakeLists.txt), [`.github/workflows/ci.yml`](.github/workflows/ci.yml), [`.github/workflows/docs.yml`](.github/workflows/docs.yml).
- **Builds réutilisés** (dossier principal `Documents/Stage_Romain/adc_cpp/`, même base de code, Release/AppleClang/série) : `build-docs`, `build-audit` (Ninja, `.ninja_log` exploitables), `build-master`, `build-py` (Makefiles, périmé). **Aucun fichier supprimé, aucun clean build.**
- **Mesures propres :** compilation isolée, **mono-thread**, sortie vers `/tmp`, flags repris à l'identique du build de tests (`-O3 -DNDEBUG -std=c++2b -arch arm64 -isysroot … -I include`), avec `-ftime-trace` pour la ventilation interne.

> **Caveat 1.** Les builds réutilisés vivent dans le dépôt principal (`adc_cpp/`), pas dans ce worktree ; les `CMakeLists` sont identiques, donc les `.ninja_log` sont représentatifs. Les sources mesurées (`system.cpp`/`amr_system.cpp`) viennent **du worktree audité**.
> **Caveat 2.** Les durées des `.ninja_log` de `build-docs` (600–934 s par TU) sont **gonflées par la contention** (build parallèle + docs concurrentes). Les chiffres autoritaires de ce rapport sont les compilations **mono-thread isolées** (469 s / 218 s).

---

## 2. Architecture du build (ce qui compile réellement)

- **Le cœur `adc` est `INTERFACE` (header-only)** — `CMakeLists.txt:57`. Il ne compile rien en soi ; tout le coût est porté par les TU qui *l'instancient*.
- **Module Python `_adc` = 3 TU seulement** : `bindings.cpp`, `system.cpp`, `amr_system.cpp` — `python/CMakeLists.txt:9`. C'est ici que se trouve ta douleur « même `_adc` est lent » : il n'y a rien à paralléliser au-delà de 3 cœurs, et le chemin critique est la TU la plus lente (`system.cpp`).
- **Tests = ~113 exécutables** (`grep -c CXX_EXECUTABLE_LINKER build.ninja` → 113 ; 133 objets `.cpp.o`). La majorité sont de petites TU, **mais 20 d'entre elles re-listent `python/system.cpp` ou `python/amr_system.cpp` comme source supplémentaire** (ex. `tests/CMakeLists.txt:204, 304, 334-336, 349, 374, 393, 411, 426, 435…`), donc recompilent intégralement la TU lourde.

---

## 3. Mesures détaillées

### 3.1 Coût par translation unit (mono-thread, propre)

| TU | `-O3` réel | user | Backend % |
|---|---|---|---|
| `python/system.cpp` | **469,4 s** | 379 s | 93 % |
| `python/amr_system.cpp` | **217,6 s** | 192 s | 93 % |

### 3.2 Où part le temps — `-ftime-trace` (`system.cpp`)

```
ExecuteCompiler                 468,1 s
  Frontend                       31,0 s   (parsing + instanciation côté front)
  Backend                       435,6 s   ← 93 %
    OptModule / OptFunction     ~221 s    (optimiseur LLVM -O3)
  Source (parsing en-têtes)       1,2 s   ← négligeable : ce n'est PAS un problème de gros headers
  InstantiateFunction            19,1 s
  PerformPendingInstantiations   18,6 s
```

**Lecture :** ni les en-têtes (1,2 s) ni l'instanciation frontend (31 s) ne sont le goulot. C'est **l'optimiseur backend `-O3`** qui domine. Et il ne s'agit d'aucune fonction chaude isolée : le top OptFunction plafonne à **0,25 s** par fonction — le coût est la **somme de milliers** de petites fonctions instanciées :

```
0,25s  SSPRK3Step::take_step<BlockRhsEvalEb<VanLeer, RusanovFlux, …>>
0,24s  SSPRK3Step::take_step<BlockRhsEvalMasked<Minmod, RusanovFlux, …>>
0,23s  SSPRK3Step::take_step<BlockRhsEval<Minmod, HLLFlux, CompositeModel<…>>>
0,21s  build_block<NoSlope, RusanovFlux, CompositeModel<Euler, PotentialForce…>>
…  (× ~1700 combinaisons)
```

### 3.3 Cause racine — explosion combinatoire des fabriques de dispatch

`include/adc/runtime/model_factory.hpp` imbrique trois dispatchers :

```
dispatch_transport : exb | compressible | isothermal                         (3)
  × dispatch_source : none | potential | gravity | magnetic | potential+mag  (~4)
    × dispatch_elliptic : charge | background | gravity                       (3)
```

≈ **36 `CompositeModel`** ; chacun traverse ensuite `make_block`/`build_block` qui dispatche sur **4 flux** (rusanov/hll/hllc/roe) × **4 limiteurs** (noslope/minmod/vanleer/weno5) × **~3 intégrateurs** (ssprk2/ssprk3/imex) → **~1700 chemins feuilles**, tous **entièrement instanciés et optimisés `-O3` dans une seule TU**. C'est exactement le motif déjà décrit dans le commentaire de `tests/CMakeLists.txt:10-25` (« ~33 CompositeModels × familles de flux × limiteurs »).

### 3.4 Duplication des TU lourdes (build de tests, série)

```
python/system.cpp      compilé  6×   (grep -c "python/system.cpp.o:" build.ninja)
python/amr_system.cpp  compilé 14×
```

- **Coût CPU dupliqué (série) :** 6 × 469 + 14 × 218 ≈ **5 866 s-CPU**.
- **Si compilé une seule fois :** 469 + 218 = **687 s**. → **~88 % de travail gaspillé** (~5 200 s-CPU récupérables).
- Avec `ADC_USE_MPI=ON`, ~8 variantes de test supplémentaires re-compilent ces TU (`test_mpi_amr_*`, `test_mpi_system_*`) → la duplication empire.

### 3.5 Impact CI

`ci.yml` build **6 configurations** (`build`, `build-py`, `build-mpi`, `build-kokkos`, `build-kokkos-omp`, `build-kokkos-py`), **toutes en `--parallel 2`** (contrainte mémoire runner ~7 Go, `ci.yml:111-119`). Chaque config recompile ces TU lourdes depuis zéro :

- `_adc` en CI (`--parallel 2`) : ~469 s + le reste ≈ **8 min minimum**, borné par `system.cpp`.
- Le build de tests complet (`build`, 20 TU lourdes + 113 petites en `--parallel 2`) ≈ **~1 h** sur runner (plus lent que ce Mac).
- Les configs **Kokkos** sont encore plus lourdes (instanciation device via `nvcc_wrapper`).

---

## 4. Causes probables, classées par impact

1. **[Dominant] Optimiseur backend `-O3` sur une explosion combinatoire d'instanciations.** 93 % du temps ; ~1700 fonctions feuilles. Preuve : `-ftime-trace` (Backend 435/468 s), scaling `-O0`→41 s.
2. **[Structurel] `_adc` n'a que 3 TU** → impossible de paralléliser ; chemin critique = `system.cpp` (469 s). Preuve : `python/CMakeLists.txt:9`.
3. **[Gaspillage] `system.cpp`/`amr_system.cpp` recompilés 20× dans les tests** (×6 en CI). Preuve : `grep` sur `build.ninja`.
4. **[Secondaire] Frontend / instanciation 31 s** par TU — réel mais 7× plus petit que le backend.
5. **Non-causes (écartées par les mesures) :** gros headers (Source = 1,2 s), linking (< 1 s/exe), LTO (désactivé — à laisser OFF), Kokkos/pybind dans `system.cpp` (la TU mesurée n'a ni l'un ni l'autre et coûte déjà 469 s).

---

## 5. Cibles les plus coûteuses

| Rang | Cible | Coût `-O3` mono-thread | Multiplicité |
|---|---|---|---|
| 1 | `python/system.cpp` | **469 s** | ×1 dans `_adc`, ×6 dans tests |
| 2 | `python/amr_system.cpp` | **218 s** | ×1 dans `_adc`, ×14 dans tests |
| 3 | `bindings.cpp` | non isolé (inclut pybind11 + façade) | ×1 dans `_adc` |
| 4 | petites TU de test instanciant `make_block` (`test_block_builder`, `test_compiled_model_parity`…) | 15–35 s chacune | ×113 |

---

## 6. Feuille de route P0 / P1 / P2

> Toutes les pistes ci-dessous préservent les résultats numériques : soit elles ne touchent pas au code généré (partition de TU, bibliothèque objet), soit elles ne changent que le niveau d'optimisation (résultats IEEE identiques sans `-ffast-math`).

### P0 — fort impact, risque ~nul

- **P0-A. Compiler `system.cpp` + `amr_system.cpp` UNE fois dans une bibliothèque objet partagée.**
  `add_library(adc_runtime STATIC python/system.cpp python/amr_system.cpp)` liée par `_adc` **et** par les 20 exécutables de test (remplacer les `add_executable(test_x test_x.cpp …/system.cpp)` par un lien vers `adc_runtime`).
  → supprime ~18 recompilations (série) / ~26 (MPI) par config, **×6 en CI**. Gain build de tests : **~88 %** du coût des TU lourdes.
  *Attention :* certains tests passent des `-D` spécifiques (`ADC_TEST_CXX`, `ENABLE_EXPORTS`) — ceux-là restent sur leur propre `.cpp` de test ; seul l'objet `system.cpp`/`amr_system.cpp` est mutualisé. `adc_cap_opt_for_kokkos_ram` (le `-O0` ciblé) s'appliquerait alors une fois à la lib, pas N fois.

- **P0-B. Splitter chaque TU lourde par axe de dispatch pour paralléliser.**
  Partitionner les instanciations de `system.cpp` (p.ex. un `.cpp` par famille de flux, ou par transport) avec instanciation explicite. `_adc` passe alors de 3 à ~8-12 TU → `-j` redevient utile, chemin critique `469 s → ~469/N`.
  → c'est **LE** correctif de ta douleur « `_adc` seul est lent ». Comportement identique (mêmes instanciations, simplement réparties).

### P1 — fort impact, à valider

- **P1-A. Baisser `-O` sur ces 2 TU.** `-O2` = **−23 %** gratuit, `-O1` = **−39 %**. La quasi-totalité des ~1700 fonctions instanciées sont du **câblage de dispatch froid** (exécuté une fois au montage), pas la boucle chaude `for_each_cell`. → mesurer l'impact runtime de `-O2` sur un cas représentatif ; si < 5 %, basculer ces TU (et leurs doublons de test) en `-O2`.
- **P1-B. Découpe chirurgicale du `-O`.** Marquer **uniquement** les fabriques froides (`dispatch_*`, `make_block` glue) en `__attribute__((optnone))` / `#pragma clang optimize off`, en gardant les kernels (`take_step`, `for_each_cell`) à `-O3`. → gros gain compile, **zéro** impact sur le chemin chaud. Plus invasif (il faut isoler le froid).
- **P1-C. Réduire l'éventail combinatoire.** Gater à la compilation les combinaisons (flux × limiteur × modèle) jamais exercées à l'exécution. → moins de feuilles à optimiser. Nécessite de savoir quels combos sont réellement utilisés (décision applicative, pas numérique).

### P2 — confort / non-régression

- **P2-A. `ccache`** (`-DCMAKE_CXX_COMPILER_LAUNCHER=ccache`) pour l'itération locale : aucun gain au 1er build ni quand un en-tête change (la signature `ADC_HEADER_SIG` invalide), mais utile quand seul un `.cpp` de test change.
- **P2-B. Garde-fou CI `-ftime-trace`** : budget de temps de compilation par TU pour détecter une régression (un nouveau flux/limiteur qui rajoute une dimension au produit combinatoire).
- **P2-C. Laisser LTO OFF** (l'activer aggraverait nettement le backend). Le linking n'est pas un sujet (< 1 s/exe) — inutile d'investir dans mold/lld.
- **PCH : faible valeur ici** (le frontend ne pèse que 31 s ; le gain plafonne sous 30 s/TU alors que le backend en coûte 435).

**Ordre recommandé :** P0-A (CI/tests, sûr et immédiat) → P0-B (débloque ta douleur `_adc`) → P1-A/P1-B (raboter le backend) → P1-C (réduire l'éventail).

---

## 7. Commandes utilisées (reproductibilité)

```bash
# Inventaire des builds et options
grep -E '^(ADC_USE_|ADC_BUILD_|CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER):' <build>/CMakeCache.txt

# Durées par cible depuis les .ninja_log (format: start_ms end_ms mtime output hash ; durée = end-start)
#   parsées en Python (dédup par output, tri décroissant) — cf. §3

# Flags exacts d'une TU lourde (depuis build.ninja du build de tests Ninja)
sed -n '2550,2556p' <build-audit>/build.ninja      # → -O3 -DNDEBUG -std=c++2b -arch arm64 -isysroot … -I include

# Mesure propre, mono-thread, avec ventilation interne
SDK=/Applications/Xcode.app/.../MacOSX26.5.sdk
CXX=/Applications/Xcode.app/.../usr/bin/c++
/usr/bin/time -p "$CXX" -O3 -DNDEBUG -std=c++2b -arch arm64 -isysroot "$SDK" -I include \
  -ftime-trace -c python/system.cpp -o /tmp/adc_system.o      # → real 469 s, trace JSON
# idem amr_system.cpp (218 s) ; idem en -O2/-O1/-O0 pour le scaling

# Comptage de la duplication
grep -c "python/system.cpp.o:"     <build-audit>/build.ninja  # → 6
grep -c "python/amr_system.cpp.o:" <build-audit>/build.ninja  # → 14
```

> Les artefacts temporaires (`/tmp/adc_*.o`, traces JSON ~150 Mo) ont été supprimés après analyse. Aucun fichier du projet n'a été modifié ou supprimé ; aucun clean build n'a été lancé.
