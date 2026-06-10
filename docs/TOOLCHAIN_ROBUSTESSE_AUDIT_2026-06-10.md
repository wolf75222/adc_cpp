# Audit toolchain & robustesse d'installation — état des lieux (2026-06-10)

**Déclencheur :** trois bugs réels chez des utilisateurs tiers (machines de collègues), tous de la
même classe « **environnement du build ≠ environnement d'exécution** » :

| # | Symptôme observé | Cause racine | Statut |
|---|---|---|---|
| 1 | `error: invalid value 'c++23' in '-std=c++23'` (env mambaforge) | le DSL compilait ses `.so` runtime avec le compilateur du **PATH** (vieux gcc/clang conda) et l'orthographe **`c++23` littérale**, alors que `_adc` est bâti en `-std=c++2b` par un autre compilateur | **CORRIGÉ** (3 protections, validées par repro) |
| 2 | `dlopen : symbol not found in flat namespace '__ZN3adc6System13install_block...'` | module `_adc` **périmé** vs en-têtes (build avant un `git pull`) : le loader DSL référence une signature C++ que le vieux `.so` n'exporte pas ; le garde ABI ne tourne **jamais** car le dlopen échoue avant ([native_loader.hpp:634](../include/adc/runtime/native_loader.hpp) dlopen < ligne 647 lecture de clé) | **CORRIGÉ** (garde pré-dlopen, validée par repro) |
| 3 | `subprocess.CalledProcessError: Command [...] returned non-zero exit status 1` | `subprocess.run(check=True)` sans capture : l'erreur du compilateur n'est pas remontée dans l'exception | **CORRIGÉ** (`_run_compile`, stderr + remèdes) |

Audit mené par workflow multi-agents (4 lentilles : `dsl.py`, CMake, classe de bugs env, réalité
conda-forge) + **contre-vérification adversariale** des findings critiques. Fait notable : les deux
findings HIGH sur le compilateur/std ont été *réfutés* par les vérificateurs **parce que les fixes
livrés pendant l'audit les corrigeaient déjà** — confirmation indépendante des causes racines.

---

## 1. Ce que le build bake dans `_adc` (après cette session)

| Attribut Python | Source CMake | Rôle |
|---|---|---|
| `__cxx_std__` (20/23) | `ADC_CXX_STD` | norme C++ du loader DSL (déjà présent) |
| `__has_kokkos__` | `ADC_HAS_KOKKOS` | backend de calcul compilé (set_threads/doctor/parité loader) |
| **`__cxx_compiler__`** *(nouveau)* | `CMAKE_CXX_COMPILER` | **le seul compilateur garanti ABI-compatible** pour les loaders DSL (la clé d'ABI encode `__VERSION__`) |
| `abi_key()` | `__VERSION__`+`__cplusplus`+`ADC_HEADER_SIG` | garde ABI C++ du chemin production |

## 2. Chaîne de résolution du compilateur DSL (nouvelle, centralisée)

`_default_cxx()` ([dsl.py](../python/adc/dsl.py)) : `cxx=` explicite → `$ADC_CXX` → **compilateur
du build** (`__cxx_compiler__`, si présent sur la machine) → PATH (`c++`/`g++`/`clang++`,
historique). Le chemin Kokkos/CUDA garde sa priorité (`ADC_KOKKOS_CXX`, `nvcc_wrapper` explicite).

Avant **chaque** compilation : `_probe_cxx_std()` vérifie `-std=` (probe `-fsyntax-only`), rétrograde
`c++23`→`c++2b` (même niveau, même `__cplusplus` sur compilateurs récents) si besoin, sinon lève une
erreur **actionnable** (compilateur utilisé, compilateur du build, 3 remèdes). Échec de compilation →
`_run_compile` remonte la **sortie du compilateur** dans l'exception.

## 3. Gardes pré-dlopen (bug #2)

- `_check_headers_match_module(include)` : compare la signature d'en-têtes **bakée** dans `_adc`
  (token `headers=` d'`abi_key()`) à celle de l'arbre `include/` utilisé. Divergence → erreur claire
  « module périmé, rebâtir » AVANT toute compilation/dlopen. Branchée sur `compile_native` et
  `HybridModel.compile` (chemins natifs).
- `adc.doctor()` expose le même check (`headers_sync`).
- L'import de `adc._adc` qui échoue donne désormais un message actionnable (extension absente /
  mauvais interpréteur cpython-3XY, avec le remède exact) au lieu du `ModuleNotFoundError` brut.

## 4. Parité Kokkos & cache (.so) — durcissements

- **Confirmé par vérif adversariale** : la clé d'ABI C++ n'encode PAS `ADC_HAS_KOKKOS` (layouts
  `allocator.hpp`/`types.hpp` divergents = UB potentiel) ni libc++/libstdc++. Chantier C++ proposé
  (§6). En attendant, **mitigation Python** : `_warn_kokkos_parity()` avertit quand
  `_adc.__has_kokkos__` et `ADC_KOKKOS_ROOT` divergent (module Kokkos + loader série = perf muette ;
  l'inverse = risque réel).
- Clé de cache des `.so` : ajout de **l'architecture CPU + `ADC_DSL_OPTFLAGS`**
  (`_platform_cache_key`) — un `.so` x86_64/`-march=native` ne sera plus réutilisé sur une autre
  machine via un cache partagé (SIGILL silencieux).

## 5. Réalité conda-forge (lentille réseau ; à reconfirmer au premier `env create`)

- **`cxx-compiler`** : clang 19 (osx-arm64) / gcc 14 (linux-64) — **tous deux C++23-capables**.
  Un toolchain C++23 garanti par conda est donc possible, MAIS en mode **tout-conda cohérent**
  (bâtir `_adc` avec lui ET `ADC_CXX=$CONDA_PREFIX/bin/...` ; ne pas mélanger avec AppleClang).
- **`kokkos` conda-forge = backend Serial UNIQUEMENT** (pas d'OpenMP ; CUDA = paquet séparé).
  Suffit pour compiler/valider `ADC_USE_KOKKOS=ON`, mais **ne scale pas en threads**. Multi-thread
  réel : Kokkos compilé `-DKokkos_ENABLE_OPENMP=ON` (local) ou Spack/ROMEO. Documenté dans
  `environment.yml` + `installation.md` (vérifier `Kokkos found ... = (...)` au configure).
- Les sélecteurs `# [linux]`/`# [osx]` **ne fonctionnent pas** dans `environment.yml`
  (conda-build seulement) → un seul yml portable, variantes via envs séparés ou conda-lock.
- Le `gcc` d'un env mambaforge macOS est un shim/clang déguisé, souvent vieux → exactement le
  piège du bug #1 ; couvert par la chaîne de résolution §2.

## 6. Chantiers restants (propositions, non faits ici)

| Prio | Chantier | Pourquoi | Nature |
|---|---|---|---|
| P1 | Jeton `kokkos=<0|1>` **dans `abi_key_string()`** (C++ + CMake + dsl) | mismatch Kokkos = UB non détecté (confirmé) ; le warning Python est une mitigation, pas une garantie | PR dédiée (change la clé d'ABI → invalide les caches, touche les tests ABI) |
| P1 | Jeton `stdlib=` (`_LIBCPP_VERSION`/`__GLIBCXX__`) dans la clé d'ABI | libc++ vs libstdc++ = ABI std::string/function divergente non détectée (confirmé) | même PR que ci-dessus |
| P2 | `import numpy` paresseux dans `dsl.py` | numpy absent tue `import adc` entier alors que le chemin production n'en a pas besoin | petit refactor dsl |
| P2 | Signer la **version Kokkos** dans la feature-key du cache | un Kokkos différent entre build et runtime n'invalide pas le cache `.so` | dsl.py |
| P2 | Warning configure si `CONDA_PREFIX` vide quand un preset conda est utilisé | presets parallel/mpi supposent l'env activé | CMakePresets/CMake |
| P3 | SDK macOS dans la clé de cache | bas risque, documenté | optionnel |

## 7. Parcours utilisateur cible (après cette session)

```bash
# 1× : tout l'outillage (cmake récent, ninja, ccache, python, numpy, pybind11, kokkos*, openmpi)
conda env create -f environment.yml && conda activate adc

# une commande par mode (flags bakés dans CMakePresets.json)
cmake --preset python          && cmake --build --preset python            # série
cmake --preset python-parallel && cmake --build --preset python-parallel   # Kokkos*

# diagnostic en 1 commande (à donner aux collègues au moindre souci)
python -c "import adc; adc.doctor()"

# threads en 1 ligne (plus de variables d'env à connaître)
python -c "import adc; adc.set_threads(); ..."
```

\* sous la réserve Kokkos-Serial du §5 pour le scaling réel.

**Toute erreur des chemins fragiles est désormais actionnable** : compilateur trop vieux (probe),
module périmé (garde signature), mauvais interpréteur (garde import), échec de compile (stderr
remonté), divergence Kokkos (warning), et `adc.doctor()` vérifie le tout a priori.
