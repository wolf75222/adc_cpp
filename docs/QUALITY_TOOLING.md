# Outillage qualité & analyse statique

`adc_cpp` est doté d'une chaîne d'analyse statique / qualité de code **volontairement séparée du
chemin critique des PR**. Le gate `ci.yml` reste l'autorité sur la compilabilité et la correction
(tests) ; la qualité tourne à part, sans ralentir le cycle de développement.

Suivi Linear : epic **ADC-105** (milestone *Qualité de code & CI durcie*).

## Déclencheurs (`.github/workflows/quality.yml`)

Le workflow `Quality` **ne tourne jamais sur un push ni sur une PR ordinaire**. Il se déclenche sur :

| Déclencheur | Quand |
| --- | --- |
| `schedule` (cron `0 4 * * 0`) | chaque **dimanche** ~04:00 UTC |
| `workflow_dispatch` | manuellement (onglet *Actions* → *Quality* → *Run workflow*) |
| label `quality` sur une PR | passe complète **opt-in** sur cette PR (PR à risque) |

> Le label `quality` doit exister dans le dépôt :
> `gh label create quality --description "Déclenche quality.yml sur cette PR" --color FBCA04`.
> Le workflow ne prend effet (cron, dispatch, label) qu'une fois présent sur la branche par défaut
> (`master`).

## Politique : informatif d'abord

Au démarrage, **rien n'est bloquant** : les remontées apparaissent en annotations GitHub, en résumé
de job et en artefacts, mais ne font pas échouer le run (pas de `-Werror`, `clang-format` en
`--dry-run`, `clang-tidy` sans `WarningsAsErrors`). Toutes les options CMake sont **OFF par défaut**
→ `ci.yml`, les builds locaux et `adc_cases` sont inchangés. Une fois la base assainie, on pourra
basculer tel ou tel job en bloquant.

## Les cinq jobs

| Job | Outil | Config | Preset / option |
| --- | --- | --- | --- |
| `format` | clang-format | `.clang-format` | — (pas de build) |
| `warnings` | gcc `-Wall -Wextra …` | `cmake/AdcDevTooling.cmake` | preset `ci-warnings` (`ADC_ENABLE_WARNINGS`) |
| `tidy` | clang-tidy | `.clang-tidy` | preset `ci-kokkos` (compile DB) |
| `sanitizers` | ASan + UBSan | `cmake/AdcDevTooling.cmake` | preset `ci-asan` (`ADC_ENABLE_SANITIZERS`) |
| `codeql` | CodeQL C++ | suite `security-and-quality` | preset `ci-kokkos` (build tracé) |

Warnings et sanitizers sont portés par une cible **`INTERFACE adc_dev_options`** que **seules** les
cibles internes lient en `PRIVATE` (les ~140 tests via `adc_add_test`, le module `_adc`). Le cœur
public `adc::adc` n'est jamais touché → aucun flag ne fuit vers les consommateurs.

CodeQL est gratuit ici car le dépôt est **public** ; les résultats remontent dans
**Security › Code scanning**.

## Reproduire en local

```bash
# Style
clang-format --dry-run --Werror include/adc/**/*.hpp     # signale ; -i pour appliquer

# Warnings stricts (Kokkos requis : env conda 'adc' actif, ou KOKKOS_PREFIX pointant une install)
cmake --preset parallel -DADC_ENABLE_WARNINGS=ON
cmake --build --preset parallel

# Sanitizers ASan+UBSan
cmake --preset parallel -DADC_ENABLE_SANITIZERS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build --preset parallel
ASAN_OPTIONS=detect_leaks=0:detect_container_overflow=0 ctest --preset parallel --output-on-failure

# clang-tidy (après un configure qui exporte compile_commands.json)
run-clang-tidy -p build 'tests/.*\.cpp'
```

> En CI, les jobs réutilisent l'install Kokkos Serial **mise en cache** par le gate (`ci.yml`), via
> l'action composite `.github/actions/setup-kokkos` (même clé de cache).

## Hors scope (extensions futures)

- Balayage `clang-format` de toute la base (reformat massif) — PR séparée.
- Passage en `-Werror` / blocage des PR — après assainissement.
- TSan, couverture, include-what-you-use, cppcheck.
