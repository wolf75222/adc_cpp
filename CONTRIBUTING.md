# Contribuer a adc_cpp

`adc_cpp` est le coeur C++23 (header-only) du solveur ADC, avec ses bindings Python
(pybind11), son chemin DSL et son packaging CMake. Ce guide resume la methodo ; le
detail technique vit dans le [README](README.md) et dans
[docs/DOC_QUALITY.md](docs/DOC_QUALITY.md).

## Build et tests (presets CMake)

Le build est pilote par les presets de `CMakePresets.json`, pas par des `-D` ad hoc.
L'environnement conda `adc` (Python 3.12) doit etre actif pour les presets `python`,
`parallel` et `mpi`.

```bash
bash scripts/setup_env.sh && conda activate adc   # env + toolchain figee

cmake --preset serial   && cmake --build --preset serial   && ctest --preset serial
cmake --preset python   && cmake --build --preset python    # module _adc (bindings)
cmake --preset mpi      && cmake --build --preset mpi      && ctest --preset mpi
cmake --preset parallel && cmake --build --preset parallel && ctest --preset parallel
```

Les presets CI (`ci-serial`, `ci-python`, `ci-mpi`, `ci-kokkos`, `ci-kokkos-python`,
`ci-bench`) refletent `.github/workflows/ci.yml` : aligner ses flags sur un job CI
plutot que d'en inventer. Les chemins GPU / GH200 ne sont pas validables hors ROMEO :
le dire explicitement dans la PR.

## Documentation

`bash scripts/build_docs.sh` construit tout le site (lint + Sphinx + Doxygen +
doxysphinx). La politique de fraicheur (docmap, lanes CI) est decrite dans
[docs/DOC_QUALITY.md](docs/DOC_QUALITY.md). La lane PR legere (`docs-pr.yml`) ne compile
rien ; le build complet tourne sur le cron hebdo et le dispatch manuel.

## Workflow

- **Linear** est la source de verite des taches : une issue `ADC-NN` = une PR.
- Branche : `adc-<n>-description-courte`. Titre de PR : `ADC-<n> Description`. Corps :
  `Fixes ADC-<n>`.
- `master` est la branche par defaut ; jamais de commit direct dessus. Livrer via une
  branche ou un `git worktree` isole depuis `master`.
- Diffs minimaux, cibles sur l'issue ; pas de reformatage parasite.

## Garde-fous

- **Aucun auteur, committer ou co-auteur IA** (Claude, Copilot, Anthropic...) nulle part
  dans l'historique : le workflow `no-ai-authors.yml` refuse ces commits a la source (le
  squash GitHub hoiste les trailers `Co-authored-by`). Identite git par defaut.
- Style des docs : ASCII strict pour `docs/sphinx/**.md`, zero em-dash partout ; ces
  regles sont verifiees par `docs/check_docs.py` (lance par `build_docs.sh` et la lane PR).

## Licence

En contribuant, vous acceptez que vos apports soient publies sous licence BSD-3-Clause
(voir [LICENSE](LICENSE)).
