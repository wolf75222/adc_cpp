# Translation glossary and conventions

Reference for translating the user-facing documentation of `adc_cpp` from French to
English (milestone "Documentation utilisateur en anglais"). Code comments are out of scope
(a later phase).

## Conventions

- **en-US**, sober technical register. No marketing, no emphasis, no AI filler. Same de-AI
  rules as the rest of the repo (see [DOC_QUALITY.md](DOC_QUALITY.md)).
- Replace the French text; the docs are monolingual English, not bilingual.
- `docs/sphinx/**` stays **ASCII strict** (English is natively ASCII). No em-dash (U+2014)
  anywhere. `docs/check_docs.py` must stay green.
- Unchanged: figures, code blocks, `literalinclude`, identifiers, API names, paths, command
  output. Only prose is translated.
- Domain terms kept as-is: diocotron, ExB, Helmholtz, Schur, Krylov, Berger-Rigoutsos,
  Berger-Oliger, reflux.

## Glossary (FR -> EN)

| French | English |
|---|---|
| maillage | mesh |
| solveur | solver |
| brique(s) | brick(s) |
| coeur | core |
| fraicheur | freshness |
| couplage | coupling |
| flux | flux |
| pas de temps | time step |
| sous-pas | substep |
| sous-cyclage | subcycling |
| bord / halo | ghost / halo |
| raffinement | refinement |
| cible (CMake) | target |
| gabarit | template |
| second membre | right-hand side |
| derive | drift |
| garde-fou | guard, safeguard |
| par defaut | default |
| obligatoire | required |
| piege | pitfall |
| recette / harnais | harness |
| etage source | source stage |
| repli | fallback |
