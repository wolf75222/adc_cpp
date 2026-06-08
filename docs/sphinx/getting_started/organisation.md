# Organisation des depots

Le projet se decoupe en **deux depots** avec une frontiere nette : la bibliotheque generique
d'un cote, les scenarios nommes de l'autre. Comprendre cette separation evite de chercher
"le diocotron" au mauvais endroit.

## `adc_cpp` -- la bibliotheque (le coeur)

<https://github.com/wolf75222/adc_cpp>

C'est ce depot. Il contient :

- le **coeur C++ header-only** (`include/adc/`) : briques physiques, schemas numeriques, pile de
  maillage AMR, solveurs de Poisson, seams de parallelisme ;
- les **bindings Python de la lib** (`python/`) : le module `adc` (pybind11), c'est-a-dire les
  facades de composition `System` / `AmrSystem` et le DSL `adc.dsl` ;
- la **suite de tests** (`tests/`, `python/tests/`) et la doc de conception (`docs/`).

Regle structurante : **le coeur est AGNOSTIQUE au modele**. Il ne nomme aucun scenario. Il ne
connait que des briques generiques (`Scalar` / `FluidState`, `ExB` / `CompressibleFlux`,
`NoSource` / `PotentialForce`, `BackgroundDensity` / `ChargeDensity`...) que l'on compose en un
modele via `adc.Model(state, transport, source, elliptic)`, ou que l'on ecrit en formules via
`adc.dsl.Model`. Il n'y a ni `diocotron`, ni `euler_poisson` dans `adc_cpp`. Voir
[`ARCHITECTURE.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md) ("Le coeur est AGNOSTIQUE au modele").

Cette regle vaut aussi pour les integrateurs sur mesure : le schema deux-fluides
asymptotic-preserving, qui n'est PAS une brique composable bloc a bloc mais un SCENARIO, a quitte
le coeur. Il vit cote applications, compile a la volee contre les en-tetes generiques.

## `adc_cases` -- la couche applicative (les scenarios nommes)

<https://github.com/wolf75222/adc_cases>

Depot frere. Il ne contient **que du code Python** qui importe le module `adc` et compose les
briques en scenarios NOMMES : `diocotron`, `euler_poisson`, deux especes, isotherme magnetique,
two-fluid AP... Chaque cas est une recette de composition (quelles briques, quel schema, quel
Poisson, quelle condition initiale), pas du nouveau coeur.

Quelques demonstrateurs de reference (executes en CI cote `adc_cases`) :

| Scenario | Fichier |
|---|---|
| ExB mono-espece (DSL) | `diocotron_dsl/run.py` |
| Deux especes (DSL) | `two_species_dsl/run.py` |
| Isotherme magnetique (DSL) | `magnetic_isothermal_dsl/run.py` |

## Qui depend de qui

```
adc_cpp  (coeur generique + bindings adc)   <-- agnostique, ne nomme aucun scenario
   ^
   | import adc  /  FetchContent adc::adc
   |
adc_cases  (scenarios Python nommes)         --> diocotron, euler_poisson, ...
```

La dependance est strictement descendante : `adc_cases` consomme `adc_cpp` (le module Python, ou
les en-tetes `adc::adc` via `FetchContent` pour les scenarios compiles a la volee). **Il n'y a
aucune dependance inverse** : le coeur ne connait pas les cas.

En pratique : un nouveau modele "bien connu" (un diocotron, un Euler-Poisson) se code dans
`adc_cases` par composition de briques. Une nouvelle BRIQUE generique (un nouveau flux, une
nouvelle fermeture elliptique) se code dans `adc_cpp`. Le [tutoriel A->Z](tutorial.md) reste, lui,
volontairement AUTONOME : il n'importe que `adc`, `numpy` et `matplotlib`, et recree le modele
diocotron a la main -- sans dependance a `adc_cases` -- pour rester reproductible depuis ce seul
depot.
