# Reference

Cette section est la reference de bas niveau d'`adc_cpp` : la surface Python publique
(autodoc), la reference C++ generee par Doxygen, la matrice de couverture des backends, les
limites connues, et la bibliographie.

C'est le complement du guide utilisateur (les pages [installation](../getting_started/installation.md),
[quickstart](../getting_started/first_run.md), [examples](../getting_started/organisation.md) et les
[tutoriels](../getting_started/tutorial.md)) : le guide montre comment FAIRE, la reference
liste ce qui EXISTE. Pour le detail de conception (contributeurs), chaque page pointe vers les
documents `docs/*.md` du depot, qui restent la source de verite.

Rappel de cadre : le coeur n'expose que des BRIQUES generiques composables (etat, transport,
source, second membre elliptique). Un MODELE est une composition de briques ; aucun scenario
physique nomme (diocotron, euler_poisson...) ne vit dans `adc` -- ces compositions nommees
sont cote application (depot `adc_cases`).

```{toctree}
:maxdepth: 1

api_python
bricks_reference
dsl_reference
api_cpp
backend_matrix
limitations
bibliography
```
