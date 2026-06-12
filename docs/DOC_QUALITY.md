# Politique de qualite documentaire

Ce document fixe la politique de qualite de la documentation d'adc_cpp : comment la doc est
classee, quelles decisions d'outillage ont ete prises et pourquoi, et comment on met la doc a jour
sans la laisser deriver du code.

Principe directeur : la documentation decrit le code reel. Une page qui promet une option, une
cible ou un comportement qui n'existe pas est un bug, au meme titre qu'un test faux. Tout
mecanisme ci-dessous existe pour rapprocher la doc du code et signaler l'ecart quand il apparait.

Suivi Linear : epic ADC-146 (milestone documentation, fraicheur et qualite automatisees). Les
briques d'outillage citees ici arrivent par des issues dediees ; l'etat reel est donne plus bas
dans la section etat du chantier.

## Trois classes de documentation

Chaque page appartient a une classe. La classe fixe la source de verite et le mode de controle.

### A. Reference generee

La source de verite est le code. La page est produite par extraction ; elle ne se relit pas a la
main, elle se reconstruit.

- `docs/sphinx/reference/api_python.md` : autodoc du module `adc` (signatures, docstrings).
- `docs/sphinx/reference/api_cpp.md` plus le site Doxygen publie sous `/cpp/` : API C++ extraite
  des en-tetes.

Controle : la fraicheur se garantit par rebuild. Si une signature change dans le code, la page
change au prochain build ; il n'y a rien a acquitter a la main.

### B. Tutoriels testables

La source de verite est un script execute en integration continue. Le texte n'inclut jamais de
code recopie : il pointe le script par `literalinclude`, donc un exemple qui casse casse le build.

- `docs/sphinx/getting_started/installation.md`
- `docs/sphinx/getting_started/first_run.md`
- `docs/sphinx/getting_started/tutorial.md`
- `docs/sphinx/reference/dsl_reference.md`
- `docs/sphinx/reference/bricks_reference.md`

Controle : le script associe tourne en CI (mode fumee, voir plus bas) et le fragment affiche est
inclus depuis ce meme script. Le code de la page et le code teste sont le meme fichier.

### C. Guides humains

La source de verite est l'intention de l'auteur : architecture, choix, algorithmes, notes de
conception. Ces pages portent un jugement que le code ne contient pas et ne se generent pas.

- `docs/ARCHITECTURE.md` (couches, modules, AMR)
- `docs/ALGORITHMS.md` (methodes, formules)
- `docs/CHOICES.md` (arbitrages assumes)
- `docs/BACKEND_COVERAGE.md` (matrice backends et tests)
- les notes de conception : `docs/DSL_MODEL_DESIGN.md`, `docs/SCHUR_CONDENSATION_DESIGN.md`, les
  `docs/AMR_*_DESIGN.md`, et autres.

Controle : revue humaine plus un controle de fraicheur. Comme rien ne se regenere ici, on suit la
fraicheur par un index (docmap) qui declenche un avertissement quand une page de classe C n'a pas
ete revue depuis une fenetre donnee.

## Decisions d'architecture (ADR)

Les choix ci-dessous sont arretes. Chaque entree donne la decision, la raison, et l'alternative
ecartee.

### Doxysphinx plutot que Breathe ou Exhale

Decision : l'API C++ est rendue par Doxygen, puis integree au site Sphinx par Doxysphinx. Le site
Doxygen brut reste publie tel quel sous `/cpp/`.

Raison : adc_cpp est lourd en C++23 (concepts, templates, foncteurs). Breathe et Exhale rejouent
le parsing C++ cote Python et derivent vite sur ces constructions ; Exhale est peu maintenu.
Doxysphinx part du HTML Doxygen deja produit, donc le rendu suit ce que Doxygen sait lire, sans
second parseur a tenir a jour.

Alternative ecartee : Breathe et Exhale (rot sur les concepts et templates C++23, maintenance
faible d'Exhale).

### docmap central plutot qu'un frontmatter par fichier

Decision : les metadonnees de fraicheur (classe, date de revue, script de test) vivent dans un seul
index, `docs/docmap.toml`, et non dans un entete au sommet de chaque page.

Raison : un index unique se lit, se valide et s'audite d'un coup ; il evite d'avoir a editer chaque
page pour changer la politique, et il laisse les pages propres pour les rendus Sphinx et Doxygen.

Pourquoi TOML : le parseur est `tomllib`, present dans la bibliotheque standard de Python 3.11+. Le
controle de doc tourne donc sans aucune dependance a installer. Les issues de cadrage parlaient de
YAML, mais la contrainte zero dependance prime ; la deviation est assumee.

Alternative ecartee : un frontmatter par fichier (dispersion des metadonnees, edition page par
page) et YAML (dependance externe a installer juste pour lire l'index).

### Exemples executes en mode fumee

Decision : les scripts de tutoriel tournent en CI avec un drapeau `--quick` ; on verifie le code de
sortie (zero) sans assertion sur la physique.

Raison : le but est de prouver que l'exemple s'importe, se compile et s'execute de bout en bout, pas
de revalider le solveur (la suite de tests du coeur s'en charge). Le mode fumee reduit la resolution
et le nombre de pas pour rester dans le budget temps d'une CI doc.

Alternative ecartee : des assertions numeriques dans les tutoriels (doublon de la suite de tests,
fragile, lent).

### Linters retenus : linkcheck, codespell, markdownlint

Decision : la chaine de lint documentaire ajoute le linkcheck Sphinx, codespell et markdownlint, en
plus de `docs/check_docs.py` deja en place.

Raison : ces trois outils couvrent des fautes que `check_docs.py` ne vise pas (liens morts, fautes
de frappe, structure Markdown). Ils sont complementaires, pas redondants.

Alternative ecartee : Vale. Sa valeur (style et termes interdits) recouvre ce que `check_docs.py`
fait deja pour ce depot (ASCII strict des pages Sphinx, em-dash, termes faux), avec un cout de
configuration en plus. Vale est ecarte pour eviter le doublon.

### Cadence CI : lane PR legere et lane lourde hebdomadaire

Decision : deux trajets distincts.

- Lane PR : legere, declenchee par filtre de chemin (seulement quand la doc change), sans
  compilation du module ni du C++. Elle fait tourner le lint et les controles rapides.
- Lane lourde : cron hebdomadaire le dimanche, plus declenchement manuel, plus push sur `master`.
  Elle construit le module pour l'autodoc, Sphinx, Doxygen et les exemples.

Raison : la grande majorite des PR ne touchent pas la doc ; leur imposer un build doc complet serait
du gaspillage. Le travail lourd va sur une cadence ou le temps n'est pas critique.

Politique informative d'abord : au demarrage, ces trajets remontent les ecarts (annotations, resume
de job) sans bloquer les PR. On bascule un controle en bloquant seulement une fois la base assainie.

## Mettre a jour la doc

### Quand editer docmap.toml

On edite `docs/docmap.toml` quand on :

- ajoute, deplace ou supprime une page de doc (mettre a jour son entree et sa classe) ;
- vient de revoir une page de classe C (mettre a jour son champ de revue, voir ci-dessous) ;
- branche un nouveau script de test sur une page de classe B (champ `tested_by`).

On ne touche pas a docmap pour les pages de classe A : leur fraicheur vient du rebuild, pas de
l'index.

### Lire et acquitter un avertissement de fraicheur

Le controle de fraicheur compare la date de revue d'une page de classe C a une fenetre. Au dela, il
emet un avertissement (non bloquant par defaut). Pour l'acquitter :

1. relire la page et corriger ce qui a derive du code ;
2. mettre a jour le champ `reviewed` de l'entree correspondante dans `docs/docmap.toml` ;
3. relancer le controle ; l'avertissement disparait.

Acquitter veut dire avoir relu, pas seulement avoir change la date. Le champ `reviewed` est une
attestation de revue.

### Ajouter un tutoriel testable

1. ecrire un script autonome qui accepte `--quick` (passage de fumee, resolution et pas reduits) et
   `--outdir DIR` (ou ecrire les figures), sur le modele de
   `docs/sphinx/tutorials/diocotron_tutorial.py` ;
2. afficher les fragments dans la page par `literalinclude` du script, jamais par copie a la main ;
3. declarer le script dans le champ `tested_by` de l'entree docmap de la page, pour que le harnais
   d'exemples le ramasse ;
4. verifier en local que le passage `--quick` sort en zero.

### Commandes locales

```bash
python docs/check_docs.py                       # lint documentaire (em-dash, ASCII, liens, termes)
bash scripts/build_docs.sh --sphinx             # lint plus Sphinx seul (iteration rapide)
python docs/run_doc_examples.py                 # joue les scripts de tutoriel en mode fumee
cmake --build --preset serial --target docs     # build doc complet via la cible CMake
```

`check_docs.py` et `build_docs.sh` existent deja. `run_doc_examples.py` et la cible CMake `docs`
arrivent avec les issues d'outillage (voir ci-dessous) ; tant qu'elles ne sont pas mergees, la
commande correspondante n'existe pas encore.

## Etat du chantier

La doctrine impose de dire ce qui existe et ce qui reste a livrer. Ce tableau est l'etat reel.

| Capacite | Etat aujourd'hui | Livre par |
| --- | --- | --- |
| Lint documentaire | present : `docs/check_docs.py` | deja la |
| Build doc unifie | present : `scripts/build_docs.sh` | deja la |
| Publication Pages opt-in | present : `.github/workflows/docs.yml` | deja la |
| Tutoriel literalinclude avec `--quick` | present : `docs/sphinx/tutorials/diocotron_tutorial.py` | deja la |
| Index docmap central | absent | ADC-147 |
| Flags `--freshness-warn-only` et `--selftest` | absents de `check_docs.py` | ADC-147 |
| Harnais d'exemples `run_doc_examples.py` et `tested_by` | absent | ADC-148 |
| Doxygen embarque (Doxysphinx) sous `docs/sphinx/doxygen/` | absent | ADC-149 |
| Linters linkcheck, codespell, markdownlint | absents | ADC-150 |
| Lane PR legere et lane lourde cron (`docs-pr.yml`) | absentes | ADC-151 |

Les fichiers `docs/docmap.toml`, `docs/run_doc_examples.py` et `.github/workflows/docs-pr.yml` ne
sont pas encore dans l'arbre ; ils arrivent par les PR des issues citees. Ce document fixe le
contrat qu'elles realisent ; il ne suppose pas qu'elles sont deja la.
