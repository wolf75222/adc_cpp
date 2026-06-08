# Manifeste des assets (adc_cpp)

Ce document recense les images suivies par git sous `docs/` du depot `adc_cpp`,
leur surface de reference reelle, leur producteur connu et une decision de
gestion. Il existe parce que la quasi-totalite de ces assets ont ete produits
hors de leur chemin committe et **ne portent aucune provenance enregistree**
(SHA `adc_cpp`, backend, resolution, commande de generation). Le seul jeu
d'assets *avec* provenance tracable est celui du tutoriel canonique
`docs/sphinx/tutorials/_assets/`, documente en section finale.

## Perimetre

Le glob `docs/*.png` + `docs/*.gif` (racine `docs/`, hors `docs/_build/`,
hors `docs/sphinx/tutorials/_assets/`) compte **33 images** : **20 PNG + 13
GIF**. La colonne "Reference par" provient d'un `grep` des fichiers `.md`
(en excluant `docs/_build/`). `docs/DOC_REFONTE_AUDIT.md` est le document
d'audit qui *catalogue* tous ces fichiers ; il n'est pas une surface de doc
vivante et n'est donc pas compte comme reference d'affichage ci-dessous.

## Etat de la provenance

**Aucune** des 33 images de `docs/` ne porte de provenance enregistree. Pour
chacune, on ignore : le SHA `adc_cpp` au moment de la generation, le backend
(prototype / aot / production), la resolution de la grille, le nombre de pas,
et la commande exacte qui l'a produite. Les figures `tut_*` et la galerie
`fig_*`/`anim_*` ont ete committees comme artefacts sortis d'un pipeline
local, pas reconstruites a leur chemin de depot.

La surface de doc *vivante* ne reference pour affichage que **deux** de ces
33 fichiers :

- `anim_romeo_diocotron_amr3.gif` -- embed HTML du hero `README.md:12` ;
- `fig_openmp_scaling.png` -- embed markdown `docs/PERFORMANCE.md:99`.

Tout le reste est soit archive-only (`docs/archive/*.md`), soit orphelin
(ne reste reference que dans le document d'audit, ou plus du tout).

## Legende des decisions

- **keep** : asset d'une surface vivante ; a conserver. Provenance a
  enregistrer (faute de quoi il reste non reproductible).
- **regenerate-with-provenance** : a reconstruire via un script versionne
  emettant un `provenance.json`, si l'asset doit revenir dans la doc.
- **move-to-archive** : asset uniquement utile aux pages d'archive ; a
  conserver avec l'archive (idealement sous `docs/archive/assets/`).
- **delete-orphan** : plus aucune reference vivante ; candidat a suppression.

## GIF (13)

| Fichier | Reference par (hors `_build`, hors audit) | Producteur | Decision |
|---|---|---|---|
| `anim_romeo_diocotron_amr3.gif` | `README.md` (hero, l.12) | inconnu -- run ROMEO/GH200 suppose, non documente | **keep** -- seul GIF de la surface vivante ; provenance ROMEO a enregistrer |
| `anim_magnetic_diocotron.gif` | `docs/archive/ROADMAP.md` | inconnu | **move-to-archive** |
| `anim_diocotron.gif` | aucune (audit seul) | inconnu -- ex-galerie Sphinx morte | **delete-orphan** ou regenerate-with-provenance si reutilise |
| `anim_diocotron_column.gif` | aucune (audit seul) | inconnu -- ex-galerie morte | **delete-orphan** |
| `anim_diocotron_amr3.gif` | aucune (audit seul) | inconnu -- ex-galerie morte | **delete-orphan** |
| `anim_diocotron_multipatch.gif` | aucune (audit seul) | inconnu -- ex-galerie morte | **delete-orphan** |
| `anim_diocotron_amr.gif` | aucune | inconnu | **delete-orphan** |
| `anim_diocotron_mpi.gif` | aucune | inconnu | **delete-orphan** |
| `anim_python_amr.gif` | aucune | inconnu | **delete-orphan** |
| `tut_diocotron_py.gif` | aucune | inconnu -- ex-tutoriels Sphinx (supprimes, commit 194c63f) | **delete-orphan** ou regenerate-with-provenance |
| `tut_diocotron_ring.gif` | aucune | inconnu -- ex-tutoriels | **delete-orphan** ou regenerate-with-provenance |
| `tut_ep_collapse.gif` | aucune | inconnu -- ex-tutoriels | **delete-orphan** ou regenerate-with-provenance |
| `tut_tfap_field.gif` | aucune | inconnu -- ex-tutoriels | **delete-orphan** ou regenerate-with-provenance |

## PNG (20)

| Fichier | Reference par (hors `_build`, hors audit) | Producteur | Decision |
|---|---|---|---|
| `fig_openmp_scaling.png` | `docs/PERFORMANCE.md` (l.99) | `scripts/plot_bench_scaling.py` (cite dans PERFORMANCE.md:92) | **keep** -- seule PNG d'une surface vivante ; suit la decision PERFORMANCE.md, provenance a enregistrer |
| `fig_diocotron_amr_vs_uniforme.png` | `docs/archive/ROADMAP.md` | inconnu | **move-to-archive** |
| `fig_diocotron_conv_modes.png` | `docs/archive/DIOCOTRON_GROWTH_RATE.md` | inconnu | **move-to-archive** |
| `fig_diocotron_highorder.png` | `docs/archive/DIOCOTRON_GROWTH_RATE.md` | inconnu | **move-to-archive** |
| `fig_diocotron_invariants.png` | `docs/archive/DIOCOTRON_GROWTH_RATE.md` | inconnu | **move-to-archive** |
| `fig_diocotron_ml_convergence.png` | `docs/archive/ROADMAP.md` | inconnu | **move-to-archive** |
| `fig_diocotron_reproduction.png` | `docs/archive/ROADMAP.md` | inconnu | **move-to-archive** |
| `romeo_amr_efficiency.png` | `docs/archive/ROMEO.md` | inconnu -- run ROMEO suppose | **move-to-archive** |
| `romeo_growth_mode4.png` | `docs/archive/ROMEO.md` | inconnu -- run ROMEO suppose | **move-to-archive** |
| `romeo_highorder_convergence.png` | `docs/archive/ROMEO.md` | inconnu -- run ROMEO suppose | **move-to-archive** |
| `fig_diocotron_growth.png` | aucune (audit seul) | inconnu -- ex-galerie morte | **delete-orphan** |
| `fig_diocotron_modes.png` | aucune (audit seul) | inconnu -- ex-galerie morte | **delete-orphan** |
| `fig_diocotron_column_growth.png` | aucune | inconnu | **delete-orphan** |
| `fig_diocotron_theory.png` | aucune | inconnu | **delete-orphan** |
| `tut_diocotron_growth.png` | aucune | inconnu -- ex-tutoriels (commit 194c63f) | **delete-orphan** ou regenerate-with-provenance |
| `tut_diocotron_sequence.png` | aucune | inconnu -- ex-tutoriels | **delete-orphan** ou regenerate-with-provenance |
| `tut_euler_poisson.png` | aucune | inconnu -- ex-tutoriels | **delete-orphan** ou regenerate-with-provenance |
| `tut_plasma.png` | aucune | inconnu -- ex-tutoriels | **delete-orphan** ou regenerate-with-provenance |
| `tut_poisson_backends.png` | aucune | inconnu -- ex-tutoriels | **delete-orphan** ou regenerate-with-provenance |
| `tut_tfap_ap.png` | aucune | inconnu -- ex-tutoriels | **delete-orphan** ou regenerate-with-provenance |

## Synthese

- **2 keep** : `anim_romeo_diocotron_amr3.gif` (README), `fig_openmp_scaling.png`
  (PERFORMANCE.md). Surface vivante ; provenance a enregistrer.
- **10 move-to-archive** : figures `fig_diocotron_*` et `romeo_*` plus
  `anim_magnetic_diocotron.gif`, referencees uniquement par `docs/archive/*.md`.
- **21 orphelins** : les 10 fichiers `tut_*` (ex-pool des tutoriels Sphinx
  partis vers `adc_cases`, suppression `tutorials/` commit 194c63f) plus les
  ex-images de la galerie morte et autres `anim_*`/`fig_*` sans reference. Pour
  chacun : **delete-orphan**, ou **regenerate-with-provenance** si l'asset doit
  revenir dans la nouvelle galerie/tutoriel.

Les `tut_*` n'ont **aucune** provenance et ne sont plus references nulle part :
ils sont deja entierement orphelins independamment de toute refonte.

## Assets du tutoriel canonique (avec provenance)

Contrairement a ce qui precede, le tutoriel A->Z vit sous
`docs/sphinx/tutorials/` et **embarque sa provenance**. Le script
`docs/sphinx/tutorials/diocotron_tutorial.py` regenere ses 4 images et ecrit
`docs/sphinx/tutorials/_assets/provenance.json` a chaque execution.

Provenance commune (extraite de `provenance.json`) :

- script : `docs/sphinx/tutorials/diocotron_tutorial.py`
- commande : `python diocotron_tutorial.py --n 96 --steps 60`
- SHA `adc_cpp` : `e58b513d2245c9258a8720b91830b9ee95cafde9`
- backend de compilation : `aot`
- backend d'execution : `serial` (defaut ; cf. getting_started pour Kokkos/MPI)
- resolution : `96x96`, `steps=60`, `cfl=0.4`, Python `3.12.2`
- metriques de controle : `growth_factor=1.5212313128`,
  `mass_drift=1.81e-16`, `amr_uniform_max_delta=0.0717869334`

| Fichier | Dimensions | Provenance |
|---|---|---|
| `docs/sphinx/tutorials/_assets/diocotron_growth.png` | 1104x432 | `provenance.json` (cle `assets`) |
| `docs/sphinx/tutorials/_assets/diocotron_cover.png` | 456x432 | `provenance.json` (cle `assets`) |
| `docs/sphinx/tutorials/_assets/diocotron.gif` | 380x360 | `provenance.json` (cle `assets`) |
| `docs/sphinx/tutorials/_assets/diocotron_uniform_vs_amr.png` | 912x432 | `provenance.json` (cle `assets`) |

Le dossier contient aussi les `.so` compiles associes au run
(`diocotron_aot.so`, `diocotron_production.so`), artefacts du meme pipeline.

Ce jeu est le modele a suivre pour toute regeneration des assets ci-dessus
marques **regenerate-with-provenance** : un script versionne, une commande
reproductible, et un `provenance.json` committe a cote des images.
