# Performance (run-time)

> **STATUT : historique (non-normatif).** Mesures datees (M1, ancien jalon). Les chiffres courants vivent dans PROFILE_RESULTS.md ; ne pas s'y fier comme etat de perf actuel.

> **MESURES HISTORIQUES (pilotes applicatifs, anciennes). NE PAS lire comme la perf actuelle.**
> Une campagne Kokkos/MPI/GPU a jour est en cours (voir docs/PROFILE_RESULTS.md quand disponible).


Methodologie CS:APP : mesurer d'abord, identifier le goulot, transformer, re-mesurer.
Banc : pas couple Euler-Poisson, N=256, Apple M1 Pro (6 coeurs perf + 2 efficiency),
AppleClang -O3 -DNDEBUG.

> Ces mesures ont ete prises avec des pilotes et bancs APPLICATIFS (scenarios nommes,
> harness de figures) qui vivent dans [`adc_cases`](https://github.com/wolf75222/adc_cases),
> pas dans ce depot (le coeur est agnostique au modele). Les chiffres restent representatifs
> du coeur (operateur elliptique, transport, reflux AMR) ; les noms de fichiers/types cites
> ci-dessous (`bench_amr`, `*Config`, scripts de figures) sont ceux du depot applicatif. Les
> figures conservees sont sous `docs/`. La trace des validations device GH200 est dans
> [GPU_RUNTIME_PORT.md](GPU_RUNTIME_PORT.md).

## Profil d'un pas couple (ou part le temps)

| phase | ms/pas | part |
| --- | --- | --- |
| pas couple complet (PerStage) | 53 | 100% |
| hyperbolique (2x assemble_rhs MUSCL + saxpy/lincomb) | 7.7 | 14% |
| **Poisson + aux (2 solves multigrille)** | **45.5** | **86%** |

Le run-time EST le solveur elliptique. L'hyperbolique est negligeable. `-mcpu=native`
ne change rien (~2%) : on n'est pas compute-bound sur l'hyperbolique.

## Leviers mesures

### 1. Politique de couplage : `OncePerStepCoupling` -> x2.6

| politique | ms/pas |
| --- | --- |
| PerStage (Poisson a chaque etage RK, 2 solves/pas) | 52.3 |
| **OncePerStep (Poisson 1x/pas)** | **19.8** |

Gain x2.6 (mieux que x2 attendu : le solve unique part d'un warm-start plus proche
de la convergence). Cout : couplage 1er ordre au lieu de 2e. Accessible via la
`CouplingPolicy` du coeur : `OncePerStepCoupling` au lieu de `PerStage`
(`coupling/coupling_policy.hpp`).

### 2. OpenMP par-kernel : PERDANT ici (ne pas activer tel quel)

| build | 1 thread | 4 threads | 8 threads |
| --- | --- | --- | --- |
| OpenMP (PerStage) | 100 | 91 | 119 |

A comparer aux **52 ms en serie sans OpenMP**. Deux raisons, confirmees a la mesure :
- l'outlining de chaque `for_each_cell` en region parallele **casse l'inlining** des
  boucles chaudes (1 thread OpenMP = 2x plus lent que la serie pure) ;
- la multigrille est **memory-bound et latency-bound** (parcours sequentiel des
  niveaux, GS red-black), mauvais candidat pour du fork/join par noyau.

Sans garde-fou, c'etait **x47 plus lent** (fork/join de 8 threads sur des boites 2x2
des niveaux grossiers). Corrige : `for_each_cell` porte une clause
`#pragma omp parallel for if (n_cells >= 4096)` -> serie sous le seuil. OpenMP n'est
toujours pas un gain ici, mais n'est plus un piege. Le bon grain serait de
paralleliser AU-DESSUS de la boucle de niveaux (region consolidee), pas par noyau.

## Poisson par FFT pour le periodique : FAIT, ~5x

La multigrille est ITERATIVE (plusieurs V-cycles x balayages GS). Pour des CL
PERIODIQUES, `PoissonFFTSolver` (`numerics/elliptic/poisson_fft_solver.hpp`) est un
solveur DIRECT (une transformee), enveloppant `PoissonFFT` au niveau MultiFab et
modelisant le concept `EllipticSolver`. Le `Coupler` est generique sur le backend :
`Coupler<Model, PoissonFFTSolver>` au lieu de `Coupler<Model>` (= MG).

Pas couple Euler-Poisson, N=256 (M1 Pro, -O3, PerStage) :

| backend elliptique | ms/pas |
| --- | --- |
| GeometricMG (iteratif) | 76 |
| **PoissonFFTSolver (direct)** | **16** |

Soit **~4.8x** sur le pas couple, a physique **bit-identique** : les deux inversent
le MEME Laplacien discret 5 points (cf. `test_elliptic_operator` : MG vs FFT, residus
a l'arrondi `~1e-14`, solutions identiques a `~1e-16`). C'est l'optimisation run-time a
fort impact sur les 86% elliptiques. Limite : FFT periodique, mono-rang (le distribue
par bandes MPI est `DistributedFFTSolver`, enveloppe `SpectralCoupler`). Cumulable avec
OncePerStep.

## Banc `bench_amr` : deux-fluides AP + coupleur AMR multi-patch

`examples/bench_amr.cpp`, chronometre sans I/O (M1 Pro, 8 coeurs = 6 perf + 2 efficiency,
Release -O3 -DNDEBUG, backend OpenMP). Run : `OMP_NUM_THREADS=k ./build-omp/bin/bench_amr n nsteps`.

**Deux-fluides AP mono-grille** (2 especes Rusanov + continuite + Poisson multigrille).
Le scaling OpenMP DEPEND DE LA TAILLE :

Balayage propre (`scripts/plot_bench_scaling.py`, figure `docs/fig_openmp_scaling.png`) :

| grille | 1 thread | 2 threads | 4 threads | 8 threads |
| --- | --- | --- | --- | --- |
| n=512 | 11.6 M mailles/s | 19.8 M | **28.3 M (x2.4)** | 26.4 M (plateau) |
| n=768 | 6.4 M mailles/s | 10.5 M | **15.7 M (x2.5)** | 15.6 M (plateau) |

![scaling OpenMP](fig_openmp_scaling.png)

A TRES petite grille (n=384, hors figure) on est overhead-bound : trop de petits noyaux
(`tfap_mstar`, 2x `div_update`, `efield`, 2x `lorentz`) + niveaux grossiers du multigrille,
le fork/join par `for_each_cell` coute plus que le travail -> OpenMP perd. Des n>=512 le
grain par noyau amortit l'overhead : **~x2.4-2.5 a partir de 4 threads** (un run isole a
meme donne x3.6 ; variabilite thermique/ordonnancement), puis **plateau net** au-dela (et
legere baisse a 8). Ce plateau n'est PAS un plafond de coeurs : le M1 Pro a **6 coeurs
performants**. C'est la **saturation de la bande passante memoire** : le stencil + le
multigrille sont a faible intensite arithmetique (bandwidth-bound), ~4 threads suffisent a
saturer le bus memoire, et ajouter des coeurs (perf ou efficiency) n'apporte plus rien.
Masse conservee a `~3e-7` (CFL `dt = 0.4 dx`).

**Elliptique FFT vs multigrille pour le deux-fluides AP : MG GAGNE (contre-intuitif).**
n=512, 60 pas, OMP=4 : MG **9.66 ms/pas**, FFT **23.1 ms/pas** (FFT x0.42, soit 2.4x plus
LENT), a physique bit-identique (`|dev_MG - dev_FFT| = 6.7e-16`). C'est l'INVERSE du pas
couple Euler-Poisson (ou le FFT gagne x4.8). Raison mesuree : le pas Euler-Poisson est
Poisson-domine (86%) et resout 2x/pas (PerStage) ; le pas deux-fluides AP est
TRANSPORT-domine (8 noyaux : 2x mstar, 2x div, efield, 2x lorentz) et ne resout le Poisson
qu'1x/pas AVEC warm-start (la multigrille part du phi du pas precedent -> 1-2 V-cycles
suffisent, moins cher qu'une paire de transformees FFT completes). Conclusion : garder
`GeometricMG` par defaut pour le deux-fluides ; l'avantage FFT est specifique aux couplages
Poisson-domines et par-etage. Lecon : mesurer, ne pas extrapoler d'un solveur a l'autre.

**Coupleur AMR multi-patch** (`AmrCouplerMP`, n=256 + 1 niveau fin, regrid Berger-Rigoutsos) :
~100 ms/pas, 8 patchs, **masse conservee a 6.4e-15 (arrondi machine)**. Gain OpenMP faible
(6.5 -> 5.9 s) : domine par le Poisson MG grossier + la reconciliation multi-box hote
(`mf_find_box`, `mf_average_down_mb` en boucle serie). La masse est conservee a l'arrondi
machine sous regrid dynamique.

Le dispatch par-noyau aide le TRANSPORT a grosse grille (x3.6) mais pas les charges
MG-dominees (couple, petit) : le bon grain reste de paralleliser au-dessus de la boucle de
niveaux.
