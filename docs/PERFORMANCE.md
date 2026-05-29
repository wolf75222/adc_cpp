# Performance (run-time)

Methodologie CS:APP : mesurer d'abord, identifier le goulot, transformer, re-mesurer.
Banc : pas couple Euler-Poisson, N=256, Apple M2, AppleClang -O3 -DNDEBUG.

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
de la convergence). Cout : couplage 1er ordre au lieu de 2e. Accessible via
`DiocotronConfig::poisson_per_stage = false` (idem EulerPoisson), ou
`Coupler::advance<Limiter, OncePerStepCoupling>`.

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

## Recommandation : Poisson par FFT pour le periodique

La multigrille est ITERATIVE (plusieurs V-cycles x balayages GS). Pour des CL
PERIODIQUES (Jeans, diocotron), `elliptic/poisson_fft.hpp` est un solveur DIRECT
(une transformee) : potentiellement bien plus rapide. Le concept `EllipticSolver`
est en place ; l'etape est d'envelopper `PoissonFFT` au niveau MultiFab et de rendre
`Coupler` generique sur le backend elliptique. C'est le prochain gros levier run-time.
