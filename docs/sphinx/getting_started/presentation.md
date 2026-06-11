# Presentation

`adc_cpp` est un solveur C++23 pour les systemes hyperbolique-elliptique couples sur une
pile de maillage AMR ecrite *from scratch*. Concretement : on transporte une ou plusieurs
densites par un schema de volumes finis (la partie hyperbolique), pendant qu'un Poisson de
systeme fournit a chaque pas le potentiel qui pilote ce transport (la partie elliptique). Le
benchmark de reference est l'instabilite diocotron (derive E x B), mais le coeur ne nomme
aucun scenario : il fournit des briques generiques que l'on compose.

## Ce qu'est `adc_cpp`

- Un coeur header-only, agnostique au modele. La physique se reduit a des lois locales
  (flux, sources, fermetures), device-callable, qui ne voient ni MPI, ni AMR, ni halos. Un
  modele est une composition de briques (`adc.Model(state, transport, source, elliptic)`), pas
  un scenario code en dur.
- Une pile de maillage `from scratch` : conteneurs `Box` / `BoxArray` / `MultiFab` /
  `Geometry`, et une hierarchie AMR block-structured multi-niveaux et multi-patch
  (clustering Berger-Rigoutsos, reflux coverage-aware).
- Un seam de parallelisme unique. Le meme code source bascule entre les espaces d'execution
  Kokkos (Serial sequentiel / OpenMP CPU multi-thread / Cuda-HIP GPU, choisis a l'install de
  Kokkos) et, en option, MPI pour le distribue, via le seam `for_each_cell` et la couche
  `comm`. On n'ecrit aucun kernel CUDA a la main : Kokkos abstrait le materiel.
- Deux solveurs de Poisson : multigrille geometrique (`GeometricMG`, V-cycle Gauss-Seidel
  red-black) et FFT spectrale directe (`PoissonFFTSolver`).
- Des bindings Python via pybind11. Le module `adc` est la facade de composition : Python
  dit quoi (quels blocs, quel schema, quel Poisson), le C++ compile fait le calcul cellule par
  cellule. Aucun va-et-vient numpy dans le hot path.

Le code s'organise en cinq couches orthogonales, ou une couche haute exprime le probleme et une
couche basse l'execute, sans dependance descendante :

| Couche | Quoi |
|---|---|
| **Physique** | lois locales : flux, equation d'etat, sources, fermetures (device-callable) |
| **Numerique** | reconstruction, flux de Riemann, operateur elliptique, conditions aux limites |
| **Maillage / donnees** | `Box`, `BoxArray`, `MultiFab`, `Geometry`, hierarchie AMR |
| **Execution** | seams : `for_each_cell` (Kokkos : Serial / OpenMP / Cuda), `comm` (MPI), allocateur |
| **Temps / couplage** | SSPRK, IMEX, splitting, reflux / average-down / subcyclage |

Le point cle : le modele physique ne depend jamais du backend parallele. Porter sur GPU est
surtout un travail de residence des donnees, pas une reecriture des noyaux de calcul.

## Perimetre honnete

- Kokkos est le seul backend on-node, et il est obligatoire (`-DADC_USE_KOKKOS=ON`, ON par
  defaut ; configurer sans Kokkos est une erreur fatale CMake). La cible se choisit a
  l'installation de Kokkos : sequentiel (Serial), CPU multi-thread (OpenMP) ou GPU (Cuda/HIP).
  Le backend OpenMP autonome (l'option `ADC_USE_OPENMP`) a ete retire ; MPI reste optionnel
  pour le distribue.
- Le GPU (NVIDIA GH200) est valide manuellement sur ROMEO, pas en CI : les runners n'ont pas
  de GPU. La CI tourne sur CPU : le gate obligatoire des PR (`build-and-test`) compile en
  Kokkos Serial (C++ + module Python), et le mode `ci-full` ajoute MPI + Kokkos Serial et
  Kokkos OpenMP. Voir [Verifier son backend](backend.md).
- Le module Python (`_adc`) est compile en Kokkos Serial : il ne route pas vers MPI. Le CPU
  multi-thread ou le GPU s'obtiennent en compilant la facade contre une install Kokkos OpenMP
  ou Cuda/HIP ; le distribue, en ajoutant `-DADC_USE_MPI=ON`.
- Le tutoriel diocotron est un modele reduit (une densite advectee par la derive E x B), le
  benchmark de normalisation, pas une reproduction du systeme Euler-Poisson complet.

## Pour aller plus loin

- Le decoupage en depots (lib vs scenarios) : [Organisation des depots](organisation.md).
- L'architecture detaillee, les seams et les decisions de design : la reference contributeur
  [`ARCHITECTURE.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md) (section 1, "cinq couches orthogonales") et
  [`CHOICES.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/CHOICES.md).
- Les algorithmes (flux, MUSCL/WENO, multigrille, reflux AMR) : [`ALGORITHMS.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md).
