# Sujets avances

Cette page rassemble les fonctionnalites qui depassent la boucle diocotron de base
(transport E x B + Poisson) : solveurs elliptiques generalises, sources couplees
inter-especes, etage source condense par Schur, geometrie polaire / disque, extension
du coeur en C++, et profilage de performance.

Chaque section resume l'essentiel pour un utilisateur et renvoie vers la reference
contributeur (`docs/*.md`) pour le detail algorithmique et les preuves de validation.
Les API montrees ici sont verifiees contre le code du depot (bindings, tests, en-tetes).

## Poisson : solveurs elliptiques

L'etage elliptique resout `lap(phi) = f` (ou une generalisation) a chaque pas, et c'est
le coeur du couplage : `f` depend de la densite, et `phi` (via `grad phi`) pilote la
derive. Le solveur se choisit par mot-cle dans `set_poisson` :

```python
import adc

sim = adc.System(n=128, L=1.0, periodic=True)
# ... add_block / add_equation ...
sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="auto")
```

Le `solver=` accepte `"geometric_mg"` (defaut) ou `"fft"`. Le `rhs=` vaut
`"charge_density"` (second membre `q n`) ou `"composite"` (somme des contributions de
blocs). `bc=` vaut `"auto"`, `"periodic"`, `"dirichlet"`.

### GeometricMG (multigrille geometrique)

`GeometricMG` est le solveur par defaut : un V-cycle multigrille avec lisseur
Gauss-Seidel rouge-noir (le coloriage rend chaque balayage independant des donnees,
donc parallelisable et device-clean). Le cycle lisse `nu1` fois, restreint le residu sur
une grille deux fois plus grossiere (`average_down`), recurse, prolonge la correction,
relisse `nu2` fois. Cout O(N), convergence quasi independante du maillage. Le coarsening
s'arrete proprement des qu'une boite ne se divise plus exactement (garde-fou
`coarsen(2).refine(2) == b`), ce qui evite les hierarchies degenerees sous AMR / multi-box.

Le meme operateur multigrille couvre trois generalisations du Laplacien, toutes opt-in
et bit-identiques au chemin historique tant qu'on ne les active pas. Cote C++
(`GeometricMG`, `numerics/elliptic/geometric_mg.hpp`) :

- `set_epsilon(eps)`, permittivite variable `div(eps(x) grad phi) = f` (moyenne
  harmonique aux faces) ;
- `set_reaction(kappa)`, operateur ecrante / Helmholtz `div(eps grad phi) - kappa phi = f`
  (ecrantage de Debye, `kappa = 1 / lambda_D^2`) ;
- `set_epsilon_anisotropic(eps_x, eps_y)`, milieu tensoriel diagonal `div(diag(eps_x, eps_y) grad phi) = f`.

Ces trois reglages sont composables ; `eps_x == eps_y` redonne l'isotrope, ne pas appeler
`set_reaction` redonne Poisson pur. Cote Python, ils sont exposes par champ NumPy
au niveau du `System` (les coefficients vivent dans les `for_each_cell` device du smoother) :

```python
import numpy as np

eps   = np.ones((n, n))            # permittivite variable (set_epsilon C++)
kappa = np.zeros((n, n))           # terme de reaction kappa >= 0 (Helmholtz/ecrante)

sim.set_epsilon_field(eps)                       # div(eps grad phi) = f
sim.set_reaction_field(kappa)                    # - kappa phi
sim.set_epsilon_anisotropic_field(eps_x, eps_y)  # diag(eps_x, eps_y)
```

### Poisson spectral (FFT)

Sur un domaine periodique, le Laplacien est diagonal en Fourier :
`phi_hat(k) = -rhs_hat(k) / (k_x^2 + k_y^2)`, mode `k=0` epingle a 0 (jauge). Une FFT
directe + division + FFT inverse resout Poisson exactement (residu machine), sans
iteration. Deux variantes existent, toutes deux modeles du concept `EllipticSolver` :

- `PoissonFFTSolver` (`numerics/elliptic/poisson_fft_solver.hpp`), mono-rang, boite
  unique. Son constructeur leve un `std::runtime_error` des que `n_ranks() != 1` ou que
  `ba.size() != 1`. Ce garde-fou est delibere et actif en Release (`NDEBUG` ne le retire
  pas) : ce solveur direct dereferencerait `fab(0)` sur un rang sans box (segfault). En
  serie, c'est le solveur exact et le plus rapide pour un domaine periodique.
- `DistributedFFTSolver` (meme en-tete), FFT distribuee par bandes (slabs) : 1 bande
  par rang, transposee parallele par `MPI_Alltoall`. C'est le pendant MPI du FFT direct,
  utilisable comme `Coupler<Model, DistributedFFTSolver>`. Contraintes : `Ny` divisible par
  `n_ranks()`, `Nx`/`Ny` puissances de 2 (un correctif gere `n` non puissance de 2 cote
  mono-rang). En serie (`n_ranks() == 1`) une seule bande couvre le domaine, identique a
  `PoissonFFTSolver`.

MG et FFT inversent prouvablement le meme Laplacien discret : le meme operateur canonique
`poisson_residual` applique a leurs deux solutions donne des residus a l'arrondi
(`~1e-14`) et des solutions identiques a `~1e-16`. Le piege du FFT : il exige le
periodique, et le second membre doit etre a moyenne nulle (sinon `phi` derive).

### Pour aller plus loin

- Algorithmes elliptiques (multigrille, FFT, eps/Helmholtz/anisotrope, cut-cell) :
  [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md), sections 9 a 12.
- Les en-tetes : `include/adc/numerics/elliptic/geometric_mg.hpp`,
  `poisson_fft_solver.hpp`, `poisson_operator.hpp`.
- Proprietes de conservation du schema couple (masse exacte FV, momentum, energie, valeurs
  mesurees par les tests) : [CONSERVATION_SUMMARY.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/CONSERVATION_SUMMARY.md).

## Sources couplees inter-especes

Au-dela du transport et de la source locale d'un bloc, on peut decrire un couplage
inter-especes (ionisation, collisions, echange thermique) en formules, sans ecrire de
C++ et sans callback Python par cellule. Le DSL `adc.dsl.CoupledSource` transporte la
formule en bytecode pile-machine, interprete cote C++ dans un `for_each_cell` device
(donc MPI-safe et GPU-clean). L'etage est applique par splitting explicite, apres le
transport.

L'exemple canonique est une ionisation a trois especes
(`d_t n_e = +k n_e n_g`, `d_t n_i = +k n_e n_g`, `d_t n_g = -k n_e n_g`) :

```python
import adc
from adc import dsl

src = dsl.CoupledSource("ionization")
ne = src.block("electrons").role("density")
ni = src.block("ions").role("density")
ng = src.block("neutrals").role("density")
kp = src.param("Kiz", 0.7)
src.add("electrons", role="density", expr=+kp * ne * ng)
src.add("ions",      role="density", expr=+kp * ne * ng)
src.add("neutrals",  role="density", expr=-kp * ne * ng)
compiled = src.compile(backend="production")

sim.add_coupling(compiled)   # branche l'etage sur System.add_coupled_source
```

`sim.add_coupling(...)` accepte aussi les couplages nommes `adc.Ionization` /
`adc.Collision` / `adc.ThermalExchange` (formule figee). Sans appel a `add_coupling`, le
`System` reste bit-identique (l'etage est inerte par defaut).

La compilation produit une ABI plate (`in_blocks`, `in_roles`, `consts`, `out_blocks`,
`out_roles`, `prog_ops`, `prog_args`, `prog_lens`) : du bytecode, jamais un callback
Python. Le test verifie que la trajectoire suit bit-pour-bit une reference NumPy
forward-Euler de la meme ODE, et que les invariants attendus tiennent (`n_i + n_g`
conserve, `n_e - n_i` constant : chaque ionisation cree une paire e/i).

### Pour aller plus loin

- Classification public / interne / deprecie des classes de couplage (dont le concept
  `CoupledSourceFor` et l'evaluateur bytecode `CoupledSourceProgram`) :
  [COUPLING_SURFACE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/COUPLING_SURFACE.md).
- Test de reference : `python/tests/test_dsl_coupled_source.py` (et la variante de
  conservation `test_dsl_coupled_source_conservation.py`).

## Schur : etage source condense

L'integrateur `adc.CondensedSchur` reproduit le splitting d'Hoffart et al.
(arXiv:2510.11808) pour la source raide potentiel / vitesse / force de Lorentz du systeme
Euler-Poisson magnetise. La cle : la condensation de Schur elimine algebriquement la
vitesse du sous-systeme implicite, ce qui reduit l'etage source a un solve elliptique
(operateur tensoriel `-div(A grad phi)` avec `A = rho B^{-1}`, en general non symetrique)
suivi d'une reconstruction explicite de la vitesse.

On le compose avec un etage transport explicite via `adc.Split` :

```python
import adc

time_policy = adc.Split(
    hyperbolic=adc.Explicit(),
    source=adc.CondensedSchur(
        kind="electrostatic_lorentz",   # seul kind supporte
        theta=1.0,                      # theta-schema : 0.5 = Crank-Nicolson, 1 = Euler retrograde
        alpha=3.0,                      # constante de couplage
    ),
)

sim.add_equation(
    "ions",
    model=model,                        # roles requis : Density / MomentumX / MomentumY (Energy optionnel)
    spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"),
    time=time_policy,
)
```

Le modele doit exposer les roles `Density`, `MomentumX`, `MomentumY` (un fluide isotherme
natif `adc.FluidState(kind="isothermal") + adc.IsothermalFlux()` les fournit). L'etage est
entierement C++ (`CondensedSchurSourceStepper`, expose `adc.CondensedSchur`) : aucun
callback Python par cellule.

> **Roles hardcodes cote C++.** Les descripteurs de role / champ ne sont pas reglables
> depuis Python. `adc.CondensedSchur(...)` accepte `kind`, `theta`, `alpha`, mais passer
> `density=`, `momentum=`, `energy=`, `magnetic_field=` ou `potential=` leve une erreur :
> l'etage source C++ fixe ces roles en dur. C'est volontaire (le contrat de
> `CondensedSchurSourceStepper` est fige).

`adc.Strang` est l'extension 2e ordre de `adc.Split` (sequence transport / source /
transport). Le defaut est inchange : un bloc en `adc.Explicit` pur ne voit jamais l'etage
source condense.

> **CondensedSchur (global) vs SourceImplicit (local).** Ne pas confondre. `adc.CondensedSchur`
> assemble et resout un operateur elliptique couplant tout le domaine (pour un couplage raide
> non local : Lorentz / electrostatique). `adc.SourceImplicit` (= IMEX source-only) est
> local : l'implicite ne couple que les composantes d'une meme cellule (relaxation, reactions,
> friction), sans solve elliptique, donc bien moins cher. Une source raide locale n'a pas
> besoin de Schur.

### Pour aller plus loin

- Conception detaillee (les cinq niveaux, la non-symetrie de l'operateur tensoriel, la
  question du solveur Krylov) : [SCHUR_CONDENSATION_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/SCHUR_CONDENSATION_DESIGN.md)
  (banniere : `implemente` ; le document est la spec d'origine, lu comme historique de
  conception).
- Proprietes de conservation du chemin Schur cartesien (valeurs mesurees) :
  [CONSERVATION_SUMMARY.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/CONSERVATION_SUMMARY.md).
- Tests : `python/tests/test_schur_via_system.py` (chemin `System -> run_source_stage`,
  briques natives, CI-safe), `test_schur_conservation.py`.

## Geometrie polaire / disque

Par defaut le domaine est cartesien carre. Deux mecanismes distincts portent une
geometrie non cartesienne.

### Anneau polaire global

Le choix de geometrie vit dans un objet maillage passe en `mesh=` (jamais dans
`FiniteVolume`, qui ne porte que reconstruction + flux + variables). `adc.PolarMesh`
decrit un anneau global `r in [r_min, r_max] x theta in [0, 2pi)`, `nr x ntheta` cellules,
theta periodique, parois physiques en `r_min` / `r_max`. La grille polaire leve le verrou
structural du diocotron sur grille cartesienne (le proto Phase-0 a mesure un rapport 73
sur la diffusion du gradient radial).

```python
import adc

mesh = adc.PolarMesh(r_min=0.3, r_max=1.0, nr=128, ntheta=256)  # axe 0 = radial, axe 1 = azimutal
sim = adc.System(mesh=mesh)   # construit un anneau global et avance dessus
```

Le `SystemConfig` porte alors `geometry="polar"`, `nr`, `ntheta`, `r_min`, `r_max`. Cote
C++, le chemin polaire est branche dans `System.step` (transport `assemble_rhs_polar` +
Poisson `PolarPoissonSolver` + derive de l'aux en base locale `(e_r, e_theta)`).

> **Portee polaire.** Le chemin polaire est limite : transport ExB scalaire /
> isotherme seulement (les flux fluides complets ne sont pas portes), mono-rang (le
> `PolarPoissonSolver` direct, FFT-en-theta + tridiagonale-en-r par Thomas, refuse MPI
> et leve sur `n_ranks() > 1` ou `ba.size() != 1`), pas de couplage cartesien <-> polaire
> (c'est un anneau global). `nr >= 3` (stencil radial decentre d'ordre 2 aux parois),
> `ntheta >= 1`. `adc.CondensedSchur` est cable en polaire (le stepper condense polaire est
> choisi cote C++ selon la geometrie).

### Masque disque (transport cartesien)

Sur grille cartesienne, on peut restreindre le transport a un disque
`hypot(x-cx, y-cy) - R < 0` :

```python
sim.set_disc_domain(cx=0.5, cy=0.5, R=0.40)   # cellule active si son centre est dans le disque
mask = sim.disc_mask()                         # masque 0/1 (ny, nx) row-major, pour verification
```

Sans `set_disc_domain`, le masque est tout actif et le chemin de transport reste
bit-identique. Le masque disque est refuse en geometrie polaire (l'anneau est deja borne
par ses parois radiales `r_min` / `r_max`).

### Pour aller plus loin

- Bindings : `python/bindings.cpp` (`geometry` / `nr` / `ntheta` / `r_min` / `r_max`,
  `set_disc_domain`, `disc_mask`).
- Solveur polaire : `include/adc/numerics/elliptic/polar_poisson_solver.hpp`.

## Extension C++ : ajouter une brique native

Le coeur est agnostique au modele : il ne nomme aucun scenario. Un modele physique est une
composition de briques generiques (etat, transport, source, elliptique), et le calcul
cellule par cellule reste du C++ compile.

Pour ecrire une nouvelle brique native, on satisfait le concept `PhysicalModel`
(`include/adc/core/physical_model.hpp`). Le contrat minimal :

```cpp
template <class M>
concept PhysicalModel =
    requires(const M m, const typename M::State u, const typename M::Aux a, int dir) {
      typename M::State;                                   // type d'etat conservatif
      typename M::Aux;                                     // == adc::Aux
      { M::n_vars } -> std::convertible_to<int>;           // nombre de composantes
      { m.flux(u, a, dir) } -> std::same_as<typename M::State>;       // flux directionnel
      { m.max_wave_speed(u, a, dir) } -> std::convertible_to<Real>;   // CFL
      { m.source(u, a) } -> std::same_as<typename M::State>;          // source LOCALE
      { m.elliptic_rhs(u) } -> std::convertible_to<Real>;             // second membre Poisson
    };
```

Toutes ces methodes doivent etre `ADC_HD` (host/device) si elles sont appelees dans des
kernels. L'extension optionnelle `HasPrimitiveVars` ajoute `to_primitive` / `to_conservative`
(reconstruction en variables primitives, plus stable pour Euler : positivite de rho et p),
et `HyperbolicPhysicalModel` ajoute le descripteur de variables (`conservative_vars()` /
`primitive_vars()`). Une fois la brique conforme, elle se compose dans un `CompositeModel`
et s'expose au runtime comme les briques existantes.

### Pour aller plus loin

- Les cinq couches orthogonales, la carte des modules, l'etage elliptique
  (probleme / operateur / solveur / post-traitement) : [ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md).
- Les choix de conception (concepts + policies, seam `for_each_cell`, `EllipticSolver`) :
  [CHOICES.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/CHOICES.md).
- Le concept et ses extensions : `include/adc/core/physical_model.hpp` ;
  la composition de reference : `include/adc/physics/composite.hpp`.

## Performance et profiling

Le depot embarque un harnais de profilage : `bench/profile_step.cpp`, pilote par
`bench/run_bench.sh`. Il reconstruit un pas de temps representatif du diocotron a partir
des seams publics (sans toucher le hot path) et chronometre chaque phase
(`transport`, `poisson`, `halos`, `aux_derive`, `reduction`, `fence`, `alloc_tmp`)
encadree de `device_fence()` pour capturer l'execution device sous Kokkos.

Le harnais est hors du build par defaut (`ADC_BUILD_BENCH=OFF`) : le CI ne le
configure ni ne le compile jamais. On l'active explicitement :

```sh
bench/run_bench.sh serie                  # Serie CPU
bench/run_bench.sh kokkos-omp  <Kroot>    # Kokkos OpenMP
bench/run_bench.sh mpi          2         # MPI CPU, 2 rangs
bench/run_bench.sh mpi-cuda    <Kroot> 4  # MPI + Kokkos Cuda (GH200), 4 rangs
```

Il accepte `--n --steps --warmup --cfl --solver {geometric_mg|fft} --limiter
{none|minmod|vanleer|weno5} --bc {periodic|dirichlet}`.

Constat principal du profil : sur les six backends mesures, la phase `poisson` domine
le pas a 96 a 99.9 %. Le transport, les halos, les reductions et les allocations
temporaires sont chacun `< 1 ms` par pas (ensemble `< 1.1 %` du pas en serie). Le verrou de
performance est, sans ambiguite, le solve elliptique (`GeometricMG::solve()`, appele a
chaque pas). Deux faits aggravants mesures : le Poisson ne profite pas du parallelisme
on-node (il regresse, le V-cycle descend jusqu'a des grilles minuscules 2x2, 4x4, et le
cout de lancement de chaque kernel ecrase le calcul utile), et sur GPU la latence de
lancement domine (ni un GPU plus large ni des GPU en plus n'aident, d'autant que le layout
`System` est mono-box).

Pas de refactor de performance sans un profil montrant le goulot.
> `PROFILE_RESULTS.md` rapporte les mesures et pointe une cible (le dispatch du V-cycle sur
> les niveaux grossiers) ; il n'applique aucune optimisation.

### Pour aller plus loin

- Profil complet (tableau phase x backend, plateformes exactes M-series + GH200, pistes a
  investiguer) : [PROFILE_RESULTS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/PROFILE_RESULTS.md).
- `docs/PERFORMANCE.md` existe mais porte des mesures historiques (anciens pilotes
  applicatifs, M1) : ne pas le lire comme la perf actuelle.
