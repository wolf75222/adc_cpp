# Architecture de adc_cpp

Solveur C++23 pour les systemes hyperbolique-elliptique couples sur AMR (pile mesh maison),
ecrit pour OpenMP + MPI + Kokkos, cible cluster ROMEO (GH200). Cas de validation fil rouge :
l'instabilite diocotron (derive E x B), l'Euler-Poisson (gravite ou plasma) et le
deux-fluides isotherme (type Hoffart, arXiv:2510.11808).

Ce document fige l'architecture cible et son etat. Le README porte la narration et les
resultats. Ici on decrit les couches, les seams, les decisions, et on distingue ce qui est
**fait** de ce qui est **cible** (refactor planifie, voir [ROADMAP.md](ROADMAP.md)).

## 1. Principe : cinq couches orthogonales

Le modele physique ne depend JAMAIS du backend parallele. Il n'expose que des lois
**locales**, pures ou quasi pures, appelables dans un kernel (`ADC_HD`, device-callable).
Il ne voit ni MPI, ni AMR, ni MultiFab, ni halos. En revanche il est ecrit pour etre
compatible avec l'execution choisie : pas d'allocation dans les boucles chaudes, pas de
`std::function`, pas de polymorphisme dynamique, donnees accessibles sur GPU.

Le code s'organise en cinq couches. **Donnees et execution sont distinctes** : les
conteneurs (ce qui STOCKE) ne sont pas la politique d'execution (comment on BOUCLE et on
COMMUNIQUE). Une couche haute exprime le probleme, une couche basse l'execute ; une couche
haute ne depend jamais d'un detail d'execution.

```
  PhysicalModel          lois locales : flux, sources, fermetures (device-callable)
        |
        v
  NumericalMethod        reconstruction, flux numerique, operateur elliptique, CL logiques
        |
        v
  DataLayout             conteneurs : Box, BoxArray, MultiFab, Geometry, hierarchie AMR
        |
        v
  ExecutionBackend       politique : for_each_cell (serie/OpenMP/Kokkos), comm, GhostExchange,
                         reductions, allocateur, BackendPolicy
        |
  TimeIntegrator         compose les operateurs (RK, IMEX, splitting, AP) sans connaitre
                         leur implementation interne
```

Les cinq couches, par contenu :

| Couche | Quoi | Ne connait pas |
|---|---|---|
| **Physique** | `PhysicalModel`, equation d'etat, termes sources, lois de fermeture | MPI, AMR, MultiFab, Kokkos, halos |
| **Numerique** | reconstruction, flux numerique, operateur spatial, operateur elliptique, CL | la decomposition en boxes/rangs (`BoxArray`, `DistributionMapping`), le backend d'execution, la strategie AMR |
| **Maillage / donnees** | Box, BoxArray, DistributionMapping, MultiFab, Geometry, hierarchie AMR | comment on boucle / communique (le backend) |
| **Execution** | `for_each_cell`, `comm`, GhostExchange, reductions, allocateur, BackendPolicy | les formules physiques, les conteneurs concrets |
| **Temps / couplage** | SSPRK, IMEX, splitting, `CouplingPolicy`, reflux / average-down / subcyclage | l'implementation des operateurs qu'il compose |

**Point delicat : le modele point-wise ne suffit pas pour les modeles couples.** Certains
termes ne sont pas purement locaux : potentiel, champ electrique, nullspace de Poisson
periodique, moyenne de charge, sources implicites. La regle :

```
PhysicalModel   = lois locales (device-callable).
CoupledProblem  = variables globales et contraintes (nullspace, <rho>, contraintes AP).
DiscreteOperator = application sur grille.
```

Aujourd'hui le fond neutralisant `rho0 = <rho>` est passe au modele comme un parametre, pas
calcule par lui : c'est deja l'esprit (la contrainte globale vit hors du `PhysicalModel`).

## 2. Couche 1 : physique (local, device-callable)

Le concept `PhysicalModel` (`core/physical_model.hpp`) n'expose que des fonctions locales :
`flux`, `source`, `max_wave_speed`, `wave_speeds`, `elliptic_rhs`, toutes `ADC_HD`. Aucun
acces au stockage ou au parallelisme.

```cpp
struct EulerPoisson {                       // bon : local, device-callable
  ADC_HD State flux(const State& u, const Aux& a, int dir) const;
  ADC_HD State source(const State& u, const Aux& a) const;     // g = -grad phi
  ADC_HD Real elliptic_rhs(const State& u) const;              // s * 4 pi G (rho - rho0)
};
```

A eviter : un modele qui connait le stockage (`void compute_flux(MultiFab& U, MultiFab& F)`)
melange physique, parallelisme et organisation memoire.

## 3. Couche 2 : numerique / discretisation

`operator/numerical_flux.hpp` (Rusanov / HLL / HLLC, politiques `ADC_HD`),
`operator/reconstruction.hpp` (MUSCL), `operator/spatial_operator.hpp`
(`compute_face_fluxes`, `assemble_rhs`), l'operateur elliptique (`elliptic/`), les CL
logiques (`mesh/physical_bc.hpp`).

C'est de la logique numerique **locale**, pas un conteneur. **Le flux numerique ne
parcourt pas les cellules lui-meme** : il est appele par l'operateur discret, qui le passe
au `for_each_cell` du backend. Le mauvais design serait
`numerical_flux.compute_all_cells(U, F, grid, mpi)` ; le bon est un `assemble_rhs` qui
compose `(modele, flux, reconstruction, CL)` puis laisse l'execution boucler.

Deux granularites a distinguer dans cette couche :

- **Politiques point-wise** (`numerical_flux`, `reconstruction`, le stencil elliptique) :
  prennent des etats / des indices, rendent un flux ou une valeur. Elles ne voient ni
  `Array4`, ni `Box`, ni le moindre conteneur ; entierement reutilisables.
- **Operateurs de grille** (`assemble_rhs`, `compute_face_fluxes`, `apply_laplacian`, le
  lisseur) : ils bouclent sur une `Box` et lisent/ecrivent une vue locale `Array4`. Ils
  dependent donc du *layout local d'un patch* (l'API `Array4` + bornes de `Box`), mais **pas**
  de la decomposition en boxes/rangs (`BoxArray` / `DistributionMapping`), **pas** du backend
  (cache derriere `for_each_cell`), **pas** de la strategie AMR. La boucle sur les fabs et la
  distribution restent au-dessus, dans la couche maillage/donnees.

## 4. Couches 3-4 : maillage/donnees ET execution (distinctes)

Deux concerns separes, volontairement. **Maillage / donnees** = ce qui STOCKE :
`mesh/box2d`, `box_array`, `distribution_mapping`, `multifab`, `geometry`, la hierarchie AMR.
Ces conteneurs ne savent pas comment on boucle ni on communique. **Execution** = la politique
(table ci-dessous) : `for_each_cell` (serie/OpenMP/Kokkos), `comm`, l'echange de halos, les
reductions, l'allocateur. Les confondre laisserait un operateur numerique dependre de la
decomposition en boxes/rangs ou du backend d'execution ; un operateur de grille voit une vue
locale `Array4` + `Box`, mais ni la `DistributionMapping` ni la politique de boucle. Cette
separation est le point structurel de la revue.

| Seam | Fichier | Role | Backends |
|---|---|---|---|
| `for_each_cell(box, f)` | `mesh/for_each.hpp` | politique d'execution, boucle sur cellules | serie / `_OPENMP` / Kokkos (Cuda) |
| `Array4` + `ADC_HD` | `mesh/fab2d.hpp`, `core/types.hpp` | vue POD device-callable | identique host/device |
| `comm` | `parallel/comm.hpp` | rang/size, all-reduce, barrier | identite serie / MPI |
| `allocator` | `core/allocator.hpp` | stockage des Fab (Arena) | `std::allocator` / `cudaMallocManaged` |
| conteneurs | `mesh/box2d`, `box_array`, `multifab`, `geometry` | maillage et champs distribues | identiques tous backends |

**`for_each_cell` exprime une politique d'execution, pas de la logique numerique.** Il
prend une box et un lambda `ADC_HD(i, j)`. S'il devenait un fourre-tout
(`for_each_cell(U, grid, ghosts, mpi, bc, amr, ...)`), on recreerait un framework opaque.
La logique numerique reste dans le lambda (couche 2), pas dans le seam.

**Trois familles de ghosts, a ne pas confondre :**

```
1. Ghosts PHYSIQUES   CL au bord du domaine : Dirichlet, Neumann/Foextrap, periodique, mur.
2. Ghosts PARALLELES  echanges entre patchs / rangs MPI.
3. Ghosts COARSE-FINE interpolation entre niveaux AMR.
```

Etat : les trois sont deja des briques SEPAREES et testees isolement.
- (1) `BoundaryCondition` = `mesh/physical_bc.hpp::fill_physical_bc` (Foextrap, Dirichlet,
  coins), teste seul (`test_physical_bc`).
- (2) `GhostExchange` = `mesh/fill_boundary.hpp` (echange intra-niveau + periodique), teste
  seul (`test_mpi_fillboundary`, `test_mpi_overlap`).
- (3) `AMRBoundaryInterpolation` = `mf_fill_fine_ghosts_*` (interp espace+temps coarse-fine).
`fill_ghosts` n'est PAS un fourre-tout : c'est une COMPOSITION explicite de (1) puis (2)
(`fill_boundary` ; `fill_physical_bc`). **Reste (cible)** : remonter (3), qui vit dans le pas
AMR, en helper nomme de premier niveau (et le rendre distribue, cf. section 8).

**Modele memoire : remplacer la discipline manuelle par une API explicite.** Aujourd'hui,
toute fonction qui fait un kernel device puis une boucle HOTE sur la meme memoire doit
appeler `device_fence()` entre les deux (sinon course memoire unifiee sur GPU, invisible en
CI CPU). C'est correct mais c'est une discipline **manuelle** : un oubli est un bug
silencieux GPU. `sum` / `norm_inf` sont aujourd'hui des boucles hote derriere un fence, pas
des reductions device. **Cible** : une API memoire explicite
(`device_reduce`, `device_norm_inf`, `sync_host`, `sync_device`) qui rend la transition
visible dans le type ou le nom, et des reductions device (pas des boucles hote protegees),
pour ne pas accumuler de synchronisations globales sur GH200.

## 5. Couche 5 : temps et couplage

`integrator/ssprk.hpp`, `imex.hpp` (AP), `splitting.hpp`, `two_fluid_ap.hpp`. Un
`TimeIntegrator` pilote explicitement les etapes (remplissage des ghosts entre stages,
evaluation du RHS, combinaison) mais **ne connait pas la formule du flux** : il ne voit
qu'une interface `RHSOperator` (evaluer le second membre en `(U, t) -> R`). Le temps
compose des operateurs, il ne possede pas la physique.

`coupling/` (`Coupler`, `AmrCoupler`, `AmrCouplerMP`, `SpectralCoupler`, `coupling_policy`)
orchestre fluide <-> Poisson. Regle stricte pour eviter la classe-dieu :

```
Une CouplingPolicy decide l'ORDRE des operations, les variables couplees, les
synchronisations, les contraintes. Elle ne possede pas la donnee, ne connait pas le
backend, ne fait ni regrid, ni I/O, ni diagnostics, ni MPI brut.
```

Les coupleurs AMR sont desormais des ordonnanceurs minces (point 6 de la revue). Trois
responsabilites sont sorties en composants nommes : la hierarchie (stockage des niveaux +
aux) dans `coupling/amr_level_storage.hpp` (`AmrLevelStack<Level>`), le regrid
Berger-Rigoutsos dans `coupling/amr_regrid_coupler.hpp` (`amr_regrid_finest`), les
diagnostics (masse, vitesse de derive) dans `coupling/amr_diagnostics.hpp` (`amr_mass`,
`amr_max_drift_speed`). `AmrCoupler` / `AmrCouplerMP` ne gardent que l'enchainement
`sync_down -> compute_aux -> step` plus la delegation de `regrid()`. L'extraction est
structurelle et bit-identique (equivalence `max|dUc|` a `0` et conservation de masse a
l'arrondi inchangees).

## 6. Carte des modules (`include/adc/`)

| Module | Couche | Role |
|---|---|---|
| `core/` | physique / exec | `types` (`ADC_HD`, `Real`), `state` (`StateVec<N>`), `physical_model` (concept), `allocator` (Arena) |
| `model/` | physique | `diocotron`, `euler`, `euler_poisson` (gravite OU plasma via `coupling_sign`) ; `langmuir`, `two_fluid_isothermal` (noyaux 0D AP) |
| `operator/` | numerique | `numerical_flux` (Rusanov/HLL/HLLC), `reconstruction` (MUSCL), `spatial_operator` (`assemble_rhs`) |
| `elliptic/` | numerique + temps | concept `EllipticSolver` ; `geometric_mg` (V-cycle) ; `poisson_fft`(+`_solver`) ; `poisson_operator` |
| `mesh/` | execution | `box2d`, `box_array`, `fab2d`/`multifab`, `for_each`, `fill_boundary`, `physical_bc`, `geometry`, `mf_arith`, `refinement`, `box_hash` |
| `parallel/` | execution | `comm` (seam MPI), `load_balance` (Z-order + knapsack) |
| `amr/` | execution | `amr_hierarchy`, `cluster` (Berger-Rigoutsos), `regrid`, `tag_box` |
| `integrator/` | temps | `ssprk`, `imex` (AP), `splitting`, `two_fluid_ap`, `amr_reflux`/`amr_multilevel` (pile Fab2D de reference), `amr_reflux_mf` (pile MultiFab, mono-box -> multi-patch N-niveaux, GPU-ready) |
| `coupling/` | temps | `coupler`, `coupling_policy`, `amr_coupler` (mono-box), `amr_coupler_mp` (multi-patch + regrid BR), `amr_level_storage` (hierarchie `AmrLevelStack`), `amr_regrid_coupler` (`amr_regrid_finest`), `amr_diagnostics` (masse, derive), `spectral_coupler` |
| `analysis/` | hors chemin chaud | `diocotron_growth` (Eigen, `#ifdef ADC_HAS_EIGEN`), `hdf5_writer` (`#ifdef ADC_HAS_HDF5`) |
| `solver/` | facade | facades PIMPL : `diocotron_solver`, `euler_poisson_solver`, `two_fluid_ap_solver` |

## 7. Solveur elliptique : probleme / operateur / solveur / post-traitement

Un solveur elliptique depend fortement de la discretisation, des CL, du layout, de la
structure AMR, des communications et du solveur lineaire disponible. Le mettre dans un seul
bloc en fait vite une classe-dieu. Decomposition cible :

```
EllipticProblem        l'equation : -div(eps grad phi) = rho, CL physiques, nullspace, coeffs.
EllipticOperator       l'operateur discret : stencil, apply(), residual(), restriction/prolongation.
LinearSolver           l'inversion : multigrille, FFT, CG, (Hypre/PETSc).
FieldPostProcess       E = -grad phi, energie, diagnostics.
```

Etat :
- **EllipticOperator FAIT** : `elliptic/poisson_operator.hpp` est l'operateur canonique,
  separe des solveurs (`apply_laplacian`, `poisson_residual`, lisseur GS rouge-noir). C'est
  l'`OperatorSpec` partage : `poisson_residual` EST la definition du Laplacien 5 points.
- **LinearSolver FAIT** : le concept `EllipticSolver` (`rhs`/`phi`/`solve`/`residual`/`geom`)
  est l'interface ; `GeometricMG` (V-cycle GS rb, seul compatible AMR et tout `n`, on-device)
  et `PoissonFFTSolver` (direct, mono-niveau periodique `n` puissance de 2, ~5x, **mono-rang /
  boite unique** : il assert `n_ranks()==1 && ba.size()==1`) en sont deux implementations. La
  variante distribuee par bandes (`MPI_Alltoall`) vit dans `SpectralCoupler`, pas ici.
  `Coupler<Model, Elliptic = GeometricMG>` depend du concept, pas d'un backend.
- **Identite MG = FFT rendue STRUCTURELLE** : `test_elliptic_operator` applique le MEME
  operateur canonique `poisson_residual` aux deux solutions -> residus `3.4e-14` (MG) et
  `7.2e-14` (FFT), solutions identiques a `1.3e-16`. Les deux inversent prouvablement le meme
  operateur (plus seulement `maxdiff` MG-vs-FFT de `test_fft_coupler`).

- **EllipticProblem et FieldPostProcess FAITS** : `elliptic/elliptic_problem.hpp` nomme les
  deux. `EllipticProblem` rassemble le coeff `eps`, les CL `BCRec` et le drapeau
  `nullspace_const` (jusqu'ici implicites : Laplacien a coefficient constant `eps = 1`, CL via
  `BCRec`, nullspace ad hoc en periodique). `eps` reste DESCRIPTIF (le stencil ne le lit pas
  encore ; le brancher changerait les valeurs des que `eps != 1`). La fabrique
  `make_elliptic_solver<Solver>(geom, ba, EllipticProblem)` est additive et delegue a la
  `BCRec` existante : aucun appelant casse, le concept `EllipticSolver` reste modele.
  `FieldPostProcess` nomme la convention de derivation `E = -grad phi` via un signe explicite
  `GradSign::Plus` (le coupleur stocke `+grad phi`, le signe physique est porte par
  `diocotron::drift_velocity`) ou `GradSign::Minus` (`two_fluid_ap::tfap_efield` stocke
  directement `-grad phi`). `field_postprocess` remplace le corps de la fonction libre
  `coupler_grad_phi` a l'identique (meme ordre, forme multiplicative `*cx`). Refactor
  structurel bit-identique, prouve par `test_elliptic_problem` (`operator==` strict).

Reste hors-perimetre tant qu'on exige le bit-identique : recabler vers `FieldPostProcess` les
sites en forme `/(2*dx)` (`amr_coupler`, `amr_coupler_mp`, `spectral_coupler`, `two_fluid_ap`),
car la division peut differer au dernier bit de la forme multiplicative `*cx` du coupleur
(IEEE754 : `a/b` et `a*(1/b)` ne coincident pas toujours). Ils instancient la meme convention
nommee, documentee, mais ne sont pas touches a cette etape.

## 8. AMR : vers un objet nativement distribue (priorite)

**Etat.** L'integrateur AMR tourne sur la pile MultiFab + seam, du mono-box au multi-patch
N-niveaux (`integrator/amr_reflux_mf.hpp`, generique `<Limiter, NumericalFlux, N-comp>`,
bulk `for_each_cell` GPU-ready) : `amr_step_2level_mf`, `amr_step_multilevel_mf`,
`amr_step_2level_multipatch`, `amr_step_multilevel_multipatch` (`subcycle_level_mp`). Le
reflux est coverage-aware (interfaces fin-grossier reelles, pas les joints fin-fin geres par
`fill_boundary`) et route la correction vers la box parente (`mf_find_box`). Les coupleurs
`AmrCoupler` (mono-box) et `AmrCouplerMP` (multi-patch + `regrid()` Berger-Rigoutsos) sont
conservatifs.

**Faiblesse 1 : la duplication par cas particulier.** Les noms
`amr_step_2level_mf` / `_multilevel_mf` / `_2level_multipatch` / `_multilevel_multipatch` /
`subcycle_level_mp` encodent le cas dans le NOM. Entree unifiee FAITE : `advance_amr(m,
LevelHierarchy&, dt)` + le type nomme `LevelHierarchy` (niveaux + base_dom + periodicite).
Verifie facade-fidele en **2 ET 3 niveaux** (`test_advance_amr`, `maxdiff = 0` vs l'appel
direct, derive masse `< 1e-12`) et conservatif. La PROMOTION des roles en types avance :
`OwnershipPolicy` est un alias reel de `DistributionMapping` ; `FluxRegister` est un VRAI TYPE
(registre grossier indexe global sur une region : `set` ecrase, `add` accumule borne, `gather`
fait l'`all_reduce`), substitue aux quatre buffers manuels du reflux ; `CoverageMask` est un
VRAI TYPE (masque grossier sur une region : `mark(box)` marque une empreinte fine clippee,
`covered(I,J)` borne) qui porte la part "couverture" de `CoarseFineInterface` (le masque
anti-double-reflux), substitue aux trois masques manuels. Les deux a l'identique (np=1/2/4
`maxdiff = 0`), contrats figes par `test_flux_register` / `test_coverage_mask`.

```
LevelHierarchy [fait, type]   OwnershipPolicy [fait, alias]
FluxRegister [fait, type]     CoverageMask [fait, type] (part "couverture" de CoarseFineInterface)
PatchRange = AmrLevelMP   CoarseFineInterface : routage bordant reste inline
SubcyclingSchedule = recursion Berger-Oliger   RegridPolicy = amr_regrid_finest (BR)
// roles restants : nommes, encore inlines dans subcycle_level_mp ; extraction en types : reste

advance_amr(m, hierarchy, dt);   // entree unifiee ; reste a absorber la famille amr_step_*
```

**Faiblesse 2 : le multi-patch distribue (priorite n.1).** Fait pour le 2-niveaux :
`amr_step_2level_multipatch` tourne **reellement distribue** (`test_mpi_amr_multipatch`,
np=1/2/4 **bit a bit identiques**, masse conservee). Le grossier mono-box est replique
(copie par-rang + remplissage periodique local au lieu du plan MPI de `fill_boundary`), les
patchs fins repartis ; `average_down` (ecrasement des cellules couvertes) et reflux
(addition aux cellules bordantes) remontent par deux buffers grossiers + `all_reduce_sum_inplace`,
chaque rang appliquant a sa copie. La couverture etait deja batie sur le `box_array()`
global. Reste : le chemin N-niveaux recursif (`subcycle_level_mp`, grossier multi-box,
routage `mf_find_box`), puis, cible finale, chaque patch portant **des le depart** :

```
owner_rank          global_box_id          parent_level
interfaces coarse-fine GLOBALES             registre de flux DISTRIBUE
politique de reduction conservative   +   load_balance SFC sur le multi-box
```

C'est l'item prioritaire de la [ROADMAP.md](ROADMAP.md).

## 9. Backends : propriete de la bibliotheque, pas un drapeau par cible

OpenMP, MPI, HDF5 et Kokkos sont attaches a la cible d'interface `adc`. **Tout ce qui lie
`adc` herite du backend** : la facade `src/` (`libadc`), les tests, les exemples. On
configure une seule fois :

```
cmake -B build                       # serie
cmake -B build -DADC_USE_OPENMP=ON   # CPU multi-thread (_OPENMP)
cmake -B build -DADC_USE_MPI=ON      # distribue (ADC_HAS_MPI + MPI::MPI_CXX)
cmake -B build -DADC_USE_KOKKOS=ON \ # GPU / CPU portable (ADC_HAS_KOKKOS)
   -DCMAKE_CXX_COMPILER=$K/bin/nvcc_wrapper -DKokkos_ROOT=$K
```

Sous Kokkos, la norme retombe a C++20 (nvcc CUDA 12.x) et `libadc` elle-meme compile pour
le GPU (les 3 facades passent sous nvcc, valide GH200 bit-identique au CPU).

Compromis assume : ce choix garantit que tests, exemples et lib partagent les memes options
(bon pour une cible recherche/HPC mono-config comme ROMEO/GH200), mais il est rigide pour
une lib publique : on ne peut pas avoir une cible CPU et une cible GPU dans le meme build.
**Cible si le projet grossit en lib reutilisable** : eclater en `adc_core` / `adc_serial` /
`adc_openmp` / `adc_mpi` / `adc_kokkos` / `adc_solver`.

## 10. Frontiere bibliotheque / demo + cout de compilation

| Couche | Contenu | Lien |
|---|---|---|
| `include/` | coeur generique (concepts, templates, seam GPU). Visible a l'instanciation. | header-only `adc::adc` |
| `src/` | facade COMPILEE `libadc` : solveurs concrets non templatises (PIMPL), instancient la pile UNE fois. API stable (apps, pybind11). | `adc::solver` |
| `examples/` (CPU) | pilotes minces. `diocotron`/`diocotron_column` lient `adc::solver` ; `diocotron_amr/mpi/theory` lient `adc::adc`. | |
| `examples/gpu/` | demos Kokkos/CUDA (GH200), heritent Kokkos de `adc`. | |
| `tests/` | CTest (+ MPI via `mpirun`). | |
| `python/` | bindings pybind11 de la facade. | |

Regle : solveur standard -> `adc::solver` ; toucher AMR/MPI/champs internes -> `adc::adc`.
Le PIMPL donne une API stable et borne le cout de compilation : le moteur template
header-only est cher a compiler (surtout sous nvcc), donc on **explicite** quelques
configurations standards (les 3 facades) instanciees une fois dans `src/`, et on reserve
`adc::adc` aux usages experts.

## 11. Validation : logicielle ET numerique

**Bit-identique = filet logiciel, pas preuve numerique.** Prouver que le multipatch est
bit-identique a la reference prouve que la refactorisation n'a rien casse. Ca ne prouve pas
que le comportement est numeriquement correct. Les deux sont necessaires.

Fait aujourd'hui :
- Tests : 60 tests CPU serie (`ctest` sur `build/`, Eigen inclus) ; 73 en build-mpi
  (= 60 + 13 tests MPI lances par `mpirun -np 4`, bit-identiques np=1/2/4).
- Bit-identique : mono-box vs pile Fab2D ; multipatch N-niveaux sur deux axes
  (`test_amr_multilevel_multipatch`, `0`) ; `AmrCouplerMP` vs `AmrCoupler` (`0`) et
  conservatif sous regrid BR (`1.3e-15`, `test_amr_coupler_mp`) ; reflux multipatch 2-niveaux
  DISTRIBUE (`test_mpi_amr_multipatch`, np=1/2/4 a `0` exact).
- Physique : Jeans 0.1%, Bohm-Gross 0.1%, dispersion deux-fluides 3.1%, cyclotron 0.00%.
- Numerique : ordre du Laplacien 5 points (`test_poisson_convergence`, L2 et Linf a 2.00,
  Dirichlet et periodique + nullspace) ; ordre MUSCL ~2 / Rusanov ~1 (`test_muscl_convergence`) ;
  tourbillon isentropique d'Euler (L1 ordre ~2, `test_euler`) ; loi de Gauss discrete du
  couplage div(grad phi) = source a 2.00 (`test_gauss_law`) ; limite AP quantifiee (uniforme
  sur 8 decades de raideur, `test_ap_limit`) ; invariants diocotron (masse, principe du
  maximum, enstrophie non croissante, `test_diocotron_stability`).
- Conservation : flux coarse-fine exact (le reflux rend la masse machine-zero,
  `test_amr_reflux_mf` / `test_amr_coupler` a `~1e-12` / `5.55e-16`).
- GPU : GH200 (CUDA 12.6) bit-identique au CPU ; MPI bit-identique a np=1/2/4.

## 12. Comparaison AMReX

Correspondances : `MultiFab`, `BoxArray`/`DistributionMapping`, `Geometry`, `AmrLevel`,
FillBoundary, Arena, reflux, MLMG ~ `GeometricMG`. Divergences assumees : pas de `MFIter`
(on itere `for_each_cell` + fab local, GPU-ready) ; l'operateur elliptique joue `LinOp`
mais Laplacien a coefficient constant (EB en escalier) ; le FluxRegister / FillPatch
multi-patch 2-niveaux est distribue (bit-identique np=1/2/4), le chemin N-niveaux recursif
restant a generaliser de meme (section 8).
