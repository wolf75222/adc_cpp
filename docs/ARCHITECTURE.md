# Architecture de adc_cpp

Coeur C++23 pour les systemes hyperbolique-elliptique couples sur AMR, ecrit pour
MPI + Kokkos (le backend OpenMP autonome est deprecie). Les briques physiques
(`include/adc/physics/`) et **les bindings Python de la lib** (module `adc` : facades de
composition `System` / `AmrSystem`) vivent ICI ; `adc_cases` ne contient que des **cas
d'utilisation en Python** qui importent ce module. Le coeur est AGNOSTIQUE au modele : il ne
nomme aucun scenario, il fournit des briques generiques composees en `CompositeModel`. Les
integrateurs SUR MESURE (deux-fluides AP) ont quitte le coeur : ce sont des scenarios qui
vivent dans `adc_cases`, compiles a la volee contre les en-tetes generiques.

Ce document decrit l'architecture et son etat reel. Le README porte la narration et les
resultats. Ici on decrit les couches, les seams, les decisions, et on distingue ce qui est
**fait** de ce qui reste **cible**. Les notes de planification et de vision sont archivees
sous [`archive/`](archive/) (hors navigation).

## Surface des classes : API actuelle / interne / legacy

Classement des principaux types selon leur surface d'exposition. Pour le detail des
responsabilites de chaque coupleur, voir [docs/COUPLER_HIERARCHY.md](COUPLER_HIERARCHY.md)
et [docs/COUPLING_SURFACE.md](COUPLING_SURFACE.md) (classification complete du couplage).

| Classe | Surface | Remarque |
|---|---|---|
| `System` (Python) | **API** | facade composition Python, point d'entree principal |
| `AmrSystem` (Python) | **API** | facade AMR Python (mono- ET multi-bloc, explicite ET IMEX) |
| `adc.dsl.Model` / `adc.Model` | **API** | DSL Python + assemblage de briques natives |
| `adc.CartesianMesh` / `adc.PolarMesh` | **API** | objet MAILLAGE passe a `adc.System(mesh=...)` : choix de geometrie (carre / anneau) |
| `adc.FiniteVolume` / `adc.Explicit` / `adc.IMEX` | **API** | objets de configuration de schema / temps passes aux facades |
| `adc.Split` / `adc.Strang` / `adc.CondensedSchur` | **API** | politiques de splitting (Lie / Strang) et etage source condense par Schur |
| `Coupler<M,E>` | interne | coupleur mono-modele mono-niveau, non expose en Python |
| `SystemCoupler` / `SystemDriver` | interne | ordonnanceur multi-especes mono-niveau |
| `AmrCouplerMP` | interne | driver AMR mono-modele multi-box |
| `AmrSystemCoupler` / `AmrSystemDriver` | interne | driver AMR multi-especes |
| `CondensedSchurSourceStepper` | interne | etage source condense par Schur (Lorentz electrostatique), opt-in `System::set_source_stage` ; pendant polaire `PolarCondensedSchurSourceStepper` |
| `numerics::TensorKrylovSolver` (`PolarTensorKrylovSolver`) | interne | BiCGStab matrice-libre pour l'operateur condense non auto-adjoint (preconditionne MG) |
| `spectral_coupler.hpp::SpectralCoupler` | **legacy** | orchestre `DistributedFFTSolver`, non route dans `System` MPI np>1 |
| `amr_multilevel.hpp` / `amr_reflux.hpp` | **legacy** | reference Fab2D mono-box, verite-terrain de `advance_amr` |

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
  ExecutionBackend       seams : for_each_cell (serie/OpenMP/Kokkos), comm, allocateur,
                         BackendPolicy. Ne voient que des vues minimales (Box2D, Array4,
                         scalaire, rang), jamais BoxArray/DistributionMapping
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
| **Execution** | seams : `for_each_cell`, `comm`, allocateur, `BackendPolicy` (vues minimales : Box2D, Array4, scalaire, rang) | les formules physiques, les methodes numeriques, `BoxArray`/`DistributionMapping` |
| **Temps / couplage** | SSPRK, IMEX, splitting, `CouplingPolicy`, reflux / average-down / subcyclage | l'implementation des operateurs qu'il compose |

`GhostExchange` (`fill_boundary`) et les reductions / `saxpy` (`mf_arith`) ne sont PAS la
politique Execution : ce sont des operateurs de grille qui ORCHESTRENT les seams (ils bouclent
`for_each_cell` sur chaque fab local et appellent `comm` / `all_reduce`), au meme titre
qu'`assemble_rhs`. La couche Execution se limite aux seams qui ne voient que des vues minimales.

**Les modeles couples ont des termes non locaux.** Certains
termes ne sont pas purement locaux : potentiel, champ electrique, nullspace de Poisson
periodique, moyenne de charge, sources implicites. La regle :

```
PhysicalModel   = lois locales (device-callable).
EquationBlock   = un U + un modele + une discretisation spatiale + une TimePolicy.
CoupledSystem   = plusieurs EquationBlock.
SystemCoupler   = assemblage global, Poisson, aux, ordonnancement des blocs.
DiscreteOperator = application sur grille.
```

Le fond neutralisant `rho0 = <rho>` ou une densite de charge multi-especes ne doit pas
etre calcule dans un `PhysicalModel` : c'est une responsabilite de l'assembleur de systeme.

## 2. Couche 1 : physique (local, device-callable)

Le concept `PhysicalModel` (`core/physical_model.hpp`) n'expose que des fonctions locales :
`flux`, `source`, `max_wave_speed`, `elliptic_rhs`, toutes `ADC_HD`. Aucun
acces au stockage ou au parallelisme. (`wave_speeds` est une methode optionnelle de brique,
pas une exigence du concept.)

```cpp
struct EulerPoisson {                       // bon : local, device-callable
  ADC_HD State flux(const State& u, const Aux& a, int dir) const;
  ADC_HD State source(const State& u, const Aux& a) const;     // g = -grad phi
  ADC_HD Real elliptic_rhs(const State& u) const;              // s * 4 pi G (rho - rho0)
};
```

**Les briques, axe par axe** (verifie contre le code, `physics/`). Le coeur ne nomme aucun
scenario : un modele est une COMPOSITION (`CompositeModel<Transport, Source, Elliptic>`) de
briques generiques, les noms de scenario (diocotron, Euler-Poisson...) vivant cote application.

| Axe | Briques (exemples) | Role |
|---|---|---|
| etat / transport | `ExBVelocity` (1 var), `Euler` / `CompressibleFlux` (4 var), `IsothermalFlux` (3 var), pendants polaires `ExBVelocityPolar` / `IsothermalFluxPolar` | `flux(U, aux, dir)` + `max_wave_speed` |
| source | `NoSource`, `PotentialForce(charge)`, `GravityForce` | `source(U, aux)` (force du champ) |
| elliptique | `ChargeDensity`, `BackgroundDensity`, `GravityCoupling(sign)` | `elliptic_rhs(U)` (second membre de Poisson) |

Deux chemins de couplage coexistent sous le MEME operateur spatial, sans specialisation : la
derive E x B exerce le chemin **aux vers flux** (le potentiel entre par le flux `ExBVelocity`) ;
Euler-Poisson le chemin **aux vers source** (le potentiel entre par `PotentialForce`/`GravityForce`,
le flux reste celui d'Euler). Le canal `aux` a un contrat de BASE `(phi, grad_x, grad_y)` mais est
desormais EXTENSIBLE : `load_aux<NComp>` / `aux_comps<Model>()` lisent des composantes supplementaires
si le modele declare `n_aux` (`B_z` comp 3 fourni, `T_e` comp 4 derive p/rho) ; `n_aux=3` (defaut)
reste strictement bit-identique a l'historique.

**Etendre : hypotheses et limite.** Ajouter un modele est simple S'IL rentre dans la famille
hyperbolique-elliptique LOCALE prevue. La promesse implicite (un nouveau modele marche partout)
ne tient que SOUS RESERVE :
- systeme conservatif ou quasi-conservatif, compatible avec un flux numerique de Riemann ;
- `flux`, `source`, `max_wave_speed`, `elliptic_rhs` tous LOCAUX (ponctuels) ;
- `aux` compatible avec le couplage existant (`phi` / `grad phi`) ;
- pas de contrainte GLOBALE exotique (au-dela du fond `rho0 = <rho>`, deja gere hors du modele) ;
- pas de source RAIDE exigeant un integrateur dedie (sinon IMEX / splitting).

Hors de cette forme (modele non conservatif, integro-differentiel, cinetique, elliptique
multi-champ), il faut une COUCHE AU-DESSUS, un concept d'operateur plus general, pas un nouveau
`PhysicalModel`. C'est la seule vraie limite architecturale a long terme.

A eviter : un modele qui connait le stockage (`void compute_flux(MultiFab& U, MultiFab& F)`)
melange physique, parallelisme et organisation memoire.

**La GEOMETRIE est un axe de CONFIG du maillage, pas un axe du modele.** Le CHOIX de la
geometrie vit dans `SystemConfig` (`runtime/system.hpp`), PAS dans le schema ni le modele : le
flux numerique, la reconstruction et les variables sont identiques. Trois geometries coexistent
cote `System` :
- **cartesien** (defaut, `geometry == "cartesian"`) : domaine carre `[0,L]^2`, chemin historique
  bit-identique (`adc.CartesianMesh`).
- **polaire** (`geometry == "polar"`, `adc.PolarMesh`) : anneau global `r in [r_min, r_max] x
  theta in [0, 2pi)` (`PolarGeometry`, `mesh/geometry.hpp`). BRANCHE dans `System::step`
  (`python/system.cpp` chemin `polar_`) : transport polaire (`assemble_rhs_polar`,
  `spatial_operator_polar.hpp`), Poisson polaire direct (`PolarPoissonSolver` : FFT-en-theta +
  tridiagonale-en-r, `numerics/elliptic/polar_poisson_solver.hpp`), aux en base locale
  `(e_r, e_theta)`. L'anneau exclut `r = 0` (`r_min > 0`, pas de singularite de coordonnee).
- **disque** (masque de domaine, `System::set_disc_domain(cx, cy, R, mode)`) : sous-domaine de
  transport DISQUE sur grille cartesienne, masque 0/1 cellule-centre (level set
  `hypot(x-cx, y-cy) - R < 0`, meme convention que la paroi conductrice du Poisson). Trois modes :
  `"none"` (defaut, masque inerte, transport plein bit-identique), `"staircase"` (transport masque
  conservatif `assemble_rhs_masked`, frontiere crenelee), `"cutcell"` (embedded-boundary conservatif
  `assemble_rhs_eb`, apertures `alpha_f` + fraction de volume `kappa`, ordre 2 interieur, MMS valide).
  Le mode disque est honore sous Lie ET Strang (`set_time_scheme`).

## 3. Couche 2 : numerique / discretisation

`numerics/numerical_flux.hpp` (Rusanov / HLL / HLLC / Roe, politiques `ADC_HD`),
`numerics/reconstruction.hpp` (MUSCL + WENO5-Z), `numerics/spatial_operator.hpp`
(`compute_face_fluxes`, `assemble_rhs`, `assemble_rhs_masked`), l'operateur elliptique
(`numerics/elliptic/`), les CL logiques (`mesh/physical_bc.hpp`). Les variantes de geometrie sont
des operateurs spatiaux SEPARES, PUREMENT ADDITIFS (le cartesien reste intouche, bit-identique) :
`numerics/spatial_operator_eb.hpp` (`assemble_rhs_eb`, cut-cell / embedded-boundary sur disque) et
`numerics/spatial_operator_polar.hpp` (`assemble_rhs_polar`, divergence en metrique annulaire).

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
  seul (`test_mpi_fillboundary` quand MPI est active).
- (3) `AMRBoundaryInterpolation` = `mf_fill_fine_ghosts_*` (interp espace+temps coarse-fine).
  Trois variantes co-existent : `_t` (mono-box, `amr_flux_helpers.hpp`), `_multi` (multi-box,
  `amr_patch_range.hpp`) et `_mb` (multi-niveau) ; les variantes multi-box sont MPI-safe (iteration
  sur les fabs locaux). `fill_ghosts` n'est PAS un fourre-tout : c'est une COMPOSITION explicite de
(1) puis (2) (`fill_boundary` ; `fill_physical_bc`). **Reste (cible)** : remonter (3), qui vit dans
le pas AMR, en helper nomme de premier niveau.

**Modele memoire : reductions device + detection des oublis de fence.** Toute fonction qui
fait un kernel device puis une boucle HOTE sur la meme memoire unifiee doit appeler
`device_fence()` entre les deux (sinon course hote/device sur GPU, invisible en CI CPU ; cf.
CHOICES.md, le bug le plus subtil). Deux reponses, et PAS une API a la MFEM (`Memory<T>` +
flags host/device) : inutile ici, la memoire est UNIFIEE (GH200, un seul buffer, rien a
desambiguiser).
- Reductions DEVICE faites : `for_each_cell_reduce_sum` / `_max` (`mesh/for_each.hpp`),
  reducteurs `Kokkos::Sum` / `Max` deterministes (bit-identique en serie/OpenMP ; sous Kokkos
  le sum reassocie le dernier bit, le max reste exact). `sum`, `norm_inf` et les diagnostics
  AMR (`amr_mass`, coupleurs `mass`) y passent ; plus de boucle hote derriere un fence. Restent
  les `max_drift_speed` (noyau `std::hypot`, a confirmer device sur ROMEO avant conversion).
- Detection des oublis de fence : `romeo/sanitizer.sbatch` (compute-sanitizer sur les exemples
  GPU) + le checksum bit-identique CPU vs GPU de `diocotron_amr_kokkos`, qui DIVERGE si un fence
  manque. Un filet de CI, plutot qu'un type qui cache la barriere dans l'accesseur (a
  contre-courant de l'idiome Kokkos/AMReX : fence explicite, separe de l'acces).

## 5. Couche 5 : temps et couplage

`numerics/time/time_integrator.hpp`, `scheduler.hpp`, `ssprk.hpp`, `imex.hpp` (AP) et
`splitting.hpp` (`lie_step` ordre 1 ; `strang_step` ordre 2 `S(dt/2) T(dt) S(dt/2)`). Une
`TimePolicy` nomme, par bloc, le traitement temporel (explicite, implicite, IMEX, prescrit) et le
nombre de sous-pas. Le scheduler lit cette politique et appelle l'operateur adapte ; il ne connait
pas la formule du flux. Le temps compose des operateurs, il ne possede pas la physique. Cote
`System`, la POLITIQUE de splitting du macro-pas (transport `H` + etage source `S`) est choisie par
`set_time_scheme("lie" | "strang")` : Strang RE-RESOUT `solve_fields` entre les demi-avances pour
que chaque `H(dt/2)` lise un `phi` coherent ; `"lie"` (defaut) reste bit-identique a l'historique.

`coupling/` (`Coupler`, `SystemCoupler`, `AmrCouplerMP`,
`SpectralCoupler`, `coupling_policy`) orchestre fluide <-> Poisson. Regle stricte
pour eviter la classe-dieu :

```
Une CouplingPolicy decide l'ORDRE des operations, les variables couplees, les
synchronisations, les contraintes. Elle ne possede pas la donnee, ne connait pas le
backend, ne fait ni regrid, ni I/O, ni diagnostics, ni MPI brut.
```

`Coupler<Model>` reste le chemin mono-modele. `SystemCoupler<CoupledSystem<...>>`
est le chemin multi-blocs mono-niveau : il assemble le RHS elliptique depuis plusieurs
etats, derive `aux`, avance les blocs explicites SSPRK, et delegue les blocs implicites
ou IMEX a un callback utilisateur.

**Etage source condense par Schur (Lorentz electrostatique RAIDE).** Pour le systeme
Euler-Poisson couple potentiel / vitesse / Lorentz (Hoffart et al., arXiv:2510.11808) a `omega_c`
eleve, `System::set_source_stage(name, "electrostatic_lorentz", theta, alpha)` active un ETAGE
SOURCE condense (`coupling/condensed_schur_source_stepper.hpp`) qui remplace la source explicite /
IMEX du bloc. La sequence (cf. `docs/SCHUR_CONDENSATION_DESIGN.md`) compose trois briques deja en
place : `ElectrostaticLorentzCondensation` (`coupling/schur_condensation.hpp`) assemble l'operateur
tensoriel plein `A = I + c rho B^{-1}` (eps_x/eps_y diag, a_xy/a_yx croises) et le second membre
condense ; `TensorKrylovSolver` (`numerics/elliptic/krylov_solver.hpp`, BiCGStab matrice-libre
preconditionne par le V-cycle `GeometricMG` sur la partie symetrique) inverse cet operateur NON
auto-adjoint des que `B_z != 0` ; `LorentzEliminator` (`numerics/lorentz_eliminator.hpp`) fournit
`B^{-1}` ferme pour reconstruire la vitesse. Pendant POLAIRE : `PolarCondensedSchurSourceStepper`
(`coupling/polar_condensed_schur_source_stepper.hpp`) + `PolarTensorKrylovSolver`
(`numerics/elliptic/polar_tensor_operator.hpp`, iteratif car la FFT-en-theta du `PolarPoissonSolver`
direct est incompatible avec le tenseur plein). Sans `set_source_stage`, le chemin reste
bit-identique.

Les coupleurs AMR sont desormais des ordonnanceurs minces (point 6 de la revue). Trois
responsabilites sont sorties en composants nommes : la hierarchie (stockage des niveaux +
aux) dans `coupling/amr_level_storage.hpp` (`AmrLevelStack<Level>`), le regrid
Berger-Rigoutsos dans `coupling/amr_regrid_coupler.hpp` (`amr_regrid_finest`), les
diagnostics (masse, vitesse de derive) dans `coupling/amr_diagnostics.hpp` (`amr_mass`,
`amr_max_drift_speed`). `AmrCouplerMP` ne garde que l'enchainement
`sync_down -> compute_aux -> step` plus la delegation de `regrid()`. L'extraction est
structurelle et bit-identique (equivalence `max|dUc|` a `0` et conservation de masse a
l'arrondi inchangees).

## 6. Carte des modules (`include/adc/`)

| Module | Couche | Role |
|---|---|---|
| `core/` | physique / systeme | `types` (`ADC_HD`, `Real`), `state` (`StateVec<N>`), `physical_model` (concept), `EquationBlock`, `CoupledSystem`, `variables` (`VariableRole`), `allocator` (Arena) |
| `physics/` | physique | briques generiques (`bricks`, `composite`), `euler`, `hyperbolic` (iso + pendants polaires `ExBVelocityPolar` / `IsothermalFluxPolar`), `source`, `elliptic`, `langmuir`, `advection_diffusion`, `two_fluid_isothermal` |
| `numerics/` | numerique | `numerical_flux` (Rusanov/HLL/HLLC/Roe), `reconstruction` (MUSCL + WENO5-Z), `spatial_operator` (`assemble_rhs` + `assemble_rhs_masked`), `spatial_operator_eb` (cut-cell EB), `spatial_operator_polar`, `spatial_discretisation`, `lorentz_eliminator` (`B^{-1}` 2x2 ferme) |
| `numerics/elliptic/` | numerique + temps | concept `EllipticSolver` ; `geometric_mg` (V-cycle, eps(x), kappa, eps anisotrope) ; `poisson_fft`(+`_solver`) ; `poisson_operator` ; `elliptic_problem` ; `elliptic_interface` (3 concepts operateur/solveur/post-trait) ; `krylov_solver` (`TensorKrylovSolver` BiCGStab) ; `polar_poisson_solver` (direct, FFT-en-theta + tridiag-en-r) ; `polar_tensor_operator` (iteratif tenseur plein) ; `cut_fraction` |
| `numerics/time/` | temps | `TimePolicy`, scheduler par sous-pas, SSPRK, `time_steppers`, `implicit_stepper`, IMEX, splitting (`lie` / `strang`), moteur AMR `advance_amr` (`amr_reflux_mf`), pile de reference `amr_multilevel` / `amr_reflux` / `amr_level` / `amr_advance` / `amr_subcycling` / `amr_patch_range` / `amr_flux_helpers` |
| `mesh/` (donnees) | maillage / donnees | `box2d`, `box_array`, `distribution_mapping`, `fab2d`/`multifab`, `geometry` (cartesien + `PolarGeometry`), `refinement`, `box_hash` |
| `mesh/` (execution) | execution | `for_each` (seam `for_each_cell`), `fill_boundary` (GhostExchange), `physical_bc`, `mf_arith` (operateurs de grille qui bouclent le seam) |
| `parallel/` | execution | `comm` (seam MPI), `load_balance` (Z-order + knapsack) |
| `amr/` | maillage adaptatif | `amr_hierarchy` (conteneur de niveaux), `cluster` (Berger-Rigoutsos, arithmetique entiere), `regrid` (politique de remaillage), `tag_box` (grille de marqueurs) |
| `coupling/` | temps / couplage | `elliptic_rhs`, `Coupler`, `SystemCoupler`, `coupling_policy`, `amr_coupler_mp`, `amr_system_coupler`, `amr_regrid_coupler`, `amr_level_storage`, `spectral_coupler`, `aux_fill`, `coupled_source`(+`_program`), `schur_condensation` + `condensed_schur_source_stepper` (+ pendant `polar_*`), diagnostics AMR |
| `runtime/` | runtime / bindings | facades `System` / `AmrSystem` (+ `amr_runtime` moteur multi-blocs), `model_factory` / `model_spec`, `block_builder` (+ `block_builder_polar`), `grid_context`, `system_field_solver` / `system_stepper` / `system_block_store` (extraits du god-class), `wall_predicate`, JIT/AOT du DSL (`dynamic_model`, `compiled_block_abi`, `dsl_block`), chemin natif (`native_loader`, `abi_key`, `add_native_block`), `add_compiled_model` cote AmrSystem (`amr_dsl_block`) ; canal aux extensible (`ensure_aux_width`, `set_magnetic_field`, `set_electron_temperature_from`) |

### 6bis. Modules detailles (par type, source de verite unique)

Relocalise depuis le README. Chaque entree nomme le header, le type ou la fonction cle, et son role.

| Module | Role |
|---|---|
| [`core/physical_model.hpp`](include/adc/core/physical_model.hpp) | concept `PhysicalModel` (flux, max_wave_speed, source, elliptic_rhs) |
| [`core::{EquationBlock,CoupledSystem}`](include/adc/core/equation_block.hpp) | bundle par espece (modele + schema spatial + politique temps + BC) et systeme de N especes |
| [`physics::bricks` / `composite`](include/adc/physics/bricks.hpp) | briques generiques (etat, transport, source, elliptique) composees en `CompositeModel` |
| [`numerics::{RusanovFlux,HLLFlux,HLLCFlux,RoeFlux}`](include/adc/numerics/numerical_flux.hpp) | flux numeriques (politiques `ADC_HD`) |
| [`numerics::reconstruction`](include/adc/numerics/reconstruction.hpp) | MUSCL ordre 2 (NoSlope / Minmod / VanLeer) + Weno5 |
| [`numerics::assemble_rhs` / `compute_face_fluxes` / `assemble_rhs_masked`](include/adc/numerics/spatial_operator.hpp) | `R = -div F + S`, flux de face pour le reflux (diffusion incluse) ; variante masquee 0/1 (disque escalier) ; GPU via `for_each_cell` |
| [`numerics::assemble_rhs_eb`](include/adc/numerics/spatial_operator_eb.hpp) | transport cut-cell / embedded-boundary conservatif sur disque (apertures `alpha_f` + fraction de volume `kappa`, ordre 2) |
| [`numerics::assemble_rhs_polar`](include/adc/numerics/spatial_operator_polar.hpp) | transport polaire additif `R = -div_polar F + S` (metrique annulaire, theta periodique) |
| [`numerics::lorentz_eliminator`](include/adc/numerics/lorentz_eliminator.hpp) | `B = I - theta dt (v x B_z)`, `B^{-1}` 2x2 ferme (reconstruction Lorentz du Schur) |
| [`numerics::time::{TimePolicy,SSPRK2,SSPRK3}`](include/adc/numerics/time/time_integrator.hpp) | par bloc : explicite / implicite / IMEX, sous-pas (`substeps`) ET cadence (`stride`) |
| [`numerics::time::{ForwardEuler,SSPRK2Step,SSPRK3Step}`](include/adc/numerics/time/time_steppers.hpp) | integrateurs en temps OBJETS (`take_step(rhs, U, dt)`) ; l'utilisateur peut fournir le sien |
| [`numerics::time::{ImplicitSourceStepper,backward_euler_source}`](include/adc/numerics/time/implicit_stepper.hpp) | defaut implicite (Newton local) ; IMEX partiel via `Model::is_implicit(c)` |
| [`numerics::time::advance_subcycled`](include/adc/numerics/time/scheduler.hpp) | scheduler : sous-pas + cadence (macro-pas) par `EquationBlock` |
| [`numerics::time::imex_euler_step`](include/adc/numerics/time/imex.hpp) | IMEX asymptotic-preserving |
| [`numerics::time::{lie_step,strang_step}`](include/adc/numerics/time/splitting.hpp) | splitting d'operateurs |
| [`numerics::time::advance_amr`](include/adc/numerics/time/amr_reflux_mf.hpp) | moteur AMR unifie : multi-patch N-niveaux, reflux coverage-aware, distribue MPI |
| [`numerics::elliptic::GeometricMG`](include/adc/numerics/elliptic/geometric_mg.hpp) | multigrille geometrique (V-cycle GS rb), AMR-compatible, on-device ; eps(x) variable cote coeur (`set_epsilon`) |
| [`numerics::elliptic::PoissonFFTSolver` / `DistributedFFTSolver`](include/adc/numerics/elliptic) | Poisson FFT spectral (mono-rang) et distribue (MPI), correctif `n` non puissance de 2 |
| [`numerics::elliptic::TensorKrylovSolver`](include/adc/numerics/elliptic/krylov_solver.hpp) | BiCGStab matrice-libre pour l'operateur condense non auto-adjoint, preconditionne par le V-cycle MG symetrique |
| [`numerics::elliptic::PolarPoissonSolver`](include/adc/numerics/elliptic/polar_poisson_solver.hpp) | Poisson polaire DIRECT sur anneau : FFT-en-theta + tridiagonale (Thomas)-en-r, exact par mode azimutal |
| [`numerics::elliptic::PolarTensorKrylovSolver`](include/adc/numerics/elliptic/polar_tensor_operator.hpp) | operateur elliptique polaire a tenseur plein (termes croises `a_rt`/`a_tr`), iteratif (Schur polaire) |
| [`numerics::elliptic::elliptic_interface`](include/adc/numerics/elliptic/elliptic_interface.hpp) | concepts `EllipticOperator` / `EllipticSolver` / `FieldPostProcessor` (contrat REELLEMENT commun, fonctions libres exclues) |
| [`coupling::{ChargeDensityRhs,CoupledSource}`](include/adc/coupling/elliptic_rhs.hpp) | RHS de systeme `f = sum_s q_s n_s` (N especes) ; source inter-especes `S(U_e,U_i,phi)` |
| [`coupling::Coupler`](include/adc/coupling/coupler.hpp) | couplage hyperbolique-elliptique mono-modele : `Coupler<Model, Elliptic>` |
| [`coupling::{SystemAssembler,SystemDriver}`](include/adc/coupling/system_coupler.hpp) | multi-especes mono-niveau : l'**assembleur** assemble (Poisson de systeme + aux), le **driver** avance (`step`, `step_cfl`, `step_adaptive`) ; `SystemCoupler` = alias du driver |
| [`coupling::AmrSystemCoupler`](include/adc/coupling/amr_system_coupler.hpp) | le systeme multi-especes porte sur **AMR** (Poisson grossier + reflux par bloc) |
| [`coupling::AmrCouplerMP`](include/adc/coupling) | couplage AMR multi-patch mono-modele (route par `advance_amr`) |
| [`coupling::{ElectrostaticLorentzCondensation,CondensedSchurSourceStepper}`](include/adc/coupling/schur_condensation.hpp) | etage source condense par Schur (Lorentz electrostatique raide) ; assemble `A = I + c rho B^{-1}` + RHS condense, resout via `TensorKrylovSolver`, reconstruit par `LorentzEliminator` |
| [`coupling::PolarCondensedSchurSourceStepper`](include/adc/coupling/polar_condensed_schur_source_stepper.hpp) | pendant POLAIRE de l'etage Schur (operateur tensoriel polaire iteratif) |
| [`coupling::CoupledSourceKernel` / `CsProgram`](include/adc/coupling/coupled_source_program.hpp) | interprete de source couplee generique (bytecode postfixe device, `adc.dsl.CoupledSource`) |
| [`coupling::aux_fill`](include/adc/coupling/aux_fill.hpp) | helpers du canal aux partages par les trois coupleurs (Coupler / SystemAssembler / AmrSystemCoupler) |
| [`amr::{cluster,regrid,tag_box}`](include/adc/amr) | tagging + clustering Berger-Rigoutsos + regrid |
| [`amr::AmrHierarchyLayout` / `same_layout_or_throw`](include/adc/amr) | garde-fou de layout AMR partage (boites + ordre + dmap + dx/dy + niveaux) ; premier pas du capstone AMR multi-blocs (#141) |
| [`mesh::{MultiFab,BoxArray,Geometry}`](include/adc/mesh) | conteneurs distribues, halos, geometrie |
| [`runtime::{System,AmrSystem}`](include/adc/runtime/system.hpp) | facades runtime de composition (assise des bindings Python) |
| [`runtime::{model_factory,model_spec}`](include/adc/runtime/model_factory.hpp) | assemblage d'un `CompositeModel` a partir d'une spec de briques |
| [`runtime::{dynamic_model,compiled_block_abi,dsl_block}`](include/adc/runtime/dsl_block.hpp) | dispatch JIT (`.so`, `IModel` virtuel) et AOT (bloc compile) d'un modele genere par le DSL |
| [`runtime::{native_loader,abi_key}`](include/adc/runtime/native_loader.hpp) | chemin DSL "production" : loader `.so` zero-copie inline `add_compiled_model<ProdModel>` ; `abi_key()` rend l'incompatibilite d'ABI EXPLICITE (`add_native_block`) |
| [`runtime::{SystemFieldSolver,SystemStepper,SystemBlockStore}`](include/adc/runtime/system_stepper.hpp) | responsabilites extraites du god-class `System::Impl` : solve_fields / avance en temps (Lie + Strang) / registre de blocs |
| [`runtime::{block_builder_polar,grid_context}`](include/adc/runtime/block_builder_polar.hpp) | fermetures de bloc POLAIRE (`assemble_rhs_polar`, `derive_aux_polar`) ; contexte de grille reel partage |
| [`runtime::AmrRuntime`](include/adc/runtime/amr_runtime.hpp) | moteur multi-blocs runtime (registre type-erase par nom) + regrid d'union des tags |
| seams [`for_each_cell`](include/adc/mesh/for_each.hpp), [`comm`](include/adc/parallel/comm.hpp) | dispatch serie/OpenMP/Kokkos, comm MPI |

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
- **EllipticOperator FAIT** : `numerics/elliptic/poisson_operator.hpp` est l'operateur canonique,
  separe des solveurs (`apply_laplacian`, `poisson_residual`, lisseur GS rouge-noir). C'est
  l'`OperatorSpec` partage : `poisson_residual` EST la definition du Laplacien 5 points.
- **LinearSolver FAIT** : le concept `EllipticSolver` (`rhs`/`phi`/`solve`/`residual`/`geom`)
  est l'interface ; `GeometricMG` (V-cycle GS rb, seul compatible AMR et tout `n`, on-device)
  et `PoissonFFTSolver` (direct, mono-niveau periodique, ~5x, **mono-rang / boite unique** : il
  assert `n_ranks()==1 && ba.size()==1`, avec correctif `n` non puissance de 2) et
  `DistributedFFTSolver` (FFT distribuee par BANDES, `MPI_Alltoall`, enveloppant le composant
  autonome `PoissonFFT` de `numerics/elliptic/poisson_fft.hpp`) en sont TROIS implementations.
  `SpectralCoupler` DELEGUE desormais a
  `DistributedFFTSolver` (il ne re-implemente plus la FFT) : il ORCHESTRE un `EllipticSolver`, il
  n'en contient pas. `Coupler<Model, Elliptic = GeometricMG>` depend du concept, pas d'un backend.
- **Identite MG = FFT rendue STRUCTURELLE** : `test_elliptic_operator` applique le MEME
  operateur canonique `poisson_residual` aux deux solutions -> residus `3.4e-14` (MG) et
  `7.2e-14` (FFT), solutions identiques a `1.3e-16`. Les deux inversent prouvablement le meme
  operateur.

- **eps(x) variable cote coeur FAIT** : `GeometricMG` resout reellement `div(eps grad phi)=f`,
  `eps` etant un champ au centre des cellules fourni par `set_epsilon(eps_fn)` (formule
  analytique evaluee niveau par niveau, ordre 2 preserve) ou `set_epsilon(eps_fine)` (champ
  discretise restreint par `average_down`). NON appele => `eps` uniforme (chemin historique).
  Le cablage `System` / Python est desormais FAIT : `System::set_epsilon_field` (eps(x) isotrope),
  `set_epsilon_anisotropic_field` (eps_x/eps_y, operateur `div(diag(eps_x, eps_y) grad phi)`) et
  `set_reaction_field` (terme `-kappa phi`, Poisson ecrante / Helmholtz) passent un coefficient non
  uniforme au solveur de systeme (`geometric_mg` seul ; demander avec `fft` leve une erreur).
- **Solveurs POLAIRES et TENSORIELS** (geometrie annulaire + Schur condense) : `PolarPoissonSolver`
  (`polar_poisson_solver.hpp`, direct FFT-en-theta + tridiag-en-r) est l'inversion polaire du chemin
  `System` polaire ; `TensorKrylovSolver` (`krylov_solver.hpp`, BiCGStab matrice-libre) et son
  pendant `PolarTensorKrylovSolver` (`polar_tensor_operator.hpp`) inversent l'operateur condense NON
  auto-adjoint de l'etage Schur. `elliptic_interface.hpp` documente les trois concepts
  (`EllipticOperator` / `EllipticSolver` / `FieldPostProcessor`) sans redefinir le contrat existant.
- **EllipticProblem et FieldPostProcess FAITS** : `numerics/elliptic/elliptic_problem.hpp` nomme
  les deux. `EllipticProblem` rassemble le coeff `eps`, les CL `BCRec` et le drapeau
  `nullspace_const`. Au niveau de CETTE fabrique additive, `eps` du `EllipticProblem` reste
  DESCRIPTIF (le brancher au stencil via cette voie reste a faire ; la voie directe
  `set_epsilon` ci-dessus, elle, est cablee dans `GeometricMG`). La fabrique
  `make_elliptic_solver<Solver>(geom, ba, EllipticProblem)` est additive et delegue a la
  `BCRec` existante : aucun appelant casse, le concept `EllipticSolver` reste modele.
  `FieldPostProcess` nomme la convention de derivation `E = -grad phi` via un signe explicite
  `GradSign::Plus` (le coupleur stocke `+grad phi`, le signe physique est porte par
  `diocotron::drift_velocity`) ou `GradSign::Minus` (un consommateur qui stocke directement
  `-grad phi`). `field_postprocess` remplace le corps de la fonction libre
  `coupler_grad_phi` a l'identique (meme ordre, forme multiplicative `*cx`). Refactor
  structurel bit-identique, prouve par `test_elliptic_problem` (`operator==` strict).

Reste hors-perimetre tant qu'on exige le bit-identique : recabler vers `FieldPostProcess` les
sites en forme `/(2*dx)` (`amr_coupler_mp`, `spectral_coupler`),
car la division peut differer au dernier bit de la forme multiplicative `*cx` du coupleur
(IEEE754 : `a/b` et `a*(1/b)` ne coincident pas toujours). Ils instancient la meme convention
nommee, documentee, mais ne sont pas touches a cette etape.

## 8. AMR distribue

**Etat.** Le coeur fournit les briques AMR dans `amr/` et le moteur MultiFab dans
`numerics/time/amr_reflux_mf.hpp`. Le schema reste le meme : tagging, clustering
Berger-Rigoutsos, regrid, sous-cyclage, average-down et reflux conservatif. Les roles
importants sont explicites : `FluxRegister` accumule les flux de face, `CoverageMask`
evite les doubles corrections, `DistributionMapping` porte l'ownership des boxes.

**Couplage.** `AmrCouplerMP` reste le driver AMR mono-modele (1 seul `add_block`). Le
`SystemCoupler` couvre le mono-niveau multi-blocs ; `AmrSystemCoupler` porte le systeme
multi-especes sur AMR (Poisson grossier + reflux par bloc). La facade `AmrSystem` supporte
desormais mono- ET multi-bloc : le multi-bloc s'active AUTOMATIQUEMENT des le 2e `add_block`
(moteur runtime `AmrRuntime`, registre type-erase par nom ; le build paresseux materialise la
hierarchie partagee, garde `same_layout_or_throw`). Reconstruction conservative | primitive, flux
Riemann rusanov | hllc | roe, integrateur explicite ET IMEX (par bloc), multirate (substeps /
stride), sources couplees inter-especes, blocs natifs ET compiles melanges ; couvert par la famille
de tests capstone (`test_amr_system_*`, `test_amr_multiblock_*`). Le regrid d'union des
tags (#199) est cable : avec `regrid_every > 0`, la hierarchie est re-grillee a partir de l'UNION
(OU cellule a cellule) des tags de TOUS les blocs (predicat par bloc, `set_refinement`) plus le tag
de `|grad phi|` (`set_phi_refinement`) ; `regrid_every == 0` -> hierarchie figee, bit-identique.
Nuance honnete restante : pas d'etage Schur GLOBAL sur AMR, et le regrid d'union reste a 2 niveaux.

**Tests coeur.** Les invariants AMR couverts dans ce depot sont le raffinement, la hierarchie,
le clustering, le regrid, le reflux, le masque de couverture, les diagnostics AMR et le load
balancing (serie), plus, sous MPI, la parite du chemin compile multi-box au nombre de rangs
(`test_mpi_mbox_parity`, `test_mpi_amr_compiled_parity`), le grossier reparti == replique
(`test_mpi_amr_distributed_coarse`) et B_z par niveau multi-box distribue
(`test_amr_system_bz_multibox_np2/4`), tous np=1/2/4. Les validations applicatives (modele
physique nomme, taux de croissance) restent dans les cas downstream (`adc_cases`).

## 9. Backends : propriete de la bibliotheque, pas un drapeau par cible

OpenMP, MPI, HDF5 et Kokkos sont attaches a la cible d'interface `adc`. **Tout ce qui lie
`adc` herite du backend** : les tests du coeur et les applications downstream. On configure
une seule fois :

```
cmake -B build                       # serie
cmake -B build -DADC_USE_OPENMP=ON   # CPU multi-thread (_OPENMP), deprecie -> Kokkos
cmake -B build -DADC_USE_MPI=ON      # distribue (ADC_HAS_MPI + MPI::MPI_CXX)
cmake -B build -DADC_USE_KOKKOS=ON \ # GPU / CPU portable (ADC_HAS_KOKKOS), recommande
   -DCMAKE_CXX_COMPILER=$K/bin/nvcc_wrapper -DKokkos_ROOT=$K
```

Le backend Kokkos est recommande : il couvre le CPU multi-thread (Serial / OpenMP) ET le GPU
(Cuda) avec un seul code, sans aucun kernel CUDA ecrit a la main. La CI joue le backend Kokkos
en Serial.

Sous Kokkos, la norme retombe a C++20 (nvcc CUDA 12.x). Les kernels `ADC_HD` et le seam
`for_each_cell` sont alors compiles pour le backend choisi.

Compromis assume : ce choix garantit que tests et applications downstream partagent les memes options
(bon pour une cible recherche/HPC mono-config comme ROMEO/GH200), mais il est rigide pour
une lib publique : on ne peut pas avoir une cible CPU et une cible GPU dans le meme build.
**Cible si le projet grossit en lib reutilisable** : eclater en `adc_core` / `adc_serial` /
`adc_openmp` / `adc_mpi` / `adc_kokkos`.

## 10. Frontiere bibliotheque / demo + cout de compilation

| Couche | Contenu | Lien |
|---|---|---|
| `include/` | coeur generique (concepts, templates, seam GPU). Visible a l'instanciation. | header-only `adc::adc` |
| `tests/` | CTest du coeur (+ MPI via `mpirun` quand active). | lie `adc::adc` |
| `docs/` | architecture, algorithmes, notes de validation/performance. | documentation |
| `include/adc/physics/` | briques physiques generiques (etat, transport, source, elliptique) composees en `CompositeModel`. | header-only `adc::adc` |
| `python/` | module Python `adc` (pybind11) : facades runtime `System` / `AmrSystem` + `adc.integrate` + DSL. L'integrateur AP deux-fluides a quitte le coeur (scenario, non brique) : il vit dans `adc_cases/two_fluid_ap/`, compile a la volee contre les en-tetes generiques. | `-DADC_BUILD_PYTHON=ON` |
| `adc_cases` | cas d'utilisation 100 % Python (un dossier par cas), importent le module `adc`. | aucun C++ |

Regle actuelle : `adc_cpp` est la bibliotheque (coeur + briques physiques + bindings). Le coeur
est AGNOSTIQUE au modele : aucun scenario n'est nomme dans `include/`, seules des briques
generiques composees par le `model_factory` ; les compositions nommees vivent dans `adc_cases`,
qui ne fait que **consommer** le module Python.

## 11. Validation : logicielle ET numerique

**Bit-identique = filet logiciel, pas preuve numerique.** Prouver que le multipatch est
bit-identique a la reference prouve que la refactorisation n'a rien casse. Ca ne prouve pas
que le comportement est numeriquement correct. Les deux sont necessaires.

Fait aujourd'hui dans `adc_cpp` :
- Tests : ctests coeur par defaut (`ctest --test-dir build`), joues en CI sur deux builds
  (Release serie ET Kokkos backend Serial) ; entrees ctest MPI quand `-DADC_USE_MPI=ON`
  (np=1/2/4, bit-identiques) ; tests Python supplementaires (bindings + DSL). Le decompte exact
  des tests est regenere dans [BACKEND_COVERAGE.md](BACKEND_COVERAGE.md) (source de verite) ;
  ne pas coder en dur ici.
- Numerique coeur : maillage, halos, AMR, reflux, multigrille, Poisson (dont eps(x) variable,
  Helmholtz/ecrante, anisotrope, cut-cell), discretisations, flux de Roe, WENO5-Z, IMEX/AP,
  splitting Lie ET Strang (`test_strang_splitting`), multirate, `EquationBlock`, `CoupledSystem`,
  `SystemCoupler`, canal aux extensible (B_z, T_e), parite du bloc compile AOT (CPU/Serial).
- Geometrie : transport EB cut-cell sur disque (`test_eb_transport`, `test_cut_cell*`, MMS ordre 2),
  primitive de fraction coupee (`test_cut_fraction*`), masque de domaine disque (`test_disc_domain_mask`).
- Polaire : Poisson polaire direct (`test_polar_poisson_mms`), transport annulaire
  (`test_polar_transport_mms`, `test_polar_ring_advection`, `test_polar_mms_vr`), chemin complet via
  `System` (`test_polar_system_step`), operateur tensoriel polaire (`test_polar_tensor_elliptic_mms`).
- Etage Schur condense : `LorentzEliminator` (`test_lorentz_eliminator`), operateur tensoriel +
  BiCGStab (`test_schur_condensation`, `test_krylov_solver`), etage source cartesien ET polaire
  (`test_condensed_schur_source_stepper`, `test_polar_condensed_schur_source_stepper`), Schur polaire
  multi-rang MPI (`test_mpi_polar_schur`).
- Les validations applicatives (diocotron, runs ROMEO, taux de croissance) vivent dans `adc_cases`.
- GPU GH200 (backend Kokkos Cuda, validations integrees au depot via harness sous
  `python/tests/gpu/`, hors CI faute de runner GPU) : composants valides bit-identiques au CPU
  (System mono-grille, ops de champ AMR, halos MPI multi-GPU, chemin compile a foncteurs nommes
  multi-box + MPI) ET validation INTEGREE AmrSystem + MPI + GPU FAITE (les trois axes dans un
  seul run, np=1/2/4 `dmax=0`, masse conservee a `0`). Caveats honnetes : un grossier multi-box
  DISTRIBUE n'est pas bit-identique sur les sommes globales (ordre de reduction FMA, le max reste
  exact) ; `add_compiled_model` a lambdas etendues n'est pas zero-copie sur device (rebond hote) ; le
  strong-scaling AMR par grossier reparti est NEGATIF a cette echelle. (La facade `AmrSystemCoupler`
  s'instancie + se compile desormais sous nvcc, limite device (b) LEVEE : concept a sonde nommee.)
  Detail : [GPU_RUNTIME_PORT.md](GPU_RUNTIME_PORT.md).

## 12. Comparaison AMReX

Correspondances : `MultiFab`, `BoxArray`/`DistributionMapping`, `Geometry`, `AmrLevel`,
FillBoundary, Arena, reflux, MLMG ~ `GeometricMG`. Divergences assumees : pas de `MFIter`
(on itere `for_each_cell` + fab local, GPU-ready) ; l'operateur elliptique joue `LinOp` et porte
desormais le coefficient variable `eps(x)`, le tenseur anisotrope `diag(eps_x, eps_y)`, le terme de
reaction `kappa` (Helmholtz / ecrantage) et le cut-cell EB (cut_fraction) -- le tenseur PLEIN
(termes croises de la condensation de Schur) sort, lui, du `GeometricMG` symetrique et passe par
`TensorKrylovSolver` ; le FluxRegister / FillPatch multi-patch 2-niveaux est distribue (bit-identique
np=1/2/4), grossier replique ou de-replique (multi-box reparti) ; reste le regrid d'un niveau
intermediaire reparti pour 3+ niveaux (section 8).

## 13. Arborescence detaillee (fichier par fichier)

Coeur header-only sous `include/adc/`, range par couche. Une ligne par fichier : ce que
c'est, et pourquoi il est la. Descriptions tirees du doc-comment de chaque en-tete.

### `core/` : types et contrat
- `types.hpp` : scalaires de base (`Real`) + macro `ADC_HD` (delegue a `KOKKOS_FUNCTION` sous Kokkos).
- `state.hpp` : `State` / `Aux`, les deux types ponctuels de la couche physique (POD device-callable).
- `physical_model.hpp` : le concept `PhysicalModel` (contrat flux / source / max_wave_speed / elliptic_rhs).
- `variables.hpp` : `VariableRole` + `VariableSet` (`index_of(role)`, `role_name`). UTILISE : les couplages inter-especes resolvent qte de mvt / densite par ROLE (fallback indices historiques), et le DSL emet les roles sur les briques generees ; chaque bloc porte son `VariableSet` (cons/prim).
- `equation_block.hpp` : un bloc d'equation = nom, `MultiFab`, modele, discretisation spatiale, politique temps.
- `coupled_system.hpp` : tuple type de plusieurs `EquationBlock`, iteration generique par bloc.
- `kokkos_env.hpp` : initialisation paresseuse de Kokkos (avant toute allocation `SharedSpace`).
- `allocator.hpp` : allocateur du `Fab2D`, std (host) ou `Kokkos::kokkos_malloc<SharedSpace>` (memoire unifiee).

### `mesh/` : donnees + seams (couche 3)
- `box2d.hpp` : `Box2D`, espace d'indices d'une grille cartesienne 2D.
- `box_array.hpp` : `BoxArray`, les boxes qui pavent un niveau (disjointes, couvrantes).
- `box_hash.hpp` : hash spatial (bins uniformes) pour retrouver les boxes voisines.
- `distribution_mapping.hpp` : `DistributionMapping`, box -> rang MPI.
- `fab2d.hpp` : `Fab2D`, donnees mono-grille sur une Box2D + ghosts (equivalent du FArrayBox).
- `multifab.hpp` : `MultiFab`, champ distribue (collection de Fab2D) (equivalent du MultiFab AMReX).
- `geometry.hpp` : `Geometry` (cartesien) + `PolarGeometry` (anneau global (r, theta) : `r_cell` / `r_face` / `dr` / `dtheta`), correspondance indices <-> coordonnees physiques.
- `for_each.hpp` : **seam d'execution** `for_each_cell` + `for_each_cell_reduce_*` (serie/OpenMP/Kokkos) + `device_fence`.
- `fill_boundary.hpp` : echange de halos intra-niveau (begin/end non-bloquant).
- `physical_bc.hpp` : CL physiques du domaine + `fill_ghosts`.
- `refinement.hpp` : transfert AMR ratio r : prolongation (interp) + restriction (average_down).
- `mf_arith.hpp` : combinaisons lineaires de MultiFab (saxpy, norm_inf, sum) pour les etages RK.

### `physics/` : briques physiques generiques (couche 1)
- `bricks.hpp` / `composite.hpp` : briques generiques (etat, transport, source, elliptique) et leur composition en `CompositeModel<Hyperbolic, Source, Elliptic>`.
- `hyperbolic.hpp` : flux hyperboliques generiques (`ExBVelocity`, `IsothermalFlux`) + pendants polaires en base locale (`ExBVelocityPolar`, `IsothermalFluxPolar`).
- `euler.hpp` : flux d'Euler compressible 4 var (+ `eigenvalues`, roles de variables).
- `source.hpp` : termes sources de potentiel / gravite.
- `elliptic.hpp` : seconds membres elliptiques (charge, fond, gravite).
- `langmuir.hpp` : briques Langmuir (oscillation electrostatique). `advection_diffusion.hpp` / `two_fluid_isothermal.hpp` : briques associees.

### `numerics/` : numerique local (couche 2)
- `numerical_flux.hpp` : flux de Riemann en politique (template) : Rusanov / HLL / HLLC / Roe.
- `reconstruction.hpp` : reconstruction d'interface : NoSlope / MUSCL (Minmod, VanLeer) / WENO5-Z.
- `spatial_operator.hpp` : `assemble_rhs` (R = -div F + S) + `compute_face_fluxes` (flux de face pour le reflux) + `assemble_rhs_masked` (masque de domaine disque escalier 0/1).
- `spatial_operator_eb.hpp` : `assemble_rhs_eb`, transport cut-cell / embedded-boundary conservatif sur disque (apertures `alpha_f` + fraction de volume `kappa`).
- `spatial_operator_polar.hpp` : `assemble_rhs_polar`, transport polaire additif (divergence en metrique annulaire, le cartesien reste intouche).
- `spatial_discretisation.hpp` : assemblage du couple (reconstruction x flux) par bloc.
- `lorentz_eliminator.hpp` : `B = I - theta dt (v x B_z)` 2x2 + son inverse analytique ferme (reconstruction de la vitesse dans l'etage Schur).

### `numerics/elliptic/` : Poisson
- `elliptic_solver.hpp` : concept `EllipticSolver` (contrat resoudre D phi = f).
- `elliptic_interface.hpp` : concepts `EllipticOperator` / `EllipticSolver` / `FieldPostProcessor` (contrat REELLEMENT commun ; fonctions libres exclues, role d'operateur porte par `GeometricMG`).
- `elliptic_problem.hpp` : types descriptifs de l'etage elliptique (`EllipticProblem`, `FieldPostProcess`, `make_elliptic_solver`).
- `poisson_operator.hpp` : Laplacien 5 points + lisseur Gauss-Seidel red-black ; `apply_laplacian` porte aussi le tenseur plein (eps_x/eps_y + croises a_xy/a_yx) + kappa.
- `geometric_mg.hpp` : multigrille geometrique (V-cycle), `solve_robust` anti-divergence, `set_epsilon` (eps(x) variable, anisotrope), terme de reaction kappa (Helmholtz / ecrante).
- `krylov_solver.hpp` : `TensorKrylovSolver`, BiCGStab matrice-libre pour l'operateur condense non auto-adjoint, preconditionne par N V-cycles `GeometricMG` symetriques.
- `poisson_fft.hpp` : Poisson spectral direct (FFT), distribue par bandes (MPI_Alltoall), correctif `n` non puissance de 2 ; `fft1d` reutilise par le solveur polaire.
- `poisson_fft_solver.hpp` : backend `EllipticSolver` FFT : `PoissonFFTSolver` (mono-rang) + `DistributedFFTSolver` (bandes MPI).
- `polar_poisson_solver.hpp` : `PolarPoissonSolver`, Poisson polaire DIRECT sur anneau (FFT-en-theta + tridiagonale-en-r, exact par mode azimutal).
- `polar_tensor_operator.hpp` : `PolarTensorKrylovSolver`, operateur elliptique polaire a tenseur plein (termes croises a_rt/a_tr), iteratif (Schur polaire).
- `cut_fraction.hpp` : primitive `detail::cut_fraction` partagee (cut-cell elliptique + transport EB : distance de coupe, fraction de volume).

### `numerics/time/` : temps + AMR (couche 5)
- `time_integrator.hpp` : tags SSPRK + `TimePolicy` explicite / implicite / IMEX / prescrit.
- `time_steppers.hpp` : integrateurs OBJETS (`ForwardEuler`, `SSPRK2Step`, `SSPRK3Step`, `take_step`).
- `implicit_stepper.hpp` : defaut implicite (Newton local), IMEX partiel via `is_implicit(c)`.
- `scheduler.hpp` : sous-cyclage generique d'un `CoupledSystem` selon les policies de blocs.
- `ssprk.hpp` : SSPRK2 / SSPRK3 (Shu-Osher, TVD).
- `imex.hpp` : IMEX asymptotic-preserving (raide implicite + non-raide explicite).
- `splitting.hpp` : splitting d'operateur Lie (ordre 1) / Strang (ordre 2).
- (l'integrateur AP deux-fluides a quitte le coeur : il vit dans `adc_cases/two_fluid_ap/`, compile a la volee contre les en-tetes generiques.)
- `amr_reflux_mf.hpp` : PARAPLUIE du **moteur AMR de production** ; inclut, dans l'ordre de dependance, `amr_flux_helpers` (avance divergence / source / average_down / ghosts coarse-fine mono-box), `amr_level` (oracle MF mono-box `detail::AmrLevelMF`, hors production), `amr_patch_range` (`PatchRange` / `FluxRegister` / `CoverageMask` + helpers multi-box MPI-safe), `amr_subcycling` (moteur N-niveaux multi-patch interne), `amr_advance` (facade publique `advance_amr` + `LevelHierarchy` / `OwnershipPolicy`).
- `amr_reflux.hpp` / `amr_multilevel.hpp` : reference Fab2D mono-box (2-niveaux / N-niveaux), verite-terrain du moteur ci-dessus.

### `coupling/` : couplage hyperbolique-elliptique (couche 5)
- `coupler.hpp` : coupleur generique (ferme Poisson -> aux -> advance), par etage.
- `system_coupler.hpp` : execution mono-niveau d'un `CoupledSystem`, avec callback pour blocs implicites/IMEX.
- `elliptic_rhs.hpp` : assembleurs de second membre elliptique mono-modele ou multi-blocs.
- `coupling_policy.hpp` : frequence du solve elliptique (PerStage / OncePerStep).
- `amr_coupler_mp.hpp` : coupleur AMR E x B multi-patch + regrid BR, parametre `replicated_coarse`.
- `amr_system_coupler.hpp` : systeme multi-especes porte sur AMR (Poisson grossier + reflux par bloc).
- `amr_regrid_coupler.hpp` : le regrid Berger-Rigoutsos extrait du coupleur multi-patch.
- `amr_level_storage.hpp` : stockage de la hierarchie (niveaux + aux) extrait des coupleurs.
- `amr_diagnostics.hpp` : masse / vitesse de derive via le seam reducteur.
- `spectral_coupler.hpp` : coupleur periodique distribue (FFT par bandes).
- `aux_fill.hpp` : helpers du canal aux partages par les trois coupleurs (Coupler / SystemAssembler / AmrSystemCoupler).
- `coupled_source.hpp` : contrat `CoupledSourceFor` d'une source de couplage inter-especes (apply(system, aux, dt), splitting forward-Euler).
- `coupled_source_program.hpp` : interprete de source couplee generique (bytecode postfixe device, `CsProgram` / `CoupledSourceKernel`, `adc.dsl.CoupledSource`).
- `schur_condensation.hpp` : `ElectrostaticLorentzCondensation` : assemble l'operateur condense A = I + c rho B^{-1} (eps_x/eps_y + croises) et son RHS (etage Schur, Lorentz electrostatique).
- `condensed_schur_source_stepper.hpp` : `CondensedSchurSourceStepper`, etage source condense complet (assemble -> TensorKrylovSolver -> reconstruit) ; opt-in `System::set_source_stage`.
- `polar_condensed_schur_source_stepper.hpp` : pendant POLAIRE de l'etage Schur (operateur tensoriel polaire iteratif).

### `runtime/` : facades runtime + DSL (assise des bindings)
- `system.hpp` : facade `System` (composition multi-blocs mono-niveau, Poisson partage ; geometrie cartesien / polaire / disque, masque `set_disc_domain`, splitting `set_time_scheme`, etage Schur `set_source_stage`, eps(x) / anisotrope / reaction).
- `amr_system.hpp` : facade `AmrSystem` (mono- ET multi-bloc sur hierarchie raffinee ; recon conservative | primitive, flux rusanov | hllc | roe, explicite ET IMEX, regrid d'union des tags ; reste hors parite : pas d'etage Schur global).
- `amr_runtime.hpp` : moteur multi-blocs runtime (`AmrRuntime`, registre type-erase par nom) + regrid d'union des tags.
- `model_spec.hpp` / `model_factory.hpp` : spec de briques -> `CompositeModel` (le coeur ne nomme aucun scenario).
- `block_builder.hpp` / `block_builder_polar.hpp` : fermetures de bloc instanciables hors `System` (fondation backend AOT) ; pendant polaire (`assemble_rhs_polar`, `derive_aux_polar`).
- `grid_context.hpp` : contexte de grille reel (maillage + CL + aux) partage par les chemins de bloc.
- `system_field_solver.hpp` / `system_stepper.hpp` / `system_block_store.hpp` : responsabilites extraites du god-class `System::Impl` (solve_fields ; avance Lie + Strang ; registre de blocs).
- `wall_predicate.hpp` : predicat de paroi conductrice (cercle centre) partage par `System` et `AmrSystem`.
- `dynamic_model.hpp` : modele type-erased a dispatch virtuel (`IModel`), charge en JIT via `.so`.
- `compiled_block_abi.hpp` : ABI `extern "C"` du bloc compile (`add_compiled_block`, marshaling hote, sans AMR/MPI).
- `dsl_block.hpp` : `add_compiled_model` (bloc compile NATIF connu a la compilation ; parite `add_block` bit-identique, validee CPU/Serial ET sur device GH200 via foncteurs nommes A==B `dres=0`).
- `native_loader.hpp` / `abi_key.hpp` : chemin DSL "production" : loader `.so` zero-copie qui inline `add_compiled_model<ProdModel>` (`add_native_block`) ; `abi_key()` rend l'incompatibilite d'ABI EXPLICITE.
- `amr_dsl_block.hpp` : `add_compiled_model` cote `AmrSystem` (pendant multi-niveau du chemin compile).

### `amr/` : maillage adaptatif
- `amr_hierarchy.hpp` : `AmrHierarchy`, la pile de niveaux raffines (niveau 0 = grossier).
- `tag_box.hpp` : grille dense de marqueurs 0/1, entree du clustering.
- `cluster.hpp` : clustering Berger-Rigoutsos (tags -> peu de boxes).
- `regrid.hpp` : regrid dynamique (tague, regroupe, proper nesting).

### `parallel/` : seam HPC
- `comm.hpp` : seam MPI (rang / taille / all-reduce / send-recv), degenere en serie.
- `load_balance.hpp` : equilibrage des boxes sur les rangs (round-robin / SFC).

### Hors `include/`
- `tests/` : suite CTest du coeur ; les tests MPI sont declares seulement si `ADC_USE_MPI=ON`.
- `docs/` : architecture, algorithmes, performance et notes de validation.
