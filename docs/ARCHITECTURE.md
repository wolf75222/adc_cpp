# Architecture de adc_cpp

Coeur C++23 pour les systemes hyperbolique-elliptique couples sur AMR, ecrit pour
OpenMP + MPI + Kokkos. Les modeles physiques, facades, exemples et bindings Python
vivent dans `adc_cases`, qui consomme ce coeur via `adc::adc`.

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
`flux`, `source`, `max_wave_speed`, `wave_speeds`, `elliptic_rhs`, toutes `ADC_HD`. Aucun
acces au stockage ou au parallelisme.

```cpp
struct EulerPoisson {                       // bon : local, device-callable
  ADC_HD State flux(const State& u, const Aux& a, int dir) const;
  ADC_HD State source(const State& u, const Aux& a) const;     // g = -grad phi
  ADC_HD Real elliptic_rhs(const State& u) const;              // s * 4 pi G (rho - rho0)
};
```

**Les modeles, axe par axe** (verifie contre le code) :

| Modele | U (`State`) | `aux` | `flux` | `source` | `elliptic_rhs` |
|---|---|---|---|---|---|
| `Diocotron` | `n_e` (1 var) | `(phi, dphi/dx, dphi/dy)` | advection par derive E x B : `n_e v_E`, `v_E = (-dphi/dy, dphi/dx)/B0` | 0 | `alpha (n_e - n_i0)` |
| `Euler` | `(rho, rho u, rho v, E)` | inutilise (present pour le concept) | flux d'Euler compressible | 0 | `rho` (densite de masse, inutilisee en Euler pur) |
| `EulerPoisson` | `(rho, rho u, rho v, E)` | `grad phi` | DELEGUE a `Euler` | force `g = -grad phi` : `(0, rho g_x, rho g_y, rho u . g)` | `s * 4 pi G (rho - rho0)`, `s = +-1` |

Diocotron exerce le chemin **aux vers flux** (le potentiel entre par le flux) ; Euler-Poisson le
chemin **aux vers source** (le potentiel entre par la source, le flux reste celui d'Euler). C'est
ce qui unifie les deux sous le MEME operateur spatial, sans specialisation.

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
  seul (`test_mpi_fillboundary` quand MPI est active).
- (3) `AMRBoundaryInterpolation` = `mf_fill_fine_ghosts_*` (interp espace+temps coarse-fine).
`fill_ghosts` n'est PAS un fourre-tout : c'est une COMPOSITION explicite de (1) puis (2)
(`fill_boundary` ; `fill_physical_bc`). **Reste (cible)** : remonter (3), qui vit dans le pas
AMR, en helper nomme de premier niveau (et le rendre distribue, cf. section 8).

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

`integrator/time_integrator.hpp`, `scheduler.hpp`, `ssprk.hpp`, `imex.hpp` (AP) et
`splitting.hpp`. Une `TimePolicy` nomme, par bloc, le traitement temporel
(explicite, implicite, IMEX, prescrit) et le nombre de sous-pas. Le scheduler lit cette
politique et appelle l'operateur adapte ; il ne connait pas la formule du flux. Le temps
compose des operateurs, il ne possede pas la physique.

`coupling/` (`Coupler`, `SystemCoupler`, `AmrCoupler`, `AmrCouplerMP`,
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
| `core/` | physique / systeme | `types` (`ADC_HD`, `Real`), `state` (`StateVec<N>`), `physical_model` (concept), `EquationBlock`, `CoupledSystem`, `allocator` (Arena) |
| `operator/` | numerique | `numerical_flux` (Rusanov/HLL/HLLC), `reconstruction` (MUSCL), `spatial_operator` (`assemble_rhs`) |
| `elliptic/` | numerique + temps | concept `EllipticSolver` ; `geometric_mg` (V-cycle) ; `poisson_fft`(+`_solver`) ; `poisson_operator` |
| `mesh/` (donnees) | maillage / donnees | `box2d`, `box_array`, `distribution_mapping`, `fab2d`/`multifab`, `geometry`, `refinement`, `box_hash` |
| `mesh/` (execution) | execution | `for_each` (seam `for_each_cell`), `fill_boundary` (GhostExchange), `physical_bc`, `mf_arith` (operateurs de grille qui bouclent le seam) |
| `parallel/` | execution | `comm` (seam MPI), `load_balance` (Z-order + knapsack) |
| `amr/` | maillage adaptatif | `amr_hierarchy` (conteneur de niveaux), `cluster` (Berger-Rigoutsos, arithmetique entiere), `regrid` (politique de remaillage), `tag_box` (grille de marqueurs) |
| `integrator/` | temps | `TimePolicy`, scheduler par sous-pas, SSPRK, IMEX, splitting, moteur AMR `advance_amr` |
| `coupling/` | temps / couplage | `elliptic_rhs`, `Coupler`, `SystemCoupler`, `coupling_policy`, `amr_coupler`, `amr_coupler_mp`, `spectral_coupler`, diagnostics AMR |

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
  boite unique** : il assert `n_ranks()==1 && ba.size()==1`) et `DistributedFFTSolver` (FFT
  distribuee par BANDES, `MPI_Alltoall`, enveloppant le composant autonome `PoissonFFT` de
  `elliptic/poisson_fft.hpp`) en sont TROIS implementations. `SpectralCoupler` DELEGUE desormais a
  `DistributedFFTSolver` (il ne re-implemente plus la FFT) : il ORCHESTRE un `EllipticSolver`, il
  n'en contient pas. `Coupler<Model, Elliptic = GeometricMG>` depend du concept, pas d'un backend.
- **Identite MG = FFT rendue STRUCTURELLE** : `test_elliptic_operator` applique le MEME
  operateur canonique `poisson_residual` aux deux solutions -> residus `3.4e-14` (MG) et
  `7.2e-14` (FFT), solutions identiques a `1.3e-16`. Les deux inversent prouvablement le meme
  operateur.

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

## 8. AMR distribue

**Etat.** Le coeur fournit les briques AMR dans `amr/` et le moteur MultiFab dans
`integrator/amr_reflux_mf.hpp`. Le schema reste le meme : tagging, clustering
Berger-Rigoutsos, regrid, sous-cyclage, average-down et reflux conservatif. Les roles
importants sont explicites : `FluxRegister` accumule les flux de face, `CoverageMask`
evite les doubles corrections, `DistributionMapping` porte l'ownership des boxes.

**Couplage.** `AmrCoupler` et `AmrCouplerMP` restent les drivers AMR mono-modele. Le nouveau
`SystemCoupler` couvre pour l'instant le mono-niveau multi-blocs ; l'etape suivante est de
faire porter la meme notion d'`EquationBlock` par le moteur AMR, au lieu de dupliquer la
logique par cas physique.

**Tests coeur.** Les invariants AMR actuellement couverts dans ce depot sont le raffinement,
la hierarchie, le clustering, le regrid, le reflux, le masque de couverture, les diagnostics
AMR et le load balancing. Les tests MPI AMR plus applicatifs vivent ou devront vivre au niveau
des cas downstream si le modele physique est necessaire.

## 9. Backends : propriete de la bibliotheque, pas un drapeau par cible

OpenMP, MPI, HDF5 et Kokkos sont attaches a la cible d'interface `adc`. **Tout ce qui lie
`adc` herite du backend** : les tests du coeur et les applications downstream. On configure
une seule fois :

```
cmake -B build                       # serie
cmake -B build -DADC_USE_OPENMP=ON   # CPU multi-thread (_OPENMP)
cmake -B build -DADC_USE_MPI=ON      # distribue (ADC_HAS_MPI + MPI::MPI_CXX)
cmake -B build -DADC_USE_KOKKOS=ON \ # GPU / CPU portable (ADC_HAS_KOKKOS)
   -DCMAKE_CXX_COMPILER=$K/bin/nvcc_wrapper -DKokkos_ROOT=$K
```

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
| `adc_cases` | modeles, facades, exemples, Python, runs applicatifs. | consomme `adc::adc` |

Regle actuelle : `adc_cpp` reste le coeur generique. Une API Python ou une facade stable
doit vivre au-dessus, dans `adc_cases`, pour ne pas faire dependre le coeur de cas physiques.

## 11. Validation : logicielle ET numerique

**Bit-identique = filet logiciel, pas preuve numerique.** Prouver que le multipatch est
bit-identique a la reference prouve que la refactorisation n'a rien casse. Ca ne prouve pas
que le comportement est numeriquement correct. Les deux sont necessaires.

Fait aujourd'hui dans `adc_cpp` :
- Tests : 32 tests CPU serie par defaut (`ctest --test-dir build`).
- Numerique coeur : maillage, halos, AMR, reflux, multigrille, Poisson, discretisations,
  IMEX/AP, splitting, `EquationBlock`, `CoupledSystem`, `SystemCoupler`.
- MPI : tests dedies actives quand `-DADC_USE_MPI=ON`.
- Les validations applicatives (diocotron, runs ROMEO, Python) vivent dans `adc_cases`.

## 12. Comparaison AMReX

Correspondances : `MultiFab`, `BoxArray`/`DistributionMapping`, `Geometry`, `AmrLevel`,
FillBoundary, Arena, reflux, MLMG ~ `GeometricMG`. Divergences assumees : pas de `MFIter`
(on itere `for_each_cell` + fab local, GPU-ready) ; l'operateur elliptique joue `LinOp`
mais Laplacien a coefficient constant (EB en escalier) ; le FluxRegister / FillPatch
multi-patch 2-niveaux est distribue (bit-identique np=1/2/4), grossier replique ou de-replique
(multi-box reparti) ; reste le regrid d'un niveau intermediaire reparti pour 3+ niveaux (section 8).

## 13. Arborescence detaillee (fichier par fichier)

Coeur header-only sous `include/adc/`, range par couche. Une ligne par fichier : ce que
c'est, et pourquoi il est la. Descriptions tirees du doc-comment de chaque en-tete.

### `core/` : types et contrat
- `types.hpp` : scalaires de base (`Real`) + macro `ADC_HD` (host/device). Minimal, inclus partout.
- `state.hpp` : `State` / `Aux`, les deux types ponctuels de la couche physique (POD device-callable).
- `physical_model.hpp` : le concept `PhysicalModel` (contrat flux / source / max_wave_speed / elliptic_rhs).
- `equation_block.hpp` : un bloc d'equation = nom, `MultiFab`, modele, discretisation spatiale, politique temps.
- `coupled_system.hpp` : tuple type de plusieurs `EquationBlock`, iteration generique par bloc.
- `allocator.hpp` : allocateur du `Fab2D`, std (host) ou `cudaMallocManaged` (memoire unifiee GH200).

### `mesh/` : donnees + seams (couche 3)
- `box2d.hpp` : `Box2D`, espace d'indices d'une grille cartesienne 2D.
- `box_array.hpp` : `BoxArray`, les boxes qui pavent un niveau (disjointes, couvrantes).
- `box_hash.hpp` : hash spatial (bins uniformes) pour retrouver les boxes voisines.
- `distribution_mapping.hpp` : `DistributionMapping`, box -> rang MPI.
- `fab2d.hpp` : `Fab2D`, donnees mono-grille sur une Box2D + ghosts (equivalent du FArrayBox).
- `multifab.hpp` : `MultiFab`, champ distribue (collection de Fab2D) (equivalent du MultiFab AMReX).
- `geometry.hpp` : `Geometry`, correspondance indices <-> coordonnees physiques.
- `for_each.hpp` : **seam d'execution** `for_each_cell` + `for_each_cell_reduce_*` (serie/OpenMP/Kokkos) + `device_fence`.
- `fill_boundary.hpp` : echange de halos intra-niveau (begin/end non-bloquant).
- `physical_bc.hpp` : CL physiques du domaine + `fill_ghosts`.
- `refinement.hpp` : transfert AMR ratio r : prolongation (interp) + restriction (average_down).
- `mf_arith.hpp` : combinaisons lineaires de MultiFab (saxpy, norm_inf, sum) pour les etages RK.

### `operator/` : numerique local (couche 2)
- `numerical_flux.hpp` : flux de Riemann en politique (template) : Rusanov / HLL / HLLC.
- `reconstruction.hpp` : reconstruction d'interface : NoSlope / MUSCL (Minmod, VanLeer, MC) / WENO5-Z.
- `spatial_operator.hpp` : `assemble_rhs` (R = -div F + S) + `compute_face_fluxes` (flux de face pour le reflux).

### `elliptic/` : Poisson
- `elliptic_solver.hpp` : concept `EllipticSolver` (contrat resoudre D phi = f).
- `elliptic_problem.hpp` : types descriptifs de l'etage elliptique.
- `poisson_operator.hpp` : Laplacien 5 points + lisseur Gauss-Seidel red-black.
- `geometric_mg.hpp` : multigrille geometrique (V-cycle), `solve_robust` anti-divergence.
- `poisson_fft.hpp` : Poisson spectral direct (FFT), distribue par bandes (MPI_Alltoall).
- `poisson_fft_solver.hpp` : backend `EllipticSolver` FFT : `PoissonFFTSolver` (mono-rang) + `DistributedFFTSolver` (bandes MPI).

### `integrator/` : temps + AMR (couche 5)
- `time_integrator.hpp` : tags SSPRK + `TimePolicy` explicite / implicite / IMEX / prescrit.
- `scheduler.hpp` : sous-cyclage generique d'un `CoupledSystem` selon les policies de blocs.
- `ssprk.hpp` : SSPRK2 / SSPRK3 (Shu-Osher, TVD).
- `imex.hpp` : IMEX asymptotic-preserving (raide implicite + non-raide explicite).
- `splitting.hpp` : splitting d'operateur Lie (ordre 1) / Strang (ordre 2).
- `amr_reflux_mf.hpp` : **moteur AMR de production** `advance_amr` (multi-patch N-niveaux distribue) + types `FluxRegister` / `CoverageMask` ; la pile mono-box `amr_*_mf` y vit en `detail::` (oracle de validation).
- `amr_reflux.hpp` / `amr_multilevel.hpp` : reference Fab2D mono-box (2-niveaux / N-niveaux), verite-terrain du moteur ci-dessus.

### `coupling/` : couplage hyperbolique-elliptique (couche 5)
- `coupler.hpp` : coupleur generique (ferme Poisson -> aux -> advance), par etage.
- `system_coupler.hpp` : execution mono-niveau d'un `CoupledSystem`, avec callback pour blocs implicites/IMEX.
- `elliptic_rhs.hpp` : assembleurs de second membre elliptique mono-modele ou multi-blocs.
- `coupling_policy.hpp` : frequence du solve elliptique (PerStage / OncePerStep).
- `amr_coupler.hpp` : coupleur AMR E x B mono-box (route par `advance_amr`).
- `amr_coupler_mp.hpp` : coupleur AMR E x B multi-patch + regrid BR, parametre `replicated_coarse`.
- `amr_regrid_coupler.hpp` : le regrid Berger-Rigoutsos extrait du coupleur multi-patch.
- `amr_level_storage.hpp` : stockage de la hierarchie (niveaux + aux) extrait des coupleurs.
- `amr_diagnostics.hpp` : masse / vitesse de derive via le seam reducteur.
- `spectral_coupler.hpp` : coupleur periodique distribue (FFT par bandes).

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
