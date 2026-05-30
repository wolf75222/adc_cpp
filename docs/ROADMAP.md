# Roadmap

Liste vivante de ce qui est fait et de ce qui reste, par intention.

## Fait

### Cœur numérique

- Volumes finis Godunov, flux Rusanov / HLL / HLLC, reconstruction MUSCL (NoSlope / Minmod /
  VanLeer), intégration SSPRK2 / SSPRK3, splitting de Strang / Lie.
- Poisson : multigrille géométrique (V-cycle, GS rouge-noir, on-device) ET FFT spectrale
  directe, derrière le concept `EllipticSolver`.
- Pile mesh maison : `MultiFab` / `BoxArray` / `DistributionMapping` / `Geometry`,
  `fill_boundary`, CL physiques, seam `for_each_cell` série / OpenMP / Kokkos.

### Couplage

- Diocotron (dérive E x B), Euler-Poisson (auto-gravité attractive OU plasma
  électrostatique répulsif, via `InteractionKind` : un seul signe sépare l'effondrement de
  Jeans de l'oscillation de Langmuir + explosion de Coulomb), deux-fluides isotherme
  asymptotic-preserving, tous via `aux = grad phi`.
- Schéma AP deux-fluides (Lorentz implicite, Poisson reformulé `beta0`), dispersion isotrope
  validée (3.1%), borne AP à `omega_pe = 1e3`.
- Continuité upwind MUSCL (anti-Gibbs) en option ; champ magnétique : rotation cyclotron
  (fréquence exacte à 0.00%).

### AMR

- Reflux 2-niveaux et N-niveaux mono-box (`amr_step_multilevel_mf`), bit-identiques à la pile
  Fab2D de référence.
- Multi-patch N-niveaux (`amr_step_multilevel_multipatch`) : reflux coverage-aware, routage
  vers la box parente, validé sur deux axes à `0` exact.
- Reflux 2-niveaux multi-patch DISTRIBUÉ MPI (`test_mpi_amr_multipatch`, np=1/2/4 bit-identique,
  grossier répliqué + gather `average_down`/reflux).
- Reflux N-niveaux multi-patch DISTRIBUÉ MPI (`amr_step_multilevel_multipatch` /
  `subcycle_level_mp`, `test_mpi_amr_multipatch3`, 3 niveaux avec niveau intermédiaire
  multi-box réparti dont le parent d'un patch fin tombe sur un autre rang ; np=1/2/4
  bit-identique). Niveau 0 répliqué, niveaux >0 répartis ; ghost-fill et échantillonnage du
  flux grossier par `parallel_copy` quand le parent est réparti, `average_down` et reflux par
  buffer grossier global + `all_reduce_sum`.
- Clustering Berger-Rigoutsos + regrid dynamique ; coupleurs `AmrCoupler` (mono-box) et
  `AmrCouplerMP` (multi-patch + regrid), conservatifs.

### Parallélisme et outils

- OpenMP (déterministe vs série), MPI (bit-identique np=1/2/4, 9 tests `mpirun`), portage GPU
  GH200 (Kokkos, bit-identique CPU).
- Validation numérique (au-delà du bit-identique) : ordre du Laplacien 5 points (L2/Linf=2.00),
  tourbillon isentropique Euler (VanLeer L1 mesuré 1.86), MUSCL mesuré 1.86 / Rusanov mesuré 0.89,
  loi de Gauss du couplage (`div(grad phi)=source`, ordre 2.00), conservation sous regrid.
- Bindings Python (3 solveurs, 1:1 avec les façades), 10 scripts exécutables (GIF/plots), banc
  `bench_amr`, figures de scaling.
- Docs : README, ALGORITHMS, ARCHITECTURE (5 couches), CHOICES, BIBLIOGRAPHY, PERFORMANCE,
  two_fluid_ap, tutoriels 00 à 09, Doxygen + Sphinx.

## En file

### Durcissement de l'architecture (revue de conception)

Issu d'une revue : la faiblesse structurelle est le mélange discrétisation / stockage /
exécution, et un AMR multi-patch pas encore pensé distribué. Voir
[ARCHITECTURE.md](ARCHITECTURE.md) (modèle en cinq couches, sections marquées « cible »).

1. **AMR multi-patch nativement distribué (priorité absolue).** Fait pour le 2-niveaux :
   `amr_step_2level_multipatch` tourne **réellement distribué** (`test_mpi_amr_multipatch`,
   np=1/2/4 **bit à bit identiques**, masse conservée). Le grossier mono-box est répliqué
   (copie par-rang + remplissage périodique local), les patchs fins répartis, `average_down`
   (écrasement couvert) et reflux (addition bordante) remontent par deux buffers grossiers +
   `all_reduce_sum_inplace`. Au passage, un bug mono-rang corrigé : les face-box des flux fins
   se bâtissaient sur les boxes **locales** avec le dmap **global** (tailles incohérentes sous
   MPI). Le même bug latent corrigé dans `subcycle_level_mp`. **Fait aussi pour le N-niveaux :**
   `subcycle_level_mp` / `amr_step_multilevel_multipatch` tourne **réellement distribué**
   (`test_mpi_amr_multipatch3`, 3 niveaux, niveau intermédiaire multi-box réparti dont le parent
   d'un patch fin tombe sur un autre rang ; np=1/2/4 **bit à bit identiques**, masse conservée).
   Niveau 0 répliqué, niveaux >0 répartis. Les cinq points sont résolus : (1) ghost-fill
   parent->enfant et (2) échantillonnage du registre grossier par `parallel_copy` quand le
   parent est réparti (lecture locale quand il est répliqué) ; (3) `average_down` et (4) reflux
   par buffer grossier global + `all_reduce_sum_inplace`, appliqué aux boxes parentes locales ;
   (5) couverture déjà globale. Reste : le coupleur `AmrCouplerMP` au-delà de 2 niveaux
   distribués (son injection d'aux `inject_aux_mb` suppose encore le parent local) ; puis, cible
   finale, chaque patch portant `owner_rank`, `global_box_id`, interfaces coarse-fine globales,
   registre distribué, `load_balance` SFC.
2. **Moteur AMR unifié.** Premier pas fait : l'entrée unifiée `advance_amr(m, LevelHierarchy&,
   dt)` + le type `LevelHierarchy` (vérifié façade-fidèle et conservatif, `test_advance_amr`).
   Reste : nommer les autres objets (`PatchRange`, `CoarseFineInterface`, `FluxRegister`,
   `SubcyclingSchedule`, `RegridPolicy` ; `OwnershipPolicy` ~ `DistributionMapping`) et y
   replier la famille `amr_step_*` (qui encode le cas dans le nom).
3. **Découper l'elliptique.** Avancé : l'`EllipticOperator` existe déjà séparé
   (`poisson_operator.hpp` : `apply_laplacian`, `poisson_residual`, lisseur) ; le
   `LinearSolver` est le concept `EllipticSolver` (MG, FFT) ; l'identité MG = FFT est rendue
   STRUCTURELLE et vérifiée (`test_elliptic_operator` applique le même opérateur canonique aux
   deux solutions, résidus ~1e-14). Fait : `EllipticProblem` (coeff `eps`, CL `BCRec`, nullspace
   `nullspace_const` en un objet, jusqu'ici implicites) et `FieldPostProcess` (convention de
   dérivation `E = -grad phi` via `GradSign::Plus`/`Minus`, jadis la fonction libre
   `coupler_grad_phi`) sont nommés dans `elliptic/elliptic_problem.hpp`. Refactor structurel
   bit-identique : `eps = 1` reste descriptif (le stencil ne le lit pas encore), la fabrique
   `make_elliptic_solver(EllipticProblem)` délègue à la `BCRec`, et `field_postprocess` reproduit
   à l'identique l'expression du coupleur. `test_elliptic_problem` prouve l'égalité bit-à-bit
   (`operator==` strict, pas une tolérance). Reste hors-périmètre tant qu'on veut le bit-identique :
   recâbler les sites en forme `/(2*dx)` (`amr_coupler`, `amr_coupler_mp`, `spectral_coupler`,
   `two_fluid_ap`), division qui peut différer au dernier bit de la forme multiplicative `*cx`.
4. **API mémoire explicite.** Réductions `sum` / `norm_inf` faites (le reste de l'API
   `sync_host` / `sync_device` reste en file). Le seam `for_each.hpp` porte désormais
   `for_each_cell_reduce_sum` et `for_each_cell_reduce_max`, à côté de `for_each_cell` et
   `device_fence()` : sous Kokkos une vraie `parallel_reduce` (MDRangePolicy, `Kokkos::Sum`
   / `Kokkos::Max`, bloquante côté hôte donc sans `device_fence()` préalable) ; en série et
   sous OpenMP une boucle hôte séquentielle. `sum` et `norm_inf` (multifab.hpp / mf_arith.hpp)
   appellent ce seam par fab local puis agrègent ; les deux `device_fence()` qui protégeaient
   leur boucle hôte ont disparu (absorbés par la réduction), les autres `device_fence()` (accès
   hôte ailleurs) restent.
   Conséquence FP : `sum` n'est plus bit-identique à la boucle hôte SOUS KOKKOS (la somme par
   tuile réassocie l'addition flottante, non associative en IEEE754). En série et sous OpenMP
   `sum` reste exact : on garde volontairement la boucle hôte séquentielle pour OpenMP, car
   `reduction(+:)` réordonnerait la somme par thread et casserait la garantie « OpenMP identique
   à la série » du repo. `norm_inf` reste EXACT partout (un max de valeurs absolues, sans
   arrondi et invariant par réordonnancement). `Kokkos::Sum` est déterministe par tuile (pas
   d'atomics flottants), donc deux `sum` sur des données inchangées rendent le même bit
   (idempotence, clé pour `test_fill_boundary/sum_unchanged`). Contrat verrouillé par
   `test_reduce` (sum_constant exact, sum varié en écart relatif < 1e-10, norm_inf strict,
   idempotence).
5. **Séparer les trois familles de ghosts** en briques nommées testables. Largement fait :
   `fill_physical_bc` (BoundaryCondition, testé seul `test_physical_bc`), `fill_boundary`
   (GhostExchange, testé `test_mpi_fillboundary`), `mf_fill_fine_ghosts_*`
   (AMRBoundaryInterpolation) sont déjà séparés ; `fill_ghosts` n'est qu'une composition
   explicite de (1) + (2). Reste : remonter le coarse-fine en helper nommé de premier niveau.
6. **CouplingPolicy mince (FAIT).** Hiérarchie, regrid et diagnostics sortis des coupleurs
   AMR en composants nommés : `coupling/amr_level_storage.hpp` (`AmrLevelStack<Level>`,
   stockage niveaux + aux, câblage et réallocation d'aux), `coupling/amr_regrid_coupler.hpp`
   (`amr_regrid_finest`, Berger-Rigoutsos en free function template, `comm.hpp` inclus
   explicitement pour `n_ranks()`), `coupling/amr_diagnostics.hpp` (`amr_mass`,
   `amr_max_drift_speed`). `AmrCoupler` / `AmrCouplerMP` ne font plus qu'ordonner
   (`sync_down -> compute_aux -> step`, `regrid()` délégué) ; les primitives d'injection
   restent des helpers `detail::`. Extraction structurelle bit-identique : équivalence
   `max|dUc| = 0` et conservation de masse à l'arrondi inchangées
   (`test_amr_coupler`, `test_amr_coupler_mp`).
7. **Suite de validation numérique (FAIT).** Le bit-identique ne prouve pas la justesse ;
   la suite couvre désormais : ordre du Laplacien 5 points (`test_poisson_convergence`,
   L2/Linf = 2.00, Dirichlet + périodique + nullspace) ; tourbillon isentropique d'Euler
   (VanLeer L1 mesuré 1.86, asserti > 1.7, `test_euler`) ; ordre MUSCL mesuré 1.86 (asserti
   > 1.7) / Rusanov mesuré 0.89 (`test_muscl_convergence`) ; loi de Gauss discrète du couplage
   `div(grad phi) = source` à 2.00 (`test_gauss_law`) ; limite AP quantifiée, uniforme sur 8
   décades de raideur (`test_ap_limit`) ; invariants diocotron (masse, principe du maximum,
   enstrophie non croissante, `test_diocotron_stability`) ; conservation flux coarse-fine
   exacte par reflux (`test_amr_reflux`, masse à 1e-12) + conservation sous regrid
   (`test_amr_coupler_mp`, dérive à 1e-9).

### Physique magnétisée (cible Hoffart)

- Push de Boris E+B combiné (au lieu du splitting de Strang externe), reformulation AP
  tensorielle sous champ fort, reproduction d'un benchmark Hoffart précis.

### Performance

- Région OpenMP consolidée au-dessus de la boucle de niveaux du multigrille (le seul levier
  identifié pour les charges MG-dominées).

### Confort

- MUSCL sur le momentum du deux-fluides ; GIF du coupleur multi-patch ; tutoriels
  supplémentaires si besoin.
