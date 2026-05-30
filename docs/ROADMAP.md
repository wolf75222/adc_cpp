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
- Clustering Berger-Rigoutsos + regrid dynamique ; coupleurs `AmrCoupler` (mono-box) et
  `AmrCouplerMP` (multi-patch + regrid), conservatifs.

### Parallélisme et outils

- OpenMP (déterministe vs série), MPI (bit-identique np=1/2/4, 9 tests `mpirun`), portage GPU
  GH200 (Kokkos, bit-identique CPU).
- Validation numérique (au-delà du bit-identique) : ordre du Laplacien 5 points (L2/Linf=2.00),
  tourbillon isentropique Euler (L1~2), MUSCL ~2 / Rusanov ~1, loi de Gauss du couplage
  (`div(grad phi)=source`, ordre 2.00), conservation sous regrid.
- Bindings Python (3 solveurs, 1:1 avec les façades), 10 scripts exécutables (GIF/plots), banc
  `bench_amr`, figures de scaling.
- Docs : README, ALGORITHMS, ARCHITECTURE (4 couches), CHOICES, BIBLIOGRAPHY, PERFORMANCE,
  two_fluid_ap, tutoriels 00 à 09, Doxygen + Sphinx.

## En file

### Durcissement de l'architecture (revue de conception)

Issu d'une revue : la faiblesse structurelle est le mélange discrétisation / stockage /
exécution, et un AMR multi-patch pas encore pensé distribué. Voir
[ARCHITECTURE.md](ARCHITECTURE.md) (modèle en quatre couches, sections marquées « cible »).

1. **AMR multi-patch nativement distribué (priorité absolue).** Fait pour le 2-niveaux :
   `amr_step_2level_multipatch` tourne **réellement distribué** (`test_mpi_amr_multipatch`,
   np=1/2/4 **bit à bit identiques**, masse conservée). Le grossier mono-box est répliqué
   (copie par-rang + remplissage périodique local), les patchs fins répartis, `average_down`
   (écrasement couvert) et reflux (addition bordante) remontent par deux buffers grossiers +
   `all_reduce_sum_inplace`. Au passage, un bug mono-rang corrigé : les face-box des flux fins
   se bâtissaient sur les boxes **locales** avec le dmap **global** (tailles incohérentes sous
   MPI). Le même bug latent corrigé dans `subcycle_level_mp`. Reste : rendre distribué le
   chemin N-niveaux récursif (`subcycle_level_mp`, grossier MULTI-box réparti, qu'on ne peut
   répliquer à bon marché). Cinq points supposent le parent local (via `mf_find_box`) et
   demandent un FillPatch façon AMReX via le `parallel_copy` déjà présent : (1) ghost-fill
   parent->enfant, (2) échantillonnage du registre grossier, (3) `average_down` routé vers
   le propriétaire de la box parente, (4) reflux routé de même, (5) couverture (déjà
   globale). Puis, cible finale, chaque patch portant `owner_rank`, `global_box_id`,
   interfaces coarse-fine globales, registre distribué, `load_balance` SFC.
2. **Moteur AMR unifié.** Replier la famille `amr_step_2level_mf` / `_multilevel_mf` /
   `_2level_multipatch` / `_multilevel_multipatch` (duplication par cas particulier) sur un
   seul `advance_amr(hierarchy, dt, operators, schedule, execution)`, au-dessus d'objets
   nommés : `LevelHierarchy`, `PatchRange`, `CoarseFineInterface`, `FluxRegister`,
   `SubcyclingSchedule`, `RegridPolicy`, `OwnershipPolicy`.
3. **Découper l'elliptique.** Avancé : l'`EllipticOperator` existe déjà séparé
   (`poisson_operator.hpp` : `apply_laplacian`, `poisson_residual`, lisseur) ; le
   `LinearSolver` est le concept `EllipticSolver` (MG, FFT) ; l'identité MG = FFT est rendue
   STRUCTURELLE et vérifiée (`test_elliptic_operator` applique le même opérateur canonique aux
   deux solutions, résidus ~1e-14). Reste : `EllipticProblem` comme type distinct (coeffs,
   CL, nullspace en un objet, aujourd'hui implicites) et `FieldPostProcess` comme composant
   nommé (`E = -grad phi`, aujourd'hui la fonction `coupler_grad_phi`).
4. **API mémoire explicite.** Remplacer la discipline manuelle `device_fence()` par
   `device_reduce` / `device_norm_inf` / `sync_host` / `sync_device` ; faire de `sum` et
   `norm_inf` de vraies réductions device (pas des boucles hôte protégées par fence).
5. **Séparer les trois familles de ghosts** en briques nommées testables. Largement fait :
   `fill_physical_bc` (BoundaryCondition, testé seul `test_physical_bc`), `fill_boundary`
   (GhostExchange, testé `test_mpi_fillboundary`), `mf_fill_fine_ghosts_*`
   (AMRBoundaryInterpolation) sont déjà séparés ; `fill_ghosts` n'est qu'une composition
   explicite de (1) + (2). Reste : remonter le coarse-fine en helper nommé de premier niveau.
6. **CouplingPolicy mince.** Sortir la hiérarchie, le regrid et les diagnostics des coupleurs
   pour que la policy ne fasse plus qu'ordonner les opérations.
7. **Suite de validation numérique (FAIT).** Le bit-identique ne prouve pas la justesse ;
   la suite couvre désormais : ordre du Laplacien 5 points (`test_poisson_convergence`,
   L2/Linf = 2.00, Dirichlet + périodique + nullspace) ; tourbillon isentropique d'Euler
   (L1 ~2, `test_euler`) ; ordre MUSCL ~2 / Rusanov ~1 (`test_muscl_convergence`) ; loi de
   Gauss discrète du couplage `div(grad phi) = source` à 2.00 (`test_gauss_law`) ; limite AP
   quantifiée, uniforme sur 8 décades de raideur (`test_ap_limit`) ; invariants diocotron
   (masse, principe du maximum, enstrophie non croissante, `test_diocotron_stability`) ;
   conservation flux coarse-fine exacte par reflux + conservation sous regrid
   (`test_amr_coupler_mp`).

### Physique magnétisée (cible Hoffart)

- Push de Boris E+B combiné (au lieu du splitting de Strang externe), reformulation AP
  tensorielle sous champ fort, reproduction d'un benchmark Hoffart précis.

### Performance

- Région OpenMP consolidée au-dessus de la boucle de niveaux du multigrille (le seul levier
  identifié pour les charges MG-dominées).

### Confort

- MUSCL sur le momentum du deux-fluides ; GIF du coupleur multi-patch ; tutoriels
  supplémentaires si besoin.
