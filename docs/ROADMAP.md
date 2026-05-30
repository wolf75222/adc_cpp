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
- Clustering Berger-Rigoutsos + regrid dynamique ; coupleurs `AmrCoupler` (mono-box) et
  `AmrCouplerMP` (multi-patch + regrid), conservatifs.

### Parallélisme et outils

- OpenMP (déterministe vs série), MPI (bit-identique np=1/2/4, 8 tests `mpirun`), portage GPU
  GH200 (Kokkos, bit-identique CPU).
- Bindings Python (3 solveurs, 1:1 avec les façades), banc `bench_amr`, figures de scaling.
- Docs : README, ALGORITHMS, ARCHITECTURE, CHOICES, BIBLIOGRAPHY, PERFORMANCE, two_fluid_ap,
  tutoriels 00 à 08, Doxygen + Sphinx.

## En file

### Durcissement de l'architecture (revue de conception)

Issu d'une revue : la faiblesse structurelle est le mélange discrétisation / stockage /
exécution, et un AMR multi-patch pas encore pensé distribué. Voir
[ARCHITECTURE.md](ARCHITECTURE.md) (modèle en quatre couches, sections marquées « cible »).

1. **AMR multi-patch nativement distribué (priorité absolue).** Avancé : la couverture est
   déjà bâtie sur le `box_array()` global (MPI-safe), et le REFLUX de `amr_step_2level_multipatch`
   est réécrit en forme distribuée (buffer grossier répliqué + `all_reduce_sum_inplace`),
   bit à bit identique en série et à np=1. Blocage actif : le grossier mono-box vit sur un
   seul rang, mais les rangs portant un patch fin ont besoin du champ grossier (ghost-fill)
   et du flux grossier (registre). Il faut donc RÉPLIQUER le grossier (broadcast du champ +
   du registre vers les rangs fins) et faire remonter `average_down` par le même buffer
   additif ; puis généraliser au chemin N-niveaux récursif (`subcycle_level_mp`). Cible
   finale : chaque patch porte `owner_rank`, `global_box_id`, interfaces coarse-fine
   globales, registre distribué, politique de réduction conservative ; `load_balance` SFC
   sur le multi-box. Repousser fige une fausse abstraction distribuée.
2. **Moteur AMR unifié.** Replier la famille `amr_step_2level_mf` / `_multilevel_mf` /
   `_2level_multipatch` / `_multilevel_multipatch` (duplication par cas particulier) sur un
   seul `advance_amr(hierarchy, dt, operators, schedule, execution)`, au-dessus d'objets
   nommés : `LevelHierarchy`, `PatchRange`, `CoarseFineInterface`, `FluxRegister`,
   `SubcyclingSchedule`, `RegridPolicy`, `OwnershipPolicy`.
3. **Découper l'elliptique.** `EllipticProblem` (équation, coeffs, CL, nullspace) /
   `EllipticOperator` (stencil, `apply`, `residual`, restriction/prolongation) /
   `LinearSolver` (MG, FFT, CG) / `FieldPostProcess` (E = -grad phi). Formaliser l'identité
   MG = FFT dans un `OperatorSpec` partagé, pas seulement dans la doc.
4. **API mémoire explicite.** Remplacer la discipline manuelle `device_fence()` par
   `device_reduce` / `device_norm_inf` / `sync_host` / `sync_device` ; faire de `sum` et
   `norm_inf` de vraies réductions device (pas des boucles hôte protégées par fence).
5. **Séparer les trois familles de ghosts** en briques nommées testables :
   `BoundaryCondition` (physique), `GhostExchange` (parallèle), `AMRBoundaryInterpolation`
   (coarse-fine).
6. **CouplingPolicy mince.** Sortir la hiérarchie, le regrid et les diagnostics des coupleurs
   pour que la policy ne fasse plus qu'ordonner les opérations.
7. **Suite de validation numérique** (le bit-identique ne prouve pas la justesse) : solutions
   manufacturées 1D/2D + ordre L1/L2/Linf, conservation sous regrid, conservation du flux
   coarse-fine, nullspace de Poisson périodique, Gauss discret div(E) = rho, limite AP,
   invariants diocotron (masse, énergie, moment, enstrophie).

### Physique magnétisée (cible Hoffart)

- Push de Boris E+B combiné (au lieu du splitting de Strang externe), reformulation AP
  tensorielle sous champ fort, reproduction d'un benchmark Hoffart précis.

### Performance

- Région OpenMP consolidée au-dessus de la boucle de niveaux du multigrille (le seul levier
  identifié pour les charges MG-dominées).

### Confort

- MUSCL sur le momentum du deux-fluides ; GIF du coupleur multi-patch ; tutoriels
  supplémentaires si besoin.
