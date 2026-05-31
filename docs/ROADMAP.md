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
- Continuité upwind MUSCL (anti-Gibbs) en option ; champ magnétique : push de Boris E+B
  combiné (`tfap_boris`, fréquence cyclotron exacte à 0.00%, dérive E x B préservée sans
  croissance séculaire), à la place du splitting de Strang externe.

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
   (5) couverture déjà globale. Le coupleur `AmrCouplerMP` est lui aussi distribué : son
   injection d'aux `coupler_inject_aux_mb` passe par `parallel_copy` quand le parent est réparti
   (niveau 0 répliqué : lecture locale), vérifié contre l'analytique np=1/2/4 par
   `test_mpi_coupler_inject`. Vers la cible finale : le `load_balance` SFC (Z-order de Morton,
   `make_sfc_distribution`) est désormais BRANCHÉ sur l'AMR distribué et vérifié, pas seulement
   testé comme algorithme en série : `test_mpi_amr_multipatch3` exécute le pas 3-niveaux sous
   répartition Morton et obtient `maxdiff = 0` vs la référence (rang 0) à np=1/2/4, déséquilibre
   1.000. Le registre de flux est aussi RESTREINT À L'INTERFACE coarse-fine : son `all_reduce`
   ne porte plus sur tout le domaine grossier (`O(NX*NY)`) mais sur la boîte englobante des
   empreintes fines (`O(interface)`), bit-identique (les cellules hors interface étaient nulles,
   sautées à l'application ; np=1/2/4 `maxdiff = 0`). Une conception dédiée (workflow lecture
   seule) a tranché le reste : le gather collectif RÉSIDUEL est irréductible tant que le grossier
   est répliqué (chaque rang doit voir la correction) ; le supprimer entièrement (registre point
   à point pur, zéro collective) exige de DÉ-RÉPLIQUER le niveau 0, ce qui casse le Poisson MG,
   `fill_periodic_local` et la mesure de masse locale des tests pour un gain marginal sur un
   grossier petit. Décision : **NO-GO sur la dé-réplication** pour ce cas (diocotron / Euler-Poisson,
   base 32x32) ; le goulot `O(NX*NY)` est déjà éliminé sans elle. Reste, optionnel : `owner_rank` /
   `global_box_id` explicites par patch (aujourd'hui dans `DistributionMapping` + indice `BoxArray`),
   et la dé-réplication seulement si un jour le niveau de base devient gros.
2. **Moteur AMR unifié.** Entrée unifiée faite : `advance_amr(m, LevelHierarchy&, dt)` + le
   type `LevelHierarchy`, vérifiée façade-fidèle en **2 et 3 niveaux** (`maxdiff = 0` vs l'appel
   direct, dérive masse `< 1e-12`) et conservatif (`test_advance_amr`). Promotion des rôles en
   types commencée : `OwnershipPolicy` est un alias réel de `DistributionMapping` ;
   `FluxRegister` est un VRAI TYPE (registre grossier indexé global sur une région : `set`
   écrase, `add` accumule borné, `gather` fait l'`all_reduce`), substitué aux quatre buffers
   manuels du reflux (2-niveaux avg/ref, N-niveaux avg/ref) bit-identique (np=1/2/4 `maxdiff=0`),
   contrat figé par `test_flux_register`. `CoverageMask` est aussi un VRAI TYPE (masque grossier
   sur une région : `mark(box)` marque une empreinte fine clippée, `covered(I,J)` borné), la part
   « couverture » de `CoarseFineInterface` (le masque anti-double-reflux), substitué aux trois
   masques manuels, bit-identique (np=1/2/4 `maxdiff=0`), contrat figé par `test_coverage_mask`.
   Reste : promouvoir les rôles restants (`PatchRange`, le routage bordant de `CoarseFineInterface`,
   `SubcyclingSchedule`, `RegridPolicy`, encore inlines dans `subcycle_level_mp`) et y replier la
   famille `amr_step_*` (qui encode le cas dans le nom).
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

- **Push de Boris E+B combiné : FAIT.** `tfap_boris` avance la quantité de mouvement sous E
  ET B en un pas symétrique (demi-impulsion électrique, rotation magnétique complète,
  demi-impulsion), au lieu du splitting de Strang externe (rotation autour de tout le pas
  électrostatique). Câblé dans `TwoFluidAP2D::step` (seul le cas magnétisé change ; à `wc = 0`
  le push se réduit exactement à `tfap_lorentz`, donc tous les tests `B = 0` restent
  bit-identiques ; à `E = 0` c'est la rotation pure, donc le cyclotron reste exact à 0.00 %).
  `test_two_fluid_boris` fige les trois propriétés : rotation conservant `|m|` sous B seul,
  réduction à l'impulsion électrique sans B, et surtout le point fixe `E x B` discret
  `m* = h cot(theta/2) (Ey, -Ex)` préservé avec rayon de giration constant (pas de croissance
  séculaire de l'énergie). `test_two_fluid_ap_amplitude` (B activé) valide le push dans la pile
  AP self-consistante.
- Reste : reformulation AP tensorielle sous champ fort.

### Reproduction Hoffart (arXiv:2510.11808) : objectif du stage

Le papier (Euler-Poisson magnétique, structure-preserving FEM) valide en Section 5 l'instabilité
**diocotron** par ses taux de croissance, dans la **limite de dérive magnétique** (eq 2.7 :
`v_dr = -∇φ×Ω/|Ω|²`, `∂tρ + ∇·(ρ v_dr) = 0`, `ωd = ρα/|Ω| = ωp²/ωc`). Cette limite EST le modèle
`Diocotron` de adc_cpp. Aucune AMR dans le papier : l'objectif est de reproduire avec NOTRE solveur
puis d'y ajouter notre AMR, puis SAMRAI.

- **M1 (en cours) : taux de croissance numérique vs analytique.** Pipeline construit et validé.
  L'analytique (`diocotron_growth.hpp`, valeurs propres de Petri/Davidson-Felice) redonne déjà les
  taux du papier (`γ₃≈0.772, γ₄≈0.911, γ₅≈0.683`, pic au mode 4). La simu non-linéaire
  (`diocotron_column`, géométrie `0.15:0.20:0.40 = 6:8:16`) reproduit **qualitativement**
  l'instabilité mode 4 (croissance linéaire puis saturation). `scripts/validate_diocotron_growth.py`
  ajuste le taux numérique sur la phase linéaire et le normalise par `ωD = ρ̄/(2π)`. Étude de
  résolution (`docs/fig_diocotron_reproduction.png`) : `γ_norm = 0.52 → 0.54 → 0.55` à `n =
  128/192/256`, croissance monotone vers `0.911` mais **limitée par la diffusion numérique du bord
  d'anneau** (anneau fin de 6 à 13 cellules). Conclusion chiffrée : sur grille uniforme, atteindre
  `0.911` demande une très haute résolution -> motive directement l'AMR (M2).
- **M2 : avec notre AMR (mono-rang, FAIT).** `examples/diocotron_column_amr.cpp` : colonne creuse
  (anneau + paroi conductrice circulaire `r=0.40` portée par le multigrille, `AmrCouplerMP` accepte
  désormais le prédicat `active`) sur AMR 2 niveaux, raffinant le **bord d'anneau** (tag couronne
  `[0.13,0.22]`, regrid Berger-Rigoutsos), diagnostic d'amplitude du mode `l` de phi à `r0`, dérive
  masse `~3e-15`. Même binaire pour les deux branches (`refine=0/1`, numériques identiques).
  Résultat (mode 4, `docs/fig_diocotron_amr_vs_uniforme.png`) : à BASE GROSSIÈRE ÉGALE (96), l'AMR
  **triple le taux** (`γ_norm = 0.38` vs uniforme `0.12`) en raffinant le transport au bord, pour
  1.8x les cellules ; il est sur une MEILLEURE courbe taux/cellule que l'uniforme (0.38 à 16k
  cellules vs ~0.22 par interpolation uniforme).
- **M2b : Poisson MULTI-NIVEAU (FAIT, mode `ml=1`).** L'étape M2 bridait le taux parce que le
  **Poisson restait résolu sur le grossier** (le coupleur injecte l'aux grossier aux patchs). Le mode
  `ml` (`diocotron_column_amr <out> <nc> <nsteps> <refine> <l> <ml>`) assemble une densité composite
  sur la grille fine (grossier prolongé + patchs fins écrasés), résout un `GeometricMG` fin dessus,
  puis restreint le potentiel vers le grossier (gradient pour `auxc`) et garde le gradient fin direct
  (`auxf`). À base 96 (mêmes 16 392 cellules) le taux remonte de `γ_norm = 0.38` à `0.42` : le Poisson
  grossier bridait bien le taux. HYPOTHÈSE CONFIRMÉE.
- **M2b-conv : convergence vérifiée par balayage de base** (`docs/fig_diocotron_ml_convergence.png`).
  En montant la base (96 -> 128 -> 160, résolution effective au bord 192 -> 256 -> 320), l'AMR
  multi-niveau converge vers l'uniforme à MÊME résolution effective, pour ~43 % des cellules :

  | résolution effective | AMR `ml` (γ_norm / cellules) | uniforme (γ_norm / cellules) | cellules AMR/unif |
  |---|---|---|---|
  | 192 | 0.42 / 16 392 | 0.50 / 36 864 | 44 % |
  | 256 | 0.526 / 28 352 | 0.526 / 65 536 | 43 % |
  | 320 | 0.563 / 44 192 | 0.565 / 102 400 | 43 % |
  | 448 | **0.592** / 82 808 | 0.577 / 200 704 | 41 % |

  À base >= 128 le taux AMR COÏNCIDE avec l'uniforme à résolution effective égale, pour moins de la
  moitié du coût ; à eff 448 il le DÉPASSE (0.592 vs 0.577) pour 41 % des cellules. La convergence
  vers `0.911` est nette et monotone (0.42 -> 0.526 -> 0.563 -> 0.592) mais LENTE : la diffusion
  numérique du bord d'anneau (limite de M1) demande une résolution bien plus haute pour atteindre
  `0.911`. C'est la SCIENCE de l'étape 3 du hero-run : pousser la résolution (l'AMR multi-niveau y
  arrive pour ~41-44 % des cellules de l'uniforme). Atteindre `0.911` à pleine échelle demande le
  driver AMR distribué (étape 2 dé-réplication + durcissement des primitives, cf. `docs/HERO_RUN_AMR.md`).
- **M2b-conv-HR : balayage poussé au-delà de eff 448 (instabilité haute résolution corrigée).**
  Le balayage plafonnait à eff 448 parce que la simu partait en `nan` au-dessus : le multigrille
  géométrique DIVERGEAIT au bord embedded sur grille fine (correction grossière incohérente avec le
  cercle re-discrétisé par niveau, rayon spectral du V-cycle > 1, ERRATIQUE selon l'alignement du
  cercle). Le warm start propageait la divergence d'un pas à l'autre -> `phi` puis le champ en `nan`.
  Ce n'était NI le pas de temps (déjà plafonné), NI le plancher de densité (la densité reste bornée
  dans `[1e-3, 1]` pendant la divergence ; seul `phi` explose). Correctif : `GeometricMG::solve_robust`
  (cf. [HERO_RUN_AMR.md](HERO_RUN_AMR.md)), qui lance le V-cycle standard (BIT-IDENTIQUE quand il
  converge ou stagne) et, SEULEMENT en cas de vraie divergence (résidu final > résidu initial),
  durcit le lissage GS (sticky) et repart à froid jusqu'à redevenir contractant. Les 8 runs
  enregistrés ci-dessus restent BIT À BIT identiques (vérifié) ; la suite elliptique reste verte.
  Le balayage monte alors sans `nan` jusqu'à eff 1024 (uniforme ET AMR `ml`, masse `~1e-14`) :

  | eff | AMR `ml` γ (lin / sat) | uniforme γ (lin / sat) | cellules AMR / unif |
  |---|---|---|---|
  | 448  | 0.631 / 0.591 | 0.632 / 0.577 | 82 808 / 200 704 = 41 % |
  | 512  | 0.664 / 0.582 | 0.650 / 0.579 | 104 632 / 262 144 = 40 % |
  | 640  | 0.663 / 0.588 | 0.670 / 0.574 | 162 144 / 409 600 = 40 % |
  | 896  | 0.695 / 0.570 | 0.699 / 0.561 | 314 340 / 802 816 = 39 % |
  | 1024 | 0.706 / 0.565 | 0.706 / 0.558 | 409 008 / 1 048 576 = 39 % |

  Deux mesures du taux. `sat` = fenêtre relative au pic (méthode historique de la table ci-dessus).
  `lin` = fenêtre PHYSIQUE FIXE en phase linéaire (`validate_diocotron_growth.py --window 5,14`,
  nouvelle option), plus robuste pour COMPARER des résolutions. La mesure `sat` PLAFONNE vers ~0.58
  puis décline au-delà de eff 448 : ce n'est PAS la physique mais un biais de fenêtre (le rollover de
  saturation se raidit avec la résolution et contamine la pente). La mesure `lin`, qui isole le régime
  exponentiel, CONTINUE sa montée MONOTONE vers `0.911` (0.63 -> 0.65 -> 0.67 -> 0.70 -> 0.71 de eff
  448 à 1024, uniforme comme AMR ; trend robuste au choix de fenêtre, `--window 6,16` donne 0.55 ->
  0.63, même montée). L'AMR `ml` SUIT l'uniforme à ~39-40 % des cellules jusqu'à eff 1024 : la promesse
  M2b (même physique, < moitié du coût) tient à l'échelle. Le verrou numérique qui bloquait le balayage
  est levé ; atteindre `0.911` reste une affaire de résolution encore plus haute (hero-run distribué).
- **M2b-recon : le vrai verrou vers `0.911` est l'ORDRE de reconstruction, pas la résolution.** M1
  attribuait le plafond à la diffusion numérique du bord d'anneau ; le transport tournait en `NoSlope`
  (reconstruction premier ordre, la plus diffusive). En passant à MUSCL ordre 2 (`VanLeer`, pente
  limitée, option `recon=1` de `diocotron_column_amr` ; 2 ghosts) le bord reste net et le taux MONTE
  FORTEMENT à résolution FIXE :

  | eff | `NoSlope` (lin 5,14) | `VanLeer` (lin 5,14) | `VanLeer` (lin 4,11, phase expo. propre) |
  |---|---|---|---|
  | 256 | 0.561 | 0.760 | **0.864** (95 % de 0.911) |
  | 512 | 0.650 | 0.753 | 0.851 |

  À eff 256 déjà, `VanLeer` atteint `γ_norm ~ 0.86` dans la fenêtre exponentielle propre (`--window
  4,11`), soit ~95 % de l'analytique `0.911`, contre `0.56` en `NoSlope` : +0.30 de taux pour la SEULE
  montée en ordre, sans toucher la résolution. Le taux `VanLeer` est quasi PLAT en résolution (0.864 ->
  0.851 de eff 256 à 512) : il est déjà CONVERGÉ en reconstruction, ce qui CONFIRME directement
  l'hypothèse M1 (le plafond ~0.58 du `NoSlope` venait de la diffusion du schéma, pas de la physique).
  Stable (limiteur TVD, aucun `nan`) et conservatif (`~1.9e-14`), uniforme ET AMR `ml` (base 320 à 66
  patchs incluse). Le défaut `recon=0` (`NoSlope`) reste BIT À BIT identique aux runs enregistrés. Les
  ~5 % restants vers `0.911` (transitoire initial, intégration en temps Euler explicite d'ordre 1,
  représentation embedded du bord) sont la cible fine ; le gros du gain est acquis à coût modeste.
- **M3 : système magnétique complet (eq 2.4, FAIT).** Au-delà de la limite de dérive : Euler
  compressible + énergie + Poisson + force de Lorentz `m × Ω`. L'architecture était déjà prête : le
  modèle `EulerPoisson` porte l'hydro, la source `-ρ∇φ`, le travail `-m·∇φ` et le second membre
  `α(ρ-ρ0)` ; il ne manquait que la rotation cyclotron `m × Ω`.
  `integrator/magnetic_euler_poisson.hpp` : `magnetic_rotate` (rotation EXACTE de la quantité de
  mouvement, `ρ` et `E` inchangés, conserve `|m|`) + `MagneticEulerPoissonCoupler`, splitting de
  Strang autour de `Coupler<EulerPoisson>` (½ rotation, transport+électrostatique SSPRK2 avec Poisson
  par étage, ½ rotation). La rotation exacte est inconditionnellement stable : schéma
  ASYMPTOTIC-PRESERVING, le pas de temps reste gouverné par la CFL hydro et NON par la fréquence
  cyclotron `ω_c = |Ω|`. `test_magnetic_euler_poisson` (60/60) prouve : rotation `ρ`/`E` bit à bit
  conservés et `|m|` préservé ; à `Ω=0` le pas est BIT À BIT le `Coupler` nu (tout le chemin
  Euler-Poisson testé est préservé) ; le point fixe de la carte de Strang converge à l'ORDRE 2
  (ratio `4.00`) vers la dérive E×B `v = (-∂_yφ, ∂_xφ)/Ω`, donc le système complet se RÉDUIT à la
  limite de dérive (M1/M2) quand `Ω` grandit. Démo `examples/magnetic_diocotron.cpp` : bande de charge
  sous le système complet, initialisée sur la variété de dérive ; tourne stablement à grand `Ω`
  (CFL hydro), masse conservée à l'arrondi (`~4e-11`), énergie du gaz quasi conservée sur la dérive
  (`docs/anim_magnetic_diocotron.gif`). La reproduction quantitative du taux à `0.911` dans le système
  complet est limitée par la diffusion (même constat que M1) et vise le hero-run.
- **M4 : SAMRAI.** Porter le diocotron sur l'AMR de SAMRAI (FetchContent + adaptateur).

Hero run ROMEO (`romeo/`) : scripts SLURM prets pour le diocotron a grande echelle sur GH200,
hybride **MPI + Kokkos/CUDA** (1 rang MPI par H100, les noyaux `for_each_cell` sur GPU ; les
232 H100 multi-noeud par Infiniband). C'est le « full machine » reel sur `armgpu` (OpenMP+MPI
est le mode CPU separe sur `x64cpu`). **Grille UNIFORME, pas d'AMR** : le binaire `diocotron_mpi`
utilise `SpectralCoupler` (Poisson FFT, bandes), donc force brute (pousser la resolution uniforme
jusqu'a resoudre le bord d'anneau et atteindre 0.911). Un hero-run AMR DYNAMIQUE est un objectif
distinct (convergence du reflux distribue fait + coupleur AMR porte multi-GPU + benchmark colonne,
cf. M2) ; le run uniforme sert de reference chiffree pour mesurer le gain de l'AMR.
`diocotron_hero.sbatch` (run) + `diocotron_scaling.sbatch` (scaling fort/faible) +
`romeo/README.md` (build + soumission).

Hero run AMR DYNAMIQUE (le vrai but de comparaison uniforme vs AMR sur ROMEO) : c'est la
convergence de plusieurs briques et le plus gros morceau restant. A l'echelle hero, le grossier
8192^2 NE PEUT PAS etre replique sur chaque rang -> il faut un **grossier decompose** (multi-box
distribue) + **Poisson MG distribue** dessus (`GeometricMG` distribue deja un grossier multi-box
via `DistributionMapping(ba.size(), n_ranks())`, mais le diocotron a un grossier MONO-box -> a
decomposer) + **reflux sans replication** (notre reflux distribue suppose le niveau 0 replique ;
les niveaux >0 sont deja distribues via `parallel_copy` + gather, a etendre au niveau 0) +
**regrid distribue** (clustering BR + redistribution des patchs) + le tout en **Kokkos/CUDA**.
Autrement dit, la **de-replication du grossier (objectif B)**, declaree NO-GO pour un PETIT
grossier, redevient REQUISE a l'echelle hero. Chemin propre : M2 d'abord (AMR sur la colonne,
mono-rang, science) pour chiffrer le gain cellules, puis assembler le driver distribue (B +
MG/regrid distribues + GPU), puis comparer les deux hero runs. Conception-d'abord recommandee
(gate conservation `maxdiff=0` np=1/2/4 a chaque etape).

Le PLAN etage de ce hero-run AMR distribue est ecrit dans
[HERO_RUN_AMR.md](HERO_RUN_AMR.md) : l'insight clef est qu'un diocotron a 2 NIVEAUX
(grossier replique + 1 niveau fin reparti) ne touche AUCUN des deux verrous durs (le regrid
tague le niveau 0 REPLIQUE, donc pas de gather-tags ; pas de de-replication tant que le grossier
reste modere), ce qui permet de livrer les etapes 0 (driver CPU) et 1 (portage GPU) en
s'appuyant sur le reflux distribue deja prouve, et de repousser le risque dur (solveur de fond
reparti + gather-tags, etape 2) jusqu'a ce qu'il devienne necessaire.

### Performance

- Région OpenMP consolidée au-dessus de la boucle de niveaux du multigrille (le seul levier
  identifié pour les charges MG-dominées).

### Confort

- MUSCL sur le momentum du deux-fluides ; GIF du coupleur multi-patch ; tutoriels
  supplémentaires si besoin.
