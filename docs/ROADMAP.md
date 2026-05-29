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

- Diocotron (dérive E x B), Euler-Poisson auto-gravitant, deux-fluides isotherme
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

### AMR distribué (le dernier morceau de la refonte)

- MPI du reflux multi-patch : disponibilité du grossier pour le ghost-fill (copie
  inter-niveaux) + gather des registres via `all_reduce_sum_inplace` (primitive déjà posée).
- `load_balance` SFC branché partout sur le multi-box.

### Physique magnétisée (cible Hoffart)

- Push de Boris E+B combiné (au lieu du splitting de Strang externe), reformulation AP
  tensorielle sous champ fort, reproduction d'un benchmark Hoffart précis.

### Performance

- Région OpenMP consolidée au-dessus de la boucle de niveaux du multigrille (le seul levier
  identifié pour les charges MG-dominées).

### Confort

- MUSCL sur le momentum du deux-fluides ; GIF du coupleur multi-patch ; tutoriels
  supplémentaires si besoin.
