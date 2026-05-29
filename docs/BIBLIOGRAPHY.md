# Bibliographie

Les références qui ont informé la conception et l'implémentation d'`adc_cpp` : codes AMR /
plasma existants consultés, manuels, articles clés. Aucun n'a été copié ; chacun a apporté
une idée.

## 1. Codes AMR / plasma consultés

### AMReX

Framework AMR block-structured de référence (LBNL), C++17 + GPU. La pile `adc_cpp` en est
un mini-clone écrit *from scratch* : les correspondances directes sont `MultiFab`,
`BoxArray` / `DistributionMapping`, `Geometry`, `FillBoundary`, le FluxRegister (reflux),
le MLMG (~ `GeometricMG`). Divergences assumées : pas de `MFIter` (on itère `for_each_cell`
+ fab local, GPU-ready), Laplacien à coefficient constant (EB en escalier), AMR mono-rang
pour le multi-patch.
[Repo](https://github.com/AMReX-Codes/amrex), Zhang et al. 2019, *AMReX*, JOSS 4(37).

### WarpX

Code PIC-AMR électromagnétique (sur AMReX) pour la physique des plasmas et des
accélérateurs. Contexte du couplage hyperbolique-elliptique sur AMR pour les plasmas non
neutres (diocotron) et le modèle de fluide.
[Repo](https://github.com/ECP-WarpX/WarpX).

### Athena++ / PLUTO

Frameworks hydro/MHD astrophysiques. Le design à **axes orthogonaux** de PLUTO (équation x
reconstruction x Riemann x intégrateur) a inspiré le découpage concept-templé d'`adc_cpp`
(`PhysicalModel` / `NumericalFlux` / `EllipticSolver` / couplage).
[Athena++](https://github.com/PrincetonUniversity/athena),
[PLUTO](http://plutocode.ph.unito.it).

## 2. Manuels

- **Birdsall & Langdon**, *Plasma Physics via Computer Simulation*, 1985. Dérive E x B,
  fréquences plasma et cyclotron, instabilité diocotron.
- **Toro**, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3e éd., 2009.
  Solveurs de Riemann (Rusanov, HLL, HLLC), reconstruction MUSCL, forme conservative.
- **Trottenberg, Oosterlee & Schüller**, *Multigrid*, 2001. V-cycle, lisseur Gauss-Seidel
  rouge-noir, restriction / prolongation.

## 3. Articles clés

- **Berger & Oliger**, 1984, *Adaptive mesh refinement for hyperbolic partial differential
  equations*, JCP 53. Sous-cyclage en temps des niveaux fins.
- **Berger & Colella**, 1989, *Local adaptive mesh refinement for shock hydrodynamics*,
  JCP 82. Reflux (FluxRegister) à l'interface fin-grossier, conservation.
- **Berger & Rigoutsos**, 1991, *An algorithm for point clustering and grid generation*,
  IEEE Trans. SMC 21. Clustering par signature pour le regrid.
- **Hoffart**, 2025, arXiv:2510.11808. Modèle deux-fluides isotherme, cible de validation
  du schéma asymptotic-preserving (`two_fluid_ap.md`).

## 4. Méthodologie performance

- **Bryant & O'Hallaron**, *Computer Systems: A Programmer's Perspective*, 3e éd., 2016.
  Profiler d'abord, identifier le goulot, transformer, re-mesurer (voir `PERFORMANCE.md`).
