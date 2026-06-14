# Bibliography

The references that informed the design and implementation of `adc_cpp` (AMR /
plasma codes consulted, manuals, key articles). None was copied; each contributed an idea. The
full, annotated list is kept in [BIBLIOGRAPHY.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/BIBLIOGRAPHY.md); below is a
reader-oriented summary.

## AMR / plasma codes consulted

- **AMReX** (Zhang et al. 2019, JOSS 4(37)) -- reference block-structured AMR framework whose
  mesh stack `adc_cpp` reimplements as a mini-clone written *from scratch* (MultiFab, BoxArray, Geometry,
  FillBoundary, FluxRegister, MLMG ~ GeometricMG).
- **WarpX** -- electromagnetic PIC-AMR code (on AMReX), the context for hyperbolic-elliptic
  coupling on AMR for non-neutral plasmas.
- **Athena++ / PLUTO** -- hydro/MHD frameworks; PLUTO's orthogonal-axes design
  (equation x reconstruction x Riemann x integrator) inspired the concept-template
  split of `adc_cpp`.

## Manuals

- **Birdsall & Langdon**, *Plasma Physics via Computer Simulation*, 1985 -- E x B drift,
  diocotron instability.
- **Chen**, *Introduction to Plasma Physics and Controlled Fusion*, 3rd ed., 2016 -- the
  repulsive side of Euler-Poisson (plasma).
- **Binney & Tremaine**, *Galactic Dynamics*, 2nd ed., 2008 -- Jeans instability, the
  attractive side of Euler-Poisson (gravity).
- **Toro**, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed., 2009 --
  Riemann solvers (Rusanov, HLL, HLLC), MUSCL reconstruction.
- **Trottenberg, Oosterlee & Schuller**, *Multigrid*, 2001 -- V-cycle, red-black
  Gauss-Seidel smoother.

## Key articles

- **Berger & Oliger**, 1984, JCP 53 -- time subcycling of fine levels.
- **Berger & Colella**, 1989, JCP 82 -- reflux (FluxRegister), conservation at the
  coarse-fine interface.
- **Berger & Rigoutsos**, 1991, IEEE Trans. SMC 21 -- signature clustering for the regrid.
- **Hoffart**, 2025, **arXiv:2510.11808** -- isothermal two-fluid model and magnetized
  Euler-Poisson, validation target for the asymptotic-preserving scheme (application scenario,
  `adc_cases/two_fluid_ap/`). See the reproduction limit of the full model in
  [limitations](known-limitations.md).

## Performance methodology

- **Bryant & O'Hallaron**, *Computer Systems: A Programmer's Perspective*, 3rd ed., 2016 --
  profile first, identify the bottleneck, transform, re-measure (see
  [PROFILE_RESULTS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/PROFILE_RESULTS.md)).
