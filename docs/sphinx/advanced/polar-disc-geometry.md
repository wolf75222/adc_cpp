# Polar and disc geometry (implementation)


By default the domain is square Cartesian. Two distinct mechanisms carry a
non-Cartesian geometry.

## Global polar ring

The geometry choice lives in a mesh object passed in `mesh=` (never in
`FiniteVolume`, which only carries reconstruction + flux + variables). `adc.PolarMesh`
describes a global ring `r in [r_min, r_max] x theta in [0, 2pi)`, `nr x ntheta` cells,
periodic theta, physical walls at `r_min` / `r_max`. The polar grid lifts the structural
lock of the diocotron on a Cartesian grid (the Phase-0 proto measured a ratio of 73
on the diffusion of the radial gradient).

```python
import adc

mesh = adc.PolarMesh(r_min=0.3, r_max=1.0, nr=128, ntheta=256)  # axe 0 = radial, axe 1 = azimutal
sim = adc.System(mesh=mesh)   # construit un anneau global et avance dessus
```

The `SystemConfig` then carries `geometry="polar"`, `nr`, `ntheta`, `r_min`, `r_max`. On the
C++ side, the polar path is wired into `System.step` (transport `assemble_rhs_polar` +
Poisson `PolarPoissonSolver` + drift of the aux in the local basis `(e_r, e_theta)`).

> **Polar scope.** The polar path is limited: scalar / isothermal ExB transport
> only (the full fluid fluxes are not ported), single-rank (the
> direct `PolarPoissonSolver`, FFT-in-theta + tridiagonal-in-r by Thomas, refuses MPI
> and raises on `n_ranks() > 1` or `ba.size() != 1`), no Cartesian <-> polar coupling
> (it is a global ring). `nr >= 3` (2nd-order one-sided radial stencil at the walls),
> `ntheta >= 1`. `adc.CondensedSchur` is wired in polar (the polar condensed stepper is
> chosen on the C++ side according to the geometry).

## Disc mask (Cartesian transport)

On a Cartesian grid, one can restrict the transport to a disc
`hypot(x-cx, y-cy) - R < 0`:

```python
sim.set_disc_domain(cx=0.5, cy=0.5, R=0.40)   # cellule active si son centre est dans le disque
mask = sim.disc_mask()                         # masque 0/1 (ny, nx) row-major, pour verification
```

Without `set_disc_domain`, the mask is fully active and the transport path stays
bit-identical. The disc mask is refused in polar geometry (the ring is already bounded
by its radial walls `r_min` / `r_max`).

### Disc helper vs the generic level-set contract

`set_disc_domain` / `disc_mask` are the stable Python *compatibility* helpers for the circle
(the Hoffart disc). Under them the geometry is a single, named, device-clean C++ contract in
`include/adc/numerics/spatial/embedded_boundary/domain.hpp`: any POD type exposing
`level_set(x, y)` (`< 0` inside), a callable `operator()`, and `cell_active` is an embedded
boundary, with the same three transport modes (`none` / `staircase` / `cutcell`). The disc
(`detail::DiscDomain`) is one instance; `detail::HalfPlaneDomain` is a built-in non-disc instance,
and the cut-cell / mask operators (`assemble_rhs_eb`, `assemble_rhs_masked`, `cut_fraction`) are
templated on the contract, so a non-disc level-set works on the C++ side without naming a shape. The
contract stays 2D (it masks cells of the fixed `(i, j)` grid; cf. ADR-0001). There is no runtime
callback path: a domain is a compile-time POD, never a `std::function`.

## Going further

- Bindings: `python/bindings.cpp` (`geometry` / `nr` / `ntheta` / `r_min` / `r_max`,
  `set_disc_domain`, `disc_mask`).
- Generic contract: `include/adc/numerics/spatial/embedded_boundary/domain.hpp` (`DiscDomain`, `HalfPlaneDomain`,
  the `LevelSetDomain` concept).
- Polar solver: `include/adc/numerics/elliptic/polar/polar_poisson_solver.hpp`.
