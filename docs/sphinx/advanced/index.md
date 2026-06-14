# Advanced topics

This page gathers the features that go beyond the basic diocotron loop
(E x B transport + Poisson): generalized elliptic solvers, coupled inter-species
sources, Schur-condensed source stage, polar / disc geometry, extending the
core in C++, and performance profiling.

Each section summarizes the essentials for a user and points to the contributor
reference (`docs/*.md`) for the algorithmic detail and the validation proofs.
The APIs shown here are checked against the repository code (bindings, tests, headers).

## Poisson: elliptic solvers

The elliptic stage solves `lap(phi) = f` (or a generalization) at each step, and it is
the core of the coupling: `f` depends on the density, and `phi` (via `grad phi`) drives the
drift. The solver is chosen by keyword in `set_poisson`:

```python
import adc

sim = adc.System(n=128, L=1.0, periodic=True)
# ... add_block / add_equation ...
sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="auto")
```

The `solver=` accepts `"geometric_mg"` (default) or `"fft"`. The `rhs=` is
`"charge_density"` (right-hand side `q n`) or `"composite"` (sum of block
contributions). `bc=` is `"auto"`, `"periodic"`, `"dirichlet"`.

### GeometricMG (geometric multigrid)

`GeometricMG` is the default solver: a multigrid V-cycle with a red-black
Gauss-Seidel smoother (the coloring makes each sweep independent of the data,
so parallelizable and device-clean). The cycle smooths `nu1` times, restricts the residual onto
a grid twice as coarse (`average_down`), recurses, prolongs the correction,
smooths again `nu2` times. Cost O(N), convergence nearly independent of the mesh. The coarsening
stops cleanly as soon as a box no longer divides exactly (safeguard
`coarsen(2).refine(2) == b`), which avoids degenerate hierarchies under AMR / multi-box.

The same multigrid operator covers three generalizations of the Laplacian, all opt-in
and bit-identical to the historical path as long as they are not activated. On the C++ side
(`GeometricMG`, `numerics/elliptic/geometric_mg.hpp`):

- `set_epsilon(eps)`, variable permittivity `div(eps(x) grad phi) = f` (harmonic
  mean at faces);
- `set_reaction(kappa)`, screened / Helmholtz operator `div(eps grad phi) - kappa phi = f`
  (Debye screening, `kappa = 1 / lambda_D^2`);
- `set_epsilon_anisotropic(eps_x, eps_y)`, diagonal tensor medium `div(diag(eps_x, eps_y) grad phi) = f`.

These three settings are composable; `eps_x == eps_y` gives back the isotropic case, not calling
`set_reaction` gives back pure Poisson. On the Python side, they are exposed by NumPy field
at the `System` level (the coefficients live in the device `for_each_cell` of the smoother):

```python
import numpy as np

eps   = np.ones((n, n))            # permittivite variable (set_epsilon C++)
kappa = np.zeros((n, n))           # terme de reaction kappa >= 0 (Helmholtz/ecrante)

sim.set_epsilon_field(eps)                       # div(eps grad phi) = f
sim.set_reaction_field(kappa)                    # - kappa phi
sim.set_epsilon_anisotropic_field(eps_x, eps_y)  # diag(eps_x, eps_y)
```

### Spectral Poisson (FFT)

On a periodic domain, the Laplacian is diagonal in Fourier:
`phi_hat(k) = -rhs_hat(k) / (k_x^2 + k_y^2)`, mode `k=0` pinned to 0 (gauge). A direct
FFT + division + inverse FFT solves Poisson exactly (machine residual), without
iteration. Two variants exist, both models of the `EllipticSolver` concept:

- `PoissonFFTSolver` (`numerics/elliptic/poisson_fft_solver.hpp`), single-rank, single
  box. Its constructor raises a `std::runtime_error` as soon as `n_ranks() != 1` or
  `ba.size() != 1`. This safeguard is deliberate and active in Release (`NDEBUG` does not remove
  it): this direct solver would dereference `fab(0)` on a rank without a box (segfault). In
  serial, it is the exact and fastest solver for a periodic domain.
- `DistributedFFTSolver` (same header), FFT distributed by slabs: 1 slab
  per rank, parallel transpose by `MPI_Alltoall`. It is the MPI counterpart of the direct FFT,
  usable as `Coupler<Model, DistributedFFTSolver>`. Constraints: `Ny` divisible by
  `n_ranks()`, `Nx`/`Ny` powers of 2 (a fix handles `n` not a power of 2 on the
  single-rank side). In serial (`n_ranks() == 1`) a single slab covers the domain, identical to
  `PoissonFFTSolver`.

MG and FFT provably invert the same discrete Laplacian: the same canonical operator
`poisson_residual` applied to their two solutions gives residuals at round-off
(`~1e-14`) and solutions identical to `~1e-16`. The FFT pitfall: it requires the
periodic case, and the right-hand side must be zero-mean (otherwise `phi` drifts).

### Going further

- Elliptic algorithms (multigrid, FFT, eps/Helmholtz/anisotropic, cut-cell):
  [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md), sections 9 to 12.
- The headers: `include/adc/numerics/elliptic/geometric_mg.hpp`,
  `poisson_fft_solver.hpp`, `poisson_operator.hpp`.
- Conservation properties of the coupled scheme (exact FV mass, momentum, energy, values
  measured by the tests): [CONSERVATION_SUMMARY.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/CONSERVATION_SUMMARY.md).

## Coupled inter-species sources

Beyond transport and a block's local source, one can describe an inter-species
coupling (ionization, collisions, thermal exchange) in formulas, without writing any
C++ and without a per-cell Python callback. The DSL `adc.dsl.CoupledSource` carries the
formula as stack-machine bytecode, interpreted on the C++ side in a device `for_each_cell`
(so MPI-safe and GPU-clean). The stage is applied by explicit splitting, after the
transport.

The canonical example is a three-species ionization
(`d_t n_e = +k n_e n_g`, `d_t n_i = +k n_e n_g`, `d_t n_g = -k n_e n_g`):

```python
import adc
from adc import dsl

src = dsl.CoupledSource("ionization")
ne = src.block("electrons").role("density")
ni = src.block("ions").role("density")
ng = src.block("neutrals").role("density")
kp = src.param("Kiz", 0.7)
src.add("electrons", role="density", expr=+kp * ne * ng)
src.add("ions",      role="density", expr=+kp * ne * ng)
src.add("neutrals",  role="density", expr=-kp * ne * ng)
compiled = src.compile(backend="production")

sim.add_coupling(compiled)   # branche l'etage sur System.add_coupled_source
```

`sim.add_coupling(...)` also accepts the named couplings `adc.Ionization` /
`adc.Collision` / `adc.ThermalExchange` (fixed formula). Without a call to `add_coupling`, the
`System` stays bit-identical (the stage is inert by default).

The compilation produces a flat ABI (`in_blocks`, `in_roles`, `consts`, `out_blocks`,
`out_roles`, `prog_ops`, `prog_args`, `prog_lens`): bytecode, never a Python
callback. The test checks that the trajectory follows bit-for-bit a NumPy forward-Euler
reference of the same ODE, and that the expected invariants hold (`n_i + n_g`
conserved, `n_e - n_i` constant: each ionization creates an e/i pair).

### Going further

- Public / internal / deprecated classification of the coupling classes (including the concept
  `CoupledSourceFor` and the bytecode evaluator `CoupledSourceProgram`):
  [COUPLING_SURFACE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/COUPLING_SURFACE.md).
- Reference test: `python/tests/test_dsl_coupled_source.py` (and the conservation
  variant `test_dsl_coupled_source_conservation.py`).

## Schur: condensed source stage

The integrator `adc.CondensedSchur` reproduces the splitting of Hoffart et al.
(arXiv:2510.11808) for the stiff potential / velocity / Lorentz-force source of the
magnetized Euler-Poisson system. The key: Schur condensation algebraically eliminates the
velocity from the implicit subsystem, which reduces the source stage to an elliptic solve
(tensor operator `-div(A grad phi)` with `A = rho B^{-1}`, in general non-symmetric)
followed by an explicit reconstruction of the velocity.

It is composed with an explicit transport stage via `adc.Split`:

```python
import adc

time_policy = adc.Split(
    hyperbolic=adc.Explicit(),
    source=adc.CondensedSchur(
        kind="electrostatic_lorentz",   # seul kind supporte
        theta=1.0,                      # theta-schema : 0.5 = Crank-Nicolson, 1 = Euler retrograde
        alpha=3.0,                      # constante de couplage
    ),
)

sim.add_equation(
    "ions",
    model=model,                        # roles requis : Density / MomentumX / MomentumY (Energy optionnel)
    spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"),
    time=time_policy,
)
```

The model must expose the roles `Density`, `MomentumX`, `MomentumY` (a native isothermal
fluid `adc.FluidState(kind="isothermal") + adc.IsothermalFlux()` provides them). The stage is
entirely C++ (`CondensedSchurSourceStepper`, exposed as `adc.CondensedSchur`): no
per-cell Python callback.

> **Roles hardcoded on the C++ side.** The role / field descriptors are not configurable
> from Python. `adc.CondensedSchur(...)` accepts `kind`, `theta`, `alpha`, but passing
> `density=`, `momentum=`, `energy=`, `magnetic_field=` or `potential=` raises an error:
> the C++ source stage fixes these roles hard. This is intentional (the contract of
> `CondensedSchurSourceStepper` is frozen).

`adc.Strang` is the 2nd-order extension of `adc.Split` (transport / source /
transport sequence). The default is unchanged: a block in pure `adc.Explicit` never sees the
condensed source stage.

> **CondensedSchur (global) vs SourceImplicit (local).** Do not confuse them. `adc.CondensedSchur`
> assembles and solves an elliptic operator coupling the whole domain (for a stiff non-local
> coupling: Lorentz / electrostatic). `adc.SourceImplicit` (= IMEX source-only) is
> local: the implicit only couples the components of the same cell (relaxation, reactions,
> friction), without an elliptic solve, so much cheaper. A local stiff source does not
> need Schur.

### Going further

- Detailed design (the five levels, the non-symmetry of the tensor operator, the
  question of the Krylov solver): [SCHUR_CONDENSATION_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/SCHUR_CONDENSATION_DESIGN.md)
  (banner: `implemente`; the document is the original spec, read as a design
  history).
- Conservation properties of the Cartesian Schur path (measured values):
  [CONSERVATION_SUMMARY.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/CONSERVATION_SUMMARY.md).
- Tests: `python/tests/test_schur_via_system.py` (path `System -> run_source_stage`,
  native bricks, CI-safe), `test_schur_conservation.py`.

## Polar / disc geometry

By default the domain is square Cartesian. Two distinct mechanisms carry a
non-Cartesian geometry.

### Global polar ring

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

### Disc mask (Cartesian transport)

On a Cartesian grid, one can restrict the transport to a disc
`hypot(x-cx, y-cy) - R < 0`:

```python
sim.set_disc_domain(cx=0.5, cy=0.5, R=0.40)   # cellule active si son centre est dans le disque
mask = sim.disc_mask()                         # masque 0/1 (ny, nx) row-major, pour verification
```

Without `set_disc_domain`, the mask is fully active and the transport path stays
bit-identical. The disc mask is refused in polar geometry (the ring is already bounded
by its radial walls `r_min` / `r_max`).

### Going further

- Bindings: `python/bindings.cpp` (`geometry` / `nr` / `ntheta` / `r_min` / `r_max`,
  `set_disc_domain`, `disc_mask`).
- Polar solver: `include/adc/numerics/elliptic/polar_poisson_solver.hpp`.

## C++ extension: adding a native brick

The core is model-agnostic: it names no scenario. A physical model is a
composition of generic bricks (state, transport, source, elliptic), and the per-cell
computation stays compiled C++.

To write a new native brick, one satisfies the `PhysicalModel` concept
(`include/adc/core/physical_model.hpp`). The minimal contract:

```cpp
template <class M>
concept PhysicalModel =
    requires(const M m, const typename M::State u, const typename M::Aux a, int dir) {
      typename M::State;                                   // type d'etat conservatif
      typename M::Aux;                                     // == adc::Aux
      { M::n_vars } -> std::convertible_to<int>;           // nombre de composantes
      { m.flux(u, a, dir) } -> std::same_as<typename M::State>;       // flux directionnel
      { m.max_wave_speed(u, a, dir) } -> std::convertible_to<Real>;   // CFL
      { m.source(u, a) } -> std::same_as<typename M::State>;          // source LOCALE
      { m.elliptic_rhs(u) } -> std::convertible_to<Real>;             // second membre Poisson
    };
```

All these methods must be `ADC_HD` (host/device) if they are called in
kernels. The optional extension `HasPrimitiveVars` adds `to_primitive` / `to_conservative`
(reconstruction in primitive variables, more stable for Euler: positivity of rho and p),
and `HyperbolicPhysicalModel` adds the variable descriptor (`conservative_vars()` /
`primitive_vars()`). Once the brick is compliant, it composes into a `CompositeModel`
and is exposed at runtime like the existing bricks.

### Going further

- The five orthogonal layers, the map of the modules, the elliptic stage
  (problem / operator / solver / post-processing): [ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md).
- The design choices (concepts + policies, `for_each_cell` seam, `EllipticSolver`):
  [CHOICES.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/CHOICES.md).
- The concept and its extensions: `include/adc/core/physical_model.hpp`;
  the reference composition: `include/adc/physics/composite.hpp`.

## Performance and profiling

The repository ships a profiling harness: `bench/profile_step.cpp`, driven by
`bench/run_bench.sh`. It rebuilds a time step representative of the diocotron from
the public seams (without touching the hot path) and times each phase
(`transport`, `poisson`, `halos`, `aux_derive`, `reduction`, `fence`, `alloc_tmp`)
bracketed by `device_fence()` to capture the device execution under Kokkos.

The harness is out of the default build (`ADC_BUILD_BENCH=OFF`): the CI never
configures or compiles it. It is enabled explicitly:

```sh
bench/run_bench.sh serie                  # Serie CPU
bench/run_bench.sh kokkos-omp  <Kroot>    # Kokkos OpenMP
bench/run_bench.sh mpi          2         # MPI CPU, 2 rangs
bench/run_bench.sh mpi-cuda    <Kroot> 4  # MPI + Kokkos Cuda (GH200), 4 rangs
```

It accepts `--n --steps --warmup --cfl --solver {geometric_mg|fft} --limiter
{none|minmod|vanleer|weno5} --bc {periodic|dirichlet}`.

Main finding of the profile: on the six measured backends, the `poisson` phase dominates
the step at 96 to 99.9 %. The transport, the halos, the reductions and the temporary
allocations are each `< 1 ms` per step (together `< 1.1 %` of the step in serial). The
performance lock is, unambiguously, the elliptic solve (`GeometricMG::solve()`, called at
each step). Two measured aggravating facts: Poisson does not benefit from the on-node
parallelism (it regresses, the V-cycle descends down to tiny grids 2x2, 4x4, and the
launch cost of each kernel crushes the useful computation), and on GPU the launch latency
dominates (neither a larger GPU nor more GPUs help, all the more so as the `System`
layout is single-box).

No performance refactor without a profile showing the bottleneck.
> `PROFILE_RESULTS.md` reports the measurements and points to a target (the V-cycle dispatch on
> the coarse levels); it applies no optimization.

### Going further

- Full profile (phase x backend table, exact platforms M-series + GH200, leads to
  investigate): [PROFILE_RESULTS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/PROFILE_RESULTS.md).
- `docs/PERFORMANCE.md` exists but carries historical measurements (old application
  drivers, M1): do not read it as the current performance.
