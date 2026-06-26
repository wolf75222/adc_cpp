# Known limitations

An honest and consolidated list of `adc_cpp`'s current limits, to avoid any wrong
expectations. Each item points to the corresponding source of truth. These are not bugs:
they are the edges of the scope validated to date.

## Spatial dimension: the core is 2D only

The solver core is structurally two-dimensional, and this is a load-bearing invariant rather than
a naming detail. The 2D assumption is baked into the data layout (`Fab2D operator()(i, j, c)`), the
paired hand-written `FaceFluxX` / `FaceFluxY` kernels, the two-component momentum (`euler.hpp`
`{rho, rho_u, rho_v, E}`), the five-point Poisson stencil, and the `Box2D` / `Geometry` index space.
The polar mesh is a second geometry at the same dimension (the ring `(r, theta)` is a two-index
`Box2D`), not a third axis; it does not make the core ND.

This is the agreed short-term contract (ADR-0001, Decision 1): 2D is declared an explicit,
introspectable invariant, not a step already taken toward an ND core. A real ND trajectory
(`IndexSpace<Dim>` / `BoxND<Dim>` with 2D adapters and 2D bit-identity gates) is a milestone-sized
change with genuine ABI and bit-identity risk, and no current scenario demands it; it is deferred
to a dedicated future milestone.

Do not confuse axis count with domain shape: this limit is about the number of axes (2D vs ND),
whereas embedded-boundary / level-set domains (ADC-327) change the shape of the domain inside the
fixed 2D plane and add no third index.

Source of truth: `pops.capabilities()['dimension']` (`== 2`), the decision record
[ADR-0001](https://github.com/wolf75222/adc_cpp/blob/master/docs/adr/ADR-0001-genericity-contracts.md),
and the `include/pops/mesh/index/box2d.hpp` header comment.

## GPU: validated manually on ROMEO, not in CI

CI never builds `-DPOPS_USE_KOKKOS=ON -DKokkos_ENABLE_CUDA=ON`. All GPU validation
(Kokkos Cuda single-GPU, and MPI + Kokkos Cuda multi-GPU) is done by hand on the
ROMEO supercomputer (GH200 node, `Kokkos_ARCH_HOPPER90`, `nvcc_wrapper`, OpenMPI
CUDA-aware), via SBATCH harnesses in `python/tests/gpu/` (excluded from the CI glob). A
"GPU-validated" mention therefore means "manually on ROMEO GH200", with the quantified evidence cited in
[GPU_RUNTIME_PORT.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/GPU_RUNTIME_PORT.md) and [BACKEND_COVERAGE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md).
Several GPU cells of the matrix remain `?` (not yet exercised on device); see the
"Notable gaps" section of the source document.

## DSL: parity asserted only if a matching C++ compiler is present

The symbolic DSL (`pops.dsl`) compiles a model to a `.so` at runtime (backends `aot` /
`production`) by invoking the C++ compiler against `adc_cpp`'s headers. The parity
verification (DSL vs native brick) therefore relies on the presence of a working C++ compiler
that supports the module standard (with `std=None` the loader derives it: C++20 under Kokkos,
since nvcc CUDA 12.x has no `-std=c++23`; see the [DSL reference](symbolic-dsl.md)). Without a
working C++ toolchain, compilation of DSL models fails; only the purely native paths
(`pops.Model(...)` / `add_block`) remain available.

## DSL backends: prototype/aot are CPU-only

`m.compile(..., backend=...)` (default `auto`: selects `production` under toolchain parity, else `aot`):

- `prototype` (JIT) and `aot` (host-marshaled) are CPU-only: no MPI, no AMR, no
  GPU, single-rank. Their flat `.so` ABI carries neither the `stride` (multirate cadence) nor
  `evolve=False` nor the partial IMEX mask; these options are rejected explicitly (explicit
  route rather than silent ignore).
- `production` is the only one pluggable on AMR (`AmrSystem.add_equation` requires
  `backend='production'`, `target='amr_system'`) and the only one that goes through the native
  zero-copy path; even there, from Python, the `stride` and the partial IMEX mask are rejected on the
  `.so` path (the loader ABI does not carry them). Detail in
  [DSL_MODEL_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/DSL_MODEL_DESIGN.md).

## Poisson FFT under MPI: periodic, constant coefficient, Ny divisible by n_ranks

The direct `PoissonFFTSolver` is single-rank by design (it would dereference `fab(0)` on a rank
without a box). Under MPI (`n_ranks > 1`), `System.set_poisson(..., "fft"|"fft_spectral")` instead
selects `RemappedFFTSolver` (ADC-287), which presents the System single box outward and runs a
box-slab scatter/gather around the distributed FFT. Constraints: periodic BCs, constant coefficient
(no wall, no variable/anisotropic eps, no kappa), and `Ny` divisible by `n_ranks()`; those cases
still raise and point to `geometric_mg`, which remains the MPI default. The potential matches
`geometric_mg` to FP tolerance.

## AMR: global Schur on a single block only

The Schur-condensed source stage (`pops.CondensedSchur` via `pops.Split` / `pops.Strang`) is
available on AMR through `AmrSystem.add_equation`. The condensed stage is assembled and solved
on the coarse level and requires a single-block hierarchy; it raises on a refined multi-block
one. `AmrSystem.add_block` rejects `pops.Split` / `pops.Strang` (use `add_equation`). On `System`
the stage is available on any cartesian or polar grid. Design:
[SCHUR_CONDENSATION_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/SCHUR_CONDENSATION_DESIGN.md),
[AMR_MULTIBLOCK_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/AMR_MULTIBLOCK_DESIGN.md).

## Polar geometry: direct Poisson single-box, no HLLC/Roe, no cartesian coupling

The polar mesh (`pops.PolarMesh`, global ring (r, theta)) is wired into `System.step`
(polar transport + polar Poisson + aux in local e_r/e_theta basis), but with sharp
edges:

- transport is scalar ExB or the isothermal fluid (`IsothermalFluxPolar`): `rusanov` on any
  polar model, `hll` on the isothermal fluid (it declares `wave_speeds`), with limiters
  minmod / vanleer / weno5. `hllc` / `roe` are not lifted on the polar side (no polar energy
  flux brick);
- the DIRECT polar Poisson solver (one box covering the ring) is single-box / single-rank: it
  refuses MPI (`n_ranks > 1` raised) and `theta_boxes > 1`. The polar transport and the
  tensorial polar Schur stage (`PolarCondensedSchurSourceStepper`), by contrast, are multi-box
  / multi-rank by azimuthal (theta) split (ADC-67);
- no cartesian <-> polar coupling: the polar ring is a separate global domain.

Source of truth: `pops.capabilities()['riemann']['system_polar']`,
`['geometry']['system_polar']` and `['schur']['system_polar']` (revalidated 2026-06).

## Two-fluid AP: scenario in adc_cases, not a core brick

The two-fluid isothermal asymptotic-preserving integrator is not a composable brick of the
core: it is a scenario. Its AP stabilization couples the stiffness to the time step in
the elliptic, which `System` composition does not reproduce. Its C++ physics lives in
`adc_cases/two_fluid_ap/` (`two_fluid_ap.hpp`), compiled on the fly against the generic
headers of `adc_cpp`; the `_pops` module does not expose it and `pops` does not re-export it.

## Hoffart full model: reproduction not established

The quantitative reproduction of Hoffart et al.'s full magnetized Euler-Poisson model
(arXiv:2510.11808) is not established to date. The reduced ExB-scalar (polar) path reaches
the diocotron's target growth rate, but the cartesian full model (`run.py`
`system-schur`) does not reproduce the paper's growth: its short runs crush the rate. The
identified lock is the geometry (the cartesian square + circular Poisson wall diffuses the
ring edge); the chosen lead is a conservative disk domain (cut-cell), not a square.
Detailed state and per-aspect statuses in
[HOFFART_FIDELITY.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/HOFFART_FIDELITY.md) (the roadmap
[FULL_MODEL_VALIDATION_ROADMAP.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/FULL_MODEL_VALIDATION_ROADMAP.md) is kept for
history but explicitly superseded by this audit).

## Import footgun: the module is tied to a specific cpython

The Python module (`pops._pops`) is a `.so` linked to the interpreter that compiled it (e.g. a
`cpython-312` `.so`). Observed consequences:

- importing it under an interpreter of another version (e.g. a system `python3` 3.9) fails,
  with a message that now names the expected tag and the rebuild command;
- without numpy, `import pops` and `pops.System` work; only `pops.dsl` (host evaluator)
  fails, with a message that asks for numpy.

You must therefore use exactly the 3.12 interpreter that built the module (with numpy), and
point `PYTHONPATH` at the corresponding `build*/python`, or reinstall with the wanted backend
(`POPS_USE_KOKKOS=ON pip install .`). See [installation](../getting-started/installation.md).
