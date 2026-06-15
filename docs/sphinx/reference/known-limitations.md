# Known limitations

An honest and consolidated list of `adc_cpp`'s current limits, to avoid any wrong
expectations. Each item points to the corresponding source of truth. These are not bugs:
they are the edges of the scope validated to date.

## GPU: validated manually on ROMEO, not in CI

CI never builds `-DADC_USE_KOKKOS=ON -DKokkos_ENABLE_CUDA=ON`. All GPU validation
(Kokkos Cuda single-GPU, and MPI + Kokkos Cuda multi-GPU) is done by hand on the
ROMEO supercomputer (GH200 node, `Kokkos_ARCH_HOPPER90`, `nvcc_wrapper`, OpenMPI
CUDA-aware), via SBATCH harnesses in `python/tests/gpu/` (excluded from the CI glob). A
"GPU-validated" mention therefore means "manually on ROMEO GH200", with the quantified evidence cited in
[GPU_RUNTIME_PORT.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/GPU_RUNTIME_PORT.md) and [BACKEND_COVERAGE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md).
Several GPU cells of the matrix remain `?` (not yet exercised on device); see the
"Lacunes notables" section of the source document.

## DSL: parity asserted only if a C++23 compiler is present

The symbolic DSL (`adc.dsl`) compiles a model to a `.so` at runtime (backends `aot` /
`production`) by invoking the C++ compiler against `adc_cpp`'s headers. The parity
verification (DSL vs native brick) therefore relies on the presence of a working C++23 compiler
on the machine. Without a C++23 toolchain, compilation of DSL models fails; only the
purely native paths (`adc.Model(...)` / `add_block`) remain available.

## DSL backends: prototype/aot are CPU-only

`m.compile(..., backend=...)` (default `aot`):

- `prototype` (JIT) and `aot` (host-marshaled) are CPU-only: no MPI, no AMR, no
  GPU, single-rank. Their flat `.so` ABI carries neither the `stride` (multirate cadence) nor
  `evolve=False` nor the partial IMEX mask; these options are rejected explicitly (explicit
  route rather than silent ignore).
- `production` is the only one pluggable on AMR (`AmrSystem.add_equation` requires
  `backend='production'`, `target='amr_system'`) and the only one that goes through the native
  zero-copy path; even there, from Python, the `stride` and the partial IMEX mask are rejected on the
  `.so` path (the loader ABI does not carry them). Detail in
  [DSL_MODEL_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/DSL_MODEL_DESIGN.md).

## Poisson FFT: refused under MPI

The spectral Poisson solver (`PoissonFFTSolver`) is single-rank by design. Under MPI (`n_ranks >
1`), the FFT path is refused with a collective error; a test locks this
non-regression. For a distributed MPI Poisson, use the geometric multigrid
(`geometric_mg`).

## AMR: global Schur on a single block only

The Schur-condensed source stage (`adc.CondensedSchur` via `adc.Split` / `adc.Strang`) is
available on AMR through `AmrSystem.add_equation`. The condensed stage is assembled and solved
on the coarse level and requires a single-block hierarchy; it raises on a refined multi-block
one. `AmrSystem.add_block` rejects `adc.Split` / `adc.Strang` (use `add_equation`). On `System`
the stage is available on any cartesian or polar grid. Design:
[SCHUR_CONDENSATION_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/SCHUR_CONDENSATION_DESIGN.md),
[AMR_MULTIBLOCK_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/AMR_MULTIBLOCK_DESIGN.md).

## Polar geometry: single-rank, scalar ExB only

The polar mesh (`adc.PolarMesh`, global ring (r, theta)) is wired into `System.step`
(polar transport + polar Poisson + aux in local e_r/e_theta basis), but with sharp
edges:

- scalar ExB transport only: fluid limiter / Riemann are not lifted on the
  polar side;
- single-rank: the direct polar solver (single box covering the ring) refuses MPI
  (`n_ranks > 1` raised); the polar counterpart of `CondensedSchur` is also single-rank;
- no cartesian <-> polar coupling: the polar ring is a separate global domain.

## Two-fluid AP: scenario in adc_cases, not a core brick

The two-fluid isothermal asymptotic-preserving integrator is not a composable brick of the
core: it is a scenario. Its AP stabilization couples the stiffness to the time step in
the elliptic, which `System` composition does not reproduce. Its C++ physics lives in
`adc_cases/two_fluid_ap/` (`two_fluid_ap.hpp`), compiled on the fly against the generic
headers of `adc_cpp`; the `_adc` module does not expose it and `adc` does not re-export it.

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

The Python module (`adc._adc`) is a `.so` linked to the interpreter that compiled it (e.g. a
`cpython-312` `.so`). Observed consequences:

- importing it under an interpreter of another version (e.g. a system `python3` 3.9) fails,
  with a message that now names the expected tag and the rebuild command;
- without numpy, `import adc` and `adc.System` work; only `adc.dsl` (host evaluator)
  fails, with a message that asks for numpy.

You must therefore use exactly the 3.12 interpreter that built the module (with numpy), and
point `PYTHONPATH` at the corresponding `build*/python`, or reinstall with the wanted backend
(`ADC_USE_KOKKOS=ON pip install .`). See [installation](../getting-started/installation.md).
