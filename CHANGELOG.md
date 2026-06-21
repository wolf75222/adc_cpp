# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
[SemVer](https://semver.org/lang/en/) (0.y.z while the public API still moves).

## Versioning policy

- **Single source**: `project(VERSION x.y.z)` in `CMakeLists.txt`. Everything derives from it:
  `adc.__version__` (bakes `ADC_VERSION` into `_adc`), the pip wheel (regex in `pyproject.toml`),
  `adcConfigVersion.cmake` (install/export). NEVER duplicate the number elsewhere; the docs build
  derives it too (`scripts/build_docs.sh` injects `PROJECT_NUMBER`, `docs/sphinx/conf.py` reads
  `project(VERSION)`), so nothing is bumped by hand outside `CMakeLists.txt`.
- **Bump**: PATCH = fixes with no API change; MINOR = backward-compatible API/brick additions;
  MAJOR (post-1.0) = API or ABI break of the DSL production path.
- **Tag**: set `git tag vX.Y.Z` on master when the bumping PR merges, then `git push --tags`.
- On every notable PR: one line in `[Unreleased]` below; on a bump, the section becomes
  `[x.y.z] - YYYY-MM-DD`.

## [Unreleased]

### Added

- **`.github/CODEOWNERS` routes review by zone** (ADC-252): the owner of a touched directory is
  auto-requested as reviewer. Solo today, `@wolf75222` owns every zone; the per-zone split documents
  the responsible owner per area and prepares reviewer routing at scale. This is the SWE-at-Google ch.10
  "Ownership" bit of the review process; turning it into a merge gate ("require review from Code
  Owners") is the branch-protection decision (ADC-166). CONTRIBUTING "Code review" links to it.
- **Regression guard: the red-black GS smoother reuses the cached halo schedule** (ADC-262):
  `tests/test_poisson_smoother_cache.cpp` asserts that a multi-level GeometricMG V-cycle builds the
  halo schedule once per MG-level layout and adds ZERO further builds across subsequent cycles, so the
  RB-GS smoother (the 86%-dominant Poisson path) does not re-enumerate the exchange jobs per sweep.
  This locks the lever-(a) win that ADC-260 (the `fill_boundary` halo-schedule cache) already shipped.
  A `bench/profile_step` measurement (serial, n=256, geometric_mg) confirmed the remaining Poisson cost
  is algorithmic (GS V-cycle iterations: 164 ms/step vs 10 ms/step with the FFT solver), not the
  per-sweep `fill_ghosts`, so on the SERIAL path the proposed face-subset/fused-exchange lever (b) is
  not worth it (sub-1% saving, and it would break bit-identity). Note: the ADC-260 cache hoists only
  the schedule ENUMERATION; under MPI the per-sweep pack/exchange still runs every sweep (the bottom
  level alone is ~100 exchanges per V-cycle), so the distributed-exchange lever (b) is NOT addressed
  here and is left as a follow-up. Test only; no behavior change.
- **`roe_abs_apply`: device-clean matrix-absolute-value for a generic moment Roe** (ADC-368): a new
  `adc::roe_abs_apply<N>(A, dU, out)` in `include/adc/numerics/dense_eig.hpp` returns the Roe
  dissipation `|A| dU = R |Lambda| R^-1 dU` of a small dense flux Jacobian via the
  infinity-norm-scaled Newton matrix-sign iteration (`|A| = A sign(A)`); `ADC_HD`, no allocation,
  `N <= 16`. For a real-diagonalizable `A` this equals the reference `flux_ROE_local.m` dissipation
  exactly (the Harten floor is inactive at O(1) wave speeds), and it returns `false` -- so the caller
  falls back to a spectral-radius (Rusanov) bound -- when the spectrum is not real or `A` is singular.
  This is the numerical kernel for the upcoming generic moment-system Roe flux (HyQMOM15); no public
  behavior changes yet. Covered by `tests/test_dense_eig.cpp` (parity to `R |Lambda| R^T` up to N=15,
  plus the complex and singular fallbacks).
- **Generic moment Roe flux: `m.roe_from_jacobian()` + `build_moment_model(roe=True)`** (ADC-368):
  a DSL emitter that makes `riemann='roe'` available for a moment hierarchy (HyQMOM) with NO fluid
  roles and NO primitive `p` (unlike `m.enable_roe`). It emits the `roe_dissipation` hook =
  `|A| (UR - UL)` with `A = dF_dir/dU` the autodiff flux Jacobian evaluated at the arithmetic-mean
  interface state, `|A|` via `adc::roe_abs_apply` (spectral-radius Rusanov fallback on a
  complex/singular spectrum) -- the reference `flux_ROE` dissipation. It is the third (mutually
  exclusive) provider of the `roe_dissipation` hook and folds into the model cache key; without a
  call nothing is emitted (bit-identical). `adc.moments.build_moment_model` gains a `roe=False`
  option (additive to `exact_speeds`, which still supplies `max_wave_speed`). This also fixes a
  latent core gap: `SourceFreeModel` (the IMEX explicit-half-step wrapper) now forwards the Roe/HLLC
  capability hooks (`roe_dissipation` / `contact_speed` / `hllc_star_state`), which it previously
  dropped (it forwarded only `pressure` / `wave_speeds`) -- so a non-Euler model on `riemann='roe'`
  (or `'hllc'`) lost the hook through the IMEX path and fell back to the Euler-4var branch, a compile
  error for `n_vars != 4`. Covered by `python/tests/test_dsl_roe_from_jacobian.py` (codegen + AOT
  compile + `System` `riemann='roe'` 10-step mass conservation + rejection without the capability).
- **Configure-time guard: Kokkos CUDA is rejected on native Windows** (ADC-168): `cmake` now fails
  fast with a clear message (use WSL2 for the GPU; native Windows is CPU Serial/OpenMP only) when
  `Kokkos_ENABLE_CUDA=ON` is requested on a native Windows configuration, instead of letting the
  unsupported Kokkos CUDA build break deep inside configure. The supported GPU path stays WSL2
  (ADC-96); Linux, macOS, and WSL2 are unaffected.
- **Per-field configurable aux halos for named aux** (ADC-369, follow-up of ADC-291): a model-declared
  named aux field can now carry its OWN ghost boundary policy via `adc.AuxHalo("foextrap")` /
  `adc.AuxHalo("dirichlet", value=v)` passed to `set_aux_field(block, name, field, halo=...)`, instead
  of inheriting the single shared phi-derived aux BC. The policy is applied to the NON-PERIODIC faces
  only (periodic faces -- a periodic domain, the polar theta direction -- keep their wrap, so a per-field
  policy never breaks the domain periodicity). Implemented by a component-scoped `fill_physical_bc`
  overload + `aux_halo_override` (`include/adc/mesh/physical_bc.hpp`), applied after the shared aux fill
  in `SystemFieldSolver` (cartesian + polar), `AmrRuntime` and `AmrCouplerMP` (coarse level). For the
  cartesian System COMPILED-block path the policy is carried to the `.so` via an append-only tail on the
  marshaled aux array (read by `compiled_block_abi.hpp` `make_grid`); the header signature (`abi_key`)
  bumps automatically so a stale `.so` is never mixed. `adc.capabilities()['aux']['named']['halo_policy']`
  reports it. Default (no halo) is strictly bit-identical; canonical fields (`B_z`, `T_e`) unchanged.
  Per-face asymmetric BC and the JIT host path remain follow-ups.
- **Named aux phase 2: AMR, polar, and a declarative `kAuxMaxExtra`** (ADC-291): model-declared named
  auxiliary fields (`m.aux_field("name")`, component `kAuxNamedBase + k`, read via `aux.extra_field(k)`)
  now work beyond the cartesian `System`. The polar `System::add_block` path widens the shared aux
  channel (`ensure_aux_width`), fixing a silent out-of-bounds read for any polar model with `n_aux > 3`
  (e.g. the polar Lorentz block) and unlocking `set_aux_field` / `aux_field` on `PolarMesh`.
  `AmrSystem.set_aux_field(block, name, array)` carries a static named field through both AMR engines
  (single-block `AmrCouplerMP`, multi-block `AmrRuntime`); the field is re-applied each `compute_aux` /
  `solve_fields` so it persists across a regrid and is injected to every level. The JIT host residual
  (`native_loader::host_residual`) now marshals named fields too (previously read as `0`). A new
  host-side C++ canonical name table (`include/adc/core/aux_names.hpp`) mirrors the Python
  `AUX_CANONICAL`, and `adc.capabilities()['aux']` is restructured (backends list, introspectable
  `limit`, `halo_radius`) and reads `kAuxMaxExtra` from the single C++ source so the Python mirror
  cannot silently drift. The only remaining compile-time aux limit (`kAuxMaxExtra`) is now declarative
  (`kAuxMaxComps` + static_asserts) and tested (`test_aux_names`, `test_native_aux_named`,
  `test_capabilities`, polar/AMR end-to-end in `test_aux_named.py`). Default paths stay bit-identical
  (empty named-aux map). The per-field configurable aux halo radius remains a documented follow-up.
- **Generic embedded-boundary / level-set domain contract** (ADC-327): the cut-cell / mask transport
  geometry is now expressed through a named, device-clean POD contract in
  `include/adc/numerics/embedded_boundary.hpp` (`level_set(x, y) < 0` inside, callable `operator()`,
  `cell_active`; built-in instances `DiscDomain` and the non-disc `HalfPlaneDomain`; a
  diagnostics-only `LevelSetDomain` concept that never constrains the hot-path templates).
  `numerics/spatial_operator_eb.hpp` no longer depends on the runtime `wall_predicate.hpp`. The
  Python helpers `set_disc_domain` / `set_geometry_mode` / `disc_mask` keep their name, signature and
  behavior (the disc is now one instance of the contract); the internal System fields are renamed
  `disc_ -> eb_domain_`, `disc_set_ -> eb_set_`, `disc_mask_ -> domain_mask_`, and runtime errors now
  speak of an embedded boundary / level-set domain. Default path strictly bit-identical;
  `tests/test_embedded_boundary_generic.cpp` locks the non-disc path end to end.
- **Generic real/complex spectrum predicate in `dense_eig.hpp`** (ADC-276): `adc::real_spectrum<N>()`
  classifies a small dense block as `Spectrum::kReal` / `kComplexPair` / `kUnknown`, with
  `EigBounds::all_real()` / `has_complex_pair()` accessors. The imaginary tolerance is relative
  (`im_tol * max(|lmin|, |lmax|, 1)`, default `1e-5`, covering a real multiplicity up to 3 -- the 3x3
  target -- since eps^(1/3) ~ 6e-6), so a quasi-degenerate real spectrum is not mislabeled complex, and
  non-convergence maps to `kUnknown` (never read as real) -- letting a native device projector test
  realizability on, e.g., a 3x3 HyQMOM15 block without NumPy or MATLAB. Header-only and additive:
  `real_eig_minmax` / `EigBounds` layout and the DSL eig path are unchanged.
- **DSL surface for the real/complex spectrum predicate** (ADC-362): `dsl.eig_all_real(rows,
  im_tol=1e-5)` exposes ADC-276's `EigBounds::all_real()` as a branchless DSL value (`1.0` if the small
  dense block's spectrum is real and the block converged, `0.0` otherwise -- complex pair or
  non-convergence), so a `m.projection` mask can ask "is this block's spectrum real?" without a branch.
  It lowers to a named device-clean functor returning `adc::Real(adc::real_eig_minmax(M).all_real(
  im_tol))`; gating on `converged` keeps a Gershgorin fallback at `0.0` (never read as real), unlike a
  raw `eig_max_im <= tol` whose `max_im` is `0` under fallback. The host NumPy mirror (LAPACK always
  converges, so no `kUnknown`) coincides with the generated brick on healthy matrices. Additive to
  `dsl.py`; the scalar `eig_max_im` / `eig_lmin` / `eig_lmax` path is bit-identical
  (`python/tests/test_projection_eig_predicate.py`).
- **Public `System.set_source_stage` on the Python facade** (ADC-308): the Schur-condensed source
  stage, already wired internally by `add_equation(time=adc.Split(source=adc.CondensedSchur(...)))`, is
  now reachable as a public `adc.System.set_source_stage(name, kind, theta, alpha, ...)` method (a thin
  pass-through to the binding with the same flat signature and defaults), so cases configure a post-hoc
  source stage without reaching into the private `_s`. Purely additive: the public call is
  bit-identical to the historical `_s.set_source_stage` path
  (`python/tests/test_set_source_stage_facade.py`).
- **One-command Python build** (ADC-358): `scripts/build_python.sh` activates the `adc` env, sizes the
  heavy-TU Ninja pool (`ADC_HEAVY_TU_POOL`) from cores capped by RAM, exports the Kokkos/CMake
  discovery vars (`Kokkos_ROOT`, `ADC_KOKKOS_ROOT`, `CMAKE_PREFIX_PATH`) and a stable cross-worktree
  ccache, runs `pip install . --no-build-isolation`, and ends on `adc.doctor()`; `--clean` drops the
  wheel cache, `--fresh` also clears ccache for a cold build. `scikit-build-core` is now pinned in
  `environment.yml` (so `--no-build-isolation` reuses the pinned stack) and `setup_env.sh` persists
  `CMAKE_PREFIX_PATH`. No change to `-O3` or generated code.
- **Prebuilt macOS arm64 wheel** (ADC-360): a `Wheels` workflow (`.github/workflows/wheels.yml`) builds a
  self-contained macOS arm64 / CPython 3.12 `_adc` wheel with `cibuildwheel` (Kokkos Serial built
  static + PIC and linked in; `delocate` bundles the rest), uploads it as a build artifact on
  build-relevant PRs, and attaches it to the GitHub Release on a `vX.Y.Z` tag. End users can then
  `pip install adc_cpp-*.whl` with no local toolchain. Separate from the required CI `gate`; no
  source or runtime behavior change. Linux/Windows wheels and PyPI publishing are follow-ups.
- **Configurable AMR regrid variable by name or role** (ADC-296): `AmrSystem.set_refinement` gains
  optional `variable=` / `role=` selectors so the multi-block union-of-tags regrid can refine on any
  conserved variable, not just component 0. Each block resolves the selector against its own conserved
  variables (`detail::resolve_selected_component`, STRICT: an absent name/role raises at build, no
  silent component-0 fallback), so a model whose refinement variable is not at component 0 refines
  correctly. The default (empty selector) stays component 0 and bit-identical; a non-default selector is
  multi-block only (mono-block `AmrCouplerMP` and the compiled `.so` loader refine on component 0 only
  and reject it). Surfaced under a new `regrid` key in `adc.capabilities()`. Per ADR-0001 Decision 5
  (Option A): the engine seam (`TagPredicate`, per-block predicates, union) is unchanged.
- **2D-core invariant published in `adc.capabilities()`** (ADC-294): `capabilities()` now exposes a
  structured `dimension` scalar (`== 2`) declaring the core's two-dimensional scope as an
  introspectable invariant, with a matching "Spatial dimension" section in
  `docs/sphinx/reference/known-limitations.md` and a cross-link in `include/adc/mesh/box2d.hpp`.
  Per ADR-0001 Decision 1 (Option A): purely additive, no API or ABI change;
  `python/tests/test_capabilities.py` pins the key. The ND core (`BoxND` / `GeometryND`) stays
  deferred to a future milestone.
- **Varying-kappa coverage for the screened-Poisson reaction term** (ADC-251):
  `tests/test_screened_poisson.cpp` gains an MMS case with a spatially varying `kappa(x,y)`, run
  through both `GeometricMG::set_reaction` overloads (`fn` and `MultiFab`). The pre-existing cases
  used only a constant kappa, so the per-cell diagonal read was unverified; order-2 convergence here
  proves the `(i,j)`-only read with 0 ghost cells is correct and sufficient, locking the deliberate
  no-ghost-fill invariant against any future stencil that would read kappa on its unfilled ghosts.
  Tests and docs only; the kappa wiring and the hot path are unchanged.
- **Multi-rank test for the collective IO gather** (ADC-257): `tests/test_mpi_system_io_gather.cpp`
  exercises `System::density_global` / `state_global` / `potential_global` (the `all_reduce_sum`
  gather behind `sim.write` / `sim.checkpoint`) under `mpirun -np {1,2,4}` -- previously covered only
  mono-rank by `python/tests/test_io_multirank.py`, which deferred the `np>1` case to a C++ file that
  did not exist. The test pins gather == known reference (bit-identical, np-invariant, catching any
  double-count), gather == local accessor on the owning rank after collective steps, and a
  checkpoint/restart round-trip that is bit-identical at np=2/4. Tests only; no change to the library,
  the API, or the hot path.
- **hyqmom15 AmrSystem multi-box GH200 validation** (ADC-320): a domain-decomposed driver
  (`docs/validation/diocotron_amr_gpu.cpp` + `diocotron_amr_mpi.sbatch`) wiring the compiled hyqmom15
  composite (`Hyqmom15Hyp/Src/Ell` + Poisson `geometric_mg`) on `AmrSystem` with
  `distribute_coarse=True`, so at `np > 1` the coarse `fill_boundary` exchanges the 15 conserved
  moments between GPUs (the real inter-GPU halo path that ADC-181's mono-box round-robin never
  exercised). Records the np=1/2/4 parity in `docs/validation/GH200_HYQMOM15.md` (cmax bit-identical,
  sums at last ulp, `coarse_local_boxes < coarse_total_boxes` proving the coarse distributed).
  Validation only; no change to the library, the API, or the hot path.
- **Coarse-level MPI ownership diagnostic** (ADC-319): `AmrSystem.coarse_local_boxes()` returns the
  number of base (level-0) boxes owned by the calling rank and `AmrSystem.coarse_total_boxes()` the
  total across all ranks. With `distribute_coarse=True` a distributed base gives `local < total` per
  rank; a replicated or single-box base gives `local == total` everywhere. Wired through the mono-block
  (`AmrCouplerMP`) and multi-block (`AmrRuntime`) paths, exposed in the pybind bindings and the Python
  facade. A general MPI strong-scaling diagnostic; no change to the numerics or the hot path.
- **Positivity floor on the AMR transport** (ADC-259): `AmrSystem.add_block` / `add_equation` now
  honor `spatial.positivity_floor > 0` (Zhang-Shu), previously rejected on the AMR path. The floor is
  threaded through both engines (single-block `AmrCouplerMP`, multi-block `AmrRuntime`) into
  `compute_face_fluxes` (Density-role face states) and adds a Density clamp on the coarse-fine fine
  ghost means (`fill_cf_ghost_cell`), the refined-patch interface the diocotron Hoffart failure
  exercised. Guarantee = face / C/F-ghost-mean Density positivity only (order-1 fallback), NOT
  updated-mean nor pressure positivity (parity with `System`). `positivity_floor == 0` is bit-identical
  to before; a model without a Density role rejects `positivity_floor > 0` explicitly.
- **Positivity floor on the compiled AMR `.so` path** (ADC-322): the production DSL loader
  (`adc_install_native_amr`) now marshals `positivity_floor` as a trailing flat argument, threaded
  through `add_compiled_model` / `set_compiled_block` into the same `compute_face_fluxes` leaf the native
  path uses (mono via `AmrBuildParams::pos_floor`, multi via a new `AmrCompiledBlockBuilder` slot). A
  `CompiledModel(target='amr_system')` block built with `positivity_floor > 0` now floors instead of
  raising (`AmrSystem.add_equation` / `add_native_block`); `positivity_floor == 0` stays bit-identical. A
  loader regenerated against pre-floor headers is rejected at load by the ABI key (the header signature
  changed), so the 9-argument call never reaches a stale 8-argument `.so`. Follow-up to ADC-259.
- **Distributed FFT Poisson under MPI** (ADC-287): `System.set_poisson(..., "fft"|"fft_spectral")` now
  runs with `n_ranks() > 1` via a box-slab remap (`RemappedFFTSolver`), replacing the previous explicit
  rejection. The new solver presents the System single round-robin box outward (so the field-solve path
  is unchanged) and hides a scatter/gather around `PoissonFFT` inside `solve()`. Periodic-only, constant
  coefficient, requires `Ny % n_ranks() == 0`; the potential matches `geometric_mg` to FP tolerance.
  `geometric_mg` stays the MPI default and the only option for walls, variable/anisotropic eps, or kappa.
  Ratified by the ADC-273 multi-agent design vote (correctness sound on every load-bearing axis; the
  elliptic `ell_` variant is not serialized, so no public-ABI break).
- **BGK collision helpers** (ADC-277): `adc.moments.maxwellian_moments` builds the local
  Maxwellian equilibrium moments of a 2D moment hierarchy (Isserlis closure, generic in the
  order and closure-free), and `adc.moments.bgk_source` returns the relaxation source
  `nu (M_eq - M)` toward it. Both work as DSL expressions or as a numeric oracle, and conserve
  mass and momentum exactly (the M00/M10/M01 rows are identically zero). BGK is meant to be
  wired through the existing source brick (`m.source` / `m.source_frequency`, explicit split or
  IMEX), so it adds no core trait, kernel, or stepper path. Strictly additive: the
  `build_moment_model` signature is unchanged. `Model.eval_source` (numpy source evaluator,
  parity with `eval_flux`) lets a host test check the emitted source without compiling.
- **Multi-GPU + MPI hyqmom15 diocotron validation harness** (ADC-181): the `docs/validation`
  diocotron driver gains an optional MPI bootstrap (`comm_init`/`comm_finalize` + rank-0 I/O
  guards) behind the new `ADC_VALIDATION_MPI` CMake option, so the same `diocotron_gpu.cpp`
  runs serial (unchanged at np=1) and under `srun -n N`. New `diocotron_mpi.sbatch` is the ROMEO
  GH200 recipe: build with CUDA-aware OpenMPI, run np=1/2/4 (one GH200 per rank), gate on per-run
  mass conservation (< 1e-12) and ulp-level global-mass parity vs np=1. Closes the System-MPI
  branch named in `GH200_HYQMOM15.md` section 3.
- **Coverage for the `step_cfl` zero / NaN wave-speed guard** (ADC-267): `tests/test_cfl_dt.cpp`
  gains two in-file cases (no new target) for the `std::max(w_max, 1e-30)` clamp that ADC-194 left
  untested. A quiescent state (`w_max == 0`) asserts `dt` is finite and equals `cfl*h/1e-30` (without
  the floor it would be `+inf`), and a model with `max_wave_speed() == NaN` asserts `cfl_dt` stays
  finite and positive (`system_max_wave_speed` does `max(0, NaN) == 0`, swallowing the NaN). Test-only;
  no change to the library, the API, or the hot path.

### Changed

- **`quality.yml` pins clang-format to 19.1.7** (ADC-381): the now-blocking `format` job installed
  whatever clang-format the `ubuntu-latest` runner shipped (~v14-18) via `apt-get`, so Google-style
  output drift between major versions could fail the gate on an otherwise-clean tree. It now installs
  the ADC-118 sweep version (`pipx install clang-format==19.1.7`) for a deterministic gate; the
  canonical version is documented in `.clang-format` and `CONTRIBUTING.md`. CI install step only; no
  source reformatting.
- **`quality.yml` clang-format check is now blocking** (ADC-219): with the tree conforming to
  `.clang-format` since the ADC-118 sweep, the `format` job's `clang-format --dry-run` step fails the
  job on any new style deviation (a regression) instead of only warning. `ruff` (Python) stays
  informational (`if: always()`, so its signal survives a format failure). This is the D15 hardening;
  the corresponding clang-tidy-family checks stay informative until their existing findings are fixed.
  Note: `quality.yml` runs on schedule, manual dispatch, or a `quality`-labeled PR, not on every PR.
- **Repo-wide clang-format sweep** (ADC-118): `include/`, `python/` and `tests/` reformatted to
  `.clang-format` in one mechanical commit (`78816b0`, no functional change), bringing the
  `format`-scanned trees to zero non-conforming. Its SHA is recorded in `.git-blame-ignore-revs` so
  `git blame` skips it (`git config blame.ignoreRevsFile .git-blame-ignore-revs`, wired into
  `scripts/setup_env.sh`; GitHub's blame UI honors the file automatically). The four tests carrying a
  generated-C++ raw string are fenced with `// clang-format off` to keep the emitted source verbatim.
- **`adc/coupling` headers grouped by family** (ADC-326): the 18 flat coupling headers are moved into
  abstraction families (`base/`, `source/`, `single/`, `static_system/`, `amr/`, `schur/`,
  `deprecated/`) so the API surface is legible; internal cross-includes now use the canonical family
  paths and a new `include/adc/coupling/README.md` documents each family's stability tier (stable,
  reference/static, AMR, Schur, deprecated). Every historical `#include <adc/coupling/<name>.hpp>`
  keeps compiling through a forwarding stub, so the move is source-compatible with no behavior change.
- **Cartesian spatial operator split into focused modules** (ADC-328): the ~975-line
  `include/adc/numerics/spatial_operator.hpp` is split into a one-way module DAG under
  `include/adc/numerics/spatial/` (`state_access`, `positivity`, `face_flux`, `wave_speed`,
  `cartesian_operator`, `masked_operator`), each documenting its own contract (state/aux access,
  reconstruction + positivity, face flux, wave-speed reductions, residual assembly, masked path).
  `spatial_operator.hpp` becomes an umbrella header that re-exports every symbol, so existing
  `#include <adc/numerics/spatial_operator.hpp>` users are unaffected and the numerics are
  bit-identical (pure code move, no behavior change).
- **Reference and deprecated AMR headers moved out of the production surface** (ADC-332): the
  single-box Fab2D/MultiFab oracle headers `numerics/time/amr_reflux.hpp` and
  `numerics/time/amr_level.hpp` move to `numerics/time/reference/`, and the deprecated N-level
  engine `numerics/time/amr_multilevel.hpp` moves to `numerics/time/deprecated/`, so the tree makes
  their status explicit. Forwarding stubs keep every historical
  `#include <adc/numerics/time/...>` path compiling (the live `amr_reflux_mf.hpp` aggregator still
  reaches `amr_level.hpp` through the stub), so there is no behavior change. Audit recorded: none of
  the three headers, nor the deprecated `coupling/spectral_coupler.hpp`, has any live includer in the
  core, the tests, the bindings, or `adc_cases` (`amr_reflux.hpp` is pulled only by the deprecated
  `amr_multilevel.hpp` and `coupling/spectral_coupler.hpp`); `spectral_coupler.hpp` is relocated to
  `coupling/deprecated/` by ADC-326.
- **Generalize runtime/system header comments** (ADC-370): follow-up to ADC-333 on the headers that
  were then in flight with ADC-369 (`runtime/system.hpp`, `runtime/amr_system.hpp`,
  `runtime/system/system_field_solver.hpp`, `coupling/amr_coupler_mp.hpp`). Diocotron/Hoffart/
  ROMEO-GH200 narration reworded to the generic contract; the provenance is recorded in
  `docs/validation/HEADER_PROVENANCE.md`. `system_stepper.hpp` is already generic (its
  `HOFFART_STEP_SEQUENCE.md` references are sanctioned validation-doc pointers). Comments and docs
  only; no behavior change.
- **Hoisted the SSPRK RK scratch out of the per-substep hot loop** (ADC-261): the explicit steppers
  (`ForwardEuler`, `SSPRK2Step`, `SSPRK3Step`) re-allocated and zero-initialized fresh full-state
  scratch (the residual `R` plus the stage copies `U1`/`U2`/`U3`) on every `take_step`, so an
  n-substep explicit advance churned malloc/zero/free n times per macro-step. Each stepper now has a
  reusable `Scratch` struct and a `take_step(rhs, U, dt, Scratch&)` overload; a `run_explicit_substeps`
  helper allocates the scratch once per advance and reuses it across substeps (with a one-shot
  fallback for custom user steppers). The one-arg `take_step` is kept for the existing callers. The
  same hoist already lived in `AdvanceImexRkArs222`; behavior is bit-identical (the buffers are fully
  overwritten each substep and stage ghosts are re-derived by the residual fill_ghosts), locked by
  `test_weno5_ssprk3` (1e-14 parity).
- **Runtime headers organized by layer** (ADC-330): the flat `include/adc/runtime/` directory is
  split into layers so the structure shows the API surface. The public facades stay at the root
  (`system.hpp`, `amr_system.hpp`, `model_spec.hpp`, `facade_options.hpp`, `export.hpp`); the ABI /
  dispatch / internal helpers move to `detail/`, the DSL and native model builders to `builders/`,
  the extracted `System` internals to `system/`, and the AMR engine to `amr/`. Every historical
  include path (`<adc/runtime/*.hpp>`) keeps compiling through a forwarding header, so the runtime
  consumers, the bindings and the tests are unaffected. Source-tree layout only; no behavior or
  ABI change.
- **Factored the geometry-free Schur source kernels into one header** (ADC-263): the five device
  functors carried byte-identical by the Cartesian and polar Schur source steppers
  (extrapolate-scalar, extrapolate-velocity, energy, extract-velocity, copy-Bz) plus the
  `set_krylov` validation now live in `include/adc/coupling/schur_source_kernels.hpp`, which depends
  only on the lightweight `Array4`/`ConstArray4` handles (not on the elliptic solver stack), so the
  polar path can share them instead of re-declaring its own copies. Only the metric-bearing kernels
  (reconstruct, operator-coeff, explicit-flux, RHS-assemble) stay local to each stepper. Behavior is
  bit-identical; covered by the existing Cartesian/polar/AMR Schur tests (serial + MPI np2/np4).
- **Centralized native AMR refinement ratio** (ADC-295): the refinement ratio of the in-house AMR
  hierarchy is now a single named invariant `kAmrRefRatio` (with a `require_supported_ref_ratio`
  guard) in `include/adc/amr/refinement_ratio.hpp`, instead of the literal `2` scattered across the
  coarse/fine paths. `AmrHierarchy` defaults to and validates that ratio, rejecting any other value
  at construction with a clear error rather than silently mis-coarsening; the parametric call sites
  (`coarsen_index`/`coarsen_grown`/`refine`, composite-FAC `CompositeFacPoisson`/`fac_bilerp_coarse`,
  the subcycling `dxc / r` spacings and `SubcyclingSchedule`) read the constant, and the
  ratio-2-structural kernels (the NxN average-down and coarse-fine flux unrolls in `numerics/time`)
  carry a `static_assert` so a future non-2 ratio fails loudly at exactly those sites. `capabilities()`
  now states `refinement_ratio = 2 only`. Behavior is bit-identical (the constant is 2);
  `test_ref_ratio` locks the invariant and the rejection.
- **Share the Newton-options range validation** (ADC-213): the four-check defensive range validation
  of the `NewtonOptions` POD (duplicated verbatim in `System::add_block` and `AmrSystem::add_block`)
  is factored into `validate_newton_options(newton, where)` next to `NewtonOptions`
  (`implicit_stepper.hpp`). The `time='imex'` gate and `newton_diagnostics` handling, which differ
  between the two callers, stay inline. The AMR `newton_fail_policy` error text is harmonized to the
  System wording (no test depends on it; the binding's string parser rejects bad policies first, so
  the integer-range check is unreachable from Python). No behavior change.
- **Elliptic solver headers organized by family** (ADC-334): `include/adc/numerics/elliptic/` is
  split into `interface/`, `poisson/`, `mg/`, `eb/`, `polar/`, and `linear/` subdirectories so the
  numeric surface shows its solver families instead of one flat directory. Every historical include
  path (`<adc/numerics/elliptic/*.hpp>`) keeps compiling through a forwarding header, so downstream
  consumers and docs are unaffected; only the source-tree layout and the headers' internal
  cross-includes change. No numerical behavior, public API, or capability surface changes.
- **Generalize core public-header comments** (ADC-333): the headers under `include/adc/**` now read
  as the generic math contract (invariants, preconditions, expected errors, maintainer warnings)
  rather than the implementation of one scenario. The diocotron/Hoffart reproduction, the
  ROMEO/GH200 machine names, and the cross-repo `adc_cases` ticket trails are reworded generically
  and the provenance is relocated to the validation docs (new `docs/validation/HEADER_PROVENANCE.md`,
  plus the existing `HOFFART_*`/`SCHUR_CONDENSATION_DESIGN`/`DIOCOTRON_GROWTH_RATE` docs). Comments
  and docs only; no behavior change.
- **Factor the stepper global-dt-bound and grid-step duplication** (ADC-213): `step_cfl` and
  `step_adaptive` shared, verbatim, the `add_dt_bound` `all_reduce_min` loop and the
  polar/Cartesian minimum-grid-step expression. Both are now private `SystemStepper` helpers
  (`apply_global_dt_bounds(dt, reason*)`, mirroring the existing `apply_coupled_freq_expr_bounds`
  idiom, and `cfl_grid_h()`). Trajectory is bit-identical and the MPI collective count/order is
  unchanged (no behavior change).
- **Factor the stepper due-block advance loop** (ADC-213): `step` (Lie) and `step_cfl` shared,
  verbatim, the per-block loop that advances every DUE block by its catch-up step (transport plus the
  opt-in Schur source stage). It is now a private `SystemStepper::advance_due_blocks(dt)` helper. The
  two other macro-steps keep their own loops (`step_strang` interleaves a re-solved source stage
  between two half advances; `step_adaptive` subcycles each block `n_b` times). The helper introduces
  no new collective and iterates the same blocks in the same order, so each rank issues the same
  sequence of transport/source halo and `all_reduce` calls as before: the trajectory is bit-identical
  and the MPI collective count/order is unchanged (no behavior change).
- **Builtin model bricks behind a single registry** (ADC-331): the transport / source / elliptic
  tag lists are centralized in `include/adc/runtime/model_registry.hpp` (constexpr `kTransports` /
  `kSources` / `kElliptics` tables plus CSV / choices / validator helpers), the model-axis
  counterpart of `dispatch_tags.hpp`. The sites that re-listed `exb|compressible|isothermal` inline
  (`model_factory.hpp`, `block_builder_polar.hpp`, `python/system.cpp`, `python/amr_system.cpp`) and
  the completeness messages of `validate_model_spec` now derive from that single table, so adding a
  builtin brick is one table row plus its compile-time dispatch case. Rejection messages stay
  byte-identical; a non-drift `static_assert` ties the registry `n_vars` to the brick types, the
  dispatch keeps an explicit registry/dispatch-consistency guard, and `test_model_registry.cpp` plus
  a routing-completeness check lock the table to generic brick tags (never an application scenario
  name). The `ModelSpec` / capabilities surface and the compiled native backend are unchanged.
- **Named user roles and strict named-coupling fallbacks** (ADC-292): `VariableSet` gains a
  string-keyed `user_roles` layer parallel to the canonical `VariableRole` enum, and a new
  `index_of(string)` resolves a component by a canonical role name OR a user-defined role label
  (removing the first-occurrence ambiguity of several `Custom` components). Coupled-source role
  resolution (`System` / `AmrRuntime::add_coupled_source`), implicit-role masks and the AMR regrid
  selector now accept a user-role label, and the label round-trips through the compiled-block `.so` roles CSV
  (`roles_csv` / new `parse_roles_into`; the flat ABI and `abi_key` are unchanged). The legacy named
  couplings (`add_collision`, `add_thermal_exchange`, `add_ionization`) now fail loud (new
  `coupling_role_index`) when a ROLES-BEARING block omits a required role instead of silently coupling
  component 0; a genuinely ROLELESS block keeps the canonical fallback. Shipped models declare full
  canonical roles, so they stay bit-identical. Per ADR-0001 Decision 4 (Option A).
- **Cache and harden the macOS wheel build** (ADC-367): the `Wheels` workflow now caches the Serial
  static+PIC Kokkos install (deterministic key on the pinned 4.4.01 commit + flags, so warm runs skip
  the Kokkos build) and wires `ccache` into the `_adc` compile (`CMAKE_CXX_COMPILER_LAUNCHER` +
  persisted `CCACHE_DIR` + `CCACHE_BASEDIR`/`NOHASHDIR`/`COMPILERCHECK=content`; a `ccache -s` step
  reports the warm hit-rate). Hardening: the Kokkos clone is retried and its checkout asserted against
  the pinned commit (re-pointed-tag / supply-chain guard), `-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0` is
  passed explicitly to the Kokkos build (no min-version skew vs `_adc`), and the tag-time release-attach
  is best-effort (a missing Release no longer red-flags a successful wheel build). CI-only.
- **Shard + instrument the gate-python CI** (ADC-366): the PR-blocking `gate-python` job is split into 3
  `matrix.shard` legs (round-robin over the sorted test list, ~1/3 of the ~18-19 min Python suite each);
  the `gate` aggregator still `needs: gate-python` so the required check name is unchanged (branch
  protection intact) and `fail-fast: false` keeps a full red/green signal. The opaque test runner is
  replaced by a per-file timing harness (slowest-first log + a TSV/JSON timings artifact) that preserves
  the exact fail semantics. ccache hit-rate is lifted on both gates (`CCACHE_BASEDIR`, `CCACHE_NOHASHDIR`,
  `CCACHE_SLOPPINESS` + a codegen-neutral `-ffile-prefix-map`; cache keys unchanged, abi_key untouched),
  and `setup-kokkos` bumps `actions/cache@v4 -> v5`. CI-only; no change to `-O` levels, the shipped
  library, or test coverage (the 3 shards are a complete, disjoint partition of all 110 test files).
- **Trim push-master CI scope and the MPI job** (ADC-366): the full suite (MPI + Kokkos OpenMP + bench)
  now runs on a push to `master` only when a build/backend path changed (a new conservative `full`
  paths-filter covering sources, tests, build files, `python/**`, `bench/`, `scripts/`, and the CI
  action/workflow); a push touching only meta files (other workflows, linter configs, LICENSE) skips
  them, with the nightly cron as the unconditional backstop and the `ci-full` label as the manual
  override (schedule/dispatch stay full). The MPI job runs `ctest --preset ci-mpi -L mpi` (the serial
  tests already run in gate-cpp), backed by a `tests/CMakeLists.txt` backfill that labels every test in
  the `ADC_HAS_MPI` block `mpi` (was ~8 of 60) plus a CI floor that fails loudly if the selection
  collapses. CI-only; no change to `-O` levels or global test coverage.
- **Split `bindings.cpp` into per-area pybind TUs** (ADC-365): the `py::class_` / `.def` surface moves
  from the single `PYBIND11_MODULE` into `init_core` (module attrs + `SystemConfig` + `ModelSpec`),
  `init_system` (`System`), and `init_amr` (`AmrSystemConfig` + `AmrSystem`), each its own TU declared in
  `python/bindings_detail.hpp` (which also holds the shared `to_2d`/`to_3d`/`flat`/`newton_fail_policy`
  helpers). `bindings.cpp` is now a thin module that calls them in order (init_core first, so the configs
  register before `System`/`AmrSystem` reference them). The bound API is byte-identical (bodies moved
  verbatim); the win is parallel compilation + lower peak pybind memory per TU, and better incrementals.
- **Memoize the fill_boundary halo schedule** (ADC-260): `fill_boundary_begin` no longer rebuilds the
  `BoxHash` and re-enumerates the full local + global (cross-rank send/recv) halo job list on every
  call. That schedule is a pure function of the layout (`BoxArray`, `DistributionMapping`, `n_grow`)
  and the per-call (`Periodicity`, domain), so it is now built once and memoized per distinct
  (`Periodicity`, domain) on the `MultiFab` (new `include/adc/mesh/halo_schedule.hpp`); only the local
  copies and the pack/MPI/unpack of the live data run per call. The plan is replayed in the SAME
  deterministic order, so packed buffers stay bit-identical and the per-rank send/recv lists stay
  aligned (`tests/test_mpi_mbox_parity`, `test_mpi_amr_compiled_parity` unchanged). The cache lives on
  the `MultiFab` and is dropped when the object is reassigned (AMR regrid builds a fresh `MultiFab`),
  so it cannot go stale. New `tests/test_fill_boundary_cache.cpp` (serial + `np=1/2/4`) proves cache-on
  equals rebuild bit-for-bit, the schedule is built once across K calls, and it is rebuilt when the
  periodicity, domain, `n_grow`, or layout changes. No API or behavior change; biggest win in the
  MG-dominated and multi-rank halo-dominated regimes.
- **Validation bricks out of the production physics surface** (ADC-329): the validation/reference
  bricks `AdvectionDiffusion`, `LangmuirMode` and `TwoFluidLinear` move from `include/adc/physics/` to
  `include/adc/validation/physics/` under the new `namespace adc::validation`, so the production brick
  surface (`physics/{hyperbolic,source,elliptic,composite}.hpp`, aggregated by `physics/bricks.hpp`)
  stays free of test-only models. The old `include/adc/physics/<name>.hpp` paths remain as deprecated
  compat forwarders that alias the type back into `adc::` (e.g. `adc::AdvectionDiffusion`), so existing
  and external includes keep compiling unchanged; `tests/test_physics_validation_compat.cpp` pins that
  both the new and legacy paths build and name the same types. No numerical behavior change.
- **Flux-subdivide the AMR compressible runtime TUs** (ADC-359, follow-up to ADC-335/342): the
  compressible (Euler 4-var) AMR seam was the heaviest remaining module TU because
  `python/amr_block_compressible.cpp` and `python/amr_compiled_compressible.cpp` each instantiated all
  four fluxes (the riemann dispatch lived inside `dispatch_amr_block` / `dispatch_amr_compiled`, whose
  hllc/roe capability guards pass for Euler). Each per-flux branch of the two dispatchers is factored
  into `dispatch_amr_block_<flux>` / `dispatch_amr_compiled_<flux>` (bodies moved verbatim), and the two
  TUs become a thin riemann dispatcher plus one per-flux TU each (`amr_{block,compiled}_compressible_{rusanov,hll,hllc,roe}.cpp`),
  so every flux compiles in parallel. `dispatch_amr_block` / `dispatch_amr_compiled` are unchanged and
  still serve the exb/isothermal seam. The reachable `build_amr_block` / `build_amr_compiled` leaf set,
  the validation, and the error messages are unchanged, so the numerics are byte-identical (guarded by
  the `dmax==0` parity suite). The 8 new TUs are added to the module, tests, bench, and docs/validation
  source lists.
- **Factor the multi-box global-gather idiom** (ADC-264): the five copy-pasted collective gather sites
  in `python/system.cpp` (`Impl::copy_comp0` / `copy_state` multi-box branches and
  `System::density_global` / `state_global` / `potential_global`) now route through a single
  anonymous-namespace `gather_global(mf, ncomp, gnx, gny)` helper (zero-init buffer, local-box write at
  global indices, `all_reduce_sum_inplace`, component-major). The loop bodies are moved verbatim, so
  every site stays bit-identical (`ncomp == 1` collapses the layout to `j*gnx + i`); the caller keeps
  the `device_fence` and the single-box fast path delegating to `SystemBlockStore`. Net -33 lines, no
  API, ABI, or behavior change. Locked by `tests/test_mpi_system_io_gather.cpp` (np=1/2/4, ADC-257) and
  `python/tests/test_polar_theta_boxes.py` (theta_boxes=2/4).
- **Explicit `ModelSpec`, no silent physics defaults** (ADC-290): `ModelSpec` no longer hard-codes the
  physics-selecting defaults `transport="compressible"` and `elliptic="charge"`; both tags are now unset
  by default and a new `detail::validate_model_spec` (called at `dispatch_model` and at the top of
  `System::add_block` / `AmrSystem::add_block`) rejects an unset `transport`/`elliptic` with a clear,
  field-naming message instead of silently composing Euler + Poisson-charge. `source="none"` (the
  explicit, neutral no-source choice) and all numeric defaults are unchanged; each numeric is read only
  once its brick is chosen, so it cannot inject physics on its own. The historical shortcuts stay at the
  Python edge (`adc.Model(...)` always sets the three tags). API note: a bare native `ModelSpec()` that
  relied on `compressible`+`charge` now raises (pre-1.0, see ADR-0001 Decision 2). Anti-regression tests
  (ADC-300): `tests/test_config_model_validation.cpp` and the `test_bindings.py` garde-fous assert the
  incomplete-spec rejection and message, so a silent Euler/charge fallback cannot return.
- **Validate `SystemConfig` / `AmrSystemConfig` before building the runtime** (ADC-299): `System` now
  validates the config (`n >= 1`, `L > 0`, plus the existing geometry/polar invariants) BEFORE
  constructing its `Impl`, which already derived the geometry, box array, distribution mapping and the
  aux `MultiFab` (all sized from `n`) ahead of the old post-construction `check_geometry`. An invalid
  `n`/`L` previously built a silent degenerate grid (`dx = L/0 = +inf` or negative `dx`); it is now
  rejected upstream with a message naming the cause. `AmrSystem` gains the same upstream guard with
  `L > 0`, `regrid_every >= 0` and `coarse_max_grid >= 0` (only `n >= 1` was checked, and after `Impl`).
  Covered by `tests/test_config_model_validation.cpp` and `test_bindings.py` (ADC-300). C++ and Python
  error messages stay aligned; no change to any valid run (bit-identical).
- **No-optimize the cold model/block factories** (ADC-337, P1-B): the host string->closure wiring
  (`dispatch_transport/_source/_elliptic/_model/_model_for`, `bind_variable_roles`,
  `resolve_implicit_components`, `make_implicit_mask`, `build_block`, `make_block`/`make_block_*`) is
  marked `ADC_COLD_FN` (clang `optnone` / gcc `optimize("O0")`, new `include/adc/core/cold.hpp`), so the
  backend stops inlining and `-O3`-optimizing the entire CompositeModel instantiation tree into one
  giant factory function -- the dominant slice of the heavy TUs' `-O3` cost (cf. `docs/BUILD_PROFILING.md`
  P1-B). The HOT kernels (`BlockRhsEval` / `Advance*` / `take_step` / Kokkos `for_each_cell`) are separate
  functions reached through `std::function` closures and stay `-O3`; the small closure-returning helpers
  (`make_max_speed` etc.) are left untouched. No `-ffast-math` and `-O0` vs `-O3` never changes IEEE
  results, so the numerics are byte-identical (guarded by the `dmax==0` parity suite). Stacks on ADC-335.
- **Flux-subdivide the isothermal runtime TU** (ADC-342, follow-up to ADC-335): `python/system_isothermal.cpp`
  instantiated both reachable fluxes (rusanov + hll) x 4 limiters x 15 models in one TU -- the post-split
  long pole (~120 `-O3` leaves). It is now split one `.cpp` per flux (`system_isothermal_rusanov.cpp` +
  `system_isothermal_hll.cpp`) via the existing per-flux seam, halving the leaves per TU so `-j` compiles
  them in parallel. System dispatches the riemann string to the matching seam (after the same
  `validate_riemann`/`validate_limiter` as `make_block`); hllc/roe (unreachable on a 3-var transport) hit
  the explicit registry throw. Byte-identical codegen for the kept combos (verbatim `make_block_<flux>`;
  guarded by the `dmax==0` parity suite), mirroring the ADC-335 compressible subdivision.
- **Pin the conda build toolchain and surface the heavy-TU pool** (ADC-338): `environment.yml` pins
  `pybind11>=2.13,<3` (the conservative/validated 2.x line; 3.x still compiles, drop `<3` to opt in) and
  documents the local-vs-validated Kokkos gap (conda ships a Serial CPU-dev `kokkos`, default per-platform
  -- dry-run verified osx-arm64 ~4.7.01 / linux-64 ~4.3.00 -- a separate artifact from the source-built
  GPU Kokkos 4.4.01 used on ROMEO/CI, so it is intentionally not hard-pinned). `scripts/setup_env.sh`
  keeps AppleClang the macOS default and installs the conda `cxx-compiler` (gcc 14.2 via cxx-compiler
  1.11.0) as the pinned Linux default -- the fix for the slow `-j40` Linux build (wrong floating host gcc)
  -- and now prints how to widen `ADC_HEAVY_TU_POOL` on a high-RAM host so `-j` parallelizes the
  (post-ADC-335) heavy sub-TUs, while CI/constrained machines keep the size-1 OOM guard. Pins verified to
  resolve by `conda create --dry-run` on osx-arm64 and (`--platform`) linux-64.
- **Parallelize the `_adc` build by splitting the heavy runtime TUs** (ADC-335): `python/system.cpp`
  and `python/amr_system.cpp` instantiated the full transport x source x elliptic x flux x limiter x
  integrator product (~1700 `-O3` leaves) in two giant TUs, so `_adc` had only 3 TUs and `-j` capped at
  3 (a colleague's `-j40` build took 2h+). The runtime dispatch is now split, by a verbatim move behind
  fixed-signature type-erased seams, into ~16 TUs: System by transport (`system_{exb,isothermal,polar}`)
  with the compressible/Euler transport further by flux (`system_compressible_{rusanov,hll,hllc,roe}`),
  and AmrSystem by transport x {single-block `AmrCouplerMP`, multi-block `AmrRuntime`}
  (`amr_{block,compiled}_{exb,isothermal,compressible}`). A new `ADC_HEAVY_TU_POOL` CMake cache var
  (default 1, anti-OOM) lets the Ninja heavy-TU pool widen so `-j` actually compiles the now-smaller
  sub-TUs in parallel. Byte-identical: the inner make_block / dispatch_amr ladders move unchanged, so the
  set of instantiated kernel symbols is identical (verified: `nm -g` exported table unchanged, 0 hot
  kernel leaves added/removed) and the production-parity suite stays `dmax==0`. Measured on an 8-core
  Mac: clean `-O3` `_adc` build 1112 s (pool=1) -> 284 s (pool=8, ~16 TUs), 3.9x. No numerics change.
- **Test build deduplicates the heavy runtime TUs** (ADC-336): `python/system.cpp` and
  `python/amr_system.cpp` are now compiled once per test configuration into two `OBJECT` libraries
  (`adc_runtime_system`, `adc_runtime_amr`) in `tests/CMakeLists.txt`, instead of being re-listed as a
  source in ~23 test executables (each a multi-GB cc1plus compile of the full dispatch product). The 23
  plain and MPI heavy-source targets link the matching object library; the 4 `ENABLE_EXPORTS`
  native-loader tests keep their own copy so the dlopen/-rdynamic resolution of the `ADC_EXPORT` runtime
  symbols is unchanged. Byte-identical: the object libraries carry exactly the flags those targets used
  before (`adc::adc` only, no extra defines), and the `-O0` RAM cap is propagated PUBLIC so each
  consumer's own `.cpp` stays at the same `-O` level, preserving the `add_compiled_model` vs `add_block`
  bit-parity (`dmax==0`) against FMA contraction. `_adc` (`python/CMakeLists.txt`) is untouched: it
  already compiled each TU once, and no preset co-builds tests and the Python module. Serial tree:
  `system.cpp` 6 -> 1 compile, `amr_system.cpp` 14 -> 5 (1 library + 4 retained loaders). No change to
  the numerics or the public API.
- **Reliable Linux/Ubuntu user install** (ADC-321): `scripts/setup_env.sh` now bootstraps a fresh
  machine end to end -- it guides the Miniforge install when `conda` is absent, configures conda-forge
  to survive HTTP 429, forces a CPU Kokkos by default via `CONDA_OVERRIDE_CUDA=""` (so `pip install .`
  no longer fails `Could not find nvcc` on a CPU host with an NVIDIA driver; `--cuda` opts in),
  persists `ADC_INCLUDE`/`ADC_KOKKOS_ROOT`/`Kokkos_ROOT`/`ADC_CACHE_DIR` in the env, and ends on
  `adc.doctor()`. `adc.doctor()` gains a `kokkos_root` check (the tutorial's "no DSL backend" blocker)
  and a CUDA-Kokkos-without-nvcc check, each with a copy-paste fix. `installation.md` gains a
  "Linux and Ubuntu: fresh install" section and a troubleshooting table; the diocotron tutorial routes
  a both-backends-failure to `adc.doctor()`. `setup_env.sh`, `environment.yml`, the diocotron tutorial
  and `pyproject.toml` are translated to English.
- **Distributed FFT Poisson hardening** (ADC-316, fast-follow to ADC-287): `RemappedFFTSolver::solve()`
  adds an owner-only `device_fence()` after the periodic-ghost wrap (PR #254 managed-buffer ordering
  discipline; the caller's post-`ell_solve` fence already brackets the read, so this is belt-and-
  suspenders that self-documents the seam, and a no-op on CPU). `test_mpi_system_fft` now asserts the
  remapped solver's `residual()` is machine-zero and covers a non-power-of-two grid (n=12), exercising
  PoissonFFT's O(n^2) DFT fallback under the box-slab remap. The System Cartesian single-box invariant
  is documented at its source (`python/system.cpp`).
- **Build parallelism derived from cores, not hardcoded** (ADC-339): the README no longer prints a
  literal `-j8`; it states the Ninja default already uses every core and gives the dynamic cap form
  (`-j$(nproc)` on Linux, `-j$(sysctl -n hw.ncpu)` on macOS). The ROMEO validation `.sbatch` builds
  that matched their `--cpus-per-task` allocation now read `-j "${SLURM_CPUS_PER_TASK:-N}"` so they
  self-adjust to the allocation. The CI `--parallel`, the WSL2 `-j 6` (RAM bound), and the parity181
  half-allocation nvcc cap stay explicit and documented (memory-bounded environments, intentional).

### Fixed

- **Quasi-vacuum velocity bound for the isothermal model** (ADC-77, third stability barrier of the
  WENO5 rollup): when the diocotron rollup evacuates the background (rho -> ~1e-7) the Schur source
  stage writes O(1) momentum onto those cells, so the raw u = m/rho exploded and collapsed the CFL.
  `IsothermalFlux` (and the inherited `IsothermalFluxPolar`) now computes the velocity as
  u = m/max(rho, vacuum_floor), which bounds both the wave speed and the advective flux at evacuated
  cells; mass and momentum are untouched (only the velocity ESTIMATE is bounded, unlike a cell density
  clamp). It is an independent opt-in knob set per model via
  `adc.FluidState("isothermal", cs2=..., vacuum_floor=...)` (carried on `ModelSpec::vacuum_floor`);
  it is deliberately SEPARATE from the spatial `positivity_floor` (the Zhang-Shu reconstruction
  limiter), which addresses a different failure mode -- coupling them would change the CFL dt of
  existing positivity_floor transport runs. With `vacuum_floor == 0` (default) the path is
  bit-identical. Covered by `tests/test_isothermal_vacuum_floor.cpp` and
  `python/tests/test_isothermal_vacuum_floor_system.py`. The energy (Euler) model is out of scope:
  a velocity bound alone does not stabilize it (the sound speed c = sqrt(gamma p / rho) also diverges
  at vacuum, a coupled pressure-positivity concern).
- **GPU validation drivers broken by the TU split** (ADC-346): `docs/validation/diocotron_gpu.cpp` and
  `diocotron_amr_gpu.cpp` compile `python/system.cpp` / `amr_system.cpp` standalone, but after the
  ADC-335 split those TUs delegate to the `build_block_*` / `build_amr_*` seams now living in per-transport
  sub-TUs -- so the drivers failed to LINK on a GH200 build (`undefined reference to
  adc::detail::build_block_compressible_rusanov`, ...). CI never builds these drivers, so it went unseen
  until a ROMEO nvcc run (the split + the `optnone` factories COMPILE cleanly under nvcc; only the link
  was missing the sub-TUs). `docs/validation/CMakeLists.txt` now compiles the seam sub-TUs into both
  drivers (same list as `_adc` / `adc_runtime_*`), and `parity181.sbatch` / `diocotron_mpi.sbatch` stage
  `diocotron_amr_gpu.cpp` (referenced by the shared CMakeLists since ADC-320). Validation-only; no library
  or hot-path change.
- **AMR seed fine patch persisted without refinement** (ADC-324): the compiled/native mono-block AMR
  builder (`build_amr_compiled`) always allocated a central seed fine patch on the explicit/imex path,
  even when `set_refinement` was never called. With the `1e30` "no refinement" threshold the build-time
  regrid tags nothing and the zero-tag regrid is a deliberate no-op, so the seed survived as a single
  un-chopped fine box pinned to rank 0 of the coarse dmap (`n_patches() == 1`), dead weight that starved
  MPI strong-scaling (rank 0 carried its coarse boxes plus the whole fine patch). The seed is now
  allocated only when refinement is configured (`refine_threshold < 1e30`): no refinement gives a
  mono-level hierarchy (`n_patches() == 0`, coarse distributes cleanly), and the refined path is
  unchanged. Regression test: `test_amr_seed_no_refine`. Follow-up to ADC-319.
- **Stale negative control in the compiled positivity-floor test** (ADC-341): ADC-324 made
  `set_refinement(1e30)` a mono-level hierarchy, so the `python/tests/test_amr_compiled_positivity_floor.py`
  section (1) assertion that the UNFLOORED `.so` run blows up on the spike became vacuous (the coarse-only
  grid diverges in neither branch), reddening the `gate-python` CI job. Mirrors the ADC-324 fix already
  applied to the native sibling test: drop the `unfloored-blows-up` control and keep the compiled-facade
  contract (floor accepted + floored run finite); the load-bearing property stays covered by
  `tests/test_positivity_floor.cpp` and the native test's refined C/F interface, and that the floor rides
  the loader by the same test's dmax==0 marshalling check and multi-block routing. Test-only; no library
  or ABI change.
- **Backend-blind DSL compile cache** (ADC-186): recompiling a `production` model onto an explicit
  `so_path` where an `aot` artifact was already loaded via dlopen in the same process re-served the
  stale aot handle (`add_native_block: adc_native_abi_key missing`), since the dynamic loader caches
  handles by path. `compile()` now keeps an in-process registry of the backend written to each path
  and redirects an explicit `so_path` already held by another backend to a distinct
  `<base>.<backend>.so` sibling, so dlopen reloads a fresh handle. The out-of-source cache was
  already keyed by backend; the three compile facades (`HyperbolicModel`, `Model`, `HybridModel`)
  share the redirect. Regression test: `test_compile_cache_backend`.
- **DSL production/AOT loaders now compile with MPI** (ADC-319): the `backend="production"` and
  `backend="aot"` model `.so` were compiled without `-DADC_HAS_MPI` even when `_adc` is built with
  MPI, so `comm.hpp` fell back to its serial stubs (`n_ranks()=1`, `my_rank()=0`) inside the loader.
  Any distributed layout built in the loader then collapsed to a single owner on every rank: an
  `AmrSystem(distribute_coarse=True)` replicated the whole coarse transport on all ranks (no MPI
  strong-scaling). `dsl.py` now re-bakes `-DADC_HAS_MPI` plus the module's MPI include dir (exposed as
  `_adc.__has_mpi__` / `__mpi_include__`), leaving the MPI symbols undefined to resolve at load against
  the libmpi already loaded by `_adc`/mpi4py (no second libmpi linked, like the Kokkos runtime). The
  MPI seam enters the loader cache key (`mpi=on|off`). Measured on ROMEO (hyqmom15 diocotron, N=256,
  cmg=64, 16 boxes): per-rank coarse box count drops from 16 to 4 at np=4 (the base now distributes),
  and ms/step falls from a flat 2554 to 1962. Serial builds are unaffected (no flag, bit-identical).
- **`bench scaling_amr` broken by the AMR TU split** (ADC-347): `bench/scaling_amr` compiles
  `python/amr_system.cpp` (which calls the `build_amr_block_*` / `build_amr_compiled_*` factories) but,
  after ADC-335/336 moved those into per-transport seam TUs, was never updated to link them, so
  `bin/scaling_amr` failed to LINK (`undefined reference to adc::detail::build_amr_block_exb`, ...) and
  the non-required `bench` job had been red since ADC-335. The six `amr_{block,compiled}_*.cpp` seam
  sources are now linked into `scaling_amr`, mirroring the `adc_runtime_amr` object library. Build-graph
  only; no behavior, API, or numerics change. Bench-side parallel to ADC-346.

## [0.2.0] - 2026-06-16

### Added

- **Pointwise projection on AMR** (ADC-312): `add_compiled_model(AmrSystem)` now accepts a model
  declaring `m.projection` (ADC-177, e.g. HyQMOM relaxation15). The projection `U <- project(U, aux)`
  is applied per level at the end of each macro-step, after the reflux and cascade (cell-local and
  idempotent, so conservation is preserved). Opt-in: a model without the projection trait is
  bit-identical to the historical trajectory. Previously the AMR path rejected such models.
- **AMR checkpoint/restart** (ADC-65): `AmrSystem.checkpoint(path)` / `restart(path)` write and read
  a bit-identical npz (per-level conservative state with fine patches, per-level phi as the multigrid
  warm-start, the fine hierarchy and the clock), replacing the old `NotImplementedError`. Restart
  imposes the saved fine layout via the new `set_hierarchy` (no re-clustering, no re-prolongation).
  Mono-block mono-rank for now: MPI np>1, multi-block and `regrid_every>0` are rejected explicitly.
  Adds the append-only level accessors `n_levels` / `n_vars` / `level_state` / `level_potential` /
  `set_hierarchy`.
- **Model-declared named aux fields** (ADC-70, phase 1, Cartesian `System`): a model declares
  persistent auxiliary fields with `m.aux_field("name")` (read in a formula via `aux.extra_field(k)`);
  `sim.set_aux_field(block, name, array)` and `sim.aux_field(block, name)` set and read them (up to
  four). The `Aux` POD stays trivially copyable and device-clean, named components are static (never
  rewritten by `solve_fields`), and a model with no named field is cache/hash-identical to before;
  `B_z` / `T_e` stay on their dedicated setters.
- **Generic 2D moment-hierarchy generator** (ADC-164): the new `adc.moments` module derives the full
  M->C->S->closure->C'->M' algebra (binomial transform plus standardization) over the DSL AST, so a
  user supplies only the closure (S -> standardized moments of order N+1). Ships `build_moment_model`,
  `gaussian_closure` (Levermore), `lorentz_sources` (Vlasov-Lorentz) and `moment_indices` /
  `moment_names`; `robust=True` adds differentiable smooth floors and `exact_speeds=True` wires HLL
  speeds via autodiff plus numeric eig.
- **SSPRK3 time integration on AMR** (ADC-64): `AmrSystem.add_block(..., time="ssprk3")` (or
  `adc.Explicit(ssprk3=True)`) runs a 3-stage Shu-Osher SSPRK3 per subcycled level, staying exactly
  conservative across coarse-fine boundaries by refluxing the effective flux
  `Feff = 1/6 F0 + 1/6 F1 + 2/3 F2`. The default stays forward Euler (bit-identical); rejected
  explicitly on the compiled `.so` paths and with `imex`.
- **First-order forward Euler explicit method** (ADC-174): `adc.Explicit(method="euler")` selects the
  `ForwardEuler` stepper on `System.add_block` (native and `backend="production"`), for
  first-order-reference fidelity; `ssprk2` stays the default. The frozen-ABI AOT path rejects `euler`
  (pointing to the production/native backends) rather than silently ignoring it.
- **Explicit signed wave speeds in the DSL** (ADC-83): `m.wave_speeds(x=(smin,smax), y=(smin,smax))`
  declares signed face speeds directly, unlocking `riemann="hll"` for models with no pressure
  primitive (moment systems, isothermal). Without `set_eigenvalues` the Rusanov/CFL bound derives
  from `max(|smin|,|smax|)`; an `add_equation` guard rejects `riemann="hll"` with no emitted speeds.
- **HLL wave speeds from the flux Jacobian** (ADC-86/87): `m.wave_speeds_from_jacobian(x=, y=,
  eig="numeric"|"fd", blocks=)` emits exact HLL speeds as the spectrum extremes of the flux Jacobian
  (new device-clean header `adc::real_eig_minmax`: closed form for N<=2, Hessenberg plus Francis QR
  otherwise, Gershgorin outer-bound fallback). `x/y=None` autodiffs the declared flux; the `Abs`
  autodiff node is now differentiable, so smooth `max(x,eps)` floors are Jacobian-able.
- **Positivity floor limiter** (ADC-76): `adc.FiniteVolume(positivity_floor=...)` imposes a density
  floor on reconstructed face states, falling back locally to the source-cell average (conservative,
  first-order) on a violating face. `floor<=0` is bit-identical to before; it is threaded through the
  Cartesian, polar and cut-cell kernels and the compiled-block ABI, and rejected explicitly on the
  paths that do not support it (prototype JIT, AOT/production, AMR).
- **Opt-in HLL wave-speed cache** (ADC-199): `adc.FiniteVolume(wave_speed_cache=True)` evaluates
  `model.wave_speeds` once per cell and direction and reuses it as the face bound instead of
  recomputing on every face and RK stage (measured speedup on moment hierarchies). Off by default,
  bit-identical to OFF in the no-slope path, and rejected outside `riemann="hll"` plus explicit time.
- **Spectral Poisson FFT variant** (ADC-175): a new elliptic kind `solver="fft_spectral"` (from
  `System.set_poisson`, `adc.EllipticSolver`, listed in `adc.capabilities()`) reuses the periodic
  single-rank FFT plumbing of `"fft"` but with the continuous Laplacian symbol `-(kx^2+ky^2)`, exact
  on sinusoids. The discrete `"fft"` and `"geometric_mg"` solvers are unchanged.
- **Native Windows (MSVC) support** (ADC-99/100/136/144): `adc_cpp` compiles and imports natively on
  Windows without WSL2. A portable dynamic-loading layer `adc::dynlib` (`LoadLibraryW` /
  `GetProcAddress`), an `ADC_EXPORT` macro (`__declspec`), an MSVC-aware ABI key, and
  `std::numbers::pi` for `M_PI` cover the runtime; the DSL `production` backend compiles a model to a
  `.dll` with `cl` and runs bit-identical to the brick path (shared Kokkos, `_adc.lib`). All `_WIN32`
  branches are dead off Windows.
- **`System.dt_hotspot(name)` CFL diagnostic** (ADC-182): returns `{w, i, j}` for the global cell
  that dominates a block's transport CFL bound, to locate a collapsing dt without an external scan.
  On-demand and off the hot path (`step` / `step_cfl` stay bit-identical), a two-pass device-clean
  reduction with an MPI all-reduce.
- **Find-or-fetch Kokkos** (ADC-263): if no Kokkos install is found, CMake downloads and builds a
  release tarball verified by SHA256, so a plain `cmake -B build` works without a pre-installed
  Kokkos. Overridable via `ADC_KOKKOS_FETCH_VERSION` (default 4.4.01) and `ADC_KOKKOS_FETCH_SHA256`.
- **Eigenvalue witness in the projection DSL** (ADC-289): `dsl.eig_max_im(rows)`, `dsl.eig_lmin(rows)`
  and `dsl.eig_lmax(rows)` build a small dense matrix from moment expressions and return a scalar
  Expr from its spectrum via `adc::real_eig_minmax` (max imaginary part as a complex-eigenvalue
  witness, or the real-part bounds). The codegen emits a named device-clean functor (no extended
  lambda), so `m.projection` can express a branchless "if a moment matrix has a complex eigenvalue,
  correct it" rule (unblocks the native relaxation15 projector, ADC-275). Additive: the existing
  expression set and `m.projection` (ADC-177) are unchanged.

- **Documentation rebuilt around a Diataxis navigation** (ADC-248): Getting started, Tutorials,
  Concepts (11 pages, ADC-249), How-to (12 task pages), Simulation, AMR, Running, Advanced topics,
  Reference and Development sections, plus a Quickstart page and an internal documentation style
  guide (ADC-250). Pages are kebab-case.
- **Embedded C++ reference** in the Sphinx site via doxysphinx (ADC-149), with a modern Doxygen
  theme and full dot diagrams (ADC-239).
- **Documentation-as-code tooling**: per-page `docmap.toml` with owner and freshness plus an
  example harness (ADC-147), a doc taxonomy and a stack ADR (ADC-146), and CI doc lanes (a light PR
  lane and a weekly heavy lane with lld and ccache, ADC-151/225).
- **Quality tooling / static analysis** (ADC-105): dedicated CI workflow `.github/workflows/quality.yml`,
  off the PR critical path (weekly Sunday cron + `workflow_dispatch` + `quality` label). Five
  *informative* (non-blocking) jobs: `clang-format` (`.clang-format`), strict warnings
  (`ADC_ENABLE_WARNINGS`), `clang-tidy` (`.clang-tidy`), ASan+UBSan sanitizers (`ADC_ENABLE_SANITIZERS`,
  `ci-warnings`/`ci-asan` presets) and CodeQL. CMake options OFF by default (empty `adc_dev_options`
  target), so `ci.yml`, local builds and `adc_cases` are unchanged. See `docs/QUALITY_TOOLING.md`.
- **Fuzzing, coverage and developer automation** (ADC-113): invariant-checked libFuzzer harnesses in
  `fuzz/` (Box2D, Berger-Rigoutsos clustering, `real_eig_minmax`; option `ADC_BUILD_FUZZING` +
  `ci-fuzz` preset, clang), gcov/gcovr coverage (`ADC_ENABLE_COVERAGE` + `ci-coverage` preset),
  Python ruff lint (`[tool.ruff]`), opt-in pre-commit hooks (`.pre-commit-config.yaml`: clang-format
  and ruff at commit), and two more `quality.yml` jobs (`fuzz`, `coverage`) on the same weekly
  informative cadence; the default build stays bit-unchanged.
- **Repository health and GitHub hygiene**: BSD-3 `LICENSE` and license declaration, `CONTRIBUTING`,
  `SECURITY`, `.gitattributes`, PR and issue templates (ADC-223/224/244/246), a root-hygiene guard
  (ADC-169), Dependabot weekly Actions bumps (ADC-117), a release workflow that turns a `v*` tag
  into a GitHub Release from this changelog (ADC-234), and `docs/VERSIONING.md` with the bump rules
  (ADC-232/237).
- **Shared C++ test harness** (`test_harness.hpp`, `bench/common.hpp`, ADC-215); a Schur-split test
  with an AMR guard on `add_block` and auto-skip without Kokkos (ADC-207).
- **CI**: `ci.yml` split into parallel `gate-cpp` / `gate-python` lanes with `mold` and path
  routing (ADC-171).
- **Moment-model documentation**: a step-by-step HyQMOM tutorial, a moments-and-closures concept
  page, a moment-models reference, and `adc.moments` (`build_moment_model`, `gaussian_closure`,
  `lorentz_sources`) added to the Python API reference (autodoc).
- **Generic pointwise post-step PROJECTION hook** (ADC-177): `m.projection([...])` on the DSL side
  (C++ trait `HasPointwiseProjection`, compiled like the flux/source), applied by `System` at the
  end of every whole macro-step (never per RK stage) on the valid cells of each block -- replaces
  the per-cell Python callback (`aot` and `production` backends; `prototype` and
  `target="amr_system"` reject it explicitly). Adds `dsl.sign(x)` (branch-free mask selections,
  differentiable). See `docs/DSL_API.md` section 5.

### Changed

- **Kokkos is the only on-node backend** (ADC-263): the standalone OpenMP path is removed (the
  `ADC_USE_OPENMP` option, `find_package(OpenMP)`, the mutual-exclusion check and every `#pragma omp`
  or manual host loop); `ADC_USE_KOKKOS` defaults ON and is fatal if disabled. Serial runs through
  Kokkos Serial, threads through Kokkos OpenMP, GPU through Kokkos Cuda/HIP, all chosen at Kokkos
  install time (`Kokkos_ENABLE_*`). The `for_each_cell` seam `#error`s without `ADC_HAS_KOKKOS`; the
  standard is pinned to C++20. The DSL loaders (`compile_aot` / `compile_native`) build against Kokkos
  and the `.so` cache is keyed on Kokkos state.
- **AOT DSL backend compiles at native optimization flags** (ADC-201): `compile_aot` (and the hybrid
  AOT path) dropped the hardcoded `-O2` (about 1.48x slower) for the native `_dsl_optflags()`
  (default `-O3 -DNDEBUG`, overridable via `$ADC_DSL_OPTFLAGS`); only the prototype JIT stays at
  `-O2`. An `aot-optflags` marker in the `.so` cache key prevents serving a stale `-O2` binary.
- **Hybrid AOT models build with the Kokkos toolchain** (ADC-103): since the Kokkos-only switch,
  `HybridModel.compile(backend="aot")` uses the native Kokkos compiler and flags (with macOS
  `-undefined dynamic_lookup`) and raises a clear `ADC_KOKKOS_ROOT` error when no Kokkos is visible;
  the prototype JIT stays pure-host.
- **Leaner DSL codegen** (ADC-200): `emit_cpp_*` now emit only the primitives transitively live in
  each method (no dead closure or its `sqrt`), values bit-identical; an opt-in `hoist_reciprocals=True`
  hoists `1/x` once for a recurring conservative denominator and turns those divisions into products
  (off by default, since it changes rounding).
- **Mixed relative/absolute stop criterion for GeometricMG** (ADC-202): `System.set_poisson(...,
  abs_tol=0.0)` adds an absolute residual floor, so the stop test becomes
  `residual <= max(rel_tol*r0, abs_tol)` with an early exit when `r0` is already under the floor
  (avoids over-solving an already-converged off-step `solve_fields`). `abs_tol=0` keeps the historical
  relative behavior bit-identical; inert for the FFT solver.
- **Higher default QR cap in `dense_eig`** (ADC-195): the default iteration cap of `real_eig_minmax`
  rises from 30 to 100 so near-degenerate companion blocks (HyQMOM 5x5, about 42 iterations) converge
  instead of falling back to Gershgorin (which over-estimated the wave speed about 9x); a new optional
  `fallback` out-parameter reports when the fallback fires.
- **Overflow-safe index arithmetic in PoissonFFT** (ADC-286): the hot indexing (row offsets,
  eigenvalue scaling, transpose buffers) widens to `size_t` before multiplying, removing an int*int
  overflow past INT_MAX on large multi-rank grids (CodeQL). Numerically neutral, bit-identical under
  INT_MAX.
- **Documentation translated to English**: the whole Sphinx site (ADC-228), the root design guides
  and this changelog (ADC-241), the remaining French docs (ADC-245), `CONTRIBUTING` / `SECURITY` /
  `DOC_QUALITY` (ADC-227), and a restructured English README with a translation glossary
  (ADC-119/236/218).
- **BREAKING (C++)**: facade parameters regrouped into a POD struct, source-incompatible for C++
  callers; the Python API is unchanged (ADC-214).
- **Docs version single-sourced** (ADC-233): the docs build derives the version from
  `project(VERSION)` in `CMakeLists.txt` (`scripts/build_docs.sh` injects the Doxygen
  `PROJECT_NUMBER`, `docs/sphinx/conf.py` reads it), so Doxygen and Sphinx no longer drift.
- **Portability (LLP64 / Windows data model)**: `long` to `int64_t` in `mesh/` and the `box_hash`
  key, removing undefined `<< 32` shifts (ADC-209/216).
- **Internal C++ cleanups**: rule of five on owning types (ADC-212), bit-exact factorizations of
  duplicated FV/MG and Newton-Jacobian code (ADC-213), residual extended device lambdas converted
  to named functors (ADC-210), hardened runtime guards (ADC-211), and a coding-standards and
  comments audit (ADC-124/125).
- **Code comments and Python docstrings translated to English** (ADC-272): all of `include/adc/**`,
  `python/adc/**` (including the pybind bindings) and the `CODE_DOCUMENTATION_CONVENTION` guide; the
  published `/cpp/` Doxygen reference and the Python autodoc now render in English. Code structure is
  byte-identical; codegen-template strings and cross-TU dispatch tokens are kept verbatim.

### Fixed

- **GPU device-clean EB path**: `aux_comps()` is now `ADC_HD` so it can be evaluated inside the
  embedded-boundary device kernels (`load_aux<aux_comps<Model>()>` in `spatial_operator_eb.hpp`).
  nvcc rejected the constexpr `__host__` call from a `__host__ __device__` kernel (#20013-D), which
  broke the CUDA build of the magnetized EB diocotron on GH200 (ADC-306). Host builds unchanged
  (`ADC_HD` is empty off nvcc).
- **Periodic theta ghosts in the polar Schur source coupling**: the polar condensed-Schur stepper
  and the polar Krylov solver now force the azimuthal (theta) ghosts of phi periodic instead of
  honoring the caller's four-face Dirichlet BCRec. The theta=0/2pi seam was filled by odd reflection
  (ghost = -phi), injecting a dipole radial-momentum kick that grew like O(1/h) and diverged the
  polar Hoffart run near t~0.01. A step with the System-style Dirichlet BCRec is now bit-identical to
  the canonical theta-periodic step.
- **Stale phi ghosts on the direct FFT Poisson path** (ADC-175): `PoissonFFTSolver::solve()` now does
  a periodic `fill_boundary` on the phi ghosts after the solve; the direct solver wrote only valid
  cells, so a centered grad-phi for an electric source read stale domain-edge ghosts (wrong Ex on the
  boundary ring with `solver="fft"`).
- **Ghost-width guard on the EB and polar FV operators** (ADC-221): the structural
  `require_reconstruction_ghosts<Limiter>` check (ADC-163) now also runs at the entry of
  `assemble_rhs_eb` and `assemble_rhs_polar`, which reused the same reconstruction stencil with no
  input validation (bound `Limiter::n_ghost`, host-only; no sane production caller triggers it).
- **Non-circular spectral check in `check_model`**: `check_model` now bounds `max_wave_speed` by the
  spectral radius of the full dense flux Jacobian (central differences, independent of any `blocks=`
  partition), catching a `set_wave_speeds_from_jacobian` partition that silently fails to bound the
  spectrum (under-estimated CFL); tunable via `jac_rtol` / `jac_atol`. Also corrects a misleading
  duplicate-index message.
- clang portability: `make_system_coupler` factory replaces CTAD on the `SystemCoupler` alias
  template, which GCC accepts but clang rejects (P1814) -- this broke every clang build of the
  coupling tests, surfaced by the new TSan job (ADC-302).
- Heap-buffer-overflow masked on disc geometry, caught by a ghost-width guard at FV operator entry
  (ADC-163).
- Comment rot: 10 inaccurate comments corrected (ADC-208); broken sister-solver links in the docs
  index.
- **Docs (Pages) build**: `suppress_warnings = ["docutils"]` set unconditionally in
  `docs/sphinx/conf.py` so `sphinx -W` no longer fails on autodoc-rendered Doxygen `@param`
  docstrings (the meaningful broken-reference and toctree checks stay strict).
- **`master` gate unblocked** (ADC-281): `test_dispatch_tags` greps the dispatch-registry error
  fragments, which ADC-272 (#105) had translated to English in `dispatch_tags.hpp` without updating
  the test -- reconciled the grepped fragments (`unknown limiter` / `unknown Riemann flux` /
  `unsupported` / `polar`).
- **Python test message-assertions reconciled with ADC-272 translation** (ADC-283): ~26 assertions
  across 21 `python/tests/` files grepped now-translated French error fragments (masked by the
  gate-python swallow, ADC-112). Reconciled to assert language-stable substrings (echoed values,
  quoted tokens) so they survive the ongoing translation; also reconciled the order-sensitive
  `ABI incompatible` greps with the source's `incompatible ABI` -- the `test_dsl_production{,_amr}.py`
  asserts and the C++ `test_amr_native_loader.cpp` guard (`test_native_abi_std.py` rides along under
  this umbrella) -- plus the stale forward-order wording in comments and hints (`bindings.cpp`,
  `__init__.py`, `dsl.py`, `python/CMakeLists.txt`).
- **MPI FFT rejection test** (ADC-282): `test_mpi_system_fft` greps the translated
  `fft solver unsupported under MPI` message, completing the post-ADC-272 test fixups alongside
  ADC-281/283.
- **CI hardening**: the weekly quality jobs are bounded against cold-cache OOM (serial instrumented
  `ci-asan` / `ci-coverage` builds, `timeout-minutes` caps, and a single Ninja pool for the heavy
  Kokkos translation units; ADC-284/290), CodeQL is scoped to the adc sources to drop 187
  vendored-Kokkos findings (ADC-285), seven orphan DSL tests are registered in ctest with a
  self-contained `ADC_KOKKOS_ROOT` (ADC-104), and a `no-ai-authors` guard rejects AI authorship or
  co-author trailers.

## [0.1.0] - 2026-06-10

First numbered release (previously `0.0.1`, never exposed; Doxygen/Sphinx already announced
0.1.0, and this bump aligns the CMake single source with it).

### Added
- `pip install .` via scikit-build-core: module in site-packages, no PYTHONPATH; backends via
  environment variables (`ADC_USE_KOKKOS=ON Kokkos_ROOT=... pip install .`).
- `find_package(adc)`: install/export rules for the header-only core (`ADC_INSTALL`).
- `adc.__version__`, `adc.doctor()` (full diagnostic), `adc.set_threads()` /
  `adc.parallel_info()` / `adc.has_kokkos()`, `_adc.kokkos_is_initialized()`.
- CMake presets (`python`, `python-parallel`, `serial`, `parallel`, `mpi` plus the `ci-*` series
  used by CI: single source of the flags).
- Conda environment (`environment.yml`) plus `scripts/setup_env.sh` (per-platform toolchain
  pinned in the env) plus `pixi.toml` (reproducible cross-platform lockfile).
- `scripts/kokkos_openmp_conda.sh` (Kokkos Serial+OpenMP in the conda env, ~2 min);
  `scripts/build_docs.sh` (lint + Sphinx + Doxygen + site in one command); machine profile
  `Tools/machines/romeo/romeo_adc.profile.example`.
- Extended ABI key of the DSL production path: `kokkos=` and `stdlib=` tokens (divergences
  previously undetected), as a preprocessor literal (insensitive to ELF interposition).
- DSL runtime toolchain guards: build compiler baked in (`__cxx_compiler__`) and preferred over
  PATH, standard probe (`c++23`->`c++2b`), pre-dlopen module/header guard (including on cache HIT),
  compilation errors surfaced with the compiler output.

### Changed
- `ADC_BUILD_TESTS` follows `PROJECT_IS_TOP_LEVEL`: a FetchContent consumer no longer builds the
  test suite.
- `import adc` works without numpy (`adc.dsl` is lazy, with a targeted error at use).
- DSL `.so` cache: machine-aware key (arch + optflags) and fingerprint of the Kokkos install
  (`KokkosCore_config.h`); a different Kokkos invalidates the cache.
- Tests: ctest labels (`core`/`mpi`) plus timeouts; memory guard `-O0` plus the ninja pool
  extended automatically to any target compiling `system.cpp`/`amr_system.cpp` (39 objects).
- `pybind11` taken from the environment before any FetchContent; ccache auto-detected;
  `ADC_PY_LTO` option (OFF by default).

### Fixed
- Three real user bugs of the DSL production path: conda PATH compiler rejecting `-std=c++23`,
  stale module leading to a cryptic dlopen `symbol not found`, `CalledProcessError` without the
  compiler output.
- `find_package(Kokkos)` failed on macOS against a Kokkos OpenMP (libomp hints set before
  KokkosConfig's `find_dependency(OpenMP)`).
- Docs: pip/PYTHONPATH contradiction, phantom Catch2 mention, `$KOKKOS_ROOT` undefined in the
  tutorial, stale numpy claim, inconsistent versions (0.0.1 vs 0.1.0).
