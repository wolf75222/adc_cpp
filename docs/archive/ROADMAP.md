# Roadmap

Living list of what is done and what remains, by intent.

> Scope note. This repository (`adc_cpp`) is the LIBRARY: core + physical bricks +
> Python bindings. The C++ EXAMPLES (`examples/`), the figure SCRIPTS (`scripts/`), the
> TUTORIALS and the APPLICATIVE validation tests (diocotron rate, ROMEO runs, physical
> invariants, convergence order of the models) now live in `adc_cases`. Several
> entries below cite files (`examples/...`, `test_euler`, `test_diocotron_*`,
> `magnetic_euler_poisson.hpp`, etc.) that are NOT in this repository: these are results
> obtained / to obtain on the `adc_cases` side, kept here as project memory. The tests
> present in THIS repository are the ~53 core ctests + ~16 Python tests (cf. `tests/` and
> `python/tests/`).

## Done

### Numerical core

- Finite volumes Godunov, Rusanov / HLL / HLLC fluxes, MUSCL reconstruction (NoSlope / Minmod /
  VanLeer), SSPRK2 / SSPRK3 integration, Strang / Lie splitting.
- Poisson: geometric multigrid (V-cycle, red-black GS, on-device) AND direct spectral FFT,
  behind the `EllipticSolver` concept.
- In-house mesh stack: `MultiFab` / `BoxArray` / `DistributionMapping` / `Geometry`,
  `fill_boundary`, physical BCs, `for_each_cell` seam serial / OpenMP / Kokkos.

### Model and coupling

- Core AGNOSTIC to the model: no scenario named in the core, only generic
  bricks (state, transport, source, elliptic right-hand side) composed into `CompositeModel`
  by the `model_factory`. Named compositions (diocotron, Euler-Poisson, two-fluid...)
  live on the application side (`adc_cases`). Coupling via the `aux` channel: base contract
  `(phi, grad_x, grad_y)` (3 components), now EXTENSIBLE (`load_aux<NComp>`, `aux_comps<Model>()`)
  for additional fields that the model declares via `n_aux`: `B_z` (comp 3, provided by
  the user) and `T_e` (comp 4, derived `p/rho` from a fluid block). Bit-exact backward compat if `n_aux=3`.
- Symbolic DSL at PROTOTYPE level, complete and tested: formulas -> C++ brick (`emit_cpp_brick`)
  + source + elliptic right-hand side + CSE; JIT path (`.so`, virtual `IModel`,
  `System.add_dynamic_block`); AOT path (`compile_or_jit(mode="compile")`,
  `System.add_compiled_block` via `compiled_block_abi.hpp`); NATIVE compiled block at parity with
  `add_block` (`add_compiled_model` / `dsl_block.hpp`, validated bit-identical on CPU/Serial by
  `test_compiled_model_parity`). MUSCL reconstruction by choice (`none`/`minmod`/`vanleer`) for
  the dynamic block.
- Roe flux in the core (`RusanovFlux` / `HLLFlux` / `HLLCFlux` / `RoeFlux`); elliptic
  operator with VARIABLE permittivity `eps(x)` on the core side (`GeometricMG::set_epsilon`) AND wired
  System/Python (`set_epsilon_field`), plus screened/Helmholtz and anisotropic operator; `VariableSet` /
  `VariableRole` now USED in inter-species couplings (`index_of(role)`) and emitted by
  the DSL on the generated bricks; reorganization `physics/` + `numerics/`.
- Two-fluid AP scheme (implicit Lorentz, Poisson reformulated `beta0`), isotropic dispersion
  validated (3.1%), AP bound at `omega_pe = 1e3`.
- MUSCL upwind continuity (anti-Gibbs) as an option; magnetic field: combined E+B Boris
  push (`tfap_boris`, cyclotron frequency exact to 0.00%, E x B drift preserved without
  secular growth), instead of the external Strang splitting.

### AMR

- 2-level and N-level mono-box reflux (`amr_step_multilevel_mf`), bit-identical to the reference
  Fab2D stack.
- N-level multi-patch (`amr_step_multilevel_multipatch`): coverage-aware reflux, routing
  to the parent box, validated on two axes at `0` exact.
- 2-level multi-patch reflux DISTRIBUTED MPI (`test_mpi_amr_multipatch`, np=1/2/4 bit-identical,
  coarse replicated + gather `average_down`/reflux).
- N-level multi-patch reflux DISTRIBUTED MPI (`amr_step_multilevel_multipatch` /
  `subcycle_level_mp`, `test_mpi_amr_multipatch3`, 3 levels with an intermediate level
  multi-box distributed whose parent of a fine patch falls on another rank; np=1/2/4
  bit-identical). Level 0 replicated, levels >0 distributed; ghost-fill and sampling of the
  coarse flux by `parallel_copy` when the parent is distributed, `average_down` and reflux by
  global coarse buffer + `all_reduce_sum`.
- Berger-Rigoutsos clustering + dynamic regrid; couplers `AmrCoupler` (mono-box) and
  `AmrCouplerMP` (multi-patch + regrid), conservative.

### Parallelism and tools

- Standalone OpenMP (deterministic vs serial, but **deprecated** in favor of Kokkos), MPI
  (bit-identical np=1/2/4, 7 `mpirun` tests in this repository), Kokkos backend (CPU Serial/OpenMP
  AND GPU Cuda, without hand-written CUDA). The CI runs 3 jobs: Release, MPI, Kokkos (Serial).
- GPU GH200: components validated SEPARATELY and bit-identical to the CPU (mono-grid System,
  AMR field ops, multi-GPU MPI halos, AOT backend of a DSL model). INTEGRATED validation
  AmrSystem + MPI + GPU now DONE (a single run, euler_poisson compiled on AMR multi-patch
  distributed, `exec=Cuda`, np=1/2/4 bit-identical `dmax=0`, mass conserved). REMAINS: the full-device
  perf (the integrated run does not scale, coarse replicated). Cf. `docs/GPU_RUNTIME_PORT.md`.
- Numerical validation (beyond bit-identical): order of the 5-point Laplacian, WENO5-Z order
  (`test_weno_convergence`), discretizations, IMEX/AP. The APPLICATIVE convergence orders
  (Euler isentropic vortex, MUSCL/Rusanov, Gauss law of the coupling, diocotron invariants)
  are measured on the `adc_cases` side.
- Python bindings of the lib (`python/`, module `pops`, `-DPOPS_BUILD_PYTHON=ON`): `pops.System`
  (block-by-block composition via `add_block`, `set_poisson`, `set_density`, `step`/`advance`/`step_cfl`,
  primitives `eval_rhs`/`get_state`/`set_state` + `pops.integrate.ssprk2_step`), the AMR
  composition `pops.AmrSystem` (mono-block, explicit; now shares the spatial operator of `System`:
  primitive reconstruction and HLLC/Roe flux via `advance_amr`, cf. `test_amr_spatial_parity`.
  The custom two-fluid AP integrator is not a generic brick but a scenario: it has
  left the core for `adc_cases/two_fluid_ap/`), plus the symbolic mini-DSL (`test_dsl_*` suite).
- Docs: README, ALGORITHMS, ARCHITECTURE (5 layers), CHOICES, BIBLIOGRAPHY, PERFORMANCE,
  two_fluid_ap, Doxygen + Sphinx (the tutorials and figures live in `adc_cases`).

## In queue

### Architecture hardening (design review)

From a review: the structural weakness is the mixing of discretization / storage /
execution, and a multi-patch AMR not yet thought out distributed. See
[ARCHITECTURE.md](ARCHITECTURE.md) (five-layer model, sections marked "target").

1. **Natively distributed multi-patch AMR (absolute priority).** Done for the 2-level:
   `amr_step_2level_multipatch` runs **really distributed** (`test_mpi_amr_multipatch`,
   np=1/2/4 **bit-for-bit identical**, mass conserved). The mono-box coarse is replicated
   (per-rank copy + local periodic fill), the fine patches distributed, `average_down`
   (covered overwrite) and reflux (border addition) come back up via two coarse buffers +
   `all_reduce_sum_inplace`. Along the way, a mono-rank bug fixed: the face-boxes of the fine fluxes
   were built on the **local** boxes with the **global** dmap (inconsistent sizes under
   MPI). The same latent bug fixed in `subcycle_level_mp`. **Done also for the N-level:**
   `subcycle_level_mp` / `amr_step_multilevel_multipatch` runs **really distributed**
   (`test_mpi_amr_multipatch3`, 3 levels, intermediate level multi-box distributed whose parent
   of a fine patch falls on another rank; np=1/2/4 **bit-for-bit identical**, mass conserved).
   Level 0 replicated, levels >0 distributed. The five points are resolved: (1) ghost-fill
   parent->child and (2) sampling of the coarse register by `parallel_copy` when the
   parent is distributed (local read when it is replicated); (3) `average_down` and (4) reflux
   by global coarse buffer + `all_reduce_sum_inplace`, applied to the local parent boxes;
   (5) coverage already global. The coupler `AmrCouplerMP` is also distributed: its
   aux injection `coupler_inject_aux_mb` goes through `parallel_copy` when the parent is distributed
   (level 0 replicated: local read), verified against the analytic np=1/2/4 by
   `test_mpi_coupler_inject`. Toward the final target: the SFC `load_balance` (Morton Z-order,
   `make_sfc_distribution`) is now WIRED on the distributed AMR and verified, not only
   tested as a serial algorithm: `test_mpi_amr_multipatch3` executes the 3-level step under
   Morton distribution and obtains `maxdiff = 0` vs the reference (rank 0) at np=1/2/4, imbalance
   1.000. The flux register is also RESTRICTED TO THE COARSE-FINE INTERFACE: its `all_reduce`
   no longer covers the whole coarse domain (`O(NX*NY)`) but the bounding box of the
   fine footprints (`O(interface)`), bit-identical (the cells outside the interface were null,
   skipped at application; np=1/2/4 `maxdiff = 0`). A dedicated design (read-only workflow)
   has settled the rest: the RESIDUAL collective gather is irreducible as long as the coarse
   is replicated (each rank must see the correction); removing it entirely (pure point-to-point
   register, zero collective) requires DE-REPLICATING level 0, which breaks the MG Poisson,
   `fill_periodic_local` and the local mass measurement of the tests for a marginal gain on a
   small coarse. Decision: **NO-GO on de-replication** for this case (diocotron / Euler-Poisson,
   base 32x32); the `O(NX*NY)` bottleneck is already eliminated without it. Remains, optional: `owner_rank` /
   `global_box_id` explicit per patch (today in `DistributionMapping` + `BoxArray` index),
   and de-replication only if one day the base level becomes large.
2. **Unified AMR engine.** Unified entry done: `advance_amr(m, LevelHierarchy&, dt)` + the
   type `LevelHierarchy`, verified facade-faithful in **2 and 3 levels** (`maxdiff = 0` vs the
   direct call, mass drift `< 1e-12`) and conservative (`test_advance_amr`). Promotion of the roles into
   types begun: `OwnershipPolicy` is a real alias of `DistributionMapping`;
   `FluxRegister` is a REAL TYPE (coarse register global-indexed on a region: `set`
   overwrites, `add` accumulates bounded, `gather` does the `all_reduce`), substituted for the four
   manual buffers of the reflux (2-level avg/ref, N-level avg/ref) bit-identical (np=1/2/4 `maxdiff=0`),
   contract frozen by `test_flux_register`. `CoverageMask` is also a REAL TYPE (coarse mask
   on a region: `mark(box)` marks a clipped fine footprint, `covered(I,J)` bounded), the
   "coverage" part of `CoarseFineInterface` (the anti-double-reflux mask), substituted for the three
   manual masks, bit-identical (np=1/2/4 `maxdiff=0`), contract frozen by `test_coverage_mask`.
   Remains: promote the remaining roles (`PatchRange`, the border routing of `CoarseFineInterface`,
   `SubcyclingSchedule`, `RegridPolicy`, still inlines in `subcycle_level_mp`) and fold into them
   the `amr_step_*` family (which encodes the case in the name).
3. **Split the elliptic.** Advanced: the `EllipticOperator` already exists separated
   (`poisson_operator.hpp`: `apply_laplacian`, `poisson_residual`, smoother); the
   `LinearSolver` is the `EllipticSolver` concept (MG, FFT); the MG = FFT identity is made
   STRUCTURAL and verified (`test_elliptic_operator` applies the same canonical operator to the
   two solutions, residuals ~1e-14). Done: `EllipticProblem` (coeff `eps`, BC `BCRec`, nullspace
   `nullspace_const` in one object, until now implicit) and `FieldPostProcess` (derivation convention
   `E = -grad phi` via `GradSign::Plus`/`Minus`, formerly the free function
   `coupler_grad_phi`) are named in `elliptic/elliptic_problem.hpp`. Structural refactor
   bit-identical: `eps = 1` remains descriptive (the stencil does not read it yet), the factory
   `make_elliptic_solver(EllipticProblem)` delegates to the `BCRec`, and `field_postprocess` reproduces
   identically the expression of the coupler. `test_elliptic_problem` proves bit-for-bit equality
   (`operator==` strict, not a tolerance). Remains out-of-scope as long as we want bit-identical:
   rewire the sites in form `/(2*dx)` (`amr_coupler`, `amr_coupler_mp`, `spectral_coupler`,
   `two_fluid_ap`), division that may differ at the last bit from the multiplicative form `*cx`.
4. **Explicit memory API.** Reductions `sum` / `norm_inf` done; explicit `sync_host()` /
   `sync_device()` now laid out. The `for_each.hpp` seam now carries
   `for_each_cell_reduce_sum` and `for_each_cell_reduce_max`, alongside `for_each_cell` and
   `device_fence()`: under Kokkos a real `parallel_reduce` (MDRangePolicy, `Kokkos::Sum`
   / `Kokkos::Max`, blocking on the host side so without prior `device_fence()`); in serial and
   under OpenMP a sequential host loop. `sum` and `norm_inf` (multifab.hpp / mf_arith.hpp)
   call this seam per local fab then aggregate; the two `device_fence()` that protected
   their host loop have disappeared (absorbed by the reduction), the other `device_fence()` (host access
   elsewhere) remain.
   FP consequence: `sum` is no longer bit-identical to the host loop UNDER KOKKOS (the sum per
   tile reassociates the floating addition, non-associative in IEEE754). In serial and under OpenMP
   `sum` remains exact: we keep deliberately the sequential host loop for OpenMP, because
   `reduction(+:)` would reorder the sum per thread and break the "OpenMP identical
   to serial" guarantee of the repo. `norm_inf` remains EXACT everywhere (a max of absolute values, without
   rounding and invariant under reordering). `Kokkos::Sum` is deterministic per tile (no
   floating atomics), so two `sum` on unchanged data give the same bit
   (idempotence, key for `test_fill_boundary/sum_unchanged`). Contract locked by
   `test_reduce` (sum_constant exact, varied sum in relative gap < 1e-10, norm_inf strict,
   idempotence).
   Explicit data residency: `sync_host()` / `sync_device()` (free functions of the seam
   `for_each.hpp` + `MultiFab` methods) encode the access INTENT, where the coherence
   relied on scattered `device_fence()` without saying which residency to make valid. Under unified
   memory (`Kokkos::SharedSpace`) `sync_host()` is a targeted `device_fence()` (drain the kernels
   in flight before a host access) and `sync_device()` a no-op (the host writes are already visible
   to the device): the behavior remains BIT-IDENTICAL to the old code (at most one fence, never a
   copy). `MultiFab::set_val` calls `sync_host()` instead of a bare `device_fence()` (same
   semantics, named intent). It is deliberately scaffolding under unified memory; the
   value is the abstraction: a future NON-unified path (separate host/device buffers) will wire into it
   a directional `deep_copy` and the per-fab residency tracking, WITHOUT touching the operators (all
   the host accesses already go through `sync_host()`), exactly as `for_each_cell` isolates the
   CPU -> GPU transition of the call sites. Idempotence and no-op locked by `test_sync_residence`.
5. **Separate the three ghost families** into named testable bricks. Largely done:
   `fill_physical_bc` (BoundaryCondition, tested alone `test_physical_bc`), `fill_boundary`
   (GhostExchange, tested `test_mpi_fillboundary`), `mf_fill_fine_ghosts_*`
   (AMRBoundaryInterpolation) are already separated; `fill_ghosts` is only an explicit composition
   of (1) + (2). Remains: raise the coarse-fine into a named first-level helper.
6. **Thin CouplingPolicy (DONE).** Hierarchy, regrid and diagnostics taken out of the AMR
   couplers into named components: `coupling/amr_level_storage.hpp` (`AmrLevelStack<Level>`,
   levels + aux storage, aux wiring and reallocation), `coupling/amr_regrid_coupler.hpp`
   (`amr_regrid_finest`, Berger-Rigoutsos in free function template, `comm.hpp` included
   explicitly for `n_ranks()`), `coupling/amr_diagnostics.hpp` (`amr_mass`,
   `amr_max_drift_speed`). `AmrCoupler` / `AmrCouplerMP` only orchestrate now
   (`sync_down -> compute_aux -> step`, `regrid()` delegated); the injection primitives
   remain `detail::` helpers. Structural extraction bit-identical: equivalence
   `max|dUc| = 0` and mass conservation to rounding unchanged
   (`test_amr_coupler`, `test_amr_coupler_mp`).
7. **Numerical validation suite (DONE).** Bit-identical does not prove correctness.
   IN THIS REPOSITORY (core): order of the 5-point Laplacian (`test_poisson_convergence`, Dirichlet +
   periodic + nullspace); AP limit quantified over several decades of stiffness
   (`test_ap_limit`); order 5 of the WENO5-Z reconstruction (`test_weno_convergence`, measured
   5.00); conservative AMR diffusion (`test_amr_diffusion`); exact reflux (`test_flux_register`,
   `test_coverage_mask`). ON THE `adc_cases` SIDE (physical model required): Euler isentropic
   vortex (VanLeer order), MUSCL / Rusanov orders, discrete Gauss law of the coupling, diocotron
   invariants (mass, maximum principle, enstrophy, field energy, angular momentum) and the
   complex eigenvalue (Re rotation + Im growth). The tests `test_euler`,
   `test_muscl_convergence`, `test_gauss_law`, `test_diocotron_*`, `test_amr_reflux`,
   `test_amr_coupler*` historically cited here now live in `adc_cases`.

### Magnetized physics (Hoffart target)

- **Combined E+B Boris push: DONE.** `tfap_boris` advances the momentum under E
  AND B in one symmetric step (electric half-impulse, full magnetic rotation,
  half-impulse), instead of the external Strang splitting (rotation around the whole electrostatic
  step). Wired in `TwoFluidAP2D::step` (only the magnetized case changes; at `wc = 0`
  the push reduces exactly to `tfap_lorentz`, so all the `B = 0` tests remain
  bit-identical; at `E = 0` it is the pure rotation, so the cyclotron remains exact to 0.00 %).
  `test_two_fluid_boris` freezes the three properties: rotation conserving `|m|` under B alone,
  reduction to the electric impulse without B, and above all the discrete `E x B` fixed point
  `m* = h cot(theta/2) (Ey, -Ex)` preserved with constant gyration radius (no secular
  growth of the energy). `test_two_fluid_ap_amplitude` (B enabled) validates the push in the
  self-consistent AP stack.
- Remains: tensorial AP reformulation under strong field.

### Hoffart reproduction (arXiv:2510.11808): internship objective

> APPLICATIVE work: the milestones M1 to M4 below are done ON THE `adc_cases` SIDE
> (`examples/` binaries, figure scripts, applicative tests, ROMEO runs). The cited files and
> tests (`diocotron_column_amr.cpp`, `diocotron_highorder.cpp`,
> `integrator/magnetic_euler_poisson.hpp`, `test_magnetic_euler_poisson`, etc.) are NOT in
> this repository. Kept here as project memory; the core (`adc_cpp`) provides only the bricks.

The paper (magnetic Euler-Poisson, structure-preserving FEM) validates in Section 5 the
**diocotron** instability by its growth rates, in the **magnetic drift limit** (eq 2.7:
`v_dr = -∇φ×Ω/|Ω|²`, `∂tρ + ∇·(ρ v_dr) = 0`, `ωd = ρα/|Ω| = ωp²/ωc`). This limit is the
composition of `ExB` + `BackgroundDensity` bricks ("diocotron" scenario on the `adc_cases` side).
No AMR in the paper: the objective is to reproduce with OUR solver then add to it
our AMR, then SAMRAI.

- **M1 (in progress): numerical vs analytic growth rate.** Pipeline built and validated.
  The analytic (`diocotron_growth.hpp`, Petri/Davidson-Felice eigenvalues) already gives back the
  rates of the paper (`γ₃≈0.772, γ₄≈0.911, γ₅≈0.683`, peak at mode 4). The nonlinear simulation
  (`diocotron_column`, geometry `0.15:0.20:0.40 = 6:8:16`) reproduces **qualitatively**
  the mode 4 instability (linear growth then saturation). `scripts/validate_diocotron_growth.py`
  fits the numerical rate on the linear phase and normalizes it by `ωD = ρ̄/(2π)`. Resolution
  study (`docs/fig_diocotron_reproduction.png`): `γ_norm = 0.52 → 0.54 → 0.55` at `n =
  128/192/256`, monotone growth toward `0.911` but **limited by the numerical diffusion of the ring
  edge** (thin ring of 6 to 13 cells). Quantified conclusion: on a uniform grid, reaching
  `0.911` demands a very high resolution -> directly motivates the AMR (M2).
- **M2: with our AMR (mono-rank, DONE).** `examples/diocotron_column_amr.cpp`: hollow column
  (ring + circular conducting wall `r=0.40` carried by the multigrid, `AmrCouplerMP` now accepts
  the `active` predicate) on 2-level AMR, refining the **ring edge** (crown tag
  `[0.13,0.22]`, Berger-Rigoutsos regrid), amplitude diagnostic of the mode `l` of phi at `r0`, mass
  drift `~3e-15`. Same binary for the two branches (`refine=0/1`, identical numerics).
  Result (mode 4, `docs/fig_diocotron_amr_vs_uniforme.png`): at EQUAL COARSE BASE (96), the AMR
  **triples the rate** (`γ_norm = 0.38` vs uniform `0.12`) by refining the transport at the edge, for
  1.8x the cells; it is on a BETTER rate/cell curve than the uniform (0.38 at 16k
  cells vs ~0.22 by uniform interpolation).
- **M2b: MULTI-LEVEL Poisson (DONE, mode `ml=1`).** Step M2 capped the rate because the
  **Poisson stayed solved on the coarse** (the coupler injects the coarse aux to the patches). The mode
  `ml` (`diocotron_column_amr <out> <nc> <nsteps> <refine> <l> <ml>`) assembles a composite density
  on the fine grid (coarse prolonged + fine patches overwritten), solves a fine `GeometricMG` on it,
  then restricts the potential toward the coarse (gradient for `auxc`) and keeps the direct fine gradient
  (`auxf`). At base 96 (same 16 392 cells) the rate rises from `γ_norm = 0.38` to `0.42`: the coarse
  Poisson was indeed capping the rate. HYPOTHESIS CONFIRMED.
- **M2b-conv: convergence verified by base sweep** (`docs/fig_diocotron_ml_convergence.png`).
  By raising the base (96 -> 128 -> 160, effective edge resolution 192 -> 256 -> 320), the
  multi-level AMR converges toward the uniform at SAME effective resolution, for ~43 % of the cells:

  | effective resolution | AMR `ml` (γ_norm / cells) | uniform (γ_norm / cells) | AMR/unif cells |
  |---|---|---|---|
  | 192 | 0.42 / 16 392 | 0.50 / 36 864 | 44 % |
  | 256 | 0.526 / 28 352 | 0.526 / 65 536 | 43 % |
  | 320 | 0.563 / 44 192 | 0.565 / 102 400 | 43 % |
  | 448 | **0.592** / 82 808 | 0.577 / 200 704 | 41 % |

  At base >= 128 the AMR rate COINCIDES with the uniform at equal effective resolution, for less than
  half the cost; at eff 448 it EXCEEDS it (0.592 vs 0.577) for 41 % of the cells. The convergence
  toward `0.911` is clear and monotone (0.42 -> 0.526 -> 0.563 -> 0.592) but SLOW: the numerical
  diffusion of the ring edge (limit of M1) demands a much higher resolution to reach
  `0.911`. This is the SCIENCE of step 3 of the hero-run: push the resolution (the multi-level AMR
  reaches it for ~41-44 % of the cells of the uniform). Reaching `0.911` at full scale demands the
  distributed AMR driver (step 2 de-replication + hardening of the primitives, cf. `docs/HERO_RUN_AMR.md`).
- **M2b-conv-HR: sweep pushed beyond eff 448 (high-resolution instability corrected).**
  The sweep capped at eff 448 because the simulation went to `nan` above: the geometric
  multigrid DIVERGED at the embedded edge on the fine grid (coarse correction inconsistent with the
  circle re-discretized per level, V-cycle spectral radius > 1, ERRATIC depending on the alignment of the
  circle). The warm start propagated the divergence from one step to the next -> `phi` then the field to `nan`.
  It was NEITHER the time step (already capped), NOR the density floor (the density stays bounded
  in `[1e-3, 1]` during the divergence; only `phi` explodes). Fix: `GeometricMG::solve_robust`
  (cf. [HERO_RUN_AMR.md](HERO_RUN_AMR.md)), which runs the standard V-cycle (BIT-IDENTICAL when it
  converges or stagnates) and, ONLY in case of true divergence (final residual > initial residual),
  hardens the GS smoothing (sticky) and restarts from cold until it becomes contracting again. The 8 runs
  recorded above remain bit-for-bit identical (verified); the elliptic suite stays green.
  The sweep then climbs without `nan` up to eff 1024 (uniform AND AMR `ml`, mass `~1e-14`):

  | eff | AMR `ml` γ (lin / sat) | uniform γ (lin / sat) | AMR / unif cells |
  |---|---|---|---|
  | 448  | 0.631 / 0.591 | 0.632 / 0.577 | 82 808 / 200 704 = 41 % |
  | 512  | 0.664 / 0.582 | 0.650 / 0.579 | 104 632 / 262 144 = 40 % |
  | 640  | 0.663 / 0.588 | 0.670 / 0.574 | 162 144 / 409 600 = 40 % |
  | 896  | 0.695 / 0.570 | 0.699 / 0.561 | 314 340 / 802 816 = 39 % |
  | 1024 | 0.706 / 0.565 | 0.706 / 0.558 | 409 008 / 1 048 576 = 39 % |

  Two measurements of the rate. `sat` = window relative to the peak (historical method of the table above).
  `lin` = FIXED PHYSICAL window in linear phase (`validate_diocotron_growth.py --window 5,14`,
  new option), more robust to COMPARE resolutions. The `sat` measurement CAPS toward ~0.58
  then declines beyond eff 448: this is NOT the physics but a window bias (the saturation rollover
  stiffens with resolution and contaminates the slope). The `lin` measurement, which isolates the
  exponential regime, CONTINUES its MONOTONE climb toward `0.911` (0.63 -> 0.65 -> 0.67 -> 0.70 -> 0.71 from eff
  448 to 1024, uniform as AMR; trend robust to the choice of window, `--window 6,16` gives 0.55 ->
  0.63, same climb). The AMR `ml` FOLLOWS the uniform at ~39-40 % of the cells up to eff 1024: the promise
  M2b (same physics, < half the cost) holds at scale. The numerical blocker that blocked the sweep
  is lifted; reaching `0.911` remains a matter of even higher resolution (distributed hero-run).
- **M2b-recon: the true blocker toward `0.911` is the reconstruction ORDER, not the resolution.** M1
  attributed the ceiling to the numerical diffusion of the ring edge; the transport ran in `NoSlope`
  (first-order reconstruction, the most diffusive). By switching to MUSCL order 2 (`VanLeer`, limited
  slope, option `recon=1` of `diocotron_column_amr`; 2 ghosts) the edge stays sharp and the rate RISES
  STRONGLY at FIXED resolution:

  | eff | `NoSlope` (lin 5,14) | `VanLeer` (lin 5,14) | `VanLeer` (lin 4,11, clean expo. phase) |
  |---|---|---|---|
  | 256 | 0.561 | 0.760 | **0.864** (95 % of 0.911) |
  | 512 | 0.650 | 0.753 | 0.851 |

  At eff 256 already, `VanLeer` reaches `γ_norm ~ 0.86` in the clean exponential window (`--window
  4,11`), i.e. ~95 % of the analytic `0.911`, against `0.56` in `NoSlope`: +0.30 of rate for the SOLE
  rise in order, without touching the resolution. The `VanLeer` rate is almost FLAT in resolution (0.864 ->
  0.851 from eff 256 to 512): it is already CONVERGED in reconstruction, which directly CONFIRMS
  the M1 hypothesis (the ~0.58 ceiling of the `NoSlope` came from the scheme diffusion, not the physics).
  Stable (TVD limiter, no `nan`) and conservative (`~1.9e-14`), uniform AND AMR `ml` (base 320 with 66
  patches included). The default `recon=0` (`NoSlope`) remains bit-for-bit identical to the recorded runs. The
  ~5 % remaining toward `0.911` (initial transient, explicit Euler order 1 time integration,
  embedded representation of the edge) are the fine target; the bulk of the gain is acquired at modest cost.
- **M2b-HO: high precision path WENO5-Z + SSPRK3 (high order space AND time), error vs analytic.**
  The research (Hoffart paper, classical methods: Jiang-Shu, Borges WENO-Z, Gottlieb-Shu-Tadmor)
  confirms the two levers: order 5 reconstruction AND order 3 time (forward Euler biases a mode in
  exponential growth, unstable on the imaginary axis). `examples/diocotron_highorder.cpp`:
  uniform diocotron WENO5-Z + SSPRK3, Poisson RE-SOLVED at each RK stage (`solve_robust`). We compare
  to the paper (modes 3/4/5 vs analytic 0.772 / 0.911 / 0.683, relative error). The analytic of the repo
  (`diocotron_growth.hpp`, Petri/Davidson-Felice) is INSENSITIVE to the smoothing of the profile (mode 4 = 0.9118
  at the sharp profile), so 0.911 is indeed the target of the sharp step we simulate, and the normalization
  (`omega_D = rho_bar/2pi`) is consistent analytic/sim.
  Result (paper window, eff 256, WENO5+SSPRK3): `gamma_norm` mode 3 = 0.838 (+8.5 %), mode 4 =
  0.985 (+8.1 %), mode 5 = 0.730 (+6.9 %). So the high order makes mode 4 pass from 0.56 (under-evaluated,
  `NoSlope`+Euler, too diffusive) to ~0.98 (OVER-evaluated), on the RIGHT side (like the coarse mesh of the paper,
  0.935), but NOT yet < 1 %. The overshoot is UNIFORM (~7-8.5 %) on the three modes, FLAT in resolution
  (eff 256 ~ eff 512 ~ eff 1024) and FLAT in reconstruction order (WENO5 ~ VanLeer): so it is NEITHER
  the scheme NOR the resolution, but the absence of a clean exponential PLATEAU (the initial transient declines
  directly toward saturation, without a clear linear phase to read the eigenmode), tied to the
  CARTESIAN staircase geometry (conducting wall + ring edges not aligned) where the paper uses a
  geometry MATCHING the edge (structure-preserving FEM, exact curved disk). Reaching < 0.5 % like the
  paper thus demands a MATCHING conducting edge (cut cells / polar grid), next work item;
  the high-order reconstruction and integration, themselves, are in place and verified
  (`test_weno_convergence`: order 5.00).
  CONFIRMATION multi-measurements (ROMEO sweep `diocotron_highorder_hero`, modes 3/4/5 x eff 256/512/1024,
  paper windows, R2 = 1.00): `gamma_norm` overshoot UNIFORM and FLAT IN RESOLUTION, mode 4 =
  0.985 / 0.988 / 0.987, mode 3 = 0.838 / 0.850 / 0.853, mode 5 = 0.730 / 0.731 / 0.729 (the resolution
  does NOT close the gap). Three diagnostics rule out the simple causes: (a) delta sweep: the
  LINEAR LIMIT (delta -> 0) rises to ~+27 % instead of dropping -> not a nonlinear contamination,
  and the apparent agreement at delta=0.1 was a compensation by the saturation; (b) DIMENSIONLESS ratio
  `gamma / |Re(omega)|` (normalization-independent, via the complex eigenvalue `diocotron_eigenvalue`)
  = 0.31 measured vs 0.331 analytic -> ~5 % is a STRUCTURAL distortion of the eigenvalue, ~3 % a
  shift of normalization `omega_D`; (c) WENO5 ~ VanLeer -> not the order. ROBUST verdict: the cause is
  the cartesian GEOMETRY (staircase wall + symmetry 4 of the square breaking the rotation invariance of the
  circular problem), INDEPENDENT of the resolution and the scheme. The < 1 % path is thus a MATCHING
  edge (cut-cell Shortley-Weller, or polar grid like the semi-Lagrangian diocotron methods),
  not more resolution. Physical invariants green over the whole run (exact mass, enstrophy = measure
  of diffusion, energy/momentum ~conserved, maximum principle): the transport itself is faithful.
  UPDATE (May 2026): the cut-cell Shortley-Weller edge has been IMPLEMENTED and validated (`test_cut_cell`:
  order 2, Poisson error 3459x lower) and it does NOT change the rate (gamma identical to 3 digits,
  cut vs staircase). The staircase wall was thus NOT the blocker: the mode is too far from the wall
  (image effect `(0.44)^8 ~ 1e-3`). Remains the STRUCTURAL distortion (b) of the cartesian eigenvalue
  (symmetry 4 grid/mode) plus the normalization. Complete and single account: `docs/DIOCOTRON_GROWTH_RATE.md`.
- **M3: full magnetic system (eq 2.4, DONE on the `adc_cases` side).** Beyond the drift
  limit: compressible Euler + energy + Poisson + Lorentz force `m × Ω`. The architecture of the
  core was already ready (bricks `CompressibleFlux` + `PotentialForce`/`GravityForce` +
  `GravityCoupling`); only the cyclotron rotation `m × Ω` was missing, added APPLICATIVELY
  (the file `magnetic_euler_poisson.hpp` cited here lives in `adc_cases`, not in this repository).
  `magnetic_rotate` (EXACT rotation of the momentum,
  `ρ` and `E` unchanged, conserves `|m|`) + `MagneticEulerPoissonCoupler`, Strang splitting
  around `Coupler<EulerPoisson>` (½ rotation, transport+electrostatic SSPRK2 with Poisson
  per stage, ½ rotation). The exact rotation is unconditionally stable: ASYMPTOTIC-PRESERVING
  scheme, the time step remains governed by the hydro CFL and NOT by the cyclotron
  frequency `ω_c = |Ω|`. `test_magnetic_euler_poisson` (60/60) proves: rotation `ρ`/`E` bit-for-bit
  conserved and `|m|` preserved; at `Ω=0` the step is bit-for-bit the bare `Coupler` (the whole
  Euler-Poisson path tested is preserved); the fixed point of the Strang map converges at ORDER 2
  (ratio `4.00`) toward the E×B drift `v = (-∂_yφ, ∂_xφ)/Ω`, so the full system REDUCES to the
  drift limit (M1/M2) when `Ω` grows. Demo `examples/magnetic_diocotron.cpp`: charge band
  under the full system, initialized on the drift manifold; runs stably at large `Ω`
  (hydro CFL), mass conserved to rounding (`~4e-11`), gas energy quasi conserved on the drift
  (`docs/anim_magnetic_diocotron.gif`). The quantitative reproduction of the rate at `0.911` in the full
  system is limited by the diffusion (same observation as M1) and targets the hero-run.
- **M4: SAMRAI.** Port the diocotron onto the SAMRAI AMR (FetchContent + adapter).

Hero run ROMEO (`romeo/`): SLURM scripts ready for the large-scale diocotron on GH200,
hybrid **MPI + Kokkos/CUDA** (1 MPI rank per H100, the `for_each_cell` kernels on GPU; the
232 H100 multi-node by Infiniband). It is the real "full machine" on `armgpu` (OpenMP+MPI
is the separate CPU mode on `x64cpu`). **UNIFORM grid, no AMR**: the binary `diocotron_mpi`
uses `SpectralCoupler` (Poisson FFT, bands), so brute force (push the uniform resolution
until resolving the ring edge and reaching 0.911). A DYNAMIC AMR hero-run is a distinct
objective (convergence of the distributed reflux done + AMR coupler ported multi-GPU + column benchmark,
cf. M2); the uniform run serves as a quantified reference to measure the gain of the AMR.
`diocotron_hero.sbatch` (run) + `diocotron_scaling.sbatch` (strong/weak scaling) +
`romeo/README.md` (build + submission).

Hero run DYNAMIC AMR (the true goal of uniform vs AMR comparison on ROMEO): it is the
convergence of several bricks and the biggest remaining piece. At hero scale, the coarse
8192^2 CANNOT be replicated on each rank -> a **decomposed coarse** is needed (multi-box
distributed) + **distributed MG Poisson** on it (`GeometricMG` already distributes a multi-box coarse
via `DistributionMapping(ba.size(), n_ranks())`, but the diocotron has a MONO-box coarse -> to
decompose) + **reflux without replication** (our distributed reflux assumes level 0 replicated;
the levels >0 are already distributed via `parallel_copy` + gather, to extend to level 0) +
**distributed regrid** (BR clustering + redistribution of the patches) + the whole in **Kokkos/CUDA**.
In other words, the **de-replication of the coarse (objective B)**, declared NO-GO for a SMALL
coarse, becomes REQUIRED again at hero scale. Clean path: M2 first (AMR on the column,
mono-rank, science) to quantify the cell gain, then assemble the distributed driver (B +
MG/regrid distributed + GPU), then compare the two hero runs. Design-first recommended
(conservation gate `maxdiff=0` np=1/2/4 at each step).

The staged PLAN of this distributed AMR hero-run is written in
[HERO_RUN_AMR.md](HERO_RUN_AMR.md): the key insight is that a 2-LEVEL diocotron
(coarse replicated + 1 distributed fine level) touches NONE of the two hard blockers (the regrid
tags level 0 REPLICATED, so no gather-tags; no de-replication as long as the coarse
stays moderate), which allows delivering steps 0 (CPU driver) and 1 (GPU port) by
relying on the distributed reflux already proven, and pushing back the hard risk (distributed background
solver + gather-tags, step 2) until it becomes necessary.

### Performance

- OpenMP region consolidated above the level loop of the multigrid (the only lever
  identified for MG-dominated loads).

### Comfort

- MUSCL on the two-fluid momentum; GIF of the multi-patch coupler; additional tutorials
  if needed.
