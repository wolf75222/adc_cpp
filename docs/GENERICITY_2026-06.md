# Genericity -- design decisions and remaining limits (audit 2026-06)

Response to the three audits of 2026-06-09 (MEETING_AUDIT / HARDCODED_GENERIC_AUDIT /
CPP_FULL_REPO_AUDIT): this document summarizes WHAT WAS CHANGED, WHY in this form, and what
remains explicitly NOT generalized. Guiding rule throughout: **default behavior
strictly bit-identical** (the new capabilities are declared opt-ins, never
silent changes).

## 1. step_cfl / step_adaptive: time-step policy with aggregated bounds

- **OPTIONAL traits of the model contract** (core/physical_model.hpp):
  `stability_speed(U, aux, dir)` (lambda* replacing max_wave_speed in the CFL -- stability !=
  accuracy, the Riemann solvers keep max_wave_speed), `source_frequency(U, aux)` (mu in
  1/s, bound WITHOUT h: the "second CFL" of the meeting), `stability_dt(U, aux)` (direct admissible step,
  cfl NOT applied). Device reductions (max / max / min-via-inverse) in spatial_operator.hpp,
  closures in block_builder.hpp, conditional forwarding in CompositeModel.
- **Effective substep**: all bounds apply to `stride*dt/substeps` (same factors as
  the historical transport bound).
- **GLOBAL bounds**: `System::add_dt_bound(label, fn)` -- one HOST evaluation per step (Python
  callback acceptable HERE, never per cell), for multi-block coupling / Schur / scheduler / user
  ramp. `step_adaptive` honors them too (clamp of the macro-step, conservative).
- **Diagnostic**: `System::last_dt_bound()` names the ACTIVE bound of the last step_cfl
  ("transport:<block>" / "source_frequency:<block>" / "stability_dt:<block>" / "global:<label>" /
  "degenerate").
- **DSL**: `m.stability_speed(expr)` / `m.stability_dt(expr)` COMPILED like flux/source
  (GPU/MPI production, no per-cell callback). Strict fallback: `max(abs(eigenvalues))`.
- **Constants**: floor 1e-30 named `kCflSpeedFloor` (core/types.hpp), shared System/AMR.

## 2. IMEX -> SourceImplicitBE: options and Newton diagnostics

- The path stays EXACTLY ForwardEuler(transport without source) + LOCAL backward-Euler on the
  source (per-cell Newton). The documentation (IMEX/SourceImplicit) now NAMES it
  explicitly and denies the IMEX-RK family; alias `adc.SourceImplicitBE = adc.SourceImplicit`.
- `NewtonOptions {max_iters, rel_tol, abs_tol, fd_eps}` (implicit_stepper.hpp): defaults = the
  historical constants (2 / 0 / 0 / 1e-7) -> path (2a) bit-identical, ZERO cost. Active tolerances
  -> instrumented path (per-cell stop on ||F||_inf <= abs_tol + rel_tol*||F0||_inf,
  detection of non-finite residual and degenerate/non-finite pivot via solve_dense -> bool).
- OPT-IN diagnostics (`newton_diagnostics=True`): per-cell scratch + reductions + all_reduce ->
  `sim.newton_report(block)` = {enabled, converged, max_residual, max_iters_used, n_failed}, aggregated
  over the substeps of the last advance. The offending cell (i, j) is NOT located (an
  arg-max device reduction would be a separate work item; the report gives the worst residual and the count).
- Plumbing: adc.IMEX / adc.SourceImplicit (kwargs newton_*) -> System::add_block -> build_block ->
  AdvanceImex*. Non-default options OUTSIDE imex or on a .so backend (ABI not carried) -> explicit
  rejection, never a silent ignore.

## 3. Flux: explicit domain of validity, HLL aligned

- HLLC/Roe documented EULER 2D ONLY (+ aliases `EulerHLLCFlux2D` / `EulerRoeFlux2D`); the entropy
  fix of Roe is the NAMED constant `kRoeEntropyFixFraction = 0.1`, documented Euler/Roe-specific.
- `hll` (generic with signed waves, requires wave_speeds) is now also routed by the AMR
  (dispatch_amr_block + dispatch_amr_compiled, same requires-gate as System) and documented everywhere
  (Spatial / FiniteVolume / system.hpp / Sphinx). Visible test:
  `adc.FiniteVolume(limiter="minmod", riemann="hll", variables="primitive")` on the isothermal 3-var
  (test_fv_hll_minmod, System + AmrSystem + explicit rejections).

## 4. Named couplings: multi-box/MPI-safe

- add_ionization / add_collision / add_thermal_exchange iterate `local_size()` (pattern
  add_coupled_source) instead of `fab(0)/box(0)`: no-op on a rank without a box, ready for a multi-box
  System. Resolution by ROLE with canonical-index fallback is preserved (named
  couplings = "canonical fluid layout" sugar, documented).

## 5. Magnetic bricks exposed + latent bug fixed

- `adc.MagneticLorentzForce(charge)` and `adc.PotentialMagneticForce(charge)` exposed in
  adc.Model(...) (C++ factory "magnetic"/"potential_magnetic" already ready) and in the hybrid path
  (_native_to_brick, including CompositeSource via nested fields a.qom/b.qom).
- **Latent bug fixed**: System::add_block (native path) NEVER called
  `ensure_aux_width(aux_comps<M>())` -- a native model with n_aux=4 (magnetic brick) read B_z
  OUT OF BOUNDS of a 3-component aux (silent garbage values). The quantitative test
  (residual == q*B*m exactly on a uniform state) locks the wiring.

## 6. check_model: generic safeguards

- `dsl.Model.check_model(...)` (formulas, before compilation): flux/source/elliptic finite,
  finite/real eigenvalues, consistency wave_speeds <-> max_wave_speed, round-trip
  to_conservative(to_primitive(U)) ~= U, positivity Density / 'p', reproducible samples.
- `System.check_model(block)` (runtime, any backend): U finite, residual -div F + S finite, positivity
  by roles, round-trip of the model conversions (state saved/restored).

## 7. Numerical constants taken out of the kernels

- `kCflSpeedFloor` (1e-30, types.hpp), `kRoeEntropyFixFraction` (0.1, numerical_flux.hpp),
  `NewtonOptions.fd_eps` (1e-7), IMEX iters -> `NewtonOptions.max_iters`,
  Schur tolerances: `set_krylov(tol, max_iters)` on the cartesian (1e-10/400)
  and polar (1e-10/600) condensed steppers, exposed via `adc.CondensedSchur(krylov_tol=, krylov_max_iters=)`.

## 8. Outputs / checkpoint

- PLAN only (docs/IO_CHECKPOINT_PLAN.md): target API write/checkpoint/restart, minimal content of the
  checkpoint (t, macro_step, mesh, blocks, U, aux/B_z, params, stride/substeps), HPC constraints
  (not one file/process; HDF5 aggregated then parallel), split into 3 PRs.

## Wave 2 (intent correction: GENERALIZE, not just mark)

The first wave handled several paths not covered by an explicit REJECTION; wave 2
WIRES them (the rejection stays only where the architecture is not ready, and it is listed below).

1. **StabilityPolicy on AMR**: AmrRuntimeBlock and the mono-block hooks carry
   source_frequency/stability_dt; build_amr_block/_compiled route the CFL speed via
   stability_speed (trait); AmrRuntime::step_cfl AND the mono-block step_cfl aggregate the bounds
   (substeps/stride formulas identical to System); AmrSystem::add_dt_bound / last_dt_bound
   (parity with System, all_reduce_min). A DSL model m.stability_dt(...) compiled
   target='amr_system' therefore constrains the AMR step (test B4).
2. **Riemann capabilities**: HLLC and Roe are no longer Euler-only algorithms in disguise --
   `HasHLLCStructure` (pressure + wave_speeds + contact_speed + hllc_star_state) and
   `HasRoeDissipation` (d = |A_roe| dU, entropy fix included) let A MODEL provide its
   structure; the core applies the generic algorithm (F* = F_k + s_k (U*_k - U_k);
   F = 1/2(F_L+F_R) - 1/2 d). The canonical Euler 2D path remains the historical implementation
   bit-identical. System AND AMR gates widened. Proofs: hooks-Euler == canonical at 1e-13;
   HLLC isothermal 3-var preserves EXACTLY a stationary shear where HLL diffuses it.
3. **Newton**: damping (0,1], fail_policy none|warn|throw (host, after reductions; pure
   observer), offending cell (i, j) + component via encoded reduction (report + throw message).
   Non-Euler proof: 3-variable nonlinear relaxation converges under tolerances; NaN
   pathology on ONE cell -> throw with exact (i, j).
4. **DSL m.source_frequency(expr)**: emitted as `frequency(U, aux)` on the generated SOURCE
   brick (the optional contract of physics/source.hpp), forwarded by CompositeModel, aggregated
   by step_cfl ("source_frequency:<block>"). Requires m.source (explicit error otherwise).
5. **Schur roles in the ABI**: set_source_stage carries density/momentum_x/momentum_y/energy
   (stable role name OR variable name) + bz_aux_component; the cartesian stepper gains an
   explicit-component constructor (the canonical ctor DELEGATES to it, bit-identical);
   adc.CondensedSchur(density=..., momentum=(...), magnetic_field=...) forwards instead of
   rejecting. Defaults = canonical roles, bit-identical.
6. **IO v1**: sim.write(path, format='vtk'|'npz', step=), sim.checkpoint / sim.restart (npz,
   atomic write); bindings macro_step() / set_clock() / set_potential() -- the restart is
   BIT-IDENTICAL (including stride cadence via macro_step, and MG warm start via restored phi).
7. **adc.capabilities()**: single truth matrix (riemann x facade, time, stability_policy,
   poisson, schur, DSL backends, io); stale docstrings corrected (PolarMesh, polar Schur
   stage "does not plug in", Phase 3c perimeter of the AMR Schur, CondensedSchur "mono-rank").

## Wave 3 (balance: polar, couplings, DSL HLLC/Jacobian, AMR Newton, polar/AMR Schur, IO, multi-block)

Confirmation at the board (advisor): `dt <= dx/|lambda_max|  <=>  CFL = dt*lambda*/dx` -- this is
exactly the `stability_speed` trait (lambda* = STABILITY speed, distinct from max_wave_speed
which stays the Riemann solver speed). Wave 3 extends this policy everywhere it was
still missing.

1. **Polar: StabilityPolicy wired** (block_builder_polar.hpp): factories
   make_cfl_speed_polar / make_source_frequency_polar / make_stability_dt_polar (functors named
   PolarStabilitySpeed/PolarSourceFreq/PolarStabilityDt, device-clean); the polar branch of
   System::add_block installs them. A polar model declaring stability_speed / stability_dt /
   source_frequency therefore bounds the step exactly as in cartesian (same substeps/stride
   formulas, same last_dt_bound). Default without trait: historical max_wave_speed,
   bit-identical.
2. **CoupledSource.frequency(mu)**: a coupled source DECLARES its frequency (1/s); the bound
   `dt <= cfl/mu` applies to the MACRO-step (the couplings are applied once per macro-step,
   not per substep); reason "coupled_source:<label>". Plumbing System AND AmrSystem
   (add_coupled_source(frequency=, label=) -> coupled_freqs_; AmrRuntime::add_coupled_frequency;
   mono-block step_cfl aggregates it too). DSL: `dsl.CoupledSource(...).frequency(mu)` carried by
   CompiledCoupledSource. Default without frequency: no change. REFINEMENT (sec. 7): mu ALSO accepts
   an Expr (same block().role() + param() fields as the terms) -> PER-CELL frequency
   mu(U), emitted in bytecode (freq_prog_ops/args) and reduced (MAX) per cell at each step
   (CoupledFreqKernel, device-clean functor; all_reduce_max global; same reason). AMR: evaluated on
   the COARSE LEVEL of the input blocks. Constant = historical path, bit-identical.
3. **DSL emits the HLLC hooks**: `m.enable_hllc()` generates contact_speed + hllc_star_state FROM
   THE ROLES (Density/MomentumX/MomentumY[/Energy] required; the variables outside the fluid roles are
   treated as passive scalars advected at the contact speed -- generalization, not a Euler
   assumption). Requires 'p' declared (pressure/pseudo-pressure). CompiledModel.has_hllc;
   CompositeModel forwards contact_speed/hllc_star_state/roe_dissipation (concept-gates);
   riemann='hllc' accepted on a 3-var NON-Euler via the capability, explicit rejection without it.
   Proof: HLLC 3-var isothermal runs finite (test_v3_features D1).
4. **Analytic Jacobian of the source**: trait `HasSourceJacobian` (jacobian(U, aux, J),
   J[r][c] = dS_r/dU_c) used by BOTH Newton paths (historical and instrumented) in place of
   finite differences (if constexpr, zero cost without the trait). DSL: `m.source_jacobian(rows)`
   (n x n of expressions) emitted on the source brick; sugar `m.implicit_source(source, jacobian=)`.
   Proof: same root as the FD at 2.8e-17 (C++), same IMEX trajectory at 1e-9 (Python).
5. **Newton options CARRIED on native multi-block AMR**: AmrSystem::add_block accepts the
   newton_* kwargs; BlockSpec carries NewtonOptions; build_amr_block captures the options in
   imex_advance (the frozen iters=2 disappeared from the multi-block closures). Still REJECTED
   explicitly: mono-block AMR (closures of the historical coupler), .so loaders (ABI), and
   newton_diagnostics (the aggregated report stays System-only).
   **BALANCE (wave 3, NOT generalized points #1)**: the OPTIONS are now also wired in
   MONO-BLOCK (threaded on the AmrCouplerMP coupler: AmrBuildParams.newton_options ->
   build_amr_compiled -> cpl->step -> advance_amr -> subcycle_level_mp -> mf_apply_source_treatment ->
   backward_euler_source; default {} bit-identical). newton_diagnostics/newton_report is wired in
   native MULTI-BLOCK (NewtonReport per AmrRuntimeBlock as shared_ptr, reset at the head of the advance in
   AmrRuntime::step, max/sum aggregation + all_reduce identical to System::AdvanceImex; binding
   AmrSystem.newton_report(name) -> dict, exact shape of the System binding). Still rejected: the
   REPORT in mono-block (rejection at build, threading the report into the coupler subcycling would
   be invasive) and the .so loaders (options AND diagnostics: flat ABI, rejection at the Python facade).
   Proofs: test_amr_newton_full (mono finite options, multi newton_report consistent, no-default-change
   dmax==0, .so rejections) + test_v3_features section (B).
6. **Schur: roles + krylov configurable on polar AND amr-schur**: explicit-component constructors
   (the canonical ctor DELEGATES, bit-identical) + set_krylov(tol, iters) on
   PolarCondensedSchurSourceStepper (1e-10/600) and AmrCondensedSchurSourceStepper (coarse
   only); set_source_stage(density=, momentum_x=, momentum_y=, energy=, bz_aux_component=)
   resolved role-OR-name on the three facades. AMR: magnetic_field != B_z stays rejected (the
   fine composite reads B_z by Aux contract).
7. **set_conservative_state MULTI-BLOCK**: wired for the native models (the full state seeds the
   coarse level via coupler_write_coarse_state; proof: the seeded momentum
   advects from the 1st step). .so loaders: explicit rejection (no state path in the v1 ABI).
8. **Extended IO**: `sim.write(format='hdf5')` (h5py OPTIONAL, clear error otherwise);
   `AmrSystem.write` npz/vtk (COARSE fields of EACH block, by name, + phi + fingerprints of
   the fine patches; the multi-level fields = PR-IO-3); checkpoint/restart AMR = explicit rejection
   pointing to the plan (PR-IO-3).

## Wave 4 (polar HLL, multi-rank IO, AMR clock) -- audit 2026-06, POLAR + IO work items

Balance of the POLAR (section 3) and IO (section 10) work items of the plan, at an **honest perimeter** (what
is wired really is; what is not is documented with file:line, never masked).

1. **POLAR HLL wired** (`include/adc/runtime/block_builder_polar.hpp`, `make_block_polar`). The polar
   RHS `assemble_rhs_polar<Limiter, NumericalFlux, Model>` ALREADY carried the numerical flux as a
   TEMPLATE PARAMETER (injection point identical to the cartesian `build_block<Limiter, Flux>`) and
   calls `nflux(model, L, aux_L, R, aux_R, dir)` -- exactly the signature of `HLLFlux`. Wiring
   HLL was therefore a small wiring job: `make_block_polar` routes `riemann='hll'` toward
   `build_block_polar<Limiter, HLLFlux>`, GATED on `model.wave_speeds` (named functor, device-clean,
   same `requires` as `block_builder.hpp` make_block). The polar isothermal fluid
   (`IsothermalFluxPolar : IsothermalFlux`) inherits `wave_speeds` -> eligible; the scalar ExB
   (`ExBVelocityPolar`, no `wave_speeds`) -> CLEAR rejection. **Default `rusanov` strictly
   bit-identical** (separate branch, untouched). HLLC/Roe stay rejected (Euler 4-var, no polar energy
   flux brick). Facade: `adc.PolarMesh` + `adc.FiniteVolume(riemann='hll')`;
   `adc.capabilities()['riemann']['system_polar'] = ['rusanov', 'hll']`. Test:
   `python/tests/test_polar_hll.py` (rusanov reproducible, hll finite AND distinct from rusanov) +
   `test_polar_rejections.test_polar_rejects_hll_on_scalar_exb`.

2. **Polar theta splitting: EXPOSED (ADC-67, updates the "NOT exposed" decision above).**
   `adc.PolarMesh(..., theta_boxes=N)` splits the ring into N theta BANDS (each box covers the whole
   radius `[0, nr-1]` and an azimuthal band; `theta_boxes` must DIVIDE `ntheta` and stay
   `<= ntheta`). `theta_boxes=1` (default) = mono-box, STRICTLY bit-identical. Plumbing: `system.cpp`
   `Impl::index_boxarray` builds the BoxArray in bands (reuses the `theta_split` decomposition of the test
   `test_polar_schur_multibox`) + `DistributionMapping(ba.size(), n_ranks())` round-robin;
   `SystemConfig.theta_boxes` (append-only) + binding; validation at BOTH levels (`PolarMesh` on the
   Python side, `check_geometry` on the C++ side). multi-box MATRIX (honest):
   - **TRANSPORT polar multi-box OK**: `assemble_rhs_polar` iterates `local_size()` and the block residual
     (`PolarBlockRhsEval`) fills the ghosts via `fill_ghosts` COLLECTIVE (inter-box halos + theta
     periodic + r physical) BEFORE assembly -- no mono-box shortcut (already validated multi-box by
     `test_polar_schur_multibox`, theta-split 8 boxes). The host marshaling (`copy_state`/`copy_comp0`/
     `write_state` on the `Impl` side, and `set_density`) reconstructs/scatters the GLOBAL ring when
     `local_size()>1` (places each box at its global indices, like `state_global`) -- the store stays
     VERBATIM (the `local_size()<=1` path delegates as-is, bit-identical cartesian + mono-box polar).
   - **DIRECT polar Poisson mono-box only**: `ensure_elliptic_polar` raises a clear UPSTREAM error if
     `ba.size()!=1` (before the construction of the `PolarPoissonSolver`, at the 1st `solve_fields`/`step`/
     `potential`). The direct solver (FFT-in-theta + tridiag-in-r) requires complete theta lines + r columns
     on a box. Message: point to `theta_boxes=1` OR the tensor Schur stage.
   - **polar tensor Schur stage multi-box**: the C++ solver is already multi-box; `theta_boxes`
     now drives the splitting on the facade side.
   Mono-rank (the direct Poisson refuses MPI). Tests: `python/tests/test_polar_theta_boxes.py`
   (isothermal transport bit-identical theta_boxes=1/2/4; scalar ExB; divisibility + direct Poisson
   multi-box rejections; round-trip get/set state multi-box). `adc.capabilities()['geometry']`.

3. **MULTI-RANK System IO** (`python/system.cpp` + `include/adc/runtime/system.hpp` + bindings +
   `python/adc/__init__.py`). Finding: `copy_state` / `copy_comp0` / `potential` read `fab(0)`
   (valid on the owner rank -- mono-box, box 0 on rank 0 -- but OUT OF BOUNDS on a rank
   without a box). Added: collective GLOBAL accessors `density_global` / `state_global` /
   `potential_global` (global buffer filled at GLOBAL indices from the local fabs, then
   `all_reduce_sum_inplace` -> each rank holds the complete field; AMR reflux pattern). The
   write/read marshalings (`write_state` / `set_potential` / `copy_*`) are now
   guarded against `local_size()==0` (no-op / empty instead of UB). Facade: `sim.write` /
   `sim.checkpoint` do the collective gather (all ranks) then write the file only on rank 0
   (`_adc.my_rank()`/`n_ranks()` exposed); `sim.restart` reads the file (shared FS) and calls
   `set_state` / `set_potential` MPI-safe (rank 0 writes, others no-op) + `set_clock`. **Mono-rank
   bit-identical** (`state_global == get_state`, all_reduce = identity, box = complete domain):
   test `python/tests/test_io_multirank.py`. GUARANTEED SEMANTICS: under MPI np>1, `write`/`checkpoint`
   produce ONE file identical to the mono-rank (mono-box System), checkpoint/restart bit-identical.
   PARALLEL HDF5 (hyperslabs) = PR-IO-3. (No MPI harness on the pytest side -> np>1 coverage to validate
   in central; the gather reuses the pattern already validated by `test_krylov_solver_np*` /
   `test_schur_condensation_np*`.)

4. **AMR clock: macro_step() / set_clock()** (`include/adc/runtime/amr_system.hpp` +
   `python/amr_system.cpp` + `amr_runtime.hpp` + `amr_dsl_block.hpp` + bindings). Parity with System:
   `AmrSystem::Impl` carries an AUTHORITATIVE macro-step counter (incremented by step/step_cfl),
   `macro_step()` returns it, `set_clock(t, ms)` restores it AND pushes it to the CADENCE counter of the
   engine (regrid/stride): `AmrRuntime::set_macro_step` (multi-block) OR `set_macro_step` hook of the
   mono-block coupler (additive at the tail of `AmrCompiledHooks`, abi_key auto-bumped via ADC_HEADER_SIG).
   Prerequisite PR-IO-3, **useful alone** (stride cadence + clock restart). Test:
   `python/tests/test_amr_clock.py`.

5. **AMR checkpoint: IMPROVED rejection** (`python/adc/__init__.py`). A bit-identical AMR checkpoint is
   IMPOSSIBLE with the current ABI; a "coarse only" fallback would be LOSSY (density() = comp0 only, the
   multi-var momentum/energy not readable) AND not bit-identical (set_conservative_state
   seeds the coarse + prolongs, does not restore the fine patches). The `NotImplementedError` rejection is
   kept but its message NOW lists PRECISELY the 4 ABI gaps (read fine states per
   patch; read complete conservative state; serialize hierarchy+ownership; write fine states
   per patch). No coarse-restart opt-in: it would not be clean (lossy + re-prolongation).

## Points still NOT generalized (explicit, updated wave 3)

1. **AMR**: no `ssprk3`; coarse/fine assumes ratio 2; `set_poisson` limited to geometric_mg +
   rhs charge_density|composite. Newton options: WIRED in mono-block (threaded on the AmrCouplerMP
   coupler: step -> advance_amr -> subcycle_level_mp -> mf_apply_source_treatment ->
   backward_euler_source, default {} bit-identical) AND in multi-block; the .so loaders reject them
   still (flat ABI). newton_diagnostics/newton_report: WIRED in native MULTI-BLOCK (NewtonReport per
   AmrRuntimeBlock, reset at the head of the advance in AmrRuntime::step, max/sum aggregation + all_reduce
   identical to System); mono-block AMR and .so loaders = explicit rejection (the report is not threaded
   into the coupler subcycling nor carried by the ABI).
2. **AMR Schur Phase 4**: composite limited to 2 levels + ONE mono-box fine patch + mono-rank;
   multi-patch, > 2 levels, MPI, multi-block = Phase 4 (explicit rejection, perimeter documented
   in the header of the stepper). set_krylov AMR only drives the coarse stage.
3. **Polar**: flux **Rusanov AND HLL** (HLL wired since wave 4, cf. below), but
   **HLLC/Roe stay NOT wired** (assume n_vars==4 Euler with energy, without polar energy flux
   brick -> explicit rejection, make_block_polar). Direct Poisson mono-rank/mono-box (tensor
   Schur = multi-box). **theta splitting NOT exposed by the facade** (decision documented
   below): the polar transport (System.step) is itself MONO-BOX
   (`python/system.cpp` ctor Impl: `ba(std::vector<Box2D>{index_domain(c)})`, a single box;
   `set_source_stage` L.~1095-1103: the facade builds ONE box covering the ring, the theta
   splitting is only drivable at the level of the C++ API PolarCondensedSchurSourceStepper). Exposing a
   `theta_boxes` parameter on `adc.PolarMesh` would be a **lying facade**: the tensor Schur
   solver knows how to split theta, but the System (transport + direct Poisson) does not, and there is
   NO plumbing (split BoxArray + DistributionMap + fill_boundary halos of the polar transport)
   to make it multi-box. Nothing is therefore exposed; the blocker is documented, not masked.
4. **Aux**: extensible by CANONICAL LIST (ADC_AUX_FIELDS + AUX_CANONICAL Python mirror: phi/grad/
   B_z/T_e) **AND, since ADC-70 (phase 1), by NAMED field declared by the model**: `m.aux_field("name")`
   reserves a component of the aux channel starting from `kAuxNamedBase = 5` (after T_e), read in C++ via
   `aux.extra_field(k)` (POD array `Real extra[kAuxMaxExtra]`, device-clean; `load_aux<NComp>` loads it
   under `if constexpr (NComp > kAuxNamedBase)` -> zero codegen at the default, bit-identical). On the
   facade side: `System.set_aux_field(block, name, array)` / `aux_field(block, name)` (name -> comp
   resolution in the Python facade from `CompiledModel.aux_extra_names`, the C++ only manipulates indices
   via `set_aux_field_component`). STATIC persistent fields (re-applied after `ensure_aux_width`,
   like B_z); at most `kAuxMaxExtra = 4` per model. B_z / T_e stay on their dedicated paths (explicit
   rejection redirecting in `set_aux_field`). PERIMETER phase 1 = **CARTESIAN System**; the `.so`
   exports an optional symbol `adc_compiled_aux_extra_names` (self-description). FOLLOW-UP (phase 2): AMR
   (aux channel per level + regrid), polar (validation), custom halos per field, and name -> comp table
   on the C++ `System::Impl` side (resolution without the Python facade).
5. **Native ROLE-AWARE bricks (done)**: source.hpp (PotentialForce / GravityForce /
   MagneticLorentzForce) and elliptic.hpp (ChargeDensity / BackgroundDensity / GravityCoupling)
   now carry index MEMBERS (c_rho / c_mx / c_my / c_E, POD integers -> device-clean),
   resolved AT CONSTRUCTION (host) by model_factory.hpp (bind_variable_roles) via
   TR::conservative_vars().index_of(role). AUTOMATIC and transparent resolution (no new user
   parameter). Defaults = canonical fluid layout -> for any NATIVE transport (Euler /
   Isothermal / ExB, canonical roles) the resolved indices == the defaults -> STRICTLY bit-identical
   (wired in polar too, dispatch_model_polar). LIMIT: from the public API the native bricks
   compose only with these CANONICAL transports; a permuted layout meets a native brick
   only via a direct C++ path (blocker: `requires` detection of the binder + role registry).
6. **IMEX-RK (done, ARS(2,2,2))**: the IMEX-RK family now EXISTS -- `adc.IMEXRK(scheme="ars222")`
   wires the Ascher-Ruuth-Spiteri (1997) scheme, **order 2** (explicit transport L = -div F coupled to
   the stiff source implicitly by a staged tableau). gamma = 1 - 1/sqrt(2), delta = 1 - 1/(2 gamma);
   stiffly accurate tableaux (b == last row of A) -> U^{n+1} = last stage. On the C++ side:
   `detail::AdvanceImexRkArs222` (block_builder.hpp), advance PARALLEL to `AdvanceImex` -- it REUSES
   `BlockRhsEval<SourceFreeModel>` (transport), `backward_euler_source` (local implicit solve) and
   saxpy/lincomb, NO new device kernel; the stage-2 source contribution is recovered by the
   consistency relation `dt*gamma*S^(2) = U^(2) - base2` (no extra source kernel). PERIMETER =
   **cartesian System**: `time="imexrk_ars222"` is rejected explicitly on AMR, polar, the .so loaders
   (prototype/aot/production) and the Strang/Schur splittings (hyperbolic != Explicit). The default
   `adc.IMEX` (= SourceImplicitBE, local backward-Euler, order 1) stays the only local implicit scheme
   and is **UNCHANGED / bit-identical** (kind "imex" != "imexrk_ars222", distinct C++ paths). LIMIT:
   the IMEX-RK source is FULLY implicit (the stage consistency relation assumes a homogeneous
   solve) -> incompatible with a partial mask `implicit_vars`/`implicit_roles` (explicit rejection;
   for a per-component partial IMEX, stay on `adc.IMEX`). The analytic Jacobian
   (`m.source_jacobian`, wave 3) also improves the IMEX-RK stage solves.
7. **CoupledSource**: still explicit forward-Euler additive, fixed capacities (kCsMaxReg=32...).
   frequency(mu) now accepts a CONSTANT (historical path) OR an Expr -> PER-CELL frequency
   mu(U) in bytecode, reduced (MAX) at each step (cf. sec. 2; AMR: bound on the coarse).
   REMAINS: the AMR bound does not see a local underestimate of mu under a fine patch (evaluated on
   the coarse, assumed choice).
8. **Backends**: DECISION TAKEN (ADC-63) -- the default becomes `backend="auto"`: PRODUCTION
   (zero-copy native loader) when toolchain parity with the _adc module is established (module
   loadable + baked compiler + matching header signature), AOT otherwise (historical
   safe default, without module). Never silent: CompiledModel.backend tells what was built,
   CompiledModel.backend_auto_reason tells why; an explicit backend short-circuits the
   policy (unchanged). Perimeter: Model.compile / HyperbolicModel.compile (BLOCK models);
   the hybrid bricks keep their aot default (distinct pipeline, follow-up). **Tag registry FACTORED (done)**: the VALIDATION of the
   tags (limiters + Riemann fluxes) and the n_ghost are now a SINGLE SOURCE
   (include/adc/runtime/dispatch_tags.hpp: kLimiters / kRiemanns + validate_limiter / validate_riemann
   / limiter_n_ghost). make_block, dispatch_amr_block, dispatch_amr_compiled and make_block_polar
   validate FIRST via this registry (historical messages preserved, by context); their final tag
   throws become a registry/dispatch inconsistency guard. RESIDUE: the DISPATCH itself
   stays a per-call-site template if/else (the Limiter / Flux types are compile-time, not tabulable
   without a heavy X-macro); the table therefore only carries the strings + n_ghost, not the type
   routing. Divergence bug fixed along the way: the hllc / roe AMR branches gained the weno5 case (strict
   parity with System).
9. **check_model on CompiledModel**: DONE (balance) -- `CompiledModel.check_runtime(n=, state=)`
   installs the .so in an EPHEMERAL System and delegates to System.check_model (finite state/residual,
   positivity by roles, round-trip of the conversions); smoke state by ROLES by default,
   state= for a precise regime. Remains: the FORMULAS of a .so without its original dsl.Model are
   not re-derivable (the symbolic source is not embedded in the .so -- assumed).
10. **IO**: System `write` / `checkpoint` / `restart` are **MULTI-RANK** (wave 4) -- collective GLOBAL
    gather (all_reduce_sum_inplace) + rank-0 write + MPI-safe scatter (MONO-BOX System: all
    the state lives on rank 0, exact gather; bit-identical to mono-rank). **PARALLEL HDF5 by
    hyperslabs: DONE on the System `write` side (ADC-66 / PR-IO-3)** -- `sim.write(format="hdf5",
    parallel=True)` OPT-IN: global datasets `(ncomp, ny, nx)` created collectively via h5py(mpio),
    each rank writes ITS boxes in hyperslabs (minimal NON-collective C++ accessors
    `System::local_boxes` / `System::local_state`; `python/system.cpp` + `system.hpp` + bindings).
    `parallel=False` (default) STRICTLY unchanged; h5py absent / without MPI / mpi4py absent ->
    CLEAR RuntimeError with remedy (never a silent write). The cartesian System being MONO-BOX,
    the TRUE hyperslab parallelism only appears in MULTI-BOX (documented honestly). Test:
    `python/tests/test_hdf5_parallel.py` (np=1: parallel True==False equivalence field by field;
    clear error if h5py without MPI; regression of the serial path). REMAINS = PR-IO-3: **PARALLEL HDF5
    AMR** (one group/dataset per level + boxes, ADC-65), **restartable parallel HDF5 checkpoint**
    (the checkpoint stays npz gather-rank-0; `checkpoint(parallel=True)` raises), **AMR checkpoint**
    (explicit rejection, ABI of the fine states per patch missing), external fields (B_z in the checkpoint).
11. **Roe on the DSL side**: DONE (balance) -- TWO complementary routes for the `roe_dissipation` hook
    (trait `HasRoeDissipation`, the core doing F = 1/2(FL+FR) - 1/2 d).
    - **ROLES route** -- `m.enable_roe()` emits `roe_dissipation` from the ROLES: with Energy =
      exact transcription of the canonical Euler algebra of the core (BIT-EXACT parity observed over
      8 steps), without Energy = same decomposition with c = sqrt(p/rho) Roe-averaged, components
      outside the fluid roles = passive scalars on the entropy wave (test_dsl_roe: stationary shear
      preserved exactly in 3-var).
    - **PROVIDED route** -- `m.roe_dissipation(x=rows_x, y=rows_y)`: for an ARBITRARY model (outside the
      fluid-role families), the user provides themselves the n lines d_i of their
      eigenstructure (same spirit as `m.source_jacobian`), written in `dsl.left(...)`/`dsl.right(...)`
      of both states UL/UR (a bare variable, without a marker, raises). The **symbolic autodiff** of the
      DSL assists this writing: `dsl.diff(expr, var)` differentiates the Expr tree (linearity, product,
      quotient, pow/sqrt chain; primitives expanded by their definition; unknown node ->
      NotImplementedError), and `m.flux_jacobian(dir)` derives from it the flux Jacobian A = dF/dU. The
      automatic symbolic DIAGONALIZATION of A (general eigenstructure) stays out of perimeter --
      out of reach of an honest generic emission: the provided route delegates it to the user.

    The two routes are EXCLUSIVE (a single provider of the hook; declaring them together raises) and
    `CompiledModel.has_roe` covers both (test_dsl_autodiff_roe: dsl.diff on analytic cases,
    flux_jacobian of the isothermal 3-var, m.roe_dissipation reproducing the hand-written isothermal Roe
    == enable_roe at ~1e-12, rejections).

## Known variation points (watch-items, not defects)

Audit ADC-189 (tracked by ADC-265) flagged two extension frictions that are deliberate trade-offs,
recorded here so they are not re-discovered later as bugs:

- **Transport dispatch is forked cartesian vs polar.** `runtime/model_factory.hpp`
  `dispatch_transport` (the cartesian table) and `runtime/block_builder_polar.hpp`
  `dispatch_transport_polar` (the polar copy) enumerate the transport bricks independently, so adding a
  transport brick with a polar variant means editing BOTH tables -- the exact drift
  `runtime/dispatch_tags.hpp` was built to remove for the source/elliptic dispatch. Source and elliptic
  dispatch are already shared; only transport forks. A future `kTransports` registry with a `polar_ok`
  flag would let the two tables validate against one source. No behaviour change is implied today.
- **`ModelSpec` is a flat all-bricks POD.** It co-locates every native brick's parameters, so a new
  brick parameter touches the central struct, the Python kwargs, and the dispatch. This flatness is a
  deliberate pybind-friendliness trade-off (one stable struct to bind), not a defect -- a watch-item only
  if the brick count grows much further.
