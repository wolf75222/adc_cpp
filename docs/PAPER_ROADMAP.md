# Hoffart reproduction roadmap (arXiv:2510.11808)

AUDIT (documentary, no implementation). Lists what is MISSING to reproduce the
diocotron benchmark of Hoffart, Maier, Shadid, Tomas (arXiv:2510.11808, Section 5.3: growth
rate of the diocotron instability of a hollow column, in the drift limit
`omega_d << omega_p << omega_c`), and classifies each gap into one of 4 baskets.

Sources read for this audit:
- `docs/ALGORITHMS.md` (numerical bricks: sections 1-3 FV, 9-12 elliptic + cut-cell,
  13-16 AMR, 19 DSL JIT/AOT);
- `docs/ARCHITECTURE.md` (sections 2-4 layers, 7 elliptic, 8 distributed AMR, 10 boundary
  lib/application);
- `docs/GPU_RUNTIME_PORT.md` (phases 8-11: nvcc limit on extended lambdas, negative AMR
  strong-scaling);
- `docs/BIBLIOGRAPHY.md` section 3 (Hoffart entry) and `docs/archive/two_fluid_ap.md`
  (method note on the two-fluid AP scheme);
- `todo.md` section 6 (M1/M2/M2b Hoffart) + sections 1-2 (aux/EPM);
- cases `adc_cases/diocotron/{run.py,README.md,band_instability.py}`,
  `adc_cases/diocotron_amr/run.py`, `adc_cases/two_fluid_ap/`, `adc_cases/cases_manifest.toml`;
- bindings: `python/bindings/system/base/system.cpp`, `python/bindings/amr/amr_system.cpp`, `python/bindings/core/bindings.cpp`,
  `python/adc/__init__.py`, `python/adc/dsl.py`, `include/adc/numerics/elliptic/mg/geometric_mg.hpp`.

## Reproduction status (factual)

Two levels in Hoffart (Section 5.3):

1. **Analytic target** (radial eigenvalue problem of Petri/Davidson-Felice).
   REPRODUCED to 3 digits in numpy on the `adc_cases/diocotron/run.py` side:
   `gamma_3 = 0.772`, `gamma_4 = 0.912`, `gamma_5 = 0.687` (see case README). Out of the core
   scope: this is pure analytic numpy verification.
2. **Numerical rate measured by `adc`**: full pipeline (composition `ExB` + `BackgroundDensity`,
   Poisson of a system with a circular conducting wall `wall="circle"`, measurement of mode `l` of
   `phi` by azimuthal FFT, fit of `exp(gamma t)`). Runs, captures the instability (exponential
   growth, correct mode ordering, `l=4` dominant), but UNDERESTIMATES the rate:
   `l=3 -22 %`, `l=4 -27 %`, `l=5 -5 %` (n=192, Minmod order 2). `todo.md` section 6: M1
   "limited by the numerical diffusion of the ring edge", M2/M2b "AMR on the ring edge triples
   the rate at equal base". The order x resolution sweep now extends this axis up to O5
   (WENO5-Z + SSPRK3) and up to n=512; see the per-mode reading in the "blocker" section below.

At this stage, the gap looks more like a numerical/structural limit of the Cartesian FV scheme
than an isolated bug, but the exact level of the floor remains to be confirmed. The identified
candidate is the **Cartesian ring edge**.

## The structural blocker: Cartesian ring edge

The Shortley-Weller cut-cell capability (`docs/ALGORITHMS.md` section 12) lives ONLY in
`include/adc/numerics/elliptic/mg/geometric_mg.hpp`: it places the circular Dirichlet conducting wall
at its REAL position for the POISSON solver. But the hyperbolic transport
(`numerics/spatial_operator.hpp`, `numerics/numerical_flux.hpp`, `numerics/reconstruction.hpp`)
has NO notion of an embedded boundary: the charge ring is advected on the full Cartesian grid.
The wall predicate (`runtime/wall_predicate.hpp`, `python/bindings/system/base/system.cpp::wall_active`)
feeds only the elliptic operator, never the flux. The net radial gradient of the ring is
therefore diffused by the Cartesian FV scheme, which damps the growth rate in an
l-dependent way (the shorter-wavelength modes, l=4, pay the most). Increasing the
resolution partially closes the gap but does not change the nature of the blocker.

MEASURE (`diocotron/SWEEP_RESULTS.md` on the adc_cases side). The order x resolution sweep
quantifies the diffusion vs structural share per mode. It now covers the high-order axis O5 = WENO5-Z + SSPRK3
(reachable from Python since adc_cpp #88, see Basket 1) and the high resolution n=384/512 (job
ROMEO x64cpu). The real order axis is thus `{O1 none, O2 minmod, O2 vanleer, O5 weno5}`. Goal of the
O5 axis: shed light on the question left open at O2 - is the l-dependent residual diffusion
(closable by order) or a structural floor of the Cartesian ring edge? Per-mode reading
(the detailed %err, the fit windows and the caveats are in `SWEEP_RESULTS.md`, source of
truth):

- **l = 3 (the CLEANEST signal, fit window homogeneous at all n)**: the `|%err|` first closes
  clearly with order and resolution, then at O5 it FLATTENS around -9 % at high
  resolution (-10.3 % at n=256, -8.6 % at n=384, -8.8 % at n=512: a flat step within the
  measurement noise). This is not the behavior of a diffusion that exhausts itself, it is the most
  CREDIBLE candidate for a structural residual.
- **l = 4 (the key mode)**: at low resolution O5 drops to ~ -4 % (n=128, n=256), which the
  O2 reading took for "diffusion almost exhausted"; but at n=384/512 it RISES BACK toward ~ -9/-10 %. Major
  caveat: these two high-resolution points have a fit window that opens early (t0 = 6.3 and
  5.4, like the n=192 point already discarded), so they probably under-read the slope. We
  therefore CANNOT conclude strongly on l=4: we can only say that the -4 % does not reproduce at either
  of the two higher resolutions.
- **l = 5**: already close to the target from O2 at n=192; small residual of variable sign (a few %),
  neither order nor high resolution makes a floor appear.

CAUTIOUS CONCLUSION (to be confirmed, does NOT constitute definitive proof). The O5 + high
resolution axis WEAKENS the hypothesis "all the gap was order-2 diffusion": at order 5, on
the best-measured mode (l=3), the residual does not keep closing but seems to plateau. The
DATA SUGGEST an l-dependent residual floor on the order of ~9-10 % at order 5 (against ~12 %
seen at O2), probably tied to the Cartesian ring edge / transport wall, REMAINS TO BE CONFIRMED. Two
limits prevent making this a firm number: (1) the l=3 plateau for now holds only on
a single flat step n=384 -> n=512 (an n=768/1024 or two `t_end` horizons would exclude a very
slow convergence); (2) the l=4 high-resolution points are biased by their early fit
window (a robust window diagnostic is to be planned before quantifying an l=4 floor). This structural candidate
remains the quantitative argument for PR-A "transport-wall", now with a plausible size
revised to ~9-10 % at order 5.

## Recommended public API (orientation)

Two recommended user entry points, both on native bricks (GPU/MPI path):

- **Compose native bricks**: `adc.Model(state, transport, source, elliptic)` assembles a
  model from state / transport / source / elliptic bricks, consumed by
  `System.add_block(...)` (or `AmrSystem`). This is the path of the sweep diocotron cases.
- **Write a model in formulas**: `adc.dsl.Model(...)` describes the equations symbolically, then
  `m.compile(...)` produces a `.so`. For production, `backend="production"` is the recommended
  default (native zero-copy loader -> `add_native_block`, GPU/MPI path).

ADVANCED / LEGACY / TEST paths, NOT the main user path:

- `backend="prototype"` (JIT, IModel, virtual dispatch, host order-1 Rusanov) and
  `backend="aot"` (host-marshaled AOT): iteration / verification, not production.
- `add_dynamic_block` (JIT prototype) and `add_compiled_block` (AOT): corresponding low-level adders.
- `adc.PythonFlux`: HOST numpy path (OUTSIDE the GPU/MPI hot path), to TEST a flux written in
  formulas, not for production.

## Gap classification (4 baskets)

### Basket 1: already possible with the CURRENT API (to launch / tune)

Capabilities wired and exposed, sufficient to push further without new code.

- **Increasing RESOLUTION and ORDER**: pure tuning (the diocotron case already runs at variable n).
  This is the M3 path of `todo.md` section 6, and the resolution x order sweep is done (see
  `SWEEP_RESULTS.md`). The WENO5-Z + SSPRK3 order increase is now reachable from Python
  (adc_cpp #88): `adc.Spatial(limiter="weno5")` (shortcut `weno5=True`) selects the
  WENO5-Z reconstruction in `make_block`, and `adc.Explicit(method="ssprk3")` (shortcut
  `ssprk3=True`) the SSPRK3 integrator, via the native path `add_block`. The default stays unchanged
  (Minmod / SSPRK2, bit-identical to pre-#88). WENO5 is now wired ALSO on the `.so` paths
  AOT and production (`add_compiled_block` / `add_native_block`, adc_cpp #102: `.so` grid at 3
  ghosts), and on the native AMR path (`AmrSystem` production, adc_cpp #105). Only the JIT
  prototype path (`add_dynamic_block`) stays at 2 ghosts and rejects `"weno5"` (see Basket 2). The sweep
  therefore covers `{O1, O2-minmod, O2-vanleer, O5 weno5}`, up to n=512.
- **Circular conducting wall on Poisson**: `wall="circle"` + `wall_radius` is wired on
  `System` (`python/bindings/core/bindings.cpp:97`) AND on `AmrSystem` (`python/bindings/core/bindings.cpp:193`,
  `python/bindings/amr/amr_system.cpp:78`). The elliptic cut-cell is validated (MMS order 2, multi-box, MPI;
  `docs/ALGORITHMS.md` section 12). Nothing to write for the Petri ring geometry.
- **AMR on the ring edge**: `adc.AmrSystem` + `set_refinement(threshold)` runs and
  conserves mass (case `adc_cases/diocotron_amr/run.py`). M2/M2b of `todo.md` note that
  AMR triples the rate at equal base. Pushing the refinement / the number of levels is a
  config tuning. `AmrSystem.potential()` (reading `phi` from Python for the azimuthal
  FFT): binding SHIPPED (python/bindings/core/bindings.cpp:272, `#135`).
- **Rate diagnostic**: the measurement chain (azimuthal FFT of mode `l` of `phi`, fit of
  the linear phase) is entirely in place on the `adc_cases` side.

### Status of the GPU / MPI execution paths (production)

Factual status of the production paths (native bricks, not DSL), independent of PR-A:

- **`System` production CPU**: validated (ctest serial; diocotron pipeline runs).
- **`AmrSystem` production CPU**: validated.
- **`System` GPU production np=1**: validated on GH200 (adc_cpp #97). #97 fixes the device segfault
  of the elliptic/mesh kernels (extended lambdas first-instantiated from an external TU ->
  named functors, robust device codegen under nvcc); Cuda vs Serial parity `dmax_abs` ~ 1e-13
  on `solve_fields`, `compute-sanitizer` clean.
- **`System::solve_fields` MPI CPU np=1/2/4**: validated (adc_cpp #99). #99 fixes the host segfault
  of the per-cell post-processing (`fab(0)` without a `local_size()` guard on ranks without a box);
  result bit-invariant to the number of ranks (`test_mpi_system_solve_fields_np{1,2,4}`, runs in MPI CI).
- **device-MPI production (multi-rank GPU)**: validated on GH200 (adc_cpp #93). The production
  DSL path (`add_compiled_model`) with `geometric_mg` is validated device + MPI np=1/2/4 (dedicated harness).
- **fft under `System` in MPI np>1**: REFUSED cleanly (adc_cpp #106: hard safeguard, no more
  segfault). In MPI `System` distributes ONE box round-robin, a layout incompatible with the FFT.
  `DistributedFFTSolver` (band layout, `MPI_Alltoall`) exists and is tested separately, but is
  NOT routed in `System` (band layout vs single box); the distributed periodic goes through it or
  through `geometric_mg`.

These paths are NOT on the critical path of the analytic target nor of the diocotron sweep (CPU),
but they condition the multi-GPU resolution increase mentioned in Basket 4.

### Basket 2: production DSL facade `m.compile(backend=...)`

The symbolic DSL exists and compiles (JIT IModel prototype, AOT, and production native loader);
the production facade is consolidated. Reproducing Hoffart does NOT depend on the DSL (the native
bricks suffice); this basket is only required if one wants to drive the full magnetized model
in formulas from Python rather than by composing bricks.

- **Consolidated `compile` facade**: `python/adc/dsl.py` exposes the ergonomic `m.compile(backend=...)`
  (auto-detection of the core include, `so_path` cache by ABI key, adc_cpp #103) on top of
  `compile_so` (JIT prototype), `compile_aot` (AOT) and `compile_native` (production, native zero-copy
  loader -> `add_native_block`, target `"system"` or `"amr_system"`). The native production
  backend is SHIPPED (#85, #92): `target="amr_system"` routes to `AmrSystem.add_native_block` (DSL
  Phase D). The DSL demonstrators are complete and all `ci = true` in adc_cases CI:
  `diocotron_dsl`, `two_species_dsl`, `magnetic_isothermal_dsl`. No diocotron sweep case
  goes through the DSL (the compositions go via `models.diocotron(...)`, native bricks), but the
  diocotron IS now written in DSL in `diocotron_dsl`.
- **WENO5 on `.so`**: SHIPPED (#102). `add_compiled_block` (AOT) and `add_native_block`
  (production) allocate a `.so` grid at 3 ghosts and accept `"weno5"`; the native AMR path
  accepts it too (#105, parity `add_native_block` == `add_compiled_model` == `add_block`, dmax=0).
  Only the JIT prototype path (`add_dynamic_block`) stays at 2 ghosts and rejects `"weno5"`.
- **Consolidated device driving**: the device-clean recipe (extended lambda -> named functor, robust device
  codegen under nvcc) covers transport (`block_builder.hpp`, adc_cpp #64), the
  elliptic/mesh kernels of `solve_fields` (#97, GPU `System` production np=1 validated GH200) AND the
  production DSL device + MPI np=1/2/4 path (#93, `geometric_mg` validated GH200). GPU validation is
  therefore no longer pending on the production side.

### Basket 3: FV disk-domain / wall capability (real circular domain, not Cartesian edge)

This is the basket that lifts the structural BLOCKER. None of these capabilities exist today.

- **Embedded boundary on the TRANSPORT side**: carry the notion of cut-cell / reflecting wall from the
  elliptic solver (`geometric_mg.hpp`) to the hyperbolic operator (`spatial_operator.hpp`)
  so that the charge ring is no longer advected on a full Cartesian grid. This is the
  gap that explains the l-dependent under-rate. `docs/ARCHITECTURE.md` section 12 (AMReX
  comparison) also notes a "staircased" EB Laplacian on the operator side, the cut-cell being only
  for the elliptic CURVED boundary.
- **Circular mesh / adapted coordinates**: an alternative to the embedded Cartesian, a domain
  that is really a disk (polar coordinates or full cut-cell mesh); not present (the core
  is adaptive Cartesian, `docs/ARCHITECTURE.md` section 1).

As long as this basket is not addressed, the fine QUANTITATIVE reproduction of the numerical rate remains
bounded by the diffusion of the Cartesian edge (finding M1, `todo.md` section 6).

### Basket 4: advanced multi-block AMR or advanced EPM

Capabilities partially present but incomplete for an advanced Hoffart use.

- **Multi-block / multi-level AMR at `System` parity**: `AmrSystem` stays MONO-block (not
  multi-species); local source IMEX OK (Gap 2 #132, backward_euler_source /
  mf_apply_source_treatment) but global Schur on AMR and multi-block AMR remain to be done. The
  native multi-box is not wired on the facade side,
  and the Python AMR facade WIRES HLLC/Roe and the primitive reconstruction with a pressure guard (see `python/adc/__init__.py`,
  safeguard `add_equation`): HLLC/Roe require a declared primitive `p` (or `enable_hllc()` / `enable_roe()`)
  (the `add_block` API accepts the primitive recon on the C++ side). WENO5 + Rusanov + conservative IS wired
  on the native AMR path (#105). A high-resolution + high-order AMR diocotron at `System`
  parity (primitive recon, Roe, multi-box) requires opening these options on the facade side.
- **Full-device AMR strong-scaling**: the distributed coarse (`replicated_coarse=false`) is wired
  but NEGATIVE at the tested scale (`docs/GPU_RUNTIME_PORT.md` phase 11). Required only for
  very large multi-GPU resolutions, not for the Section 5.3 target.
- **Advanced EPM**: the extended elliptic operator (eps(x), Helmholtz/screened, anisotropic) is done
  and device-validated (`docs/ALGORITHMS.md` section 11, `todo.md` section 2), but the wiring
  `EllipticProblem` -> stencil via the additive factory remains DESCRIPTIVE, and the Schur EPM
  splitting is deferred (`docs/ARCHITECTURE.md` section 7, `todo.md` section 7). Not required for the
  drift diocotron (pure Poisson + wall), relevant if one targets the full magnetized
  two-fluid model.
- **Full magnetized model**: the two-fluid AP scheme (`adc_cases/two_fluid_ap/`,
  `docs/archive/two_fluid_ap.md`) carries the exact cyclotron rotation (section 6 of the note) but
  NOT yet the `E x B` + inhomogeneous diamagnetic coupling to transport. This is the Hoffart
  extension beyond the drift limit.

## Ordered plan

### DONE (to date)

- **WENO5-Z / SSPRK3 reachable from Python** (adc_cpp #88): `adc.Spatial(limiter="weno5")` +
  `adc.Explicit(method="ssprk3")` via the native path `add_block`, default unchanged.
- **Order x resolution sweep extended to O5 and to n=384/512** (see `SWEEP_RESULTS.md`): order
  `{O1, O2 minmod, O2 vanleer, O5 weno5}`, up to n=512 (high resolution on ROMEO x64cpu).
  Reading: l=3 plateaus ~ -9 % at O5 high resolution (cleanest structural candidate); l=4 does not
  reproduce its -4 % low resolution but its high-resolution points are biased by their
  fit window; l=5 already at target (see cautious conclusion in the blocker section).
- **GPU `System` production np=1** validated on GH200 (adc_cpp #97).
- **`solve_fields` MPI CPU np=1/2/4** validated (adc_cpp #99).
- **device-MPI production** (`add_compiled_model` + `geometric_mg`) validated on GH200 np=1/2/4
  (adc_cpp #93).
- **WENO5 on the `.so` paths** (AOT + production) validated (adc_cpp #102), and on the native AMR
  path (adc_cpp #105, parity `add_native_block` == `add_compiled_model` == `add_block`, dmax=0).
- **`AmrSystem` native production** (`add_native_block`, `target="amr_system"`) shipped (adc_cpp #92,
  DSL Phase D).
- **fft under `System` MPI np>1 refused cleanly** (adc_cpp #106: hard safeguard, no more segfault).
- **ergonomic `compile()` + `so_path` cache** (adc_cpp #103).
- **Complete DSL demonstrators** (`diocotron_dsl`, `two_species_dsl`, `magnetic_isothermal_dsl`),
  all `ci = true` in adc_cases CI.

### Next scientific BLOCKER (the only one that lifts the structural under-rate)

- **Basket 3 - transport boundary / disk-domain / embedded boundary**: carry an embedded boundary /
  wall on the transport side (or a domain that is really a disk) so that the charge ring is no longer
  advected on a full Cartesian grid. This is the only path that addresses the structural floor
  candidate ~9-10 % highlighted by the O5 sweep. The heaviest and most rewarding work item for
  the numerical rate. The sweep is NOT a proof of it: it SUGGESTS the candidate and remains to be confirmed
  (n=768/1024 or two `t_end` horizons for l=3; robust window diagnostic for l=4).
  The "transport-wall Phase 1" attempt (adc_cpp #109) is EXPERIMENTAL and was CLOSED WITHOUT
  merge: it placed a transport boundary on the external CONDUCTOR (wrong boundary), which masks
  the real blocker. The scientific blocker stays the RING EDGE (the net radial gradient of the charge
  ring), not the conducting wall. PR-A "transport-wall" must target this ring edge.

### Next infrastructure blockers (can land in parallel)

- **`AmrSystem.potential()` on the Python side**: binding SHIPPED (python/bindings/core/bindings.cpp:272, `#135`);
  `phi` readable from the AMR for the azimuthal FFT of the rate diagnostic.
- **distributed fft routed in `System`**: `DistributedFFTSolver` (band layout) exists and is
  tested separately but is NOT routed in `System` (band layout vs single box); wiring it
  would allow the distributed periodic MPI without falling back on `geometric_mg`. Not required for the target.
- **AMR facade <-> `System` parity**: open on the Python facade side the native multi-box, the primitive
  recon and the HLLC/Roe fluxes (wired on the AMR facade with a pressure guard) to
  push the AMR to high resolution + high order at `System` parity.
- **Basket 4 depending on the ambition**: full-device AMR strong-scaling (distributed coarse negative at
  the tested scale), then the full magnetized model (`two_fluid_ap` coupled to transport) if one
  leaves the drift limit.

## Blocker summary

Reproducing the analytic TARGET of Hoffart is done (numpy, 3 digits). Reproducing the
NUMERICAL rate at parity requires lifting the Cartesian ring edge (basket 3): today the
cut-cell serves only Poisson, the transport stays Cartesian, hence an l-dependent under-rate that the
resolution attenuates without removing. The sweep extended to O5 (WENO5-Z + SSPRK3, reachable from
Python since adc_cpp #88) and to high resolution n=384/512 WEAKENS the hypothesis "all the gap
was order-2 diffusion": on the best-measured mode (l=3), the O5 residual no longer closes
but plateaus around -9 %. The data therefore SUGGEST a candidate structural floor on the
order of ~9-10 % at order 5, REMAINS TO BE CONFIRMED (a single flat step on l=3; early fit window
biasing l=4) - NOT definitive proof, but the quantitative argument for PR-A
"transport-wall". Driving WENO5-Z/SSPRK3 from Python is therefore no longer a blocker (done, #88);
the remaining blocker is indeed the transport boundary (basket 3).
