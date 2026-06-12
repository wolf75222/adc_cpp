<!--
Provenance : document de DESIGN produit par un workflow multi-agents (mapping parallele
des sous-systemes adc_cpp -> design Phase B/C -> revue adversariale -> synthese), 2026-06-08.
C'est une FEUILLE DE ROUTE d'implementation, PAS du code valide. Les sections REVIEW FIX
et la section 3 (risques ouverts) sont les conclusions adversariales a respecter avant code.
Findings decisifs : R0 (le solve_fields de tete de pas ecrase le phi du Schur -> lien direct
avec GaussPolicy), R1 (le taux mesure est POLAIRE, incompatible patches AMR), R4 (pas de
reduction globale deterministe -> 'bit-identique entre np' est FAUX).
-->

# AMR Condensed-Schur -- Implementation Design (Phase B + C)

> **STATUT (2026-06-10) : design HISTORIQUE, largement REALISE depuis.** Conserve pour ses
> conclusions adversariales (R0-R4, G1/G5), toujours valables. Realise sur master :
> Phase B -- `AmrSystem::set_conservative_state` (mono-bloc, puis MULTI-BLOCS natif via PR #267) ;
> Phase C -- solveur elliptique COMPOSITE FAC + Schur tenseur AMR (PR #262), etage source condense
> par Schur COMPOSITE multi-niveau "Phase 3c" (PR #266), tolerances Krylov + roles configurables
> (PR #267). Restes EXACTS du perimetre : multi-patch fin / >2 niveaux / MPI / multi-blocs de
> l'etage Schur AMR ("Phase 4", rejet explicite dans amr_condensed_schur_source_stepper.hpp) ;
> R0 (re-solve electrostatique de tete de pas vs phi du Schur) reste la decision structurante,
> cf. HOFFART_STEP_SEQUENCE.md ; G5 (AMR-polaire incompatible RadialLine) reste vrai.


## 0. Context & goal

**Goal.** Reproduce the Hoffart diocotron run (arXiv:2510.11808) on the AMR engine with a paper-faithful source stage. Two chantiers:

- **Phase B** -- give `AmrSystem` a `set_conservative_state` so the AMR run can start from the *drift* state (`[rho, rho*u0, rho*v0(, E)]`), not density-only. Pure additive plumbing; existing density-only path stays bit-identical.
- **Phase C** -- `AmrCondensedSchurSourceStepper`: the paper source stage (`A = I + theta^2 dt^2 alpha rho B^-1`, condensed tensor-Poisson `-div(A grad phi) = rhs`, velocity reconstruction `v_theta = B^-1(v^n - theta dt grad phi_theta)`) on the AMR hierarchy.

**Decisive reframe (Phase C).** The current AMR field solve is **NOT composite**. `solve_fields()` (`amr_runtime.hpp:466-493`) does `mg_.solve()` on the **coarse level only** (line 479), then `field_postprocess` (486) + **piecewise-constant injection** of `phi`/`grad phi` to fine levels (`coupler_inject_aux_mb`, 490-492). `GeometricMG`'s `lev_[]` are V-cycle domain-coarsening levels of a *single* `BoxArray` (`geometric_mg.hpp:86-111`), **not** AMR refinement levels. So Phase C is a genuinely new elliptic capability (composite AMR tensor-Poisson), not a re-wiring.

> **REVIEW FIX [recon-extrap G1] -- the actual decisive fact is upstream of the reframe.** `step()` (`amr_runtime.hpp:589`) opens **every macro-step** with `solve_fields()` (line 603), which re-solves the *bare electrostatic* Poisson on `rho^{n+1}` and **overwrites** any Schur-evolved `phi` (`docs/HOFFART_STEP_SEQUENCE.md:9-11`, `HOFFART_FIDELITY.md` Gauss-restart row). Transport reads only `aux`. **Consequence:** with the current Lie ordering the Schur `phi` is discarded before it can drive transport -- the composite solve influences the trajectory **only** through the reconstructed momentum `m` feeding `rho^{n+1}`. **This must be resolved before investing in §2.3-2.4 reflux machinery** (see §3 R0). Either (a) suppress the top-of-step electrostatic re-solve on steps the Schur stage owns `phi` and forward Schur `phi` into `aux` (the fix `HOFFART_STEP_SEQUENCE.md:61` names), or (b) accept the documented `difference_intentionnelle`, right-size the elliptic effort (Tier 0, loose tol), and **measure growth-rate sensitivity to solver tol on the existing mono path first**.

> **REVIEW FIX [recon-extrap G5] -- geometry scope.** The *measured* −0.38% diocotron rate lives on the **polar** path (`PolarCondensedSchurSourceStepper`), whose `PolarTensorKrylovSolver` precond is `RadialLine` and **requires every box to span the full radial extent** (`polar_condensed_schur_source_stepper.hpp:317-325`, `check_radial_columns`). AMR fine *patches* do not span full r -- **AMR-polar is structurally incompatible with the existing polar solver**. This entire Phase C design targets the **cartesian** path only. If the deliverable is the polar measurement, Phase C as written is wrong-geometry and a separate AMR-polar elliptic story is required. State the target geometry explicitly before building.

Two tiers are carried throughout:
- **Tier 0** -- per-level solve, coarse-Dirichlet C/F BC, **no** elliptic reflux. First-order-conservative at the C/F interface; staging milestone + debugging oracle, **not** the deliverable.
- **Tier 1** -- composite single-grid (AMReX MLMG-style) solve with elliptic flux reflux; the conservation deliverable.

---

## 1. Phase B -- `AmrSystem::set_conservative_state` (drift initial state) [implementation-ready, with the type-erasure fixes]

Component layout (matches `to_3d`/`set_primitive_state`): **component-major, row-major within a component** -- `Uflat[c*n*n + j*n + i]`. Guiding invariant **NO-DEFAULT-CHANGE**: when `has_state == false` every path falls through to the unchanged `coupler_write_coarse` density branch; the state path is purely additive.

### 1.1 Struct / field additions

- **`AmrBuildParams`** -- `include/adc/runtime/amr_system.hpp:75-90`. Add after the density fields (≈86-87):
  ```cpp
  bool has_state = false;
  std::vector<double> state;   // ncomp*n*n, component-major c*n*n+j*n+i; ncomp == Model::n_vars
  ```
  Prioritaire sur `density` quand `has_state`. ncomp not stored -- it is `Model::n_vars`, validated `state.size()==ncomp*n*n` at write.
- **`BlockSpec`** -- `python/amr_system.cpp:98-100`. Mirror: `bool has_state=false; std::vector<double> state;`.
- **`AmrCompiledHooks`** -- unchanged.

### 1.2 New kernel `coupler_write_coarse_state` -- `include/adc/coupling/amr_coupler_mp.hpp` (add after `coupler_write_coarse`, ≈line 133)

New function; `coupler_write_coarse` (line 116) left untouched (density path provably bit-identical). Distribution-aware, multi-box; copies all `ncomp` components positionally (no rho/momentum/energy hardwired -- caller does prim→cons). `gamma` omitted (energy supplied by caller).

```cpp
inline void coupler_write_coarse_state(MultiFab& U, const std::vector<double>& state, int n, int ncomp) {
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  if (state.size() != nn * static_cast<std::size_t>(ncomp))
    throw std::runtime_error("AMR coupler : etat initial de taille != ncomp*n*n");
  device_fence();
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 u = U.fab(li).array();
    const Box2D v = U.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        for (int c = 0; c < ncomp; ++c)
          u(i, j, c) = state[static_cast<std::size_t>(c) * nn
                           + static_cast<std::size_t>(j) * static_cast<std::size_t>(n) + i];
  }
}
```

> **REVIEW FIX [phaseB-typeerasure L1] -- compute `j*n+i` in `std::size_t`.** The flat index above casts `j` and `n` to `size_t` (shown). `coupler_write_coarse` validates in `int` (`amr_coupler_mp.hpp:118`); the new kernel validates in `size_t` (better, but note the kernels are *not* arithmetically identical despite the "pendant exact" claim). Latent `int` overflow at `n >= ~46341` is thereby avoided.

Box iteration matches `coupler_write_coarse` (122-131) exactly → identical multi-box/replicated/distributed behaviour; only the per-cell write differs. `device_fence()` preserved.

### 1.3 Builder threading (mono + multi)

- **`build_amr_compiled`** (mono) -- `amr_dsl_block.hpp:90`. Replace seed line:
  ```cpp
  if (bp.has_state)        coupler_write_coarse_state(cpl->coarse(), bp.state, bp.n, nc);
  else if (bp.has_density) coupler_write_coarse(cpl->coarse(), bp.density, bp.n, nc, bp.gamma);
  ```
  `coupler_inject_coarse_to_fine_mb` (92-93) already injects all `nc` comps piecewise-constant (kernel loop `k<nc`, line 106) -- momentum prolongs free, **no change**.
- **`build_amr_block`** (multi) -- `amr_dsl_block.hpp:215-235`. Append `const std::vector<double>& state = {}, bool has_state = false` to the signature; seed block (232-233) gets the same `if/else if` branch.
- **`dispatch_amr_block`** -- `amr_dsl_block.hpp:334-394`. Append the two params; thread `, state, has_state` through **all 10** `build_amr_block<...>` call sites (343/346/349/352 rusanov; 360/363/366 hllc; 378/381/384 roe).
- **`multi_builder` lambda in `add_compiled_model`** -- `amr_dsl_block.hpp:520-531`. Extend lambda params `const std::vector<double>& bstate, bool bhas_state` and forward to `dispatch_amr_block(..., bstate, bhas_state)`.

### 1.4 Type-erased boundary -- `AmrCompiledBlockBuilder` typedef (`amr_system.hpp:113-117`)

Append the two params to the `std::function` signature:
```cpp
... bool imex, int stride, const std::vector<std::string>& implicit_vars,
const std::vector<std::string>& implicit_roles,
const std::vector<double>& state, bool has_state)>;
```

> **REVIEW FIX [phaseB-typeerasure C1+H1] -- the ABI guard does NOT catch a header *edit*; this is the central Phase B risk.** The guard at `amr_system.cpp:501` compares `loader_key` vs `detail::abi_key_string()` = `compiler;std;ADC_HEADER_SIG` (`abi_key.hpp:46-49`). The module-side `ADC_HEADER_SIG` is injected by CMake via `file(GLOB_RECURSE ... CONFIGURE_DEPENDS)` (`python/CMakeLists.txt:32-41`), whose own comment (26-30) warns: *"Une EDITION de contenu d'en-tete n'invalide pas le glob."* Phase B **edits existing headers, adds no new files** → `cmake --build` (without re-`cmake -S`) keeps a **stale** signature. Stale module + stale loader produce **matching** keys → the guard passes → `build_multi` (`amr_system.cpp:232`) invokes the new 12-arg typedef against a 10-arg lambda across the `dlopen` boundary → **stack corruption on the density-only path** (NO-DEFAULT-CHANGE violated by the ABI surface change itself). The same hazard hits the `AmrBuildParams` struct-layout change (§1.1), via the mono `mono_builder(make_build_params())` path. **Mandatory fix:** add a literal, glob-independent sentinel to `abi_key_string()` (e.g. `";amr_builder_v=2"`) so both header-side and python-side keys change deterministically the instant the typedef/struct changes. **AND** document that landing this requires a full reconfigure (`cmake -S . -B build-py`), not just `cmake --build`, plus `dsl.compile_native(target='amr_system')` to regenerate loaders. The design's original R1 ("typedef perturbs nothing in the hash") is backwards and must not be relied on.

### 1.5 Python marshaling

- **`make_build_params`** -- `python/amr_system.cpp:183-184`: `bp.has_state = b.has_state; bp.state = b.state;`.
- **compiled multi dispatch** -- `python/amr_system.cpp:232-234`: append `, b.state, b.has_state`.
- **native multi dispatch** -- `python/amr_system.cpp:248-250`: append `, b.state, b.has_state`.
- `set_compiled_block` (387-430) needs **no signature change** -- `state`/`has_state` default-init and are filled post-registration by `set_conservative_state` (mirrors `set_density`). *(But its correctness depends entirely on the §1.4 sentinel bump -- qualify the "no change" claim accordingly [phaseB-typeerasure H1].)*

### 1.6 `set_conservative_state` + binding

- **Declare** -- `amr_system.hpp` after line 258. Full conservative state on coarse level, component-major, prioritaire sur `set_density` (last call wins; flags independent). `name` cosmetic mono / indexes block multi.
- **Define** -- `python/amr_system.cpp` after `set_density` (≈588). Mirror `set_density`: guard `built`, guard empty blocks, **size guard** `U.size() % (n*n) == 0`, multi-block name→index, set `blocks[idx].state/.has_state`. Leave `has_density` as-is (inoffensive; never read when `has_state`).

  > **REVIEW FIX [phaseB-typeerasure M1] -- reject empty state explicitly.** `0 % nn == 0` is true, so a zero-length `U` would set `has_state=true` with empty `state` and throw only deep inside the first `step()`. Add an explicit `U.size()==0` rejection distinct from the modulus check, and surface the count mismatch (`ncomp != n_vars`) as early as the size check allows. The fine `ncomp*n*n` check stays deferred to `coupler_write_coarse_state` at build (ncomp = `Model::n_vars` known only then -- same deferral as density's `n*n` check).

- **pybind** -- `python/bindings.cpp` after the AMR `set_density` binding (≈318). Mirror the `set_density` lambda; `flat(arr)` (line 51) C-contiguates `(ncomp,n,n)` → `c*n*n+j*n+i`.

  > **REVIEW FIX [phaseB-typeerasure L2] -- assert `arr.ndim()==3` in the binding.** `flat(arr)` flattens *any* C-contiguous array (`bindings.cpp:51-53`), so a 2-D `(n,n)` density passed by mistake silently becomes a 1-component state (passes `%nn==0`), writing comp 0 and leaving momentum/energy at `set_val(0)` defaults -- a silent density-masquerade with wrong physics. Add `if (arr.ndim()!=3) throw ...` before `flat`, mirroring `to_3d`'s validation (`bindings.cpp:38-49`).

- **facade** -- `python/adc/__init__.py`: `AmrSystem` delegates via `__getattr__`; binding auto-exposed, no facade edit (confirm `set_density` is not re-declared in the AMR facade -- it is not).

### 1.7 New read binding `coarse_state()` -- REQUIRED (not optional)

> **REVIEW FIX [phaseB-typeerasure M3] -- without this, momentum prolongation is UNTESTED.** Tests B and C below need to read back all `nc` components; the proposed "indirect" momentum check via `potential()` is **vacuous** (potential depends only on comp 0). Add a small `coupler_read_coarse_all(U, n, ncomp, replicated)` mirroring `coupler_read_coarse` but looping all comps, plus `AmrSystem::coarse_state()` and a one-line pybind. This makes the momentum claim -- the second half of Phase B -- actually verifiable.

### 1.8 `run.py` drift seed -- `adc_cases/hoffart_euler_poisson_dsl/run.py:186-214`

`AmrSystem` has no `set_primitive_state`/`solve_fields`, and `potential()` triggers `ensure_built()` which freezes the system (so `set_*` must precede any read). Seed recipe: build a **throwaway probe** AMR with the same config + density, read `probe.potential()` for `phi0`, discard it, compute drift `v0 = -(grad phi0 x Omega)/|Omega|^2` (zeroed outside disc, same recipe as `build_uniform` 177-182), build `Uflat = stack([rho, rho*u0, rho*v0(, E)])`, then `set_conservative_state` on a fresh system.

> **REVIEW FIX [phaseB-typeerasure H2a] -- single-pass vs two-pass; do not overstate fidelity.** `build_uniform` (the `system-schur` reference) solves Poisson → drift → `set_primitive_state` → **solves again** (two-pass relaxation). The AMR probe path solves **once** (the probe is discarded; `set_conservative_state` must precede `ensure_built`, so no second solve is possible). Do **not** set `paper_initial_drift=true` for amr-imex as if equivalent -- use a distinct flag (`paper_initial_drift_amr_1pass`) or keep the single-pass limitation in `known_differences` (metadata `run.py:464,482`). Update the module docstring (15-17) which states "AMR initializes only density".

> **REVIEW FIX [phaseB-typeerasure H2b] -- wrap the probe in try/except; do not make a Poisson solve a hard build dependency.** The old `build_amr` never solved Poisson at build. The probe runs `set_poisson(...geometric_mg, wall=circle)` which can throw on a degenerate MG hierarchy (known cut-cell/MG coarsening fragility). If it throws, the **entire** `build_amr` fails, including the density-only fallback that worked before -- a NO-DEFAULT-CHANGE *build-robustness* regression. Wrap the probe in try/except and fall back to `set_density` (old behavior) on failure, or compute `phi0` from a cheap one-level `System` solve.

> **REVIEW FIX [phaseB-typeerasure R4] -- component-order guard is mandatory, not optional.** Build `Uflat` from the model's declared `conservative_vars()` count/order, not hardcoded `3`. Add a hard `assert Uflat.shape[0] == nc`. The kernel guards count (`ncomp*n*n`) but **not order**; a wrong order silently corrupts the run. For `nc==4` append `E = rho/(gamma-1) + 0.5*rho*(u0^2+v0^2)`.

### 1.9 Phase B verification (CPU-laptop, serial, `python/tests/test_amr_conservative_state.py`)

- **Test A -- round-trip / density fall-through (NO-DEFAULT-CHANGE).** `n=32`, 2 levels, `regrid_every=0`, `refine_threshold=1e30`. Seed via `set_density` vs an equivalent `set_conservative_state([rho,0,...])`; assert `density()`/`potential()` identical.
  > **REVIEW FIX [phaseB-typeerasure M2] -- bit-identity only holds for `nc==1`; demote `nc==4` to `allclose`.** `coupler_write_coarse_state` drops `gamma`, so the state path's energy comes from Python (float64) while the density path uses the kernel's `Real(gamma)-Real(1)` (possibly float32 on GPU builds). `np.array_equal` then fails on a ULP. Use the **scalar 1-var** case as the exact NO-DEFAULT-CHANGE gate; for `nc==4` build the equivalent state in the same `Real` dtype and assert `np.allclose` with documented tol. Also note `refine_threshold=1e30` tags nothing → `regrid` is a **no-op**, so the fine seed survives as the injected copy (the design's R2 "immediately overwritten" is wrong for this test config).
- **Test B -- full-state read-back at step 0** via `coarse_state()` (§1.7): assert comp 0 == `rho`, comps 1,2 == `rho*u0`, `rho*v0` to machine precision.
- **Test C -- momentum prolongation + coarse==avg(fine).** With an actual fine patch, assert fine-level state = piecewise-constant injection of coarse momentum, **and** add the step-0 assertion `coarse == average_down(fine)` (the real invariant R3 claims but never tests; [phaseB-typeerasure M4]). Exact (injection is a copy).
- **Test D -- mass conservation** over N=20 steps from a momentum-carrying state (`< 1e-12` rel).
- **Test E -- momentum drives motion**: seed `[rho, rho*u0, 0]`, one step, assert density centroid shifts +x.

---

## 2. Phase C -- `AmrCondensedSchurSourceStepper` [implementation-ready]

> **REVIEW FIX [ellip-cf C1+C3+C4 -- the single load-bearing decision].** Adopt the **composite single-grid (AMReX MLMG-style)** formulation. The original design conflated three mutually inconsistent choices (RHS reflux vs matvec reflux; covered identity-rows vs covered exclusion; "MLAT" without the τ-correction). Resolve all three at once:
> - **Covered coarse cells are eliminated slaves**, not unknowns: `phi_c = avg(phi_f)`, set by `mf_average_down_mb` *after* each matvec, and **zeroed in every Krylov vector** (`r,p,v,s,t,phat,shat`) -- not merely masked in the dots.
> - **Unknowns** = uncovered-coarse ∪ all-fine.
> - **C/F-bordering coarse rows carry the fine-flux reflux *inside the operator* (`out`)**, never in the RHS. `rhs` stays the plain assembled `-Δphi^n - θdt α div(ρ B^-1 v^n)` with **no reflux term and no τ**.
> - This makes the residual `rhs - A_amr·phi` exactly the discrete-Gauss defect, so driving it to zero *is* conservation; the "re-solve until residual<tol" outer loop collapses into the single BiCGStab loop. **Delete** the original §3.3-step-4 outer iteration and the §2.2 "Tier 1 RHS correction" term.

### 2.1 Per-level storage

Every mono-level field (`condensed_schur_source_stepper.hpp:189-367`) becomes per AMR level `l`:

| Mono field | Per-level | ncomp/ghost | role |
|---|---|---|---|
| `phi_n_` | `phi_n_[l]` | 1/1 | frozen `phi^n` (extrapolation) |
| `rhs_schur_` | `rhs_[l]` | 1/0 | condensed RHS |
| `eps_x_,eps_y_` | `eps_x_[l],eps_y_[l]` | 1/1 | `A_xx,A_yy` |
| `a_xy_,a_yx_` | `a_xy_[l],a_yx_[l]` | 1/1 | `A_xy,A_yx` |
| `bz_` | reuse `aux_[l]` B_z comp | -- | `CopyBzKernel` |
| `vx_n_,vy_n_` | `vx_n_[l],vy_n_[l]` | 1/0 | `v^n` |
| `vx_t_,vy_t_` | `vx_t_[l],vy_t_[l]` | 1/0 | `v^{n+θ}`→`v^{n+1}` |

**Ownership.** The stepper owns its own `std::vector<MultiFab>` per field, sized once in ctor against each level's `(*b.levels)[l].U.box_array()/dmap()`. **Not** in `AmrLevelMP` (transport state, must stay lean) and **not** in `AmrLevelStack::aux_` (address-stability invariant `amr_level_storage.hpp:33`, fixed width). Mirrors the mono "allocate-once" discipline.

**Regrid hazard.** On `regrid()` (`amr_runtime.hpp:502-566`, fine `BoxArray` changes at 540) every fine scratch MultiFab is invalid.
> **REVIEW FIX [mpi-krylov E2 + recon-extrap G7 -- `reattach` is insufficient; rebuild the whole solver].** A `reattach(l, ba, dm)` that re-allocates only scratch MultiFabs is **not enough**. After regrid the stepper *also* holds stale: (a) per-level `GeometricMG` precond instances (built on each level's `BoxArray`), (b) `CoverageMask`/`CoarseFineInterface` (built from `fine_ba`, `amr_patch_range.hpp:152-155`), (c) the `AmrTensorKrylovSolver`'s own per-level Krylov work vectors (`r_,p_,v_,s_,t_,phat_,shat_`, analogues of `krylov_solver.hpp:285-286`). **Cleanest fix:** since the solver is built lazily at first `step()`, **rebuild the entire stepper/solver after regrid** (full lazy re-construction), not a partial reattach. Wire this into `AmrRuntime::regrid()` after R7 (546-548). Scratch holds no time-persistent data (rewritten every step), so re-construction needs no prolongation. A forgotten rebuild of *any* of (a)-(c) gives silent wrong answers (stale mask → wrong covered cells) or out-of-bounds (stale work vectors), and crashes V5 on the first regrid.

### 2.2 Per-level assembly: `A_l` and condensed RHS `rhs_l`

**`A_l = I + θ²dt²α ρ_l B_l^{-1}`** -- pure reuse, indexed by `l`: `ElectrostaticLorentzCondensation::assemble_operator((*b.levels)[l].U, bz_l, geom_l, bcPhi_, eps_x_[l], eps_y_[l], a_xy_[l], a_yx_[l])` (`condensed_schur_source_stepper.hpp:252-253`), with `geom_l` at `dx_l = dx_coarse/2^l`. `SchurOperatorCoeffKernel` (`schur_condensation.hpp:67-85`) is cell-local -- no cross-level coupling in assembly. There is **no restriction of `A`** -- each level assembles from its own `ρ_l` at full resolution.

**`rhs_l = -Δ_h phi^n_l - θdt α div(ρ^n B^{-1} v^n)_l`** -- reuse `assemble_rhs(phi_l, U_l, bz_l, geom_l, bcPhi_, rhs_[l])` (`schur_condensation.hpp:210-249`). The `-Δphi^n` term is the **plain 5-point Laplacian** (`assemble_rhs` line 217-222, no tensor coef).

> **REVIEW FIX [ellip-cf C8 -- RHS `phi^n` ghost and matvec `phi` ghost must use the SAME C/F transfer].** If `assemble_rhs`'s internal `fill_ghosts(phi_n)` uses `coupler_inject_aux_mb` (constant) while the matvec uses a proper interpolation (§2.3/2.6 below), the residual `rhs - A phi` at `phi=phi^n` is non-zero on the C/F ring even when it should be → a spurious interface source. **Route both `phi^n` (RHS) and `phi` (matvec) through the same C/F interpolation routine.**

> **REVIEW FIX [ellip-cf C2/G8 -- C/F face coefficient must be shared; coefficient-ghost accuracy is NOT free].** The matvec uses a **harmonic mean** of center and neighbor coefficients per face (`poisson_operator.hpp:153-156`). At the C/F ring the neighbor is the injected-coarse `A`; a harmonic mean of fine `A` and constant-injected coarse `A` is **not** the face coefficient the coarse side sees, breaking `F_c = ½(F_{f,0}+F_{f,1})` at the *coefficient* level even with exact flux reflux. **Define one canonical C/F face coefficient** (restrict fine `A` to the coarse face, or evaluate `A` from interpolated `(ρ,B_z)` at the face) and use it on **both** sides. The original §2.1 claim "constant injection of A is sufficient because conservation comes from the flux match" is wrong here.

### 2.3 Coarse-fine elliptic flux (the hard, load-bearing part)

**Conservation statement.** For `L phi = -div(A grad phi)`, face-normal flux `F = A grad phi`. At a ratio-2 C/F interface (2D → 2 fine faces per coarse face): `F_c == ½(F_{f,0}+F_{f,1})`. The diagonal harmonic term *is* a clean face-normal flux → averaging two fine to one coarse is exact. The reflux moves the coarse bordering equation onto the averaged fine flux, **inside the operator**:
```
out[l-1](c) = (interior coarse fluxes) - (½ Σ fine-face F)/dx_c
```

> **REVIEW FIX [ellip-cf C2 + recon-extrap G3 -- the cross term does NOT telescope, and `cross_div` is not a face flux].** `cross_div` (`poisson_operator.hpp:101-124`) computes `d_x(A_xy d_y phi) + d_y(A_yx d_x phi)` with **4-corner tangential averages** -- it is a *cell-centered divergence*, not a per-face normal flux you can store in a register. The tangential neighbours straddling a C/F face are at **different resolutions on the two sides**, so `F_c^cross ≠ ½(F_{f,0}^cross + F_{f,1}^cross)` even for smooth phi; the register difference converges to a nonzero `O(dx)` tangential mismatch -- leaking the antisymmetric part **exactly at the cartesian ring edge** where the known diocotron overshoot lives (`project_adc_cpp_overshoot`). The original "reuse `FluxRegister`/`route_reflux` verbatim" is **not structurally available** for the cross term. **Resolution, pick one and state it:**
> - **(option 1, Tier 1 true)** Refactor `cross_div` to *emit* a face-normal cross flux `F_x^cross = A_xy^face · (∂_y phi)^face` evaluated from a **C/F-conforming** tangential derivative (interpolate the fine tangential values to the coarse tangential locations and vice versa -- the MLMG cross-term treatment). This touches `poisson_operator.hpp` (contradicting the original §7 "no modification") and is the genuinely new, un-budgeted code.
> - **(option 2, honest Tier 0/1-symmetric)** Reflux **only the diagonal harmonic flux**; document conservation as exact for the symmetric part and `O(dx)` for the antisymmetric part. Since the antisymmetric part is `O(θ²dt²α B_z)` (small at source CFL), this may be acceptable -- **but then the V0 Gauss gate must assert `O(dx)`, not `1e-13`** (see §4 V0).

**Reflux bookkeeping reuse.** The transport `FluxRegister` buffer + deterministic `gather()` (`amr_patch_range.hpp:83-100`) and `CoverageMask` (107-124) are reusable for *bookkeeping*; write `route_elliptic_reflux` paralleling `route_reflux` (162-178) but **omit `·dt`** (purely spatial) and land the correction in the operator's `out`, not the state/RHS.

> **REVIEW FIX [ellip-cf C9 -- re-derive the per-face sign; do not copy transport's literally].** `route_reflux` signs encode transport's `cL/cR` flux orientation. For `-div`, the C/F face contribution at the bordering coarse cell is `-(±F_face)/dx_c` with the sign set by the **outward normal of the coarse cell** and the operator being `-div(A∇phi)`. State the convention explicitly and **unit-test each of the 4 faces independently** (V0b) -- a flipped single-face sign still passes a symmetric-patch test by cancellation.

### 2.4 `AmrTensorKrylovSolver` + MPI

New `include/adc/numerics/elliptic/amr_tensor_krylov_solver.hpp` -- multi-level BiCGStab. Reuses the BiCGStab skeleton + `KrylovResult` (`krylov_solver.hpp:56-60,104-...`), `apply_laplacian` full-tensor matvec (`poisson_operator.hpp:137-179`), `saxpy/lincomb` (`mf_arith.hpp:82,114`), `all_reduce_sum` (`comm.hpp:56`).

**Global reductions.** One `all_reduce_sum` per inner product (level-summed local dots), covered-coarse cells excluded.
```cpp
Real dot_amr(const std::vector<MultiFab>& X, const std::vector<MultiFab>& Y) {
  Real loc = 0;
  for (int l = 0; l < nlev_; ++l) loc += local_dot_uncovered(X[l], Y[l], l);  // global-index covered test
  return all_reduce_sum(loc);   // ONE collective, all ranks
}
```
> **REVIEW FIX [mpi-krylov A1 -- do NOT reuse `dot()`; coverage test is in global coords].** The existing `dot()` ends with its own `all_reduce_sum` (`mf_arith.hpp:152`); calling it per level does `nlev_` collectives per inner product. **Write a new local-only dot** (`reduce_sum_cell`+`DotKernel`, `mf_arith.hpp:71-77,150`) and call `all_reduce_sum` **exactly once**. The §5/§7 "reuse `dot`" line is wrong. `CoverageMask` is indexed in **global coarse coords** (`amr_patch_range.hpp:110,152-155`) while the local fab loop is in fab-local index space -- the covered test must map `(i,j)_local → (I,J)_global` and query `cmask_[l].covered(I,J)` (mask for level `l` built from level `l+1`'s fine `BoxArray`). A coordinate-system mismatch here is a silent wrong-answer bug.

> **REVIEW FIX [mpi-krylov A2 -- pin the L2 norm path].** The codebase has two norm conventions: `GeometricMG::current_residual` uses `norm_inf`+`all_reduce_max` (`geometric_mg.hpp:463`), `TensorKrylovSolver` uses L2=`sqrt(dot)` (`krylov_solver.hpp:260`). **All four AMR norm/dot quantities (`bnorm,rnorm,snorm` + dots) must go through the covered-excluded L2 path.** Do **not** reuse `current_residual`.

**Multi-level matvec.**
```cpp
void apply_operator_amr(const std::vector<MultiFab>& in, std::vector<MultiFab>& out) {
  for (int l = nlev_-1; l >= 0; --l) {
    fill_cf_ghosts(in, l);                 // interpolating C/F ghost of in[l-1] into in[l] ring
    fill_boundary(in[l], dom_l, bcPhi_);   // fine-fine + physical
    apply_laplacian(in[l], out[l], eps_x_[l], eps_y_[l], a_xy_[l], a_yx_[l]);
  }
  for (int l = nlev_-1; l >= 1; --l) reflux_matvec(in, out, l);        // §2.3, in operator
  for (int l = nlev_-1; l >= 1; --l) enforce_covered_slave(in, out, l); // phi_c = avg(phi_f), then zero in vectors
}
```
> **REVIEW FIX [mpi-krylov C1 -- every rank iterates ALL levels; no `local_size()==0` short-circuit].** `fill_boundary`/`fill_ghosts` (`krylov_solver.hpp:182`→`fill_boundary.hpp:131+`, `physical_bc.hpp:158-160`) post MPI sends/recvs enumerated over the **global** `BoxArray`; `mf_fill_fine_ghosts_mb` non-replicated path uses `parallel_copy` (`amr_subcycling.hpp:242-255`), also collective. Under AMR many ranks own **zero** fine fabs, but they **must still call** `fill_boundary`/`parallel_copy`/`average_down`/reflux-`gather` for every level or the matvec **deadlocks**. **Invariant: the per-level matvec and precond bodies are entered unconditionally on every rank for every level; empty-rank-on-level contributes zero but participates in every collective. No `if (local_size()==0) continue` anywhere.** Pin the level traversal order (finest→coarsest) as a documented invariant; forbid rank-dependent branching [mpi-krylov C2].

> **REVIEW FIX [ellip-cf C6 + recon-extrap G4 -- C/F ghost for the matvec/gradient must be interpolation, NOT constant injection].** `mf_fill_fine_ghosts_mb` with `Po==Pn` (spatial-only) does **piecewise-constant** `fill_cf_ghost_cell` -- sets `phi_ghost = phi_coarse` at the coarse *center*, which is `dx_f/2 + dx_c/2` from the fine center, not `dx_f`, mis-scaling the normal derivative by ~3× at a ratio-2 interface. For an elliptic matvec under Krylov this is not `O(dx)`-benign (as in transport) -- it makes the operator **inconsistent at the C/F line** and BiCGStab converges to a phi with a spurious C/F gradient layer. **Use a geometric-offset-aware (≥linear, MLMG-style) C/F interpolation**, and the **same** routine must feed (a) the matvec ghost, (b) the RHS `phi^n` ghost (§2.2/C8), and (c) the velocity-reconstruction gradient ghost (§2.6/G4) -- otherwise the conservation won in §2.3 is thrown away in reconstruction.

### 2.5 Preconditioner

**Tier 0 (first):** per-level independent `GeometricMG` V-cycles on the **symmetric part** `A_sym,l = (A+Aᵀ)/2` (drop cross terms; `set_epsilon_anisotropic` **without** `set_cross_terms`). Block-Jacobi-over-levels -- reuses `GeometricMG` verbatim, one instance per level.

> **REVIEW FIX [mpi-krylov D2 -- this is NOT "low-severity / merely weak"; it can break BiCGStab].** A preconditioner must be a **fixed linear operator**. Two concrete failure modes the original design missed:
> - **Affine drift.** The mono `apply_precond` subtracts the affine Dirichlet offset `d_bc` (`krylov_solver.hpp:222-234`) to keep `M^{-1}` linear. **Each per-level precond application must replicate its own `d_bc` subtraction**, or the precond becomes affine and BiCGStab accumulates a parasitic constant (the failure the mono header documents at 208-213).
> - **Definiteness loss.** `A_amr` couples levels (reflux + covered-slave rows) but block-Jacobi `M^{-1}` does not, so `M^{-1}A_amr` can lose definiteness → `ω→0` breakdown. This is "preconditioned operator loses definiteness," not "more iterations." **Keep flexible GMRES (FGMRES) as a named fallback** (BiCGStab tolerates a *variable* precond poorly; FGMRES is built for it).

> **REVIEW FIX [ellip-cf C7 -- per-level precond needs Dirichlet-ZERO at the C/F boundary, not `bcPhi_`].** Each fine-level `GeometricMG` precond is built on its own patch `BoxArray` with physical/periodic BC at the patch edge -- but the correct **linearized C/F BC for a correction is homogeneous Dirichlet** on the C/F ring (physical/periodic only on true domain faces). Without this the precond smooths toward the wrong boundary and **re-injects a persistent C/F error mode every iteration** → the fine residual *plateaus* (V3 would then wrongly conclude "Tier 0 precond inadequate" for the wrong reason). Set per-level patch-edge BC = Dirichlet-0 on C/F faces.

**Tier 1 (only if V3 demands):** a separate `AmrMgPreconditioner` (MLAT V-cycle across AMR levels) using per-level `GeometricMG` as smoothers/bottom-solvers + inter-AMR-level transfers. **Do not** retrofit `GeometricMG::vcycle_rec` (hard-wired to its own domain-coarsening `lev_[]`, `geometric_mg.hpp:525-566`).

> **REVIEW FIX [mpi-krylov D3 -- verify diagonal dominance at the C/F ring; antisym grows with refinement].** Two AMR threats to `A_sym,l` SPD-ness: (a) at the C/F ring the diagonal is from fine `ρ_l` but face coefs are coarse-injected -- **diagonal dominance is not guaranteed**; add a cheap per-row diagonal-dominance assertion in V0/V2 incl. the C/F ring, and if it fails assemble the symmetric face coef from a **harmonic mean of fine values** on the ring. (b) The dropped antisymmetric part scales with `1/dx_l` at the C/F face, so its size **relative to the kept symmetric part grows with level** -- the "small at source CFL" is a mono statement; **V3 must compare iteration counts at 2 vs 3 levels with CFL held in physical units** (fixed `dt`, shrinking `dx_l`).

> **REVIEW FIX [mpi-krylov E1 -- `op != precond` must be an `nlev_`-wide pairwise invariant].** The mono assert (`krylov_solver.hpp:87`) guards one pair. Assert distinctness for **every level** *and* that no two levels alias the same `GeometricMG` instance (e.g. a default-constructed precond vector all pointing at `lev_[0]`) -- one aliased pair silently corrupts that level's solve. Keep operator coefficients in the stepper's own MultiFab vectors (§2.1) separate from the precond's `GeometricMG` instances.

### 2.6 Reconstruction / extrapolation (per level)

Pure per-level reuse of the mono kernels; the only AMR addition is C/F ghost fills before gradient reads and a final cascade. After the composite solve yields `phi_θ,l`:

1. **C/F-interpolating + `fill_boundary` of `phi_θ,l`** -- precede `condensed_schur_source_stepper.hpp:270`'s `fill_ghosts(phi)` with the **same interpolating** C/F fill as the matvec (§2.4/C6). The reconstruction `∂_x phi, ∂_y phi` (`SchurReconstructKernel`, lines 94-95) reads `phi(i±1,j±1)` across the patch edge.
   > **REVIEW FIX [recon-extrap G4 -- restated, this is where conservation is silently lost].** `SchurReconstructKernel`'s centered gradient straddling the C/F line, fed by **constant injection**, is `O(dx)`-inconsistent and pushes that error straight into `m = ρ v_θ` at the ring edge (the overshoot mechanism). The C/F gradient ghost **must** use the same interpolating routine as the matvec, term-for-term, so the `B v = v^n - θdt∇phi` identity (`condensed_schur_source_stepper.hpp:81`) holds at the interface. §5.1's original "coarse-inject" wording contradicts §4.2's `mf_fill_fine_ghosts_mb` -- unify on interpolation.
2. **`SchurReconstructKernel`** per level → `v_θ = B^{-1}(v^n - θdt∇phi_θ)`, writes `m = ρ^n v_θ`. Verbatim, `half_idx_l = 1/(2 dx_l)`.
3. **`SchurExtrapolateScalarKernel`** (phi) + **`SchurExtrapolateVelocityKernel`** ((v,m)): `f^{n+1}=f^n+(1/θ)(f^{n+θ}-f^n)`. Verbatim, per level.
4. **`SchurEnergyKernel`** per level if Energy role present. Verbatim.
5. **Coverage cascade (mandatory):** `for l=nlev-1..1: mf_average_down_mb(phi_[l],phi_[l-1])` and same for state `m` comps -- the invariant-restore the runtime already does (`amr_runtime.hpp:456-458,471`). Reuse verbatim.
6. **Final `fill_boundary`** of state + phi per level (`condensed_schur_source_stepper.hpp:308-311`, `bcU_default()` mirroring `bcPhi_`). `ρ` stays frozen on all levels (header invariant 54-55); `average_down(ρ)` done at step start makes the C/F momentum recomposition consistent.

### 2.7 Python API + binding + step insertion

Remove the `isinstance(time, Split)` rejection guards in `python/adc/__init__.py` (`add_block` ≈1240-1244, `add_equation` ≈1287-1295). Reuse the **existing** `adc.Strang`/`adc.Split`/`adc.CondensedSchur` descriptors (no new `AmrStrang` class -- they are topology-agnostic). Route as `System.add_equation` (912-924): add the transport block, then `self._s.set_source_stage(name, kind, theta, alpha)` + `self._s.set_time_scheme(scheme)`.

C++ `set_source_stage`/`set_time_scheme` on `AmrSystem` (header `amr_system.hpp`, impl `python/amr_system.cpp`, pybind `python/bindings.cpp`) with the **same validation chain** as `system.cpp:922-990`: kind=="electrostatic_lorentz"; θ∈(0,1]; cartesian geometry; Density/MomentumX/MomentumY roles present; **B_z mandatory** (`set_magnetic_field` + `ensure_aux_width(kAuxBaseComps+1)`, `system.cpp:969-973`). Store spec per block; build `AmrCondensedSchurSourceStepper` lazily at first `step()` (after hierarchy/layout final).

**Step insertion (`AmrRuntime::step`, `amr_runtime.hpp:589-630`).** Lie: insert `amr_schur_step(dt)` **after** `coupled_source_step(dt)` (628), before `++macro_step_`. The stage freezes `phi_n_[l]`/`v_n_[l]`, copies B_z from `aux_[l]`, assembles `A_l`/`rhs_l` (§2.2), runs the solver (§2.4), reconstructs/extrapolates (§2.6).

> **REVIEW FIX [recon-extrap G1/G2 + recon-extrap G11/ellip-cf C11 -- resolve the phi data-flow contract BEFORE Strang, and pick `phi^n` source].** Two coupled problems:
> - **Strang collision (G2).** The proposed Strang sequence `H(dt/2)→solve_fields→S(dt)→solve_fields→H(dt/2)` inserts `solve_fields()` *between* the Schur stage and the 2nd half-transport, and `solve_fields` **re-solves electrostatic Poisson and overwrites the Schur phi** (`HOFFART_STEP_SEQUENCE.md:61` calls this "fatal" for Strang). The 2nd `H` then reads the electrostatic phi, not `phi^{n+θ}`. **Fix:** add a new `AmrRuntime` entry that **publishes the Schur phi into `aux` via `field_postprocess` only (no Poisson re-solve)** before the 2nd half-step. Note `amr_runtime.hpp` exposes **no half-step hook today** -- Strang requires real new orchestration, not a re-wire.
> - **`phi^n` provenance (C11).** On fine levels `aux_[l].phi` is the *injected* coarse phi, not a fine elliptic solution. Seeding `phi_n_[l]` from it while the Schur solve produces a *composite* `phi^{n+θ}` makes the extrapolation `phi^{n+1}=phi^n+(1/θ)(phi^{n+θ}-phi^n)` mix an injected `phi^n` with a composite `phi^{n+θ}` -- a spurious "injected-vs-composite" component, not time evolution. **Decide and document:** either `phi^n` for the Schur stage is itself a composite solve (replace `solve_fields` provenance), or prove the extrapolation error is bounded. The original §0 reframe (injected fine phi inadequate) contradicts silently assuming `solve_fields` is an acceptable `phi^n` provider.

**GaussPolicy.** No `GaussPolicy` type exists in the code today (grep-confirmed).
> **REVIEW FIX [recon-extrap G1 -- GaussPolicy is about the phi-restart, not Tier 0/1].** The prompt's "GaussPolicy" is the documented **once-per-step Poisson re-solve that discards the Schur phi** (`HOFFART_FIDELITY.md` Gauss-restart row), **not** a Tier-0-vs-Tier-1 reflux knob. The source stage does not change `ρ` (frozen, `condensed_schur_source_stepper.hpp:54`), so the charge-side Gauss law is untouched regardless. If a `GaussPolicy` type is later introduced it should select: `Restart` (current behavior -- re-impose Gauss each step, Schur phi is a warm-start only) vs `Evolve` (paper restart-free `-Δphi` evolution -- suppress the re-solve, forward Schur phi). The Tier-0/Tier-1 *reflux* choice is a separate, secondary `ConservationPolicy`. Wire the phi-restart decision (§3 R0) before exposing any policy enum; until then hard-wire the decision made in R0.

---

## 3. Open correctness risks (from the adversarial review -- NOT fully resolved, ranked)

**R0 [recon-extrap G1] -- CRITICAL, chantier-defining, UNRESOLVED.** The top-of-step `solve_fields()` overwrites the Schur phi, making the entire composite elliptic solve **inert on the trajectory's dominant pathway** (it influences the run only through reconstructed `m`). The reframe in §0 identified the wrong decisive fact. **Gate the whole chantier on resolving the phi data-flow contract** (suppress-and-forward vs accept-and-right-size) before building §2.3-2.4. Cheapest first step: measure growth-rate sensitivity to solver tol on the existing **mono** path -- if insensitive, Tier 1 reflux to machine precision is over-engineering.

**R1 [recon-extrap G5] -- CRITICAL feasibility/scope, UNRESOLVED.** The measured diocotron rate is **polar**; the polar solver's `RadialLine` precond needs full-radius boxes, structurally incompatible with AMR fine patches. If the deliverable is the polar measurement, this cartesian design is wrong-geometry. **Decide target geometry first.**

**R2 [ellip-cf C2 / recon-extrap G3] -- CRITICAL, partially resolved (decision pending).** The cross-term flux does not telescope across C/F and `cross_div` is not a face flux. Resolved *to a decision* (§2.3 option 1 conforming cross-flux vs option 2 diagonal-only + relaxed gate), but **the conforming cross-flux code is un-built and un-budgeted**, and touches `poisson_operator.hpp`. This is the dominant *implementation* risk and sits exactly at the cartesian ring edge of the known overshoot.

**R3 [recon-extrap G2] -- CRITICAL for Strang, partially resolved.** Strang needs a phi-publish-without-resolve entry and a half-step hook that does not exist today. Design given (§2.7), not yet implemented.

**R4 [mpi-krylov B1] -- HIGH, UNRESOLVED.** **"Bit-identical across np" is false** -- there is **no deterministic global reduction** in the codebase (`mf_arith.hpp:138-140` explicitly: dot is same-value-all-ranks-at-fixed-np, not across np; changing np changes the fab partition → different summation tree). The historical "bit-identical" GH200 results were on the *transport/reflux* path (`FluxRegister::gather`=`all_reduce_sum_inplace`, fixed-size deterministic), **not** Krylov dots. **Fix the acceptance criterion** (V4/V6) to *rank-invariant convergence + scaled-tolerance solution agreement*, OR add a deterministic reduction (FluxRegister-style canonical-order sum / Kahan) if tight np-invariance is truly required. The growth-rate fit window is narrow (±0.024 band) -- quantify the reassociation noise against it [recon-extrap G9].

**R5 [mpi-krylov D2/D3] -- HIGH, partially resolved.** Block-Jacobi-over-levels precond can be affine (missing per-level `d_bc` subtraction) or indefinite (level coupling in `A_amr` not in `M^{-1}`) → BiCGStab breakdown, not just slowdown. Fixes specified (§2.5), but **FGMRES fallback is named, not built**, and definiteness is asserted, never proven. V3 must log breakdown-guard firing (`ρ/ω` underflow), not just `iters`.

**R6 [ellip-cf C6 + recon-extrap G4] -- HIGH, resolved-in-design.** First-order C/F ghost makes the matvec inconsistent at the interface and corrupts the reconstructed `m` at the ring edge. Fix (geometric-offset interpolation, shared across matvec/RHS/reconstruction) is specified but is **new kernel work** coupled to R2's conforming-flux machinery.

**R7 [ellip-cf C7] -- MEDIUM, resolved-in-design.** Per-level precond must use Dirichlet-0 C/F BC, not domain `bcPhi_`, else fine residual plateaus and V3 misjudges Tier 0.

**R8 [ellip-cf C9] -- MEDIUM, resolved-in-design.** Per-face reflux sign must be re-derived for `-div`; symmetric-patch tests mask single-face sign flips → per-face V0b tests required.

**R9 [ellip-cf C2/G8] -- MEDIUM, resolved-in-design.** Harmonic-mean C/F face coefficient of fine-vs-injected-coarse `A` breaks `F_c=½ΣF_f` at the coefficient level even with exact reflux → define one shared C/F face coefficient.

**R10 [ellip-cf C8] -- MEDIUM, resolved-in-design.** RHS `phi^n` ghost and matvec `phi` ghost must use the same C/F transfer or the residual at `phi=phi^n` carries a spurious interface source.

**R11 [recon-extrap G6 / ellip-cf C3] -- MEDIUM, resolved-in-design.** Covered-cell identity rows (O(1)) mixed with operator rows (O(1/dx²)) destabilize `ω`; resolved by eliminating covered DOFs entirely (slaves zeroed in all Krylov vectors), not just masking dots.

**R12 [mpi-krylov C1 + recon-extrap G7 + mpi-krylov E2] -- MEDIUM, resolved-in-design.** Matvec deadlock if any rank short-circuits an empty level; regrid invalidates precond/mask/work-vectors (rebuild whole solver). Both specified; easy to regress if not encoded as invariants.

---

## 4. Staged validation plan (CPU-laptop reachable vs ROMEO-GH200-only)

| Stage | What | Where | Pass criterion |
|---|---|---|---|
| **B-A..E** | Phase B Tests A-E (§1.9) | **CPU laptop, serial** | A bit-identical (`nc==1`) / `allclose` (`nc==4`); B comps read back exact via `coarse_state()`; C injection exact + coarse==avg(fine); D mass `<1e-12`; E centroid shifts. |
| **R0-probe** | **Growth-rate sensitivity to solver tol on the existing MONO path** (the cheap experiment that decides whether Tier 1 is worth building) | **CPU laptop, serial** | Quantify d(rate)/d(tol). If rate insensitive → Tier 0 / loose tol suffices; do **not** build Tier 1 reflux. **Gate R0 before any §2.3-2.4 code.** |
| **V0a** | Uniform grid, **full tensor**, random phi: `Σ_Ω L phi == Σ_∂Ω F·n` to `1e-13` (validates conservation form -- `cross_div` *is* conservation-form globally) | **CPU laptop, serial** | `~1e-13`. |
| **V0b** | 2-level grid: composite operator row-sum telescopes; **per-face** sign tests; covered-cell slave identity `phi_c==avg(phi_f)` | **CPU laptop, serial** | `1e-13` for the **diagonal** part; `O(dx)` for the cross term **unless** the conforming cross-flux (§2.3 option 1) is built [ellip-cf C5]. Distinguishes "operator not conservative" / "cross doesn't telescope" / "reflux arithmetic wrong" -- three different bugs. Also assert per-row diagonal dominance incl. C/F ring [mpi-krylov D3]. |
| **V1** | Single AMR Schur step, **Tier 0** (no reflux), serial 2-level vs uniform-fine | **CPU laptop, serial** | Bounded `O(dx)` C/F discrepancy; interior matches uniform to solver tol (`1e-10`). Validates per-level physics + plumbing. |
| **V2** | Single AMR Schur step, **Tier 1** (reflux on), serial 2-level vs uniform | **CPU laptop, serial** | `‖phi_amr-phi_uniform‖` on covering region `≤ C·solver_tol`, no C/F spike in `m`. Headline conservation pass. |
| **V3** | BiCGStab convergence audit: log `iters`, `rel_residual`, **breakdown-guard firing** (`ρ/ω` underflow), at 2 and 3 levels with **CFL in physical units** | **CPU laptop, serial** | Decide Tier 1 precond / FGMRES need [mpi-krylov D1/D2/D3]. If iters ≲ 2× mono and no breakdown → ship Tier 0 precond. |
| **V4** | MPI rank-invariance np=1/2/4, multi-box fine level, single Schur step | **CPU laptop** (local MPI ranks -- **no ROMEO**) | **Rank-invariant convergence (same `iters`/`rel_residual` to all ranks at fixed np) + solution agreement across np within `O(eps·√N·cond)`-scaled tol -- NOT bit-identity** [mpi-krylov B1]. No deadlock on empty ranks. Quantify reassociation noise vs the ±0.024 rate band [recon-extrap G9]. |
| **V5** | Multi-step diocotron (Lie then Strang), 2-level vs uniform; watch the known `l`-dependent overshoot | **CPU laptop** serial + np=2 | Mode growth matches uniform within documented tol; **no new C/F-localized overshoot** beyond the pre-existing structural one. Strang requires R3 entry built first. |
| **V6** | **GPU device-clean + bit-identity on GH200**: named-functor kernels (no extended lambdas, the nvcc limit), single + multi-box, np=1/2/4 GPUs | **ROMEO GH200 ONLY** (no CUDA on laptop; `rsync` not clone, direct `g++`/`nvcc`, account `r250127`/user `rmdraux`) | A==B device parity vs CPU within fp tol; multi-GPU MPI **rank-invariant convergence** (NOT bit-identical Krylov dots -- Kokkos dot is not bit-identical between backends, `mf_arith.hpp:9,138`; only the `FluxRegister::gather` reflux path is deterministic) [mpi-krylov B1]. |

ROMEO-only = **V6**. Everything through V5 (incl. MPI up to np=4) runs on the laptop with local MPI ranks. ROMEO is required solely to prove the named-functor kernels are device-clean and multi-GPU MPI convergence is rank-invariant.

---

## 5. New/changed files checklist (exact paths + one-line purpose)

**Phase B -- changed:**
- `include/adc/runtime/amr_system.hpp` -- `AmrBuildParams` += `has_state,state`; `AmrCompiledBlockBuilder` typedef += `state,has_state`; declare `set_conservative_state`, `coarse_state`.
- `include/adc/runtime/abi_key.hpp` -- **add glob-independent ABI sentinel** (`amr_builder_v=2`) [phaseB C1/H1].
- `include/adc/coupling/amr_coupler_mp.hpp` -- **new** `coupler_write_coarse_state`; **new** `coupler_read_coarse_all`.
- `include/adc/runtime/amr_dsl_block.hpp` -- mono seed branch (90); `build_amr_block`/`dispatch_amr_block` (10 sites)/`multi_builder` += `state,has_state`.
- `python/amr_system.cpp` -- `BlockSpec` += `has_state,state`; `make_build_params` packs state; 2 multi dispatches += args; **new** `set_conservative_state` + `coarse_state` bodies.
- `python/bindings.cpp` -- **new** `set_conservative_state` (with `ndim()==3` guard) + `coarse_state` pybind.
- `adc_cases/hoffart_euler_poisson_dsl/run.py` -- drift-seed `build_amr` (probe in try/except, `nc` guard); docstring (15-17) + metadata (464,482) honesty fix.
- `python/tests/test_amr_conservative_state.py` -- **new** Tests A-E.

**Phase C -- new:**
- `include/adc/numerics/elliptic/amr_tensor_krylov_solver.hpp` -- multi-level BiCGStab; covered-excluded L2 reductions; per-level matvec with reflux + covered-slave; FGMRES fallback hook.
- `include/adc/numerics/elliptic/amr_elliptic_reflux.hpp` -- `TensorFluxRegister` + `route_elliptic_reflux` (diagonal flux; conforming cross-flux if §2.3 option 1).
- `include/adc/coupling/amr_condensed_schur_source_stepper.hpp` -- per-level scratch, `step()`, lazy build, full rebuild-on-regrid.
- `include/adc/numerics/elliptic/amr_mg_preconditioner.hpp` -- *(Tier 1 only)* MLAT V-cycle across AMR levels.
- `python/tests/test_amr_schur_*.py` -- V0a/V0b/V1/V2/V3/V4 drivers.

**Phase C -- changed:**
- `include/adc/numerics/elliptic/poisson_operator.hpp` -- *(only if §2.3 option 1)* `cross_div` refactor to emit C/F-conforming face cross-flux [recon-extrap G3, contradicts original "no modification"].
- `include/adc/runtime/amr_runtime.hpp` -- `amr_schur_step`; Strang restructure (new phi-publish-without-resolve entry, half-step orchestration); **rebuild stepper after `regrid()`**; phi-restart contract (R0).
- `include/adc/runtime/amr_system.hpp` / `python/amr_system.cpp` / `python/bindings.cpp` -- `set_source_stage` + `set_time_scheme` (validation chain mirroring `system.cpp:922-990`).
- `python/adc/__init__.py` -- remove `isinstance(time, Split)` AMR guards; route `set_source_stage`/`set_time_scheme`.

**Unchanged (reused as-is):** `GeometricMG` (except do **not** retrofit `vcycle_rec`), `TensorKrylovSolver`, `schur_condensation.hpp`, mono `condensed_schur_source_stepper.hpp`. *(`poisson_operator.hpp` moves to "changed" only under §2.3 option 1.)*

---

## 6. Suggested PR sequence (small landable steps, each independently testable)

**Phase B:**
1. **PR-B1 -- ABI sentinel + kernels.** `abi_key.hpp` sentinel bump; `coupler_write_coarse_state` + `coupler_read_coarse_all`. Unit-test the kernels directly (write/read round-trip, multi-box). Lands the safety net before any signature change. [phaseB C1/H1, M3]
2. **PR-B2 -- builder threading + `AmrBuildParams`/typedef.** Thread `state,has_state` through `build_amr_block`/`dispatch_amr_block`/`multi_builder`/`AmrCompiledBlockBuilder`/`AmrBuildParams`/`BlockSpec`/`make_build_params`. Density path unchanged. Regenerate loaders. Test: existing AMR suite still green (NO-DEFAULT-CHANGE), full reconfigure.
3. **PR-B3 -- `set_conservative_state` + `coarse_state` + bindings** (with `ndim()==3`, empty-state, `nc`-count guards). Test A (`nc==1` bit-identical), Test B (read-back).
4. **PR-B4 -- Tests C/D/E** (prolongation, mass, motion).
5. **PR-B5 -- `run.py` drift seed** (probe try/except, `nc` guard, fidelity-honest metadata/docstring).

**Phase C:**
6. **PR-C0 -- R0/R1 decision doc + mono tol-sensitivity probe.** No solver code. Resolve phi-restart contract and target geometry; run the mono growth-rate-vs-tol experiment. **Gates everything below.**
7. **PR-C1 -- elliptic operator + V0a.** Per-level `apply_laplacian` wiring + uniform-grid full-tensor conservation-form test (`1e-13`). No AMR yet.
8. **PR-C2 -- C/F interpolation kernel** (geometric-offset, shared matvec/RHS/reconstruction) + shared C/F face coefficient. Unit-test gradient consistency at a static C/F interface. [R6/R9/R10]
9. **PR-C3 -- `route_elliptic_reflux` (diagonal) + `TensorFluxRegister` + V0b.** Per-face sign tests, covered-slave identity, diagonal Gauss to `1e-13`, cross to `O(dx)`. [R2 option 2 baseline, R8]
10. **PR-C4 -- `AmrTensorKrylovSolver` (Tier 0 precond)** + covered-excluded L2 reductions + matvec deadlock invariants. V1 (Tier 0 vs uniform) + V3 (convergence/breakdown audit). [R4 criterion, R5, R11, R12]
11. **PR-C5 -- `AmrCondensedSchurSourceStepper` + lazy build + rebuild-on-regrid** + `set_source_stage`/`set_time_scheme` + Python routing (Lie only). V2 (Tier 1 diagonal reflux vs uniform). [R3 Lie path]
12. **PR-C6 -- conforming cross-flux** *(only if V0b/V2 show the cross-term leak matters per R0/R2)*: `cross_div` face-flux refactor + tighten V0b cross to `1e-13`.
13. **PR-C7 -- Strang orchestration** (phi-publish-without-resolve entry, half-step hook). V5 (multi-step Lie+Strang). [R3 Strang path]
14. **PR-C8 -- MPI** (V4 rank-invariant convergence) then **ROMEO V6** (device-clean + multi-GPU). [R4]
15. **PR-C9 -- Tier 1 MG precond** *(only if V3 demands)*: `amr_mg_preconditioner.hpp` + FGMRES fallback. [R5]

Each PR is independently testable on the CPU laptop except V6 (ROMEO). PR-C0 is a hard gate: do not build §2.3-2.4 machinery until the phi-restart contract (R0) and geometry (R1) are resolved.
