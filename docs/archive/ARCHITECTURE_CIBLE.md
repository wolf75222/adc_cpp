# Target architecture (north star)

VISION doc (not the current state), drawn from the supervisor's description. To re-read before the
whiteboard session. Two levels:
- **(A)** the layered OO architecture (composable models): ALREADY largely realized in adc_cpp;
- **(B)** the symbolic DSL where Python WRITES the formulas: CPU interpreter + C++ codegen
  (flux / brick / source / elliptic) + CSE + JIT dlopen of the kernel done (`adc.dsl`); remaining are the
  dispatch in the template solver (type-erased) and Kokkos/CUDA + GPU run.

---

## 0. Principle (to carve in stone)

```
Un modele physique ne sait pas avancer le temps.
Un schema spatial ne sait pas ce qu'est Euler.
Un flux numerique ne sait pas ce qu'est un electron.
Un EPM lit le systeme entier.
Une source couplee lit plusieurs blocs.
Le driver orchestre, mais ne contient pas la physique.
```

PDE:
```
HPM : d_t U + div F(U, aux) = S(U, U_all, aux)
EPM : D(phi, aux) = f(U_all, aux)
Chaine : Variables -> Flux physique -> Flux numerique -> SpaceMethod -> TimeMethod
         CoupledSystem orchestre HPM + EPM + sources couplees.
```

---

## 1. Target tree (condensed)

```
solver/
  core/        scalar, state_vector, field, direction, variable, concepts
  physics/
    hyperbolic/  hyperbolic_model, variables, primitive_conservative, flux, eigenvalues
    source/      source_model, local_source, coupled_source
    elliptic/    elliptic_model, elliptic_operator, elliptic_rhs, elliptic_field
    model/       physical_model, equation_block, coupled_system
  numerics/
    reconstruction/  reconstruction (cons|prim), minmod, vanleer, weno
    flux/            numerical_flux, rusanov, hll, hllc
    space/           space_method, finite_volume
    time/            time_method, explicit, implicit, imex, ssprk, scheduler
  mesh/    mesh, geometry, boundary_conditions, field_layout
  amr/     hierarchy, refinement_criterion, regrid, prolongation, restriction, reflux
  runtime/ registry, factory, simulation, backend
  python/  bindings.cpp
```

### State of the REAL tree (June 2026)

The `include/adc/` tree was reorganized toward this target (the `<adc/...>` prefix is kept):
`core/`, `physics/` (euler + hyperbolic / source / elliptic / composite, cf. the split of `bricks.hpp`),
`numerics/` (flux + spatial_operator + `time/` + `elliptic/`), `mesh/`, `amr/`, `runtime/`,
`coupling/`, `parallel/`. The brick categories are now separate files
(`physics/{hyperbolic,source,elliptic,composite}.hpp`), with `physics/bricks.hpp` remaining an umbrella for
compat. Some remain flat (no fine subdirectories `numerics/{flux,space,reconstruction}`) for lack of a
1:1 file<->concept correspondence; `physical_model.hpp` (concepts) and `variables.hpp` remain in
`core/` (fundamental contracts). Renames: `Variables`->`VariableSet`, `HyperbolicModel`->
`HyperbolicPhysicalModel` (compat aliases kept).

---

## 2. OO layers (and current state)

| Target layer | State in adc_cpp today |
|---|---|
| `Variables` (cons/prim, conversions) | DONE: `core/variables.hpp` (kind, names, size) + `using State/Prim` + `to_primitive`/`to_conservative` on the hyperbolic brick. MISSING: semantic `VariableRole` (Density/MomentumX/...), `index_of(role)`. |
| `HyperbolicModel` (Vars + flux + lambda) | DONE: concept `HyperbolicModel` + bricks Euler / IsothermalFlux / ExBVelocity. MISSING: `eigenvalues()` returning the full VECTOR (we have `max_wave_speed` + signed `wave_speeds`); `Direction` enum (we have `int dir`). |
| `LocalSource` / `CoupledSource` | DONE: local source bricks (PotentialForce/GravityForce/NoSource); coupled ionization/collision/exchange sources (operator-split). MISSING: a proper `CoupledSource` object hierarchy + composition `source = a + b`. |
| `EllipticPhysicalModel` | DONE (1st order): `add_elliptic_model` + bricks (div_eps_grad, charge_density, electric_field_from_potential). MISSING: variable eps(x), alternative operators (diffusion, projection), rhs at the EPM level. |
| `PhysicalModel` = HPM + source + elliptic | DONE: `CompositeModel<Hyperbolic, Source, Elliptic>`. |
| `EquationBlock` (model + space + time + bc + evolve) | DONE: blocks of the runtime System (model + spatial + time + substeps + evolve). |
| `SpaceMethod` / `FiniteVolume` | DONE: `assemble_rhs<Limiter, Flux>` (finite volume, generic over the model). |
| `Reconstruction` (cons|prim) | DONE: recon cons/prim + minmod/vanleer/weno. |
| `NumericalFlux` (Rusanov/HLL/HLLC) | DONE: generic over the model (`m.flux`, `m.max_wave_speed`). |
| `TimeMethod` (explicit/imex/ssprk/scheduler) | DONE: ForwardEuler/SSPRK2 + partial IMEX + multirate (`step_adaptive`). MISSING: FULL implicit, `stride` (every-N) exposed cleanly. |
| `CoupledSystem` + `Assembler`/`Driver` | DONE: the runtime System (vector of blocks + Poisson + couplings); Assembler/Driver split done on the core side. |
| `mesh` / `amr` / `runtime` / `python` | DONE: MultiFab/Geometry/BC, AMR (hierarchy, regrid, reflux), factory (dispatch), Simulation (System), bindings. |

Summary: **layers 1-12 of the OO design are ~80% in place** (different organization and names, but the abstraction is there, and the recent refactor has reinforced it: HyperbolicModel, Variables contract, EPM, coupled sources, primitive recon).

---

## 3. The symbolic DSL (the endgame, NEW)

Goal: Python WRITES the formulas (not a function called per cell), ADC turns them into a solver.

```python
e = adc.dsl.HyperbolicModel("electrons")
rho, rhou, rhov, E = e.conservative_vars("rho", "rho_u", "rho_v", "E")
u = e.primitive("u", rhou / rho)
p = e.primitive("p", (gamma - 1) * (E - 0.5 * rho * (u*u + v*v)))
c = adc.dsl.sqrt(gamma * p / rho)
e.set_flux(x=[rhou, rhou*u + p, rhou*v, rho*H*u], y=[...])
e.set_eigenvalues(x=[u - c, u, u + c], y=[...])
e.set_source([...]); e.set_elliptic_rhs(-qe * rho / me)
```

`rho`, `u`, `p`... are not floats: they are SYMBOLIC EXPRESSIONS. Python builds a
GRAPH of formulas; ADC can then: (1) interpret it on CPU (proto), (2) generate C++,
(3) generate Kokkos/CUDA, (4) JIT, (5) verify the dependencies between variables. `Euler`,
`diocotron`, `two-fluid` become simple Python files of formulas.

Additional tree:
```
adc/symbolic/  expression (Var/Const/Add/Mul/Sqrt...), vector_expr, formula_graph, simplifier, codegen
```

### Honest assessment (engineering)

- It is a **domain compiler**, not a refinement. Realistic effort: **multi-month**, not a
  refactor. The main risk: GPU codegen (a Python graph -> correct and performant Kokkos/CUDA kernel)
  is exactly the hard part.
- **It exists** in pieces: SymPy (C/Fortran codegen), **Pystencils** (stencils -> C/CUDA),
  **Devito** (finite-difference DSL -> optimized C), UFL/Firedrake (finite elements). For the
  hyperbolic finite-volume case (flux + eigenvalues + reconstruction), less off-the-shelf, but
  the foundations exist; do not rebuild a symbolic engine from scratch (reuse SymPy).
- **Tradeoff vs the current state**: today we have two paths, COMPILED bricks (template, GPU/MPI,
  production) + `adc.PythonFlux` (CPU numpy proto). The double-coding (a formula written in C++ AND
  in numpy) is the cost the DSL would remove: ONE source of formulas -> CPU interpreter + generated
  GPU kernel. The gain grows with the number of models (multi-species plasma: many variants).
- **Recommendation**: keep the current compiled solver (it works, GPU-ready) as the production
  target; start the DSL as a **separate prototype** (`adc/dsl.py`), first a CPU interpreter
  of a formula graph (validate the concept on Euler), THEN a C++ codegen. Do NOT rewrite the
  solver while waiting for the DSL to mature.

### State: CPU interpreter PROTOTYPE done (`adc.dsl`)

The `python/adc/dsl.py` module realizes step (1) (interpret on CPU) and (5) (verify the
dependencies):
- expression tree (`Expr`: `Const`, `Var`, `Add/Sub/Mul/Div/Pow/Neg`, `Sqrt`) built by
  operator overloading; `eval(env)` applies it to numpy arrays (the whole domain at once);
- declarative `HyperbolicModel`: `conservative_vars` / `primitive` (formulas) / `aux` / `set_flux` /
  `set_eigenvalues` / `set_source` / `set_elliptic_rhs` / `check()` (dependencies);
- `to_python_flux()` wires the tree onto the host backend `adc.PythonFlux` -> **the model RUNS**.

Verified: `python/tests/test_dsl.py` (symbolic Euler flux == reference numpy flux,
`max_wave_speed` consistent, `check()` detects an undefined variable, mass conserved at runtime)
and the `adc_cases/dsl_euler/` case (Euler written as formulas, acoustic expansion, mass conserved).

Step (2) DONE for Euler (host codegen + wrapping as a brick):
- `emit_cpp()` generates the flux function `template <class Real> void <name>_flux(const Real*, Real*, int)`
  from the tree (each `Expr` node knows how to write itself via `to_cpp()`);
- `emit_cpp_brick()` generates a complete BRICK: a struct (`StateVec` / `Aux` / `ADC_HD`) with flux,
  max_wave_speed, to_primitive, to_conservative, conservative_vars / primitive_vars, which SATISFIES the
  `adc::HyperbolicModel` concept (so usable in a CompositeModel / the solver). The
  non-invertible conversions (to_conservative) are provided by the user (`set_conservative_from`), the DSL
  not knowing how to invert symbolically.

- `emit_cpp_source()` generates a composable SOURCE BRICK (`apply(U, a)`), with aux locals read
  as `a.<field>` (convention: aux names = fields of `adc::Aux`, e.g. grad_x / grad_y).

Verified (all in CI):
- `test_dsl_codegen.py`: generated flux == numpy interpreter;
- `test_dsl_brick.py`: the brick COMPILES against the adc headers, `static_assert(adc::HyperbolicModel<...>)`,
  and == `adc::Euler` (4 vars, no aux) AND `adc::ExBVelocity` (1 var, aux-dependent flux), zero deviation;
- `test_dsl_source.py`: the generated source == `adc::PotentialForce`;
- `test_dsl_compose.py`: `CompositeModel<EulerGen, NoSource, ChargeDensity>` satisfies `adc::PhysicalModel`
  and equals the hand-written version;
- `test_dsl_jit.py`: JIT-lite end-to-end (Python formulas -> generated `.hpp` -> g++ compiles a finite-volume
  driver -> Rusanov residual identical to `adc::Euler`).
Also verified on ROMEO (g++ 11, C++20): compilation and bit-exact equality.

DONE since: elliptic codegen (`emit_cpp_elliptic`, == `adc::ChargeDensity`); CSE (`cse=True` by
default: H / c factored into locals `cseK_`, verified identical to the version without CSE and to `adc::Euler`);
REAL JIT of the kernel (`test_dsl_jitlib`: generated flux -> `.so` compiled on the fly -> loaded into the
Python process via ctypes -> == numpy interpreter).

GPU RUN DONE: the generated brick compiles with nvcc (`-arch=sm_90`) and runs on NVIDIA GH200
(ROMEO, aarch64 node), result BIT-IDENTICAL to `adc::Euler` (cf. docs/GPU_ROMEO.md). It is
device-ready by construction (`ADC_HD` -> `__host__ __device__`, device-safe ops, `std::sqrt`).

KOKKOS DONE (brick verification): the generated brick runs via `Kokkos::parallel_for` on
the `Cuda` execution space (GH200; Kokkos 4.4 + CUDA 12.6, `HOPPER90`), == `adc::Euler` to one ULP
(5.6e-17, FMA contraction), cf. docs/GPU_ROMEO.md. It is the same dispatch primitive as
`adc/mesh/for_each.hpp`.

(a) DONE: TYPE-ERASED interface `adc::IModel<NV>` + `ModelAdapter` (include/adc/runtime/dynamic/dynamic_model.hpp)
AND wiring into the runtime. `System::add_dynamic_block(name, so)` loads at runtime (dlopen) a generated
brick compiled into a `.so` and creates a block driven by the IModel (host Rusanov order 1), advanced via
eval_rhs / step / step_cfl like any block; `dsl.HyperbolicModel.compile_so` does the JIT.
End-to-end verified: `test_dsl_block` (DSL -> .so -> add_dynamic_block; eval_rhs == adc.PythonFlux to
9e-16; 25 steps in the System, mass conserved); `test_dynamic_model` (C++) and `test_dsl_dynamic`
(dlopen from a main ignorant of the type) lock down the mechanism. HOST path (vtable, off GPU;
during COMPILE of PythonFlux). The GPU hot path remains the TEMPLATE path.

(b) DONE: a COMPLETE 2D Euler CASE (80 steps, CFL, Rusanov, periodic) advances on GH200 through the
Kokkos seam of adc (`for_each_cell` / `for_each_cell_reduce_*`), mass exactly conserved, generated
brick == `adc::Euler` to 9e-16 over 80 steps (cf. docs/GPU_ROMEO.md). Remaining: the ENTIRE runtime
stack (System / AMR / MPI) on GPU, and `sim.add_dynamic_block` on the Python side (item (a)).

The host codegen and the type-erased dispatch do NOT replace the production compiled bricks (template
path, GPU/MPI).

---

## 4. Pragmatic path (incremental, without blocking on the DSL)

Small concrete steps that get closer to the whiteboard WITHOUT the compiler:
1. `VariableRole` (Density/MomentumX/.../Pressure/Temperature) + `index_of(role)` on `Variables`:
   gives meaning to the components (useful for coupled sources: "the velocity of such-and-such species").
2. `eigenvalues()` returning the full vector of eigenvalues (beyond `max_wave_speed`).
3. `Direction` enum (X/Y) instead of `int dir` (readability, type-safe).
4. Composition of local sources: `source = ElectricForce(...) + LorentzForce(...)`.
5. Progressive reorg of the directories toward the target tree (cosmetic, to do when touching a corner).

The symbolic DSL (section 3) is the long-term goal, treated as a separate sub-project.
```
