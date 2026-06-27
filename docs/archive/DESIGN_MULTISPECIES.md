# pops : Multi-species milestone : from `PhysicalModel` to `CoupledSystem`

*Design document for the whiteboard session (with Sacha). Updates the roadmap
after the supervisor's remarks and the increments already done.*

---

## 1. The frame (what the supervisor validates, what is missing)

The supervisor **does not call `PhysicalModel` into question**. From the transcript:
"the physical model, I have nothing to say, it looks flexible; the structure with
templates goes in the right direction". `PhysicalModel` is the **right local level**: an
equation for one species (flux, max_wave_speed, source, elliptic_rhs), device-callable.

What is missing is the **level above**: assembling **several** `PhysicalModel`,
each with **its** spatial method, **its** time integrator, **its** time step,
and **global** couplings (Poisson and sources see all species).

```
PhysicalModel   describes a LOCAL equation             (already good)
EquationBlock   = State + Model + Spatial + Time     (done, core)
CoupledSystem   = several EquationBlock               (done, core)
Scheduler       = order of steps, substeps, IMEX     (done, core)
PoissonRHS      = assembled from all species          (done, core)
```

The core guarantees that these choices stay compatible with **AMR / MPI / GPU**.

## 1bis. Posture (what he expects NOW)

Four principles that frame the immediate objective (do not oversell, do not freeze):

- **A skeleton, not the final solver.** The objective is to lay down the right bricks and
  **test them on simple cases**, not to code everything or optimize.
- **Abstraction before data structure.** He distinguishes three levels: (1) abstraction,
  (2) architecture / class interaction, (3) data structure. Level (3) "is easily
  changed later". We first show *who depends on whom, who assembles what*, not the
  memory layout (`MultiFab`, stacked storage, etc.).
- **Performance after stabilization.** "Optimizing, I will know how to do it once the code is
  clean and frozen." Message: good abstractions first, optimization next, not
  the other way around.
- **Validated by a user, not by the compiler.** Sacha is a key user. The
  real test is not "does it compile?" but: **can a user describe
  their physical system without understanding AMR / MPI / GPU?** Central question: *what minimal API
  lets Sacha describe their case?*

**Diffusion = a flux, not a new layer.** The parabolic part is one additional flux
contribution in the spatial operator, not a large abstraction. (Done on uniform grid
via `DiffusiveModel`; on AMR the engine works by **face flux** for the
reflux, so diffusion must go through it as a **diffusive face flux**: this is the
follow-up, not a new level.)

## 2. Translating the supervisor's remarks into requirements

| He says | It means |
|---|---|
| "U for the ions, U for the electrons" | `State` must no longer be unique: several `U_k` fields. |
| "isothermal ions, Euler electrons" | each species has its own `PhysicalModel`. |
| "10 electron steps, 1 ion step" | one `Scheduler`/`TimePolicy` per block (subcycling). |
| "implicit electrons, explicit ions" | the `TimeIntegrator` is **per block**, not global. |
| "Rusanov ions, HLL electrons" | the `SpatialDiscretisation` is **per block**, not global. |
| "they interact in `f(U)` and in `S`" | local `elliptic_rhs(u)` and `source(u,aux)` are too narrow: we need an inter-species **coupling** source + a **global** elliptic RHS. |

The target `Coupler` is no longer "hyperbolic + elliptic" but a **system
assembler**: it takes `{U_e, U_i, U_n, ...}` + spatial methods + time methods +
elliptic solvers + execution order.

## 3. Target vision: C++ abstractions

```cpp
// Un bloc = un état + un modèle + une discrétisation spatiale + une politique temporelle.
template <class Model, class Spatial, class Time>
struct EquationBlock {
  Model    model;
  MultiFab* U;         // vue non-possédante vers l'état du bloc
  BCRec    bc;         // conditions au bord PAR BLOC (cf. ci-dessous)
  // Spatial = SpatialDiscretisation<Limiter, NumericalFlux>
  // Time    = ExplicitTime / ImplicitTime / IMEXTime / PrescribedTime
};

// Un systeme = plusieurs blocs.
template <class... Blocks>
struct CoupledSystem {
  std::tuple<Blocks...> blocks;
};

// Le driver mono-niveau connecte systeme + Poisson + scheduler.
SystemCoupler sim(system, geom, ba, bc_phi, rhs_assembler);
```

- **`EquationBlock`**: done. It groups `PhysicalModel`, `SpatialDiscretisation`,
  `TimePolicy`, `MultiFab` state and boundary conditions.
- **`CoupledSystem`**: done. Assembly layer for N blocks.
- **`Scheduler`**: done. Encodes the order (e.g. 10 electron substeps per ion step),
  the targeted implicit (partial IMEX) via `TimePolicy`.
- **`PoissonCoupling`**: first brick done. `elliptic_rhs.hpp` provides the
  single-model case and the two-field / two-block case; `SystemCoupler` accepts a user
  assembler to generalize `f = Σ_s α_s · model_s.elliptic_rhs(U_s)`.
  Compliant with the request: coupling in `f(U)`, not in `F`.

Non-regression: `Coupler<Model>` stays the single-block path; `SystemCoupler` is added
alongside, without breaking the historical API.

### 3bis. Architecture points to settle

- **Per-block boundary conditions.** Today the `BCRec` are global in the couplers.
  Multi-species requires per-block BC: periodic electrons, wall/extrapolation ions,
  neutrals with imposed profile. -> `bc` in `EquationBlock` (above), no more single global BC.
- **AMR is an orchestrator, not a mesh option.** It takes a definition
  that is *local, cell by cell* and projects it onto a whole hierarchy of grids:
  `fill_ghosts → subcycling → average_down → reflux → regrid`. So `AmrCoupler` /
  `AmrCouplerMP` are not simple technical variants but **execution
  orchestrators**: they already play the role of the future `Scheduler` at the level of a block.
- **`Coupler`: a historical name.** Conceptually this layer is an **Assembler /
  Simulation Driver** (it takes all the ingredients and puts them on the mesh). The name
  "Coupler" comes from "couple hyperbolic + elliptic"; we do not defend it at all
  costs, we accept that it denotes the assembly.
- **Templates, but we draw the concepts.** The supervisor accepts the templates ("it goes
  in the right direction") while noting that it is less readable than virtual. So we
  present the **conceptual objects** (`PhysicalModel`, `SpatialDiscretisation`,
  `TimeIntegrator`, `EquationBlock`, `CoupledSystem`, `Scheduler`), even if technically
  they are templates.

## 4. Target Python API (composition, not computation)

Python **configures** the system; the cell / AMR / MPI / GPU loops stay in C++.

```python
import pops
from pops.numerics.riemann import Rusanov, HLLC
from pops.numerics.reconstruction.limiters import Minmod, VanLeer

mesh = pops.Mesh2D(nx=512, ny=512, xlim=(0,1), ylim=(0,1),
                  amr=pops.AMR(levels=3, ratio=2))
sim  = pops.Simulation(mesh, backend="kokkos")   # cpu / openmp / mpi / kokkos

sim.add_equation(name="electrons",
    model=pops.models.ElectronEuler(charge=-1, mass=1, gamma=5/3),
    spatial=pops.FiniteVolume(reconstruction=VanLeer(), riemann=HLLC()),
    time=pops.Implicit(scheme="imex", substeps=10))

sim.add_equation(name="ions",
    model=pops.models.IonEuler(charge=+1, mass=1836, gamma=5/3),
    spatial=pops.FiniteVolume(reconstruction=Minmod(), riemann=Rusanov()),
    time=pops.Explicit(scheme="ssprk2", substeps=1))

sim.add_poisson(unknown="phi",
    rhs=pops.ChargeDensity(positive=["ions"], negative=["electrons"]),
    solver=pops.GeometricMG(tol=1e-10, max_iter=200))

sim.run(t_end=1.0, cfl=0.4,
        output=pops.Output(path="runs/two_species", every=20,
                          fields=["electrons.rho", "ions.rho", "phi"]))
```

The strings (`flux="hllc"`, `time="imex"`) **select compiled C++ bricks**;
they are never cell-by-cell callbacks (slow, not GPU/MPI-friendly). An
advanced user writes their `PhysicalModel` in C++ (`StateVec<N>`, `POPS_HD`) and then exposes
it to Python, always in composition, never in a Python inner loop.

## 5. Current state vs target

| | Today | Target |
|---|---|---|
| States | a single `U` | `{U_e, U_i, U_n, ...}` |
| Model | one `PhysicalModel` | one per block |
| Spatial | selectable (uniform + AMR) | **per block** |
| Time | SSPRK2/3 selectable at the coupler | **per block** (+ partial IMEX, subcycling) |
| Poisson | `f = model.elliptic_rhs(U)` (1 state) | `f = Σ_s q_s n_s` (all species) |
| Coupler | `Coupler<Model, Elliptic>` single-block | `CoupledSystem<Blocks...>` |
| Python | drives facades (`DiocotronSolver`, ...) | composes a system |

## 6. What is already done (prepares the ground, does not pre-empt the data decision)

All pushed, all green (adc_cpp 30/30, adc_cases 44/44; MPI 7+7):

1. **Core/applications split**: `adc_cpp` = generic engine (zero model), `adc_cases`
   = models/facades/examples/Python, via FetchContent (`pops::pops`).
2. **`SpatialDiscretisation<Limiter, NumericalFlux>`** + tags `SSPRK2`/`SSPRK3` +
   `Coupler::step<Disc, TimeInteg, Policy>`: spatial discretization and time
   integrator **selectable** (the future "per block" will plug into it).
3. **Diffusion as one more flux**: trait `DiffusiveModel` (`+ν∆U` in
   `assemble_rhs`), model `AdvectionDiffusion`. (Uniform grid; AMR = follow-up.)
4. **Elliptic solver chosen at runtime** in the diocotron facade (MG/FFT, `variant`
   pattern).
5. **AMR applies `model.source`** (the AMR path ignored it: correction bug) +
   **`SpatialDiscretisation` wired in AMR** (`AmrCoupler/MP::step<Disc>`, conservative MUSCL).

## 7. Modifications to make (ordered)

**Master decision (at the whiteboard), data design:**
- `tuple<Blocks...>` (each block keeps its `StateVec<n_k>`, variadic composition) **vs**
  stacked `StateVec<N_total>` (one contiguous memory block, per-species offsets). Conditions
  everything else (locality perf, views, GPU). To be settled with Sacha.

**Then, in order:**
1. **`EllipticRhsAssembler` / `PoissonCoupling`**: take `elliptic_rhs` out of the
   single-state path. `f = Σ_s α_s · model_s.elliptic_rhs(U_s)`, shared φ.
   Files: `coupling/coupler.hpp` (`detail::coupler_eval_rhs`), `core/physical_model.hpp`.
2. **`EquationBlock<Model, Spatial, Time>`**: aggregation type (state + policies).
   Adapter `SingleFieldSystem<Model>` for non-regression.
3. **`CoupledSystem<Blocks...>` + `Scheduler`**: assembles N blocks; handles the order of
   substeps and the implicit/explicit per block.
4. **Inter-species coupling source**: distinguish the model's local `source` from
   coupling (collisions, exchange) that sees the other states.
5. **Partial IMEX**: integrator handling implicitly a **subset** of the
   variables (electrons) and explicitly the rest (ions). Trait `which_implicit()`.
6. **Per-species time subcycling**: reuse the AMR subcycling (Berger-Oliger)
   for the blocks (10 electron steps / 1 ion step).
7. **Compiled facade `MultiSpeciesSolver(vector<SpeciesConfig>)`** + **composition
   Python API** (`sim.add_equation(...)`, `sim.add_poisson(...)`, `sim.run()`).
8. **Cleanups**: `SpectralCoupler` must no longer hardcode the diocotron physics
   (call `model.elliptic_rhs` / `model.max_wave_speed`); clarify the `Aux` contract
   (fixed `phi, grad φ` or actually `typename M::Aux`); diffusion on AMR (diffusive face
   flux to stay conservative at the reflux).

## 7bis. Simple cases to test the skeleton

The architecture is validated on simple **user** cases (not a production solver):

- Euler electrons + **isothermal** Euler ions + Poisson (the canonical two-species case);
- **fixed-ion diocotron** (approx. what exists: constant `n_i0`);
- **mobile-ion diocotron** (the ions become a second block);
- **resolved** or **prescribed** gas / neutrals (imposed profile, not resolved each step).

Success criterion: a user (Sacha) can describe these cases **without touching**
AMR / MPI / GPU. If the API does not allow it, the abstraction is incomplete.

## 7ter. Glossary (for the slides, to define before dropping the acronyms)

| Term | Short meaning |
|---|---|
| `BoxArray` | partition of the domain into blocks (boxes) |
| `MultiFab` | the `U` fields stored on these blocks (distributed collection) |
| `BCRec` | boundary conditions of a field |
| `aux` | auxiliary variable transported (here `phi, grad phi`) |
| seam | seam where parallelism lives (`for_each_cell`, `comm`) |

Python note: never call `flux()` cell by cell from Python (slow, not
GPU/MPI). Python **configures** the system, or provides **vectorized** fields (numpy on
all cells). The cell hot path stays in C++.

## 8. Summary (whiteboard sentence)

> `pops` already knows how to take a **local physical law** and run it on a mesh
> with Poisson, AMR, MPI and GPU. What is missing to become a **solver-building
> library** is a level of **multi-block assembly**: several
> states, several models, several numerical methods, several time steps, and
> global couplings in Poisson and in the sources.
>
> The `PhysicalModel` describes a local equation. The `CoupledSystem` describes a physical
> system. The `Scheduler` describes the execution order. The `pops` core guarantees that these
> choices stay compatible with AMR / MPI / GPU.
