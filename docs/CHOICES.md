# Design choices

The architecture decisions of `adc_cpp`, with their context and their cost. Complements
[ARCHITECTURE.md](ARCHITECTURE.md) (which describes the state) by saying *why*.

---

## D-1. AMR stack written *from scratch*, not on top of `pde_core_cpp`

**Context.** `euler_cpp` and `advection_cpp` depend on `pde_core_cpp` (AMR primitives
shared via FetchContent). One option was to do the same for `adc_cpp`.

**Decision.** MultiFab + BoxArray + DistributionMapping + seam stack written entirely in
`include/adc/`, independent.

**Why.** The hyperbolic-elliptic coupling on distributed AMR (diocotron, two-fluid
plasma) requires a mesh layer close to AMReX (distributed MultiFab, FluxRegister, FillPatch)
that `pde_core_cpp` did not have. Building it from scratch freezes an AMR stack mastered
end to end, the pedagogical goal of the internship.

**Cost.** Conceptual duplication with AMReX; no mature optimizations (MFIter, curved EB).
Accepted: we stay readable and portable rather than optimal.

---

## D-2. Three orthogonal axes + concepts

**Decision.** `PhysicalModel` (concept), `NumericalFlux` (policy), `EllipticSolver`
(concept), coupling. A solver = one point in this product.

**Why.** Add a flux (HLLC), an elliptic (FFT), or a model (two-fluid) without
touching the rest. `compute_face_fluxes<Limiter, NumericalFlux, Model>` is the junction
point. Inspired by the design of PLUTO (see BIBLIOGRAPHY).

---

## D-3. The `for_each_cell` seam (single dispatch)

**Decision.** A single loop primitive `for_each_cell(box, lambda ADC_HD)` compiles
to `Kokkos::parallel_for` (Serial / OpenMP / Cuda execution space depending on the Kokkos install).
Device-callable POD `Array4`, `device_fence()`, `comm.hpp` for MPI.

**Why.** The physics is written once and runs everywhere. Kokkos is the only on-node
backend and it is required (`-DADC_USE_KOKKOS=ON`, ON by default); the seam does not compile
without `ADC_HAS_KOKKOS`. The backend remains a **property of the `adc` target**
(target_compile_definitions INTERFACE), not a per-solver flag: the on-node target is
chosen at the Kokkos install (`Kokkos_ENABLE_SERIAL` / `_OPENMP` / `_CUDA`), nothing in
the code.

**Cost.** GPU fence discipline (any kernel-device function then host-loop on the
same unified memory must `device_fence()` between the two). The most subtle bug encountered.

---

## D-4. MultiFab / BoxArray / DistributionMapping in the AMReX style

**Decision.** **Global** BoxArray (all ranks know all boxes),
DistributionMapping (per-box owner, SFC balancing), MultiFab allocating only the
local fabs.

**Why.** This is the model that makes distributed AMR possible and the multi-patch coverage
correct under any distribution (the coverage is computed from the global BoxArray).

---

## D-5. `EllipticSolver` concept: multigrid AND FFT

**Decision.** `GeometricMG` (iterative, warm-start, on-device) and `PoissonFFTSolver`
(direct, periodic) model the same concept; the coupler is generic over it.

**Why.** The right solver depends on the load (measured: FFT wins ~4.8x on Poisson-dominated
Euler-Poisson, MG wins ~2.4x on the transport-dominated two-fluid). The concept lets
you choose without rewriting the coupler.

---

## D-6. Runtime composition facade + bindings

**Decision.** The bindings (`python/bindings/core/bindings.cpp`) expose runtime COMPOSITION facades
(`System`, `AmrSystem`), not named solvers. A model is a composition of
generic bricks (`adc.Model(state, transport, source, elliptic)`) assembled by the
`model_factory`; no scenario is named in the lib.

**Why.** A stable and bindable surface, never `Coupler<Model, Elliptic>` on the outside. The
generic core stays header-only; the runtime facade (`runtime/system.hpp`) gives a clean Python
API and a compilation boundary. The concrete named solvers have disappeared in favor of
agnostic composition: the scenario names live on the application side (`adc_cases`).

---

## D-7. Reference Fab2D stack kept as a test oracle

**Decision.** The old mono-box `Fab2D` stack (`amr_reflux.hpp`, `amr_multilevel.hpp`)
is no longer in production but stays compiled and tested.

**Why.** It serves as an **oracle**: every MultiFab brick is proven bit-identical to
it (`test_amr_*_mf`). Safe rework by equivalence rather than by trust.
