# Plan: Variables level + HPM / EPM (abstraction work item)

Working reference for the work item arising from the tutor meetings (01-02/06/2026). To re-read
before the whiteboard session with Sacha. The `adc_cpp` core is already agnostic (bricks) and its
numerical flow is already generic over the model; the missing abstraction is the
**Variables** level (conservative U / primitive P + conversions + reconstruction choice), and the
second correction is to **not hard-code Poisson** but to make it an instance of an
**EllipticPhysicalModel (EPM)**.

---

## 0. Current status

### Done (Phase 1, tested: ctest 46/46 + end-to-end recon Python)
- `PhysicalModel` concept extended by the OPTIONAL `HasPrimitiveVars` extension
  (`core/physical_model.hpp`): `using Prim`, `to_primitive(U)->P`, `to_conservative(P)->U`.
- Conversions implemented in the transport bricks (`model/euler.hpp`, `model/bricks.hpp`):
  CompressibleFlux P=(rho,u,v,p), IsothermalFlux P=(rho,u,v), ExBVelocity P=cons (identity).
  `CompositeModel` forwards Prim + conversions.
- `max_wave_speed` / `wave_speeds` computed VIA the primitive (centralization: no more recomputing
  u=rho u/rho, p=(g-1)(E-...) scattered around).
- Reconstruction in primitive variables (`operator/spatial_operator.hpp`): `reconstruct`
  converts the stencil U->P, limits on P, reconverts P->U; numerical flux unchanged; update
  always conservative. Choice carried by a RUNTIME flag `recon_prim` (no template explosion),
  conservative fallback if the model does not expose the conversions.
- Exposed: `add_block(..., recon="conservative"|"primitive", ...)`; Python
  `pops.Spatial(recon="primitive")` or `Spatial(primitive=True)`. AMR rejects the primitive
  cleanly (the AMR cases use NoSlope, or prim == cons).
- Tests: `tests/test_primitive_recon.cpp` (round-trip + concept) + Python test (Euler recon
  cons vs prim: mass conserved ~1e-15 in both, positivity, finite).

### Done (priorities 5-7 + cases, this work item)
- #57 case "two independent Euler" (adc_cases/two_euler): same HLLC scheme + primitive recon,
  multirate; mass/block, positivity, electrons faster.
- #54 FROZEN species: `evolve` flag on the block (not advanced, seen by Poisson) + `add_background`.
- #52 CoupledSource mechanism: operator-split pass in the System runtime (after transport, reads
  several blocks at the same point).
- #53 coupling bricks: `add_ionization` (n_g -> n_i + n_e, mass n_i+n_g conserved),
  `add_collision` (friction, momentum conserved), `add_thermal_exchange` (energy conserved).
- #55 (partial) plasma case (adc_cases/plasma): e + i + n, Poisson + ionization + collision wired
  end to end; n_i+n_g conserved at ~1e-15, Poisson active, densities positive. models.py:
  `euler()` and `neutral_isothermal()` recipes added.
- #58 first-order EPM: add_elliptic_model + bricks (elliptic / div_eps_grad / charge_density /
  electric_field_from_potential); Poisson = instance; set_poisson shortcut; eps!=1 and alternative
  operators rejected (refinements). Pure Python on top of the existing solver.
- #55/#56/#59: system recipes models.two_fluid / models.plasma; coupling objects
  (pops.Ionization / Collision / ThermalExchange) + sim.add_coupling; variable descriptor
  (sim.variable_names cons/prim per block, introspection).
- Refinements (done): `HyperbolicModel` concept (Vars + conversions + flux + wave speeds,
  Variables REQUIRED); `Euler` split into a PURE hyperbolic brick (without source or elliptic_rhs);
  `CompositeModel<Hyperbolic, Source, Elliptic>` (ex-Transport) with static_assert(HyperbolicModel);
  `Variables` descriptor (core/variables.hpp) carried by the hyperbolic; pops.PythonFlux (host
  prototyping backend, Rusanov numpy residual, outside Kokkos/GPU).
- Verified: test_bindings (bricks + frozen + 3 couplings + EPM + introspection + safeguards),
  two_euler and plasma cases, core ctest 46/46 unchanged.

### Already generic: DO NOT redo
- `RusanovFlux` / `HLLFlux` / `HLLCFlux` are `template<class Model>` and call `m.flux`,
  `m.max_wave_speed`, `m.wave_speeds`, `m.pressure` (`operator/numerical_flux.hpp`). The scheme
  does NOT know Euler. (It is `euler_cpp` that had this problem, not `adc_cpp`.)
- `System` / `AmrSystem` / dispatch (transport x source x elliptic), separation of two repos
  (adc_cpp core, adc_cases cases), multirate, partial IMEX, multi-species Poisson Sum_s q_s n_s.

---

## 1. Target architecture (tutor whiteboard)

```
PhysicalModel
├── HPM : HyperbolicPhysicalModel        d_t U + div F(U, aux) = S(U_self, U_all, aux)
│     Vars   : U (conservatif), P (primitif) ; cons_to_prim(U), prim_to_cons(P)
│     Flux   : F(U) [F(P) optionnel] ; lambda ; |lambda|max = |u| + c, c = sqrt(gamma p/rho)
│     Source : locale + COUPLEE (peut lire les autres especes)
│
└── EPM : EllipticPhysicalModel          D(eps, phi) = f(U_all, aux)
      unknown  : phi
      operator : D = div(eps grad .)        (Poisson : un cas particulier)
      coeff    : eps(x)
      rhs      : f(U_all) = Sum_s q_s n_s    (densite de charge)
      output   : phi, E = -grad phi          (reinjecte dans les sources des HPM)

SpaceMethod (deja generique)
      Reconstruction (conservative | primitive) + NumericalFlux (Rusanov/HLL/HLLC) + div(F*)

Coupler / Assembler
      tient plusieurs HPM + les EPM ; construit les sources inter-especes ; construit le rhs
      des EPM ; reinjecte phi / E dans les sources des HPM. AMR = assembleur sur grille raffinee.
```

### Reference plasma case (3 HPM coupled by the sources)
- H1 electrons (Euler + Lorentz, 4 var): source = ionization `m_e n_e n_g K_iz`, force
  `-e n_e E`, magnetic `-e n_e (u_e x B)`, collisions `-m_e n_e nu_e u_e`, thermal exchange
  `(3 m_e/M) n_e nu_e (T_e - T_g)`.
- H2 ions (isothermal, 3 var): ionization `M n_e n_g K_iz`, force `e n_i E`, collisions
  `-M n_i nu_i u_i` (unmagnetized ions: no u_i x B).
- H3 neutrals (isothermal, 3 var): `-M_n n_e n_g K_iz`, collisions `-M_n n_i nu_i (u_g - u_i)`.
- The SAME ionization term appears with OPPOSITE SIGNS: we lose a neutral, we gain an
  ion and an electron (conservation). Coupling NOT mediated by Poisson.
- Poisson: `div(eps grad phi) = e(n_e - n_i)/eps0`, then `E = -grad phi` comes back into the sources.

---

## 2. Variables level (DONE): reconstruction pipeline

```
Conservatif :  U_i  -> limiteur sur U -> U_L, U_R -> flux numerique -> update conservatif
Primitif    :  U_i  -> P_i = cons_to_prim(U_i) -> limiteur sur P -> P_L, P_R
                    -> U_L = prim_to_cons(P_L), U_R = prim_to_cons(P_R) -> flux numerique
            update TOUJOURS conservatif : U^{n+1} = U^n - dt div(F*) + dt S
```
For Euler the primitive is more robust (positivity of rho and p).

### Vars: carried by the HYPERBOLIC brick (design choice)

On the whiteboard the professor draws `PhysicalModel = Vars + Flux + Source` (three children). IMPLEMENTATION
DECISION: `Vars` is NOT an independent freely combinable brick (you cannot pair Euler's variables
with the isothermal flux). `Vars`, the cons<->prim conversions and the wave speeds are
CARRIED by the HYPERBOLIC brick (`HyperbolicModel` concept), because they are physically tied to the
flux. The composition is on (hyperbolic, source, elliptic):
```
PhysicalModel = CompositeModel<Hyperbolic, Source, Elliptic>
   Hyperbolic --> Vars (cons U / prim P + conversions + descripteur Variables) + Flux + |lambda|max
   Source     --> S(U, aux)            (composable independamment)
   Elliptic   --> elliptic_rhs / EPM   (composable independamment)
```
The hyperbolic brick (Euler, IsothermalFlux, ExBVelocity) therefore exposes a `Variables`
DESCRIPTOR object, a REQUIRED contract (`conservative_vars()` / `primitive_vars()`):
```cpp
enum class VariableKind { Conservative, Primitive };
struct Variables {
  VariableKind kind;
  std::vector<std::string> names;  // ex. {"rho","rho_u","rho_v","rho_E"} ou {"rho","u","v","p"}
  int size;
};
```
and the model carries several sets:
```cpp
Variables conservative_vars;   // (rho, rho u, rho v, rho E)
Variables primitive_vars;      // (rho, u, v, p)
State to_primitive(const State& U) const;
State to_conservative(const State& P) const;
```
Examples (names):
```
Euler      cons (rho, rho u, rho v, rho E)   prim (rho, u, v, p)
isotherme  cons (rho, rho u, rho v)          prim (rho, u, v)
diocotron  cons (n)                          prim (n)
```
The numerical core manipulates these Variables objects GENERICALLY, without knowing whether the
components are rho/u/p, n, rho*u... The reconstruction is written conceptually:
```cpp
if (recon == Conservative) reconstruct(model.conservative_vars, U);
if (recon == Primitive)  { P = model.to_primitive(U); reconstruct(model.primitive_vars, P);
                           U = model.to_conservative(P); }
```
Status (#59, DONE): a `Variables` descriptor object (kind + names + size) is carried by the
HYPERBOLIC brick (`core/variables.hpp`; `conservative_vars()` / `primitive_vars()` on Euler / isothermal /
ExB, forwarded by `CompositeModel`), and it is a REQUIRED CONTRACT of the `HyperbolicModel` concept (not
a separate brick: Vars is physically tied to the flux). The runtime draws from it the names exposed by
`sim.variable_names` (single source of truth). Host metadata, does not drive the computation (the core
works per component via the cons<->prim conversions).

---

## 3. EPM: the correction (DO NOT hard-code Poisson)

Poisson must not be a special case at the architecture level. It needs an
`EllipticPhysicalModel`; Poisson is only one instance of it (unknown phi, operator
div(eps grad), rhs charge_density, output E = -grad phi).

### Target API (instead of `set_poisson(...)` as the final abstraction)
```python
sim.add_elliptic_model(
    name="phi",
    model=models.elliptic(
        unknown="phi",
        operator=models.div_eps_grad(epsilon=1.0),
        rhs=models.charge_density(species={"ne": -1.0}, background=n_i0),
        output=models.electric_field_from_potential(),   # E = -grad phi
    ),
    solver=pops.EllipticSolver("geometric_mg"),
)
```
`div_eps_grad`, `charge_density`, `electric_field_from_potential` are BRICKS, not
`Poisson` hard-coded. You can replace ONLY the EPM without touching System/AmrSystem:
```python
models.elliptic(unknown="T", operator=models.diffusion(coeff="kappa"), rhs=models.heat_source())
models.elliptic(unknown="p", operator=models.pressure_projection(), rhs=models.divergence_constraint())
```
`set_poisson(...)` STAYS as a practical shortcut (= add_elliptic_model with the Poisson bricks),
but not as the central architecture.

Status (first order, DONE, #58): add_elliptic_model + bricks exposed in Python; Poisson = instance;
set_poisson shortcut. For now the operator div(eps grad) at eps=1 + charge density (charges on
the blocks); eps(x), charges at the EPM level, and other operators (diffusion, projection) = refinements
(rejected cleanly by NotImplementedError).

### Math form
```
HPM : d_t U + div F(U) = S(U, aux)
EPM : D(phi, aux) = f(U_all, aux)
Poisson : D = div(eps grad) ; f = e(n_e - n_i)/eps0 ; plus generalement f = Sum_s q_s n_s ; E = -grad phi
```

---

## 4. Inter-species sources (CoupledSource)

Today `source(U, aux)` only sees one state + the field; direct coupling (ionization,
collisions, thermal exchange) must read the OTHER species. Target signature:
```
source(block_id, local_state U_self, SystemView all, aux) -> S
```
Location: at the Coupler/Assembler level (as the elliptic rhs already sums Sum_s q_s n_s). Bricks:
ionization `m n_e n_g K_iz` (opposite signs), collisions `nu (u_a - u_b)`, exchange `(T_a - T_b)`.
Depends on the primitive variables (velocities, temperatures). Test: sum of mass sources = 0.

---

## 5. Frozen species (evolved / non-evolved background)

A block can be FROZEN: `evolve=False`, not advanced in time, but visible to the coupled sources
(ionization needs n_g) and to the EPM rhs. Generalizes `BackgroundDensity` (constant n0)
toward a real background species n_g(x). Switching evolved <-> frozen must be "one line".

---

## 6. Tutor test scale (same SpaceMethod everywhere, without copy-paste)

1. scalar advection
2. isothermal Euler
3. full Euler (1 block)
4. TWO INDEPENDENT Euler (electrons + ions, mass ratio, different velocities, expansion, uncoupled)
5. + coupled source (ionization / collisions)
6. + EPM / Poisson (≈ current multispecies)
7. Sacha's plasma model

---

## 7. TODO (tutor's priority order)

```
[x] 1. Cons/Prim + conversions dans les modeles            (#47, #48)
[x] 2. max_wave_speed via le primitif                       (#48)
[x] 3. reconstruction en primitif + flag recon expose       (#49, #50)
[x] 4. test Euler recon=cons vs recon=prim                  (#51 + test Python)
[x] 5. cas "deux Euler independants" (adc_cases)            (#57, etape 4 de l'echelle)
[x] 6. CoupledSource (operator-split) + ionisation/collision/echange thermique + espece gelee
       (#52, #53, #54) : sim.add_ionization / add_collision / add_thermal_exchange / add_background
[x] 7. EPM premier ordre (#58) : add_elliptic_model + briques elliptic / div_eps_grad /
       charge_density / electric_field_from_potential ; Poisson = instance ; set_poisson raccourci.
       eps(x) et operateurs alternatifs (diffusion, projection) = raffinements (rejetes proprement).
[x] -- niveau systeme dans adc_cases : models.py recipes (two_fluid, plasma) + couplages (#55)
[x] -- Python : Spatial(recon=), objets de couplage (add_coupling), add_background, add_elliptic_model (#56)
[x] -- descripteur Variables : sim.variable_names (cons/prim par bloc), introspection (#59)
```

Note: priority order = the tutor's. #59 (Variables descriptor object) is a conceptual
ENRICHMENT, not blocking; the computation already runs with the functional subset
of Phase 1.

Critical path to "two Euler, same code" requested by the tutor: 1-4 (done) + 5.

---

## 8. Code sketches (from the whiteboard)

### HPM
```cpp
struct HPM {
  using ConsVars = ...;   // U
  using PrimVars = ...;   // P
  PrimVars cons_to_prim(const ConsVars& U) const;   // = to_primitive
  ConsVars prim_to_cons(const PrimVars& P) const;   // = to_conservative
  ConsVars flux(const ConsVars& U, const Aux& aux, int dir) const;
  double   max_wave_speed(const ConsVars& U, const Aux& aux, int dir) const;  // |u| + c
  ConsVars source(const ConsVars& U_self, const SystemView& all, const Aux& aux) const;
};
```

### EPM
```cpp
struct EPM {
  using Unknown = ...;   // phi
  auto   coefficient(Position x) const;            // eps(x), operateur D = div(eps grad)
  double rhs(const SystemView& all, Position x) const;   // f(U_all) = Sum_s q_s n_s
  AuxFields postprocess(const Unknown& phi) const;       // E = -grad phi
};
```

### Generic numerical flux (already in place in adc_cpp)
```cpp
template <class Model>
struct RusanovFlux {
  typename Model::State operator()(const Model& m, const State& UL, const State& UR,
                                   const Aux& aux, int dir) const {
    auto FL = m.flux(UL, aux, dir), FR = m.flux(UR, aux, dir);
    double a = std::max(m.max_wave_speed(UL, aux, dir), m.max_wave_speed(UR, aux, dir));
    return 0.5 * (FL + FR) - 0.5 * a * (UR - UL);
  }
};
```

### Finite volumes + numerical flux at the interface (whiteboard convention)
```
dU_ij/dt + (1/dx)(F*_{i+1/2,j} - F*_{i-1/2,j}) + (1/dy)(F*_{i,j+1/2} - F*_{i,j-1/2}) = S_ij
F*_{i+1/2,j} = NF(U^R_{i,j}, U^L_{i+1,j})        # etats RECONSTRUITS (pas U_i et U_{i+1})
Rusanov directionnel : a = max(|lambda^x_max(U^L)|, |lambda^x_max(U^R)|)   (idem en y)
|lambda|_max = |sqrt(dp/drho) +- u| = |u| + c    # generique ; Euler : c = sqrt(gamma p/rho)
valeurs propres (Euler, dir x) : u - c, u, u + c
```
NF is GENERIC: it calls `model.flux` and `model.max_wave_speed`, never Euler hard-coded (already
the case in adc_cpp). `|lambda|_max = |sqrt(dp/drho) +- u|` is the form valid for any EOS,
not only the ideal gas.

### Diocotron case, target with explicit EPM
```python
sim = pops.AmrSystem(n=n, L=L, regrid_every=10, periodic=True)
sim.add_block("ne", model=models.diocotron(B0=1.0, alpha=1.0),
              spatial=pops.Spatial(none=True, flux="rusanov"))
sim.add_elliptic_model("phi", model=models.elliptic(
    unknown="phi", operator=models.div_eps_grad(epsilon=1.0),
    rhs=models.charge_density(species={"ne": -1.0}, background=n_i0),
    output=models.electric_field_from_potential()),
    solver=pops.EllipticSolver("geometric_mg"))
```

---

## 9. Decision: single interface (Vars + Flux + Source), two Flux backends

Design converged with the user. The model INTERFACE stays the same everywhere: `Vars + Flux + Source`
(+ EPM for the elliptic). All the numerics (reconstruction, numerical flux NF, assembler) consume
ONLY this interface. Behind the `Flux` interface, two interchangeable implementations:

- **CompiledFlux** (default, PRODUCTION): compiled C++ bricks `POPS_HD` (CompressibleFlux=Euler,
  IsothermalFlux, ExBVelocity...). Fast, compatible with Kokkos / GPU / MPI. They live in adc_cpp as
  generic operators (on the same footing as Rusanov or the limiters). This is the production path.
- **PythonFlux** (PROTOTYPING): a function supplied from Python / adc_cases (vectorized numpy:
  flux(U)->F, max_wave_speed(U)->a, cons<->prim). Slow, CPU / HOST only. Allows prototyping
  a novel flux without recompiling. GOLDEN RULE: must NEVER be used in a Kokkos / GPU kernel
  (safeguard: if Kokkos/GPU backend active -> clear error).

On the Python side, `pops.Model(...)` can therefore EITHER select a compiled brick (`pops.CompressibleFlux()`),
OR supply a prototype function (PythonFlux). Same interface, same assembler; only the flux backend
changes. Performance imposes CompiledFlux; PythonFlux is for iterating fast, outside the GPU/MPI hot path.

adc_cpp / adc_cases boundary:
- adc_cpp: generic engine + compiled bricks (flux, sources, elliptic operators, numerics,
  reconstruction, time steppers, Poisson) + the interface contract. No named SCENARIO.
- adc_cases: SCENARIOS = Python compositions (diocotron, two-fluid, plasma...) + possible prototyping
  PythonFlux. The scenario NAMES live here.

The `custom_scheme` pattern (task #40) is the ancestor of PythonFlux; DONE (#62): `pops.PythonFlux`
(flux + max_wave_speed in numpy, finite-volume Rusanov residual on the HOST side) formalizes this
prototyping backend, OUTSIDE Kokkos/GPU (pure host path, never in a kernel). The named fluxes
(Euler/isothermal/ExB) are NOT moved out of adc_cpp: they are the CompiledFlux (production).
