# DSL_API -- Short reference of the Python DSL (adc.dsl)

USER reference document. For design, reasoning and history,
see [docs/DSL_MODEL_DESIGN.md](DSL_MODEL_DESIGN.md).

---

## 1. Writing a symbolic model

```python
import adc
from adc import dsl

m = dsl.Model("mon_modele")

# Variables conservatives : conservative_vars(...) RENVOIE un tuple de Var a depacker
# (roles physiques optionnels).
rho, mx, my = m.conservative_vars("rho", "mx", "my",
                                  roles=["Density", "MomentumX", "MomentumY"])

# Variables primitives (kwargs : nom=expression symbolique)
m.primitive_vars(rho=rho, ux=mx/rho, uy=my/rho)

# Flux physique (declarateur symbolique ; m.eval_flux(...) = evaluateur numpy)
m.flux(x=[mx, mx*mx/rho, mx*my/rho],
       y=[my, mx*my/rho, my*my/rho])

# Source (optionnel -- force du potentiel)
# m.aux('grad_x') declare et renvoie le champ auxiliaire grad_x (Var).
phi_x, phi_y = m.aux("grad_x"), m.aux("grad_y")
m.source([-rho*phi_x, -rho*phi_y])  # exemple ExB / force

# Second membre elliptique (optionnel -- couplage Poisson)
m.elliptic_rhs(rho)

# Parametre nomme (constante inlinee a la compilation)
g = m.param("gamma", 1.4)
```

---

## 2. Compile

```python
# Le DEFAUT de m.compile(...) est backend="aot" : il faut donc demander explicitement
# "production" pour le chemin natif zero-copie (recommande en MPI/AMR).
compiled = m.compile(backend="production", target="system")
# Pour AMR :
compiled_amr = m.compile(backend="production", target="amr_system")
```

Available backends (`backend=` ; DEFAULT = `aot`) :

| backend | CPU | MPI | AMR | GPU | Note |
|---|---|---|---|---|---|
| `production` | yes | yes (np=1/2/4) | via `AmrSystem` | GH200 (C++ side) | **recommended** in MPI/AMR ; native zero-copy. `_BACKEND_CAPS["production"]["gpu"]` is reported `False` on the Python side (the tested host module is not built with Kokkos/CUDA) |
| `aot` | yes | no | no | no | **DEFAULT** ; marshaling `.so` ; CPU debug/bench. Also carries runtime params (`set_block_params`) |
| `prototype` | yes (Rusanov o1) | no | no | no | JIT proto ; do not use in production |

The `.so` is cached by `model_hash` : an unchanged model is not recompiled.

---

## 3. Wiring onto System / AmrSystem

```python
sim = adc.System(n=256, periodic=True)
sim.add_equation("fluide",
                 model=compiled,
                 spatial=adc.FiniteVolume(limiter="vanleer", riemann="rusanov"),
                 time=adc.Explicit(substeps=1))
# 1er argument positionnel = rhs (valide dans {charge_density, composite}) : passer le solveur
# par MOT-CLE, pas en positionnel.
sim.set_poisson(solver="geometric_mg")
sim.run(t_end=10.0, cfl=0.4)
```

```python
# AMR : AmrSystemConfig n'a PAS de champ max_level. Champs reels : n, L, regrid_every, periodic,
# distribute_coarse, coarse_max_grid (regrid_every=0 -> hierarchie figee).
amr = adc.AmrSystem(n=128, L=1.0, regrid_every=4, periodic=True)
amr.add_equation("fluide",
                 model=compiled_amr,
                 spatial=adc.FiniteVolume(limiter="vanleer", riemann="rusanov"),
                 time=adc.Explicit(substeps=1))
```

Important points :
- `riemann=` names the NUMERICAL flux (`rusanov`/`hllc`/`roe`) ; `m.flux(...)` is the PHYSICAL flux.
- `fft` is not supported under `System` in MPI `np>1` : use `geometric_mg`.
- `backend="production"` with `target="amr_system"` : `AmrSystem` is single- AND multi-block,
  explicit ; HLLC/Roe/`primitive` are wired on the Python AMR facade with a pressure guard: HLLC/Roe require a
  declared primitive `p` (or the `enable_hllc()` / `enable_roe()` capability), otherwise `add_equation` raises.

---

## 4. Cache and reproducibility

`m.compile()` returns a `CompiledModel` object that carries :
- `so_path` : path of the compiled `.so`.
- `model_hash` : stable hash (formulas + roles + params) -- cache key.
- `abi_key` : compiler/std/headers key -- explicit refusal if incompatible at load time.
- `params` : dict of named parameters declared via `m.param(...)`.

---

## 5. Points of attention

- `m.param(name, value)` : by default (`kind="const"`) constant INLINED at compile time ; changing
  the value requires a new call to `m.compile()`. The `runtime` mode (`kind="runtime"`) is SUPPORTED
  on the `aot` backend : the value is modifiable WITHOUT recompiling via `System.set_block_params`.

  ```python
  m = dsl.Model("iso")
  rho, mx, my = m.conservative_vars("rho", "rho_u", "rho_v")
  cs2 = m.param("cs2", 1.0, kind="runtime")          # param RUNTIME (defaut = 1.0)
  u, v = m.primitive("u", mx/rho), m.primitive("v", my/rho)
  p = m.primitive("p", cs2 * rho)
  m.primitive_vars(rho=rho, u=u, v=v, p=p)
  m.conservative_from([rho, rho*u, rho*v])
  m.flux(x=[mx, mx*u+p, my*u], y=[my, mx*v, my*v+p])
  cs = dsl.sqrt(cs2)
  m.eigenvalues(x=[u-cs, u, u+cs], y=[v-cs, v, v+cs])

  compiled = m.compile(backend="aot")                 # cache key inclut les params
  compiled.runtime_param_names                         # -> ['cs2'] (ordre des indices C++)

  sim = adc.System(n=64, periodic=True)
  sim.add_equation("gas", model=compiled,
                   spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"))
  sim.set_block_params("gas", [4.0])                   # change cs2 au RUNTIME, sans recompiler
  ```
- `adc.PythonFlux` : numpy host TEST tool, outside the GPU/MPI hot path. Never use in
  production.
- Physical roles (`Density`, `MomentumX`, `MomentumY`, ...) : required for inter-species
  couplings and for the `System` to recover quantities by role. To be supplied to
  `conservative_vars(roles=...)` or to `m.compile(require_metadata=True)`.

---

## 6. Reference demonstrators (adc_cases, ci=true)

| Case | File |
|---|---|
| Single-species ExB DSL | `diocotron_dsl/run.py` |
| Two species DSL | `two_species_dsl/run.py` |
| Magnetic isothermal DSL | `magnetic_isothermal_dsl/run.py` |
