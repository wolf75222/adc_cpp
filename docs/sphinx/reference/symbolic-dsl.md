# Reference: the symbolic DSL (pops.dsl)

The `pops.dsl` DSL lets you write a model's physics as a tree of symbolic
expressions (the Python operators `+ - * / ** -` and `dsl.sqrt` build the tree, not a
function called per cell), which the DSL translates into compilable C++ then compiles into a `.so`
attachable to an `pops.System` / `pops.AmrSystem`. Two entry points: `pops.dsl.Model(name)`,
the recommended stable facade (pure sugar, composition of a private `HyperbolicModel` `_m`), and
`pops.dsl.HyperbolicModel(name)`, the lower-level backend object (`set_*` naming, always
usable directly). Both expose `compile()`. This page is the canonical registry of the
DSL. Source: [python/pops/dsl.py](https://github.com/wolf75222/adc_cpp/blob/master/python/pops/dsl.py) ;
design: [DSL_MODEL_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/DSL_MODEL_DESIGN.md).

## Declaring a model

All the methods below are on the `pops.dsl.Model` facade. They delegate to
`HyperbolicModel` (implicit right column). `flux`, `eigenvalues`, `source`, `elliptic_rhs`
map one to one to the functions of the `pops::PhysicalModel` concept read by the core.

| method | what it declares | example |
|---|---|---|
| `conservative_vars(*names, roles=None)` | the conservative variables (the state `U`) ; returns a tuple of `Var` | `rho, mx, my, E = m.conservative_vars("rho","rho_u","rho_v","E")` |
| `primitive(name, expr)` | a primitive by its formula ; returns a `Var` | `u = m.primitive("u", mx/rho)` |
| `primitive_vars(*vars, roles=None, **named)` | the ordered layout of `Prim` (and, in kwargs, defines each primitive) | `prho,pu,pv,pp = m.primitive_vars(rho=rho,u=u,v=v,p=p)` |
| `aux(name)` | a fixed canonical auxiliary field read at runtime ; returns a `Var` | `gx = m.aux("grad_x")` |
| `aux_field(name)` | a NAMED auxiliary field, set per block (see [aux vs aux_field](#aux-vs-aux-field)) ; returns a `Var` | `mu = m.aux_field("mobility")` |
| `flux(x, y)` | the physical flux `F(U)`, one `Expr` per component and direction | `m.flux(x=[...], y=[...])` |
| `eigenvalues(x, y)` | the characteristic speeds per direction | `m.eigenvalues(x=[u-c,u,u+c], y=[v-c,v,v+c])` |
| `wave_speeds(x, y)` | explicit signed speeds `(smin, smax)` per direction (enables `hll` without a primitive `p`) | `m.wave_speeds(x=(sl, sr), y=(sl, sr))` |
| `wave_speeds_from_jacobian(...)` | signed speeds from the eigenvalues of the flux Jacobian (autodiff) | `m.wave_speeds_from_jacobian()` |
| `enable_hllc()` | emit the HLLC capability from the roles + primitive `p` (enables `hllc` beyond Euler-4) | `m.enable_hllc()` |
| `enable_roe()` | emit the Roe capability from the roles + primitive `p` (enables `roe` beyond Euler-4) | `m.enable_roe()` |
| `roe_from_jacobian()` | generic moment Roe from the flux Jacobian, roles-free (see below) | `m.roe_from_jacobian()` |
| `conservative_from(exprs)` | the inverse `Prim -> U` (to be supplied, the DSL does not invert) | `m.conservative_from([rho, rho*u, rho*v, ...])` |
| `source(s)` | the source term `S(U, aux)` (optional), one `Expr` per component | `m.source([0.0, -rho*gx, -rho*gy, ...])` |
| `source_frequency(mu)` | local source frequency `mu(U, aux)` driving the `source` step bound (requires `source`) | `m.source_frequency(nu_coll)` |
| `source_jacobian(rows)` | analytic `dS/dU` for the implicit Newton (instead of finite differences ; requires `source`) | `m.source_jacobian([[...], ...])` |
| `implicit_source(jacobian=None)` | sugar for the local implicit (residual from `source` ; optional analytic Jacobian) | `m.implicit_source()` |
| `stability_speed(expr)` | custom stability speed `lambda*` driving the block CFL (default: `max(abs(eigenvalues))`) | `m.stability_speed(c_fast)` |
| `stability_dt(expr_dt)` | direct admissible step bound `dt(U, aux)` (the CFL is not applied to it) | `m.stability_dt(dt_relax)` |
| `elliptic_rhs(e)` | the contribution to the right-hand side of the system Poisson (optional) | `m.elliptic_rhs(-1.0*(rho - 1.0))` |
| `projection(p)` | pointwise post-step projection `U <- P(U, aux)` (optional, realizability/positivity) | `m.projection([rho, (rhou+abs_(rhou))/2, ...])` |
| `gamma(value)` | the adiabatic index (EOS), exported in the `.so` | `m.gamma(1.4)` |
| `param(name, value, kind="const")` | a named parameter usable in formulas ; returns a `Param` | `g = m.param("gamma", 1.4)` |
| `check()` | verifies that every referenced variable is declared | `m.check()` |

### conservative_vars

Declares the conservative state vector and returns one `Var` per name. The `roles=` is optional and of
the same length as `names` ; without it, the canonical name -> role mapping applies (`rho`/`n` ->
`Density`, `rho_u` -> `MomentumX`, `E` -> `Energy`...), an unknown name stays `Custom`. A length
of `roles` that differs raises a `ValueError`. The roles let the `System` resolve inter-species
couplings by `index_of(role)` rather than by a literal index.

```python
rho, rhou, rhov, E = m.conservative_vars(
    "rho", "rho_u", "rho_v", "E",
    roles=["Density", "MomentumX", "MomentumY", "Energy"])
```

### primitive

Defines a primitive by its formula (as a function of the conservatives, the preceding primitives or
the aux). Insertion order = dependency order.

```python
u = m.primitive("u", rhou / rho)
p = m.primitive("p", (g - 1.0) * (E - 0.5 * rho * (u*u + v*v)))
```

### primitive_vars

Sets the ordered layout of `Prim` ; two exclusive forms (mixing them raises a `ValueError`) :

- kwargs form (target style) : `primitive_vars(rho=expr, u=expr, v=expr, p=expr)` defines each
  primitive and sets the layout in the insertion order of the kwargs (Python 3.7+). Returns a tuple of
  `Var`.
- positional form : `primitive_vars(rho, u, v, p, roles=...)` only sets the layout from
  names / `Var` already defined. Returns `None`.

Self-reference safeguard : in kwargs, if the value is the `Var` of the same name (e.g. `u=u` with `u`
already coming from `m.primitive("u", ...)`) or if the name is already a conservative (`rho=rho`), the
primitive is not redefined ; it just joins the layout. Without this safeguard, the codegen
would emit `const Real u = u;` (auto-init -> NaN).

```python
prho, pu, pv, pp = m.primitive_vars(rho=rho, u=u, v=v, p=p)   # defines AND orders
```

### aux

Declares an auxiliary field read at runtime (`pops::Aux` channel) ; returns a `Var` read in C++ as
`a.<name>`. The name must be a key of the fixed table (an unknown name raises a `ValueError`) :

| name | index | role |
|---|---|---|
| `phi` | 0 | potential |
| `grad_x` | 1 | x gradient of the potential |
| `grad_y` | 2 | y gradient of the potential |
| `B_z` | 3 | magnetic field (extended channel) |
| `T_e` | 4 | electron temperature (extended channel) |

`phi`/`grad_x`/`grad_y` are the base contract (3 components). Using `B_z` or `T_e` widens the
channel : the generated brick then declares `n_aux` (4 or 5) so the system dimensions the shared
channel.

```python
gx, gy = m.aux("grad_x"), m.aux("grad_y")   # field E = -grad phi
```

(aux-vs-aux-field)=
### aux vs aux_field

`aux(name)` and `aux_field(name)` both return a `Var` read at runtime, but they target different
parts of the `pops::Aux` channel:

- `aux(name)` reads a CANONICAL component of the fixed table above (`phi` / `grad_x` / `grad_y` /
  `B_z` / `T_e`, indices 0..4), DERIVED by the solver (the potential and its gradient come from the
  Poisson solve). An unknown name raises.
- `aux_field(name)` reserves a NAMED, user-supplied component. The name is arbitrary; the k-th
  declared field is component `AUX_NAMED_BASE + k` (base 5, just after `T_e`), read in C++ via
  `aux.extra_field(k)`. At most `AUX_NAMED_MAX` (= 4) named fields per model, so the named block
  starts strictly after the canonical table and never collides with it.

A named field is static (it persists across steps; the solver never rewrites it) and is provided per
block via `System.set_aux_field(block, name, array, halo=None)` -- the `name` must have been declared
by `m.aux_field(name)` and the block attached via `add_equation`; the array is 2D `(ny, nx)` or flat
row-major. Read it back with `System.aux_field(block, name)`. The canonical `B_z` / `T_e` keep their
dedicated paths (`set_magnetic_field` / `set_electron_temperature_from`) and are NOT settable through
`set_aux_field`.

By default a named field inherits the SHARED aux ghost behavior (periodic preserved, otherwise
zero-gradient). Pass `halo=pops.AuxHalo(...)` (ADC-369) to give that one field its own non-periodic
boundary policy instead: `pops.AuxHalo("foextrap")` (zero-gradient) or `pops.AuxHalo("dirichlet",
value=v)` (fixed boundary value). The policy applies only to the non-periodic faces (periodic faces,
and the polar `theta` direction, keep their wrap), works on `System` (Cartesian + polar) and the AMR
coarse level, and is bit-identical to the shared behavior when omitted. The same `halo=` is accepted
by `AmrSystem.set_aux_field`.

```python
mu = m.aux_field("mobility")               # named field, component 5
m.source([0.0, -mu * mx, -mu * my, 0.0])
# ... after add_equation("ions", compiled):
s.set_aux_field("ions", "mobility", mob_2d)                       # static, per block
s.set_aux_field("ions", "mobility", mob_2d, halo=pops.AuxHalo("foextrap"))  # per-field zero-gradient BC
```

### flux

Declares the physical flux `F(U)`. `x` and `y` are lists of `Expr`, one per conservative
component. Not to be confused with the numpy evaluator `m.eval_flux(U, aux, dir)` (debug / host
proto, which returns a stacked numpy array) nor with the numerical flux `riemann=` of
`pops.FiniteVolume` (Rusanov / HLLC / Roe).

```python
m.flux(x=[rhou, rhou*u + p, rhou*v, rho*H*u],
       y=[rhov, rhov*u, rhov*v + p, rho*H*v])
```

### eigenvalues

Declares the characteristic speeds per direction (lists of `Expr`). The core derives
`max_wave_speed` from them (Rusanov bound and CFL time step) ; if a primitive named `p` (pressure)
exists, the generated brick also exposes `pressure` / `wave_speeds`, which makes it compatible with
the HLLC / Roe fluxes. Without `p`, the model stays limited to Rusanov.

```python
m.eigenvalues(x=[u-c, u, u+c], y=[v-c, v, v+c])
```

### Riemann capabilities (wave_speeds, enable_hllc/enable_roe, roe_from_jacobian)

By default a model carries only the Rusanov bound from `eigenvalues`. Several declarators widen the
set of `riemann=` fluxes a model accepts at `add_equation`:

- `wave_speeds(x, y)` declares explicit signed speeds `(smin, smax)` per direction, which enables
  `riemann="hll"` for a model WITHOUT a primitive `p` (a moment or isothermal system).
  `wave_speeds_from_jacobian(...)` derives the same pair from the eigenvalues of the flux Jacobian.
- `enable_hllc()` / `enable_roe()` emit the HLLC / Roe capability from the variable roles plus a
  primitive `p`, so `riemann="hllc"` / `"roe"` work beyond the four-variable Euler system.
- `m.roe_from_jacobian()` is the GENERIC moment Roe emitter: it builds the `roe_dissipation` hook as
  `|A| (UR - UL)` with `A = dF/dU` the autodiff flux Jacobian at the mean interface state, applied via
  the matrix-sign kernel `pops::roe_abs_apply` (spectral-radius Rusanov fallback on a complex or
  singular spectrum). Unlike `enable_roe`, it needs NEITHER fluid roles NOR a primitive `p`, so it
  makes `riemann="roe"` available for a full moment hierarchy. It is one of THREE mutually exclusive
  providers of `roe_dissipation` (with `enable_roe` and the hand-written `roe_dissipation(x, y)`):
  declaring more than one raises. Folded into the model cache key only when used, so an unused model
  stays bit-identical. This is the path the moment generator emits -- see
  [moment models](moment-models.md).

### conservative_from

Gives the inverse `Prim -> U`, one `Expr` per conservative in the order of `conservative_vars`. The
DSL cannot invert the primitives symbolically ; the user supplies the inverse by hand.
Generates `to_conservative`. Required for the complete codegen of the brick (`emit_cpp_brick`).

```python
m.conservative_from([rho, rho*u, rho*v, p/(g-1.0) + 0.5*rho*(u*u + v*v)])
```

### source

Optional source term `S(U, aux)`, one `Expr` per component (scalars are promoted to
`Const`). Reads the external state through the `pops::Aux` channel (e.g. `grad_x` / `grad_y` for a
potential force). Without `source`, the brick is `pops::NoSource`.

```python
m.source([0.0, -rho*gx, -rho*gy, -(rhou*gx + rhov*gy)])
```

### Named sources (source_term)

`m.source_term(name, exprs)` declares a NAMED local source `S_name(U, primitives, aux, params)` --
exactly `n_cons` expressions, free to depend on conservative / primitive / aux / params / constants.
A named source is **opt-in**: it is emitted only when a compiled time {doc}`Program <time-program>`
requests it (`P.rhs(..., sources=[name])` / `ctx.source(name)`), and is **never** summed implicitly
into the legacy total source. `m.source([...])` is exactly `source_term("default")` (the
backward-compatible alias). Other names must be valid identifiers, unique, and must not collide with a
`linear_source`.

```python
m.source_term("electric", [0.0, -rho*gx, -rho*gy])   # opt-in; requested via rhs(sources=["electric"])
```

Because named terms are never summed implicitly, a model that declares only named sources (no
`m.source` default) cannot answer the legacy total-source query: an old stepper asking for the total
source is rejected fail-loud (`ValueError: model has multiple named sources; use
pops.compile_problem(...) or define m.source(...) explicitly`). Use a compiled time `Program` (which
names the sources it wants) for such a model.

### Local linear sources (linear_source)

`m.linear_source(name, matrix)` declares a NAMED local linear operator `L_name(aux, params)` -- an
`n_cons x n_cons` matrix whose coefficients may depend on constants / params / aux **only**, NOT on
conservative or primitive variables (otherwise `S(U) = L U` would not be linear in `U` and could not
be treated as a local linear source). A coefficient that reads a conservative or primitive variable is
rejected fail-loud (`ValueError: linear_source '<name>' coefficients must not depend on conservative
or primitive variables`).

The operator is **opt-in**: never folded into `m.source` or `P.rhs`. A compiled time `Program` uses it
explicitly via `P.linear_source(name)` -- as `ctx.apply(L, U)` (`L U`, the explicit Lorentz force), or
inside `P.solve_local_linear(operator=P.I - dt*P.linear_source(name), rhs=...)` (the per-cell implicit
solve `(I - dt L) U' = rhs`). The canonical example is the 3x3 Lorentz operator (rho, mx, my) with the
magnetic field `B_z` read from the aux channel:

```python
bz = m.aux("B_z")
m.linear_source("lorentz", [[0.0,  0.0, 0.0],
                            [0.0,  0.0,  bz],
                            [0.0,  -bz, 0.0]])   # used via P.apply / P.solve_local_linear in a Program
```

See `examples/time_programs/predictor_corrector_poisson_lorentz.py` for a full predictor-corrector
step driving a named `electric` source and the `lorentz` linear source.

### elliptic_rhs

Contribution to the elliptic right-hand side (system Poisson coupling : charge density, neutralizing
background, gravity), a single `Expr`. The system Poisson sums the contributions of all the
blocks. Without it, the block's rhs is zero.

```python
m.elliptic_rhs(-1.0 * (rho - 1.0))   # self-consistent gravity, sign=-1, rho0=1
```

### projection

`projection([...])` declares a POINTWISE post-step projection `U <- P(U, aux)`, one `Expr` per
conservative component, for realizability or positivity (e.g. the HyQMOM `relaxation15` moment
projection). It is emitted as the C++ trait `HasPointwiseProjection` and compiled like the flux and
source (CSE included), replacing a per-cell Python callback.

```python
m.projection([rho, (rhou + abs_(rhou)) / 2, ...])   # ex. plancher de positivite branche-libre
```

- **Semantics**: applied once at the END of each whole macro-step (after transport + source stage +
  couplings; never per RK stage, including under Strang), on the valid cells only (ghosts are rebuilt
  by the next step).
- **Contract**: `P` must be idempotent (a true projection) and pointwise (no neighbor). Write the
  clamps BRANCH-FREE in max/min via `dsl.abs_` / `dsl.sign` (differentiable through `dsl.diff`), e.g.
  positivity `(q + abs_(q)) / 2`.
- **Where it runs**: backends `aot` and `production`, on BOTH `pops.System` and `pops.AmrSystem` -- on
  AMR (ADC-312) the projection is applied per level after the reflux and cascade, so the conservative
  correction is preserved. Only the `prototype` (JIT) backend rejects it. A model without a projection
  is bit-identical to the historical trajectory (opt-in via the trait).

### Eigenvalue spectrum predicates

`pops.dsl` exposes scalar `Expr` nodes built from the spectrum of a SMALL dense matrix assembled from
expressions (a Jacobian sub-block, a companion matrix...). The matrix is `rows`, a list of `k` rows
of `k` `Expr` (row-major, `k <= 16`), diagonalized device-clean by `pops::real_eig_minmax`
(`dense_eig.hpp`). They are designed for the branch-free `m.projection`: a test like "if the spectrum
is complex, correct" is written as a max/min/sign mask on these scalars, with no dynamic branch.

| function | scalar value |
|---|---|
| `eig_max_im(rows)` | the largest `abs(Im(lambda))` (0 = a real, hyperbolic spectrum) |
| `eig_lmin(rows)` | the smallest real part of the spectrum |
| `eig_lmax(rows)` | the largest real part of the spectrum |
| `eig_all_real(rows, im_tol=1e-5)` | `1.0` iff the block CONVERGED and its spectrum is real, else `0.0` |
| `EigWitness(rows, field, im_tol=...)` | the node behind the four helpers (`field` selects `max_im` / `lmin` / `lmax` / `all_real`) |

`eig_all_real` (ADC-362) maps to `pops::EigBounds::all_real`: it returns `1.0` only when the small
dense block converged AND its spectrum is real (the largest imaginary part is within `im_tol` times a
relative scale `max(abs(lmin), abs(lmax), 1)`), and `0.0` otherwise. Crucially it is
CONVERGED-GATED, so a Gershgorin fallback (a non-converged block) yields `0.0` (not real). Prefer it
to a raw `eig_max_im(rows) <= tol`, which would read the fallback's conventional `max_im = 0` as a
real spectrum. Compose it branch-free, for example `complex = 1.0 - eig_all_real(rows)` and then a
max/min blend:

```python
complex_mask = 1.0 - eig_all_real(rows)        # 1.0 if a complex pair (or non-converged), else 0.0
m.projection([rho, mx + complex_mask * correction_x, ...])
```

### gamma and param

`gamma(value)` sets the adiabatic index, carried by the `.so` (optional symbol) so the
`System` couplings use the right gamma instead of the historical default 1.4.

`param(name, value, kind="const")` declares a named parameter usable as an `Expr`, stored
in `m.params` (introspection / reproducibility). Two modes :

- `kind="const"` (default) : the value is inlined as a literal at codegen (literal in the `.so`), while
  keeping its identity for introspection.
- `kind="runtime"` : the value emits `params.get(<index>)` (read of an
  `pops::RuntimeParams` member), modifiable at runtime via `System.set_block_params(name, values)` without
  recompiling. Supported by the `aot` backend only ; on `prototype` / `production` a runtime param
  is frozen to its declaration value.

Special case `name == "gamma"` : `param` also calls `set_gamma(value)` so the ABI metadata
stays consistent.

```python
g   = m.param("gamma", 1.4)                   # const : inline + set_gamma
cs2 = m.param("cs2", 1.0, kind="runtime")     # runtime: params.get(0), overwritable (aot)
```

`pops.dsl.RuntimeParam(name, value)` is sugar equivalent to `Param(name, value, kind="runtime")`.

```{note}
On `HyperbolicModel`, these declarators carry the `set_` prefix (`set_flux`, `set_eigenvalues`,
`set_source`, `set_elliptic_rhs`, `set_gamma`, `set_primitive_state`, `set_conservative_from`) ;
`cons(name)` adds a single conservative. The `Model` facade renames them into declarative forms.
`param` exists only on the `Model` facade.
```

## Expression algebra

Every formula is a tree of `Expr`. The leaves are `Var`, `Const`, `Param`, `RuntimeParamRef`,
`_CsField` ; the operators build compound nodes. Supported operators :

| Python | node | C++ emitted |
|---|---|---|
| `a + b` | `Add` | `(a + b)` |
| `a - b` | `Sub` | `(a - b)` |
| `a * b` | `Mul` | `(a * b)` |
| `a / b` | `Div` | `(a / b)` |
| `a ** b` | `Pow` | `std::pow(a, b)` |
| `-a` | `Neg` | `(-a)` |
| `+a` | identity | returns the inner node |

Python scalars (`int` / `float`) are auto-promoted to `Const(float(o))` (via `_wrap`). A
`Param` is promoted by its inner node (`Const` for const, `RuntimeParamRef` for runtime), so
`dsl.sqrt(param_runtime)` correctly emits `params.get(...)` and not the frozen value.

`dsl.sqrt(x)` (-> `std::sqrt(...)`) is the only named math function of the algebra. Everything
else goes through the operators : for a square or a root you write `x*x` or `x**0.5`.

What is not supported, on purpose :

- no `exp`, `log`, `min`, `max`, `abs`, nor conditional / ternary. The `min` / `max` on the
  eigenvalues are generated internally on the C++ side from `eigenvalues`, not exposed as
  operators.
- no `grad` / `div` operator nor symbolic differentiation : the spatial derivatives arrive
  through the `aux("grad_x")` / `aux("grad_y")` fields supplied by the solver, not through the algebra.
- no indexing on an `Expr` : components are addressed by position in the Python
  lists of `flux` / `source` / `conservative_from`.

CSE (common subexpression elimination) : `cse=True` (default of all emitters) factors
out the repeated compound subexpressions into `cseK_` locals, in dependency order (the smallest
first), via a structural key per node (two identical subtrees share a local ; a
runtime param is keyed by its name).

## Compiling

`m.compile(...)` translates the symbolic model into a `.so` and returns a `CompiledModel`. The facade :

```python
Model.compile(so_path=None, include=None, backend="auto", target="system",
              name=None, cxx=None, std=None, require_metadata=False)
```

The backing `HyperbolicModel.compile(...)` has the same signature (order `backend, name, cxx, std,
require_metadata, target`) and returns the `so_path` (string).

Argument semantics :

- `so_path=None` : out-of-source cache (`pops_cache_dir()` : `$POPS_CACHE_DIR`, otherwise
  `$XDG_CACHE_HOME/pops/dsl`, otherwise `~/.cache/pops/dsl`). The file name is keyed on `model_hash`
  + `abi_key` (+ backend / target / name). Cache hit (the `.so` already exists for this key) -> no
  recompilation. Passing `so_path=` forces that path and always recompiles.
- `include=None` : auto-detected by `pops_include()` (`$POPS_INCLUDE`, otherwise the installed `pops`
  package, otherwise the sibling repo). Validity criterion : `pops/mesh/storage/multifab.hpp` exists ; otherwise `RuntimeError`.
- `cxx=None` : autodetect `c++` / `g++` / `clang++` (via `shutil.which`).
- `std=None` : default per backend. For `production` (native), the loader standard via
  `loader_cxx_std()` (= `_pops.__cxx_std__` : c++20 under Kokkos because CUDA 12.x has no `-std=c++23`,
  c++23 otherwise). For `prototype` / `aot`, `"c++20"`.
- `require_metadata=False` : if `True`, requires useful physical roles and an explicit `gamma`,
  failing which the `.so` would fall back on the `System` defaults (`custom` roles / gamma 1.4). This
  safeguard runs before the cache. Incompatible with `prototype` (raises a `ValueError`).
- `name` : optional override of the base name of the generated struct / type.

### The three backends

| backend | engine | System adder | numerics | CPU | MPI | AMR | GPU | when |
|---|---|---|---|---|---|---|---|---|
| `prototype` | JIT (`compile_so`) | `add_dynamic_block` | virtual `IModel`, host residual, Rusanov order 1 only | yes | no | no | no | fast iteration / debug |
| `aot` | AOT (`compile_aot`) | `add_compiled_block` | flat ABI `.so`, production path (HLLC/Roe, order 2, WENO5) but local single-rank grid with marshaling (non zero-copy) | yes | no | no | no | `auto` fallback ; debug / CPU bench ; only one to carry runtime params |
| `production` | native (`compile_native`) | `add_native_block` | `.so` loader that inlines `add_compiled_model<ProdModel>` on the `grid_context()` -> zero-copy, same path as `add_block`, named functors | yes | yes | via `AmrSystem` | reports `False` (non-Kokkos host) | recommended in MPI / AMR |

The code default is `backend="auto"` : it auto-selects `production` under toolchain parity with
the installed `_pops` (loadable module + known baked compiler + matching header signature),
otherwise falls back to `aot`. The explicit values `prototype | aot | production` are still
available and short-circuit this policy. The capabilities are materialized in `_BACKEND_CAPS` :
`production` declares `{cpu, mpi, amr} = True`. `gpu` is reported `False` out of caution : the native
path is device-clean in C++ (validated GH200, named functors), but the end-to-end validation from
Python on a module built Kokkos/CUDA remains a dedicated step and the host module tested in CI is not
built GPU. These capabilities are diagnostic flags, checked at attachment (`add_equation`) or
at runtime, and not a `device=` argument frozen at compile time.

### Hybrid models (native + DSL)

You can mix native bricks and partial DSL bricks in a single model via
`pops.CompositeModel(transport, source, elliptic)`, which returns a `dsl.HybridModel` ; its
`.compile(backend="aot")` returns a `CompiledModel` attachable via `add_equation`. At least one slot must
be a DSL brick (otherwise use `pops.Model(...)`). Brick catalog and example :
[brick reference](native-bricks.md).

### ABI key (production)

The `production` loader calls off-line methods of the already-loaded `_pops` module
(`install_block` / `grid_context` / `ensure_aux_width` ; `set_compiled_block` for AMR), so it
is compiled with `-undefined dynamic_lookup` on macOS and bakes `-DPOPS_HEADER_SIG=<signature>`
identical to the module's build. Loader and module must share the same ABI (headers + compiler
+ C++ standard). `add_native_block` compares `pops_native_abi_key()` to `module.abi_key()` and rejects with
"ABI incompatible" if they diverge. A different `std` changes `__cplusplus` so the ABI key : that is
why `std=None` derives the standard from the loader instead of freezing c++23. The `prototype` / `aot`
backends only target `System` and do not cross-attach.

### target='system' vs 'amr_system'

- `target="system"` (default) : `pops::System` facade. The native loader emits the symbol
  `pops_install_native(System&, ..., evolve, stride)`. All backends are permitted.
- `target="amr_system"` : `pops::AmrSystem` facade. Valid only with `backend="production"` (the
  others raise a `ValueError`, there is no AMR `.so` path outside native). The loader includes
  `amr_dsl_block.hpp` and emits a distinct symbol `pops_install_native_amr(AmrSystem&, ...)` (without
  `evolve` argument) -> `add_compiled_model(AmrSystem&)` (conservative reflux, regrid). You attach via
  `AmrSystem.add_native_block`. A System loader does not attach on AmrSystem and vice versa.

### CompiledModel

The object returned by `Model.compile`. It carries : `so_path`, `backend`, `target`, `adder`,
`cons_names`, `cons_roles`, `prim_names`, `n_vars`, `gamma`, `n_aux`, `params` (dict name -> `Param`),
`caps` (cpu/mpi/amr/gpu), `abi_key`, `model_hash`, `cxx`, `std`. Properties :
`runtime_param_names` (sorted runtime params, = the order of the C++ indices and the order expected by
`set_block_params`) and `runtime_param_values()`. The metadata is not re-read from the `.so` : Python
already holds it. You attach via `System.add_equation(name, compiled, ...)`, which routes according to
`compiled.adder`.

## Inter-species coupled sources (CoupledSource)

`pops.dsl.CoupledSource` describes an arbitrary inter-species exchange in formulas (beyond the named
Ionization / Collision / ThermalExchange couplings). It compiles into flat bytecode (stack machine, no
`.so`, no per-cell Python callback) interpreted in a device `for_each_cell`. Applied in
explicit splitting, after the transport.

- `CoupledSource(name="coupled_source")` creates the source.
- `src.block(name).role(role)` -> a `_CsField` (a `Var` with environment name `"<block>::<role>"`).
  The roles are canonicalized to lowercase (`density`, `momentum_x`, `energy`, `velocity_x`, `pressure`,
  `temperature`, `scalar`...).
- `src.param(name, value)` -> a const `Param` (inlined as a constant register of the bytecode).
- `src.add(block, role=, expr=)` adds a term `d_t (block.role) += expr` ; several `add` on the
  same `(block, role)` add up.
- `src.add_pair(block_a, block_b, role=, expr=)` : exchange conservative by construction. `block_a`
  gains `+expr`, `block_b` loses `-expr` (the same subtree, in `Neg`), so `sum(role)` is conserved
  cell by cell. `block_a != block_b` required. Chainable (returns `self`).
- `src.compile(backend="production", verify_conservation=False)` -> `CompiledCoupledSource`. The
  backend documents the intent ; the numerics is identical (bytecode interpreted on the C++ side).

Opcodes (mirror of `pops::CsOp`) : `PUSHREG`(0), `ADD`(1), `SUB`(2), `MUL`(3), `DIV`(4), `NEG`(5),
`POW`(6), `SQRT`(7). Only `+ - * / ** -unary sqrt` plus field and constant are supported (any
other node raises a `TypeError`). Frozen capacity limits, diagnosed on the Python side before the
C++ boundary : 32 registers (inputs + constants), 16 source terms, 256 opcodes per term.

`CompiledCoupledSource` carries the flat ABI (`in_blocks`, canonical `in_roles`, `consts`,
`out_blocks`, `out_roles`, `prog_ops`, `prog_args`, `prog_lens`) and a reference numpy evaluator
`reference_terms(fields)` (same `Expr` as the bytecode). You attach via `sim.add_coupling(compiled)`
(-> `System.add_coupled_source`).

```python
from pops import dsl

src = dsl.CoupledSource("ionization")
ne = src.block("electrons").role("density")
ni = src.block("ions").role("density")
ng = src.block("neutrals").role("density")
k  = src.param("Kiz", 0.7)
src.add("electrons", role="density", expr=+k*ne*ng)
src.add("ions",      role="density", expr=+k*ne*ng)
src.add("neutrals",  role="density", expr=-k*ne*ng)
sim.add_coupling(src.compile(backend="production"))
```

`verify_conservation=True` checks symbolically, role by role, that the sum of the terms cancels
(each `+E` offset by a `-E` of the same structural body on another block) ; it raises a
`ValueError` explicitly otherwise. This is what `add_pair` guarantees by construction ; the flag extends
the check to hand-written couplings. The check is conservative : it may wrongly flag a
coupling written with algebraically equal but structurally different forms (`k*ne` vs
`ne*k`), never the reverse. Off by default : a deliberate net creation / destruction (ionization)
stays valid without the flag.

## Validation (check)

`m.check()` collects `known = cons_names + prim_defs + aux_names`, then `used` = all the
dependencies (`deps()`) appearing in each primitive definition, in `flux` x/y, in
`eigenvalues` x/y, in `source` and in `elliptic_rhs`. If `used - known` is non-empty, it raises
`ValueError("model '<name>': undefined variables [...]")`. Returns `True` otherwise.

`check()` does not verify : that `flux` / `eigenvalues` / `source` have been set, the completeness of the
layout, the validity of the roles, nor conservation. These errors surface later :

- `emit_cpp_brick` requires `set_primitive_state`, `set_conservative_from` (n_vars exprs), `set_flux`
  (n_vars/dir) and `set_eigenvalues`, otherwise `ValueError` at emission.
- `aux(name)` raises on an unknown aux name.
- the assignment of runtime indices raises if the model exceeds `kMaxRuntimeParams=32`.
- at attachment, `add_equation` rejects HLLC/Roe without a primitive `p`, an incorrect length of
  `names=`, and `names=` on the native `production` path (the names come from the `.so`).
- `CoupledSource.compile(verify_conservation=True)` checks conservation role by role.

## Complete example

Euler model with gravity-Poisson coupling, from declaration to run (copy-paste). Compilation
requires pops headers and a C++ compiler ; without them, `compile` raises.

```python
import numpy as np
import pops
from pops import dsl

GAMMA = 1.4

def build_euler_poisson():
    m = dsl.Model("euler_poisson")
    rho, rhou, rhov, E = m.conservative_vars(
        "rho", "rho_u", "rho_v", "E",
        roles=["Density", "MomentumX", "MomentumY", "Energy"])
    g = m.param("gamma", GAMMA)                      # const : inline + set_gamma
    u = m.primitive("u", rhou / rho)
    v = m.primitive("v", rhov / rho)
    p = m.primitive("p", (g - 1.0) * (E - 0.5 * rho * (u*u + v*v)))
    H = (E + p) / rho
    c = dsl.sqrt(g * p / rho)
    m.flux(x=[rhou, rhou*u + p, rhou*v, rho*H*u],
           y=[rhov, rhov*u, rhov*v + p, rho*H*v])
    m.eigenvalues(x=[u-c, u, u+c], y=[v-c, v, v+c])
    gx, gy = m.aux("grad_x"), m.aux("grad_y")        # E = -grad phi
    m.source([0.0, -rho*gx, -rho*gy, -(rhou*gx + rhov*gy)])
    m.elliptic_rhs(-1.0 * (rho - 1.0))               # self-consistent gravity
    prho, pu, pv, pp = m.primitive_vars(rho=rho, u=u, v=v, p=p)
    m.conservative_from([prho, prho*pu, prho*pv,
                         pp/(g-1.0) + 0.5*prho*(pu*pu + pv*pv)])
    m.check()
    return m

m = build_euler_poisson()
compiled = m.compile(backend="production")           # include / so_path auto, cache

n = 32
s = pops.System(n=n, L=1.0, periodic=True)
s.add_equation("gas", compiled,
               spatial=pops.FiniteVolume(limiter="minmod", riemann="hllc",
                                        variables="primitive"))
s.set_poisson(rhs="charge_density", solver="geometric_mg")

xs = (np.arange(n) + 0.5) / n
X, Y = np.meshgrid(xs, xs)
U = np.zeros((4, n, n))
U[0] = 1.0 + 0.3*np.exp(-((X-0.5)**2 + (Y-0.5)**2)/0.02)
U[3] = 1.0/(GAMMA - 1.0)
s.set_state("gas", U.reshape(-1).tolist())

nsteps = s.run(t_end=0.02, cfl=0.4)
final = np.array(s.get_state("gas")).reshape(4, n, n)
```

Note on `pops.FiniteVolume(limiter=, riemann=, variables=)` : `riemann` is the numerical flux
(`rusanov` / `hll` / `hllc` / `roe`), distinct from the physical flux `m.flux` ; `limiter` among
`none` / `minmod` / `vanleer` / `weno5` ; `variables` among `conservative` / `primitive`. HLLC / Roe
require a primitive named `p`.

## Pitfalls

1. **Two `Model`**. `pops.Model(state, transport, source, elliptic)` (in `__init__.py`) composes
   pre-compiled native bricks (`ModelSpec`) ; `pops.dsl.Model(name)` writes symbolic
   formulas. Different signatures and files.
2. **`flux` vs `eval_flux`**. `m.flux(x=, y=)` declares ; `m.eval_flux(U, aux, dir)` evaluates (numpy).
   Distinct methods. On `HyperbolicModel`, the collision is resolved the other way :
   `flux(U, aux, dir)` evaluates and `set_flux` declares.
3. **`primitive_vars(u=u, ...)` self-reference**. Passing a `Var` already defined of the same name (or a
   conservative) does not redefine it, otherwise the codegen would emit `const Real u = u;` -> NaN.
4. **`conservative_from` is required and manual**. The DSL does not invert the primitives ; you must
   supply `cons = f(prim)` explicitly for the complete codegen of the brick.
5. **Fixed aux names**. Only `phi` / `grad_x` / `grad_y` / `B_z` / `T_e`. An unknown name raises.
   `B_z` / `T_e` widen `n_aux` (4 / 5).
6. **ABI match (production)**. Loader and `_pops` module must share headers + compiler
   + C++ standard ; a discrepancy -> `add_native_block` raises "ABI incompatible". Do not force `std` for
   `production` : leave `std=None` to derive the standard from the loader (c++20 under Kokkos, c++23 otherwise).
7. **Runtime params on `aot` only**. `kind="runtime"` is frozen to the declaration value on
   `prototype` / `production`. `set_block_params` on a 100% const block raises. `runtime_param_names`
   sorted = the order of the C++ indices = the order expected by `set_block_params(name, values)`.
8. **`target="amr_system"` requires `backend="production"`**. No AMR `.so` path outside native. In AMR,
   HLLC / Roe / `primitive` are rejected at the Python facade (the C++ engine supports them : facade
   limitation).
9. **Cache key**. `so_path=None` caches by `model_hash` (formulas + roles + n_aux + gamma + params,
   including runtime declaration values) and `abi_key`. Changing a formula / a param / the
   toolchain -> cache miss -> recompilation. Passing `so_path=` forces the recompilation. Runtime
   params change at runtime via `set_block_params` without recompiling (on `aot`).
10. **Device-clean**. `production` uses named functors, not extended lambdas
    `__host__ __device__` (which segfaulted under nvcc). It is the only valid GPU / MPI path.
11. **CoupledSource**. No `.so`, no Python per cell ; bytecode interpreted in C++. Only
    `+ - * / ** -unary sqrt` ; limits 32 registers / 16 terms / 256 opcodes. `add_pair` guarantees
    conservation by construction ; `verify_conservation=True` checks the hand-written `.add`
    (conservative check : never a false positive in reverse).
