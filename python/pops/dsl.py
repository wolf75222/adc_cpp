"""pops.dsl : SYMBOLIC mini-DSL for physical models.

Python WRITES the formulas (named variables, expressions), not a function called per cell.
The operations (+, -, *, /, **, pops.dsl.sqrt) build an expression TREE. A
HyperbolicModel declares its conservative variables, its primitives (defined by formulas),
its flux, its eigenvalues, its source and its elliptic contribution.

    e = pops.dsl.HyperbolicModel("euler")
    rho, rhou, rhov, E = e.conservative_vars("rho", "rho_u", "rho_v", "E")
    u = e.primitive("u", rhou / rho)
    p = e.primitive("p", (gamma - 1.0) * (E - 0.5 * rho * (u*u + v*v)))
    e.set_flux(x=[rhou, rhou*u + p, ...], y=[...])
    e.set_eigenvalues(x=[u - c, u, u + c], y=[...])

Backends available via m.compile(backend=..., target=...) :
  - "prototype" : NumPy/host JIT evaluator, first-order Rusanov, TEST only (host alone) ;
  - "aot"       : generates a flat-ABI .so (debug ABI), ahead-of-time compilation ;
  - "production": zero-copy native loader (add_native_block), RECOMMENDED by default ; device-clean
                  path (named functors, validated GH200, #97/#93) and MPI/AMR-ready.

Target (target=) :
  - "System"    : single-level system ;
  - "AmrSystem" : AMR-wired system (#92/#105), only with backend "production".

Ergonomics :
  - m.compile() auto-detects the pops includes and caches the .so by (model_hash, abi_key) (#103).
  - The physical roles (gamma, n_aux, B_z, T_e) are preserved and passed through to the C++.

Inter-species coupling (#131, #167) :
  - CoupledSource describes an exchange between blocks via DSL formulas (arbitrary cross source).
  - CoupledSource.add_pair(block_a, block_b, role, expr) guarantees conservation by
    construction (+expr on A, -expr on B, SAME subtree) ; DSL equivalent of
    add_collision / add_thermal_exchange on the C++ side. compile(verify_conservation=True) checks
    the property symbolically over all the terms (add and add_pair).
"""
import os
import shutil
import sys

import numpy as np

# Operator-first type system (Spec 2): a typed VIEW only. dsl.py is sometimes loaded as a
# top-level module without its package (the standalone-import test trick, e.g.
# test_projection_eig loads dsl.py via spec_from_file_location), where a relative import has no
# parent package; fall back to loading the sibling model/ PACKAGE by path (it is stdlib-only).
# The package uses relative intra-imports, so we register it under a name and give the spec its
# submodule search path; `from .spaces import ...` then resolves without importing pops/_pops.
try:
    from . import model as _model
except ImportError:  # pragma: no cover - exercised only by the standalone-import path
    import importlib.util as _ilu

    _mdir = os.path.join(os.path.dirname(__file__), "model")
    _mspec = _ilu.spec_from_file_location(
        "pops_model", os.path.join(_mdir, "__init__.py"),
        submodule_search_locations=[_mdir])
    _model = _ilu.module_from_spec(_mspec)
    sys.modules["pops_model"] = _model  # so the package's relative imports resolve
    _mspec.loader.exec_module(_model)


# --- Symbolic IR (single source of truth: pops.ir) ----------------------------
# All symbolic node classes, ops, visitors, and algebraic helpers now live in
# pops.ir.  They are re-imported here so every existing `dsl.X` reference keeps
# working without any change at the call site.
from pops.ir.expr import (Expr, _wrap, Const, Var, _Bin, Add, Sub, Mul, Div, Pow, Neg, Sqrt, Abs, Sign)  # noqa: E402,F401
from pops.ir.ops import sqrt, abs_, sign, eig_max_im, eig_lmin, eig_lmax, eig_all_real, left, right  # noqa: E402,F401
from pops.ir.values import _EIG_FIELDS, _EIG_PREDICATES, EigWitness, StateRef, RuntimeParamRef  # noqa: E402,F401
from pops.ir.visitors import _children, _expr_uses_cons_or_prim, _key  # noqa: E402,F401
from pops.ir.lowering import _is_const, _s_add, _s_neg, _s_sub, _s_mul, _s_div, _s_pow, diff  # noqa: E402,F401


# --- codegen infrastructure (single source of truth: pops.codegen) ----------------------------
# All build/compile infrastructure (pops_header_signature, pops_include, loader_cxx_std,
# resolve_auto_backend, _check_headers_match_module, cache helpers, ABI guards, native flags)
# now live in pops.codegen.  They are re-imported here so every existing `dsl.X` reference
# keeps working without any change at the call site.
from pops.codegen.toolchain import (  # noqa: E402,F401
    pops_header_signature, pops_include, loader_cxx_std, _pops_cxx_std_from_module, _pops_module,
    loader_cxx_compiler, resolve_auto_backend, _check_headers_match_module, _default_cxx,
    _STD_ALIAS, _run_compile, _probe_cache, _probe_cxx_std,
    _native_kokkos_root, _libomp_prefix, _native_feature_key,
    _warn_kokkos_parity, _env_truthy, _native_kokkos_compiler, _pops_import_lib,
    _native_kokkos_flags, pops_loader_build_flags,
)
from pops.codegen.cache import (  # noqa: E402,F401
    pops_cache_dir, _cache_so_path, _process_so_backend, _backend_distinct_so_path,
    _record_so_backend, _native_mpi_flags, _platform_cache_key, _dsl_optflags,
    _DSL_OPTFLAGS_DEFAULT,
)
from pops.codegen.abi import (  # noqa: E402,F401
    module_header_signature, check_compiled_matches_module, _abi_key_python,
)
from pops.codegen.cpp_writer import (  # noqa: E402,F401
    _cpp_expand, _cpp_cse, _cse_emit, _collect_eig_witnesses, _eig_witness_helpers,
    _count_cons_denoms, _recip_rewrite, _dir_key, _roe_validate, _cpp_roe,
)
from pops.codegen.compile import (  # noqa: E402,F401
    _BACKEND_CAPS, _BACKENDS,
    model_hash as _cg_model_hash,
    adder_for as _cg_adder_for,
    emit_cpp_so_source as _cg_emit_cpp_so_source,
    emit_cpp_aot_source as _cg_emit_cpp_aot_source,
    emit_cpp_native_loader as _cg_emit_cpp_native_loader,
    compile_so as _cg_compile_so,
    compile_aot as _cg_compile_aot,
    compile_native as _cg_compile_native,
    compile_or_jit as _cg_compile_or_jit,
    compile_model as _cg_compile_model,
    compile_problem, _module_to_model,
)
from pops.codegen.loader import CompiledModel, CompiledProblem  # noqa: E402,F401


# --- Aux channel: canonical layout ------------------------------------------
# The named auxiliary fields (aux('...')) are FIXED-index COMPONENTS of the aux channel
# (cf. pops::Aux / kAuxBaseComps on the C++ side). phi/grad_x/grad_y = BASE contract (3 components);
# the following ones (B_z, ...) WIDEN the channel -> the generated brick then declares n_aux so that
# the system sizes and populates the shared channel (cf. CompositeModel::n_aux, ensure_aux_width).
#
# INHERENT C++ <-> Python DUPLICATION: the table below MUST stay the MIRROR of the single C++
# source POPS_AUX_FIELDS (include/pops/core/state.hpp), from which load_aux (device read)
# and the host marshaling (python/system.cpp) are generated. Python does not read the C++ headers, so
# we cannot generate it: adding an extra aux field = 1 line here AND 1 line in POPS_AUX_FIELDS,
# with the SAME {name, index}. This is the only remaining duplication; the 3 C++ sites are now unified.
AUX_CANONICAL = {"phi": 0, "grad_x": 1, "grad_y": 2, "B_z": 3, "T_e": 4}
AUX_BASE_COMPS = 3

# Aux fields NAMED by the model (ADC-70 phase 1): m.aux_field("name"). Components starting from
# AUX_NAMED_BASE (= 5, just after T_e=4) -- the k-th declared name is component AUX_NAMED_BASE + k,
# read in C++ via aux.extra_field(k). MIRRORS of kAuxNamedBase / kAuxMaxExtra (include/pops/core/state.hpp,
# single C++ source). Decouples the user names from the canonical channel: B_z / T_e keep their indices
# 3 / 4 and their dedicated paths (set_magnetic_field / set_electron_temperature_from).
AUX_NAMED_BASE = 5
AUX_NAMED_MAX = 4  # maximum number of named aux fields per model (= kAuxMaxExtra on the C++ side)

# Bound on the number of RUNTIME parameters per block (P7-b). MIRROR of kMaxRuntimeParams
# (include/pops/runtime/runtime_params.hpp): the C++ carrier RuntimeParams has an array of this FIXED
# size (device-copiable without allocation), so a model exceeding the bound is rejected at codegen.
_K_MAX_RUNTIME_PARAMS = 32


def aux_n_aux(aux_names):
    """Aux channel width required by these CANONICAL fields: max(3, largest index + 1).
    Raises ValueError on an unknown name (a canonical aux field MUST be a component of pops::Aux)."""
    w = AUX_BASE_COMPS
    for nm in aux_names:
        if nm not in AUX_CANONICAL:
            raise ValueError("unknown aux field '%s': expected %s (components of pops::Aux)"
                             % (nm, sorted(AUX_CANONICAL)))
        w = max(w, AUX_CANONICAL[nm] + 1)
    return w


def aux_total_n_aux(aux_names, aux_extra_names):
    """TOTAL width of the aux channel: max of the canonical width (aux_n_aux) and, if NAMED fields
    (aux_field) are declared, AUX_NAMED_BASE + number of names (the last name = component
    AUX_NAMED_BASE + len-1). Without a named field -> aux_n_aux (historical path, bit-identical)."""
    w = aux_n_aux(aux_names)
    if aux_extra_names:
        w = max(w, AUX_NAMED_BASE + len(aux_extra_names))
    return w


# --- Physical roles: variable name -> VariableRole -------------------------
# CANONICAL mapping name -> physical role (cf. pops::VariableRole / role_name on the C++ side). Lets a
# generated brick DECLARE the MEANING of its components (density, momentum, energy...) instead of
# empty roles, so that inter-species couplings (System::add_collision / add_thermal_exchange)
# resolve via index_of(role) rather than via a literal index. The usual names of fluid models
# (rho, rho_u, u, p, E, n...) are recognized; an unknown name stays 'Custom'. A model can impose
# its roles explicitly (conservative_vars(..., roles=[...]) / set_primitive_state(..., roles=[...]))
# for a non-standard layout. Key = EXACT variable name, value = member of pops::VariableRole.
CANONICAL_ROLES = {
    "rho": "Density", "n": "Density", "density": "Density",
    "rho_u": "MomentumX", "rhou": "MomentumX", "mom_x": "MomentumX", "mx": "MomentumX",
    "rho_v": "MomentumY", "rhov": "MomentumY", "mom_y": "MomentumY", "my": "MomentumY",
    "rho_w": "MomentumZ", "rhow": "MomentumZ", "mom_z": "MomentumZ", "mz": "MomentumZ",
    "E": "Energy", "rho_E": "Energy", "ener": "Energy", "energy": "Energy",
    "u": "VelocityX", "v": "VelocityY", "w": "VelocityZ",
    "vx": "VelocityX", "vy": "VelocityY", "vz": "VelocityZ",
    "p": "Pressure", "pressure": "Pressure",
    "T": "Temperature", "temperature": "Temperature",
}


def role_of(name):
    """CANONICAL physical role of name @p name (member of pops::VariableRole), 'Custom' if unknown."""
    return CANONICAL_ROLES.get(name, "Custom")


def roles_for(names, override=None):
    """List of roles (pops::VariableRole members) parallel to @p names. @p override (optional):
    list of the same length explicitly fixing the roles (string 'Density'... or None to fall back
    on the canonical mapping of the name). Used for non-standard layouts where names are not enough."""
    if override is None:
        return [role_of(nm) for nm in names]
    if len(override) != len(names):
        raise ValueError("roles: %d roles for %d variables" % (len(override), len(names)))
    return [(r if r is not None else role_of(nm)) for nm, r in zip(names, override, strict=True)]



# --- Declarative hyperbolic model -----------------------------------------
class HyperbolicModel:
    """Hyperbolic model written as FORMULAS: conservative variables, primitives (defined by
    expressions), flux, eigenvalues, source, elliptic contribution. cf. module docstring."""

    def __init__(self, name):
        self.name = name
        self.cons_names = []
        self.prim_defs = {}     # name -> Expr (in terms of the cons / previous prims / aux)
        self.aux_names = []      # CANONICAL aux fields read (phi/grad/B_z/T_e), cf. AUX_CANONICAL
        self.aux_extra_names = []  # NAMED aux fields (aux_field): order = index AUX_NAMED_BASE + k
        self._flux = {}         # "x" / "y" -> list of Expr (one per conservative component)
        self._flux_terms = {}   # NAMED physical fluxes (flux_term, ADC-419): name -> {"x": [Expr],
                                # "y": [Expr]} (n_cons each). The implicit "default" flux lives in
                                # self._flux (m.flux / set_flux), so a model that only ever calls m.flux
                                # keeps this dict EMPTY -- the named keys enter _model_hash ONLY when
                                # non-empty (cache key preserved). A compiled Program selects a SUM of
                                # named fluxes via ctx.rhs(..., fluxes=[name, ...]); fluxes=["default"]
                                # (or no list) keeps the historical -div F (rhs_into), byte-identical.
        self._eig = {}          # "x" / "y" -> list of Expr (eigenvalues)
        self._wave_speeds = None  # {"x"/"y": (smin Expr, smax Expr)}: explicit SIGNED speeds
                                  # (set_wave_speeds); None = derived from the eigenvalues if 'p' (historical)
        self._ws_jacobian = None  # {"x"/"y": [[Expr]]} + meta (eig, blocks): EXACT signed speeds
                                  # from the eigenvalues of the flux jacobian (set_wave_speeds_from_jacobian)
        self._source = None     # list of Expr (one per component) or None
        self._source_terms = {}   # NAMED local sources (source_term): name -> [Expr] (n_cons each).
                                  # The implicit "default" source lives in self._source (m.source), so a
                                  # model that only ever calls m.source keeps this dict EMPTY -- the named
                                  # keys enter _model_hash ONLY when non-empty (cache key preserved).
        self._linear_sources = {}  # NAMED local linear operators (linear_source): name -> [[Expr]]
                                   # (n_cons x n_cons), coefficients linear in U (no cons/prim dependency).
        self._elliptic = None   # Expr (contribution to the elliptic right-hand side) or None
        self._elliptic_fields = {}  # NAMED elliptic fields (elliptic_field, ADC-419): name -> dict
                                    # {"rhs": Expr, "operator": str, "aux": [str]}. The unnamed default
                                    # stays in self._elliptic (m.elliptic_rhs); the named keys enter
                                    # _model_hash ONLY when non-empty (cache key preserved). The runtime
                                    # (a second elliptic operator / aux channel) is DEFERRED: the IR +
                                    # validation + hash + codegen-IR land, but ctx.solve_fields(field=)
                                    # raises NotImplementedError on lowering (cf. time.py).
        self._stab_speed = None  # Expr: STABILITY speed lambda* (None = fallback eigenvalues)
        self._stab_dt = None     # Expr: direct ADMISSIBLE step dt(U, aux) (None = no bound)
        self._src_freq = None    # Expr: frequency mu(U, aux) of the SOURCE (None = no bound)
        self._proj = None        # [Expr]: PROJECTION ponctuelle post-pas U <- P(U, aux) (ADC-177)
        self._src_jac = None     # [[Expr]] n x n: ANALYTIC Jacobian dS/dU (None = finite differences)
        self._hllc = False       # True: emit the HLLC capability (contact_speed + star state)
        self._riemann_hook_forms = {}  # ARBITRARY-formula overrides of the role-derived Riemann hooks
                                   # (ADC-456): name -> Expr. Codegen'd key: 'pressure' (single-state
                                   # signature, overrides the pressure(U) hook body). A descriptor or
                                   # None leaves the role-derived default. Folded into _model_hash.
        self._roe = False        # True: emit the ROE capability (roe_dissipation from the roles)
        self._roe_rows = None    # {"x": [Expr], "y": [Expr]}: roe_dissipation PROVIDED (outside roles)
        self._roe_jacobian = None  # {"x"/"y": [[Expr]]}: roe_dissipation from the FLUX JACOBIAN
                                   # (roe_from_jacobian, generic moment Roe via pops::roe_abs_apply)
        self.prim_state = []    # ordered names of the primitive state (Prim layout); for the codegen
        self.cons_from = None   # list of Expr: conservative in terms of the primitives (to_conservative)
        self.cons_roles = None  # explicit override of the conservative roles (otherwise canonical mapping)
        self.prim_roles = None  # explicit override of the primitive roles (otherwise canonical mapping)
        self.gamma = None       # adiabatic index of the block (EOS), read by the inter-species couplings
                                # on the System side. None -> symbol pops_compiled_gamma not emitted (the System
                                # then falls back to its historical default 1.4, strict backward compatibility).
        self._rate_operators = {}  # NAMED composite rate operators (rate_operator, Spec 2): name ->
                                   # {"flux": bool, "sources": [str], "fluxes": [str] | None}. A pure
                                   # Program-side ALIAS for ctx.rhs(flux=..., sources=..., fluxes=...): a
                                   # typed P.call(name) lowers to the SAME rhs IR, so the alias never enters
                                   # the model hash nor the codegen (its flux/sources are already hashed).

    def cons(self, name):
        self.cons_names.append(name)
        return Var(name, "cons")

    def conservative_vars(self, *names, roles=None):
        """Declare the conservative variables. @p roles (optional): list of the same length explicitly
        setting the physical role of each component (string 'Density'/'MomentumX'... or None
        to fall back on the canonical mapping of the name); useful for a non-standard layout where the names
        do not suffice to deduce the meaning. Without roles, the canonical name -> role mapping applies."""
        if roles is not None and len(roles) != len(names):
            raise ValueError("conservative_vars: %d roles for %d variables" % (len(roles), len(names)))
        self.cons_roles = list(roles) if roles is not None else None
        return tuple(self.cons(n) for n in names)

    def primitive(self, name, expr):
        """Define a primitive by its formula (in terms of the cons / previous primitives)."""
        self.prim_defs[name] = _wrap(expr)
        return Var(name, "prim")

    def aux(self, name):
        """CANONICAL auxiliary field (e.g. grad_x, grad_y, B_z, T_e) provided at execution. The name
        MUST be a key of AUX_CANONICAL. For an arbitrary NAMED field, see aux_field."""
        self.aux_names.append(name)
        return Var(name, "aux")

    def aux_field(self, name):
        """NAMED auxiliary field (ADC-70 phase 1) provided at execution per block via
        System.set_aux_field(bloc, name, array). Unlike aux(...) (CANONICAL components
        phi/grad/B_z/T_e), name is ARBITRARY: the k-th call reserves component
        AUX_NAMED_BASE + k of the aux channel (read in C++ via aux.extra_field(k)). Returns a Var
        usable in flux / source / eigenvalues / elliptic_rhs like any other aux variable.

        At most AUX_NAMED_MAX named fields per model (FIXED bound on the C++ side, Aux POD). A name already
        canonical (B_z, T_e, phi...) is REJECTED: those fields have their dedicated paths (aux('B_z') +
        set_magnetic_field, etc.); a duplicate named name is also rejected."""
        # The name becomes a C++ LOCAL in the generated formula (cf. _aux_locals_lines) AND the key of the
        # facade table: it must be a valid C++ identifier (letters/digits/_, not a
        # leading digit). Explicit rejection rather than a .so that does not compile.
        if not (isinstance(name, str) and name.isidentifier()):
            raise ValueError("aux_field(%r): invalid name (C++ identifier expected: "
                             "letters/digits/_, without a leading digit)" % (name,))
        if name in AUX_CANONICAL:
            raise ValueError(
                "aux_field('%s') : '%s' is a CANONICAL aux field; use aux('%s') (and the "
                "dedicated path, e.g. set_magnetic_field for B_z, set_electron_temperature_from "
                "for T_e)" % (name, name, name))
        if name in self.aux_extra_names:
            raise ValueError("aux_field('%s') : field already declared" % name)
        if len(self.aux_extra_names) >= AUX_NAMED_MAX:
            raise ValueError("aux_field('%s') : at most %d named aux fields per model "
                             "(kAuxMaxExtra bound on the C++ side)" % (name, AUX_NAMED_MAX))
        self.aux_extra_names.append(name)
        return Var(name, "aux")

    def _aux_locals_lines(self):
        """C++ locals for the aux fields read in a formula: canonical '<n>' <- a.<n> ;
        named '<n>' <- a.extra_field(k) (k = position in aux_extra_names). The local name is
        IDENTICAL to the one the Expr emits (Var.to_cpp), so the formula references it directly."""
        lines = ["    const pops::Real %s = a.%s;" % (n, n) for n in self.aux_names]
        lines += ["    const pops::Real %s = a.extra_field(%d);" % (n, k)
                  for k, n in enumerate(self.aux_extra_names)]
        return lines

    def _reads_aux(self):
        """True if a formula reads an aux field (canonical or named): drives the naming of the Aux
        parameter ('a' vs anonymous) so as not to trigger an unused-parameter warning."""
        return bool(self.aux_names) or bool(self.aux_extra_names)

    def _total_n_aux(self):
        """TOTAL width of the model's aux channel (canonical + named fields)."""
        return aux_total_n_aux(self.aux_names, self.aux_extra_names)

    def set_flux(self, x, y): self._flux = {"x": list(x), "y": list(y)}
    def set_eigenvalues(self, x, y): self._eig = {"x": list(x), "y": list(y)}

    def flux_term(self, name, x, y):
        """Declare a NAMED physical flux F_name(U, primitives, aux, params): exactly n_cons
        expressions per direction (x= the x-flux, y= the y-flux), free to depend on cons / primitives /
        aux / aux_field / params / constants -- the same dependency surface as set_flux. A named flux is
        OPT-IN: it is emitted only when a compiled time Program selects it (ctx.rhs(..., fluxes=[name,
        ...])) and is NEVER folded into the historical -div F (rhs_into). name == "default" is the
        backward-compatible alias of m.flux(...) (stored in self._flux, hash unchanged): so
        ctx.rhs(fluxes=["default"]) is byte-identical to the historical flux-only RHS. Other names must
        be valid identifiers, unique, and not collide with another named flux. A Program that requests
        several named fluxes assembles -div of their SUM (the codegen emits one kernel per name and
        sums them), so splitting the physical flux into named pieces that sum to it reproduces -div F."""
        n = self.n_vars
        if n == 0:
            raise ValueError("flux_term(%r): declare conservative_vars(...) first" % (name,))
        if not isinstance(name, str) or not name:
            raise ValueError("flux_term: name must be a non-empty string")
        x = [_wrap(e) for e in x]
        y = [_wrap(e) for e in y]
        if len(x) != n or len(y) != n:
            raise ValueError("flux_term('%s'): %d/%d expressions (x/y) for %d conservative variables"
                             % (name, len(x), len(y), n))
        if name == "default":
            self._flux = {"x": x, "y": y}   # equivalent to m.flux(...) -- the legacy default flux
            return
        if not name.isidentifier():
            raise ValueError("flux_term('%s'): name must be a valid identifier "
                             "(letters/digits/_, no leading digit)" % name)
        if name in self._flux_terms:
            raise ValueError("flux_term('%s'): already declared" % name)
        self._flux_terms[name] = {"x": x, "y": y}

    def set_wave_speeds(self, x, y):
        """Explicit SIGNED wave speeds per direction: x = (smin_x, smax_x), y = (smin_y,
        smax_y), expressions of cons / prims / aux. Emits ``wave_speeds(U, aux, dir, smin, smax)``
        on the generated brick WITHOUT requiring a primitive 'p': riemann='hll' becomes available for
        a pressureless model (moment system, isothermal...). The core only gates HLL on
        requires { m.wave_speeds(...) } (block_builder.hpp): no C++ change.

        Takes priority over the historical path (primitive 'p' -> wave_speeds = min/max of
        eigenvalues) when both exist. WITHOUT a call: strictly historical emission.
        If set_eigenvalues is NOT called, max_wave_speed (Rusanov / CFL) is derived from
        ``max(|smin|, |smax|)`` over the two expressions of the direction."""
        x, y = tuple(x), tuple(y)
        if len(x) != 2 or len(y) != 2:
            raise ValueError("set_wave_speeds : expected x=(smin, smax) and y=(smin, smax) "
                             "(got x=%d expression(s), y=%d)" % (len(x), len(y)))
        if self._ws_jacobian is not None:
            raise ValueError("set_wave_speeds : set_wave_speeds_from_jacobian already declared -- one "
                             "single wave_speeds provider")
        self._wave_speeds = {"x": (_wrap(x[0]), _wrap(x[1])),
                             "y": (_wrap(y[0]), _wrap(y[1]))}

    def set_wave_speeds_from_jacobian(self, x=None, y=None, eig="numeric", blocks=None):
        """EXACT signed wave speeds: smin/smax = extremes of the flux jacobian's eigenvalues
        A = dF/dU, computed NUMERICALLY per cell (pops::real_eig_minmax, Francis QR
        on a stack buffer, Gershgorin fallback on non-convergence = safe outer bound). Emits
        ``wave_speeds(U, aux, dir, smin, smax)`` (core HLL gate) and, without set_eigenvalues,
        ``max_wave_speed`` = ``max(|smin|, |smax|)`` over the same blocks.

        @p x, @p y : n_vars x n_vars matrices of expressions dA[i][j] = dF_dir[i]/dU[j]. None
        (default) = AUTODIFF of the declared flux via flux_jacobian(dir) (dsl.diff, primitives
        expanded by the chain rule) -- the jacobian can then not desynchronize from the
        flux. Providing explicit x/y only makes sense to bypass autodiff (hand-simplified
        forms); check_model then confronts them against the flux finite differences.

        @p eig : "numeric" (default) = jacobian entries emitted as formulas, per-block
        eigenvalues at runtime; "fd" = jacobian built BY COLUMNS from the finite differences of the
        COMPILED flux ((flux(U + eps e_k) - flux(U))/eps, ``eps = 1e-6 |U[0]| + 1e-30``, mirror of the
        flagsym != 1 branch of the reference MATLAB) -- generic bring-up/debug, never
        production (O(eps) truncation).

        @p blocks : None (default) = ONE full n_vars x n_vars block, the only unconditionally
        correct mode. Otherwise, a list of INDEX LISTS (possibly non-contiguous, e.g.
        [[0, 1, 4], [2, 3]]) applied to BOTH directions, or a dict {"x": [...], "y": [...]}
        (the block-triangular structures of dFx/dU and dFy/dU differ in general: for a
        moment system, the chains in x are contiguous and those in y are not).
        The extremes are taken over the union of the spectra of the diagonal sub-blocks A[idx][idx].
        CONTRACT: the caller ASSERTS that A is block-(lower-)triangular according to this
        partition (up to permutation) -- on an arbitrary matrix the sub-block extremes
        DO NOT BOUND the spectrum (counter-example [[0, k], [k, 0]]: spectrum +-k, 1x1 sub-blocks
        zero). Indices may be omitted (rows/columns carrying no extreme eigenvalue,
        cf. the skipped block of the reference MATLAB).

        Diagnostics: QR non-convergence silently falls back to the block's Gershgorin bound
        (WIDER, never wrong -- HLL stays stable, only more diffusive); a loss
        of hyperbolicity (complex eigenvalues) is not reported per cell -- verify it
        offline (check_model, golden type eigenvalues15_2D)."""
        if self._wave_speeds is not None:
            raise ValueError("set_wave_speeds_from_jacobian : set_wave_speeds already declared -- one "
                             "single wave_speeds provider")
        if eig not in ("numeric", "fd"):
            raise ValueError("set_wave_speeds_from_jacobian : eig 'numeric' | 'fd' (got %r)" % (eig,))
        nv = self.n_vars
        if (x is None) != (y is None):
            raise ValueError("set_wave_speeds_from_jacobian : provide x AND y, or neither (autodiff)")
        if eig == "fd" and x is not None:
            raise ValueError("set_wave_speeds_from_jacobian : eig='fd' builds the jacobian from "
                             "finite differences of the compiled flux -- x/y make no sense here")
        rows = {}
        if eig == "numeric":
            if x is None:
                if not self._flux:
                    raise ValueError("set_wave_speeds_from_jacobian : call set_flux(...) first "
                                     "(jacobian autodiff)")
                rows = {"x": self.flux_jacobian(0), "y": self.flux_jacobian(1)}
            else:
                for key, mat in (("x", x), ("y", y)):
                    if len(mat) != nv or any(len(r) != nv for r in mat):
                        raise ValueError("set_wave_speeds_from_jacobian : jacobian %s expected "
                                         "%d x %d" % (key, nv, nv))
                    rows[key] = [[_wrap(e) for e in r] for r in mat]
        def norm_blocks(blk, label):
            blk = [list(int(i) for i in b) for b in blk]
            seen = set()
            for b in blk:
                if not b:
                    raise ValueError("set_wave_speeds_from_jacobian : empty block (%s)" % label)
                local = set()
                for i in b:
                    if not (0 <= i < nv):
                        raise ValueError("set_wave_speeds_from_jacobian : index %d out of [0, %d) "
                                         "(%s)" % (i, nv, label))
                    if i in local:
                        raise ValueError("set_wave_speeds_from_jacobian : index %d present twice "
                                         "in the same block (%s)" % (i, label))
                    if i in seen:
                        raise ValueError("set_wave_speeds_from_jacobian : index %d present in "
                                         "two blocks (%s)" % (i, label))
                    local.add(i)
                    seen.add(i)
            return blk

        if blocks is None:
            per_dir = {"x": [list(range(nv))], "y": [list(range(nv))]}
        elif isinstance(blocks, dict):
            if set(blocks) != {"x", "y"}:
                raise ValueError("set_wave_speeds_from_jacobian : blocks dict expected with "
                                 "keys 'x' and 'y' (got %r)" % sorted(blocks))
            per_dir = {k: norm_blocks(blocks[k], k) for k in ("x", "y")}
        else:
            shared = norm_blocks(blocks, "x and y")
            per_dir = {"x": shared, "y": [list(b) for b in shared]}
        self._ws_jacobian = {"rows": rows or None, "eig": eig, "blocks": per_dir,
                             "explicit": x is not None}
    def set_source(self, s): self._source = [_wrap(e) for e in s]
    def set_elliptic_rhs(self, e): self._elliptic = _wrap(e)

    def elliptic_field(self, name, rhs, operator="poisson", aux=None):
        """Declare a NAMED elliptic field (ADC-419): an elliptic solve ``operator(field) = rhs(U)``
        whose solution + derived quantities populate the NAMED aux fields @p aux (default
        ``["phi", "grad_x", "grad_y"]``, the canonical electrostatic triple). @p rhs is an Expr of
        cons / primitives / aux / params (the elliptic right-hand side assembled from the state, the
        same surface as set_elliptic_rhs). @p operator names the elliptic operator (only ``"poisson"``
        is hosted by the runtime today). A named elliptic field is OPT-IN; the unnamed default stays in
        self._elliptic (m.elliptic_rhs). name must be a valid identifier, unique, and not collide with
        the default.

        SCOPE: the IR + validation + hash + codegen-IR for the named field land here, but the RUNTIME
        (a SECOND elliptic operator with its own aux-channel allocation) is DEFERRED -- the System hosts
        a single elliptic solve + the shared aux channel, so ctx.solve_fields(field=name) raises a clear
        NotImplementedError on lowering rather than mis-solving (cf. time.py / report)."""
        n = self.n_vars
        if n == 0:
            raise ValueError("elliptic_field(%r): declare conservative_vars(...) first" % (name,))
        if not isinstance(name, str) or not name:
            raise ValueError("elliptic_field: name must be a non-empty string")
        if name == "default":
            raise ValueError("elliptic_field('default'): the default elliptic field is m.elliptic_rhs "
                             "(set_elliptic_rhs); pass a distinct name")
        if not name.isidentifier():
            raise ValueError("elliptic_field('%s'): name must be a valid identifier "
                             "(letters/digits/_, no leading digit)" % name)
        if operator != "poisson":
            raise ValueError("elliptic_field('%s'): operator '%s' is not supported (only 'poisson')"
                             % (name, operator))
        if name in self._elliptic_fields:
            raise ValueError("elliptic_field('%s'): already declared" % name)
        aux = list(aux) if aux is not None else ["phi", "grad_x", "grad_y"]
        if not aux:
            raise ValueError("elliptic_field('%s'): aux must list at least one field" % name)
        for a in aux:
            if not (isinstance(a, str) and a.isidentifier()):
                raise ValueError("elliptic_field('%s'): aux field %r is not a valid identifier"
                                 % (name, a))
        rhs = _wrap(rhs)
        # The elliptic RHS brick (emit_cpp_elliptic_field, like the default emit_cpp_elliptic) reads
        # ONLY the conservative state (+ primitives derived from it), never the aux channel: the System
        # assembles f(U) per cell from the block state, before any aux is solved. An rhs reading an aux
        # field would compile to an undefined local -> reject it loud (the default set_elliptic_rhs has
        # the same surface). A source/flux READING the named field's solved aux is the supported pattern;
        # it is the named-elliptic RHS itself that must be a function of U only.
        rhs_aux = rhs.deps() & (set(AUX_CANONICAL) | set(self.aux_extra_names) | {"phi", "grad_x",
                                                                                 "grad_y", "B_z",
                                                                                 "T_e"})
        if rhs_aux:
            raise ValueError("elliptic_field('%s'): rhs may not read aux fields %s; the elliptic "
                             "right-hand side is a function of the conservative state only (the same "
                             "surface as m.elliptic_rhs). Read the SOLVED field's aux in a source/flux."
                             % (name, sorted(rhs_aux)))
        self._elliptic_fields[name] = {"rhs": rhs, "operator": operator, "aux": aux}

    def source_term(self, name, exprs):
        """Declare a NAMED local source S_name(U, primitives, aux, params): exactly n_cons
        expressions, free to depend on cons / primitives / aux / aux_field / params / constants. A
        named source is OPT-IN -- it is emitted only when a compiled time Program asks for it
        (ctx.rhs(..., sources=[name]) / ctx.source(name)) and is NEVER summed implicitly into the
        legacy total source. name == "default" is the backward-compatible alias of m.source([...])
        (stored in self._source, hash unchanged). Other names must be valid identifiers, unique, and
        must not collide with a linear_source."""
        n = self.n_vars
        if n == 0:
            raise ValueError("source_term(%r): declare conservative_vars(...) first" % (name,))
        if not isinstance(name, str) or not name:
            raise ValueError("source_term: name must be a non-empty string")
        exprs = [_wrap(e) for e in exprs]
        if len(exprs) != n:
            raise ValueError("source_term('%s'): %d expressions for %d conservative variables"
                             % (name, len(exprs), n))
        if name == "default":
            self._source = exprs   # equivalent to m.source([...]) -- the legacy default source
            return
        if not name.isidentifier():
            raise ValueError("source_term('%s'): name must be a valid identifier "
                             "(letters/digits/_, no leading digit)" % name)
        if name in self._source_terms:
            raise ValueError("source_term('%s'): already declared" % name)
        if name in self._linear_sources:
            raise ValueError("source_term('%s'): name collides with a linear_source" % name)
        self._source_terms[name] = exprs

    def linear_source(self, name, matrix):
        """Declare a NAMED local linear operator L_name(aux, params): an n_cons x n_cons matrix whose
        coefficients may depend on constants / params / aux / aux_field ONLY -- NOT on conservative or
        primitive variables (otherwise S(U) = L U is not linear in U and could not be treated as a
        local linear source by solve_local_linear). The operator is OPT-IN: never folded into m.source
        or ctx.rhs; a Program uses it explicitly via ctx.linear_source(name) / ctx.apply /
        ctx.solve_local_linear. Name must be a valid identifier, unique, and must not collide with a
        source_term."""
        n = self.n_vars
        if n == 0:
            raise ValueError("linear_source(%r): declare conservative_vars(...) first" % (name,))
        if not isinstance(name, str) or not name:
            raise ValueError("linear_source: name must be a non-empty string")
        if not name.isidentifier():
            raise ValueError("linear_source('%s'): name must be a valid identifier "
                             "(letters/digits/_, no leading digit)" % name)
        rows = [list(r) for r in matrix]
        if len(rows) != n or any(len(r) != n for r in rows):
            raise ValueError("linear_source('%s'): expected a %dx%d matrix (n_cons x n_cons)"
                             % (name, n, n))
        wrapped = [[_wrap(c) for c in row] for row in rows]
        for row in wrapped:
            for coeff in row:
                if _expr_uses_cons_or_prim(coeff):
                    raise ValueError("linear_source '%s' coefficients must not depend on "
                                     "conservative or primitive variables" % name)
        if name in self._linear_sources:
            raise ValueError("linear_source('%s'): already declared" % name)
        if name in self._source_terms:
            raise ValueError("linear_source('%s'): name collides with a source_term" % name)
        self._linear_sources[name] = wrapped

    def rate_operator(self, name, *, flux=True, sources=("default",), fluxes=None):
        """Declare a NAMED composite rate operator ``R_name = -div F + sum(sources)`` (Spec 2,
        operator-first). It is a Program-side ALIAS for ``ctx.rhs(flux=, sources=, fluxes=)``: a typed
        ``P.call(name, U[, fields])`` lowers to the SAME rhs IR as the explicit ``P.rhs(...)`` shortcut,
        so a model-free Program can address the RHS by one operator name instead of spelling out
        flux/sources. The alias carries no new numerics (its flux/sources are already in the model and
        the hash) -- it never enters the model hash nor the codegen. ``flux`` / ``sources`` / ``fluxes``
        have the same meaning as :meth:`Program.rhs`. ``name`` must be a valid identifier, unique among
        rate operators, and must not collide with a source_term / linear_source."""
        if self.n_vars == 0:
            raise ValueError("rate_operator(%r): declare conservative_vars(...) first" % (name,))
        if not (isinstance(name, str) and name.isidentifier()):
            raise ValueError("rate_operator(%r): name must be a valid identifier "
                             "(letters/digits/_, no leading digit)" % (name,))
        if name in self._rate_operators:
            raise ValueError("rate_operator('%s'): already declared" % name)
        if name in self._source_terms or name in self._linear_sources:
            raise ValueError("rate_operator('%s'): name collides with a source_term/linear_source"
                             % name)
        flx = list(fluxes) if fluxes else None
        if not flux and flx:
            raise ValueError("rate_operator('%s'): named fluxes require flux=True "
                             "(a source-only rate has no flux to divide)" % name)
        srcs = list(sources) if sources is not None else None
        self._rate_operators[name] = {"flux": bool(flux), "sources": srcs, "fluxes": flx}

    def stability_speed(self, expr):
        """STABILITY speed lambda* (expression of cons / prims / aux): drives the block CFL
        instead of ``max(|eigenvalues|)``. Emitted as ``stability_speed(U, aux, dir)`` (C++ trait
        ``HasStabilitySpeed``): System::step_cfl then uses it for the transport bound
        dt <= cfl*h/lambda*, while the Riemann solvers keep reading max_wave_speed
        (stability != accuracy). WITHOUT a call, the FALLBACK is strictly historical:
        max(abs(eigenvalues)) via max_wave_speed. Compiled like flux/source (no per-cell Python
        callback: compatible with GPU/MPI production). Wired into System AND AmrSystem (mono and
        multi-block; on the AMR side the reduction is evaluated on the COARSE level, where the CFL lives)."""
        self._stab_speed = _wrap(expr)

    def stability_dt(self, expr_dt):
        """Direct ADMISSIBLE step dt(U, aux) (expression > 0, in time units): local step
        bound, emitted as ``stability_dt(U, aux)`` (C++ trait ``HasStabilityDt``). System::step_cfl
        imposes dt <= min_cells(stability_dt) * substeps / stride (the cfl is NOT applied: the
        model already declares an admissible step). The most general form (stiff source, local coupling,
        non-reducible transport+source formula). WITHOUT a call, no additional bound (historical
        step policy). Compiled like flux/source (GPU/MPI production). Wired into System AND
        AmrSystem (mono and multi-block; on the AMR side evaluated on the COARSE level)."""
        self._stab_dt = _wrap(expr_dt)

    def source_frequency(self, expr_mu):
        """Local FREQUENCY mu(U, aux) [1/time] of the SOURCE (relaxation, collision, reaction):
        the 'second CFL' of the meeting -- bound dt <= cfl * substeps / (stride * max_cells(mu)),
        WITHOUT a space step (a source bounded in 1/time). Emitted as ``frequency(U, aux)`` on the
        generated SOURCE BRICK (C++ contract of source bricks, cf. physics/source.hpp);
        CompositeModel forwards it (HasSourceFrequency trait) and System/AmrSystem::step_cfl
        aggregate it. REQUIRES set_source/m.source (the frequency is a property of the source).
        WITHOUT a call, the source does not constrain the step (historical). Compiled (GPU/MPI production)."""
        self._src_freq = _wrap(expr_mu)

    def projection(self, exprs):
        """PROJECTION PONCTUELLE post-pas (ADC-177) : U <- P(U, aux), une expression par composante
        conservative (en fonction des cons / prims / aux). Emise comme ``project(U, aux)`` sur la
        brique hyperbolique generee (trait C++ ``HasPointwiseProjection``) ; le System l'applique sur
        les cellules VALIDES de chaque bloc a la FIN de chaque macro-pas ENTIER (apres transport +
        etage source + couplages ; jamais par etage RK -- semantique POST-PAS). CONTRAT : P doit etre
        une PROJECTION (idempotente : P(P(U)) == P(U)) et PONCTUELLE (aucune lecture de voisin). Les
        formules de realisabilite restent cote cas ; les clamps s'ecrivent SANS branche, en max/min
        via abs_ / sign : p.ex. positivite q >= 0 : (q + abs_(q)) / 2. Compilee comme flux/source
        (CSE comprise, production GPU/MPI -- remplace le callback Python par cellule). Backends
        'aot' (add_compiled_block) et 'production' System (add_native_block) ; le backend 'prototype'
        et target='amr_system' la REJETTENT explicitement (jamais d'ignore silencieux). SANS appel :
        aucun hook emis, chemin bit-identique."""
        exprs = [_wrap(e) for e in exprs]
        if len(exprs) != self.n_vars:
            raise ValueError("projection : %d expressions attendues (une par composante "
                             "conservative), recu %d" % (self.n_vars, len(exprs)))
        self._proj = exprs

    def projection_value(self, U, aux=None):
        """EVALUATEUR numpy de la projection ponctuelle emise (miroir exact du project(U, aux) C++) :
        U (n_vars, ...) -> U projete. Reference de test / prototypage hote. ValueError si
        projection([...]) n'a pas ete appelee."""
        if self._proj is None:
            raise ValueError("projection_value : appeler projection([...]) d'abord")
        env = self._env(U, aux)
        shape = np.asarray(U[0]).shape
        return np.stack([np.broadcast_to(e.eval(env), shape) for e in self._proj], axis=0)

    def source_jacobian(self, rows):
        """ANALYTICAL JACOBIAN of the source: dS/dU, n_vars x n_vars matrix of expressions
        (rows[r][c] = dS_r/dU_c, as a function of cons / prims / aux). Emitted as
        ``jacobian(U, aux, J)`` on the generated SOURCE brick, forwarded by CompositeModel
        (C++ trait ``HasSourceJacobian``): the Newton of the implicit source (IMEX /
        SourceImplicitBE) uses it INSTEAD of finite differences -- exactness (no more
        fd_eps noise) and saved source evaluations. REQUIRES m.source. WITHOUT a call: historical
        finite differences, bit-identical."""
        self._src_jac = [[_wrap(e) for e in row] for row in rows]

    def enable_hllc(self):
        """Emits the HLLC CAPABILITY (audit wave 3): ``contact_speed`` (Toro) + ``hllc_star_state``
        GENERATED from the block's ROLES (Density / MomentumX / MomentumY, Energy optional) and the
        primitive 'p' -- the core's contact-resolving HLLC solver (C++ trait HasHLLCStructure)
        then becomes available for THIS model, EVEN outside 4-variable Euler (3-var isothermal,
        moments with passive scalars: any component without a particular role is advected
        passively in the star state, Us[c] = fac*U[c]/rho). REQUIRES: roles Density/MomentumX/
        MomentumY declared + primitive 'p' (explicit error at emission otherwise)."""
        self._hllc = True
        return self

    # Riemann capability hooks whose body is a SINGLE-state formula (signature ``hook(U)``) and so
    # can be codegen'd from an arbitrary board Expr in the model's own symbols (ADC-456). The
    # two-state hooks (contact_speed / hllc_star_state, signature over UL/UR/pL/pR/sL/sR) are NOT
    # in this set: an arbitrary single-Expr override is ill-defined for them, so they keep the
    # role-derived default (selected by a capability-hook descriptor).
    _FORMULA_HOOKS = ("pressure",)

    def set_riemann_hooks(self, **forms):
        """Record ARBITRARY-formula overrides of the role-derived Riemann hooks (ADC-456, Spec 3
        section 11). Each keyword is a hook name; the value is an :class:`Expr` (an ``pops.math`` /
        dsl formula) that REPLACES the canonical role-derived body at codegen, or ``None`` to keep
        the default. Currently codegen'd hook: ``pressure`` (replaces the ``pressure(U)`` body,
        default = the primitive 'p' formula).

        Only :class:`Expr` values are recorded; a non-Expr (a :class:`pops.lib.BrickDescriptor`
        selecting a canonical scheme, or ``None``) is ignored and the role-derived default stands.
        An Expr passed for a two-state hook (``contact_speed`` / ``star_state`` / ``sound_speed``)
        raises: those keep the role-derived default selected via a hook descriptor. A formula
        referencing a quantity the model cannot provide still raises the clear capability error at
        emission. Without any Expr override the module hash and the emitted brick are bit-identical
        to the role-derived path."""
        for name, form in forms.items():
            if not isinstance(form, Expr):
                continue  # descriptor / None: role-derived default stands
            if name not in self._FORMULA_HOOKS:
                raise NotImplementedError(
                    "riemann hook %r does not yet accept an arbitrary board formula (its C++ "
                    "signature spans two states); pass a capability-hook descriptor "
                    "(e.g. pops.lib.riemann.hllc.contact_speed.euler()) for the role-derived "
                    "default. Formula override is available for: %s"
                    % (name, ", ".join(self._FORMULA_HOOKS)))
            self._riemann_hook_forms[name] = form
        return self

    def enable_roe(self):
        """Emits the ROE CAPABILITY (audit balance, GENERICITY_2026-06.md point 11):
        ``roe_dissipation(UL, AL, UR, AR, dir)`` = ``|A_roe| (UR - UL)`` GENERATED from the block's
        ROLES -- the core's Roe-like solver (C++ trait HasRoeDissipation, F = 1/2(FL+FR) - 1/2 d)
        becomes available for THIS model, EVEN outside 4-variable Euler:

        - roles Density/MomentumX/MomentumY + Energy: ideal-gas Roe algebra, exact
          TRANSCRIPTION of the canonical C++ path (sqrt(rho)-weighted averages, gamma-1 deduced from
          ``p/(E - 1/2 rho |v|^2)``, Harten entropy fix on the acoustic waves);
        - roles Density/MomentumX/MomentumY WITHOUT Energy (isothermal / pseudo-pressure): same
          decomposition without the energy row, LOCAL sound speed c = sqrt(p/rho) Roe-averaged
          (standard generalization outside ideal gas);
        - any component OUTSIDE the fluid roles is treated as a PASSIVE SCALAR carried by the
          entropy wave (row identical to the tangential momentum, phi = q/rho).

        REQUIRES: roles Density/MomentumX/MomentumY declared + primitive 'p' (explicit error at
        emission otherwise). Without a call: nothing emitted, riemann='roe' stays Euler-4-var-only.

        EXCLUSIVE with m.roe_dissipation: the capability from the roles and the dissipation PROVIDED by
        the user are two providers of the SAME roe_dissipation hook -- declaring both
        raises (one single provider)."""
        if self._roe_rows is not None:
            raise ValueError("enable_roe : roe_dissipation(...) already provided -- one single provider "
                             "of the roe_dissipation hook (capability from the roles OR provided)")
        if self._roe_jacobian is not None:
            raise ValueError("enable_roe : roe_from_jacobian() already declared -- one single provider "
                             "of the roe_dissipation hook")
        self._roe = True

    def roe_dissipation(self, x, y):
        """Roe dissipation PROVIDED by the user (outside the fluid-role families): n_vars
        expressions per direction (rows d_i), emitted as the C++ hook
        ``roe_dissipation(UL, AL, UR, AR, dir)`` = d (HasRoeDissipation trait; the core does
        F = 1/2(FL+FR) - 1/2 d, cf. RoeFlux). It is the 'provided' counterpart of m.enable_roe (generated
        from the ROLES): here the user writes THEIR eigenstructure -- same spirit as
        m.source_jacobian (provided, not invented). The helper m.flux_jacobian(dir) (A = dF/dU
        auto-derived by dsl.diff) assists this writing.

        TWO-STATE VOCABULARY: each variable/primitive must be wrapped by dsl.left(...) (state
        UL) or dsl.right(...) (state UR); a Roe average is therefore written explicitly
        (left(sqrt(rho))*left(u) + right(sqrt(rho))*right(u)) / (left(sqrt(rho)) + right(sqrt(rho))).
        A BARE variable (without a marker) raises at declaration (undetermined state).

        @p x, @p y : lists of n_vars expressions (rows for dir=0 and dir=1). TWO EXPLICIT sets
        (no role mapping here): at dir=0 the normal component is the x axis, at dir=1 the y axis.

        Guards: length n_vars per direction; each variable under left/right; conflict with
        enable_roe (one single provider of the hook) -> error. WITHOUT a call: nothing emitted (bit-identical).
        Requires the 'aot' or 'production' backend (the hook is emitted in the generated brick)."""
        if self._roe:
            raise ValueError("roe_dissipation : enable_roe() already called -- one single provider of the "
                             "roe_dissipation hook (capability from the roles OR provided)")
        if self._roe_jacobian is not None:
            raise ValueError("roe_dissipation : roe_from_jacobian() already declared -- one single "
                             "provider of the roe_dissipation hook")
        rx, ry = list(x), list(y)
        if len(rx) != self.n_vars or len(ry) != self.n_vars:
            raise ValueError("roe_dissipation : %d expressions expected per direction (got x=%d, "
                             "y=%d)" % (self.n_vars, len(rx), len(ry)))
        rows = {"x": [_wrap(e) for e in rx], "y": [_wrap(e) for e in ry]}
        for key in ("x", "y"):
            for e in rows[key]:
                _roe_validate(e, False)  # rejects any variable outside a left()/right() marker
        self._roe_rows = rows

    def roe_from_jacobian(self):
        """Generic moment Roe: emit the hook ``roe_dissipation(UL, AL, UR, AR, dir)`` =
        ``|A| (UR - UL)`` with ``A = dF_dir/dU`` the flux Jacobian (m.flux_jacobian, autodiff)
        evaluated at the ARITHMETIC MEAN interface state ``Uavg = 1/2 (UL + UR)``, and ``|A|`` via
        the matrix-sign kernel ``pops::roe_abs_apply`` (dense_eig.hpp): for a real-diagonalizable A
        this is ``R |Lambda| R^-1`` exactly, the dissipation of the reference flux_ROE. On a complex
        or singular spectrum the kernel returns false and the hook FALLS BACK to a spectral-radius
        (Rusanov) dissipation ``rho (UR - UL)``, ``rho = max(|lmin|, |lmax|)`` of
        ``pops::real_eig_minmax(A)`` -- so the dissipation is always well defined.

        Unlike m.enable_roe (which needs fluid roles Density/MomentumX/MomentumY + primitive 'p'),
        this path needs NEITHER -- it is the GENERIC provider for a moment hierarchy (HyQMOM), making
        riemann='roe' available with no Euler-4-var assumption. The FULL n_vars x n_vars Jacobian is
        always eigendecomposed (as the reference flux_ROE does), not a block partition.

        EXCLUSIVE with m.enable_roe and m.roe_dissipation: the three are providers of the SAME
        roe_dissipation hook (declaring more than one raises). Requires set_flux(...) and the 'aot'
        or 'production' backend (the hook is emitted in the generated brick). WITHOUT a call: nothing
        emitted (bit-identical cache key)."""
        if self._roe:
            raise ValueError("roe_from_jacobian : enable_roe() already called -- one single provider "
                             "of the roe_dissipation hook")
        if self._roe_rows is not None:
            raise ValueError("roe_from_jacobian : roe_dissipation(...) already provided -- one single "
                             "provider of the roe_dissipation hook")
        self._roe_jacobian = {"x": self.flux_jacobian(0), "y": self.flux_jacobian(1)}

    def flux_jacobian(self, dir):
        """Flux jacobian A = dF_dir/dU : n_vars x n_vars matrix of expressions, A[i][j] =
        d(flux_dir[i])/d(cons[j]), auto-derived from the declared fluxes (via dsl.diff with primitive
        substitution). CONSTRUCTION HELPER (the user uses it to write m.roe_dissipation):
        EMITS NOTHING by itself. @p dir : 0/'x' (x axis) or 1/'y' (y axis). REQUIRES set_flux(...).

        The primitives are expanded by their definition (chain); a non-derived primitive
        stays a symbol in the result (evaluating it numerically requires an env containing its values,
        e.g. HyperbolicModel._env)."""
        if not self._flux:
            raise ValueError("flux_jacobian : call set_flux(...) first")
        key = _dir_key(dir)
        comps = self._flux.get(key, [])
        if len(comps) != self.n_vars:
            raise ValueError("flux_jacobian : flux %s expected with %d components (got %d)"
                             % (key, self.n_vars, len(comps)))
        defs = self.prim_defs
        return [[diff(comps[i], self.cons_names[j], defs) for j in range(self.n_vars)]
                for i in range(self.n_vars)]

    def left(self, expr):
        """Marks @p expr as evaluated on the LEFT state UL (sugar for dsl.left, m.roe_dissipation)."""
        return left(expr)

    def right(self, expr):
        """Marks @p expr as evaluated on the RIGHT state UR (sugar for dsl.right, m.roe_dissipation)."""
        return right(expr)

    def set_gamma(self, gamma):
        """Adiabatic index of the block (compressible EOS). Carried by the generated .so via the
        optional symbol pops_compiled_gamma, so that the System's inter-species couplings (collision,
        thermal exchange, T_e) use the RIGHT gamma instead of the historical default 1.4. Without a call,
        no gamma symbol is emitted (backward compat: the System keeps its default)."""
        self.gamma = float(gamma)

    def set_primitive_state(self, *vars_or_names, roles=None):
        """Declares the ORDERED layout of the primitive state (Prim): component names, in order.
        Necessary for the brick codegen (to_primitive fills Prim in this order). Each name must
        be a conservative variable or an already-defined primitive. @p roles (optional): same
        convention as conservative_vars (explicit per-component override, None = canonical mapping)."""
        self.prim_state = [v.name if isinstance(v, Var) else str(v) for v in vars_or_names]
        if roles is not None and len(roles) != len(self.prim_state):
            raise ValueError("set_primitive_state : %d roles for %d variables"
                             % (len(roles), len(self.prim_state)))
        self.prim_roles = list(roles) if roles is not None else None

    def set_conservative_from(self, exprs):
        """Formulas of the conservative state as a function of the primitives (one per conservative
        variable, in conservative_vars order). Used to generate to_conservative: the DSL cannot invert
        the primitives symbolically, so the user provides the inverse explicitly."""
        self.cons_from = [_wrap(e) for e in exprs]

    @property
    def n_vars(self): return len(self.cons_names)

    # --- operator-first typed view (Spec 2, S2-1) --------------------------------
    # A DERIVED, typed view of the model as spaces + a registry of typed operators
    # (pops.model). It carries NO numerics and does NOT touch the model hash or the
    # codegen: source_term / linear_source / elliptic_field / flux lower into typed
    # operators so a model-free Program can address them by signature (P.call, S2-2)
    # and the C++ codegen can dispatch by integer id (S2-6). The public
    # pops.model.Module front-end (S2-3) builds on the same primitives.
    def _aux_name_set(self):
        """Names that denote an auxiliary field read by a formula (canonical + named)."""
        return set(AUX_CANONICAL) | set(self.aux_extra_names)

    def _aux_requirements(self, exprs):
        """{'aux': [...]} of the aux fields the expressions read, or {} if none."""
        aux_set = self._aux_name_set()
        read = sorted({d for e in exprs for d in (e.deps() & aux_set)})
        return {"aux": read} if read else {}

    def state_space(self, name="U"):
        """Typed :class:`pops.model.StateSpace` view of the conservative state: its
        components and canonical physical roles. Derived; carries no data."""
        role_list = roles_for(self.cons_names, self.cons_roles)
        roles = dict(zip(self.cons_names, role_list, strict=True))
        return _model.StateSpace(name=name, components=tuple(self.cons_names),
                                 roles=roles, layout="cell")

    def field_space(self, name="fields"):
        """Typed :class:`pops.model.FieldSpace` view of the auxiliary surface the model
        reads (canonical aux + named aux fields, in read order, de-duplicated)."""
        comps = []
        for nm in list(self.aux_names) + list(self.aux_extra_names):
            if nm not in comps:
                comps.append(nm)
        return _model.FieldSpace(name=name, components=tuple(comps), layout="cell")

    def operator_registry(self, state_name="U"):
        """Typed :class:`pops.model.OperatorRegistry` derived from this model.

        Lowers the PDE shortcuts into typed operators (ids follow registration order):
        flux -> grid_operator ``(State) -> Rate(State)``; source_term -> local_source
        ``(State[, Fields]) -> Rate(State)``; linear_source -> local_linear_operator
        ``(Fields?) -> LocalLinearOperator(State, State)``; elliptic_field ->
        field_operator ``(State) -> FieldSpace``; projection -> projection
        ``(State) -> State``. The implicit defaults surface as ``flux_default`` /
        ``source_default`` / ``fields_from_state``. Pure view: no hash / codegen impact.
        """
        reg = _model.OperatorRegistry()
        state = self.state_space(state_name)
        fields = self.field_space()
        aux_set = self._aux_name_set()

        def reads_fields(exprs):
            return any(e.deps() & aux_set for e in exprs)

        # Flux divergence (grid_operator: State -> Rate(State)).
        if self._flux:
            reg.register(_model.Operator(
                "flux_default", "grid_operator",
                _model.Signature([state], _model.Rate(state)),
                capabilities={"local": False, "linear": False, "produces_rate": True,
                              "requires_ghosts": 1, "supports_device": True,
                              "default": True},
                source="dsl.flux"))
        for nm in sorted(self._flux_terms):
            reg.register(_model.Operator(
                nm, "grid_operator", _model.Signature([state], _model.Rate(state)),
                capabilities={"local": False, "linear": False, "produces_rate": True,
                              "requires_ghosts": 1, "supports_device": True},
                source="dsl.flux_term"))

        # Local sources (local_source: State[, Fields] -> Rate(State)).
        if self._source is not None:
            rf = reads_fields(self._source)
            reg.register(_model.Operator(
                "source_default", "local_source",
                _model.Signature([state, fields] if rf else [state],
                                 _model.Rate(state)),
                capabilities={"local": True, "linear": False, "requires_fields": rf,
                              "produces_rate": True, "supports_device": True,
                              "default": True},
                requirements=self._aux_requirements(self._source),
                source="dsl.source"))
        for nm in sorted(self._source_terms):
            exprs = self._source_terms[nm]
            rf = reads_fields(exprs)
            reg.register(_model.Operator(
                nm, "local_source",
                _model.Signature([state, fields] if rf else [state],
                                 _model.Rate(state)),
                capabilities={"local": True, "linear": False, "requires_fields": rf,
                              "produces_rate": True, "supports_device": True},
                requirements=self._aux_requirements(exprs),
                source="dsl.source_term"))

        # Local linear operators (local_linear_operator: Fields? -> L(State, State)).
        for nm in sorted(self._linear_sources):
            coeffs = [c for row in self._linear_sources[nm] for c in row]
            rf = reads_fields(coeffs)
            reg.register(_model.Operator(
                nm, "local_linear_operator",
                _model.Signature([fields] if rf else [],
                                 _model.LocalLinearOperator(state, state)),
                capabilities={"local": True, "linear": True, "solve_i_minus_a": True,
                              "matrix_available": True, "supports_device": True},
                requirements=self._aux_requirements(coeffs),
                source="dsl.linear_source"))

        # Field operators (field_operator: State -> FieldSpace).
        if self._elliptic is not None:
            reg.register(_model.Operator(
                "fields_from_state", "field_operator",
                # The Poisson solve PRODUCES the canonical electrostatic triple; an externally
                # imposed aux (e.g. B_z) read by sources is part of field_space() but not produced
                # here, so the produced FieldSpace is the triple, not the full read surface.
                _model.Signature([state], _model.FieldSpace(
                    "fields", components=("phi", "grad_x", "grad_y"))),
                capabilities={"requires_solver": True, "supports_device": True,
                              "default": True},
                requirements={"elliptic_operator": "poisson"},
                source="dsl.elliptic_rhs"))
        for nm in sorted(self._elliptic_fields):
            info = self._elliptic_fields[nm]
            reg.register(_model.Operator(
                nm, "field_operator",
                _model.Signature([state],
                                 _model.FieldSpace(nm, components=tuple(info["aux"]))),
                capabilities={"requires_solver": True, "supports_device": True},
                requirements={"elliptic_operator": info["operator"]},
                source="dsl.elliptic_field"))

        # Pointwise projection (projection: State -> State).
        if self._proj is not None:
            reg.register(_model.Operator(
                "projection", "projection", _model.Signature([state], state),
                capabilities={"local": True, "idempotent": True,
                              "supports_device": True},
                source="dsl.projection"))

        # Composite rate operators (local_rate: State[, Fields] -> Rate(State)); aliases
        # for ctx.rhs(flux=, sources=, fluxes=), carried as a lowering hint for P.call.
        for nm in sorted(self._rate_operators):
            cfg = self._rate_operators[nm]
            src_names = cfg["sources"] if cfg["sources"] is not None else ["default"]
            needs = False
            for s in src_names:
                if s == "default":
                    needs = needs or (self._source is not None
                                      and reads_fields(self._source))
                elif s in self._source_terms:
                    needs = needs or reads_fields(self._source_terms[s])
            reg.register(_model.Operator(
                nm, "local_rate",
                _model.Signature([state, fields] if needs else [state],
                                 _model.Rate(state)),
                capabilities={"local": False, "linear": False, "requires_fields": needs,
                              "produces_rate": True, "supports_device": True},
                lowering={"flux": cfg["flux"], "sources": cfg["sources"],
                          "fluxes": cfg["fluxes"]},
                source="dsl.rate_operator"))
        return reg

    # --- evaluation (CPU interpreter, numpy) ---
    def _env(self, U, aux):
        """Environment: cons (from U), aux (provided), then derived primitives (insertion
        order = dependency order)."""
        env = {self.cons_names[i]: U[i] for i in range(len(self.cons_names))}
        if aux:
            env.update(aux)
        for pname, pexpr in self.prim_defs.items():
            env[pname] = pexpr.eval(env)
        return env

    def flux(self, U, aux, dir):
        """Physical flux in direction dir (0=x, 1=y). U: numpy (n_vars, ...)."""
        env = self._env(U, aux)
        comps = self._flux["x" if dir == 0 else "y"]
        return np.stack([np.broadcast_to(c.eval(env), U[0].shape) for c in comps], axis=0)

    def max_wave_speed(self, U, aux, dir):
        """max_k max_cells ``|lambda_k|``: Rusanov / CFL bound. Source: eigenvalues
        (legacy); WITHOUT set_eigenvalues, ``max(|smin|, |smax|)`` of the explicit signed speeds
        (set_wave_speeds), an exact mirror of the C++ emission."""
        env = self._env(U, aux)
        key = "x" if dir == 0 else "y"
        if not self._eig.get(key) and self._ws_jacobian is not None:
            lo, hi = self._ws_jacobian_value(U, env, key)
            return max(float(np.max(np.abs(lo))), float(np.max(np.abs(hi))))
        exprs = self._eig.get(key) or (list(self._wave_speeds[key])
                                       if self._wave_speeds is not None else None)
        if not exprs:
            raise ValueError("max_wave_speed: neither set_eigenvalues(...) nor set_wave_speeds(...) nor "
                             "set_wave_speeds_from_jacobian(...) declared on model '%s'"
                             % self.name)
        return max(float(np.max(np.abs(np.asarray(e.eval(env))))) for e in exprs)

    def _ws_jacobian_value(self, U, env, key):
        """Numpy evaluator of the jacobian path: extremes of the real parts of the eigenvalues
        of the sub-blocks, per sample (mirror of the emitted wave_speeds; np.linalg.eigvals)."""
        ws = self._ws_jacobian
        nv = self.n_vars
        nsmp = int(np.asarray(U[0]).reshape(-1).shape[0])
        if ws["eig"] == "fd":
            base = np.stack([np.broadcast_to(np.asarray(c.eval(env), dtype=float), (nsmp,))
                             if hasattr(c, "eval") else np.full((nsmp,), float(c))
                             for c in (self._flux[key])], axis=0)
            J = np.empty((nsmp, nv, nv))
            Uflat = np.stack([np.broadcast_to(np.asarray(env[c], dtype=float), (nsmp,))
                              for c in self.cons_names], axis=0)
            for k in range(nv):
                eps = 1e-6 * np.abs(Uflat[0]) + 1e-30
                Up = Uflat.copy()
                Up[k] += eps
                envp = self._env(Up, {n: env[n] for n in self.aux_names} if self.aux_names else None)
                Fp = np.stack([np.broadcast_to(np.asarray(c.eval(envp), dtype=float), (nsmp,))
                               for c in self._flux[key]], axis=0)
                J[:, :, k] = ((Fp - base) / eps).T
        else:
            rows = ws["rows"][key]
            J = np.empty((nsmp, nv, nv))
            for i in range(nv):
                for j in range(nv):
                    J[:, i, j] = np.broadcast_to(
                        np.asarray(rows[i][j].eval(env), dtype=float), (nsmp,))
        lo = np.full((nsmp,), np.inf)
        hi = np.full((nsmp,), -np.inf)
        for b in ws["blocks"][key]:
            idx = np.asarray(b)
            lam = np.linalg.eigvals(J[:, idx[:, None], idx[None, :]])
            lo = np.minimum(lo, lam.real.min(axis=1))
            hi = np.maximum(hi, lam.real.max(axis=1))
        return lo, hi

    def _flux_jacobian_spectral_radius(self, U, aux, dir):
        """Spectral radius max_cells max_k |Re(lambda_k)| of the FULL dense Jacobian A = dF_dir/dU,
        evaluated by CENTRAL finite differences on the interpreted flux. Independent of any declared
        partition (set_wave_speeds_from_jacobian blocks=...): serves as a non-circular reference bound
        against max_wave_speed. Returns None if a perturbed state leaves the domain (non-finite flux)
        -- in which case nothing can be concluded."""
        nv = self.n_vars
        U = np.asarray(U, dtype=float)
        nsmp = U.shape[1]
        J = np.empty((nsmp, nv, nv))
        for j in range(nv):
            eps = 1e-6 * np.abs(U[j]) + 1e-7
            Up = U.copy()
            Up[j] = Up[j] + eps
            Um = U.copy()
            Um[j] = Um[j] - eps
            Fp = self.flux(Up, aux, dir)
            Fm = self.flux(Um, aux, dir)
            if not (bool(np.all(np.isfinite(Fp))) and bool(np.all(np.isfinite(Fm)))):
                return None
            for i in range(nv):
                J[:, i, j] = (np.broadcast_to(Fp[i], (nsmp,))
                              - np.broadcast_to(Fm[i], (nsmp,))) / (2.0 * eps)
        lam = np.linalg.eigvals(J)
        return float(np.max(np.abs(lam.real)))

    def wave_speeds_value(self, U, aux, dir):
        """Numpy evaluator of the signed speeds (smin, smax) -- mirror of the emitted wave_speeds:
        explicit pair (set_wave_speeds) if declared, otherwise min/max of the eigenvalues (legacy
        path, which requires 'p' to be EMITTED but remains evaluable here)."""
        env = self._env(U, aux)
        key = "x" if dir == 0 else "y"
        if self._wave_speeds is not None:
            lo, hi = self._wave_speeds[key]
            return (np.asarray(lo.eval(env), dtype=float),
                    np.asarray(hi.eval(env), dtype=float))
        if self._ws_jacobian is not None:
            return self._ws_jacobian_value(U, env, key)
        eigs = [np.asarray(e.eval(env), dtype=float) for e in self._eig.get(key, [])]
        if not eigs:
            raise ValueError("wave_speeds_value: neither set_wave_speeds(...) nor set_eigenvalues(...) "
                             "declared on model '%s'" % self.name)
        eigs = list(np.broadcast_arrays(*eigs)) if len(eigs) > 1 else eigs  # mixed shapes (constant lambda)
        return (np.min(np.stack(eigs), axis=0), np.max(np.stack(eigs), axis=0))

    def source_value(self, U, aux):
        """Source term (numpy (n_vars, ...)), or zeros if not defined. A model that declares only
        NAMED sources (no m.source default) cannot answer the legacy total-source query: the named
        terms are never summed implicitly, so an old stepper asking for the total source is rejected
        (use pops.compile_problem(...) with a time Program, or define m.source(...) explicitly)."""
        if self._source is None:
            if self._source_terms:
                raise ValueError("model has multiple named sources; use pops.compile_problem(...) "
                                 "or define m.source(...) explicitly")
            return np.zeros_like(U)
        env = self._env(U, aux)
        return np.stack([np.broadcast_to(s.eval(env), U[0].shape) for s in self._source], axis=0)

    def to_python_flux(self, aux=None):
        """Produces an pops.PythonFlux (host backend) from the formulas: the model RUNS
        (interpreted on CPU). aux: dict name -> array (auxiliary fields), frozen for this flux."""
        import pops
        a = aux or {}
        return pops.PythonFlux(
            lambda U, d: self.flux(U, a, d),
            lambda U: max(self.max_wave_speed(U, a, 0), self.max_wave_speed(U, a, 1)))

    def check(self):
        """Checks that every referenced variable (primitives, flux, eigenvalues, source) is
        properly declared (cons / prim / aux). Raises ValueError otherwise (dependency check)."""
        known = (set(self.cons_names) | set(self.prim_defs) | set(self.aux_names)
                 | set(self.aux_extra_names))  # named aux fields (aux_field): ADC-70
        used = set()
        groups = [self._flux.get("x", []), self._flux.get("y", []),
                  self._eig.get("x", []), self._eig.get("y", []), self._source or [],
                  [e for e in (self._stab_speed, self._stab_dt, self._src_freq)
                   if e is not None],
                  self._proj or [],  # projection ponctuelle post-pas (ADC-177)
                  [e for row in (self._src_jac or []) for e in row]]
        if self._wave_speeds is not None:
            groups.append(list(self._wave_speeds["x"]) + list(self._wave_speeds["y"]))
        if self._ws_jacobian is not None and self._ws_jacobian["rows"] is not None:
            for d in ("x", "y"):
                groups.append([e for row in self._ws_jacobian["rows"][d] for e in row])
        if self._roe_rows is not None:
            groups.append(self._roe_rows["x"])
            groups.append(self._roe_rows["y"])
        for exprs in self._source_terms.values():  # NAMED sources (source_term)
            groups.append(exprs)
        for mat in self._linear_sources.values():  # NAMED linear operators (linear_source)
            groups.append([e for row in mat for e in row])
        for flx in self._flux_terms.values():  # NAMED fluxes (flux_term, ADC-419)
            groups.append(list(flx["x"]) + list(flx["y"]))
        for e in self.prim_defs.values():
            used |= e.deps()
        for grp in groups:
            for e in grp:
                used |= e.deps()
        if self._elliptic is not None:
            used |= self._elliptic.deps()
        for fld in self._elliptic_fields.values():  # NAMED elliptic fields (elliptic_field, ADC-419)
            used |= fld["rhs"].deps()
        missing = used - known
        if missing:
            raise ValueError("model '%s': undefined variables %s" % (self.name, sorted(missing)))
        # source_frequency is a property of the SOURCE (emitted on the generated source brick):
        # declaring it without a source would be SILENTLY lost -> explicit error.
        if self._src_freq is not None and self._source is None:
            raise ValueError("model '%s': source_frequency(...) declared without a source "
                             "(call m.source([...]) -- the frequency is emitted on the generated "
                             "source brick)" % self.name)
        if self._src_jac is not None and self._source is None:
            raise ValueError("model '%s': source_jacobian(...) declared without a source "
                             "(call m.source([...]) -- the Jacobian is emitted on the generated "
                             "source brick)" % self.name)
        # roe_dissipation and enable_roe are two providers of the SAME hook: exclusive (defensive;
        # already rejected at declaration). Structural re-check of the rows (left/right) along the way.
        if self._roe_rows is not None:
            if self._roe:
                raise ValueError("model '%s': enable_roe() and roe_dissipation(...) declared "
                                 "together -- a single provider of the roe_dissipation hook" % self.name)
            for key in ("x", "y"):
                for e in self._roe_rows[key]:
                    _roe_validate(e, False)
        # linear_source coefficients stay linear in U (defensive re-check: also caught at declaration).
        for nm, mat in self._linear_sources.items():
            for row in mat:
                for coeff in row:
                    if _expr_uses_cons_or_prim(coeff):
                        raise ValueError("linear_source '%s' coefficients must not depend on "
                                         "conservative or primitive variables" % nm)
        return True

    def check_model(self, samples=None, n_samples=64, seed=0, aux=None, rtol=1e-8, atol=1e-10,
                    raise_on_error=True, jac_rtol=1e-3, jac_atol=1e-9):
        """Generic NUMERICAL verification of the symbolic model (audit 2026-06, work item 6):
        evaluates the formulas on sample states and checks, when the piece exists:

        - finite flux (both directions);
        - finite source;
        - finite elliptic_rhs;
        - finite and real eigenvalues; finite max_wave_speed and >= 0;
        - consistency wave_speeds <-> max_wave_speed: ``max(|lambda_min|, |lambda_max|) <= mws``;
        - NON-CIRCULAR bounding of the spectrum: the spectral radius of the full dense flux Jacobian
          (central finite differences, independent of any blocks= partition) does not exceed
          max_wave_speed -- catches a set_wave_speeds_from_jacobian partition that does NOT bound the
          eigenvalues (mws underestimated, unsafe CFL) where the consistency above, derived from the
          SAME partition, still holds;
        - round-trip to_conservative(to_primitive(U)) ~= U (if prim_state + cons_from declared);
        - positivity of the Density-role components (and of the primitive 'p' if declared) on the
          samples (which are generated positive for these roles).

        @p jac_rtol, @p jac_atol: tolerances of the spectral bounding (radius <= mws*(1+jac_rtol)
        + jac_atol); relaxed to absorb the noise of the finite-difference Jacobian.

        @p samples: array (n_vars, N) of conservative states to test; None -> N = n_samples random
        states (fixed seed, reproducible): Density-role components in [0.1, 2], the others
        in [-1, 1]; an Energy-role component gets ``1 + |kinetic|`` to stay physical.
        @p aux: dict name -> value(s) of the auxiliary fields (default: zeros).
        @return dict {"ok": bool, "failures": [str], "n_samples": N}. raise_on_error=True (default)
        raises ValueError listing the failures. PRE-COMPILATION: checks the FORMULAS (the compiled .so
        emits exactly these formulas); the RUNTIME counterpart on an installed block is
        System.check_model(block)."""
        self.check()  # declared dependencies (raises if a variable does not exist)
        rng = np.random.default_rng(seed)
        nv = self.n_vars
        roles = roles_for(self.cons_names, self.cons_roles)
        if samples is None:
            U = rng.uniform(-1.0, 1.0, size=(nv, int(n_samples)))
            kinetic = np.zeros(int(n_samples))
            for i, r in enumerate(roles):
                if r == "Density":
                    U[i] = rng.uniform(0.1, 2.0, size=int(n_samples))
            for i, r in enumerate(roles):
                if r in ("MomentumX", "MomentumY"):
                    kinetic += U[i] ** 2
            for i, r in enumerate(roles):
                if r == "Energy":
                    U[i] = 1.0 + kinetic  # above the kinetic: pressure > 0 for an ideal gas
        else:
            U = np.asarray(samples, dtype=float)
            if U.ndim != 2 or U.shape[0] != nv:
                raise ValueError("check_model: samples must be (n_vars=%d, N)" % nv)
        a = {n: np.zeros(U.shape[1]) for n in (self.aux_names + self.aux_extra_names)}
        if aux:
            for k, v in aux.items():
                a[k] = np.broadcast_to(np.asarray(v, dtype=float), (U.shape[1],)).copy()
        failures = []

        def finite(x):
            return bool(np.all(np.isfinite(np.asarray(x, dtype=float))))

        for d, dn in ((0, "x"), (1, "y")):
            if not finite(self.flux(U, a, d)):
                failures.append("flux %s non-finite on the samples" % dn)
        if self._source is not None and not finite(self.source_value(U, a)):
            failures.append("source non-finite on the samples")
        if self._elliptic is not None:
            env = self._env(U, a)
            if not finite(self._elliptic.eval(env)):
                failures.append("elliptic_rhs non-finite on the samples")
        env = self._env(U, a)
        for d in ("x", "y"):
            for k, e in enumerate(self._eig.get(d, [])):
                lam = np.asarray(e.eval(env), dtype=float)
                if np.iscomplexobj(lam):
                    failures.append("eigenvalue %s[%d] complex (non-hyperbolic system?)" % (d, k))
                elif not finite(lam):
                    failures.append("eigenvalue %s[%d] non-finite" % (d, k))
        if self._wave_speeds is not None:
            for d in ("x", "y"):
                lo = np.asarray(self._wave_speeds[d][0].eval(env), dtype=float)
                hi = np.asarray(self._wave_speeds[d][1].eval(env), dtype=float)
                if not (finite(lo) and finite(hi)):
                    failures.append("wave_speeds %s (explicit) non-finite" % d)
                elif bool(np.any(lo > hi)):
                    failures.append("wave_speeds %s (explicit): smin > smax on some samples" % d)
        for d, dn in ((0, "x"), (1, "y")):
            mws = self.max_wave_speed(U, a, d)
            if not np.isfinite(mws) or mws < 0:
                failures.append("max_wave_speed %s non-finite or negative (%r)" % (dn, mws))
            else:
                # consistency wave_speeds <-> max_wave_speed: the SIGNED extremes actually emitted
                # (explicit pair if declared, otherwise eigenvalues) must be covered by the
                # Rusanov / CFL bound.
                lo, hi = self.wave_speeds_value(U, a, d)
                ext = max(float(np.max(np.abs(lo))), float(np.max(np.abs(hi))))
                if ext > mws * (1.0 + rtol) + atol:
                    failures.append("wave_speeds %s inconsistent with max_wave_speed (%g > %g)"
                                    % (dn, ext, mws))
                # NON-CIRCULAR bounding: the spectral radius of the dense flux Jacobian (central
                # FD, no partition) must be bounded by max_wave_speed. A blocks= partition that is
                # not really block-triangular yields sub-block extremes that do NOT bound the
                # spectrum -> mws too small, detected here.
                if self._flux:
                    radius = self._flux_jacobian_spectral_radius(U, a, d)
                    if radius is not None and radius > mws * (1.0 + jac_rtol) + jac_atol:
                        failures.append(
                            "partition %s: max_wave_speed (%g) does not bound the spectrum of the "
                            "flux Jacobian (spectral radius %g) -- the blocks= partition of "
                            "set_wave_speeds_from_jacobian does not bound the eigenvalues, the "
                            "Rusanov/CFL bound is underestimated" % (dn, mws, radius))
        # round-trip cons -> prim -> cons (when both directions are declared)
        if self.prim_state and self.cons_from is not None:
            penv = {nm: np.broadcast_to(np.asarray(env[nm], dtype=float), (U.shape[1],))
                    for nm in self.prim_state}
            U2 = np.stack([np.broadcast_to(np.asarray(e.eval(penv), dtype=float), (U.shape[1],))
                           for e in self.cons_from], axis=0)
            if not finite(U2):
                failures.append("to_conservative(to_primitive(U)) non-finite")
            elif not np.allclose(U2, U, rtol=rtol, atol=atol):
                err = float(np.max(np.abs(U2 - U)))
                failures.append("round-trip to_conservative(to_primitive(U)) != U (max deviation %g: "
                                "inconsistent conversions)" % err)
        # positivity: Density roles (conservative) and primitive 'p' (pressure) if declared
        for i, r in enumerate(roles):
            if r == "Density" and not bool(np.all(U[i] > 0)):
                failures.append("component '%s' (Density role) not strictly positive on the "
                                "samples" % self.cons_names[i])
        if "p" in self.prim_defs:
            p = np.asarray(env["p"], dtype=float)
            if not finite(p):
                failures.append("primitive 'p' (pressure) non-finite")
            elif not bool(np.all(p > 0)):
                failures.append("primitive 'p' (pressure) not strictly positive on physical "
                                "states (suspicious EOS)")
        report = {"ok": not failures, "failures": failures, "n_samples": int(U.shape[1])}
        if failures and raise_on_error:
            raise ValueError("check_model('%s'): %d failure(s):\n  - %s"
                             % (self.name, len(failures), "\n  - ".join(failures)))
        return report

    # --- RUNTIME parameters (P7-b): collection + index assignment + generated member -------------
    def _all_exprs(self):
        """All the Expr of the model (primitives, flux, eigenvalues, source, elliptic,
        cons_from). Used to discover the RuntimeParamRef nodes hidden in the tree."""
        out = list(self.prim_defs.values())
        for d in ("x", "y"):
            out += self._flux.get(d, [])
            out += self._eig.get(d, [])
        if self._wave_speeds is not None:  # explicit signed speeds: runtime params included
            for d in ("x", "y"):
                out += list(self._wave_speeds[d])
        if self._ws_jacobian is not None and self._ws_jacobian["rows"] is not None:
            for d in ("x", "y"):  # jacobian entries: runtime params included
                out += [e for row in self._ws_jacobian["rows"][d] for e in row]
        if self._source is not None:
            out += [_wrap(e) for e in self._source]
        if self.cons_from is not None:
            out += list(self.cons_from)
        if self._elliptic is not None:
            out.append(self._elliptic)
        if self._roe_rows is not None:  # Roe rows provided: discover their runtime params (via StateRef)
            out += self._roe_rows["x"] + self._roe_rows["y"]
        return out

    def runtime_param_nodes(self):
        """RuntimeParamRef nodes PRESENT in the formulas, deduplicated by name (the same param may
        appear several times but shares the SAME node object). Order SORTED by name (stable index
        = position in this list, mirror of RuntimeParams on the C++ side)."""
        seen = {}

        def walk(e):
            if isinstance(e, RuntimeParamRef):
                seen.setdefault(e.name, e)
                return
            for c in _children(e):
                walk(c)

        for e in self._all_exprs():
            walk(e)
        return [seen[k] for k in sorted(seen)]

    def assign_runtime_indices(self):
        """Assigns to each RuntimeParamRef its STABLE index (sorted order of names) and returns the
        ordered list of nodes. CALLED before any brick codegen: without this call, to_cpp() would raise
        (index -1). Idempotent (reassigns the same indices). Rejects a model exceeding the C++ bound
        kMaxRuntimeParams (otherwise the fixed-size array would overflow)."""
        nodes = self.runtime_param_nodes()
        if len(nodes) > _K_MAX_RUNTIME_PARAMS:
            raise ValueError(
                "model '%s': %d runtime parameters > kMaxRuntimeParams bound=%d "
                "(include/pops/runtime/runtime_params.hpp); reduce the number of runtime params"
                % (self.name, len(nodes), _K_MAX_RUNTIME_PARAMS))
        for k, node in enumerate(nodes):
            node.index = k
        return nodes

    def _runtime_params_member(self):
        """C++ line declaring the RuntimeParams member of a generated brick, initialized to the
        DECLARATION values (default without a runtime set call). Empty string if the model has no runtime
        param (brick strictly identical to history -> bit-identity of const params preserved)."""
        nodes = self.assign_runtime_indices()
        if not nodes:
            return ""
        vals = ", ".join(repr(node.value) for node in nodes)
        return ("  pops::RuntimeParams params{%d, {%s}};  // params RUNTIME (P7-b) : ecrasables a "
                "l'execution\n" % (len(nodes), vals))

    def has_runtime_params(self):
        """True if at least one formula reads a runtime parameter (kind='runtime')."""
        return bool(self.runtime_param_nodes())

    def _validate_hook_form(self, hook, form, allow_aux=True):
        """Reject an arbitrary-formula Riemann hook (ADC-456) that references a quantity the model
        cannot provide -- the same dependency rule as :meth:`check`, surfaced as a clear capability
        error. @p allow_aux: a single-state hook (e.g. pressure(U)) takes no Aux parameter, so an
        aux dependency is also a missing capability there."""
        known = set(self.cons_names) | set(self.prim_defs)
        if allow_aux:
            known |= set(self.aux_names) | set(self.aux_extra_names)
        missing = sorted(form.deps() - known)
        if missing:
            raise ValueError(
                "riemann hook %r references undeclared quantity %s: the formula needs model "
                "capabilities %s that are not provided (declare them, or use the role-derived "
                "default)" % (hook, missing, missing))

    # --- codegen (step 2 : symbolic tree -> compilable C++) ---
    def _codegen_exprs(self, exprs, cse, real="pops::Real", indent="    "):
        from pops.codegen import module_codegen as _cg
        return _cg._codegen_exprs(self, exprs, cse, real=real, indent=indent)

    def _live_prims(self, exprs, seed=()):
        from pops.codegen import module_codegen as _cg
        return _cg._live_prims(self, exprs, seed=seed)

    def _prim_block(self, live=None, hoist=False):
        from pops.codegen import module_codegen as _cg
        return _cg._prim_block(self, live=live, hoist=hoist)

    def _jac_entries(self):
        from pops.codegen import module_codegen as _cg
        return _cg._jac_entries(self)

    def emit_cpp(self, func=None, cse=True):
        """Generates a compilable C++ function computing the physical flux from the symbolic
        tree (each Expr node knows how to write itself in C++ via to_cpp).

        Produced signature : template <class Real> void <func>_flux(const Real* U, Real* F, int dir).
        Constants inlined ; each primitive becomes a local variable. cse=True (default) factors
        the common subexpressions (H, c...) into ``cseK_`` locals ; cse=False recomputes them inline.

        Step (2) of the DSL (see docs/ARCHITECTURE_CIBLE.md sect. 3) : HOST C++ (templatable on Real)."""
        from pops.codegen import module_codegen as _cg
        return _cg.emit_cpp(self, func=func, cse=cse)

    def emit_cpp_brick(self, name=None, namespace="pops_generated", cse=True,
                       hoist_reciprocals=False):
        """Generates a C++ BRICK satisfying the pops::HyperbolicModel concept (wrapping : step
        2bis). The produced struct uses StateVec / Aux / POPS_HD / Variables and exposes flux,
        max_wave_speed, to_primitive, to_conservative, conservative_vars, primitive_vars : it can
        therefore enter a CompositeModel and run in the compiled solver.

        Requires set_primitive_state(...) (Prim layout) and set_conservative_from([...]) (to_conservative,
        which the DSL cannot invert on its own). cse=True (default) factors the common
        subexpressions (H, c...) into ``cseK_`` locals. Still to do (see ARCHITECTURE_CIBLE.md sect. 3) :
        Kokkos/CUDA codegen, JIT."""
        from pops.codegen import module_codegen as _cg
        return _cg.emit_cpp_brick(self, name=name, namespace=namespace, cse=cse,
                                  hoist_reciprocals=hoist_reciprocals)

    def emit_cpp_source(self, name=None, namespace="pops_generated", cse=True,
                        hoist_reciprocals=False):
        """Generate a composable C++ SOURCE BRICK (in the pops sense) from self._source.

        The produced struct exposes apply(U, a) returning the source term S(U, aux), with one line per
        conservative component (S[i] = self._source[i].to_cpp()). It has the same form as the source
        bricks written by hand (NoSource, PotentialForce in pops/model/bricks.hpp) and can therefore
        enter as the Source parameter of a CompositeModel.

        CONVENTION: the auxiliary names (set via aux(...)) must be FIELDS of pops::Aux,
        because they are read directly as a.<name> (e.g. aux('grad_x') -> a.grad_x, aux('grad_y') ->
        a.grad_y). This convention is the same as that of the manual bricks, where the source reads
        the outer state only through the pops::Aux channel (potential and its gradient).

        Style identical to emit_cpp_brick (inlined constants, cons -> locals, primitives -> locals;
        plus, aux -> locals); cse=True factors the common sub-expressions. Raises ValueError if
        set_source(...) has not been called."""
        from pops.codegen import module_codegen as _cg
        return _cg.emit_cpp_source(self, name=name, namespace=namespace, cse=cse,
                                   hoist_reciprocals=hoist_reciprocals)

    def _emit_bricks(self, name=None, hoist_reciprocals=False):
        """Generate the bricks (hyperbolic + source + elliptic) and the CompositeModel<...> type
        shared by BOTH backends (JIT IModel and AOT). Source / elliptic OPTIONAL: without
        set_source -> pops::NoSource; without set_elliptic_rhs -> zero rhs (no Poisson coupling).
        @p hoist_reciprocals: codegen option propagated to the bricks (cf. emit_cpp_brick).
        Returns (nv, bricks_code, composite_type)."""
        from pops.codegen import module_codegen as _cg
        return _cg._emit_bricks(self, name=name, hoist_reciprocals=hoist_reciprocals)

    def _elliptic_field_registrations(self, nm):
        """Per named elliptic field (ADC-428): (field, brick_struct, phi_comp, gx_comp, gy_comp) for the
        native loader. The aux component of each output name is its channel index: a CANONICAL name
        (phi/grad_x/...) maps via AUX_CANONICAL; a model-named aux (aux_field) maps to
        AUX_NAMED_BASE + its position in aux_extra_names. A name the model never declared as an aux is
        rejected (the solve would write a component no source can read). gx/gy default to -1 (phi only)
        when the field lists fewer than 3 aux names."""
        from pops.codegen import module_codegen as _cg
        return _cg._elliptic_field_registrations(self, nm)

    def _emit_metadata(self, model_alias):
        """OPTIONAL metadata symbols of the .so block, read by dlsym on the System side. SHARED by both
        backends (JIT and AOT). The NAMES + ROLES are always emitted (POPS_EXPORT_BLOCK_METADATA):
        they come from the model's VariableSet (single source of truth), the System reads them instead of
        the u0.. fallback / no roles. The GAMMA is emitted (POPS_EXPORT_BLOCK_GAMMA) only if set_gamma(...)
        has been called; otherwise no gamma symbol -> the System keeps its default 1.4 (backward-compat).

        @p model_alias must be an alias WITHOUT a top-level comma (the preprocessor splits
        macro arguments on commas): callers pass a `using ... = CompositeModel<...>`."""
        from pops.codegen import module_codegen as _cg
        return _cg._emit_metadata(self, model_alias)

    def emit_cpp_so_source(self, name=None, hoist_reciprocals=False):
        """Thin wrapper: delegates to pops.codegen.compile.emit_cpp_so_source."""
        return _cg_emit_cpp_so_source(self, name=name, hoist_reciprocals=hoist_reciprocals)

    def compile_so(self, so_path, include=None, name=None, cxx=None, std="c++20",
                   hoist_reciprocals=False):
        """Thin wrapper: delegates to pops.codegen.compile.compile_so."""
        return _cg_compile_so(self, so_path, include=include, name=name, cxx=cxx, std=std,
                              hoist_reciprocals=hoist_reciprocals)

    def emit_cpp_aot_source(self, name=None, hoist_reciprocals=False):
        """Thin wrapper: delegates to pops.codegen.compile.emit_cpp_aot_source."""
        return _cg_emit_cpp_aot_source(self, name=name, hoist_reciprocals=hoist_reciprocals)

    def compile_aot(self, so_path, include=None, name=None, cxx=None, std="c++20",
                    hoist_reciprocals=False):
        """Thin wrapper: delegates to pops.codegen.compile.compile_aot."""
        return _cg_compile_aot(self, so_path, include=include, name=name, cxx=cxx, std=std,
                               hoist_reciprocals=hoist_reciprocals)

    def emit_cpp_native_loader(self, name=None, target="system", hoist_reciprocals=False):
        """Thin wrapper: delegates to pops.codegen.compile.emit_cpp_native_loader."""
        return _cg_emit_cpp_native_loader(self, name=name, target=target,
                                          hoist_reciprocals=hoist_reciprocals)

    def compile_native(self, so_path, include=None, name=None, cxx=None, std="c++23", target="system",
                       hoist_reciprocals=False):
        """Thin wrapper: delegates to pops.codegen.compile.compile_native."""
        return _cg_compile_native(self, so_path, include=include, name=name, cxx=cxx, std=std,
                                  target=target, hoist_reciprocals=hoist_reciprocals)

    def compile_or_jit(self, so_path, include=None, mode="jit", name=None, cxx=None, std="c++20",
                       target="system", hoist_reciprocals=False):
        """Thin wrapper: delegates to pops.codegen.compile.compile_or_jit."""
        return _cg_compile_or_jit(self, so_path, include=include, mode=mode, name=name, cxx=cxx,
                                  std=std, target=target, hoist_reciprocals=hoist_reciprocals)

    # --- production facade: a single entry point per INTENTION (backend) -----------------
    # Routes the compilation backend by INTENTION rather than by implementation detail. Each
    # entry designates one of the existing engines (compile_so / compile_aot) AND the System adder to use
    # at runtime -- coupled here so that a caller does not wire an AOT .so onto add_dynamic_block (or
    # vice versa), which would load but with an inconsistent ABI/numerics.
    #   "prototype"  -> compile_so  (JIT, IModel, virtual dispatch, host first-order Rusanov; fast
    #                   iteration, to be wired via System.add_dynamic_block);
    #   "aot"        -> compile_aot (AOT, host-marshaled PRODUCTION path: assemble_rhs<Limiter,
    #                   Flux>, HLLC/Roe, second order, SSPRK2/IMEX on a LOCAL grid of the .so; numerics
    #                   identical to native but marshaled arrays, via add_compiled_block);
    #   "production" -> compile_native (NATIVE LOADER): the .so inlines add_compiled_model<ProdModel>, which
    #                   installs the generated model as a NATIVE System block (closures over the REAL
    #                   grid_context). The block runs ZERO-COPY the SAME path as add_block (no
    #                   marshaling); device-clean by construction (named functors from block_builder).
    #                   To be wired via System.add_native_block (ABI key verified). This is the path
    #                   prepared for a real production backend (Kokkos/CUDA codegen = later PR).
    # Single source of truth now lives in pops.codegen.compile._BACKENDS; class attribute kept
    # for backward-compat references via HyperbolicModel._BACKENDS or self._BACKENDS.
    _BACKENDS = _BACKENDS  # noqa: F821 -- re-bound from the module-level import above

    def _model_hash(self, params=None):
        """Stable hash of the model; delegates to pops.codegen.compile.model_hash."""
        return _cg_model_hash(self, params=params)

    def _check_require_metadata(self, require_metadata, backend):
        """require_metadata guard rails (pure-Python, deterministic on the model + backend). Factored out
        to be called BEFORE the cache (in HyperbolicModel AND Model): a cache HIT must never
        mask a metadata requirement. Without require_metadata, no-op."""
        if not require_metadata:
            return
        # backend "prototype" (add_dynamic_block, VIRTUAL dispatch, host first-order Rusanov): NOT a
        # device-clean production path -> requesting metadata on it is inconsistent (clear error).
        if backend == "prototype":
            raise ValueError(
                "compile: backend 'prototype' (JIT, host virtual dispatch) incompatible with "
                "require_metadata=True; use backend='aot' or 'production' for the "
                "device-clean path with guaranteed metadata")
        missing = []
        roles = roles_for(self.cons_names, self.cons_roles)
        if all(r == "Custom" for r in roles):
            missing.append("physical roles (conservative_vars(..., roles=[...]) or canonical names)")
        if self.gamma is None:
            missing.append("gamma (set_gamma(...))")
        if missing:
            raise ValueError(
                "compile(require_metadata=True): model '%s' does not provide %s; the .so "
                "would fall back to the System fallback (roles 'custom' / gamma 1.4)"
                % (self.name, " nor ".join(missing)))

    def compile(self, so_path=None, include=None, backend="auto", name=None, cxx=None, std=None,
                require_metadata=False, target="system", hoist_reciprocals=False):
        """Thin wrapper: delegates to pops.codegen.compile.compile_model."""
        return _cg_compile_model(self, so_path=so_path, include=include, backend=backend,
                                 name=name, cxx=cxx, std=std,
                                 require_metadata=require_metadata, target=target,
                                 hoist_reciprocals=hoist_reciprocals)

    @classmethod
    def adder_for(cls, backend):
        """Name of the System method to use to wire the .so produced by compile(backend=...):
        'add_dynamic_block' (prototype/JIT), 'add_compiled_block' (aot) or 'add_native_block'
        (production/native). Delegates to pops.codegen.compile.adder_for."""
        return _cg_adder_for(backend)

    def emit_cpp_elliptic(self, name=None, namespace="pops_generated", cse=True,
                          hoist_reciprocals=False):
        """Generates a composable elliptic RIGHT-HAND SIDE BRICK from self._elliptic.

        The produced struct exposes rhs(U) -> Real (charge density, background, gravity...), same shape as
        the manual bricks (ChargeDensity, BackgroundDensity in pops/model/bricks.hpp): it enters
        as the Elliptic parameter of a CompositeModel. Inlined constants, cons/primitives -> locals,
        cse=True factors out common sub-expressions. ValueError if set_elliptic_rhs(...) is missing."""
        from pops.codegen import module_codegen as _cg
        return _cg.emit_cpp_elliptic(self, name=name, namespace=namespace, cse=cse,
                                     hoist_reciprocals=hoist_reciprocals)

    def emit_cpp_elliptic_field(self, field, struct_name, namespace="pops_generated",
                                hoist_reciprocals=False, cse=True):
        """Generates a SELF-CONTAINED elliptic RHS brick for the NAMED field @p field (ADC-428).

        Unlike emit_cpp_elliptic (which emits only ``rhs(U)``, consumed by CompositeModel), this brick
        is shaped like a minimal Model so the runtime can pair it with pops::make_poisson_rhs directly:
        it declares ``n_vars`` + ``State`` (so load_state<Brick> reads the conservative state) and
        exposes ``elliptic_rhs(State)`` (what detail::PoissonRhs<Brick> calls per cell). The native
        loader builds one std::function per named field via make_poisson_rhs(Brick{}) and attaches it to
        the block (System::set_block_elliptic_field). The RHS reads ONLY the conservative state (+
        primitives), never the aux (enforced at declaration). Reuses _codegen_exprs / _prim_block so the
        formula lowers IDENTICALLY to the default elliptic brick."""
        from pops.codegen import module_codegen as _cg
        return _cg.emit_cpp_elliptic_field(self, field, struct_name, namespace=namespace,
                                           hoist_reciprocals=hoist_reciprocals, cse=cse)


# === Phase A: pure-Python user facade =================================
# The STABLE surface the user writes (dsl.Model / Param / CompiledModel). Pure sugar:
# no new numerics, no engine change. dsl.Model COMPOSES a private HyperbolicModel
# (_m) and delegates each call to an existing method; Param is a NAMED constant that inlines
# at codegen; CompiledModel packages the .so + the metadata already known on the Python side (no
# re-reading of the .so). cf. docs/DSL_MODEL_DESIGN.md (Phase A).

# _BACKEND_CAPS and _BACKENDS: single source of truth is pops.codegen.compile.
# Re-imported at the top of this file; available as pops.dsl._BACKEND_CAPS etc.


class Param:
    """NAMED parameter of a DSL model, usable like an Expr in formulas.

    Mode (a), constant fixed at compilation: `kind="const"` (default). The codegen INLINES every
    constant (Const.to_cpp -> repr(value)), so the param inlines as Const(value) at codegen (value
    written HARD-CODED in the .so) while keeping its IDENTITY (name/value/kind) for introspection
    (m.params), diagnostics and reproducibility. UNCHANGED by P7-b: bit-identical to the history.

    Mode (b), RUNTIME parameter (modifiable WITHOUT recompiling): `kind="runtime"` (P7-b). At codegen, the
    param emits `params.get(<index>)` (read of an pops::RuntimeParams member of the brick) instead
    of a constant; its value is carried at runtime by the AOT .so ABI and can be CHANGED
    (block.set_param / System.set_block_params) without recompiling. The value passed at declaration serves
    as the DEFAULT (without a set call, the block behaves as with a const param of that value).
    SUPPORTED by the "aot" backend (add_compiled_block). The "prototype" (JIT) and "production"
    (native) backends compile a runtime param as its declaration value (fixed): a set_param there has no
    effect, the API reports it (cf. CompiledModel.runtime_param_names / System.set_block_params).

    Param DOES NOT INHERIT from Expr (see NB below): it EXPOSES the same tree hooks
    (`eval`/`to_cpp`/`deps`) and operators by DELEGATING to an internal NODE (Const for 'const',
    RuntimeParamRef for 'runtime'), so `g * (E - ...)` builds the expected tree directly. The
    value is not an environment variable -> no dependency to check in check()."""

    # NB: Param DOES NOT INHERIT from Expr to avoid embedding its state (name/kind) in the CSE
    # structural key; it EXPOSES the tree hooks instead by delegating to an internal node.
    def __init__(self, name, value, kind="const"):
        if kind not in ("const", "runtime"):
            raise ValueError("Param: kind 'const' | 'runtime' (got %r)" % (kind,))
        self.name = name
        self.value = float(value)
        self.kind = kind
        if kind == "runtime":
            # SHARED RUNTIME node: all occurrences of the param in formulas point to this same
            # object, so setting its .index at compilation (Model._assign_runtime_indices) is enough to
            # route all its reads to params.get(<index>). index=-1 while not assigned.
            self._node = RuntimeParamRef(self.name, self.value)
        else:
            self._node = Const(self.value)  # inlines at codegen: value written HARD-CODED in the .so

    # --- tree hooks (delegated to the internal node): Param usable like an Expr ---
    def eval(self, env): return self._node.eval(env)
    def to_cpp(self): return self._node.to_cpp()
    def deps(self): return set()  # neither const nor runtime has a dependency (nothing to check in check())

    # --- operators: Param combines like an Expr (promotion via _wrap of the internal node) ---
    def __add__(self, o): return Add(self._node, _wrap(o))
    def __radd__(self, o): return Add(_wrap(o), self._node)
    def __sub__(self, o): return Sub(self._node, _wrap(o))
    def __rsub__(self, o): return Sub(_wrap(o), self._node)
    def __mul__(self, o): return Mul(self._node, _wrap(o))
    def __rmul__(self, o): return Mul(_wrap(o), self._node)
    def __truediv__(self, o): return Div(self._node, _wrap(o))
    def __rtruediv__(self, o): return Div(_wrap(o), self._node)
    def __neg__(self): return Neg(self._node)
    def __pos__(self): return self._node  # +param = identity (Expr), for +k*ne*ng
    def __pow__(self, o): return Pow(self._node, _wrap(o))

    def __float__(self): return self.value
    def __repr__(self): return "Param(%r, %r, kind=%r)" % (self.name, self.value, self.kind)


def RuntimeParam(name, value):
    """Sugar: a RUNTIME parameter (modifiable without recompiling). Equivalent to Param(name, value,
    kind='runtime'). cf. Param mode (b) and include/pops/runtime/runtime_params.hpp (P7-b)."""
    return Param(name, value, kind="runtime")


# CompiledProblem: re-exported from pops.codegen.loader (imported above)


# _module_to_model: re-exported from pops.codegen.compile (imported above)


# compile_problem: re-exported from pops.codegen.compile (imported above)


# CompiledModel: re-exported from pops.codegen.loader (imported above)


class Model:
    """STABLE facade of a DSL model (Phase A). COMPOSES a private HyperbolicModel (_m, composition and
    NOT inheritance) and delegates each call to an existing method: no new numerics.

        m = pops.dsl.Model("euler")
        rho, rhou, rhov, E = m.conservative_vars("rho", "rho_u", "rho_v", "E")
        g = m.param("gamma", 1.4)                 # NAMED constant, inlined at codegen
        u = m.primitive("u", rhou / rho)
        p = m.primitive("p", (g - 1.0) * (E - 0.5 * rho * (u*u + ...)))
        m.flux(x=[...], y=[...])                   # symbolic DECLARATOR of the physical flux
        m.eval_flux(U, aux, dir)                   # numpy EVALUATOR (debug), DISTINCT name
        m.primitive_vars(rho=rho, u=u, v=v, p=p)   # ordered Prim layout (kwargs order)
        compiled = m.compile(so_path, include, backend="aot")  # -> CompiledModel

    cf. docs/DSL_MODEL_DESIGN.md sections 1-3."""

    def __init__(self, name):
        self._m = HyperbolicModel(name)
        self.params = {}   # name -> Param (introspection / reproducibility)

    @property
    def name(self): return self._m.name

    # --- variable declaration (direct delegation to HyperbolicModel) ---
    def conservative_vars(self, *names, roles=None):
        """Declares the conservative variables. @p roles: same convention as HyperbolicModel."""
        return self._m.conservative_vars(*names, roles=roles)

    def primitive(self, name, expr):
        """Defines a primitive by its formula (as a function of the cons / preceding primitives)."""
        return self._m.primitive(name, expr)

    def primitive_vars(self, *vars, roles=None, **named):
        """Declares the primitives AND the ORDERED layout of Prim. Two forms:

        - KWARGS (target style): `primitive_vars(rho=expr, u=expr, v=expr, p=expr)`: each kwarg
          DEFINES a primitive (m.primitive(name, expr)) AND fixes the layout of Prim in the
          insertion order of the kwargs (Python 3.7+: order guaranteed). @p roles (list) optional.
        - POSITIONAL: `primitive_vars(rho, u, v, p, roles=...)`: names/Var already defined, fixes
          only the layout (delegates to set_primitive_state, like HyperbolicModel).

        The two forms are exclusive (mixing named kwargs and positional raises)."""
        if named and vars:
            raise ValueError("primitive_vars: mixing positional form and named kwargs "
                             "(choose one; kwargs define AND order the primitives)")
        if named:
            # kwargs: define each primitive, then fix the layout in insertion order.
            # A primitive is NOT (re)defined if the kwarg is the Var of the SAME name -- otherwise the codegen
            # would emit `const Real x = x;` (auto-init -> NaN). Two cases of self-reference, both
            # left to JOIN the layout without redefinition (target style primitive_vars(rho=rho, u=u, ...)):
            #  - name ALREADY CONSERVATIVE (e.g. rho=rho: the density, primitive == conservative);
            #  - PRIMITIVE Var ALREADY DEFINED of the same name (e.g. u=u when u comes from m.primitive('u', ...)).
            # Otherwise (kwarg = expression, e.g. p=cs2*rho or u=mx/rho) the primitive is defined normally.
            ordered = list(named.keys())
            for nm in ordered:
                val = named[nm]
                self_ref = isinstance(val, Var) and getattr(val, "name", None) == nm
                if nm in self._m.cons_names or self_ref:
                    continue
                self._m.primitive(nm, val)
            self._m.set_primitive_state(*ordered, roles=roles)
            return tuple(Var(nm, "prim") for nm in ordered)
        # positional form: fixes the layout from already-defined names/Var.
        self._m.set_primitive_state(*vars, roles=roles)
        return None

    def aux(self, name):
        """CANONICAL auxiliary field (must be a key of AUX_CANONICAL: phi/grad_x/grad_y/B_z/T_e)."""
        return self._m.aux(name)

    def aux_field(self, name):
        """NAMED auxiliary field (ADC-70 phase 1) provided by a block via System.set_aux_field(block, name,
        array). name is ARBITRARY (identifier); the k-th call reserves the aux channel component
        AUX_NAMED_BASE + k (read in C++ via aux.extra_field(k)). At most AUX_NAMED_MAX per model.
        Returns a Var usable in flux / source / eigenvalues. Delegates to HyperbolicModel.aux_field."""
        return self._m.aux_field(name)

    def conservative_from(self, exprs):
        """Inverse prim -> cons (the DSL cannot invert symbolically)."""
        self._m.set_conservative_from(exprs)

    # --- flux: symbolic DECLARATOR vs numpy EVALUATOR (DISTINCT names, settled decision) ---
    def flux(self, x, y):
        """Symbolic DECLARATOR of the physical flux (delegates to set_flux). x/y: lists of Expr, one
        per conservative component. DO NOT confuse with the numpy evaluator eval_flux."""
        self._m.set_flux(x, y)

    def flux_term(self, name, x, y):
        """NAMED physical flux F_name(U, primitives, aux, params): exactly n_cons expressions per
        direction (delegates to HyperbolicModel.flux_term). Opt-in -- emitted only when a compiled time
        Program selects it (ctx.rhs(..., fluxes=[name, ...])), never folded into the historical -div F.
        name='default' is the backward-compatible alias of m.flux(...): ctx.rhs(fluxes=['default']) is
        byte-identical to the historical flux-only RHS. A Program requesting several named fluxes
        assembles -div of their SUM."""
        self._m.flux_term(name, x, y)

    def eval_flux(self, U, aux, dir):
        """numpy EVALUATOR of the physical flux (debug / host proto; delegates to HyperbolicModel.flux).
        U: numpy (n_vars, ...); aux: dict name -> array; dir: 0=x, 1=y."""
        return self._m.flux(U, aux, dir)

    def eval_source(self, U, aux):
        """numpy EVALUATOR of the source term (debug / host proto; delegates to
        HyperbolicModel.source_value). U: numpy (n_vars, ...); aux: dict name -> array. Returns
        zeros when no source was declared. Lets a host test check the emitted source (e.g. a BGK
        collision) against an oracle without compiling."""
        return self._m.source_value(U, aux)

    def eigenvalues(self, x, y):
        """Eigenvalues (characteristic speeds) per direction (delegates to set_eigenvalues)."""
        self._m.set_eigenvalues(x, y)

    def wave_speeds(self, x, y):
        """Explicit SIGNED wave speeds per direction: x = (smin_x, smax_x), y = (smin_y,
        smax_y). Emits ``wave_speeds(U, aux, dir, smin, smax)`` on the brick WITHOUT requiring a
        primitive 'p': riemann='hll' becomes available for a model without pressure (moment
        system, isothermal...). Takes priority over the historical path (eigenvalues + 'p'); if
        eigenvalues is not declared, max_wave_speed (Rusanov / CFL) derives from ``max(|smin|, |smax|)``.
        Delegates to set_wave_speeds; cf. HyperbolicModel.set_wave_speeds."""
        self._m.set_wave_speeds(x, y)

    def wave_speeds_from_jacobian(self, x=None, y=None, eig="numeric", blocks=None):
        """EXACT signed wave speeds from the eigenvalues of the flux jacobian (delegates to
        set_wave_speeds_from_jacobian, see its full contract): x/y = dF/dU as Expr (None =
        AUTODIFF of the declared flux via flux_jacobian); eig = 'numeric' | 'fd' (finite differences
        of the compiled flux, debug); blocks = lists of indices of the diagonal sub-blocks (None = full
        block, the only unconditionally correct mode -- the blocks ASSERT a
        block-triangular structure)."""
        self._m.set_wave_speeds_from_jacobian(x=x, y=y, eig=eig, blocks=blocks)

    def eval_wave_speeds(self, U, aux, dir):
        """numpy EVALUATOR of the emitted signed speeds (smin, smax) (delegates to
        HyperbolicModel.wave_speeds_value): explicit pair, jacobian (numpy eig per blocks) or
        min/max of the eigenvalues."""
        return self._m.wave_speeds_value(U, aux, dir)

    def stability_speed(self, expr):
        """STABILITY speed lambda* driving the block CFL (OPTIONAL; delegates to
        HyperbolicModel.stability_speed). Fallback without a call: max(abs(eigenvalues)), strictly
        the historical behavior. Compiled like flux/source (production GPU/MPI, no per-cell callback)."""
        self._m.stability_speed(expr)

    def stability_dt(self, expr_dt):
        """Direct ADMISSIBLE step dt(U, aux) local bound of the step (OPTIONAL; delegates to
        HyperbolicModel.stability_dt). The cfl is not applied to this bound. Fallback without a call:
        no additional bound (historical step policy)."""
        self._m.stability_dt(expr_dt)

    def source(self, s):
        """Source term S(U, aux), one expression per component (optional; delegates to set_source)."""
        self._m.set_source(s)

    def source_term(self, name, exprs):
        """NAMED local source S_name(U, primitives, aux, params): exactly n_cons expressions
        (delegates to HyperbolicModel.source_term). Opt-in -- emitted only when a compiled time
        Program requests it (ctx.rhs(..., sources=[name]) / ctx.source(name)), never summed
        implicitly. name='default' is the backward-compatible alias of m.source([...])."""
        self._m.source_term(name, exprs)

    def linear_source(self, name, matrix):
        """NAMED local linear operator L_name(aux, params) U, an n_cons x n_cons matrix whose
        coefficients are independent of U / primitives (delegates to HyperbolicModel.linear_source).
        Used explicitly by a Program (ctx.linear_source / ctx.apply / ctx.solve_local_linear);
        never folded into m.source or ctx.rhs."""
        self._m.linear_source(name, matrix)

    def rate_operator(self, name, *, flux=True, sources=("default",), fluxes=None):
        """NAMED composite rate operator R_name = -div F + sum(sources) (Spec 2, operator-first):
        a Program-side alias for ctx.rhs(flux=, sources=, fluxes=) so a model-free Program can call
        P.call(name, U[, fields]) instead of P.rhs(...). Delegates to HyperbolicModel.rate_operator."""
        self._m.rate_operator(name, flux=flux, sources=sources, fluxes=fluxes)

    def source_frequency(self, expr_mu):
        """Local frequency mu(U, aux) [1/s] of the source -- the 'source' step bound from the meeting
        (dt <= cfl*substeps/(stride*max mu), without a space step). Emitted on the generated SOURCE
        brick (frequency(U, aux)), forwarded by CompositeModel, aggregated by System/AmrSystem
        step_cfl. REQUIRES m.source([...]). Delegates to HyperbolicModel.source_frequency."""
        self._m.source_frequency(expr_mu)

    def source_jacobian(self, rows):
        """ANALYTIC Jacobian dS/dU of the source (rows[r][c] = dS_r/dU_c, an n_vars x n_vars
        matrix of expressions): the implicit Newton (IMEX/SourceImplicitBE) uses it instead of
        finite differences. REQUIRES m.source. Delegates to HyperbolicModel.source_jacobian."""
        self._m.source_jacobian(rows)

    def projection(self, exprs):
        """PROJECTION PONCTUELLE post-pas U <- P(U, aux) (ADC-177, OPTIONNEL) : une expression par
        composante conservative, appliquee par le System a la FIN de chaque macro-pas ENTIER (jamais
        par etage RK) sur les cellules valides. CONTRAT : idempotente et ponctuelle ; clamps en
        max/min via abs_/sign. Backends 'aot'/'production' (System) ; 'prototype' et AMR rejetes.
        Delegue a HyperbolicModel.projection (cf. son contrat complet)."""
        self._m.projection(exprs)

    def projection_value(self, U, aux=None):
        """EVALUATEUR numpy de la projection emise (reference de test ; delegue a
        HyperbolicModel.projection_value)."""
        return self._m.projection_value(U, aux)

    def implicit_source(self, jacobian=None):
        """GROUPED declaration of the local implicit (wave 3 audit, sugar): the RESIDUAL is already
        implied by m.source (backward-Euler: F = W - U^n - dt*S(W)); @p jacobian (optional) =
        analytic dS/dU matrix (cf. source_jacobian). Without jacobian: finite differences."""
        if jacobian is not None:
            self._m.source_jacobian(jacobian)

    def enable_hllc(self):
        """Emits the HLLC capability (contact_speed + hllc_star_state generated from the ROLES +
        primitive 'p'): riemann='hllc' becomes available for this model EVEN outside 4-variable
        Euler. Delegates to HyperbolicModel.enable_hllc."""
        self._m.enable_hllc()

    def set_riemann_hooks(self, **forms):
        """Record ARBITRARY-formula overrides of the role-derived Riemann hooks (ADC-456): e.g.
        ``pressure=<Expr>`` codegen's that formula as the ``pressure(U)`` hook body. Descriptors /
        ``None`` keep the role-derived default. Delegates to HyperbolicModel.set_riemann_hooks."""
        self._m.set_riemann_hooks(**forms)
        return self

    def enable_roe(self):
        """Emits the ROE capability (roe_dissipation = ``|A_roe| dU`` generated from the ROLES +
        primitive 'p'): riemann='roe' becomes available for this model EVEN outside 4-variable
        Euler (without Energy: c = sqrt(p/rho) averaged Roe-style; components outside the fluid
        roles = passive scalars on the entropy wave). Delegates to HyperbolicModel.enable_roe."""
        self._m.enable_roe()

    def roe_dissipation(self, x, y):
        """Roe dissipation PROVIDED by the user (outside the fluid roles): n_vars expressions per
        direction (x=, y=), written with m.left(...)/m.right(...) (or dsl.left/right) of the two states,
        emitted as the C++ hook roe_dissipation(UL, AL, UR, AR, dir). During the 'provided' mode of enable_roe
        (a single provider: supplying both together raises). The helper m.flux_jacobian assists the writing.
        Delegates to HyperbolicModel.roe_dissipation (cf. its doc)."""
        self._m.roe_dissipation(x, y)

    def flux_jacobian(self, dir):
        """Flux Jacobian A = dF_dir/dU (an n_vars x n_vars matrix of Expr, A[i][j]=d(F_i)/d(U_j)),
        auto-derived from the fluxes declared via dsl.diff (expanded primitives). HELPER for building
        m.roe_dissipation, emits nothing. @p dir: 0/'x' or 1/'y'. Delegates to HyperbolicModel."""
        return self._m.flux_jacobian(dir)

    def roe_from_jacobian(self):
        """Generic moment Roe: emits roe_dissipation = ``|A| (UR-UL)`` with A the flux Jacobian at
        Uavg = 1/2(UL+UR) and |A| via pops::roe_abs_apply (matrix-sign), spectral-radius Rusanov
        fallback on a complex/singular spectrum. Roles-free (no Density/Momentum, no 'p'): makes
        riemann='roe' available for a moment hierarchy. Exclusive with enable_roe / roe_dissipation.
        Delegates to HyperbolicModel.roe_from_jacobian (cf. its doc)."""
        self._m.roe_from_jacobian()

    def left(self, expr):
        """Marks @p expr as evaluated on the LEFT state UL (m.roe_dissipation). Sugar for dsl.left."""
        return left(expr)

    def right(self, expr):
        """Marks @p expr as evaluated on the RIGHT state UR (m.roe_dissipation). Sugar for dsl.right."""
        return right(expr)

    def elliptic_rhs(self, e):
        """Contribution to the elliptic right-hand side (Poisson coupling; delegates to set_elliptic_rhs)."""
        self._m.set_elliptic_rhs(e)

    def elliptic_field(self, name, rhs, operator="poisson", aux=None):
        """NAMED elliptic field: an elliptic solve operator(field) = rhs(U) populating the named @p aux
        fields (default ['phi', 'grad_x', 'grad_y']); delegates to HyperbolicModel.elliptic_field. The
        IR + validation + hash land; the multi-field RUNTIME (a second elliptic operator + aux channel)
        is DEFERRED -- ctx.solve_fields(field=name) raises NotImplementedError on lowering."""
        self._m.elliptic_field(name, rhs, operator=operator, aux=aux)

    def gamma(self, value):
        """Adiabatic index (EOS), carried by POPS_EXPORT_BLOCK_GAMMA (delegates to set_gamma)."""
        self._m.set_gamma(value)

    def param(self, name, value, kind="const"):
        """NAMED parameter usable in the formulas. Mode (a) (`kind="const"`, default): constant
        frozen at compile time, inlined at codegen; stored in m.params (introspection /
        reproducibility). Mode (b) (`kind="runtime"`, P7-b): SUPPORTED on the "aot" backend -- the param
        emits `params.get(<index>)` (member of pops::RuntimeParams) and its value can be CHANGED at runtime
        via System.set_block_params(name, values) WITHOUT recompiling (the declaration value serves as the
        default); cf. CompiledModel.runtime_param_names. The "prototype"/"production" backends freeze a
        runtime param at its declaration value.

        gamma CASE: if name == "gamma", ALSO calls set_gamma(value) so that the ABI metadata
        stays consistent (otherwise the System falls back to 1.4)."""
        p = Param(name, value, kind=kind)  # 'runtime' -> RuntimeParamRef (P7-b), 'const' -> inline
        self.params[name] = p
        if name == "gamma":
            self._m.set_gamma(p.value)
        return p

    def check(self):
        """Checks the dependencies (referenced variables are declared). Raises ValueError otherwise."""
        return self._m.check()

    def check_model(self, samples=None, n_samples=64, seed=0, aux=None, rtol=1e-8, atol=1e-10,
                    raise_on_error=True, jac_rtol=1e-3, jac_atol=1e-9):
        """Generic NUMERICAL verification of the model (finite flux/source/elliptic, real and finite
        eigenvalues, wave_speeds/max_wave_speed consistency, non-circular bounding of the spectrum by
        the dense Jacobian, cons<->prim round-trip, positivity of Density/'p') on sample states.
        Delegates to HyperbolicModel.check_model (cf. its doc). To be called BEFORE compile(); the
        runtime counterpart of an installed block is System.check_model."""
        return self._m.check_model(samples=samples, n_samples=n_samples, seed=seed, aux=aux,
                                   rtol=rtol, atol=atol, raise_on_error=raise_on_error,
                                   jac_rtol=jac_rtol, jac_atol=jac_atol)

    # --- introspection (read-only, delegated to the backing model) ---
    @property
    def cons_names(self): return self._m.cons_names

    @property
    def prim_state(self): return self._m.prim_state

    @property
    def n_vars(self): return self._m.n_vars

    def state_space(self, name="U"):
        """Typed pops.model.StateSpace view of the conservative state (Spec 2).
        Delegates to HyperbolicModel.state_space."""
        return self._m.state_space(name)

    def field_space(self, name="fields"):
        """Typed pops.model.FieldSpace view of the auxiliary surface (Spec 2).
        Delegates to HyperbolicModel.field_space."""
        return self._m.field_space(name)

    def operator_registry(self, state_name="U"):
        """Typed pops.model.OperatorRegistry derived from this model (Spec 2): the PDE
        shortcuts (source_term / linear_source / elliptic_field / flux) lower into
        typed operators. Pure view -- no hash or codegen impact. Delegates to
        HyperbolicModel.operator_registry."""
        return self._m.operator_registry(state_name)

    @property
    def module(self):
        """The pops.model.Module view of this PDE model (Spec 2, operator-first): its typed
        StateSpace / FieldSpace and the OperatorRegistry that source_term / linear_source /
        elliptic_field / flux / rate_operator populate. dsl.Model is the PDE convenience
        facade; the Module is the model-free view a generic Program binds to (P.bind_operators).
        The Module carries no numerics; codegen still reads this Model via compile_problem."""
        mod = _model.Module(self.name)
        st = self._m.state_space()
        mod.state_space(st.name, st.components, roles=st.roles, layout=st.layout,
                        storage=st.storage)
        fs = self._m.field_space()
        mod.field_space(fs.name, fs.components, layout=fs.layout)
        mod.adopt_registry(self._m.operator_registry())
        return mod

    # --- operator introspection (Spec 2, S2-5) ---
    def list_operators(self):
        """Names of the typed operators this model exposes (registration order)."""
        return self._m.operator_registry().names()

    def list_state_spaces(self):
        """Names of the model's state spaces (one, the conservative state)."""
        return [self._m.state_space().name]

    def list_field_spaces(self):
        """Names of the model's field spaces (one, the auxiliary surface)."""
        return [self._m.field_space().name]

    def operator_signature(self, name):
        """The pops.model.Signature of operator ``name``."""
        return self._m.operator_registry().get(name).signature

    def operator_requirements(self, name):
        """The requirements dict of operator ``name``."""
        return dict(self._m.operator_registry().get(name).requirements)

    def operator_capabilities(self, name):
        """The capabilities dict of operator ``name``."""
        return dict(self._m.operator_registry().get(name).capabilities)

    def _model_hash(self):
        """Stable hash of the model: formulas (flux/eig/source/elliptic/primitives/cons_from) + roles +
        n_aux + NAMED params (m.params). Used to identify/reuse an already-compiled .so (cache key)
        and to trace the run. Delegates to the shared computation HyperbolicModel._model_hash, passing it
        the Param of the facade (otherwise two models differing only by a param would have the same hash)."""
        return self._m._model_hash(params=self.params)

    def compile(self, so_path=None, include=None, backend="auto", target="system", name=None,
                cxx=None, std=None, require_metadata=False, hoist_reciprocals=False):
        """Compiles the model into a CompiledModel (Phase A). Delegates the GENERATION + compilation to
        HyperbolicModel.compile (engines unchanged: compile_so / compile_aot / compile_native), then
        packages the .so with the already-known metadata (no re-reading of the .so).

        - ``backend``: "prototype" | "aot" | "production" (cf. HyperbolicModel.compile).
        - ``target``: "system" (default) | "amr_system" (DSL Phase D). "amr_system" requires
          backend="production" (the native loader inlines add_compiled_model(AmrSystem&), the only
          .so AMR path; cf. compile_or_jit) -> to be wired via AmrSystem.add_equation. Another backend
          with target="amr_system" raises ValueError (no AMR path outside native).

        NO ``device`` argument: the GPU/MPI/AMR capabilities are checked at wiring time
        (add_equation) / at execution, not frozen at compile time (DSL_MODEL_DESIGN.md point 7).

        ERGONOMICS (does not change the numerics):

        - ``include`` None -> auto-detected (pops_include()); passing include= remains possible;
        - ``so_path`` None -> .so in an out-of-source cache (pops_cache_dir()), file name keyed on
          model_hash (PARAMS INCLUDED) + abi_key (+ backend/target/name). Cache HIT (.so already present)
          -> reuse without recompilation; cache MISS (model/param/toolchain change) ->
          recompilation + storage. Passing so_path= forces that path and recompiles (backward-compat).

        Returns a CompiledModel carrying so_path, backend, target, adder, names/roles/gamma/n_aux/params,
        caps, abi_key, model_hash, cxx, std."""
        import os
        # 'auto' DEFAULT (ADC-63): production if toolchain parity is established, aot otherwise. The reason
        # is recorded on the CompiledModel (backend_auto_reason) -- never a silent choice.
        auto_reason = None
        if backend == "auto":
            backend, auto_reason = resolve_auto_backend(include)
        if backend not in HyperbolicModel._BACKENDS:
            raise ValueError("compile: unknown backend %r (expected %s + 'auto')"
                             % (backend, sorted(HyperbolicModel._BACKENDS)))
        if target not in ("system", "amr_system"):
            raise ValueError("compile: target 'system' | 'amr_system' (got %r)" % (target,))

        m = self._m
        # effective std: same per-backend default as HyperbolicModel.compile. The native one follows the
        # loader's standard (c++20 under Kokkos, c++23 otherwise, cf. loader_cxx_std); the others stay c++20.
        mode = HyperbolicModel._BACKENDS[backend][0]
        if target == "amr_system" and mode != "native":
            raise ValueError("compile: target='amr_system' only exists for backend='production' "
                             "(native AMR path); got backend=%r" % (backend,))
        eff_std = std if std is not None else (loader_cxx_std() if mode == "native" else "c++20")
        # native AND aot (mode "compile") compile the pops headers -> real Kokkos (compiler +
        # kokkos feature-key) so that the cache key MATCHES the produced .so (cf. compile_aot).
        kokkos_like = mode in ("native", "compile")
        eff_cxx = _native_kokkos_compiler(cxx) if kokkos_like else _default_cxx(cxx)
        if include is None:  # ergonomics: auto-detection of the pops headers folder
            include = pops_include()

        # Metadata guards BEFORE the cache (a HIT must not mask them; cf.
        # HyperbolicModel._check_require_metadata).
        m._check_require_metadata(require_metadata, backend)

        # PARAMS-INCLUDED model_hash (the one carried by the CompiledModel) AND the ABI key: both also
        # serve as cache keys, so we compute them here to reuse them (key/metadata consistency).
        model_hash = self._model_hash()
        abi_key = _abi_key_python(include, eff_cxx, eff_std)

        # OUT-OF-SOURCE cache when so_path is omitted: we RESOLVE the keyed path here (with the
        # params-included hash) and pass it explicitly to the engine -- the cache of HyperbolicModel.compile
        # would otherwise use the hash WITHOUT params (the Model facade adds the Param). HIT -> we skip the
        # compilation. Explicit so_path -> forced path, always recompiles (strict backward-compat).
        cache_hit = False
        if so_path is None:
            # kokkos feature-key in the key (cf. compile_native): a SERIAL .so is not reused
            # on a Kokkos module. MUST match the engine's key, otherwise repeated recompilations.
            cache_backend = (backend + ";" + _native_feature_key()) if kokkos_like else backend
            if hoist_reciprocals:  # distinct codegen -> distinct key (cf. HyperbolicModel.compile)
                cache_backend += ";hoist"
            so_path = _cache_so_path(model_hash, abi_key, cache_backend, target, name)
            cache_hit = os.path.exists(so_path)

        if cache_hit:
            out_path = so_path  # .so already compiled for this key: no recompilation
        else:
            # Compilation (engines unchanged, require_metadata/backend/target guards of
            # HyperbolicModel.compile: the loader emits pops_install_native_amr for target="amr_system").
            out_path = m.compile(so_path, include, backend=backend, name=name, cxx=cxx, std=std,
                                 require_metadata=require_metadata, target=target,
                                 hoist_reciprocals=hoist_reciprocals)
        # The keyed path (cache HIT) or the path retained by the engine carries the written backend: we
        # record it so a cross-backend reuse of the SAME path in this process is detected.
        _record_so_backend(out_path, backend)

        adder = HyperbolicModel.adder_for(backend)
        cons_roles = roles_for(m.cons_names, m.cons_roles)
        cm = CompiledModel(
            so_path=out_path, backend=backend, adder=adder, target=target,
            cons_names=m.cons_names, cons_roles=cons_roles, prim_names=m.prim_state,
            n_vars=m.n_vars, gamma=m.gamma, n_aux=aux_total_n_aux(m.aux_names, m.aux_extra_names),
            params=self.params, caps=_BACKEND_CAPS[backend],
            abi_key=abi_key, model_hash=model_hash,
            cxx=eff_cxx, std=eff_std, hllc=m._hllc,
            roe=(m._roe or getattr(m, '_roe_rows', None) is not None
                 or getattr(m, '_roe_jacobian', None) is not None),
            aux_extra_names=m.aux_extra_names,
            wave_speeds=(m._wave_speeds is not None or m._ws_jacobian is not None
                         or "p" in m.prim_defs))
        # Trace of the 'auto' policy (ADC-63): None if the backend was explicit. Diagnostic,
        # never a silent choice -- cm.backend says what was built, this says WHY.
        cm.backend_auto_reason = auto_reason
        return cm


# === Phase B (prototype): HYBRID composition of a native brick + a DSL brick in ONE model ========
# Until now, mixing native and DSL was done at the SYSTEM level (a native add_block block + a DSL
# add_equation block). One could not mix a native brick and a DSL brick INSIDE A SINGLE
# model. This prototype allows it, in BOTH directions (DSL transport + native source/elliptic, AND
# native transport + DSL source/elliptic).
#
# ARCHITECTURE (option B): the C++ core already COMPOSES heterogeneous brick types effortlessly --
# pops::CompositeModel<Hyperbolic, Source, Elliptic> accepts any type conforming to its slot,
# native (include/pops/physics/) or generated by the DSL, and physics/bricks.hpp is already included by all
# the backends. The mix is therefore generated at the compilation of the final COMPOSITE: a single .so, on the
# SAME path as full DSL models (inlined numerics, identical to a native block). No .so
# per brick nor a partial virtual ABI (that would be option A: host virtual dispatch, without GPU
# inlining -- discarded).
#
# The only subtlety: the backends carry only the model TYPE (default-constructed), so
# the PARAMETERS of a native brick (qom, q, cs2...) must be BAKED into the type. For this we emit
# a small derived struct that fixes the public fields in its constructor (host):
#   namespace pops_generated { struct NatSrc : pops::PotentialForce { NatSrc() { qom = pops::Real(-1.0); } }; }
# It inherits EXACTLY the native numerics (true native path, zero re-derivation) and satisfies the
# slot contract. Without a parameter -> a simple `using` alias.
#
# PROTOTYPE state: "aot" backend (add_compiled_block, self-sufficient .so, production host-marshaled
# path). The hybrid "production" (native zero-copy) and "prototype" (JIT) backends, the
# fine propagation of roles/gamma/n_aux and the amr_system target arrive in the following PRs.


class NativeBrick:
    """Descriptor of a NATIVE brick (include/pops/physics/) for hybrid composition.

    Carries the C++ type of the brick, the PARAMETERS to bake into the type (public field -> value) and,
    for a hyperbolic brick, the variable layout (conservative names, n_vars, primitives,
    gamma). emit(struct_name) returns the C++ text to sew into the composite .so: a derived struct that
    fixes the parameters (or a simple `using` alias if the brick has no parameter).

    - ``kind``: 'hyperbolic' | 'source' | 'elliptic' (target slot).
    - ``fields``: dict {public C++ field name -> value}; ORDER preserved (insertion).
    - ``var_names`` / ``n_vars`` / ``prim_names`` / ``gamma``: layout metadata (hyperbolic
      slot only).
    - ``min_vars``: minimal number of variables that a TEMPLATED brick (source/elliptic) requires;
      e.g. PotentialForce indexes s[1]/s[2] so it requires >= 3 variables. Checked by HybridModel.
    - ``n_aux``: width of the aux channel that the brick READS (>= 3 if it reads B_z/T_e)."""

    def __init__(self, cpp_type, kind, fields=None, var_names=None, n_vars=None, prim_names=None,
                 gamma=None, min_vars=1, n_aux=AUX_BASE_COMPS):
        if kind not in ("hyperbolic", "source", "elliptic"):
            raise ValueError("NativeBrick: kind 'hyperbolic' | 'source' | 'elliptic' (got %r)" % (kind,))
        self.cpp_type = cpp_type
        self.kind = kind
        self.fields = dict(fields or {})
        self.var_names = list(var_names) if var_names else None
        self.n_vars = n_vars
        self.prim_names = list(prim_names) if prim_names else (list(var_names) if var_names else None)
        self.gamma = gamma
        self.min_vars = min_vars
        self.n_aux = n_aux

    def emit(self, struct_name, namespace="pops_generated"):
        """C++ text of the brick sewn into the composite .so. Without a parameter -> `using` alias
        (zero cost); with parameters -> a derived struct that fixes them in its host constructor
        (the values are WRITTEN HARD, like an inlined DSL constant)."""
        if not self.fields:
            return "namespace %s { using %s = %s; }\n" % (namespace, struct_name, self.cpp_type)
        sets = " ".join("%s = pops::Real(%s);" % (k, repr(float(v))) for k, v in self.fields.items())
        return ("namespace %s { struct %s : %s { %s() { %s } }; }\n"
                % (namespace, struct_name, self.cpp_type, struct_name, sets))


class CompiledBrick:
    """Result of <partial DSL brick>.compile(): the C++ of ONE brick (the generated struct) + its
    metadata, ready to be sewn into a hybrid CompositeModel. The MACHINE compilation happens at the
    level of the composite (a single .so); this object carries the brick already GENERATED and frozen."""

    def __init__(self, kind, struct_src, type_name, n_vars=None, n_aux=AUX_BASE_COMPS,
                 cons_names=None, cons_roles=None, prim_names=None, gamma=None, hash_part="",
                 wave_speeds=True):
        self.kind = kind                 # 'hyperbolic' | 'source' | 'elliptic'
        self.struct_src = struct_src     # C++ text of the struct (namespace pops_generated { struct ... })
        self.type_name = type_name       # qualified type to place in CompositeModel<...>
        self.n_vars = n_vars             # layout (hyperbolic) or declared number of variables (src/ell)
        self.n_aux = n_aux
        self.cons_names = list(cons_names) if cons_names else []
        self.cons_roles = list(cons_roles) if cons_roles else []
        self.prim_names = list(prim_names) if prim_names else []
        self.gamma = gamma
        self.hash_part = hash_part       # stable hash slice (formulas) for the composite cache key
        # wave_speeds emitted by the struct (DSL hyperbolic brick: 'p' OR explicit pair); True by
        # default = unknown (native brick): we let the C++ requires-gate decide (historical).
        self.has_wave_speeds = bool(wave_speeds)

    def __repr__(self):
        return "CompiledBrick(kind=%r, type=%r, n_vars=%r)" % (self.kind, self.type_name, self.n_vars)


class CompiledHyperbolicBrick(CompiledBrick):
    """Compiled DSL hyperbolic brick (vars/flux/eigenvalues/conversions)."""
    def __init__(self, **kw): super().__init__("hyperbolic", **kw)


class CompiledSourceBrick(CompiledBrick):
    """Compiled DSL source brick (apply(U, aux))."""
    def __init__(self, **kw): super().__init__("source", **kw)


class CompiledEllipticBrick(CompiledBrick):
    """Compiled DSL elliptic right-hand side brick (rhs(U))."""
    def __init__(self, **kw): super().__init__("elliptic", **kw)


class HyperbolicBrick:
    """PARTIAL hyperbolic DSL brick (variables/flux/eigenvalues/conversions), composable with
    native or DSL bricks for the source and the elliptic. Same surface as dsl.Model but limited
    to the hyperbolic slot. compile() -> CompiledHyperbolicBrick."""

    def __init__(self, name):
        self._m = HyperbolicModel(name)

    @property
    def name(self): return self._m.name

    def conservative_vars(self, *names, roles=None):
        return self._m.conservative_vars(*names, roles=roles)

    def primitive(self, name, expr):
        return self._m.primitive(name, expr)

    def primitive_vars(self, *vars, roles=None):
        """ORDERED layout of Prim (positional form, names/Var already defined)."""
        self._m.set_primitive_state(*vars, roles=roles)

    def aux(self, name): return self._m.aux(name)
    def flux(self, x, y): self._m.set_flux(x, y)
    def eigenvalues(self, x, y): self._m.set_eigenvalues(x, y)

    def wave_speeds(self, x, y):
        """Explicit SIGNED wave speeds (smin, smax) per direction, WITHOUT requiring 'p' --
        same contract as Model.wave_speeds (the brick struct goes through emit_cpp_brick, which
        emits wave_speeds from the pair ; the hybrid CompositeModel forwards it to the HLL gate)."""
        self._m.set_wave_speeds(x, y)

    def conservative_from(self, exprs): self._m.set_conservative_from(exprs)
    def gamma(self, value): self._m.set_gamma(value)
    def check(self): return self._m.check()

    def compile(self):
        """Validate + emit the hyperbolic C++ struct (emit_cpp_brick) -> CompiledHyperbolicBrick."""
        self._m.check()
        struct_name = "Hyp" + self._m.name.capitalize()
        struct_src = self._m.emit_cpp_brick(name=struct_name)
        return CompiledHyperbolicBrick(
            struct_src=struct_src, type_name="pops_generated::" + struct_name,
            n_vars=self._m.n_vars, cons_names=list(self._m.cons_names),
            cons_roles=roles_for(self._m.cons_names, self._m.cons_roles),
            prim_names=list(self._m.prim_state), gamma=self._m.gamma,
            n_aux=aux_n_aux(self._m.aux_names), hash_part=self._m._model_hash(),
            wave_speeds=("p" in self._m.prim_defs or self._m._wave_speeds is not None))


class SourceBrick:
    """PARTIAL DSL brick for a source S(U, aux), composable with a native or DSL transport and
    elliptic. Declares its conservatives (the layout must match the transport) + its aux fields
    + the source formula. compile() -> CompiledSourceBrick."""

    def __init__(self, name):
        self._m = HyperbolicModel(name)

    def conservative_vars(self, *names, roles=None):
        return self._m.conservative_vars(*names, roles=roles)

    def primitive(self, name, expr): return self._m.primitive(name, expr)
    def aux(self, name): return self._m.aux(name)
    def source(self, s): self._m.set_source(s)

    def compile(self):
        """Validate + emit the source C++ struct (emit_cpp_source) -> CompiledSourceBrick."""
        if self._m._source is None:
            raise ValueError("SourceBrick.compile: call source([...]) first")
        struct_name = "Src" + self._m.name.capitalize()
        struct_src = self._m.emit_cpp_source(name=struct_name)
        return CompiledSourceBrick(
            struct_src=struct_src, type_name="pops_generated::" + struct_name,
            n_vars=self._m.n_vars, n_aux=aux_n_aux(self._m.aux_names),
            hash_part=self._m._model_hash())


class EllipticBrick:
    """PARTIAL DSL brick for an elliptic right-hand side rhs(U), composable with a native or DSL
    transport and source. Declares its conservatives (layout) + the right-hand side formula.
    compile() -> CompiledEllipticBrick."""

    def __init__(self, name):
        self._m = HyperbolicModel(name)

    def conservative_vars(self, *names, roles=None):
        return self._m.conservative_vars(*names, roles=roles)

    def primitive(self, name, expr): return self._m.primitive(name, expr)
    def elliptic_rhs(self, e): self._m.set_elliptic_rhs(e)

    def compile(self):
        """Validate + emit the elliptic C++ struct (emit_cpp_elliptic) -> CompiledEllipticBrick."""
        if self._m._elliptic is None:
            raise ValueError("EllipticBrick.compile: call elliptic_rhs(...) first")
        struct_name = "Ell" + self._m.name.capitalize()
        struct_src = self._m.emit_cpp_elliptic(name=struct_name)
        return CompiledEllipticBrick(
            struct_src=struct_src, type_name="pops_generated::" + struct_name,
            n_vars=self._m.n_vars, n_aux=AUX_BASE_COMPS, hash_part=self._m._model_hash())


class HybridModel:
    """Composer of a HYBRID model: three slots (transport, source, elliptic), each provided by a
    NATIVE brick (NativeBrick) or a DSL brick (CompiledBrick). Assembles a mixed pops::CompositeModel<...>
    and compiles it into ONE .so (prototype: backend 'aot'). Returns a CompiledModel pluggable via
    System.add_equation (adder add_compiled_block).

    The transport (hyperbolic) slot FIXES the layout: n_vars, conservative names, primitives, gamma. A
    DSL source/elliptic brick must declare the SAME n_vars ; a templated native brick (source/
    elliptic) only needs to satisfy its min_vars (e.g. PotentialForce requires >= 3 variables)."""

    def __init__(self, transport, source, elliptic, name="hybrid"):
        self.name = name
        hyp = self._norm(transport, "hyperbolic", "NatHyp")
        src = self._norm(source, "source", "NatSrc")
        ell = self._norm(elliptic, "elliptic", "NatEll")

        nv = hyp["n_vars"]
        if nv is None:
            raise ValueError("HybridModel: the transport slot must fix n_vars (hyperbolic brick)")
        for role, slot in (("source", src), ("elliptic", ell)):
            if slot["provider"] == "dsl":
                if slot["n_vars"] != nv:
                    raise ValueError(
                        "HybridModel: the DSL brick %s declares %d variables but the transport has %d ; "
                        "align conservative_vars(...)" % (role, slot["n_vars"], nv))
            elif slot["min_vars"] > nv:
                raise ValueError(
                    "HybridModel: the native brick %s requires >= %d variables (transport=%d) ; e.g. a "
                    "fluid force makes no sense on a scalar transport"
                    % (role, slot["min_vars"], nv))

        self.n_vars = nv
        self.cons_names = list(hyp["cons_names"])
        self.cons_roles = list(hyp["cons_roles"])
        self.prim_names = list(hyp["prim_names"])
        self.gamma = hyp["gamma"]
        self.n_aux = max(hyp["n_aux"], src["n_aux"], ell["n_aux"])
        self._has_wave_speeds = bool(hyp.get("wave_speeds", True))
        self._slots = (hyp, src, ell)

    @staticmethod
    def _norm(prov, role, native_struct_name):
        """Normalize a slot (DSL CompiledBrick or NativeBrick) into a common dict."""
        if isinstance(prov, CompiledBrick):
            if prov.kind != role:
                raise ValueError("HybridModel: DSL brick of type %r placed in slot %r"
                                 % (prov.kind, role))
            d = dict(provider="dsl", struct_text=prov.struct_src, type_name=prov.type_name,
                     n_vars=prov.n_vars, min_vars=prov.n_vars, n_aux=prov.n_aux)
            if role == "hyperbolic":
                d.update(cons_names=prov.cons_names, cons_roles=prov.cons_roles,
                         prim_names=prov.prim_names, gamma=prov.gamma,
                         wave_speeds=getattr(prov, "has_wave_speeds", True))
            return d
        if isinstance(prov, NativeBrick):
            if prov.kind != role:
                raise ValueError("HybridModel: native brick of type %r placed in slot %r"
                                 % (prov.kind, role))
            d = dict(provider="native", struct_text=prov.emit(native_struct_name),
                     type_name="pops_generated::" + native_struct_name,
                     n_vars=prov.n_vars, min_vars=prov.min_vars, n_aux=prov.n_aux)
            if role == "hyperbolic":
                names = prov.var_names or []
                d.update(cons_names=list(names), cons_roles=roles_for(names),
                         prim_names=list(prov.prim_names or names), gamma=prov.gamma,
                         wave_speeds=True)  # native: unknown, the C++ requires-gate decides (historical)
            return d
        raise TypeError("HybridModel: slot %r must be a native brick (pops.* / NativeBrick) or a "
                        "compiled DSL brick (CompiledBrick) ; got %r" % (role, type(prov).__name__))

    def _emit_aot_source(self):
        """C++ source of the hybrid composite .so, behind the extern \"C\" ABI of compiled_block_abi.hpp
        (aot backend: same flat ABI as emit_cpp_aot_source). The bricks (generated DSL or native binding
        structs) are stitched together, then assembled into pops::CompositeModel<...>."""
        hyp, src, ell = self._slots
        parts = ['#include <pops/runtime/builders/compiled/compiled_block_abi.hpp>\n',
                 '#include <pops/physics/bricks/bricks.hpp>\n',   # CompositeModel + native bricks
                 '#include <pops/core/state/variables.hpp>\n']   # POPS_EXPORT_BLOCK_METADATA / _GAMMA
        for slot in self._slots:
            if slot["struct_text"]:
                parts.append(slot["struct_text"])
        parts.append('\nnamespace pops_generated { using AotModel = pops::CompositeModel<%s, %s, %s>; }\n'
                     % (hyp["type_name"], src["type_name"], ell["type_name"]))
        parts.append('POPS_DEFINE_COMPILED_BLOCK(pops_generated::AotModel)\n')
        parts.append('POPS_EXPORT_BLOCK_METADATA(pops_generated::AotModel)\n')
        if self.gamma is not None:
            parts.append('POPS_EXPORT_BLOCK_GAMMA(%r)\n' % self.gamma)
        return "".join(parts)

    def _model_hash(self):
        """Stable hash of the composite: provider + type + generated text of each slot (the text encodes
        the DSL formulas and the baked native parameters). Used as a cache key."""
        import hashlib
        parts = ["hybrid", self.name]
        for slot in self._slots:
            parts.append("%s|%s|%s" % (slot["provider"], slot["type_name"], slot.get("struct_text", "")))
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()

    def _bricks_and_composite(self):
        """C++ text of the stitched bricks (generated DSL + native binding structs) + composite type."""
        hyp, src, ell = self._slots
        bricks = "".join(s["struct_text"] for s in self._slots if s["struct_text"])
        composite = ("pops::CompositeModel<%s, %s, %s>"
                     % (hyp["type_name"], src["type_name"], ell["type_name"]))
        return bricks, composite

    def _emit_metadata(self, alias):
        """ABI metadata symbols (names/roles from conservative_vars, optional gamma), SHARED
        by the backends. @p alias: an alias WITHOUT a top-level comma (the preprocessor splits
        macro arguments on commas)."""
        out = '\nPOPS_EXPORT_BLOCK_METADATA(%s)\n' % alias
        if self.gamma is not None:
            out += 'POPS_EXPORT_BLOCK_GAMMA(%r)\n' % self.gamma
        return out

    def _emit_jit_source(self):
        """Source of the JIT library (backend 'prototype'): the hybrid composite behind an
        extern \"C\" factory (pops_make_model via pops::ModelAdapter). Host VIRTUAL dispatch (order-1
        Rusanov residual): fast iteration, to be plugged via System.add_dynamic_block. Hybrid
        counterpart of emit_cpp_so_source."""
        bricks, composite = self._bricks_and_composite()
        return ('#include <pops/runtime/dynamic/dynamic_model.hpp>\n'
                '#include <pops/physics/bricks/bricks.hpp>\n'
                '#include <pops/core/state/variables.hpp>\n'
                + bricks
                + '\nnamespace pops_generated { using JitModel = %s; }\n' % composite
                + 'extern "C" int pops_model_nvars() { return %d; }\n' % self.n_vars
                + 'extern "C" void* pops_make_model() { return new pops::ModelAdapter<pops_generated::JitModel>(); }\n'
                + 'extern "C" void pops_destroy_model(void* p) { delete static_cast<pops::IModel<%d>*>(p); }\n'
                % self.n_vars
                + self._emit_metadata("pops_generated::JitModel"))

    def _emit_native_source(self, target="system"):
        """C++ source of the NATIVE LOADER (backend 'production'): the hybrid composite as CompositeModel<...>
        behind a THIN extern \"C\" ABI. Like emit_cpp_native_loader, the .so does NOT carry the
        numerics: it INSTALLS the generated model as a native block of the facade via add_compiled_model<>,
        which builds the closures on the REAL CONTEXT of the facade -> same path as add_block,
        ZERO-COPY (MPI by construction, device-clean). pops_native_abi_key() freezes the ABI key at
        compile time, compared against the module's abi_key() by add_native_block (explicit rejection if
        headers/compiler/std diverge).

        @p target: 'system' -> pops_install_native (System&, evolve) ; 'amr_system' ->
        pops_install_native_amr (AmrSystem&, without evolve) inline add_compiled_model(AmrSystem&): the block
        runs the SAME AMR hierarchy as AmrSystem.add_block (reflux, regrid). DISTINCT symbols per
        target (a System loader is not pluggable onto AmrSystem.add_native_block, and vice versa)."""
        if target not in ("system", "amr_system"):
            raise ValueError("_emit_native_source: target 'system' | 'amr_system' (got %r)" % (target,))
        bricks, composite = self._bricks_and_composite()
        head = ('#include <pops/runtime/dynamic/abi_key.hpp>\n'        # POPS_ABI_KEY_LITERAL (key frozen at compile time)
                '#include <pops/physics/bricks/bricks.hpp>\n'         # CompositeModel + native bricks
                '#include <pops/core/state/variables.hpp>\n'
                '#include <string>\n')
        # Header template of the target (selective: do not pull the AMR machinery into a System loader).
        head += ('#include <pops/runtime/builders/compiled/dsl_block.hpp>\n' if target == "system"
                 else '#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>\n')
        # Preprocessor LITERAL, no call to abi_key_string(): an inline would be interposed
        # (ELF/RTLD_GLOBAL) toward the module's copy -> module key returned -> tautological guard.
        key = ('#if defined(_WIN32)\n'
               '#define POPS_LOADER_API extern "C" __declspec(dllexport)\n'
               '#else\n'
               '#define POPS_LOADER_API extern "C"\n'
               '#endif\n'
               'POPS_LOADER_API const char* pops_native_abi_key() {\n'
               '  return POPS_ABI_KEY_LITERAL;\n'
               '}\n')
        if target == "system":
            # pos_floor (ADC-76, Zhang-Shu positivity limiter): final flat argument, marshaled
            # down to the loader's make_block via add_compiled_model. Old signature = old .so =
            # rejected by the ABI key (the headers changed), never a wrong argument layout.
            install = ('POPS_LOADER_API void pops_install_native(void* sys, const char* name, const char* limiter,\n'
                       '                                    const char* riemann, const char* recon,\n'
                       '                                    const char* time, double gamma, int substeps,\n'
                       '                                    int evolve, int stride, double pos_floor) {\n'
                       '  pops::System* s = reinterpret_cast<pops::System*>(sys);\n'
                       '  pops::add_compiled_model<pops_generated::ProdModel>(*s, name, pops_generated::ProdModel{},\n'
                       '                                                    limiter, riemann, recon, time, gamma,\n'
                       '                                                    substeps, evolve != 0, stride,\n'
                       '                                                    pos_floor);\n'
                       '}\n')
        else:  # amr_system: AmrSystem overload (no evolve parameter, mono-block AMR)
            install = ('POPS_LOADER_API void pops_install_native_amr(void* sys, const char* name,\n'
                       '                                        const char* limiter, const char* riemann,\n'
                       '                                        const char* recon, const char* time,\n'
                       '                                        double gamma, int substeps) {\n'
                       '  pops::AmrSystem* s = reinterpret_cast<pops::AmrSystem*>(sys);\n'
                       '  pops::add_compiled_model<pops_generated::ProdModel>(*s, name, pops_generated::ProdModel{},\n'
                       '                                                    limiter, riemann, recon, time, gamma,\n'
                       '                                                    substeps);\n'
                       '}\n')
        return (head + bricks
                + '\nnamespace pops_generated { using ProdModel = %s; }\n' % composite
                + key + install + self._emit_metadata("pops_generated::ProdModel"))

    def compile(self, backend="aot", so_path=None, include=None, name=None, cxx=None, std=None,
                target="system"):
        """Compile the hybrid composite into a CompiledModel.

        ``backend`` :

        - 'prototype' -> add_dynamic_block: JIT, host VIRTUAL dispatch (order-1 Rusanov), fast
          iteration ; no MPI/AMR, no HLLC/Roe flux nor primitive recon ;
        - 'aot' -> add_compiled_block: self-sufficient .so (flat ABI, host-marshaled), mono-rank
          production path ; without MPI/AMR ;
        - 'production' -> add_native_block: zero-copy native loader that inlines add_compiled_model<>, SAME
          path as add_block (closures on the facade's real context), MPI by construction.
          The names/roles/gamma come from the .so metadata (no names=).

        ``target`` : 'system' (default) | 'amr_system'. 'amr_system' REQUIRES backend='production': the loader
        inlines add_compiled_model(AmrSystem&) (symbol pops_install_native_amr), the only AMR .so path ; to be
        plugged via AmrSystem.add_equation. The other backends have no AMR counterpart.

        so_path None -> out-of-source cache (key = model_hash + abi_key + backend + target)."""
        import os
        import sys
        import tempfile
        if backend not in ("prototype", "aot", "production"):
            raise ValueError("HybridModel.compile: backend 'prototype' | 'aot' | 'production' (got %r)"
                             % (backend,))
        if target not in ("system", "amr_system"):
            raise ValueError("HybridModel.compile: target 'system' | 'amr_system' (got %r)" % (target,))
        if target == "amr_system" and backend != "production":
            raise ValueError("HybridModel.compile: target='amr_system' only exists for "
                             "backend='production' (native AMR path) ; got backend=%r" % (backend,))
        mode = {"prototype": "jit", "aot": "aot", "production": "native"}[backend]
        if include is None:
            include = pops_include()
        if std is None:  # the native loader shares the module ABI (std derived from the loader: c++20 under
            # Kokkos, c++23 otherwise, cf. loader_cxx_std/compile_native); jit/aot stay at c++20.
            std = loader_cxx_std() if mode == "native" else "c++20"
        # NATIVE (production) AND AOT: compiler following the Kokkos backend (g++ by default,
        # nvcc_wrapper if explicit), Kokkos flags without linking libkokkos (single runtime), feature-key
        # kokkos in the cache. KOKKOS-ONLY: the hybrid aot includes the pops headers
        # (compiled_block_abi.hpp -> multifab/for_each) which require POPS_HAS_KOKKOS, same flags as
        # compile_aot; only the jit (prototype) stays pure host (-O2, dynamic_model/bricks without
        # multifab). kokkos_like also serves the cache key.
        native = (mode == "native")
        kokkos_like = native or mode == "aot"
        if mode == "aot" and _native_kokkos_root() is None:
            raise RuntimeError(
                "HybridModel.compile: adc_cpp is Kokkos-only -- the AOT model includes the pops "
                "headers which require Kokkos. Point to an installed Kokkos via POPS_KOKKOS_ROOT (or "
                "Kokkos_ROOT), e.g. `export POPS_KOKKOS_ROOT=/path/to/kokkos` (Serial is enough "
                "on CPU). Run `python -c \"import pops; pops.doctor()\"` for a full diagnosis.")
        if native:  # pre-dlopen guard: headers != build of _pops -> clear remedy (cf. compile_native)
            _check_headers_match_module(include)
            _warn_kokkos_parity()
        eff_cxx = _native_kokkos_compiler(cxx) if kokkos_like else _default_cxx(cxx)
        if not eff_cxx:
            raise RuntimeError("HybridModel.compile: no C++ compiler found")
        std = _probe_cxx_std(eff_cxx, std)  # ACTIONABLE error if the std is not supported
        model_hash = self._model_hash()
        abi_key = _abi_key_python(include, eff_cxx, std)
        if so_path is None:
            cache_backend = (("hybrid-" + backend + ";" + _native_feature_key()) if kokkos_like
                             else "hybrid-" + backend)
            so_path = _cache_so_path(model_hash, abi_key, cache_backend, target, name)
            if os.path.exists(so_path):
                _record_so_backend(so_path, "hybrid-" + backend)
                return self._compiled_model(so_path, backend, target, abi_key, model_hash, eff_cxx, std)
        else:
            # Explicit so_path: avoid the dlopen handle cache re-serving ANOTHER backend already loaded at
            # this path in the process (cf. _backend_distinct_so_path). The hybrid backend is distinct from
            # the non-hybrid backend of the same name (different ABI) -> 'hybrid-' prefix.
            so_path = _backend_distinct_so_path(so_path, "hybrid-" + backend)

        # aot AND native run the production path -> same optimization flags (cf. _dsl_optflags);
        # only the jit/prototype stays at -O2 (Rusanov host residue, perf out of scope).
        optflags = _dsl_optflags() if kokkos_like else ["-O2"]
        flags = ["-shared", "-fPIC", "-std=" + std, *optflags]
        kokkos_link_flags = []
        if mode == "jit":
            source = self._emit_jit_source()
        elif mode == "aot":
            # Like compile_aot: Kokkos flags without linking libkokkos (the _pops module has already loaded the
            # runtime, singleton), undefined symbols resolved at load time; Apple-ld then requires
            # -undefined dynamic_lookup (on ELF/Linux -shared already allows them).
            source = self._emit_aot_source()
            kokkos_compile_flags, kokkos_link_flags = _native_kokkos_flags()
            flags += kokkos_compile_flags
            if sys.platform == "darwin":
                flags += ["-undefined", "dynamic_lookup"]
        else:  # native: header signature + Kokkos backend parity (cf. compile_native / _native_kokkos_flags)
            source = self._emit_native_source(target=target)  # undefined symbols resolved at load time (_pops module)
            flags.append('-DPOPS_HEADER_SIG="%s"' % pops_header_signature(include))
            kokkos_compile_flags, kokkos_link_flags = _native_kokkos_flags()
            flags += kokkos_compile_flags
            if sys.platform == "darwin":  # Apple-ld: explicitly allow undefined symbols (cf. compile_native)
                flags += ["-undefined", "dynamic_lookup"]
        with tempfile.TemporaryDirectory() as tmp:
            cpp = os.path.join(tmp, "model_hybrid.cpp")
            with open(cpp, "w") as f:
                f.write(source)
            _run_compile([eff_cxx, *flags, "-I", include, cpp, "-o", so_path, *kokkos_link_flags],
                         "HybridModel, backend " + backend)
        _record_so_backend(so_path, "hybrid-" + backend)
        return self._compiled_model(so_path, backend, target, abi_key, model_hash, eff_cxx, std)

    def _compiled_model(self, so_path, backend, target, abi_key, model_hash, cxx, std):
        return CompiledModel(
            so_path=so_path, backend=backend, adder=HyperbolicModel.adder_for(backend),
            target=target, cons_names=self.cons_names, cons_roles=self.cons_roles,
            prim_names=self.prim_names, n_vars=self.n_vars, gamma=self.gamma, n_aux=self.n_aux,
            params={}, caps=_BACKEND_CAPS[backend], abi_key=abi_key, model_hash=model_hash,
            cxx=cxx, std=std, hllc=getattr(self, "_hllc", False),
            roe=getattr(self, "_roe", False),
            wave_speeds=getattr(self, "_has_wave_speeds", True))


# --- Generic COUPLED inter-species source (P5 phase 1, EXPLICIT splitting) -----------------------
#
# pops.dsl.CoupledSource describes an ARBITRARY coupling between species as FORMULAS, beyond the named
# couplings (Ionization / Collision / ThermalExchange) which freeze a formula. We read fields (block,
# role) as INPUT and WRITE source terms (block, role) given by symbolic expressions
# (same Expr as the models DSL: +, *, -, /, **, sqrt, params). compile(backend) compiles each
# expr into postfix BYTECODE (stack machine) that C++ evaluates in the same for_each_cell device as the
# named couplings -- NO .so nor Python callback per cell. Applied AFTER transport (split).

# Inverse of pops::role_name: DSL role name (CamelCase, cf. CANONICAL_ROLES) -> canonical lowercase
# name expected by pops::role_from_name (C++ boundary). Single source of the correspondence.
_ROLE_TO_CANONICAL = {
    "Density": "density",
    "MomentumX": "momentum_x", "MomentumY": "momentum_y", "MomentumZ": "momentum_z",
    "Energy": "energy",
    "VelocityX": "velocity_x", "VelocityY": "velocity_y", "VelocityZ": "velocity_z",
    "Pressure": "pressure", "Temperature": "temperature", "Scalar": "scalar",
}


def _role_canonical(role):
    """Canonical role name (lowercase, C++ boundary) for a DSL role. Accepts already-canonical."""
    if role in _ROLE_TO_CANONICAL:
        return _ROLE_TO_CANONICAL[role]
    if role in _ROLE_TO_CANONICAL.values():
        return role
    raise ValueError("CoupledSource: unknown role %r (roles: %s)"
                     % (role, ", ".join(sorted(_ROLE_TO_CANONICAL))))


# Stack machine opcodes: MIRROR of pops::CsOp (coupled_source_program.hpp). FROZEN values
# (transported as-is by the Python -> C++ ABI).
_CS_PUSHREG = 0
_CS_ADD = 1
_CS_SUB = 2
_CS_MUL = 3
_CS_DIV = 4
_CS_NEG = 5
_CS_POW = 6
_CS_SQRT = 7

# FROZEN capacities, mirror of coupled_source_program.hpp (kCsMaxReg / kCsMaxTerms / kCsMaxProg). We
# diagnose on the Python side (clear error) before reaching the C++ boundary.
_CS_MAX_REG = 32
_CS_MAX_TERMS = 16
_CS_MAX_PROG = 256


class _CsField(Var):
    """Symbolic handle of a (block, role): it is a Var (hence a full Expr) whose environment NAME
    '<block>::<role>' indexes both the numpy eval (env) and the input register at the
    bytecode codegen. Subclassing Var directly gives operators / _wrap / to_cpp / eval / deps, so
    `+k * ne * ng` builds the expected Expr tree without delegation."""

    def __init__(self, block, role):
        super().__init__("%s::%s" % (block, role), "coupled_field")
        self.block = block
        self.role = role

    def __repr__(self): return "_CsField(%r, %r)" % (self.block, self.role)


class _CsBlock:
    """Construction helper: `src.block("electrons").role("density")` -> _CsField. Records the
    (block, role) requested on the source to fix the order of the input registers."""

    def __init__(self, src, name):
        self._src = src
        self.name = name

    def role(self, role):
        return self._src._field(self.name, role)


class CompiledCoupledSource:
    """Result of CoupledSource.compile(...): packages the FLAT ABI (bytecode) ready for
    System.add_coupled_source, + the REFERENCE numpy evaluator (same Expr) for tests / a Python
    integrator. No .so: the coupling is interpreted on the C++ side (device stack machine)."""

    def __init__(self, name, backend, in_blocks, in_roles, consts, out_blocks, out_roles,
                 prog_ops, prog_args, prog_lens, terms, reg_order, frequency=0.0,
                 freq_prog_ops=None, freq_prog_args=None, frequency_expr=None):
        self.name = name
        self.backend = backend
        self.frequency = float(frequency)  # mu [1/s] declared CONSTANT (0 = no constant bound)
        self.in_blocks = list(in_blocks)
        self.in_roles = list(in_roles)        # canonical (lowercase, C++ boundary)
        self.consts = list(consts)
        self.out_blocks = list(out_blocks)
        self.out_roles = list(out_roles)      # canonical
        self.prog_ops = list(prog_ops)
        self.prog_args = list(prog_args)
        self.prog_lens = list(prog_lens)
        # PER-CELL frequency mu(U): bytecode (same inputs/constants as the source). EMPTY =
        # constant frequency only. Transported to System/AmrSystem.add_coupled_source.
        self.freq_prog_ops = list(freq_prog_ops) if freq_prog_ops else []
        self.freq_prog_args = list(freq_prog_args) if freq_prog_args else []
        self._frequency_expr = frequency_expr  # reference Expr (numpy eval for tests); None if constant
        self._terms = list(terms)             # [(block, role_canonical, Expr)]: numpy reference
        self._reg_order = list(reg_order)     # env names '<block>::<role>' in register order

    def __repr__(self):
        return ("CompiledCoupledSource(name=%r, backend=%r, n_in=%d, n_const=%d, n_terms=%d)"
                % (self.name, self.backend, len(self.in_blocks), len(self.consts),
                   len(self.out_blocks)))

    def reference_terms(self, fields):
        """Evaluates the source terms on numpy arrays (REFERENCE for tests). @p fields:
        dict (block, role_canonical) -> array; returns [(block, role_canonical, dS)] with dS = S
        (the evaluated symbolic term), BEFORE multiplication by dt. Same Expr as the C++ codegen."""
        env = {}
        for (block, role), arr in fields.items():
            env["%s::%s" % (block, _role_canonical(role))] = arr
        return [(b, r, e.eval(env)) for (b, r, e) in self._terms]

    def reference_frequency(self, fields):
        """Evaluates the PER-CELL frequency mu(U) on numpy arrays (REFERENCE for tests):
        same Expr / register table as the C++ bytecode. @p fields: dict (block, role_canonical) ->
        array; returns the mu array (same formulas). Returns None if the frequency is CONSTANT
        (no Expr) -- use .frequency in that case."""
        if self._frequency_expr is None:
            return None
        env = {}
        for (block, role), arr in fields.items():
            env["%s::%s" % (block, _role_canonical(role))] = arr
        return self._frequency_expr.eval(env)


class CoupledSource:
    """Generic COUPLED inter-species source (pops.dsl), Phase 1 (EXPLICIT splitting). Reuses the
    Expr of the models DSL for the source formulas; no coupling is hard-coded.

        src = pops.dsl.CoupledSource("ionization")
        ne = src.block("electrons").role("density")
        ng = src.block("neutrals").role("density")
        k  = src.param("Kiz", 1.0)
        src.add("electrons", role="density", expr=+k * ne * ng)
        src.add("neutrals",  role="density", expr=-k * ne * ng)
        sim.add_coupling(src.compile(backend="production"))

    compile(backend) -> CompiledCoupledSource: flat ABI (bytecode) consumed by
    System.add_coupled_source (C++ side: stack machine evaluated in a for_each_cell device, MPI-safe,
    named functor). The backend (production / prototype) does NOT change the numerics: the bytecode is
    interpreted on the C++ side in both cases (no .so per coupling); it is kept for introspection
    and API parity with the models DSL."""

    def __init__(self, name="coupled_source"):
        self.name = name
        self._fields = {}    # '<block>::<role>' -> _CsField (single input register per (block, role))
        self._reg_order = []  # order of appearance of the input fields (-> register order)
        self._params = {}    # name -> Param
        self._terms = []     # [(block, role_canonical, Expr)]
        # Indices (in self._terms) of the terms EMITTED BY add_pair, by pair: [(idx_gain, idx_loss)].
        # add_pair guarantees that the two terms carry EXACTLY the same evaluated Expr, one +expr,
        # the other -expr (Neg) -> conservative exchange by construction. verify_conservation=True
        # revisits them at compile time to CHECK the property (and detect a breach on the manual add side).
        self._pairs = []
        self._frequency = 0.0  # mu [1/s] declared CONSTANT (coupling step bound; 0 = no bound)
        # optional PER-CELL mu(U): an Expr (same vocabulary as the terms: block().role() +
        # param()) emitted into bytecode at compile() against the SAME register table. None = constant.
        self._frequency_expr = None

    def frequency(self, mu):
        """Declared coupling FREQUENCY mu [1/s] (vague 3 audit): step bound dt <= cfl / mu
        aggregated by System/AmrSystem::step_cfl (reason 'coupled_source:<name>'). Couplings are
        applied ONCE per MACRO-step (splitting, apply_couplings(dt)): the bound applies to the
        macro-dt, WITHOUT a substeps/stride factor. mu <= 0 = no bound (historical).

        @p mu accepts TWO forms:
          - a number (float / int) -> CONSTANT frequency (historical path, bit-identical);
          - an Expr of the SAME vocabulary as the terms (fields block().role() + param()) ->
            PER-CELL frequency mu(U), emitted into bytecode at compile() and evaluated per cell on the
            C++ side (MAX + all_reduce_max -> dt <= cfl / max(mu)). The referenced fields MUST be
            declared via .block(...).role(...) (as for the terms); otherwise compile() raises an
            EXPLICIT ValueError (field used without .block(...).role(...)).

        Returns self (chainable)."""
        # An Expr/_CsField/Param -> per-cell frequency (bytecode); a scalar -> constant.
        if isinstance(mu, (int, float)) and not isinstance(mu, bool):
            self._frequency = float(mu)
            self._frequency_expr = None
        else:
            self._frequency_expr = _wrap(mu)  # Expr / _CsField / Param -> tree node (cf. _wrap)
            self._frequency = 0.0             # the bytecode carries the bound; no duplicate constant
        return self

    # --- symbolic construction ----------------------------------------------------------------
    def block(self, name):
        """Handle of a block: .role(role) derives a symbolic field (block, role) from it."""
        return _CsBlock(self, name)

    def _field(self, block, role):
        canon = _role_canonical(role)
        key = "%s::%s" % (block, canon)
        if key not in self._fields:
            f = _CsField(block, canon)
            self._fields[key] = f
            self._reg_order.append(key)
        return self._fields[key]

    def param(self, name, value):
        """NAMED constant parameter, usable like an Expr (inlines as a real in the bytecode)."""
        p = Param(name, value, kind="const")
        self._params[name] = p
        return p

    def add(self, block, role=None, expr=None):
        """Adds a source TERM: d_t (block.role) += expr. @p expr is an Expr / _CsField / Param /
        number. Several adds on the same (block, role) ADD UP (sum of source terms)."""
        if role is None:
            raise ValueError("CoupledSource.add: role= required")
        if expr is None:
            raise ValueError("CoupledSource.add: expr= required")
        e = expr if isinstance(expr, Expr) else _wrap(expr)  # _CsField / Var are already Expr; Param/scalar -> _wrap
        self._terms.append((block, _role_canonical(role), e))
        return self

    def add_pair(self, block_a, block_b, role=None, expr=None):
        """Adds a CONSERVATIVE EXCHANGE of the quantity @p role between @p block_a and @p block_b, described
        by a SINGLE expression @p expr (Expr / _CsField / Param / number).

        Sign convention (to remember): @p block_a GAINS +expr, @p block_b LOSES -expr, on the SAME
        evaluated value of @p expr. In other words:

            d_t (block_a.role) += +expr
            d_t (block_b.role) += -expr

        It is the DSL equivalent of the NAMED C++ couplings (add_collision / add_thermal_exchange), which
        compute ONE value and apply it with two opposite signs: the sum over the two blocks of the
        exchanged term is zero at each cell and at each step, so the total quantity sum(role) over
        (block_a, block_b) is CONSERVED by construction -- independently of the chosen formula, the dt
        and the state. Choose @p expr >= 0 for a transfer from B to A (A gains, B loses); a negative
        sign of expr simply reverses the transfer direction (conservation holds in all cases).

        Contrast with two hand-written .add(...) (+expr on A, -expr on B): add_pair guarantees that
        the TWO legs carry the SAME Expr (the second is exactly Neg of the first), whereas by hand
        nothing prevents writing two slightly different formulas by mistake -> conservation broken
        silently. add_pair removes this risk; compile(verify_conservation=True) also checks it
        for hand-written couplings.

        @p block_a and @p block_b must be distinct. add_pair is purely ADDITIVE on top of .add:
        the manual API stays available and unchanged. Returns self (chainable)."""
        if role is None:
            raise ValueError("CoupledSource.add_pair: role= required")
        if expr is None:
            raise ValueError("CoupledSource.add_pair: expr= required")
        if block_a == block_b:
            raise ValueError("CoupledSource.add_pair: block_a and block_b must be distinct "
                             "(received %r for both)" % (block_a,))
        canon = _role_canonical(role)
        gain = expr if isinstance(expr, Expr) else _wrap(expr)  # +expr (gaining leg)
        loss = Neg(gain)                                        # -expr: SAME subtree, opposite sign
        idx_gain = len(self._terms)
        self._terms.append((block_a, canon, gain))
        idx_loss = len(self._terms)
        self._terms.append((block_b, canon, loss))
        self._pairs.append((idx_gain, idx_loss))
        return self

    # --- bytecode codegen -----------------------------------------------------------------------
    def _emit_program(self, expr, reg_index):
        """Compile @p expr (Expr tree) into postfix bytecode (parallel ops/args lists) against the
        register table @p reg_index (env name '<bloc>::<role>' OR constant value -> register index).
        Constants (Const / inline Param) become a dedicated constant register. Postfix traversal
        (recursion over the tree structure): a Var pushes its register, a binary emits its two
        subtrees then the opcode, etc. -- exactly the semantics of CsProgram::eval on the C++ side."""
        ops, args = [], []

        def emit(node):
            if isinstance(node, Var):
                if node.name not in reg_index:
                    raise ValueError("CoupledSource: field %r used without .block(...).role(...)"
                                     % node.name)
                ops.append(_CS_PUSHREG)
                args.append(reg_index[node.name])
            elif isinstance(node, Const):
                ops.append(_CS_PUSHREG)
                args.append(self._const_reg(node.value, reg_index))
            elif isinstance(node, Neg):
                emit(node.a)
                ops.append(_CS_NEG)
                args.append(0)
            elif isinstance(node, Sqrt):
                emit(node.a)
                ops.append(_CS_SQRT)
                args.append(0)
            elif isinstance(node, Add):
                emit(node.a)
                emit(node.b)
                ops.append(_CS_ADD)
                args.append(0)
            elif isinstance(node, Sub):
                emit(node.a)
                emit(node.b)
                ops.append(_CS_SUB)
                args.append(0)
            elif isinstance(node, Mul):
                emit(node.a)
                emit(node.b)
                ops.append(_CS_MUL)
                args.append(0)
            elif isinstance(node, Div):
                emit(node.a)
                emit(node.b)
                ops.append(_CS_DIV)
                args.append(0)
            elif isinstance(node, Pow):
                emit(node.a)
                emit(node.b)
                ops.append(_CS_POW)
                args.append(0)
            else:
                raise TypeError("CoupledSource: expression node not supported in Phase 1: %r "
                                "(supported: +, -, *, /, **, unary -, sqrt, field, constant)"
                                % type(node).__name__)

        emit(expr)
        return ops, args

    def _const_reg(self, value, reg_index):
        """Register index of a constant @p value (deduplicated). Constants occupy the
        registers AFTER the input fields (cf. CoupledSourceKernel: r[n_in + c] = consts[c])."""
        key = ("const", float(value))
        if key not in reg_index:
            reg_index[key] = len(self._reg_order) + len(self._consts)
            self._consts.append(float(value))
        return reg_index[key]

    @staticmethod
    def _signed_key(expr):
        """SIGNED STRUCTURAL key of @p expr: (sign, body_key), where sign is +1 / -1 and
        body_key is the structural key (_key) of the expression stripped of ALL its leading Neg
        (the sign folds in each peeled Neg). Two expressions are structurally OPPOSITE iff they have
        the SAME body_key and opposite signs (e.g. E and Neg(E), or -E and Neg(-E)=E). Peeling
        ALL leading Neg makes the key robust to the +expr / -expr pair from add_pair EVEN when expr is
        already a Neg (add_pair sets loss = Neg(gain), hence one more Neg). We do NOT normalize the
        internal algebra (k*ne vs ne*k): at worst a false 'non-conservative' on differently written
        forms, NEVER a false 'conservative' (the check stays conservative, hence sound)."""
        sign = 1
        while isinstance(expr, Neg):
            sign = -sign
            expr = expr.a
        return (sign, _key(expr))

    def _verify_conservation(self):
        """Verify that, role by role, the sum of the source terms CANCELS structurally: each
        contribution +E on one block is compensated by a contribution -E (same structural body) on
        another block. Raises an EXPLICIT ValueError otherwise. This is exactly the property add_pair
        guarantees by construction; this check extends it to hand-written couplings (two .add) and detects
        a break (slightly different formulas, forgotten sign, orphan term). Purely symbolic
        (no numerical evaluation): same structural key as the codegen CSE."""
        from collections import Counter
        per_role = {}
        for (_block, role, expr) in self._terms:
            sign, body = self._signed_key(expr)
            # Signed counter per structural body: +1 for +E, -1 for -E. Everything cancels => conservative.
            c = per_role.setdefault(role, Counter())
            c[body] += sign
        offenders = []
        for role in sorted(per_role):
            for body, net in per_role[role].items():
                if net != 0:
                    offenders.append((role, body, net))
        if offenders:
            details = "; ".join(
                "role '%s': term %r not compensated (net=%+d)" % (role, body, net)
                for (role, body, net) in offenders)
            raise ValueError(
                "CoupledSource.compile(verify_conservation=True): NON-conservative coupling. "
                "Each contribution +E on one block must be compensated by -E (same expression) "
                "on another block (use add_pair to guarantee it). Uncompensated terms: "
                + details)

    def compile(self, backend="production", verify_conservation=False):
        """Compile the source into a CompiledCoupledSource (flat bytecode ABI). @p backend documents
        the intent (API parity with the model DSL); the numerics are identical (C++ interpreter).

        @p verify_conservation (opt-in, default False): SYMBOLIC check that the coupling conserves
        each quantity (role) -- the sum of the source terms of a same role cancels structurally
        (each +E compensated by a -E on another block). add_pair satisfies this property by
        construction; this mode extends it to hand-written couplings (two .add) and raises an EXPLICIT
        ValueError if a term is not compensated (divergent formula, forgotten sign, orphan term).
        Off by default: a deliberately NON-conservative coupling (net creation/destruction, e.g.
        ionization creating an e/i pair) stays legal without passing the flag."""
        if not self._terms:
            raise ValueError("CoupledSource.compile: no term (.add(...) required)")
        if verify_conservation:
            self._verify_conservation()
        # Register table: input fields first (order of appearance), constants next.
        reg_index = {key: i for i, key in enumerate(self._reg_order)}
        self._consts = []
        prog_ops, prog_args, prog_lens = [], [], []
        out_blocks, out_roles = [], []
        for (block, role, expr) in self._terms:
            ops, args = self._emit_program(expr, reg_index)
            if len(ops) > _CS_MAX_PROG:
                raise ValueError("CoupledSource: program of term (%s.%s) too long (%d > %d)"
                                 % (block, role, len(ops), _CS_MAX_PROG))
            prog_ops += ops
            prog_args += args
            prog_lens.append(len(ops))
            out_blocks.append(block)
            out_roles.append(role)
        # Optional PER-CELL FREQUENCY: its program is emitted AFTER the terms, against the SAME
        # register table (reg_index) and the SAME constant list (self._consts) -- the referenced fields
        # must be declared via .block().role() (otherwise _emit_program raises: field used
        # without .block(...).role(...)). The frequency's own constants are appended after those of the
        # terms; on the C++ side they occupy the same registers r[n_in ..] (CoupledFreqKernel loads them
        # like the source). Constant frequency (or none) -> empty program (historical path).
        freq_prog_ops, freq_prog_args = [], []
        if self._frequency_expr is not None:
            freq_prog_ops, freq_prog_args = self._emit_program(self._frequency_expr, reg_index)
            if len(freq_prog_ops) > _CS_MAX_PROG:
                raise ValueError("CoupledSource: frequency program too long (%d > %d)"
                                 % (len(freq_prog_ops), _CS_MAX_PROG))
        n_reg = len(self._reg_order) + len(self._consts)
        if n_reg > _CS_MAX_REG:
            raise ValueError("CoupledSource: too many registers (inputs + constants = %d > %d)"
                             % (n_reg, _CS_MAX_REG))
        if len(out_blocks) > _CS_MAX_TERMS:
            raise ValueError("CoupledSource: too many source terms (%d > %d)"
                             % (len(out_blocks), _CS_MAX_TERMS))
        in_blocks = [self._fields[key].block for key in self._reg_order]
        in_roles = [self._fields[key].role for key in self._reg_order]
        return CompiledCoupledSource(
            name=self.name, backend=backend, in_blocks=in_blocks, in_roles=in_roles,
            consts=list(self._consts), out_blocks=out_blocks, out_roles=out_roles,
            prog_ops=prog_ops, prog_args=prog_args, prog_lens=prog_lens,
            terms=self._terms, reg_order=self._reg_order, frequency=self._frequency,
            freq_prog_ops=freq_prog_ops, freq_prog_args=freq_prog_args,
            frequency_expr=self._frequency_expr)
