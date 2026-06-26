"""HyperbolicModel and Param: the symbolic mini-DSL core.

Python WRITES the formulas (named variables, expressions); the operations build
an expression TREE. :class:`HyperbolicModel` declares the conservative variables,
the primitives (defined by formulas), the flux, the eigenvalues, the source and
the elliptic contribution.

The 1500-line ``HyperbolicModel`` is assembled here from topical authoring mixins
so no file exceeds the Spec-4 500-line bound (see ``physics/_authoring_*``):
variables, flux, sources, Riemann, the operator-first view, the numpy evaluators,
the runtime parameters, and the codegen wrappers. The stable user facade
:class:`pops.physics.facade.Model` COMPOSES a private ``HyperbolicModel``.

Import-graph rule (Spec 4): this module imports only :mod:`pops.ir`; any
:mod:`pops.codegen` / ``_pops`` use is LAZY, inside method bodies (the codegen
wrappers in ``_authoring_codegen`` and the lazy ``_BACKENDS`` metaclass).
"""
from pops.ir import (  # noqa: F401  -- node classes used by Param's operator overloads
    _wrap, Const, Add, Sub, Mul, Div, Pow, Neg, Var)
from pops.ir.values import RuntimeParamRef

from ._authoring_vars import _VariablesMixin
from ._authoring_flux import _FluxMixin
from ._authoring_sources import _SourceMixin
from ._authoring_riemann import _RiemannMixin
from ._authoring_view import _OperatorViewMixin
from ._authoring_eval import _EvalMixin
from ._authoring_params import _RuntimeParamsMixin
from ._authoring_codegen import _CodegenMixin


class _BackendLazyMeta(type):
    """Metaclass exposing ``HyperbolicModel._BACKENDS`` / ``._BACKEND_CAPS`` lazily.

    The single source of truth is :mod:`pops.codegen.compile`; resolving it at class
    body time would make ``import pops.physics`` pull in codegen, breaking the Spec-4
    import-graph rule. The metaclass resolves the tables on first CLASS-level access
    instead, so ``HyperbolicModel._BACKENDS[backend]`` keeps working unchanged.
    """

    def __getattr__(cls, name):
        if name in ("_BACKENDS", "_BACKEND_CAPS"):
            from pops.codegen import compile as _cg
            return getattr(_cg, name)
        raise AttributeError(name)


class HyperbolicModel(_VariablesMixin, _FluxMixin, _SourceMixin, _RiemannMixin,
                      _OperatorViewMixin, _EvalMixin, _RuntimeParamsMixin, _CodegenMixin,
                      metaclass=_BackendLazyMeta):
    """Hyperbolic model written as FORMULAS: conservative variables, primitives (defined by
    expressions), flux, eigenvalues, source, elliptic contribution. cf. module docstring.

    The behaviour lives in the topical authoring mixins; this concrete class only assembles
    them and owns ``__init__`` (the full instance-attribute layout the mixins operate on)."""

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
