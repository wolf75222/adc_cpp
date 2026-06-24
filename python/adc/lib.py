"""adc.lib -- a catalog of typed brick descriptors and IR macros (Spec 3).

adc.lib is NOT a Python numerics library. Every entry is one of:

* a NATIVE brick -- a descriptor naming a C++ symbol already in ``include/adc``
  (``adc.lib.riemann.HLLC()`` -> ``adc::HLLCFlux``); a catalogued brick with no native
  symbol yet carries ``available=False`` and an empty ``native_id`` (never a fake id);
* a GENERATED brick -- a descriptor of a DSL-authored brick compiled to C++;
* a MACRO brick -- a Python function that builds Program IR
  (``adc.lib.time.predictor_corrector`` delegates to :mod:`adc.time` ``std``);
* an EXTERNAL C++ brick -- a descriptor of a user ``.so`` registered by id
  (``adc.lib.riemann.User("my_hllc")``).

A descriptor carries metadata only -- a native id, a runtime scheme string,
requirements and capabilities. It computes nothing; the codegen and runtime
consume it. The namespaces mirror the Spec 3 catalog (riemann, reconstruction,
limiters, spatial, fields, solvers, preconditioners, diagnostics, projections,
invariants, time).
"""
from types import SimpleNamespace

__all__ = ["BrickDescriptor", "riemann", "reconstruction", "limiters", "spatial",
           "fields", "solvers", "preconditioners", "diagnostics", "projections",
           "invariants", "time"]

BRICK_TYPES = ("native", "generated", "macro", "external_cpp")


class BrickDescriptor:
    """A typed, numerics-free descriptor of a numerical brick.

    Identity is by all metadata fields so two descriptors of the same brick
    compare equal (used to detect a re-selected brick and to key the artifact
    hash). It is intentionally inert: it has no ``eval`` / ``compile`` / call.
    """

    def __init__(self, name, brick_type, *, category="brick", native_id="",
                 scheme=None, requirements=None, capabilities=None, options=None,
                 available=True, expression=None):
        if brick_type not in BRICK_TYPES:
            raise ValueError("brick_type %r must be one of %s"
                             % (brick_type, ", ".join(BRICK_TYPES)))
        self.name = str(name)
        self.brick_type = str(brick_type)
        self.category = str(category)
        self.native_id = str(native_id)
        self.scheme = scheme
        self.requirements = dict(requirements or {})
        self.capabilities = dict(capabilities or {})
        self.options = dict(options or {})
        self.available = bool(available)
        # Optional board value carried by a generated/macro brick; kept OFF the
        # identity key (it may be an unhashable board node).
        self.expression = expression

    def _key(self):
        return (self.category, self.name, self.brick_type, self.native_id,
                self.scheme, tuple(sorted(self.options.items())))

    def __eq__(self, other):
        return isinstance(other, BrickDescriptor) and self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def __repr__(self):
        return "BrickDescriptor(%r, %r, scheme=%r)" % (
            self.name, self.brick_type, self.scheme)


# Native ids below are the REAL C++ symbols in include/adc (verified): the FV bricks
# live at top level in ``namespace adc`` (e.g. adc::HLLCFlux), not under a numerics/fv
# namespace. Some catalogued bricks have no native type yet -- they are emitted with
# ``available=False`` and an EMPTY native_id rather than a fabricated symbol.
def _native(name, native_id, scheme, *, category, caps=None, **options):
    """A native-brick descriptor; ``caps`` lists required model capabilities."""
    req = {"capabilities": list(caps)} if caps is not None else {}
    return BrickDescriptor(name, "native", category=category, native_id=native_id,
                           scheme=scheme, requirements=req, options=options or None)


def _planned(name, scheme, *, category, **options):
    """A catalogued brick with NO native C++ symbol yet (available=False, no id).

    It names the slot in the catalog without overclaiming a symbol; wiring a native
    type for it is tracked as a follow-up.
    """
    return BrickDescriptor(name, "native", category=category, native_id="",
                           scheme=scheme, options=options or None, available=False)


# --- riemann ---------------------------------------------------------------
def _riemann(name, native_id, caps):
    return _native(name, native_id, name, category="riemann", caps=caps)


riemann = SimpleNamespace(
    Rusanov=lambda: _riemann("rusanov", "adc::RusanovFlux", ["max_wave_speed"]),
    HLL=lambda: _riemann("hll", "adc::HLLFlux", ["physical_flux", "wave_speeds"]),
    HLLC=lambda: _riemann("hllc", "adc::HLLCFlux",
                          ["physical_flux", "pressure", "wave_speeds",
                           "contact_speed", "hllc_star_state"]),
    Roe=lambda: _riemann("roe", "adc::RoeFlux", ["physical_flux", "roe_average"]),
    User=lambda native_id, **opts: BrickDescriptor(
        native_id, "external_cpp", category="riemann", native_id=native_id,
        scheme="user", options=opts or None),
)


def _hook(name, scheme):
    """A capability-hook selector descriptor: it picks a canonical model hook (e.g. the Euler
    contact speed / star state, the Einfeldt wave speeds) that the native solver consumes. It
    computes nothing; the hook C++ is generated from the model (roles) by the dsl backend."""
    return BrickDescriptor(name, "macro", category="riemann_hook", scheme=scheme)


# Canonical capability-hook selectors used by m.riemann(..., wave_speeds=, contact_speed=, star_state=).
riemann.speeds = SimpleNamespace(
    einfeldt=lambda: _hook("einfeldt", "einfeldt"),
    davis=lambda: _hook("davis", "davis"),
)
riemann.hllc = SimpleNamespace(
    contact_speed=SimpleNamespace(euler=lambda: _hook("euler_contact", "euler")),
    star_state=SimpleNamespace(euler=lambda: _hook("euler_star", "euler")),
)


# --- reconstruction --------------------------------------------------------
# adc::Weno5 IS the WENO5-Z reconstruction (it wraps weno5z()); WENO5 and WENO5Z both
# select it. MUSCL is reconstruction-by-limiter; its native limiter type is adc::Minmod.
reconstruction = SimpleNamespace(
    FirstOrder=lambda: _native("firstorder", "adc::NoSlope", "firstorder",
                               category="reconstruction"),
    MUSCL=lambda limiter="minmod": _native(
        "muscl", "adc::Minmod", limiter, category="reconstruction", limiter=limiter),
    WENO5=lambda: _native("weno5", "adc::Weno5", "weno5", category="reconstruction"),
    WENO5Z=lambda: _native("weno5z", "adc::Weno5", "weno5", category="reconstruction"),
    User=lambda native_id, **opts: BrickDescriptor(
        native_id, "external_cpp", category="reconstruction", native_id=native_id,
        scheme="user", options=opts or None),
)


# --- limiters --------------------------------------------------------------
limiters = SimpleNamespace(
    Minmod=lambda: _native("minmod", "adc::Minmod", "minmod", category="limiter"),
    VanLeer=lambda: _native("vanleer", "adc::VanLeer", "vanleer", category="limiter"),
    # MC / Superbee are catalogued but have no native type yet (available=False).
    MC=lambda: _planned("mc", "mc", category="limiter"),
    Superbee=lambda: _planned("superbee", "superbee", category="limiter"),
)


# --- spatial ---------------------------------------------------------------
# The finite-volume residual is assembled by the adc::SpatialDiscretisation<Limiter,
# NumericalFlux> tag-type bundle (spatial_discretisation.hpp); there are no separate
# residual/divergence/source-assembly types, so these name that bundle.
spatial = SimpleNamespace(
    FiniteVolumeResidual=lambda **o: _native(
        "fv_residual", "adc::SpatialDiscretisation", "fv", category="spatial", **o),
    FluxDivergence=lambda **o: _native(
        "flux_divergence", "adc::SpatialDiscretisation", "fv", category="spatial", **o),
    SourceAssembly=lambda **o: _native(
        "source_assembly", "adc::SpatialDiscretisation", "fv", category="spatial", **o),
)


# --- fields (elliptic) -----------------------------------------------------
# The default Poisson coupling is solved by adc::GeometricMG (geometric_mg.hpp); there
# is no standalone adc::Poisson / Helmholtz / FieldSolver type yet.
fields = SimpleNamespace(
    Poisson=lambda **o: _planned("poisson", "poisson", category="field", **o),
    Helmholtz=lambda **o: _planned("helmholtz", "helmholtz", category="field", **o),
    EllipticSolve=lambda **o: _planned("elliptic_solve", "elliptic",
                                       category="field", **o),
    GeometricMG=lambda **o: _native("geometric_mg", "adc::GeometricMG", "geometric_mg",
                                    category="field", **o),
)


# --- solvers (linear / nonlinear) ------------------------------------------
# The matrix-free Krylov solvers are FREE FUNCTIONS in namespace adc (generic_krylov.hpp);
# Newton/FixedPoint have no standalone solver type (Newton is the implicit_stepper kernel).
def _solver(name, native_id, **o):
    return _native(name, native_id, name, category="solver", **o)


solvers = SimpleNamespace(
    CG=lambda **o: _solver("cg", "adc::cg_solve", **o),
    BiCGStab=lambda **o: _solver("bicgstab", "adc::bicgstab_solve", **o),
    GMRES=lambda **o: _solver("gmres", "adc::gmres_solve", **o),
    Richardson=lambda **o: _solver("richardson", "adc::richardson_solve", **o),
    Newton=lambda **o: _planned("newton", "newton", category="solver", **o),
    FixedPoint=lambda **o: _planned("fixed_point", "fixed_point", category="solver", **o),
    Schur=lambda **o: _solver("schur", "adc::SchurCondensationOperator", **o),
)


# --- preconditioners -------------------------------------------------------
# Only the geometric-multigrid preconditioner has a native type; identity/jacobi/
# block-jacobi have none yet (the polar solver has its own PolarPrecond enum).
preconditioners = SimpleNamespace(
    Identity=lambda: _planned("identity", "identity", category="preconditioner"),
    Jacobi=lambda: _planned("jacobi", "jacobi", category="preconditioner"),
    BlockJacobi=lambda: _planned("block_jacobi", "block_jacobi",
                                 category="preconditioner"),
    GeometricMG=lambda **o: _native("geometric_mg", "adc::GeometricMG", "geometric_mg",
                                    category="preconditioner", **o),
    User=lambda native_id, **opts: BrickDescriptor(
        native_id, "external_cpp", category="preconditioner", native_id=native_id,
        scheme="user", options=opts or None),
)


# --- diagnostics -----------------------------------------------------------
def _diag(_dname, **o):
    return BrickDescriptor(_dname, "macro", category="diagnostic", scheme=_dname,
                           options=o or None)


diagnostics = SimpleNamespace(
    integral=lambda expr=None, **o: _diag("integral", expr=expr, **o),
    norm=lambda kind="l2", **o: _diag("norm", kind=kind, **o),
    mass=lambda **o: _diag("mass", **o),
    momentum=lambda **o: _diag("momentum", **o),
    energy=lambda **o: _diag("energy", **o),
    invariant_error=lambda name=None, **o: _diag("invariant_error", name=name, **o),
    residual=lambda **o: _diag("residual", **o),
)


# --- projections -----------------------------------------------------------
# Positivity is the adc::zhang_shu_scale free function (positivity.hpp); the others have
# no native symbol yet (a generated brick or a planned native type).
projections = SimpleNamespace(
    positivity=lambda **o: _native("positivity", "adc::zhang_shu_scale", "positivity",
                                   category="projection", **o),
    bound_preserving=lambda **o: _planned("bound_preserving", "bound_preserving",
                                          category="projection", **o),
    conservative_fix=lambda **o: BrickDescriptor(
        "conservative_fix", "generated", category="projection",
        scheme="conservative_fix", options=o or None),
    divergence_cleaning=lambda **o: BrickDescriptor(
        "divergence_cleaning", "generated", category="projection",
        scheme="divergence_cleaning", options=o or None),
)


# --- invariants ------------------------------------------------------------
def _invariant(name, expression=None, over=None):
    """A catalog invariant descriptor; the value ``expression`` is kept off the
    identity key (it may be an unhashable board node) as a plain attribute."""
    return BrickDescriptor(name, "macro", category="invariant", scheme="invariant",
                           options={"over": tuple(over) if over else ()},
                           expression=expression)


invariants = SimpleNamespace(
    invariant=_invariant,
    conservation_check=lambda name, tolerance=1e-10, **o: BrickDescriptor(
        name, "macro", category="invariant", scheme="conservation_check",
        options={"tolerance": tolerance, **o}),
)


# --- time (MACRO bricks: build Program IR via adc.time.std) ----------------
def _std():
    from . import time as _time
    return _time.std


def _time_macro(std_name):
    """A macro brick that forwards to ``adc.time.std.<std_name>``; builds IR only."""
    def macro(P, block, *args, **kwargs):
        return getattr(_std(), std_name)(P, block, *args, **kwargs)
    macro.__name__ = std_name
    macro.__doc__ = "Build the %r time scheme into Program P (adc.time.std)." % std_name
    return macro


time = SimpleNamespace(
    forward_euler=_time_macro("forward_euler"),
    ssprk2=_time_macro("ssprk2"),
    ssprk3=_time_macro("ssprk3"),
    rk4=_time_macro("rk4"),
    rk=_time_macro("rk"),
    adams_bashforth=_time_macro("adams_bashforth"),
    bdf=_time_macro("bdf"),
    strang=_time_macro("strang"),
    lie=_time_macro("lie"),
    imex=_time_macro("imex_local"),
    predictor_corrector=_time_macro("predictor_corrector_local_linear"),
)
