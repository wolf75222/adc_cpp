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
import json
from types import SimpleNamespace

__all__ = ["BrickDescriptor", "riemann", "reconstruction", "limiters", "spatial",
           "fields", "solvers", "preconditioners", "diagnostics", "projections",
           "invariants", "time", "solver", "build_solver_ir", "generate_solver_cpp",
           "SolverContext", "SolverIR", "load_cpp_library", "external"]

BRICK_TYPES = ("native", "generated", "macro", "external_cpp")


class BrickDescriptor:
    """A typed, numerics-free descriptor of a numerical brick.

    Identity is by all metadata fields so two descriptors of the same brick
    compare equal (used to detect a re-selected brick and to key the artifact
    hash). It is intentionally inert: it has no ``eval`` / ``compile`` / call.
    """

    def __init__(self, name, brick_type, *, category="brick", native_id="",
                 scheme=None, requirements=None, capabilities=None, options=None,
                 available=True, expression=None, builder=None):
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
        # Optional Python builder of a GENERATED-brick solver (``@adc.lib.solver``):
        # the function that AUTHORS the solver IR. Like ``expression`` it is kept OFF
        # the identity key (a callable is not part of the brick's value identity).
        self.builder = builder

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


# --- external C++ bricks (Spec 3 section 21-22 / criterion 20) -------------
# A user ships a brick in a standalone ``.so`` that registers a manifest entry at
# static-init time (the C++ ``ADC_REGISTER_BRICK`` macro -> ``BrickRegistry``) and exports
# a C ``adc_brick_manifest()`` returning JSON. ``load_cpp_library`` dlopens it, parses that
# JSON and registers the ids in this in-process catalog; ``riemann.User(id)`` /
# ``external(id)`` then surface an ``external_cpp`` descriptor carrying the manifest's
# requirements/capabilities. An id that was never loaded raises a clear error -- a
# descriptor is NEVER fabricated for an unregistered brick.
_EXTERNAL_BRICKS = {}


def _clear_external_catalog():
    """Drop every loaded external brick (test isolation; not part of the public API)."""
    _EXTERNAL_BRICKS.clear()


def _split_csv(value):
    """Split a manifest CSV field into a stripped, non-empty token list ([] when absent)."""
    if value is None:
        return []
    if not isinstance(value, str):
        raise ValueError("manifest requirements/capabilities must be a CSV string; got %r"
                         % (value,))
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def _register_manifest(manifest_json):
    """Parse a brick manifest (the JSON ``adc_brick_manifest()`` returns) and register it.

    The manifest is ``{"bricks": [{"id", "category", "requirements", "capabilities"}, ...]}``
    where ``requirements``/``capabilities`` are optional CSV strings. Each entry's id is
    registered in the in-process catalog (last load wins on a repeated id). Returns the number
    of bricks registered. A malformed manifest or an entry missing its id raises ``ValueError``
    rather than silently drop a brick. This is the seam ``load_cpp_library`` calls after dlopen;
    it is also usable directly (a test does not need a compiled ``.so``).
    """
    try:
        doc = json.loads(manifest_json)
    except (json.JSONDecodeError, TypeError) as err:
        raise ValueError("external brick manifest is not valid JSON: %s" % (err,)) from err
    bricks = doc.get("bricks") if isinstance(doc, dict) else None
    if not isinstance(bricks, list):
        raise ValueError("external brick manifest must be {\"bricks\": [...]}; got %r"
                         % (manifest_json,))
    count = 0
    for entry in bricks:
        if not isinstance(entry, dict) or not entry.get("id"):
            raise ValueError("external brick manifest entry must carry a non-empty 'id'; "
                             "got %r" % (entry,))
        brick_id = str(entry["id"])
        _EXTERNAL_BRICKS[brick_id] = {
            "id": brick_id,
            "category": str(entry.get("category") or "brick"),
            "requirements": _split_csv(entry.get("requirements")),
            "capabilities": _split_csv(entry.get("capabilities")),
        }
        count += 1
    return count


def load_cpp_library(path):
    """Load an external C++ brick ``.so`` and register the bricks it manifests (criterion 20).

    Opens @p path with :func:`ctypes.CDLL` (its static initializers run the
    ``ADC_REGISTER_BRICK`` registrations), calls the exported C function
    ``const char* adc_brick_manifest()`` to read the registered bricks as JSON, and registers
    the ids in the in-process catalog so ``riemann.User(id)`` / :func:`external` resolve. The
    ``.so`` must export ``adc_brick_manifest`` (a missing symbol is a clear ``ValueError``).
    Returns the number of bricks registered.
    """
    import ctypes
    handle = ctypes.CDLL(str(path))  # raises OSError if the path is not a loadable library
    try:
        manifest_fn = handle.adc_brick_manifest
    except AttributeError as err:
        raise ValueError("external brick library %r does not export adc_brick_manifest(); it "
                         "is not an adc brick .so" % (path,)) from err
    manifest_fn.restype = ctypes.c_char_p
    raw = manifest_fn()
    if raw is None:
        raise ValueError("external brick library %r: adc_brick_manifest() returned NULL"
                         % (path,))
    return _register_manifest(raw.decode("utf-8"))


def _external_descriptor(brick_id, *, expect_category=None):
    """The ``external_cpp`` descriptor for a loaded brick @p brick_id (raise if not loaded).

    An unloaded id raises :class:`LookupError` naming the id and :func:`load_cpp_library`; a
    category mismatch (selecting via ``riemann.User`` a brick registered as a preconditioner)
    raises :class:`ValueError`. The manifest requirements/capabilities become list metadata on
    the descriptor (mirroring the native bricks' ``requirements={"capabilities": [...]}``).
    """
    entry = _EXTERNAL_BRICKS.get(str(brick_id))
    if entry is None:
        raise LookupError(
            "external brick %r not loaded; call adc.lib.load_cpp_library(...) on the brick "
            ".so first (loaded: %s)" % (brick_id, sorted(_EXTERNAL_BRICKS) or "none"))
    if expect_category is not None and entry["category"] != expect_category:
        raise ValueError("external brick %r is registered as category %r, not %r"
                         % (brick_id, entry["category"], expect_category))
    req = {"capabilities": list(entry["requirements"])} if entry["requirements"] else {}
    caps = {"provides": list(entry["capabilities"])} if entry["capabilities"] else {}
    return BrickDescriptor(entry["id"], "external_cpp", category=entry["category"],
                           native_id=entry["id"], scheme="user",
                           requirements=req or None, capabilities=caps or None)


def external(brick_id):
    """An ``external_cpp`` descriptor for a loaded brick of ANY category (criterion 20).

    The category-agnostic counterpart of ``riemann.User`` / ``preconditioner.User``: it surfaces
    whatever category the manifest registered. An unloaded id raises a clear :class:`LookupError`.
    """
    return _external_descriptor(brick_id)


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
    User=lambda brick_id: _external_descriptor(brick_id, expect_category="riemann"),
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
    User=lambda brick_id: _external_descriptor(brick_id, expect_category="reconstruction"),
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
    # The whole finite-volume spatial brick selected per instance by the unified sim.install (Spec 3
    # section 22): it carries the runtime scheme options (riemann / reconstruction / positivity_floor)
    # that System.install lowers to the existing add_equation spatial args. ``riemann`` names the
    # NUMERICAL Riemann flux (not the model's physical flux); ``reconstruction`` is the limiter
    # (none/minmod/vanleer/weno5).
    FiniteVolume=lambda **o: _native(
        "finite_volume", "adc::SpatialDiscretisation", "fv", category="spatial", **o),
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


# --- custom solver DSL (Spec 3 section 20 / criterion 23) -------------------
# ``@adc.lib.solver`` registers a GENERATED-brick solver whose body is a Python
# builder that AUTHORS a solver IR (matrix-free Krylov primitives). The builder
# computes nothing in Python; the IR lowers to C++ -- that lowering is the
# deferred follow-up (``generate_solver_cpp`` raises a clear ADC-462 error).
_CUSTOM_SOLVERS = {}


def solver(name=None, signature=None):
    """Register a custom solver written in the IR-authoring DSL (criterion 23).

    Decorates a builder ``f(ctx, *args)`` that AUTHORS a solver IR using the
    matrix-free Krylov primitives of :class:`SolverContext` (``ctx.norm2`` /
    ``ctx.dot`` / ``ctx.residual`` / affine ``x + omega*r`` / ``ctx.while_``).
    The builder builds IR ONLY -- it never performs Python numerics. Returns a
    ``generated`` :class:`BrickDescriptor` in the ``solver`` category, carrying the
    builder off its identity key, selectable wherever a native solver is (its
    ``scheme`` mirrors ``adc.lib.solvers.GMRES()``).

    The generated C++ lowering + run is the deferred C++ follow-up: see
    :func:`generate_solver_cpp` (it raises a clear ADC-462 ``NotImplementedError``;
    it is never faked as a Python solve).
    """
    if not isinstance(name, str) or not name:
        raise ValueError("@adc.lib.solver requires a non-empty name=")
    if signature is not None and not isinstance(signature, str):
        raise TypeError("@adc.lib.solver signature= must be a string (e.g. '(A, b)')")

    def decorate(builder):
        if not callable(builder):
            raise TypeError("@adc.lib.solver must decorate a callable builder; got %r"
                            % (builder,))
        opts = {"signature": signature} if signature is not None else None
        desc = BrickDescriptor(name, "generated", category="solver", scheme=name,
                               options=opts, builder=builder)
        _CUSTOM_SOLVERS[name] = desc
        return desc

    return decorate


def _custom_solver(name):
    """The registered custom-solver descriptor named @p name (KeyError if absent)."""
    return _CUSTOM_SOLVERS[name]


def _registered_solvers():
    """The names of the registered custom solvers (registration order)."""
    return list(_CUSTOM_SOLVERS)


solvers.custom = _custom_solver
solvers.registered = _registered_solvers


def _as_descriptor(solver_brick):
    """Coerce a ``@adc.lib.solver`` argument to its descriptor: accept the descriptor
    itself or a registered name. A non-generated/non-solver brick is rejected loud."""
    if isinstance(solver_brick, BrickDescriptor):
        desc = solver_brick
    elif isinstance(solver_brick, str):
        desc = _CUSTOM_SOLVERS.get(solver_brick)
        if desc is None:
            raise KeyError("no custom solver named %r is registered" % (solver_brick,))
    else:
        raise TypeError("expected a custom solver descriptor or its name; got %r"
                        % (solver_brick,))
    if desc.brick_type != "generated" or desc.category != "solver" or desc.builder is None:
        raise ValueError("%r is not a custom (@adc.lib.solver) solver descriptor"
                         % (desc.name,))
    return desc


class SolverIR:
    """The IR authored by a custom-solver builder: an inert graph of typed ops.

    It is a thin view over the building :class:`adc.time.Program` -- it records the
    flat op list and the returned solution value. It holds NO numeric data: every
    node is a typed SSA record (see :class:`adc.time.Value`). The C++ lowering of
    this IR is deferred (ADC-462); :func:`generate_solver_cpp` raises rather than
    fake a Python solve.
    """

    def __init__(self, descriptor, program, result):
        self.descriptor = descriptor
        self.program = program
        self.result = result

    def nodes(self):
        """The IR value nodes the builder authored, including control-flow body ops.

        Walks the flat SSA list AND the recorded ``cond``/``body`` sub-blocks of ``while``
        nodes (those blocks are owned by the op, not the top-level list), in build order."""
        out = []
        _walk_nodes(self.program._values, out)
        return out

    def op_kinds(self):
        """The set of op kinds present in the IR (e.g. ``norm2`` / ``apply`` / ``while``)."""
        kinds = set()
        for node in self.nodes():
            kinds.add(node.attrs.get("kind", node.op))
        return kinds

    def __repr__(self):
        return "SolverIR(%r, nodes=%d)" % (self.descriptor.name, len(self.program._values))


class SolverContext:
    """The matrix-free Krylov authoring context handed to a custom-solver builder.

    It wraps an :class:`adc.time.Program` and exposes the primitives a solver needs
    -- ``norm2`` / ``dot`` / ``scalar_int`` / ``logical_and`` / ``while_`` (a context
    manager) / operator apply / affine ``x + omega*r``. Every primitive BUILDS an IR
    node and returns an IR value; NOTHING is computed in Python. The unknown ``x``,
    the residual ``r`` and the operator apply ``A(x)`` are IR values, not arrays.
    """

    def __init__(self, program, block="solve"):
        self._p = program
        self._block = block

    # --- operands -----------------------------------------------------------
    def unknown(self, name=None):
        """A fresh solver unknown (the iterate ``x`` / the rhs ``b``): a State IR value."""
        return self._p.state(self._block)

    def zeros_like(self, value):
        """A zero-initialized iterate over the same block as @p value (the warm start)."""
        _require_field(value, "zeros_like")
        return self._p.state(value.block or self._block)

    def scalar_int(self, n):
        """A COMPILE-TIME integer literal as a Scalar IR value (a loop count / index). It
        is an IR node, never a live Python counter the loop mutates."""
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError("scalar_int expects a Python int; got %r" % (n,))
        return self._p._scalar_binop(float(n), 0.0, "add")

    # --- reductions ---------------------------------------------------------
    def norm2(self, x):
        """The Euclidean norm ``||x||_2`` as a Scalar IR value (a collective reduction)."""
        return self._p.norm2(x)

    def dot(self, a, b):
        """The inner product ``<a, b>`` as a Scalar IR value (a collective reduction)."""
        return self._p.dot(a, b)

    # --- operator apply / residual ------------------------------------------
    def apply(self, operator, x):
        """Apply the matrix-free operator ``A(x)`` as an IR node (an RHS-like value)."""
        if not (hasattr(x, "vtype") and x.vtype == "state"):
            raise TypeError("apply: x must be a State IR value")
        return self._p.apply(operator=_operator_name(operator), state=x)

    def residual(self, operator, x, b):
        """The residual ``r = b - A(x)`` as an affine IR combine (no Python math)."""
        ax = self.apply(operator, x)
        return self._p.linear_combine(expr=b - ax)

    def combine(self, expr):
        """Materialize an affine IR expression (e.g. ``x + omega*r``) into a State IR node.

        The affine ``x + omega*r`` is a deferred IR expression; this records it as one
        ``linear_combine`` node (the next iterate). ``omega`` is an IR literal coefficient,
        never multiplied against data in Python."""
        return self._p.linear_combine(expr=expr)

    # --- predicates ---------------------------------------------------------
    def logical_and(self, a, b):
        """The conjunction of two Bool predicates as a Bool IR node (re-evaluated each
        loop pass). Builds an ``and`` node; it never short-circuits in Python."""
        for nm, val in (("a", a), ("b", b)):
            if not (hasattr(val, "vtype") and val.vtype == "bool"):
                raise TypeError("logical_and: %s must be a Bool IR value" % nm)
        return self._p._new("bool", "logical_and", (a, b), {}, None, a.block)

    # --- control flow -------------------------------------------------------
    def while_(self, cond_fn):
        """A convergence loop as a context manager: ``with ctx.while_(cond_fn):`` records the
        loop body, then RE-EVALUATES the convergence predicate against the loop-updated
        iterate and emits one IR ``while`` node owning both blocks.

        @p cond_fn is a zero-argument builder that BUILDS a Bool IR value each time it is
        called (e.g. ``lambda: ctx.norm2(ctx.residual(A, x, b)) > tol``). It is recorded
        into a SEPARATE ``cond_block`` after the body, so the predicate references the
        mutated iterate -- not the pre-loop ``x`` -- and re-runs every pass (mirroring
        :meth:`adc.time.Program.while_`). The loop is DYNAMIC (C++-side); it never iterates
        in Python.

        Wiring the predicate to a pre-built Bool value would freeze it on the initial
        iterate (a constant convergence test), so a bare Bool value is rejected loud."""
        if not callable(cond_fn):
            raise TypeError(
                "while_: condition must be a zero-argument builder that BUILDS the Bool "
                "predicate against the loop-updated iterate (e.g. "
                "lambda: ctx.norm2(ctx.residual(A, x, b)) > tol), not a pre-built Bool value "
                "(that would freeze the test on the initial iterate)")
        return _SolverWhile(self._p, cond_fn, self._block)


class _SolverWhile:
    """The context manager :meth:`SolverContext.while_` returns: it records the loop body
    ops into a sub-block and, on exit, RE-RECORDS the convergence predicate into a separate
    ``cond_block`` so it references the loop-updated iterate (mirroring
    :meth:`adc.time.Program.while_`). It then emits one ``while`` IR node owning both blocks.
    The blocks are re-emitted inside the generated C++ loop; they are never replayed in
    Python."""

    def __init__(self, program, cond_fn, block):
        self._p = program
        self._cond_fn = cond_fn
        self._block = block
        self._body = None

    def __enter__(self):
        self._body = []
        self._p._recording.append(self._body)
        return self

    def __exit__(self, exc_type, exc, tb):
        self._p._recording.pop()
        if exc_type is not None:
            return False
        # Record the predicate AFTER the body so it builds against the loop-updated iterate.
        # Its ops live in a separate cond_block (re-run each pass), not the body block.
        cond_block = []
        self._p._recording.append(cond_block)
        try:
            cond = self._cond_fn()
        finally:
            self._p._recording.pop()
        if not (hasattr(cond, "vtype") and cond.vtype == "bool"):
            raise TypeError("while_: the condition builder must return a Bool IR value "
                            "(e.g. ctx.norm2(r) > tol); got %r" % (cond,))
        self._p._new("state", "while", (),
                     {"cond_block": cond_block, "cond": cond, "body_block": self._body},
                     None, self._block)
        return False


def build_solver_ir(solver_brick):
    """Run a custom-solver builder to AUTHOR its IR (no Python numerics).

    @p solver_brick is a ``@adc.lib.solver`` descriptor (or its registered name). The
    builder receives a :class:`SolverContext` and two unknowns (the operator ``A`` and
    the rhs ``b``) and returns the solution IR value. Returns a :class:`SolverIR`.
    """
    from . import time as _time
    desc = _as_descriptor(solver_brick)
    program = _time.Program("solver_" + desc.name)
    ctx = SolverContext(program)
    a_op = program.linear_source("A")    # the matrix-free operator A, an IR operator value
    b_rhs = ctx.unknown("b")             # the right-hand side b, an IR State value
    result = desc.builder(ctx, a_op, b_rhs)
    # A builder may return an affine expression (``x + omega*r``); materialize it into a
    # State IR node so the solution is always a recorded value, never a deferred Python expr.
    if not (hasattr(result, "vtype")) and result is not None:
        result = ctx.combine(result)
    if not (hasattr(result, "vtype") and result.vtype == "state"):
        raise ValueError("a custom solver builder must return the solution State IR value; "
                         "got %r" % (result,))
    return SolverIR(desc, program, result)


def generate_solver_cpp(solver_brick):
    """Lower a custom solver IR to C++ -- DEFERRED to the ADC-462 C++ follow-up.

    The IR-authoring slice lands the DSL and its IR; emitting the generated C++ kernel
    and running it is the deferred C++ work. This raises a clear ADC-462 error rather
    than silently no-op or fake a Python solve.
    """
    desc = _as_descriptor(solver_brick)
    raise NotImplementedError(
        "ADC-462: the custom solver DSL authors IR; lowering %r to generated C++ is the "
        "deferred C++ follow-up (the IR-authoring slice does not generate or run the "
        "solver). Build the IR with adc.lib.build_solver_ir() instead." % (desc.name,))


def _walk_nodes(values, out):
    """Append @p values and any ops recorded in their ``while`` cond/body sub-blocks to @p
    out, depth-first in build order (a loop's cond and body blocks are owned by its op, not
    the flat list). The cond block is walked too so the re-evaluated convergence predicate's
    ops (its ``residual`` / ``apply`` over the loop-updated iterate) are visible."""
    for node in values:
        out.append(node)
        attrs = node.attrs if hasattr(node, "attrs") else {}
        for key in ("cond_block", "body_block"):
            block = attrs.get(key)
            if isinstance(block, list):
                _walk_nodes(block, out)


def _require_field(value, where):
    if not (hasattr(value, "is_field") and value.is_field()):
        raise TypeError("%s: a State/RHS IR value is required; got %r" % (where, value))


def _operator_name(operator):
    """The linear-source name of a matrix-free operator IR value (or a bare string)."""
    if isinstance(operator, str):
        return operator
    name = getattr(operator, "attrs", {}).get("linear_source") if hasattr(operator, "attrs") else None
    if name is None:
        raise TypeError("apply: operator must be a linear-source IR value or a name; got %r"
                        % (operator,))
    return name


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
    User=lambda brick_id: _external_descriptor(brick_id, expect_category="preconditioner"),
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
