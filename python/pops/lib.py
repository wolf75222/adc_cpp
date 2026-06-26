"""pops.lib -- a catalog of typed brick descriptors and IR macros (Spec 3).

pops.lib is NOT a Python numerics library. Every entry is one of:

* a NATIVE brick -- a descriptor naming a C++ symbol already in ``include/adc``
  (``pops.lib.riemann.HLLC()`` -> ``pops::HLLCFlux``); a catalogued brick with no native
  symbol yet carries ``available=False`` and an empty ``native_id`` (never a fake id);
* a GENERATED brick -- a descriptor of a DSL-authored brick compiled to C++;
* a MACRO brick -- a Python function that builds Program IR
  (``pops.lib.time.predictor_corrector`` delegates to :mod:`pops.time` ``std``);
* an EXTERNAL C++ brick -- a descriptor of a user ``.so`` registered by id
  (``pops.lib.riemann.User("my_hllc")``).

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
        # Optional Python builder of a GENERATED-brick solver (``@pops.lib.solver``):
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
# static-init time (the C++ ``POPS_REGISTER_BRICK`` macro -> ``BrickRegistry``) and exports
# a C ``pops_brick_manifest()`` returning JSON. ``load_cpp_library`` dlopens it, parses that
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
    """Parse a brick manifest (the JSON ``pops_brick_manifest()`` returns) and register it.

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
    ``POPS_REGISTER_BRICK`` registrations), calls the exported C function
    ``const char* pops_brick_manifest()`` to read the registered bricks as JSON, and registers
    the ids in the in-process catalog so ``riemann.User(id)`` / :func:`external` resolve. The
    ``.so`` must export ``pops_brick_manifest`` (a missing symbol is a clear ``ValueError``).
    Returns the number of bricks registered.
    """
    import ctypes
    handle = ctypes.CDLL(str(path))  # raises OSError if the path is not a loadable library
    try:
        manifest_fn = handle.pops_brick_manifest
    except AttributeError as err:
        raise ValueError("external brick library %r does not export pops_brick_manifest(); it "
                         "is not an adc brick .so" % (path,)) from err
    manifest_fn.restype = ctypes.c_char_p
    raw = manifest_fn()
    if raw is None:
        raise ValueError("external brick library %r: pops_brick_manifest() returned NULL"
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
            "external brick %r not loaded; call pops.lib.load_cpp_library(...) on the brick "
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
# live at top level in ``namespace pops`` (e.g. pops::HLLCFlux), not under a numerics/fv
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
    Rusanov=lambda: _riemann("rusanov", "pops::RusanovFlux", ["max_wave_speed"]),
    HLL=lambda: _riemann("hll", "pops::HLLFlux", ["physical_flux", "wave_speeds"]),
    HLLC=lambda: _riemann("hllc", "pops::HLLCFlux",
                          ["physical_flux", "pressure", "wave_speeds",
                           "contact_speed", "hllc_star_state"]),
    Roe=lambda: _riemann("roe", "pops::RoeFlux", ["physical_flux", "roe_average"]),
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
# pops::Weno5 IS the WENO5-Z reconstruction (it wraps weno5z()); WENO5 and WENO5Z both
# select it. MUSCL is reconstruction-by-limiter; its native limiter type is pops::Minmod.
reconstruction = SimpleNamespace(
    FirstOrder=lambda: _native("firstorder", "pops::NoSlope", "firstorder",
                               category="reconstruction"),
    MUSCL=lambda limiter="minmod": _native(
        "muscl", "pops::Minmod", limiter, category="reconstruction", limiter=limiter),
    WENO5=lambda: _native("weno5", "pops::Weno5", "weno5", category="reconstruction"),
    WENO5Z=lambda: _native("weno5z", "pops::Weno5", "weno5", category="reconstruction"),
    User=lambda brick_id: _external_descriptor(brick_id, expect_category="reconstruction"),
)


# --- limiters --------------------------------------------------------------
limiters = SimpleNamespace(
    Minmod=lambda: _native("minmod", "pops::Minmod", "minmod", category="limiter"),
    VanLeer=lambda: _native("vanleer", "pops::VanLeer", "vanleer", category="limiter"),
    # MC / Superbee are catalogued but have no native type yet (available=False).
    MC=lambda: _planned("mc", "mc", category="limiter"),
    Superbee=lambda: _planned("superbee", "superbee", category="limiter"),
)


# --- spatial ---------------------------------------------------------------
# The finite-volume residual is assembled by the pops::SpatialDiscretisation<Limiter,
# NumericalFlux> tag-type bundle (spatial_discretisation.hpp); there are no separate
# residual/divergence/source-assembly types, so these name that bundle.
spatial = SimpleNamespace(
    FiniteVolumeResidual=lambda **o: _native(
        "fv_residual", "pops::SpatialDiscretisation", "fv", category="spatial", **o),
    FluxDivergence=lambda **o: _native(
        "flux_divergence", "pops::SpatialDiscretisation", "fv", category="spatial", **o),
    SourceAssembly=lambda **o: _native(
        "source_assembly", "pops::SpatialDiscretisation", "fv", category="spatial", **o),
    # The whole finite-volume spatial brick selected per instance by the unified sim.install (Spec 3
    # section 22): it carries the runtime scheme options (riemann / reconstruction / positivity_floor)
    # that System.install lowers to the existing add_equation spatial args. ``riemann`` names the
    # NUMERICAL Riemann flux (not the model's physical flux); ``reconstruction`` is the limiter
    # (none/minmod/vanleer/weno5).
    FiniteVolume=lambda **o: _native(
        "finite_volume", "pops::SpatialDiscretisation", "fv", category="spatial", **o),
)


# --- fields (elliptic) -----------------------------------------------------
# The default Poisson coupling is solved by pops::GeometricMG (geometric_mg.hpp); there
# is no standalone pops::Poisson / Helmholtz / FieldSolver type yet.
fields = SimpleNamespace(
    Poisson=lambda **o: _planned("poisson", "poisson", category="field", **o),
    Helmholtz=lambda **o: _planned("helmholtz", "helmholtz", category="field", **o),
    EllipticSolve=lambda **o: _planned("elliptic_solve", "elliptic",
                                       category="field", **o),
    GeometricMG=lambda **o: _native("geometric_mg", "pops::GeometricMG", "geometric_mg",
                                    category="field", **o),
)


# --- solvers (linear / nonlinear) ------------------------------------------
# The matrix-free Krylov solvers are FREE FUNCTIONS in namespace pops (generic_krylov.hpp);
# Newton/FixedPoint have no standalone solver type (Newton is the implicit_stepper kernel).
def _solver(name, native_id, **o):
    return _native(name, native_id, name, category="solver", **o)


solvers = SimpleNamespace(
    CG=lambda **o: _solver("cg", "pops::cg_solve", **o),
    BiCGStab=lambda **o: _solver("bicgstab", "pops::bicgstab_solve", **o),
    GMRES=lambda **o: _solver("gmres", "pops::gmres_solve", **o),
    Richardson=lambda **o: _solver("richardson", "pops::richardson_solve", **o),
    Newton=lambda **o: _planned("newton", "newton", category="solver", **o),
    FixedPoint=lambda **o: _planned("fixed_point", "fixed_point", category="solver", **o),
    Schur=lambda **o: _solver("schur", "pops::SchurCondensationOperator", **o),
)


# --- custom solver DSL (Spec 3 section 20 / criterion 23) -------------------
# ``@pops.lib.solver`` registers a GENERATED-brick solver whose body is a Python
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
    ``scheme`` mirrors ``pops.lib.solvers.GMRES()``).

    The generated C++ lowering + run is the deferred C++ follow-up: see
    :func:`generate_solver_cpp` (it raises a clear ADC-462 ``NotImplementedError``;
    it is never faked as a Python solve).
    """
    if not isinstance(name, str) or not name:
        raise ValueError("@pops.lib.solver requires a non-empty name=")
    if signature is not None and not isinstance(signature, str):
        raise TypeError("@pops.lib.solver signature= must be a string (e.g. '(A, b)')")

    def decorate(builder):
        if not callable(builder):
            raise TypeError("@pops.lib.solver must decorate a callable builder; got %r"
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
    """Coerce a ``@pops.lib.solver`` argument to its descriptor: accept the descriptor
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
        raise ValueError("%r is not a custom (@pops.lib.solver) solver descriptor"
                         % (desc.name,))
    return desc


class SolverIR:
    """The IR authored by a custom-solver builder: an inert graph of typed ops.

    It is a thin view over the building :class:`pops.time.Program` -- it records the
    flat op list and the returned solution value. It holds NO numeric data: every
    node is a typed SSA record (see :class:`pops.time.Value`). The C++ lowering of
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

    It wraps an :class:`pops.time.Program` and exposes the primitives a solver needs
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
        :meth:`pops.time.Program.while_`). The loop is DYNAMIC (C++-side); it never iterates
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
    :meth:`pops.time.Program.while_`). It then emits one ``while`` IR node owning both blocks.
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

    @p solver_brick is a ``@pops.lib.solver`` descriptor (or its registered name). The
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


def generate_solver_cpp(solver_brick, func=None):
    """Lower a custom solver IR to GENERATED C++ that RUNS (Spec 3 section 20, criterion 23).

    @p solver_brick is a ``@pops.lib.solver`` descriptor (or its registered name). Builds the
    solver IR with :func:`build_solver_ir` and walks the SSA, emitting a self-contained C++
    free function that drives the solve entirely in C++ -- it calls the SHARED matrix-free HPC
    primitives (``pops::dot`` / ``pops::saxpy`` / ``pops::lincomb`` from ``mf_arith.hpp``) and
    runs the convergence loop as a REAL C++ ``for (;;)`` whose predicate re-evaluates each
    iteration. Returns the C++ source string (a header-only templated kernel).

    The emitted signature is::

        template <class Op>
        pops::KrylovResult <name>_solve(const Op& A, pops::MultiFab& x,
                                       const pops::MultiFab& b);

    The operator ``A`` is a value-typed TEMPLATE parameter (a callable
    ``void(pops::MultiFab&, const pops::MultiFab&)``, the same shape ``pops::ApplyFn`` and the
    native Krylov loops take), so it inlines at the call site -- there is NO ``std::function``
    in the kernel, NO Python callback in the loop, NO heap allocation inside the loop (the
    scratch fields are allocated ONCE before it), and NO per-cell string lookup (criterion
    24.9). The DSL path is for CUSTOM solvers; a DSL solver that maps onto a native scheme
    keeps the ``pops::*_solve`` free functions as its backend.
    """
    desc = _as_descriptor(solver_brick)
    ir = build_solver_ir(desc)
    return _SolverCppLowering(ir, func or desc.name).emit()


# A hard upper bound on a custom solver's convergence loop: a generated kernel MUST terminate even if
# the authored predicate never goes false (a stalled / diverging custom solver). The authored
# ``it < max_iter`` cap normally stops the loop first; this is the backstop.
_SOLVER_MAX_ITERS = 1000000


class _SolverCppLowering:
    """Walk a :class:`SolverIR` and emit a standalone, self-contained C++ solver kernel.

    The walker mirrors the proven ``pops.time.Program`` op lowering for the solver subset
    (``state`` / ``linear_source`` / ``reduce`` (norm2/dot) / ``apply`` / ``linear_combine`` /
    ``scalar_op`` / ``compare`` / ``logical_and`` / ``while``) but emits a free function that
    is NOT bound to a ProgramContext / model: the matrix-free operator is the template
    parameter ``A`` and the vector operands are bare ``pops::MultiFab`` scratch fields combined
    with the shared ``pops::dot`` / ``pops::saxpy`` / ``pops::lincomb`` primitives. It is kept in
    ``pops.lib`` (the solver-codegen path) so it does not perturb the shared time/dsl codegen."""

    def __init__(self, ir, func):
        self._ir = ir
        self._func = str(func)
        self._var = {}          # IR value id -> C++ token (a MultiFab lvalue or a scalar / bool expr)
        self._scratch = []      # MultiFab scratch field declarations (alloc-once, before the loop)
        # Iteration counters (the ``it`` of ``it = it + 1``): an SSA scalar literal cannot mutate, so
        # the authored counter is a constant in the IR. We bind the FIRST such counter to the real C++
        # loop counter ``pops_iters`` so ``it < max_iter`` is a GENUINE trip bound, not a frozen test.
        self._counter_id = self._find_counter()
        # Every scalar_op that aliases the counter (the base ``it`` and its ``it + 1`` increments,
        # transitively) lowers to the live C++ ``pops_iters``. Pre-resolved so the cond_block (emitted
        # before the body) can reference the body's counter-update id, which is the same loop counter.
        self._counter_aliases = self._counter_alias_ids()
        # The rhs ``b`` is the first State the builder receives (ctx.unknown('b')); the iterate ``x``
        # is the warm-start State the result folds back onto (ctx.zeros_like(b)). Both lower to a
        # SINGLE C++ variable each, so the in-place loop update mutates one field, the cond re-reads it.
        self._b_id = self._first_state_id()
        self._iterate_id = self._find_iterate_id()
        if self._iterate_id is None:
            # Fail loud: a result with no iterate State (other than the rhs ``b``) would otherwise
            # lower to a kernel that silently returns its input. build_solver_ir validates the
            # result upstream, so a well-formed @pops.lib.solver never reaches this.
            raise ValueError(
                "solver IR has no solution iterate to lower (the result must fold onto a "
                "warm-start State distinct from the rhs); cannot generate C++")

    def emit(self):
        """The full C++ translation unit: the templated kernel + an explicit-double ABI wrapper."""
        body = self._emit_kernel_body()
        scratch = "\n".join("  " + ln for ln in self._scratch)
        body_src = "\n".join("  " + ln for ln in body)
        return _SOLVER_CPP_TEMPLATE.format(
            name=json.dumps(self._ir.descriptor.name), func=self._func,
            scratch=scratch, body=body_src, max_iters=_SOLVER_MAX_ITERS)

    # --- top-level walk ----------------------------------------------------
    def _emit_kernel_body(self):
        """Lower the top-level SSA list into the kernel body lines (the while op recurses).

        ``r`` (the result iterate) is the function's output ``x``; the warm start (``zeros_like``)
        zeroes it. Every other field op allocates an alloc-once scratch field. Reductions /
        scalars / compares lower inline; the ``while`` op becomes a real C++ loop."""
        lines = []
        result_id = self._ir.result.id
        for v in self._ir.program._values:
            self._emit_op(v, lines, result_id)
        return lines

    def _emit_op(self, v, lines, result_id):
        """Lower a SINGLE solver-IR op to C++, appending statements to @p lines and recording its
        C++ token in ``self._var``. Shared by the top-level walk and the while sub-blocks."""
        op = v.op
        if op == "linear_source":
            # The matrix-free operator A: its token is the kernel's template-parameter callable.
            self._var[v.id] = "A"
        elif op == "state":
            # A solver unknown / rhs / warm-start iterate. The rhs ``b`` is the kernel argument; the
            # warm-start iterate (``zeros_like(b)``) IS the output ``x`` zeroed (the loop updates it in
            # place); any other bare state is an alloc-once scratch field.
            if v.id == self._b_id:
                self._var[v.id] = "b"
            elif v.id == self._iterate_id:
                self._var[v.id] = "x"
                lines.append("x.set_val(static_cast<pops::Real>(0));  // warm start: zeros_like(b)")
            else:
                self._var[v.id] = self._alloc_field(None, v)
        elif op == "reduce":
            self._emit_reduce(v, lines)
        elif op == "apply":
            self._emit_apply(v, lines)
        elif op == "linear_combine":
            self._emit_combine(v, lines, result_id)
        elif op == "scalar_op":
            self._emit_scalar_op(v, lines)
        elif op == "compare":
            self._emit_compare(v)
        elif op == "logical_and":
            a, b = v.inputs
            self._var[v.id] = "(%s && %s)" % (self._var[a.id], self._var[b.id])
        elif op == "while":
            self._emit_while(v, lines, result_id)
        else:
            raise NotImplementedError(
                "generate_solver_cpp: solver-IR op %r (value %r) is not lowerable yet; the custom "
                "solver DSL supports zeros_like / norm2 / dot / apply / residual / combine / "
                "scalar_int / logical_and / while_ (Spec 3 section 20)" % (op, v.name))

    # --- per-op lowering ---------------------------------------------------
    def _emit_reduce(self, v, lines):
        """A collective reduction -> a C++ scalar local. ``norm2 = sqrt(pops::dot(u,u))``;
        ``dot`` calls ``pops::dot`` directly (the SAME shared primitive the native Krylov uses)."""
        tok = "s%d" % v.id
        self._var[v.id] = tok
        kind = v.attrs["kind"]
        if kind == "norm2":
            (u,) = v.inputs
            lines.append("const pops::Real %s = std::sqrt(pops::dot(%s, %s));"
                         % (tok, self._var[u.id], self._var[u.id]))
        elif kind == "dot":
            a, b = v.inputs
            lines.append("const pops::Real %s = pops::dot(%s, %s);"
                         % (tok, self._var[a.id], self._var[b.id]))
        else:
            raise NotImplementedError(
                "generate_solver_cpp: reduction kind %r is not lowerable in a custom solver "
                "(use norm2 / dot)" % (kind,))

    def _emit_apply(self, v, lines):
        """``A(x)`` -> the matrix-free matvec: call the template-parameter operator into an
        alloc-once scratch field. NO std::function -- ``A`` is the inlined template callable."""
        operator = v.inputs[0]
        state = v.inputs[1] if len(v.inputs) > 1 else None
        # The solver context records apply inputs as (state[, fields]); the operator name is in
        # attrs. The first State input is the vector A is applied to.
        x_in = next((inp for inp in v.inputs if getattr(inp, "vtype", None) == "state"), state)
        out = self._alloc_field(None, v)
        self._var[v.id] = out
        lines.append("A(%s, %s);" % (out, self._var[x_in.id]))
        _ = operator  # the operator is the single template callable A (no per-name dispatch)

    def _emit_combine(self, v, lines, result_id):
        """An affine combine ``sum_k c_k * field_k`` -> a scratch field (or the in-place iterate
        update). ``c_k`` are IR literal coefficients.

        The Richardson update ``x <- x + omega*r`` is the in-place iterate write: its term ``x`` is
        the loop variable (the kernel output ``x``), so the combine TARGETS ``x`` and keeps its
        content, adding only the other terms with ``pops::saxpy``. The residual ``r = b - A x``
        (whose terms are ``b`` / ``A(x)``, neither the iterate) targets a fresh alloc-once scratch
        that is zeroed first. So both the warm-start state ``#result`` and every in-place update map
        to the single token ``x`` -- the loop's cond re-reads ``A(x)`` on the updated iterate."""
        terms = list(zip(v.inputs, v.attrs["coeffs"], strict=True))
        # The combine carries the iterate iff one of its terms IS the iterate token "x" (the
        # x + omega*r update) or it is the returned warm-start itself.
        carries_iterate = v.id == result_id or any(self._var[inp.id] == "x" for inp, _ in terms)
        target = "x" if carries_iterate else self._alloc_field(None, v)
        self._var[v.id] = target
        self_term = next((c for inp, c in terms if self._var[inp.id] == target), None)
        if self_term is None:
            lines.append("%s.set_val(static_cast<pops::Real>(0));" % target)
        else:
            scale = float(self_term.get(0, 0.0))
            if scale != 1.0:
                lines.append("pops::saxpy(%s, static_cast<pops::Real>(%s), %s);  // scale self term"
                             % (target, repr(scale - 1.0), target))
        for inp, coeff in terms:
            tok = self._var[inp.id]
            if tok == target:
                continue
            lines.append("pops::saxpy(%s, static_cast<pops::Real>(%s), %s);"
                         % (target, repr(float(coeff.get(0, 0.0))), tok))

    def _emit_scalar_op(self, v, lines):
        """Scalar arithmetic (add/sub/mul/div). The bound iteration counter lowers to the real C++
        loop counter ``pops_iters`` (so ``it < max_iter`` is a genuine trip bound); the counter's
        ``it = it + 1`` increment is the loop's own ``++pops_iters`` and emits nothing. A pure-literal
        ``scalar_int(n)`` collapses to its compile-time value; any other scalar arithmetic lowers to
        a C++ expression over its operands."""
        # The counter and its increments alias the live C++ ``pops_iters`` (no statement emitted).
        if v.id in self._counter_aliases:
            self._var[v.id] = "static_cast<pops::Real>(pops_iters)"
            return
        operands = v.attrs["operands"]
        fn = v.attrs["fn"]
        # A literal-only scalar_int (scalar_int(n) is built as n + 0.0): fold to its constant value.
        if all(kind == "c" for kind, _ in operands):
            self._var[v.id] = "static_cast<pops::Real>(%s)" % repr(_fold_scalar_literal(operands, fn))
            return
        toks = []
        for kind, val in operands:
            if kind == "v":
                toks.append(self._var[v.inputs[val].id])
            else:
                toks.append("static_cast<pops::Real>(%s)" % repr(float(val)))
        cppop = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[fn]
        self._var[v.id] = "(%s %s %s)" % (toks[0], cppop, toks[1])

    def _counter_alias_ids(self):
        """The set of scalar_op ids that alias the loop counter: the base ``it`` plus every
        ``it + literal`` increment (transitively). All lower to the live C++ ``pops_iters``."""
        if self._counter_id is None:
            return frozenset()
        aliases = {self._counter_id}
        nodes = list(_iter_all_nodes(self._ir.program._values))
        changed = True
        while changed:
            changed = False
            for v in nodes:
                if v.op != "scalar_op" or v.id in aliases:
                    continue
                reads = [v.inputs[idx].id for kind, idx in v.attrs["operands"] if kind == "v"]
                if any(rid in aliases for rid in reads):
                    aliases.add(v.id)
                    changed = True
        return frozenset(aliases)

    def _find_counter(self):
        """The IR id of the iteration counter: a top-level literal ``scalar_int`` (built as ``n+0``)
        that a body ``scalar_op`` reads to increment it (the ``it = it + scalar_int(1)`` idiom). Bound
        to the live C++ loop counter so the authored ``it < max_iter`` cap actually bounds the
        generated loop. ``None`` if the solver authors no counter (the loop then relies on the
        residual test / the hard safety cap to terminate)."""
        prog = self._ir.program
        # A counter SEED is a pure-literal scalar_op declared at the top level (before the loop).
        seeds = {v.id for v in prog._values
                 if v.op == "scalar_op" and all(k == "c" for k, _ in v.attrs["operands"])}
        # The counter is a seed that some OTHER scalar_op reads (its increment); a bare unread literal
        # (e.g. a max_iter constant compared directly) is not a counter.
        for v in _iter_all_nodes(prog._values):
            if v.op != "scalar_op":
                continue
            reads = [v.inputs[idx].id for kind, idx in v.attrs["operands"] if kind == "v"]
            for rid in reads:
                if rid in seeds:
                    return rid
        return None

    def _emit_compare(self, v):
        """A scalar predicate -> an inline C++ boolean expression (no statement; the while op embeds
        it in ``if (!(<expr>)) break;``)."""
        lhs = v.inputs[0]
        if len(v.inputs) == 2:
            rhs_tok = self._var[v.inputs[1].id]
        else:
            rhs_tok = "static_cast<pops::Real>(%s)" % repr(float(v.attrs["rhs"]))
        self._var[v.id] = "(%s %s %s)" % (self._var[lhs.id], v.attrs["cmp"], rhs_tok)

    def _emit_while(self, v, lines, result_id):
        """Lower the convergence loop to a REAL C++ ``for (;;)`` with a break: the cond sub-block is
        re-emitted each pass (the predicate re-evaluates against the loop-updated iterate), then the
        body sub-block runs. The C++ loop counter ``pops_iters`` is the authored iteration counter (so
        the author's ``it < max_iter`` cap really bounds the loop); a hard safety cap also breaks the
        loop so a stalled custom solver can never spin forever even if its IR omits a cap."""
        # The convergence predicate (cond_block) was RECORDED AFTER the body so it reads the
        # loop-updated iterate -- in the IR it references the body's iterate-update State (result_id).
        # In the C++ loop the iterate is the single in-place variable ``x``, so pre-bind the body's
        # iterate-update id to ``x``: the cond (emitted first, runs each pass against the prior body's
        # x) and the body (which writes x in place) then both resolve to the one loop variable.
        self._var[result_id] = "x"
        # Pre-bind every counter alias to the live loop counter, so the cond_block (emitted before the
        # body) can reference the body's counter-update id -- they are all the one C++ ``pops_iters``.
        for cid in self._counter_aliases:
            self._var[cid] = "static_cast<pops::Real>(pops_iters)"
        lines.append("pops_iters = 0;")  # function-scope counter (declared in the kernel template)
        lines.append("for (;; ++pops_iters) {")
        body = ["if (pops_iters >= %d) break;  // hard safety cap (custom solver)" % _SOLVER_MAX_ITERS]
        for w in v.attrs["cond_block"]:
            self._emit_op(w, body, result_id)
        cond_expr = self._var[v.attrs["cond"].id]
        body.append("if (!(%s)) break;" % cond_expr)
        for w in v.attrs["body_block"]:
            self._emit_op(w, body, result_id)
        lines += ["  " + ln for ln in body]
        lines.append("}")

    # --- helpers -----------------------------------------------------------
    def _alloc_field(self, fixed, v):
        """The C++ token of a vector field for IR value @p v. ``fixed="x"`` is the kernel output (no
        new scratch); otherwise allocate an ALLOC-ONCE scratch field shaped like the rhs ``b`` (one
        ghost, ncomp(b)) before the loop -- never inside it (no heap churn in the kernel)."""
        if fixed == "x":
            return "x"
        tok = "v%d" % v.id
        self._scratch.append("pops::MultiFab %s(b.box_array(), b.dmap(), b.ncomp(), 1);" % tok)
        return tok

    def _first_state_id(self):
        """The id of the rhs ``b`` IR State: the first State the builder receives (build_solver_ir
        binds it as ``ctx.unknown('b')`` right after the operator linear_source 'A')."""
        for v in self._ir.program._values:
            if v.op == "state":
                return v.id
        return None

    def _find_iterate_id(self):
        """The id of the solution iterate ``x``: the warm-start State the result folds back onto. The
        result is either a State (the warm start returned directly) or a ``linear_combine`` whose State
        input (not the rhs ``b``) is the iterate base (the ``x`` of ``x + omega*r``)."""
        res = self._ir.result
        if res.op == "state":
            return res.id
        # Walk the result combine's State inputs for the non-``b`` State (the iterate base).
        seen = set()
        frontier = [res]
        while frontier:
            node = frontier.pop()
            if node.id in seen:
                continue
            seen.add(node.id)
            if node.op == "state" and node.id != self._b_id:
                return node.id
            for inp in node.inputs:
                if getattr(inp, "vtype", None) == "state":
                    frontier.append(inp)
        return None


def _fold_scalar_literal(operands, fn):
    """Fold a pure-literal scalar_op (the ``scalar_int(n) = n + 0.0`` idiom) to its constant float."""
    a = float(operands[0][1])
    b = float(operands[1][1]) if len(operands) > 1 else 0.0
    if fn == "add":
        return a + b
    if fn == "sub":
        return a - b
    if fn == "mul":
        return a * b
    return a / b


def _iter_all_nodes(values):
    """Yield @p values and the ops in any ``while`` cond/body sub-blocks, depth-first (build order)."""
    for v in values:
        yield v
        for key in ("cond_block", "body_block"):
            blk = v.attrs.get(key) if hasattr(v, "attrs") else None
            if isinstance(blk, list):
                yield from _iter_all_nodes(blk)


_SOLVER_CPP_TEMPLATE = '''\
// GENERATED by pops.lib.generate_solver_cpp (epic ADC-462, Spec 3 section 20 / criterion 23).
// Do not edit by hand. A CUSTOM solver authored in the @pops.lib.solver IR-DSL, lowered to a
// self-contained C++ kernel that drives the solve entirely in C++ via the SHARED matrix-free
// HPC primitives (pops::dot / pops::saxpy / pops::lincomb). The matrix-free operator A is a
// value-typed TEMPLATE parameter (a callable void(MultiFab&, const MultiFab&), the same shape
// the native Krylov loops take), so it INLINES at the call site: no type-erased indirection in
// the kernel, no Python callback in the loop, no heap allocation inside the loop (the scratch
// fields below are allocated once, before it), and no per-cell name dispatch (criterion 24.9).
#include <pops/mesh/storage/multifab.hpp>                     // pops::MultiFab
#include <pops/mesh/storage/mf_arith.hpp>                     // pops::dot / pops::saxpy / pops::lincomb
#include <pops/numerics/elliptic/linear/krylov_result.hpp>    // pops::KrylovResult
#include <pops/core/foundation/types.hpp>                     // pops::Real
#include <cmath>                                             // std::sqrt in lowered norms

// The custom solver kernel: solve A x = b, writing the solution into x (warm-started from x).
// Returns the iteration count / final relative residual in an pops::KrylovResult.
template <class Op>
inline pops::KrylovResult {func}_solve(const Op& A, pops::MultiFab& x, const pops::MultiFab& b) {{
  int pops_iters = 0;  // convergence-loop counter (0 for a loop-free solver)
{scratch}
{body}
  // Final relative residual ||b - A x|| / ||b|| over the SAME shared primitives (a diagnostic, once,
  // after the loop): A(x) into a scratch, r = b - A x, then the global L2 norms via pops::dot.
  pops::MultiFab pops_resid(b.box_array(), b.dmap(), b.ncomp(), 1);
  A(pops_resid, x);
  pops::lincomb(pops_resid, static_cast<pops::Real>(1), b, static_cast<pops::Real>(-1), pops_resid);
  const pops::Real pops_bnorm = std::sqrt(pops::dot(b, b));
  const pops::Real pops_rnorm = std::sqrt(pops::dot(pops_resid, pops_resid));
  pops::KrylovResult res;
  res.iters = pops_iters;
  res.rel_residual = pops_bnorm > pops::Real(0) ? pops_rnorm / pops_bnorm : pops_rnorm;
  res.converged = pops_iters < {max_iters};
  return res;
}}
'''


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
    GeometricMG=lambda **o: _native("geometric_mg", "pops::GeometricMG", "geometric_mg",
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
# Positivity is the pops::zhang_shu_scale free function (positivity.hpp); the others have
# no native symbol yet (a generated brick or a planned native type).
projections = SimpleNamespace(
    positivity=lambda **o: _native("positivity", "pops::zhang_shu_scale", "positivity",
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


# --- time (MACRO bricks: build Program IR via pops.time.std) ----------------
def _std():
    from . import time as _time
    return _time.std


def _time_macro(std_name):
    """A macro brick that forwards to ``pops.time.std.<std_name>``; builds IR only."""
    def macro(P, block, *args, **kwargs):
        return getattr(_std(), std_name)(P, block, *args, **kwargs)
    macro.__name__ = std_name
    macro.__doc__ = "Build the %r time scheme into Program P (pops.time.std)." % std_name
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
