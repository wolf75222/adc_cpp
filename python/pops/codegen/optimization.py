"""pops.codegen.optimization -- typed codegen optimization policy (Spec 5 sec.13.8).

The optimization policy is a typed object, not a string ``optimization="fast"``. It selects
which IR / expression transforms the codegen may apply (CSE, dead-node elimination,
redundant-solve elimination, local fusion, reciprocal hoisting) and the numeric math mode
(:mod:`pops.codegen.math_options`). It is inert: it describes + inspects the choices; the
codegen consumes it. A non-strict numeric transform is never implicit -- it must be selected
here (Spec 5 sec.13.10).
"""
from pops.descriptors import Descriptor, reject_string_selector
from .math_options import StrictMath, _MathMode


class _Fusion(Descriptor):
    category = "fusion_policy"


class Disabled(_Fusion):
    """No kernel fusion / no fast-math (explicit off-switch)."""

    def options(self):
        return {"enabled": False}


class ConservativeFusion(_Fusion):
    """Fuse only safe local operations; never across a solve / reduction / halo refresh."""

    def __init__(self, local_sources=True, projections=True,
                 flux_divergence=False, field_solves=False):
        self.local_sources = bool(local_sources)
        self.projections = bool(projections)
        self.flux_divergence = bool(flux_divergence)
        self.field_solves = bool(field_solves)

    def options(self):
        return {"local_sources": self.local_sources, "projections": self.projections,
                "flux_divergence": self.flux_divergence, "field_solves": self.field_solves}

    def capabilities(self):
        # Fusion across an elliptic solve / global reduction / halo refresh is never allowed.
        return {"crosses_solve": False, "crosses_reduction": False}


_MATH_SUGGEST = ('a typed StrictMath()/FastMath()/DebugMath()/GpuRegisterAware(), not the '
                 'string "fast"')
_FUSE_SUGGEST = "a typed ConservativeFusion()/Disabled(), not a string"


def _check_math(math):
    """Validate the ``math`` selector: a typed math mode, never a bare string (Spec 5 sec.14.2).

    ``None`` keeps the conservative :class:`StrictMath` default. A bare ``str`` is REJECTED via
    :func:`pops.descriptors.reject_string_selector` -- Spec 5 forbids naming a math mode with a
    string; the message points at the typed alternatives. Any other non-:class:`_MathMode` value
    is a clear ``TypeError`` rather than a silent mis-set that crashes later in :meth:`options`.
    """
    if math is None:
        return StrictMath()
    if isinstance(math, str):
        reject_string_selector(math, "optimization math", _MATH_SUGGEST)  # always raises
    if not isinstance(math, _MathMode):
        raise TypeError("Optimization: math must be a typed StrictMath()/FastMath()/DebugMath()/"
                        "GpuRegisterAware() (got %r)" % (type(math).__name__,))
    return math


def _check_fuse(fuse):
    """Validate the ``fuse`` selector: a typed fusion policy, never a bare string (sec.14.2).

    ``None`` means no fusion policy attached. A bare ``str`` is REJECTED via
    :func:`pops.descriptors.reject_string_selector`; any other value lacking the descriptor
    ``options()`` surface is a clear ``TypeError`` rather than a silent mis-set.
    """
    if fuse is None:
        return None
    if isinstance(fuse, str):
        reject_string_selector(fuse, "optimization fuse", _FUSE_SUGGEST)  # always raises
    if not isinstance(fuse, _Fusion):
        raise TypeError("Optimization: fuse must be a typed ConservativeFusion()/Disabled() "
                        "(got %r)" % (type(fuse).__name__,))
    return fuse


class Optimization(Descriptor):
    """A typed codegen optimization policy (Spec 5 sec.13.8).

    ``Optimization(cse=True, eliminate_dead_nodes=True, eliminate_redundant_solves=True,
    fuse=ConservativeFusion(), hoist_reciprocals=True, math=StrictMath())``. The ``math`` mode
    defaults to :class:`StrictMath` (conservative); a non-strict mode must be chosen
    explicitly. The ``math`` / ``fuse`` algorithm selectors are TYPED objects, never strings
    (Spec 5 sec.7 / sec.14.2): a bare ``math="fast"`` / ``fuse="conservative"`` is rejected at
    construction with an actionable message, not silently accepted and crashed later.
    :meth:`to_emit_kwargs` maps onto the existing codegen emit knobs so the policy can drive the
    C++ emitter without a breaking signature change.
    """

    category = "optimization"

    def __init__(self, cse=True, eliminate_dead_nodes=True, eliminate_redundant_solves=True,
                 fuse_local_ops=True, hoist_reciprocals=True, fuse=None, math=None):
        self.cse = bool(cse)
        self.eliminate_dead_nodes = bool(eliminate_dead_nodes)
        self.eliminate_redundant_solves = bool(eliminate_redundant_solves)
        self.fuse_local_ops = bool(fuse_local_ops)
        self.hoist_reciprocals = bool(hoist_reciprocals)
        self.fuse = _check_fuse(fuse)
        self.math = _check_math(math)

    def options(self):
        return {"cse": self.cse, "eliminate_dead_nodes": self.eliminate_dead_nodes,
                "eliminate_redundant_solves": self.eliminate_redundant_solves,
                "fuse_local_ops": self.fuse_local_ops,
                "hoist_reciprocals": self.hoist_reciprocals,
                "fuse": self.fuse.name if self.fuse is not None else None,
                "math": self.math.name}

    def capabilities(self):
        return {"strict_math": isinstance(self.math, StrictMath)}

    def to_emit_kwargs(self):
        """Map this policy onto the existing ``emit_cpp*`` knobs (cse / hoist_reciprocals)."""
        return {"cse": self.cse, "hoist_reciprocals": self.hoist_reciprocals}

    @classmethod
    def default(cls):
        """The recommended conservative default (StrictMath, CSE on, no risky fusion)."""
        return cls()


__all__ = ["Optimization", "ConservativeFusion", "Disabled"]
