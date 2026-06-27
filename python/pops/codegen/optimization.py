"""pops.codegen.optimization -- typed codegen optimization policy (Spec 5 sec.13.8).

The optimization policy is a typed object, not a string ``optimization="fast"``. It selects
which IR / expression transforms the codegen may apply (CSE, dead-node elimination,
redundant-solve elimination, local fusion, reciprocal hoisting) and the numeric math mode
(:mod:`pops.codegen.math_options`). It is inert: it describes + inspects the choices; the
codegen consumes it. A non-strict numeric transform is never implicit -- it must be selected
here (Spec 5 sec.13.10).
"""
from pops.descriptors import Descriptor
from .math_options import StrictMath


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


class Optimization(Descriptor):
    """A typed codegen optimization policy (Spec 5 sec.13.8).

    ``Optimization(cse=True, eliminate_dead_nodes=True, eliminate_redundant_solves=True,
    fuse=ConservativeFusion(), hoist_reciprocals=True, math=StrictMath())``. The ``math`` mode
    defaults to :class:`StrictMath` (conservative); a non-strict mode must be chosen
    explicitly. :meth:`to_emit_kwargs` maps onto the existing codegen emit knobs so the policy
    can drive the C++ emitter without a breaking signature change.
    """

    category = "optimization"

    def __init__(self, cse=True, eliminate_dead_nodes=True, eliminate_redundant_solves=True,
                 fuse_local_ops=True, hoist_reciprocals=True, fuse=None, math=None):
        self.cse = bool(cse)
        self.eliminate_dead_nodes = bool(eliminate_dead_nodes)
        self.eliminate_redundant_solves = bool(eliminate_redundant_solves)
        self.fuse_local_ops = bool(fuse_local_ops)
        self.hoist_reciprocals = bool(hoist_reciprocals)
        self.fuse = fuse
        self.math = math if math is not None else StrictMath()

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
