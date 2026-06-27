"""pops.physics : math/physics model AUTHORING layer.

Two authoring surfaces share this package:

* the BLACKBOARD facade ``pops.physics.Model`` (Spec 3) -- a state, primitives, a
  flux, an elliptic field-solve, sources and local linear operators tied together
  by board equations -- lives in :mod:`pops.physics.board`. This is the public
  ``pops.physics.Model`` (unchanged surface).
* the PDE facade (the symbolic mini-DSL ``Model`` / ``HyperbolicModel`` / ``Param``)
  lives in :mod:`pops.physics.facade` / :mod:`pops.physics.model`; it is what
  ``pops.dsl.Model`` re-exports. Exposed here as ``PdeModel`` (and ``HyperbolicModel``)
  so consumers that need the PDE engine can reach it without the ``dsl`` shim.

Hybrid composition (``NativeBrick`` / ``HybridModel`` / partial DSL bricks) and the
generic inter-species ``CoupledSource`` round out the surface.

Import-graph rule (Spec 4): this package imports only :mod:`pops.ir` and
:mod:`pops.model` (plus stdlib / numpy) at module scope. Any :mod:`pops.codegen`
or ``_pops`` use is LAZY, inside method bodies (the codegen wrappers on
``HyperbolicModel`` / ``Model.compile`` / ``HybridModel.compile``). The
architecture test (ADC-474) enforces this.
"""
# Aux-channel layout + physical roles (single Python-side source; mirror of the C++ headers).
from .aux import (
    AUX_CANONICAL, AUX_BASE_COMPS, AUX_NAMED_BASE, AUX_NAMED_MAX, CANONICAL_ROLES,
    aux_n_aux, aux_total_n_aux, role_of, roles_for)

# PDE-model symbolic mini-DSL (the engine pops.dsl.Model wraps).
from .model import HyperbolicModel, Param, RuntimeParam
from .facade import Model as PdeModel

# Hybrid composition: native + DSL bricks into one CompositeModel .so.
from .bricks import (
    NativeBrick, CompiledBrick, CompiledHyperbolicBrick, CompiledSourceBrick,
    CompiledEllipticBrick, HyperbolicBrick, SourceBrick, EllipticBrick)
from .hybrid import HybridModel

# Generic coupled inter-species source (explicit splitting; pure bytecode codegen).
from .multispecies import CoupledSource, CompiledCoupledSource

# Blackboard board facade: the public pops.physics.Model surface (Spec 3).
from .board import Model
# Spec 5 sec.5.16 / sec.11 preferred name. ALIAS, not a rename: it is the SAME class object
# (PhysicsModel is Model), so every existing pops.physics.Model consumer keeps working and the
# class __name__ stays "Model" (a `type(x).__name__ == "Model"` check is unaffected).
PhysicsModel = Model
from .board_handles import (
    Invariant, FluxHandle, SourceHandle, FieldsHandle, FieldHandle,
    LocalLinearOperatorExpr, CallableOperator, StateHandle, VectorHandle,
    _roles_for)  # restore the flat physics.py module-level access (test_riemann_capabilities)

__all__ = [
    # board surface (the historical pops.physics public names)
    "Model", "PhysicsModel", "Invariant", "FluxHandle", "SourceHandle", "FieldsHandle", "FieldHandle",
    "LocalLinearOperatorExpr", "CallableOperator", "StateHandle", "VectorHandle",
    # aux + roles
    "AUX_CANONICAL", "AUX_BASE_COMPS", "AUX_NAMED_BASE", "AUX_NAMED_MAX", "CANONICAL_ROLES",
    "aux_n_aux", "aux_total_n_aux", "role_of", "roles_for",
    # PDE-model engine
    "PdeModel", "HyperbolicModel", "Param", "RuntimeParam",
    # hybrid + bricks
    "NativeBrick", "CompiledBrick", "CompiledHyperbolicBrick", "CompiledSourceBrick",
    "CompiledEllipticBrick", "HyperbolicBrick", "SourceBrick", "EllipticBrick", "HybridModel",
    # coupled source
    "CoupledSource", "CompiledCoupledSource",
]
