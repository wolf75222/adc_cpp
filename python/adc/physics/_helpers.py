"""Internal helpers shared across the physics sub-modules."""
import re


def _safe_name(name):
    """A C-identifier-safe operator name derived from a board display name."""
    s = re.sub(r"[^0-9a-zA-Z_]", "_", str(name)).strip("_")
    if not s:
        raise ValueError("operator name %r has no identifier characters" % (name,))
    if s[0].isdigit():
        s = "_" + s
    return s


# Board role vocabulary -> dsl canonical role (adc::VariableRole). The dsl roles_for() uses an
# explicit role override verbatim, so a board role must already be canonical for the native HLLC/Roe
# role lookup (which indexes "Density"/"MomentumX"/"MomentumY"/"Energy") to find it.
_BOARD_ROLE = {
    "density": "Density",
    "momentum_x": "MomentumX", "momentum_y": "MomentumY", "momentum_z": "MomentumZ",
    "energy": "Energy", "pressure": "Pressure", "temperature": "Temperature",
}


def _canon_role(role):
    """Canonicalize a board role string to a dsl role; pass through None and unknown roles."""
    if role is None:
        return None
    return _BOARD_ROLE.get(str(role).lower(), role)


def _roles_for(hyp):
    """The canonical dsl roles of a HyperbolicModel's conservative state."""
    from adc import dsl as _dsl
    return _dsl.roles_for(hyp.cons_names, hyp.cons_roles)
