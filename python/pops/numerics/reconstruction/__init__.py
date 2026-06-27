"""pops.numerics.reconstruction -- the spatial-reconstruction brick catalog (Spec 3 / Spec 5).

FirstOrder / MUSCL / WENO5 / WENO5Z selectors plus a ``User`` selector for an
external C++ reconstruction brick. The slope limiters are catalogued separately
in :mod:`pops.numerics.reconstruction.limiters`.

pops::Weno5 IS the WENO5-Z reconstruction (it wraps weno5z()); WENO5 and WENO5Z both
select it. MUSCL is reconstruction-by-limiter; its native limiter type is pops::Minmod.
"""
from types import SimpleNamespace

from pops.descriptors import _native, _external_descriptor
from .limiters import limiters

# Spec 5 sec.7 / criterion 11: the GHOST (halo) depth a reconstruction stencil NEEDS, by its
# lowered scheme token. A first-order scheme reads the cell mean (1 ghost); a second-order
# MUSCL/slope-limited reconstruction reads one neighbour (2); the fifth-order WENO5(-Z) stencil
# reads two neighbours each side (3). Keyed on the token so the runtime Spatial brick (which
# carries the lowered token, not the descriptor) can look the requirement up.
REQUIRED_GHOST_DEPTH = {
    "firstorder": 1,
    "none": 1,
    "minmod": 2,
    "vanleer": 2,
    "muscl": 2,
    "weno5": 3,
    "weno5z": 3,
}

#: The conservative second-order-MUSCL ghost depth the INSPECTION surface assumes for a memory
#: estimate (``pops.codegen.inspect_compiled._ghost_depth``). NOTE this is NOT a hard block
#: limit: the native runtime GROWS each block's halo to match its reconstruction
#: (``include/pops/runtime/system.hpp`` ``block_n_ghost(lim)`` -> 3 for weno5, 2 for MUSCL), so
#: WENO5 is served today. The ghost-depth validation therefore checks the reconstruction's
#: DECLARED requirement only against an EXPLICITLY-constrained block depth (a fixed / external
#: halo a caller passes), never against this assumption -- rejecting WENO5 by default would be a
#: FALSE POSITIVE that breaks a working problem.
INSPECT_GHOST_DEPTH_ASSUMPTION = 2

reconstruction = SimpleNamespace(
    FirstOrder=lambda: _native("firstorder", "pops::NoSlope", "firstorder",
                               category="reconstruction", ghost_depth=1),
    MUSCL=lambda limiter="minmod": _native(
        "muscl", "pops::Minmod", limiter, category="reconstruction", limiter=limiter,
        ghost_depth=2),
    WENO5=lambda: _native("weno5", "pops::Weno5", "weno5", category="reconstruction",
                          ghost_depth=3),
    WENO5Z=lambda: _native("weno5z", "pops::Weno5", "weno5", category="reconstruction",
                           ghost_depth=3),
    User=lambda brick_id: _external_descriptor(brick_id, expect_category="reconstruction"),
)


def required_ghost_depth(reconstruction_or_token):
    """The ghost depth a reconstruction NEEDS (Spec 5 sec.7 / criterion 11).

    Accepts a reconstruction descriptor (reads its declared ``ghost_depth`` option / scheme) or a
    bare lowered scheme token (``"weno5"`` / ``"muscl"`` / ...). Returns ``None`` when the
    requirement is not declared/known -- the caller then does NOT reject (a missing requirement is
    not a known incompatibility; no false positive).
    """
    if isinstance(reconstruction_or_token, str):
        return REQUIRED_GHOST_DEPTH.get(reconstruction_or_token)
    descriptor = reconstruction_or_token
    declared = (getattr(descriptor, "options", None) or {}).get("ghost_depth")
    if isinstance(declared, int) and not isinstance(declared, bool):
        return declared
    return REQUIRED_GHOST_DEPTH.get(getattr(descriptor, "scheme", None))


def validate_ghost_depth(reconstruction_or_token, available=None, block=None):
    """Reject a reconstruction whose DECLARED ghost depth exceeds an EXPLICIT block depth.

    Spec 5 sec.7 / criterion 11: a high-order stencil (WENO5 needs 3 ghost cells) reading past a
    too-thin halo is a correctness bug, so the requirement must be checked before runtime. This
    raises a clear, actionable error when the requirement is KNOWN and exceeds @p available.

    The OVERRIDING discipline is NO FALSE POSITIVE. The native runtime GROWS each block's halo to
    match its reconstruction (``block_n_ghost(lim)`` in include/pops/runtime/system.hpp: 3 for
    weno5, 2 for MUSCL), so WENO5 is served today on a default block. Hence:

    * @p available defaults to ``None`` == "the block halo is allocated to match the scheme" ->
      the check NEVER fires (rejecting WENO5 by default would break a working problem);
    * the check fires ONLY when a caller passes an EXPLICIT @p available that constrains the
      block below the requirement (a fixed / external halo);
    * an undeclared reconstruction (``required_ghost_depth`` is ``None``) is never rejected.

    Args:
        reconstruction_or_token: A reconstruction descriptor or its lowered scheme token.
        available: An EXPLICIT block ghost depth, or ``None`` to defer to the scheme-matched
            runtime allocation (no rejection).
        block: Optional block name, woven into the message ("block 'plasma' has ghost_depth=2").

    Returns:
        bool: ``True`` when the depth is sufficient, undeclared, or scheme-matched.

    Raises:
        ValueError: When the reconstruction's declared ghost depth exceeds an explicit
            @p available.
    """
    if available is None:
        return True  # runtime grows the halo to the scheme; no explicit constraint to check.
    needed = required_ghost_depth(reconstruction_or_token)
    if needed is None or needed <= available:
        return True
    if isinstance(reconstruction_or_token, str):
        name = reconstruction_or_token.upper()
    else:
        name = getattr(reconstruction_or_token, "name",
                       type(reconstruction_or_token).__name__).upper()
    where = ("block %r has" % block) if block is not None else "the block provides"
    raise ValueError(
        "%s requires ghost_depth >= %d, %s ghost_depth=%d; use a lower-order reconstruction "
        "(pops.numerics.reconstruction.MUSCL()) or a block with a deeper halo."
        % (name, needed, where, available))

# Spec 5: expose the schemes at module scope (``from pops.numerics.reconstruction import MUSCL``).
FirstOrder = reconstruction.FirstOrder
MUSCL = reconstruction.MUSCL
WENO5 = reconstruction.WENO5
WENO5Z = reconstruction.WENO5Z
User = reconstruction.User

__all__ = ["reconstruction", "limiters", "FirstOrder", "MUSCL", "WENO5", "WENO5Z", "User",
           "REQUIRED_GHOST_DEPTH", "INSPECT_GHOST_DEPTH_ASSUMPTION", "required_ghost_depth",
           "validate_ghost_depth"]
