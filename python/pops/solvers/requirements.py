"""pops.solvers.requirements -- the solver capability vocabulary (Spec 5 sec.5.7).

A solver descriptor declares what routes it SUPPORTS through a fixed set of capability tags
(uniform / amr / mpi / gpu / variable_epsilon / anisotropic / screened / periodic_bc /
wall_bc). This module names that vocabulary ONCE so the elliptic descriptors do not each
re-spell the keys, and :func:`capability_map` builds the plain-dict ``supports_*`` view a
descriptor's :meth:`capabilities` returns. It is pure metadata -- no descriptor, no compute.
"""

#: The capability tags a solver descriptor may advertise (Spec 5 sec.5.7). Ordered so the
#: ``supports_*`` capability dict has a stable, documented key order.
CAPABILITY_TAGS = (
    "uniform",
    "amr",
    "mpi",
    "gpu",
    "variable_epsilon",
    "anisotropic",
    "screened",
    "periodic_bc",
    "wall_bc",
)


def capability_map(**supported):
    """Build the ``{"supports_<tag>": bool}`` capability dict from named tags (default False).

    Each keyword must name a tag in :data:`CAPABILITY_TAGS`; an unknown tag raises so a typo in
    a descriptor's capability declaration is caught at import, not silently dropped. Tags not
    passed default to ``False`` -- a descriptor opts in to what it supports.
    """
    unknown = sorted(set(supported) - set(CAPABILITY_TAGS))
    if unknown:
        raise ValueError(
            "unknown solver capability tag(s) %s; valid tags: %s"
            % (unknown, ", ".join(CAPABILITY_TAGS)))
    return {"supports_%s" % tag: bool(supported.get(tag, False)) for tag in CAPABILITY_TAGS}


__all__ = ["CAPABILITY_TAGS", "capability_map"]
