"""pops.lib.time.macros -- the Spec-3 MACRO-brick time catalog (internal artifact).

A catalog SimpleNamespace of MACRO ``BrickDescriptor``-style entries that forward to
``pops.lib.time.std.<name>`` (build Program IR only). This is the OLD Spec-3 ``pops.lib.time``
surface; under Spec 4 the public ``pops.lib.time`` is the scheme-builder package
(:mod:`pops.lib.time`'s ``forward_euler`` / ``ssprk2`` / ...), so this catalog ns is kept
here as an internal artifact and is NOT re-exported as ``lib.time``.
"""
from types import SimpleNamespace

from .std import std as _std


def _time_macro(std_name):
    """A macro brick that forwards to ``pops.lib.time.std.<std_name>``; builds IR only."""
    def macro(P, block, *args, **kwargs):
        return getattr(_std, std_name)(P, block, *args, **kwargs)
    macro.__name__ = std_name
    macro.__doc__ = "Build the %r time scheme into Program P (pops.lib.time.std)." % std_name
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

__all__ = ["time"]
