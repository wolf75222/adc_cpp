"""pops.codegen.math_options -- typed numeric/codegen math modes (Spec 5 sec.13.8).

A math mode is a typed object (``StrictMath()``), not a string ``optimization="fast"``. It
selects which numeric transformations the codegen may apply. ``StrictMath`` is the
conservative default; ``FastMath`` allows rounding-changing transforms; ``DebugMath`` favours
readable generated C++; ``GpuRegisterAware`` limits temporaries / register pressure. Inert.
"""
from pops.descriptors import Descriptor


class _MathMode(Descriptor):
    category = "math_mode"


class StrictMath(_MathMode):
    """Conservative numeric behaviour (recommended default): no fast-math, no implicit FMA."""

    def options(self):
        return {"fast_math": False, "implicit_fma": False}

    def capabilities(self):
        return {"bit_reproducible": True}


class FastMath(_MathMode):
    """Aggressive numeric transforms -- EXPLICIT only; may change rounding."""

    def options(self):
        return {"fast_math": True}

    def capabilities(self):
        return {"may_change_rounding": True, "bit_reproducible": False}


class DebugMath(_MathMode):
    """Fewer optimizations, more readable generated C++ (debugging the codegen)."""

    def options(self):
        return {"optimize": False, "readable": True}


class GpuRegisterAware(_MathMode):
    """Limit temporaries / register pressure (large moment systems on GPU)."""

    def options(self):
        return {"limit_temporaries": True}

    def capabilities(self):
        return {"register_pressure_aware": True}


__all__ = ["StrictMath", "FastMath", "DebugMath", "GpuRegisterAware"]
