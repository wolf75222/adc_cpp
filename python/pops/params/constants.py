"""pops.params.constants -- dimensioned constant values (Spec 5 sec.5.12).

A :class:`Constant` is a named, optionally-dimensioned compile-time value. It is the
"valeur dimensionnee" of Spec 5 sec.5.12 -- a :class:`pops.params.ConstParam` that also
carries a unit string for documentation / dimensional bookkeeping. Inert.
"""
from pops.params.runtime import ConstParam


class Constant(ConstParam):
    """A named constant with an optional unit (e.g. ``Constant("c", 2.998e8, unit="m/s")``)."""

    category = "constant"

    def __init__(self, name, value, unit=None, dtype=None):
        from pops.math import Real
        super().__init__(name, value, dtype=dtype if dtype is not None else Real)
        self.unit = unit

    def options(self):
        opt = super().options()
        opt["unit"] = self.unit
        return opt


__all__ = ["Constant"]
