"""SourceHandle: a declared local source term."""
from adc import math as _bm


class SourceHandle(_bm.RateTerm):
    """A declared local source term -- a summand of a rate equation."""

    def __init__(self, display_name, reg_name):
        self.name = str(display_name)
        self.reg_name = str(reg_name)

    def _rate_terms(self):
        return [("source", self, 1.0)]

    def __repr__(self):
        return "SourceHandle(%r)" % (self.name,)
