"""pops.moments.speeds -- the wave-speed strategy descriptor (Spec 5 sec.6).

Maps the moment-model wave-speed strategy onto the engine's ``exact_speeds`` / ``roe``
flags:

* ``EXACT_EIGENVALUES`` -- exact wave speeds by autodiff of the flux + per-cell numeric
  eigenvalues (``exact_speeds=True``).
* ``ROE_DISSIPATION`` -- additionally emit the generic Roe dissipation (``roe=True``).
* ``BOUNDED`` -- the caller sets the wave speeds itself (``exact_speeds=False``).

It CHOOSES the wave-speed algorithm, so it is a typed :class:`pops.descriptors.Descriptor`
(Spec 5 sec.6): it declares its options / capabilities and is inspectable. Inert -- it
records the choice and exposes it as the engine flags on ``.build()``; the eigenvalue / Roe
arithmetic is generated and runs in C++.
"""
from pops.descriptors import Descriptor


class ExactSpeeds(Descriptor):
    """The wave-speed strategy a moment hierarchy uses (route-choosing descriptor).

    ``EXACT_EIGENVALUES`` / ``ROE_DISSIPATION`` / ``BOUNDED`` map to the engine's
    ``exact_speeds`` / ``roe`` flags. It records the choice; the eigenvalue / Roe
    arithmetic is generated and runs in C++.
    """

    category = "wave_speed"

    EXACT_EIGENVALUES = "exact_eigenvalues"
    ROE_DISSIPATION = "roe_dissipation"
    BOUNDED = "bounded"
    _KINDS = (EXACT_EIGENVALUES, ROE_DISSIPATION, BOUNDED)

    def __init__(self, kind=EXACT_EIGENVALUES):
        if kind not in ExactSpeeds._KINDS:
            raise ValueError("ExactSpeeds kind %r must be one of %s"
                             % (kind, ", ".join(ExactSpeeds._KINDS)))
        self.kind = kind

    @classmethod
    def from_flags(cls, exact_speeds, roe):
        """The descriptor matching the engine flags (``roe`` wins, then ``exact_speeds``)."""
        if roe:
            return cls(cls.ROE_DISSIPATION)
        return cls(cls.EXACT_EIGENVALUES if exact_speeds else cls.BOUNDED)

    @property
    def exact_speeds(self):
        """The engine ``exact_speeds`` flag (True unless the BOUNDED strategy)."""
        return self.kind != ExactSpeeds.BOUNDED

    @property
    def roe(self):
        """The engine ``roe`` flag (True only for the ROE_DISSIPATION strategy)."""
        return self.kind == ExactSpeeds.ROE_DISSIPATION

    def options(self):
        return {"kind": self.kind}

    def capabilities(self):
        return {"exact_speeds": self.exact_speeds, "roe": self.roe}

    def __repr__(self):
        return "ExactSpeeds(%r)" % (self.kind,)


__all__ = ["ExactSpeeds"]
