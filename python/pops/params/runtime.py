"""pops.params.runtime -- typed scalar parameters (Spec 5 sec.5.12).

A parameter declares whether it is compile-time or runtime, its typed dtype (from
:mod:`pops.math`), an optional default, and an optional typed domain constraint -- instead
of the string form ``Param(kind="runtime")``. These are inert descriptors; the codegen /
runtime consume them (a runtime param appears in ``compiled.arguments()``; a const param
participates in the cache key).
"""
from pops.descriptors import Descriptor
from pops.math import Real


class RuntimeParam(Descriptor):
    """A runtime parameter: changeable without recompilation if the ABI is unchanged.

    ``RuntimeParam("alpha", dtype=Real, default=1.0, domain=Positive())``. It appears in
    ``compiled.arguments()`` and is set at bind time; it does NOT participate in the codegen
    hash (changing it must not force a recompile while the ABI holds).
    """

    category = "runtime_param"

    def __init__(self, name, dtype=Real, default=None, domain=None):
        self._name = str(name)
        self.dtype = dtype
        self.default = default
        self.domain = domain

    @property
    def name(self):
        return self._name

    def options(self):
        return {"name": self._name, "dtype": getattr(self.dtype, "name", self.dtype),
                "default": self.default,
                "domain": self.domain.name if self.domain is not None else None}

    def capabilities(self):
        return {"runtime": True, "compile_time": False}

    def validate(self, context=None):
        super().validate(context)  # honour the explainable available() route check too
        if self.domain is not None and self.default is not None:
            self.domain.check(self.default, who="%s default" % self._name)
        return True


class ConstParam(Descriptor):
    """A compile-time constant: frozen into the generated code; in the cache key.

    ``ConstParam("gamma", value=5.0/3.0)``. Changing it can require a recompile (it changes
    the codegen hash). Use a :class:`RuntimeParam` for values that must change at run time.
    """

    category = "const_param"

    def __init__(self, name, value, dtype=Real):
        self._name = str(name)
        self.value = value
        self.dtype = dtype

    @property
    def name(self):
        return self._name

    def options(self):
        return {"name": self._name, "value": self.value,
                "dtype": getattr(self.dtype, "name", self.dtype)}

    def capabilities(self):
        return {"runtime": False, "compile_time": True, "in_cache_key": True}


class DerivedParam(Descriptor):
    """A parameter derived from others by a PoPS expression (computed in C++, not Python)."""

    category = "derived_param"

    def __init__(self, name, expression):
        self._name = str(name)
        self.expression = expression

    @property
    def name(self):
        return self._name

    def options(self):
        return {"name": self._name,
                "expression": getattr(self.expression, "name", repr(self.expression))}


__all__ = ["RuntimeParam", "ConstParam", "DerivedParam"]
