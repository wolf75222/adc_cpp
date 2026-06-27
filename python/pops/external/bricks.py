"""pops.external.bricks -- typed references to compiled-out-of-core bricks (Spec 5 sec.5.17).

A :class:`CompiledBrickRef` names a brick that lives in a ``.so`` / manifest outside the
standard PoPS core but is compatible with its ABI manifests. Spec 5 forbids a free string to
a native brick: a reference carries a manifest + a native id, and resolves to the typed
``external_cpp`` :class:`pops.descriptors.BrickDescriptor` (with the manifest's requirements /
capabilities). A missing manifest or an unregistered id is a clear error before runtime.
"""
from pops.descriptors import Availability, Descriptor, _external_descriptor
from .manifests import register


class CompiledBrickRef(Descriptor):
    """A reference to a compiled brick: a manifest (``.json`` or ``.so``) + a native id.

    ``CompiledBrickRef(manifest="build/my_riemann.json", native_id="my_hll_variant")``.
    :meth:`resolve` registers the manifest and returns the validated external descriptor;
    :meth:`available` reports whether it resolves, with the reason if not.
    """

    category = "external_brick"

    def __init__(self, manifest, native_id, *, expect_category=None):
        self.manifest = str(manifest)
        self.native_id = str(native_id)
        self.expect_category = expect_category
        self._registered = False

    def options(self):
        return {"manifest": self.manifest, "native_id": self.native_id,
                "expect_category": self.expect_category}

    def _ensure_registered(self):
        if not self._registered:
            register(self.manifest)
            self._registered = True

    def resolve(self):
        """Register the manifest and return the typed ``external_cpp`` BrickDescriptor."""
        self._ensure_registered()
        return _external_descriptor(self.native_id, expect_category=self.expect_category)

    def requirements(self):
        try:
            return dict(self.resolve().requirements)
        except Exception:
            return {}

    def capabilities(self):
        try:
            return dict(self.resolve().capabilities)
        except Exception:
            return {}

    def available(self, context=None):
        try:
            self.resolve()
            return Availability.yes()
        except Exception as err:
            return Availability.no(
                "compiled brick %r could not be resolved: %s" % (self.native_id, err),
                alternatives=["check the manifest path and native_id"])


# Spec 5 uses both names; ExternalBrick is the same typed reference.
ExternalBrick = CompiledBrickRef

__all__ = ["CompiledBrickRef", "ExternalBrick"]
