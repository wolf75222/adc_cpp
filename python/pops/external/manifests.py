"""pops.external.manifests -- read + register a compiled-brick manifest (Spec 5 sec.5.17).

A manifest is the JSON ``pops_brick_manifest()`` exports (``{"bricks": [{"id", "category",
"requirements", "capabilities"}, ...]}``). It can be read from a ``.json`` file or from a
``.so`` (dlopened). Both register the ids in the in-process catalog owned by
:mod:`pops.descriptors`; nothing here computes.
"""
from pops.descriptors import load_cpp_library, _register_manifest


def register_manifest_file(path):
    """Register the bricks in a manifest ``.json`` file. Returns the count registered."""
    with open(str(path), "r", encoding="utf-8") as handle:
        return _register_manifest(handle.read())


def register(path):
    """Register a manifest from a ``.json`` file or a brick ``.so`` (dlopen). Returns the count.

    A ``.json`` path is parsed directly; anything else is treated as a loadable ``.so`` and
    dlopened via :func:`pops.descriptors.load_cpp_library` (its static initializers register
    the bricks and the exported ``pops_brick_manifest()`` is read).
    """
    p = str(path)
    if p.endswith(".json"):
        return register_manifest_file(p)
    return load_cpp_library(p)


__all__ = ["register", "register_manifest_file"]
