"""Spec 5 (sec.15): pops.physics stays codegen-free, and typed descriptors print short.

Two complementary architecture guarantees:

1. ``pops.physics`` is an authoring layer: none of its modules may import ``pops.codegen``
   at module scope (the codegen engine pulls numpy and the heavy emitter; physics imports
   it LAZILY inside ``Model.compile``). This focused source-only AST check mirrors the
   technique in ``test_no_runtime_imports.py`` but pins the specific physics -> codegen edge.

2. A typed descriptor's ``repr()`` / ``str()`` is a short, human-readable summary -- never a
   raw ``array(...)`` numeric dump (Spec 5 sec.12.1, the print-summary contract).

Part 1 is source-only (it parses the tree, it does not import ``pops``). Part 2 imports
``pops`` and so skips when ``_pops`` is absent; the source-only check still runs bare.
"""
import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PHYSICS = REPO_ROOT / "python" / "pops" / "physics"


def _module_scope_import_targets(tree):
    """Yield every module-scope (col_offset==0) absolute import target in a parsed module."""
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if node.col_offset != 0:  # nested in a function/class body -> a lazy import, allowed
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def test_physics_never_imports_codegen_at_module_scope():
    assert PHYSICS.is_dir(), "missing %s" % PHYSICS
    offenders = []
    for path in sorted(PHYSICS.rglob("*.py")):
        tree = ast.parse(path.read_text(), str(path))
        for target in _module_scope_import_targets(tree):
            if target == "pops.codegen" or target.startswith("pops.codegen."):
                offenders.append("%s imports %s at module scope"
                                 % (path.relative_to(REPO_ROOT), target))
    assert not offenders, (
        "pops.physics must import pops.codegen LAZILY (in Model.compile), never at module "
        "scope:\n  " + "\n  ".join(offenders))


# Part 2 imports pops; skip if the native extension cannot be loaded in this interpreter.
try:
    import pops._pops  # noqa: F401
    _HAVE_POPS = True
except Exception:  # pragma: no cover - exercised only without a built extension
    _HAVE_POPS = False


@pytest.mark.skipif(not _HAVE_POPS, reason="compiled _pops extension not importable")
def test_typed_descriptor_repr_is_short_and_has_no_array_dump():
    from pops.diagnostics import energy, mass, norm
    from pops.numerics.reconstruction import FirstOrder, MUSCL, WENO5
    from pops.numerics.riemann import HLL, HLLC, Roe, Rusanov

    descriptors = [HLL(), Rusanov(), HLLC(), Roe(),
                   FirstOrder(), MUSCL(), WENO5(),
                   norm(), mass(), energy()]
    for descriptor in descriptors:
        for text in (repr(descriptor), str(descriptor)):
            assert len(text) < 800, "descriptor print too long (%d chars): %r" % (
                len(text), text[:80])
            assert "array(" not in text, "descriptor print contains a raw array dump: %r" % (
                text[:120])
            assert text.strip(), "descriptor print is empty"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
