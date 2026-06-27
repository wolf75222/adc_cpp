"""Spec 4 (36.1 / 38 / 7): forbidden paths must never exist in the pops/ tree.

The clean-break restructure bans the ``std`` and ``custom`` escape hatches and the
flat ``models`` package: a time scheme is a library macro called by its explicit name
(``pops.lib.time.ssprk3``), never a ``std`` bundle; a closure is a decorated callable, not a
``pops.lib.models.moments.custom`` module; and concrete models live under
``pops.lib.models.*``, never a top-level ``pops.models``.

Spec 4 s7 bans the *name* ``std`` outright -- not only the ``std/`` directory (s38) but any
``std.py`` module or ``std`` namespace. test_no_std_name_anywhere enforces that.

This test reads the source tree only; it does not import ``pops`` or ``_pops``.
"""
import pathlib

import pytest

# tests/architecture/<this file> -> repo root is parents[2].
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
POPS = REPO_ROOT / "python" / "pops"

# Paths the restructure forbids outright (files or directories).
FORBIDDEN = (
    "lib/models/moments/custom.py",
    "time/std",
    "lib/std",
    "models",
)


@pytest.mark.parametrize("rel", FORBIDDEN)
def test_forbidden_path_absent(rel):
    target = POPS / rel
    assert not target.exists(), (
        "forbidden Spec 4 path is present: python/pops/%s "
        "(36.1/38: no std/custom escape hatch, no flat models package)" % rel
    )


def test_no_std_name_anywhere():
    """Spec 4 s7: the name ``std`` is banned in the architecture, not just the ``std/`` directory
    (s38). No ``std.py`` module may exist anywhere under python/pops -- schemes are exposed by
    their explicit names (``pops.lib.time.ssprk3``), never via a ``std`` bundle.
    """
    offenders = sorted(str(p.relative_to(REPO_ROOT)) for p in POPS.rglob("std.py"))
    assert not offenders, (
        "Spec 4 s7 forbids the `std` name; remove: %s "
        "(call schemes by explicit name, e.g. pops.lib.time.ssprk3)" % offenders
    )
