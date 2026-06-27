"""Spec 4 (36.1 / 38): forbidden paths must never exist in the pops/ tree.

The clean-break restructure bans the ``std`` and ``custom`` escape hatches and the
flat ``models`` package: a time scheme is a library macro (``pops.lib.time.*``), not a
``pops.time.std`` submodule; a closure is a decorated callable, not a
``pops.lib.models.moments.custom`` module; and concrete models live under
``pops.lib.models.*``, never a top-level ``pops.models``.

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
