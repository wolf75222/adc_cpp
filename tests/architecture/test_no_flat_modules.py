"""Spec 4 (38): the flat single-file modules must be gone at the END state.

The pre-restructure tree had monolithic ``python/pops/<name>.py`` files (``dsl``,
``model``, ``time``, ``physics``, ``lib``, ``library``, ``moments``, ``integrate``).
Spec 4 replaces each with an acyclic sub-package of <=500-line files. This test asserts
the flat files are absent.

On the PR-D branch some of these still exist (``dsl.py``, ``library.py``,
``moments.py``, ``integrate.py``); this test is written as the TARGET and is the
PR-E / PR-F punch-list -- it is expected to fail until those modules are decomposed
or removed.

The test reads the source tree only; it does not import ``pops`` or ``_pops``.
"""
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
POPS = REPO_ROOT / "python" / "pops"

# Flat modules that must not remain as single .py files at the package root.
FLAT_MODULES = (
    "dsl",
    "model",
    "time",
    "physics",
    "lib",
    "library",
    "moments",
    "integrate",
)


@pytest.mark.parametrize("name", FLAT_MODULES)
def test_flat_module_absent(name):
    flat = POPS / ("%s.py" % name)
    assert not flat.exists(), (
        "flat module still present: python/pops/%s.py "
        "(Spec 4 38: replace with an acyclic sub-package of <=500-line files)" % name
    )
