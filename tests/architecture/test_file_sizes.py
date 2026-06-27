"""Spec 4 (36.3): every pops/ source file stays small.

The restructure caps each ``python/pops/**/*.py`` at 500 lines so no module grows back
into a monolith. The sole allowlisted exception is the package facade
``python/pops/__init__.py``, capped at 120 lines (it should re-export, not implement).

On the PR-D branch ``__init__.py`` is still the giant runtime facade and several codegen
files exceed 500 lines; this test is the TARGET and is expected to fail until PR-F trims
them. The failure message lists every offending file with its line count -- that list is
the punch-list.

The test reads the source tree only; it does not import ``pops`` or ``_pops``.
"""
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
POPS = REPO_ROOT / "python" / "pops"

DEFAULT_MAX_LINES = 500
# Allowlist: path relative to python/pops -> permitted max line count.
ALLOWLIST = {"__init__.py": 120}


def _line_count(path):
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def test_every_module_within_line_budget():
    violations = []
    for path in sorted(POPS.rglob("*.py")):
        rel = path.relative_to(POPS).as_posix()
        limit = ALLOWLIST.get(rel, DEFAULT_MAX_LINES)
        lines = _line_count(path)
        if lines > limit:
            violations.append("python/pops/%s: %d lines (limit %d)" % (rel, lines, limit))
    assert not violations, (
        "files exceed the Spec 4 36.3 line budget:\n  " + "\n  ".join(violations)
    )
