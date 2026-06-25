"""Spec 4 architecture test: pure-layer packages must not import runtime or _adc.

Checks that source files under the listed package directories do not contain
import statements that pull in _adc, adc.runtime, or (for physics) adc.codegen.

Expected to FAIL against the current (pre-Spec-4) tree.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ADC_PKG = REPO_ROOT / "python" / "adc"

# Map: package sub-path (relative to REPO_ROOT) -> forbidden token strings.
# A token is forbidden if any source line matches the pattern:
#   (^|\s)(import|from)\b.*<token>
PACKAGE_FORBIDDEN: dict[str, list[str]] = {
    "python/adc/ir":      ["_adc", "adc.runtime"],
    "python/adc/model":   ["_adc", "adc.runtime"],
    "python/adc/physics": ["_adc", "adc.runtime", "adc.codegen"],
    "python/adc/time":    ["_adc", "adc.runtime"],
    "python/adc/lib":     ["_adc", "adc.runtime"],
}

# Pre-compiled pattern: matches a line that has an import statement referencing
# a given token.  We build one pattern per token.
def _import_pattern(token: str) -> re.Pattern[str]:
    # Matches lines like:
    #   import _adc
    #   from _adc import ...
    #   import adc.runtime
    #   from adc.runtime import ...
    #   from adc.runtime.foo import ...
    #   adc._adc  (attribute access that references _adc)
    escaped = re.escape(token)
    return re.compile(
        r"(?:^|(?<=\s))(?:import|from)\b.*" + escaped,
        re.MULTILINE,
    )


def _check_package(pkg_relpath: str, forbidden_tokens: list[str]) -> list[str]:
    """Return a list of violation strings 'relpath:lineno: line text'."""
    pkg_dir = REPO_ROOT / pkg_relpath
    if not pkg_dir.is_dir():
        return []  # package does not exist yet — skip (forward-looking)

    patterns = {tok: _import_pattern(tok) for tok in forbidden_tokens}
    violations: list[str] = []

    for py_file in sorted(pkg_dir.rglob("*.py")):
        # Skip __pycache__
        if "__pycache__" in py_file.parts:
            continue

        source = py_file.read_text(encoding="utf-8", errors="replace")
        rel = str(py_file.relative_to(REPO_ROOT))

        for lineno, line in enumerate(source.splitlines(), 1):
            stripped = line.strip()
            for tok, pat in patterns.items():
                if pat.search(stripped):
                    violations.append(f"{rel}:{lineno}: [{tok}] {line.rstrip()}")

    return violations


def test_ir_no_runtime_imports():
    """python/adc/ir must not import _adc or adc.runtime."""
    v = _check_package("python/adc/ir", PACKAGE_FORBIDDEN["python/adc/ir"])
    assert not v, (
        "python/adc/ir contains forbidden runtime imports:\n"
        + "\n".join(f"  {x}" for x in v)
    )


def test_model_no_runtime_imports():
    """python/adc/model must not import _adc or adc.runtime."""
    v = _check_package("python/adc/model", PACKAGE_FORBIDDEN["python/adc/model"])
    assert not v, (
        "python/adc/model contains forbidden runtime imports:\n"
        + "\n".join(f"  {x}" for x in v)
    )


def test_physics_no_runtime_imports():
    """python/adc/physics must not import _adc, adc.runtime, or adc.codegen."""
    v = _check_package("python/adc/physics", PACKAGE_FORBIDDEN["python/adc/physics"])
    assert not v, (
        "python/adc/physics contains forbidden runtime imports:\n"
        + "\n".join(f"  {x}" for x in v)
    )


def test_time_no_runtime_imports():
    """python/adc/time must not import _adc or adc.runtime."""
    v = _check_package("python/adc/time", PACKAGE_FORBIDDEN["python/adc/time"])
    assert not v, (
        "python/adc/time contains forbidden runtime imports:\n"
        + "\n".join(f"  {x}" for x in v)
    )


def test_lib_no_runtime_imports():
    """python/adc/lib must not import _adc or adc.runtime."""
    v = _check_package("python/adc/lib", PACKAGE_FORBIDDEN["python/adc/lib"])
    assert not v, (
        "python/adc/lib contains forbidden runtime imports:\n"
        + "\n".join(f"  {x}" for x in v)
    )


# ---------------------------------------------------------------------------
# Script runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_ir_no_runtime_imports,
        test_model_no_runtime_imports,
        test_physics_no_runtime_imports,
        test_time_no_runtime_imports,
        test_lib_no_runtime_imports,
    ]
    for t in tests:
        try:
            t()
            print(f"OK  {t.__name__}")
        except AssertionError as exc:
            print(f"FAIL {t.__name__}: {exc}")
