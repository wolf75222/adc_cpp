"""Spec 4 architecture test: no source file under python/adc/ may exceed MAX_LINES.

An allowlist handles legitimately-large files (e.g. __init__.py).
Expected to FAIL against the current (pre-Spec-4) tree.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ADC_PKG = REPO_ROOT / "python" / "adc"

MAX_LINES = 500

# Allowlist: repo-root-relative path -> permitted line count.
ALLOWLIST: dict[str, int] = {
    "python/adc/__init__.py": 120,
}


def test_no_oversized_files():
    """Every *.py under python/adc/ must stay within its line-count budget."""
    if not ADC_PKG.is_dir():
        return  # package not present — nothing to check

    violations: list[str] = []

    for py_file in sorted(ADC_PKG.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue

        relpath = str(py_file.relative_to(REPO_ROOT))
        limit = ALLOWLIST.get(relpath, MAX_LINES)

        line_count = len(py_file.read_text(encoding="utf-8", errors="replace").splitlines())
        if line_count > limit:
            violations.append(
                f"{relpath}: {line_count} lines (limit {limit})"
            )

    assert not violations, (
        "Files exceeding their line-count budget (Spec 4 requires small, focused modules):\n"
        + "\n".join(f"  {v}" for v in violations)
    )


# ---------------------------------------------------------------------------
# Script runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        test_no_oversized_files()
        print("OK  test_no_oversized_files")
    except AssertionError as exc:
        print(f"FAIL test_no_oversized_files: {exc}")
