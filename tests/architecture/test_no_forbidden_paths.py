"""Spec 4 architecture test: forbidden flat-file paths must not exist.

These tests encode the Spec 4 acceptance criteria for the python/adc package
layout.  They are expected to FAIL against the current (pre-Spec-4) tree.
"""

from pathlib import Path

# Repo root: this file lives at tests/architecture/test_no_forbidden_paths.py
REPO_ROOT = Path(__file__).resolve().parents[2]
ADC_PKG = REPO_ROOT / "python" / "adc"


# ---------------------------------------------------------------------------
# Forbidden flat files
# ---------------------------------------------------------------------------

FORBIDDEN_FILES = [
    "python/adc/dsl.py",
    "python/adc/model.py",
    "python/adc/time.py",
    "python/adc/moments.py",
    "python/adc/physics.py",
    "python/adc/lib.py",
    "python/adc/library.py",
    "python/adc/library_codegen.py",
    "python/adc/integrate.py",
    "python/adc/lib/models/moments/custom.py",
]

FORBIDDEN_DIRS = [
    "python/adc/time/std",
    "python/adc/lib/std",
    "python/adc/lib/models/std",
    "python/adc/lib/moments/std",
]


def test_no_forbidden_flat_files():
    """Each explicitly-forbidden flat file must be absent."""
    present = [p for p in FORBIDDEN_FILES if (REPO_ROOT / p).exists()]
    assert not present, (
        "Forbidden flat files still exist (Spec 4 requires package dirs):\n"
        + "\n".join(f"  {p}" for p in present)
    )


def test_no_forbidden_std_dirs():
    """Each explicitly-forbidden 'std' directory must be absent."""
    present = [p for p in FORBIDDEN_DIRS if (REPO_ROOT / p).exists()]
    assert not present, (
        "Forbidden 'std' directories still exist:\n"
        + "\n".join(f"  {p}" for p in present)
    )


def test_no_std_dir_anywhere_under_adc():
    """No directory named exactly 'std' may appear anywhere under python/adc/."""
    if not ADC_PKG.is_dir():
        return  # package does not exist yet — nothing to check
    std_dirs = [
        str(p.relative_to(REPO_ROOT))
        for p in ADC_PKG.rglob("std")
        if p.is_dir()
    ]
    assert not std_dirs, (
        "Directories named 'std' found under python/adc/ (must be removed for Spec 4):\n"
        + "\n".join(f"  {p}" for p in std_dirs)
    )


def test_no_custom_py_under_lib_models():
    """No file named 'custom.py' may exist under python/adc/lib/models/."""
    lib_models = ADC_PKG / "lib" / "models"
    if not lib_models.is_dir():
        return  # directory not yet present — nothing to check
    custom_files = [
        str(p.relative_to(REPO_ROOT))
        for p in lib_models.rglob("custom.py")
    ]
    assert not custom_files, (
        "Files named 'custom.py' found under python/adc/lib/models/ (forbidden in Spec 4):\n"
        + "\n".join(f"  {p}" for p in custom_files)
    )


# ---------------------------------------------------------------------------
# Script runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_no_forbidden_flat_files,
        test_no_forbidden_std_dirs,
        test_no_std_dir_anywhere_under_adc,
        test_no_custom_py_under_lib_models,
    ]
    for t in tests:
        try:
            t()
            print(f"OK  {t.__name__}")
        except AssertionError as exc:
            print(f"FAIL {t.__name__}: {exc}")
