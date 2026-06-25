"""Spec 4 architecture test: required public API must be importable from adc.

Verifies that all Spec 4 public symbols are reachable from the installed
(or source-tree) package.  If _adc is not built, the entire test is skipped.

Expected to FAIL/SKIP against the current (pre-Spec-4) tree.
"""


def test_public_api():
    """All Spec 4 public symbols must be importable and non-None."""
    try:
        import adc  # noqa: F401
    except Exception:
        try:
            import pytest
            pytest.skip("adc/_adc not importable in this environment")
        except ImportError:
            return  # pytest not available; treat as skip

    # --- adc.physics ---
    from adc.physics import Model  # noqa: F401

    # --- adc.time ---
    from adc.time import Program  # noqa: F401

    # --- adc.lib.time ---
    from adc.lib.time import predictor_corrector_local_linear  # noqa: F401

    # --- adc.lib.models.moments ---
    from adc.lib.models.moments import HyQMOM15  # noqa: F401

    # --- adc.lib.moments ---
    from adc.lib.moments import CartesianVelocityMoments, MomentModel  # noqa: F401

    # --- top-level adc attributes ---
    import adc as _adc_mod
    assert _adc_mod.System is not None, "adc.System must not be None"
    assert _adc_mod.compile_problem is not None, "adc.compile_problem must not be None"
    assert _adc_mod.compile_library is not None, "adc.compile_library must not be None"


# ---------------------------------------------------------------------------
# Script runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        import adc  # noqa: F401
    except Exception:
        print("SKIP  test_public_api: adc/_adc not importable in this environment")
    else:
        try:
            test_public_api()
            print("OK  test_public_api")
        except AssertionError as exc:
            print(f"FAIL test_public_api: {exc}")
        except ImportError as exc:
            print(f"FAIL test_public_api: {exc}")
