#!/usr/bin/env python3
"""Spec 5 sec.12.4: the ``POPS_*`` runtime-default environment variables.

These are the env vars that supply a DEFAULT for a no-argument Python call. The contract
(shared with ``POPS_PROFILE``) is additive: the env only sets the default, an explicit Python
argument always wins, and an unparseable value is ignored (no stricter rejection than passing the
argument directly).

All checks here are PURE: they exercise the resolver and ``set_threads`` argument handling without
``_pops`` or numpy, so they run in any interpreter. They never fake the engine -- ``set_threads``
writes only the standard ``OMP_NUM_THREADS`` / ``KOKKOS_NUM_THREADS`` knobs, which we read back.
"""
import os
import sys
import warnings

import pytest

from pops.runtime import threading as th
from pops.runtime.profile import Profile


# ---- POPS_THREADS resolver (lenient parsing) --------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    (None, None),       # unset -> caller falls back to os.cpu_count()
    ("", None),         # blank -> ignored
    ("   ", None),      # whitespace -> ignored
    ("8", 8),           # positive integer
    ("  4 ", 4),        # surrounding whitespace tolerated
    ("0", None),        # non-positive -> ignored, not raised
    ("-3", None),       # negative -> ignored
    ("nope", None),     # non-integer -> ignored, not raised
    ("2.5", None),      # float string -> ignored (int() would raise -> swallowed)
])
def test_threads_from_env_is_lenient(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("POPS_THREADS", raising=False)
    else:
        monkeypatch.setenv("POPS_THREADS", raw)
    assert th._threads_from_env() == expected


# ---- POPS_THREADS supplies the set_threads() default ------------------------------------------
def _run_set_threads(monkeypatch, *args):
    """Call set_threads with the late-init / serial guards neutralized, capture the threads written.

    We force a serial-but-not-yet-initialized module so set_threads reaches the env-write path
    without needing _pops; the SERIAL warning is expected and suppressed. Returns the int written to
    OMP_NUM_THREADS, or None if nothing was written (the early-return paths).
    """
    monkeypatch.setattr(th, "_first_system_built", False)
    monkeypatch.setattr(th, "has_kokkos", lambda: False)

    class _FakePops:
        @staticmethod
        def kokkos_is_initialized():
            return False

    import pops
    monkeypatch.setattr(pops, "_pops", _FakePops, raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        th.set_threads(*args)
    written = os.environ.get("OMP_NUM_THREADS")
    return int(written) if written is not None else None


def test_pops_threads_sets_default_for_no_argument(monkeypatch):
    monkeypatch.setenv("POPS_THREADS", "7")
    assert _run_set_threads(monkeypatch) == 7
    # Both backend-agnostic knobs are written to the same value.
    assert os.environ.get("KOKKOS_NUM_THREADS") == "7"


def test_explicit_argument_always_wins_over_pops_threads(monkeypatch):
    monkeypatch.setenv("POPS_THREADS", "7")
    # Explicit n overrides the env default -- never the other way around.
    assert _run_set_threads(monkeypatch, 3) == 3


def test_unset_pops_threads_falls_back_to_cpu_count(monkeypatch):
    monkeypatch.delenv("POPS_THREADS", raising=False)
    assert _run_set_threads(monkeypatch) == (os.cpu_count() or 1)


def test_unparseable_pops_threads_falls_back_to_cpu_count(monkeypatch):
    monkeypatch.setenv("POPS_THREADS", "garbage")
    assert _run_set_threads(monkeypatch) == (os.cpu_count() or 1)


# ---- POPS_PROFILE: documented default resolver is wired (coverage guard) ----------------------
def test_pops_profile_resolver_is_wired(monkeypatch):
    monkeypatch.setenv("POPS_PROFILE", "advanced")
    assert Profile.from_env() == Profile.Advanced()
    monkeypatch.setenv("POPS_PROFILE", "off")
    assert Profile.from_env(default=Profile.Basic()) == Profile.Basic()
    monkeypatch.delenv("POPS_PROFILE", raising=False)
    assert Profile.from_env() is None  # unset + no default -> profiling stays off


# ---- Coverage guard: every POPS_* runtime default in the doc table is actually read -----------
def test_documented_runtime_defaults_are_implemented(monkeypatch):
    # POPS_THREADS -> threading._threads_from_env; POPS_PROFILE -> Profile.from_env. If either
    # resolver stops reading its variable, this fails -- the doc would then be describing a dead knob.
    monkeypatch.setenv("POPS_THREADS", "5")
    assert th._threads_from_env() == 5
    monkeypatch.setenv("POPS_PROFILE", "full")
    assert Profile.from_env() == Profile.Advanced()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
