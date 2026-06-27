"""Spec 5 (sec.13.8): pops.codegen.Optimization + typed numeric math modes.

The codegen optimization policy and the numeric math mode are typed objects, not a string
``optimization="fast"``. A non-strict math transform is never implicit -- it must be chosen
explicitly. These are inert; the codegen consumes them.
"""
import sys

import pytest

pytest.importorskip("pops")
from pops.codegen import (  # noqa: E402
    Optimization, ConservativeFusion, Disabled,
    StrictMath, FastMath, DebugMath, GpuRegisterAware)


def test_default_optimization_is_strict():
    opt = Optimization()
    assert isinstance(opt.math, StrictMath)
    assert opt.capabilities()["strict_math"] is True
    assert opt.options()["cse"] is True and opt.options()["math"] == "StrictMath"
    assert opt.to_emit_kwargs() == {"cse": True, "hoist_reciprocals": True}
    assert Optimization.default().capabilities()["strict_math"] is True


def test_explicit_fast_math_is_not_strict():
    opt = Optimization(cse=False, hoist_reciprocals=False, math=FastMath())
    assert opt.capabilities()["strict_math"] is False
    assert FastMath().capabilities()["may_change_rounding"] is True
    assert opt.to_emit_kwargs() == {"cse": False, "hoist_reciprocals": False}


def test_math_modes():
    assert StrictMath().options()["fast_math"] is False
    assert DebugMath().options()["readable"] is True
    assert GpuRegisterAware().capabilities()["register_pressure_aware"] is True


def test_fusion_never_crosses_solve():
    fuse = ConservativeFusion(local_sources=True, flux_divergence=False, field_solves=False)
    assert fuse.capabilities()["crosses_solve"] is False
    assert fuse.capabilities()["crosses_reduction"] is False
    assert fuse.options()["field_solves"] is False
    assert Disabled().options()["enabled"] is False
    opt = Optimization(fuse=fuse)
    assert opt.options()["fuse"] == "ConservativeFusion"


def test_printable_summaries():
    s = str(Optimization())
    assert s.startswith("Optimization") and len(s) < 300
    assert repr(StrictMath()).startswith("StrictMath")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
