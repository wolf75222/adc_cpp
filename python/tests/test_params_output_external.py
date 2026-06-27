"""Spec 5 (sec.5.12 / 5.14 / 5.17): the pops.params / pops.output / pops.external surface.

Typed scalar params (compile-time vs runtime, typed dtype, typed domain), typed output /
checkpoint / format / level policies, and typed compiled-brick references with manifest +
native id. All inert; the runtime consumes them. Needs only `import pops`.
"""
import sys

import pytest

pops = pytest.importorskip("pops")

from pops.math import Real, Integer, Bool  # noqa: E402
from pops.params import (RuntimeParam, ConstParam, DerivedParam,  # noqa: E402
                         Positive, NonNegative, Range, In, Constant)
from pops.output import (OutputPolicy, CheckpointPolicy, HDF5, Plotfile,  # noqa: E402
                         AllLevels, CoarseOnly, SelectedLevels)
from pops.external import CompiledBrickRef, ExternalBrick  # noqa: E402
import pops.descriptors as _desc  # noqa: E402


def test_math_dtypes():
    assert Real.name == "Real" and str(Integer) == "Integer" and repr(Bool) == "Bool"


def test_runtime_and_const_params():
    a = RuntimeParam("alpha", dtype=Real, default=1.0, domain=Positive())
    assert a.name == "alpha" and a.capabilities()["runtime"] is True
    assert a.options()["dtype"] == "Real"
    a.validate()  # default 1.0 satisfies Positive
    g = ConstParam("gamma", value=5.0 / 3.0)
    assert g.capabilities()["in_cache_key"] is True and g.value == 5.0 / 3.0
    assert DerivedParam("Te", expression="p/rho").category == "derived_param"


def test_param_domain_rejects_bad_default():
    bad = RuntimeParam("nu", dtype=Real, default=-1.0, domain=Positive())
    with pytest.raises(ValueError):
        bad.validate()


def test_constraints():
    Positive().check(2.0)
    with pytest.raises(ValueError):
        Positive().check(0.0, who="alpha")
    NonNegative().check(0.0)
    Range(0.0, 1.0).check(0.5)
    with pytest.raises(ValueError):
        Range(0.0, 1.0).check(2.0)
    with pytest.raises(ValueError):
        Range(1.0, 0.0)  # lo > hi
    In("a", "b").check("a")
    with pytest.raises(ValueError):
        In("a", "b").check("z")


def test_constant_with_unit():
    c = Constant("c", 2.998e8, unit="m/s")
    assert c.options()["unit"] == "m/s" and c.value == 2.998e8


def test_output_and_checkpoint_policies():
    out = OutputPolicy(format=HDF5(parallel=True), cadence=20, fields=["phi", "E"],
                       levels=AllLevels(), require_parallel=True)
    assert out.options()["format"] == "HDF5" and out.options()["levels"] == "all"
    assert out.requirements()["parallel_io"] is True
    assert HDF5(parallel=True).requirements()["parallel_io"] is True
    assert Plotfile().capabilities()["per_level"] is True
    assert SelectedLevels(0, 1).options()["levels"] == (0, 1)
    assert CoarseOnly().options()["levels"] == "coarse"
    chk = CheckpointPolicy(restartable=True, require_bit_identical=True)
    assert chk.options()["restartable"] is True


def test_external_brick_ref_resolves_from_json_manifest(tmp_path):
    _desc._clear_external_catalog()
    manifest = tmp_path / "bricks.json"
    manifest.write_text(
        '{"bricks": [{"id": "my_ext_hll", "category": "riemann", '
        '"requirements": "physical_flux,wave_speeds"}]}', encoding="utf-8")
    ref = CompiledBrickRef(manifest=str(manifest), native_id="my_ext_hll",
                           expect_category="riemann")
    assert ref.available()  # registers + resolves
    d = ref.resolve()
    assert d.brick_type == "external_cpp"
    assert "physical_flux" in d.requirements.get("capabilities", [])
    assert ExternalBrick is CompiledBrickRef
    _desc._clear_external_catalog()


def test_external_brick_ref_missing_is_explainable(tmp_path):
    _desc._clear_external_catalog()
    ref = CompiledBrickRef(manifest=str(tmp_path / "none.json"), native_id="nope")
    av = ref.available()
    assert not av.ok and "could not be resolved" in av.reason


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
