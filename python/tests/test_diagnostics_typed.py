"""Spec 5 (sec.5.13 / 14.2.7): the typed pops.diagnostics measure descriptors.

``Norm`` / ``Integral`` / ``MinMax`` / ``ConservationCheck`` are inert typed descriptors: a
diagnostic is a typed OBJECT, not ``diagnostics.norm(kind="l2")``. ``Norm`` takes a typed
``pops.linalg.norms`` object and rejects a string; ``ConservationCheck`` takes a diagnostic
descriptor and rejects a string. They construct, expose ``options()`` / ``inspect()`` /
``__repr__``, carry their reduction metadata (kind, MPI reduction, cadence slot, AMR
compatibility) and compute NOTHING in Python. These tests assert that contract. Needs only
``import pops``.
"""
import subprocess
import sys

import pytest

pops = pytest.importorskip("pops")

from pops.descriptors import Descriptor  # noqa: E402
from pops.diagnostics import (ConservationCheck, Integral, MinMax,  # noqa: E402
                             Norm)
from pops.linalg.norms import L1, L2, LInf  # noqa: E402


class _Role:
    """A minimal named role handle (a model role reference) for the tests.

    A real role is a typed role object carrying a ``name``; the measures surface it by that
    name without interpreting it. This stub models the intended (named) usage.
    """

    def __init__(self, name):
        self.name = name


# --- package surface --------------------------------------------------------------------
def test_typed_measures_exported():
    import pops.diagnostics as diag
    for name in ("Norm", "Integral", "MinMax", "ConservationCheck"):
        assert hasattr(diag, name), name
        assert name in diag.__all__, name


# --- Norm: typed norm kind, string rejected ---------------------------------------------
@pytest.mark.parametrize("cls,kind", [(L1, "l1"), (L2, "l2"), (LInf, "linf")])
def test_norm_accepts_typed_norm_kind(cls, kind):
    n = Norm(cls(), block="ne", role=_Role("Density"))
    assert isinstance(n, Descriptor)
    assert n.category == "diagnostic_norm"
    assert n.options()["norm"] == kind
    assert n.options()["scheme"] == "norm"
    assert n.options()["block"] == "ne"
    assert n.options()["role"] == "Density"
    assert n.capabilities()["norm_kind"] == kind


def test_norm_rejects_string_kind():
    with pytest.raises(TypeError):
        Norm("l2")
    with pytest.raises(TypeError):
        Norm("l2", block="ne")


def test_norm_rejects_non_norm_object():
    with pytest.raises(TypeError):
        Norm(object())
    with pytest.raises(TypeError):
        Norm(Integral())  # an Integral is a measure, not a norm kind


# --- Integral / MinMax ------------------------------------------------------------------
def test_integral_is_a_sum_reduction():
    mass = Integral(role=_Role("Density"))
    assert isinstance(mass, Descriptor)
    assert mass.category == "diagnostic_integral"
    assert mass.options()["scheme"] == "integral"
    assert mass.options()["role"] == "Density"
    assert mass.options()["block"] is None
    assert mass.capabilities()["reduction"] == "sum"


def test_minmax_is_a_minmax_reduction():
    mm = MinMax(block="ne")
    assert mm.category == "diagnostic_minmax"
    assert mm.options()["scheme"] == "min_max"
    assert mm.options()["block"] == "ne"
    assert mm.capabilities()["reduction"] == "min_max"


# --- ConservationCheck: takes a diagnostic descriptor, rejects a string -----------------
def test_conservation_check_accepts_a_diagnostic():
    chk = ConservationCheck(Integral(role=_Role("Density")), tolerance=1e-12)
    assert isinstance(chk, Descriptor)
    assert chk.category == "conservation_check"
    assert chk.options()["tolerance"] == 1e-12
    assert chk.options()["quantity"] == "Integral"
    assert chk.validate() is True
    assert chk.available().ok


def test_conservation_check_default_tolerance():
    chk = ConservationCheck(Integral())
    assert chk.tolerance == 1e-12


def test_conservation_check_accepts_a_norm():
    chk = ConservationCheck(Norm(L2(), block="ne"))
    assert chk.validate() is True
    assert chk.options()["quantity"] == "Norm"


def test_conservation_check_rejects_string():
    chk = ConservationCheck("mass")
    with pytest.raises(TypeError):
        chk.validate()
    av = chk.available()
    assert not av.ok and av.status == "no"
    assert "quantity" in av.missing


def test_conservation_check_rejects_non_descriptor():
    chk = ConservationCheck(object())
    with pytest.raises(TypeError):
        chk.validate()


# --- metadata: reduction kind, MPI reduction, cadence, AMR/multi-level ------------------
@pytest.mark.parametrize("measure", [
    Norm(L2(), block="ne"), Integral(role="Density"), MinMax(block="ne"),
])
def test_measures_declare_reduction_metadata(measure):
    caps = measure.capabilities()
    assert caps["mpi_reduction"] is True
    assert caps["amr_compatible"] is True
    assert caps["multi_level"] is True
    assert caps["cadence_slot"] == "diagnostic"
    # the requirement mirrors the capability: a distributed reduction needs an all-reduce.
    assert measure.requirements()["mpi_reduction"] is True


def test_conservation_check_declares_metadata():
    chk = ConservationCheck(Integral(), tolerance=1e-9)
    caps = chk.capabilities()
    assert caps["reduction"] == "check"
    assert caps["mpi_reduction"] is True and caps["amr_compatible"] is True
    assert caps["multi_level"] is True and caps["cadence_slot"] == "diagnostic"
    assert chk.requirements()["quantity"] is True


# --- inspect() / options() / __repr__ (Spec 5 sec.12.1 printable rule) ------------------
@pytest.mark.parametrize("measure,cls_name,category", [
    (Norm(L2(), block="ne"), "Norm", "diagnostic_norm"),
    (Integral(role="Density"), "Integral", "diagnostic_integral"),
    (MinMax(block="ne"), "MinMax", "diagnostic_minmax"),
    (ConservationCheck(Integral()), "ConservationCheck", "conservation_check"),
])
def test_inspect_options_and_repr(measure, cls_name, category):
    info = measure.inspect()
    assert info["name"] == cls_name
    assert info["category"] == category
    assert info["scheme"] == measure.options()["scheme"]
    assert info["options"] == measure.options()
    # __repr__ is short and stable: it names the class and is deterministic across calls.
    assert cls_name in repr(measure)
    assert repr(measure) == repr(measure)
    # lower() is metadata only, carrying the native reduction scheme.
    assert measure.lower()["scheme"] == measure.options()["scheme"]


# --- inertness: the typed measures compute nothing and pull no runtime ------------------
def test_measures_module_is_numpy_and_runtime_free_at_source():
    import pops.diagnostics.measures as mod
    text = open(mod.__file__).read()
    for forbidden in ("import numpy", "import _pops", "pops.runtime", "pops.codegen"):
        assert forbidden not in text, "%s must not appear in %s" % (forbidden, mod.__file__)


def test_importing_measures_alone_does_not_pull_pops_runtime():
    # In a FRESH interpreter, importing only pops.diagnostics.measures must not load _pops.
    code = "import sys; import pops.diagnostics.measures; print('_pops' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", out.stdout + out.stderr


# --- factory / typed-class scheme dedup (ADC-506) ---------------------------------------
# The legacy diagnostics factories and the typed measure classes are NOT the same descriptor
# family (the factory is a BrickDescriptor with category "diagnostic" that the architecture
# consumers depend on; the typed class is a Descriptor with a finer category). What they MUST
# share -- and what ADC-506 dedups -- is the native reduction SCHEME label, now sourced ONCE
# from the typed class. These tests pin that the two paths lower to the same inert scheme and
# that neither computes.
def test_norm_factory_and_typed_norm_share_one_scheme():
    import pops.diagnostics as diag

    factory = diag.norm()
    typed = Norm(L2())
    # The scheme label is sourced from the typed class -- the factory reads Norm.scheme, so the
    # two agree by construction (no second literal to drift).
    assert typed.scheme == Norm.scheme
    assert factory.scheme == Norm.scheme
    assert factory.scheme == typed.options()["scheme"]
    # The historical factory return shape consumers depend on is untouched.
    assert factory.category == "diagnostic"
    assert factory.brick_type == "macro"


def test_integral_factory_and_typed_integral_share_one_scheme():
    import pops.diagnostics as diag

    factory = diag.integral()
    typed = Integral()
    assert typed.scheme == Integral.scheme
    assert factory.scheme == Integral.scheme
    assert factory.scheme == typed.options()["scheme"]
    assert factory.category == "diagnostic"


def test_conservation_check_factory_and_typed_class_share_one_scheme():
    import pops.diagnostics as diag

    factory = diag.invariants.conservation_check("charge")
    typed = ConservationCheck(Integral())
    assert typed.scheme == ConservationCheck.scheme
    assert factory.scheme == ConservationCheck.scheme
    # The invariant factory keeps its own (historical) category; only the scheme is unified.
    assert factory.category == "invariant"


def test_dedup_does_not_fabricate_a_runtime_path():
    # Both the factory descriptor and the typed measure stay inert: lower() returns a plain
    # metadata dict carrying the shared scheme, and neither exposes a compute/eval/call.
    import pops.diagnostics as diag

    factory = diag.norm()
    typed = Norm(L2())
    assert factory.lower()["scheme"] == typed.lower()["scheme"] == Norm.scheme
    for obj in (factory, typed):
        for attr in ("eval", "compile", "__call__", "run"):
            assert not hasattr(obj, attr), "%r must stay inert (has %s)" % (obj, attr)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
