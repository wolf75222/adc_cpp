"""Tests for pops.linalg -- the abstract algebraic layer (Spec 5 sec.5.6).

The package NAMES ``A x = b``, the operators, the typed norms and the reductions. Every object
is an inert typed descriptor: it constructs, exposes ``options()`` / ``inspect()`` / ``__repr__``,
validates its operand types, and computes NOTHING in Python (no numpy). These tests assert that
contract.
"""
import pytest

import pops
from pops import linalg
from pops.descriptors import Descriptor
from pops.linalg import (
    LinearOperator, MatrixFreeOperator, LinearProblem, Residual,
    L1, L2, LInf, Dot, Norm2, dot, norm2,
)


class _Handle:
    """A minimal named vector handle (an unknown / rhs / operand reference) for the tests.

    A real operand is a typed field/state handle carrying a ``name``; the descriptors surface
    operands by that name. A bare string has no ``name``, so the descriptors fall back to its
    repr -- the tests use this stub to model the intended (named-handle) usage.
    """

    def __init__(self, name):
        self.name = name


# --- package surface --------------------------------------------------------------------
def test_package_exposed_on_pops():
    assert pops.linalg is linalg
    assert "linalg" in pops.__all__


def test_reexports_are_present():
    for name in ("LinearOperator", "MatrixFreeOperator", "LinearProblem", "Residual",
                 "L1", "L2", "LInf", "Dot", "Norm2", "dot", "norm2",
                 "operator", "problem", "norms", "reductions"):
        assert hasattr(linalg, name), name
        assert name in linalg.__all__, name


# --- operators --------------------------------------------------------------------------
def test_linear_operator_is_inert_descriptor():
    A = LinearOperator("laplacian", native_id="pops::DivEpsGrad")
    assert isinstance(A, Descriptor)
    assert A.category == "linear_operator"
    assert A.name == "laplacian"
    assert A.native_id == "pops::DivEpsGrad"
    assert A.options() == {"name": "laplacian"}
    assert A.capabilities() == {"matrix_free": False}
    info = A.inspect()
    assert info["category"] == "linear_operator"
    assert info["native_id"] == "pops::DivEpsGrad"
    assert "laplacian" in repr(A)


def test_linear_operator_default_native_id_is_none():
    A = LinearOperator("A")
    assert A.native_id is None


def test_matrix_free_operator_is_matrix_free():
    M = MatrixFreeOperator("stencil_apply")
    assert isinstance(M, Descriptor)
    assert M.category == "linear_operator"
    assert M.name == "stencil_apply"
    assert M.native_id is None
    assert M.capabilities() == {"matrix_free": True}
    assert M.options() == {"name": "stencil_apply"}
    assert "stencil_apply" in repr(M)


# --- problem ----------------------------------------------------------------------------
def test_linear_problem_constructs_and_inspects():
    A = LinearOperator("A", native_id="pops::DivEpsGrad")
    p = LinearProblem(operator=A, unknown=_Handle("phi"), rhs=_Handle("b"))
    assert isinstance(p, Descriptor)
    assert p.category == "linear_problem"
    opts = p.options()
    assert opts["operator"] == "A"
    assert opts["unknown"] == "phi"
    assert opts["rhs"] == "b"
    assert p.requirements() == {"operator": True, "unknown": True, "rhs": True}
    assert p.capabilities() == {"linear": True, "matrix_free": False}
    info = p.inspect()
    assert info["operator"] == "A" and info["unknown"] == "phi" and info["rhs"] == "b"
    assert "LinearProblem" in repr(p)


def test_linear_problem_named():
    A = LinearOperator("A")
    p = LinearProblem(operator=A, unknown="x", rhs="b", name="poisson")
    assert p.name == "poisson"
    assert p.options()["name"] == "poisson"


def test_linear_problem_matrix_free_capability_propagates():
    M = MatrixFreeOperator("apply")
    p = LinearProblem(operator=M, unknown="x", rhs="b")
    assert p.capabilities() == {"linear": True, "matrix_free": True}


def test_linear_problem_validate_accepts_linear_operator():
    A = LinearOperator("A")
    p = LinearProblem(operator=A, unknown="x", rhs="b")
    assert p.validate() is True
    assert p.available().ok


def test_linear_problem_validate_accepts_matrix_free_operator():
    M = MatrixFreeOperator("apply")
    p = LinearProblem(operator=M, unknown="x", rhs="b")
    assert p.validate() is True


def test_linear_problem_rejects_non_operator():
    p = LinearProblem(operator="laplacian", unknown="x", rhs="b")
    with pytest.raises(TypeError):
        p.validate()
    avail = p.available()
    assert not avail.ok
    assert avail.status == "no"
    assert "operator" in avail.missing


def test_linear_problem_rejects_none_operator():
    p = LinearProblem(operator=None, unknown="x", rhs="b")
    with pytest.raises(TypeError):
        p.validate()


# --- residual ---------------------------------------------------------------------------
def test_residual_names_b_minus_ax():
    A = LinearOperator("A")
    p = LinearProblem(operator=A, unknown="x", rhs="b", name="sys")
    r = Residual(p)
    assert isinstance(r, Descriptor)
    assert r.category == "residual"
    assert r.options() == {"problem": "sys"}
    assert r.validate() is True
    assert r.available().ok
    assert "Residual" in repr(r)


def test_residual_rejects_non_problem():
    r = Residual("not a problem")
    with pytest.raises(TypeError):
        r.validate()
    avail = r.available()
    assert not avail.ok
    assert "problem" in avail.missing


# --- norms (typed objects, not strings) -------------------------------------------------
@pytest.mark.parametrize("cls,kind", [(L1, "l1"), (L2, "l2"), (LInf, "linf")])
def test_norms_are_typed_descriptors(cls, kind):
    n = cls()
    assert isinstance(n, Descriptor)
    assert n.category == "norm"
    assert n.kind == kind
    assert n.options() == {"kind": kind}
    assert n.name == cls.__name__
    assert cls.__name__ in repr(n)


def test_norms_are_distinct_types():
    assert type(L1()) is not type(L2())
    assert type(L2()) is not type(LInf())
    # The whole point of Spec 5 sec.5.6: a norm is a TYPED object, not a string.
    assert not isinstance(L2(), str)


# --- reductions (name only; compute nothing) --------------------------------------------
def test_dot_builds_a_reduction_descriptor():
    a, b = _Handle("a"), _Handle("b")
    d = dot(a, b)
    assert isinstance(d, Dot)
    assert isinstance(d, Descriptor)
    assert d.category == "reduction"
    assert d.options() == {"op": "dot", "a": "a", "b": "b"}
    assert d.requirements() == {"operands": 2}
    # It only references the operands; it did NOT compute an inner product.
    assert d.a is a and d.b is b


def test_norm2_builds_a_reduction_descriptor():
    x = _Handle("x")
    n = norm2(x)
    assert isinstance(n, Norm2)
    assert isinstance(n, Descriptor)
    assert n.category == "reduction"
    assert n.options() == {"op": "norm2", "x": "x"}
    assert n.requirements() == {"operands": 1}
    assert n.x is x


def test_reductions_reference_handles_by_name():
    A = LinearOperator("A")
    p = LinearProblem(operator=A, unknown="x", rhs="b", name="sys")
    r = Residual(p)
    # norm2 of a residual references it by its descriptor name, computing nothing.
    n = norm2(r)
    assert n.options()["x"] == "Residual"


def test_reductions_compute_nothing_numeric():
    # Passing numbers must NOT trigger a numeric reduction -- the descriptor only NAMES it.
    d = dot(3, 4)
    assert d.options() == {"op": "dot", "a": "3", "b": "4"}
    n = norm2(5)
    assert n.options() == {"op": "norm2", "x": "5"}


# --- inertness: nothing in the package imports a runtime/compute backend ----------------
def test_modules_are_numpy_free_at_module_scope():
    import sys
    # importing pops.linalg must not have pulled numpy / _pops in on its own behalf.
    for mod in (linalg.operator, linalg.problem, linalg.norms, linalg.reductions):
        src = mod.__file__
        text = open(src).read()
        assert "import numpy" not in text, src
        assert "import _pops" not in text, src


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
