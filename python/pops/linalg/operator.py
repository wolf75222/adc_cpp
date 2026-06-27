"""pops.linalg.operator -- typed linear-operator descriptors (Spec 5 sec.5.6).

The linear-algebra layer NAMES the operator ``A`` in ``A x = b``; it does not apply it.
:class:`LinearOperator` references an assembled / matrix-backed operator (optionally a real
native symbol), while :class:`MatrixFreeOperator` names an operator known only by its action
``x -> A x`` (no stored matrix). Both are inert descriptors: they carry the operator name and
a matrix-free capability flag, and compute nothing -- the C++ runtime applies the operator.
"""
from pops.descriptors import Descriptor


class LinearOperator(Descriptor):
    """A typed linear operator ``A`` (the matrix in ``A x = b``).

    ``LinearOperator("laplacian", native_id="pops::DivEpsGrad")`` names the operator and,
    optionally, the native C++ symbol that materialises it. It is inert: it declares the
    operator name and that it is NOT matrix-free (an assembled / matrix-backed operator); the
    runtime applies it. Use :class:`MatrixFreeOperator` when only the action ``x -> A x`` is
    available.
    """

    category = "linear_operator"

    def __init__(self, name, native_id=None):
        self._name = str(name)
        self.native_id = native_id if native_id is None else str(native_id)

    @property
    def name(self):
        return self._name

    def options(self):
        return {"name": self._name}

    def capabilities(self):
        return {"matrix_free": False}


class MatrixFreeOperator(Descriptor):
    """A matrix-free linear operator: known only by its action ``x -> A x``.

    ``MatrixFreeOperator("stencil_apply")`` names an operator that is never assembled into a
    stored matrix (the common case for a stencil / FFT / Schur action). It is inert: it carries
    the operator name and the matrix-free capability flag; the runtime supplies the action. It
    has no native symbol of its own, so :attr:`native_id` stays ``None``.
    """

    category = "linear_operator"

    def __init__(self, name):
        self._name = str(name)

    @property
    def name(self):
        return self._name

    def options(self):
        return {"name": self._name}

    def capabilities(self):
        return {"matrix_free": True}


__all__ = ["LinearOperator", "MatrixFreeOperator"]
