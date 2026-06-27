"""pops.solvers.options -- typed smoother / coarse-solver sub-descriptors (Spec 5 sec.5.7).

The multigrid elliptic solver (:class:`pops.solvers.elliptic.GeometricMG`) is configured by
TYPED objects, not strings: a smoother (:class:`Chebyshev` / :class:`RedBlackGaussSeidel`) and
a coarse-grid solver (:class:`DirectSmallGrid`). These are inert descriptors -- they record
the choice and its knobs and compute nothing; the C++ multigrid kernel runs the smoother and
the coarse solve.
"""
from pops.descriptors import Descriptor


class Chebyshev(Descriptor):
    """A Chebyshev polynomial smoother of the given ``degree`` (pre/post-smooth stage).

    Inert: it names the smoother and its degree; the C++ multigrid kernel applies it.
    """

    category = "smoother"
    native_id = None

    def __init__(self, degree=2):
        if isinstance(degree, bool) or not isinstance(degree, int):
            raise TypeError("Chebyshev(degree=) must be a Python int; got %r" % (degree,))
        if degree < 1:
            raise ValueError("Chebyshev(degree=) must be >= 1; got %d" % degree)
        self.degree = int(degree)

    @property
    def name(self):
        return "chebyshev"

    def options(self):
        return {"degree": self.degree}

    def lower(self, context=None):
        return {"kind": "chebyshev", "degree": self.degree}


class RedBlackGaussSeidel(Descriptor):
    """A red-black Gauss-Seidel smoother (color-ordered for stencil parallelism). Inert."""

    category = "smoother"
    native_id = None

    @property
    def name(self):
        return "red_black_gauss_seidel"

    def options(self):
        return {}

    def lower(self, context=None):
        return {"kind": "red_black_gauss_seidel"}


class DirectSmallGrid(Descriptor):
    """A direct coarse-grid solve for hierarchies whose coarse level is below ``threshold``.

    ``threshold`` is the coarse-grid unknown count under which a dense direct solve replaces
    further coarsening. Inert: the C++ multigrid kernel performs the coarse solve.
    """

    category = "coarse_solver"
    native_id = None

    def __init__(self, threshold=100):
        if isinstance(threshold, bool) or not isinstance(threshold, int):
            raise TypeError(
                "DirectSmallGrid(threshold=) must be a Python int; got %r" % (threshold,))
        if threshold < 1:
            raise ValueError("DirectSmallGrid(threshold=) must be >= 1; got %d" % threshold)
        self.threshold = int(threshold)

    @property
    def name(self):
        return "direct_small_grid"

    def options(self):
        return {"threshold": self.threshold}

    def lower(self, context=None):
        return {"kind": "direct_small_grid", "threshold": self.threshold}


__all__ = ["Chebyshev", "RedBlackGaussSeidel", "DirectSmallGrid"]
