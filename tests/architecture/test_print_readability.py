"""Spec 5 sec.12.1 (criteria #40-41): print(obj) is short, stable and array-free.

A user printing one of the main typed objects must get a SHORT, deterministic, readable
summary -- never an array dump, a full C++ struct, or a multi-kilobyte IR. This test builds
each headline object and asserts three properties of ``str(obj)``:

* short -- under 800 characters (a one-line / few-line summary, not a data dump);
* deterministic -- ``str(x) == str(x)`` (no memory address, no run-dependent ordering);
* array-free -- no ``array(`` / ``ndarray`` substring (no numpy payload leaks into print).

Like the other importing architecture tests, this one needs the compiled ``_pops`` extension
(the runtime bricks ``pops.Spatial`` / ``pops.FiniteVolume`` come from the runtime layer). If
``_pops`` cannot be loaded the whole module is skipped, not failed.
"""
import pytest

# Skip the whole module if the native extension cannot be loaded in this interpreter (mirrors
# test_public_imports: _bootstrap raises a custom ImportError whose .name does not match
# "pops._pops", so importorskip would re-raise instead of skipping).
try:
    import pops._pops  # noqa: F401
except Exception as _exc:  # pragma: no cover - exercised only without a built extension
    pytest.skip("compiled _pops extension not importable: %s" % _exc, allow_module_level=True)

import pops  # noqa: E402
import pops.codegen as codegen  # noqa: E402
import pops.diagnostics as diagnostics  # noqa: E402
import pops.fields as fields  # noqa: E402
import pops.linalg.norms as norms  # noqa: E402
import pops.math as pmath  # noqa: E402
import pops.mesh.layouts as layouts  # noqa: E402
import pops.output as output  # noqa: E402
import pops.params as params  # noqa: E402
import pops.solvers.elliptic as elliptic  # noqa: E402
import pops.solvers.krylov as krylov  # noqa: E402

_MAX_PRINT_LEN = 800


def _objects():
    """Construct one instance of every headline user object (label -> obj).

    Every object is INERT (a typed descriptor / authoring record); nothing here touches
    _pops, numpy, the runtime or codegen.
    """
    mesh = pops.CartesianMesh(n=8, L=1.0)
    phi = pmath.Unknown("phi")
    prog = pops.time.Program("demo")
    prog.state("plasma")  # one op so the summary reports a non-zero op count
    return {
        # numerics scheme bricks (runtime layer).
        "Spatial": pops.Spatial(),
        "FiniteVolume": pops.FiniteVolume(),
        # compiled time-program authoring object (pure-Python SSA builder).
        "time.Program": prog,
        # mesh layouts.
        "layouts.Uniform": layouts.Uniform(mesh),
        "layouts.AMR": layouts.AMR(mesh),
        # field problems.
        "fields.FieldProblem": fields.FieldProblem(unknown=phi),
        "fields.PoissonProblem": fields.PoissonProblem(unknown=phi),
        # elliptic + krylov solvers.
        "solvers.elliptic.GeometricMG": elliptic.GeometricMG(),
        "solvers.krylov.GMRES": krylov.GMRES(),
        "solvers.krylov.CG": krylov.CG(),
        # codegen optimization.
        "codegen.Optimization": codegen.Optimization(),
        # runtime params.
        "params.RuntimeParam": params.RuntimeParam("nu", default=0.1),
        # output policy.
        "output.OutputPolicy": output.OutputPolicy(),
        # diagnostics measures.
        "diagnostics.Norm": diagnostics.Norm(norms.L2()),
        "diagnostics.Integral": diagnostics.Integral(),
    }


def test_print_is_short():
    for label, obj in _objects().items():
        text = str(obj)
        assert len(text) < _MAX_PRINT_LEN, (
            "str(%s) is %d chars (>= %d): print(obj) must be a short summary, not a dump"
            % (label, len(text), _MAX_PRINT_LEN))


def test_print_is_deterministic():
    for label, obj in _objects().items():
        first, second = str(obj), str(obj)
        assert first == second, (
            "str(%s) is not deterministic (a memory address or run-dependent ordering "
            "leaked into print)" % label)


def test_print_has_no_array_dump():
    for label, obj in _objects().items():
        text = str(obj)
        assert "array(" not in text and "ndarray" not in text, (
            "str(%s) contains a raw array dump; print(obj) must stay numerics-free" % label)


def test_print_is_not_default_object_repr():
    # A readable summary must not be the bare ``<module.Class object at 0x...>`` Python falls
    # back to when no __str__ is defined: that leaks a memory address and tells the user nothing.
    for label, obj in _objects().items():
        text = str(obj)
        assert text, "str(%s) is empty" % label
        assert "object at 0x" not in text, (
            "str(%s) is the default object repr (leaks a memory address, unreadable): %r"
            % (label, text))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
