#!/usr/bin/env python3
"""Emit a GENERATED custom-solver C++ kernel from the @pops.lib.solver IR (ADC-462).

Lowers a textbook Richardson solver authored in the ``@pops.lib.solver`` IR-DSL to a
self-contained C++ kernel (``pops.lib.generate_solver_cpp``) and writes it to a header
the C++ validation test (``tests/test_solver_codegen_generated.cpp``) includes. This is
the build-time half of the codegen->compile->run validation: the test compiles the
emitted kernel against the real ``pops::pops`` runtime and runs it on a known linear
system, comparing to the native ``pops::richardson_solve``.

Run standalone (no ``_pops`` extension needed -- the IR-authoring + codegen layers are
pure Python): ``python3 scripts/gen_solver_kernel.py <out_header>``.
"""
import importlib
import os
import sys
import types


def _load_lib():
    """Load ``pops.lib`` from source WITHOUT importing the ``pops`` package ``__init__``
    (which needs the compiled ``_pops`` extension). The IR-authoring and codegen layers
    are pure Python, so ``pops.lib`` and its ``pops.ir`` / ``pops.model`` dependencies
    import standalone under a synthetic ``pops`` package rooted at the source tree. Using
    the normal import machinery resolves ``pops.lib`` whether it is a flat module or a
    package (Spec 4) and follows its submodule imports without hard-coded file paths."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pops_dir = os.path.join(root, "python", "pops")
    pkg = types.ModuleType("pops")
    pkg.__path__ = [pops_dir]
    sys.modules["pops"] = pkg
    return importlib.import_module("pops.lib")


# omega / tol of the generated Richardson solver. They MUST match the constants the C++ validation
# test (tests/test_solver_codegen_generated.cpp) feeds the NATIVE richardson_solve so the two trace
# the same iterates and stop at the same residual level (parity). omega = 1e-3 under-relaxes the SPD
# Helmholtz operator A = I - 0.1*Lap on the 32x32 grid (lambda_max ~ 820, stable for omega < ~2.4e-3);
# tol is the ABSOLUTE residual L2 norm the loop breaks on.
GEN_OMEGA = 2.0e-3
GEN_ABS_TOL = 1.0e-8


def _build_richardson(lib):
    """Register the Richardson solver IR (the spec example): x <- x + omega*(b - A x),
    looping while ||b - A x|| > tol and it < max_iter. omega / tol are IR literals."""
    @lib.solver(name="richardson_gen", signature="(A, b)")
    def richardson(ctx, a, b):  # noqa: D401 - the IR builder
        x = ctx.zeros_like(b)
        it = ctx.scalar_int(0)

        def converging():
            return ctx.logical_and(
                ctx.norm2(ctx.residual(a, x, b)) > GEN_ABS_TOL,
                it < ctx.scalar_int(500000))

        with ctx.while_(converging):
            r = ctx.residual(a, x, b)
            x = ctx.combine(x + GEN_OMEGA * r)
            it = it + ctx.scalar_int(1)
        return x

    return richardson


def main(argv):
    if len(argv) != 2:
        sys.stderr.write("usage: gen_solver_kernel.py <out_header>\n")
        return 2
    lib = _load_lib()
    solver = _build_richardson(lib)
    src = lib.generate_solver_cpp(solver)
    out = argv[1]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="ascii") as handle:
        handle.write(src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
