"""Spec 3 section 20 / criterion 23: a custom solver written in a Python DSL.

``@pops.lib.solver`` registers a GENERATED-brick solver whose body BUILDS a solver
IR using matrix-free Krylov primitives (``ctx.norm2`` / ``ctx.residual`` / affine
``x + omega*r`` / ``ctx.while_``). The builder authors IR ONLY -- it never performs
Python numerics. The descriptor is selectable wherever a native solver is (it
mirrors ``pops.lib.solvers.GMRES()``), and the generated-C++ lowering + run is the
deferred C++ follow-up (``generate_solver_cpp`` raises a clear ADC-462 error rather
than fake a Python solve).

This example builds the IR of a textbook Richardson iteration and prints it.

Run: python3 examples/spec3/custom_richardson_solver.py
"""
import pops.lib as lib


@lib.solver(name="richardson", signature="(A, b)")
def richardson(ctx, A, b, *, omega=0.5, tol=1e-8, max_iter=200):
    """Richardson iteration x <- x + omega*(b - A x), authored as IR.

    Every operation builds an IR node: ``omega`` / ``tol`` are IR literals, the
    residual ``b - A x`` is an affine combine, and the convergence test is a
    runtime Bool. No float arithmetic runs on real data here.
    """
    x = ctx.zeros_like(b)
    it = ctx.scalar_int(0)
    # The convergence predicate is a BUILDER re-evaluated against the loop-updated x / it
    # each pass; it never freezes on the initial (zero) iterate.
    def converging():
        return ctx.logical_and(ctx.norm2(ctx.residual(A, x, b)) > tol,
                               it < ctx.scalar_int(max_iter))
    with ctx.while_(converging):
        r = ctx.residual(A, x, b)        # r = b - A x  (an affine IR combine)
        x = ctx.combine(x + omega * r)   # x <- x + omega*r  (omega is an IR literal)
        it = it + ctx.scalar_int(1)
    return x


def main():
    # The decorator returns a generated solver descriptor, selectable like a native one:
    # same category and a scheme string as pops.lib.solvers.GMRES().
    assert richardson.brick_type == "generated"
    assert richardson.category == lib.solvers.GMRES().category == "solver"
    assert richardson.scheme == "richardson"
    assert "richardson" in lib.solvers.registered()

    # Running the builder authors the IR (no Python numerics).
    ir = lib.build_solver_ir(richardson)
    print("custom solver:", richardson.name, "(scheme=%r)" % richardson.scheme)
    print("ir ops:", sorted(ir.op_kinds()))
    print("solution value:", ir.result.vtype, "<-", ir.result.op)
    print("nodes:", len(ir.nodes()))

    # The matrix-free Krylov primitives are all present, in the IR (not Python).
    ops = ir.op_kinds()
    for needed in ("norm2", "apply", "linear_combine", "while", "logical_and"):
        assert needed in ops, "missing IR op: %s" % needed

    # The generated-C++ lowering is honestly deferred.
    try:
        lib.generate_solver_cpp(richardson)
    except NotImplementedError as exc:
        assert "ADC-462" in str(exc)
        print("\ngenerated C++ lowering: deferred (%s)" % str(exc).splitlines()[0])

    print("\nOK: the custom solver is authored as IR, not computed in Python.")


if __name__ == "__main__":
    main()
