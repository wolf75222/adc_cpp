# Custom solvers

A solver in adc is a typed brick. It can be native C++, written in the Python DSL and
generated to C++, provided by an external C++ library, or specialized into `problem.so`. In
every case it runs in C++ and uses the shared HPC primitives (dot, norm, axpy, MPI
reductions, the scratch manager, the profiler); Python never iterates a Krylov loop.

## Native solvers (today)

The matrix-free Krylov solvers are native C++ free functions in
`include/adc/numerics/elliptic/linear/generic_krylov.hpp`:

| Solver | Symbol |
| --- | --- |
| Richardson | `adc::richardson_solve` |
| CG | `adc::cg_solve` |
| BiCGStab | `adc::bicgstab_solve` |
| GMRES | `adc::gmres_solve` |

They are named by descriptors ({doc}`typed-bricks`): `adc.lib.solvers.CG()`,
`adc.lib.solvers.BiCGStab()`, `adc.lib.solvers.GMRES()`, `adc.lib.solvers.Richardson()`. A
compiled time Program drives them through `P.solve_linear(...)` ({doc}`time-program`); the
elliptic field solve uses the geometric multigrid (`adc::GeometricMG`).

## Generated solvers (design)

A solver can be written in the Python DSL and generated to C++ -- it builds an IR, it does not
compute in Python:

```python
@adc.lib.solver(name="richardson", signature="(A, b)")
def richardson(ctx, A, b, *, omega=0.5, tol=1e-8, max_iter=200):
    x = ctx.zeros_like(b)
    it = ctx.scalar_int(0)
    # The convergence predicate is a BUILDER re-evaluated against the loop-updated x
    # each pass; passing a pre-built Bool would freeze the test on the initial iterate.
    def converging():
        return ctx.logical_and(ctx.norm2(ctx.residual(A, x, b)) > tol,
                               it < ctx.scalar_int(max_iter))
    with ctx.while_(converging):
        r = ctx.residual(A, x, b)          # r = b - A x
        x = ctx.combine(x + omega * r)     # x <- x + omega*r
        it = it + ctx.scalar_int(1)
    return x
```

The lowering emits C++ that uses the core primitives (dot / norm / axpy / linear_combine /
MPI reductions / scratch / matrix-free apply / profiler). Modes: `native`, `generated`,
`library`, `specialized`, `auto`.

```{admonition} Status
:class: note
The native Krylov solvers and the `adc.lib.solvers` descriptors exist; `adc.compile_library`
compiles a brick library to a real `.so` (see typed-bricks). The solver DSL `@adc.lib.solver`
authors a solver IR, but lowering that IR to a callable generated C++ kernel, the specialization
modes and external C++ solver registration are follow-ups; the Program `solve_linear` over the
native Krylov solvers is the supported path today.
```
