# Custom solvers

A solver in adc is a typed brick. It can be native C++, written in the Python DSL and
generated to C++, provided by an external C++ library, or specialized into `problem.so`. In
every case it runs in C++ and uses the shared HPC primitives (dot, norm, axpy, MPI
reductions, the scratch manager, the profiler); Python never iterates a Krylov loop.

## Native solvers (today)

The matrix-free Krylov solvers are native C++ free functions in
`include/pops/numerics/elliptic/linear/generic_krylov.hpp`:

| Solver | Symbol |
| --- | --- |
| Richardson | `pops::richardson_solve` |
| CG | `pops::cg_solve` |
| BiCGStab | `pops::bicgstab_solve` |
| GMRES | `pops::gmres_solve` |

They are named by descriptors ({doc}`typed-bricks`): `pops.lib.solvers.CG()`,
`pops.lib.solvers.BiCGStab()`, `pops.lib.solvers.GMRES()`, `pops.lib.solvers.Richardson()`. A
compiled time Program drives them through `P.solve_linear(...)` ({doc}`time-program`); the
elliptic field solve uses the geometric multigrid (`pops::GeometricMG`).

## Generated solvers (design)

A solver can be written in the Python DSL and generated to C++ -- it builds an IR, it does not
compute in Python:

```python
@pops.lib.solver(name="richardson", signature="(A, b)")
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

`pops.lib.generate_solver_cpp(solver)` lowers the IR to a self-contained C++ kernel:

```cpp
template <class Op>
pops::KrylovResult richardson_solve(const Op& A, pops::MultiFab& x, const pops::MultiFab& b) {
  // scratch fields allocated ONCE, before the loop
  x.set_val(0);                              // warm start: zeros_like(b)
  for (;; ++pops_iters) {                     // a REAL C++ loop; the predicate re-evaluates
    A(v, x);                                 // matrix-free A(x) (template Op, inlined)
    pops::saxpy(r, 1.0, b); pops::saxpy(r, -1.0, v);     // r = b - A x
    if (!((std::sqrt(pops::dot(r, r)) > tol) && (pops_iters < max_iter))) break;
    pops::saxpy(x, omega, r);                 // x <- x + omega*r
  }
}
```

The operator `A` is a value-typed **template parameter** (the shape the native Krylov loops
take), so it inlines: no type-erased indirection in the kernel, no Python callback in the loop,
no heap allocation inside it, and no per-cell name dispatch (criterion 24.9). The kernel calls
the shared matrix-free primitives (`pops::dot` / `pops::saxpy` / `pops::lincomb`); a DSL solver that
maps onto a native scheme keeps the `pops::*_solve` free functions as its backend. Modes:
`native`, `generated`, `library`, `specialized`, `auto`.

```{admonition} Status
:class: note
The native Krylov solvers, the `pops.lib.solvers` descriptors, the solver DSL (`@pops.lib.solver`
authoring + `generate_solver_cpp` C++ lowering), and `pops.compile_library` (compiling a brick
library to a real `.so`, see typed-bricks) exist. The specialization modes and external C++
solver registration are follow-ups; the Program `solve_linear` over the native Krylov solvers is
the supported runtime path today.
```
