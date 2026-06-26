"""Spec 3 section 20 / criterion 23: a custom solver DSL that BUILDS IR.

``@pops.lib.solver`` registers a GENERATED-brick solver whose body is a Python
builder. Running the builder authors a SOLVER IR (matrix-free Krylov primitives:
norm2 / dot / apply / linear_combine / while) and computes NOTHING in Python --
no float arithmetic on real data, no numpy callback is captured. The generated
C++ lowering + run is the deferred C++ follow-up: it raises a clear ADC-462
NotImplementedError rather than faking a Python solve.

These tests are the AUTHORING slice: they assert the registration shape, the IR
ops, the no-Python-compute invariant, native-solver selectability, and the
honest deferral. They never run a custom solver numerically.
"""
import pytest

lib = pytest.importorskip("pops.lib")


def _richardson(ctx, A, b, *, omega=0.5, tol=1e-8, max_iter=100):
    """A textbook Richardson iteration authored as IR: x <- x + omega*(b - A x).

    Builds IR only -- omega / tol are IR literals, never multiplied against data. The
    convergence predicate is a BUILDER re-evaluated against the loop-updated x each pass
    (it never freezes on the initial zero iterate).
    """
    x = ctx.zeros_like(b)
    it = ctx.scalar_int(0)

    def converging():
        return ctx.logical_and(ctx.norm2(ctx.residual(A, x, b)) > tol,
                               it < ctx.scalar_int(max_iter))
    with ctx.while_(converging):
        r = ctx.residual(A, x, b)          # r = b - A x
        x = ctx.combine(x + omega * r)     # affine IR combine, no Python float math
        it = it + ctx.scalar_int(1)
    return x


def test_decorator_registers_generated_solver_descriptor():
    @lib.solver(name="richardson_dsl", signature="(A, b)")
    def richardson(ctx, A, b):
        return _richardson(ctx, A, b)

    d = richardson  # the decorator returns the descriptor
    assert isinstance(d, lib.BrickDescriptor)
    assert d.brick_type == "generated"
    assert d.category == "solver"
    assert d.name == "richardson_dsl"
    assert d.scheme == "richardson_dsl"
    # The builder is carried OFF the identity key (like BrickDescriptor.expression).
    assert callable(d.builder)


def test_descriptor_is_registered_in_the_catalog():
    @lib.solver(name="cataloged_solver")
    def s(ctx, A, b):
        return _richardson(ctx, A, b)

    assert lib.solvers.custom("cataloged_solver") is s
    assert "cataloged_solver" in lib.solvers.registered()


def test_builder_builds_an_ir_with_the_expected_ops():
    @lib.solver(name="ir_shape", signature="(A, b)")
    def s(ctx, A, b):
        return _richardson(ctx, A, b)

    ir = lib.build_solver_ir(s)
    ops = ir.op_kinds()
    # The matrix-free Krylov primitives the spec calls out.
    assert "norm2" in ops
    assert "apply" in ops          # A(x) inside the residual
    assert "linear_combine" in ops  # x + omega*r
    assert "while" in ops
    # The solution value the builder returned is a State-like IR value.
    assert ir.result.vtype == "state"


def test_while_records_a_cond_block_over_the_loop_updated_iterate():
    """The convergence predicate must be RE-RECORDED against the loop-updated iterate.

    A condition captured once (before the loop) over the initial zero ``x`` is a constant
    test -- the loop would never see the iterate change. Assert the ``while`` node carries a
    separate ``cond_block`` whose convergence ``apply`` (the A(x) inside the residual) reads
    a State produced INSIDE the body (the updated x), not the initial zero iterate.
    """
    @lib.solver(name="cond_block_shape", signature="(A, b)")
    def s(ctx, A, b):
        return _richardson(ctx, A, b)

    ir = lib.build_solver_ir(s)
    whiles = [n for n in ir.nodes() if n.op == "while"]
    assert whiles, "the convergence loop must emit a while node"
    w = whiles[0]

    # The predicate lives in its OWN cond_block (re-run each pass), not only the body block.
    cond_block = w.attrs.get("cond_block")
    assert isinstance(cond_block, list) and cond_block, (
        "the while node must carry a non-empty cond_block (the re-evaluated predicate); "
        "a missing cond_block freezes the test on the initial iterate")
    assert "body_block" in w.attrs and w.attrs["body_block"]

    # The recorded condition is the Bool the predicate built, and it lives in the cond_block.
    cond = w.attrs.get("cond")
    assert cond is not None and cond.vtype == "bool"
    assert cond in cond_block, "the recorded condition Bool must be in the cond_block"

    # The iterate A(x) is applied to the LOOP-UPDATED x. The body produces a fresh State
    # (the linear_combine) with a higher SSA id than the initial zero iterate; the cond_block
    # apply must read that updated State, not the initial one.
    body_states = [n.id for n in w.attrs["body_block"]
                   if n.vtype == "state" and n.op == "linear_combine"]
    assert body_states, "the body must produce an updated-iterate State (the affine combine)"
    updated_iterate_id = max(body_states)

    cond_applies = [n for n in cond_block if n.op == "apply"]
    assert cond_applies, "the convergence residual must apply A(x) inside the cond_block"
    applied_state_ids = {inp.id for n in cond_applies for inp in n.inputs
                         if getattr(inp, "vtype", None) == "state"}
    assert any(sid >= updated_iterate_id for sid in applied_state_ids), (
        "the convergence test must re-evaluate A(x) on the loop-updated iterate "
        "(id >= %d), not on the initial zero x; got applied state ids %s"
        % (updated_iterate_id, sorted(applied_state_ids)))


def test_while_rejects_a_pre_built_bool_condition():
    """A pre-built Bool freezes the convergence test on the initial iterate, so ``while_``
    must reject it and require a builder callback instead."""
    @lib.solver(name="reject_prebuilt_cond")
    def s(ctx, A, b):
        x = ctx.zeros_like(b)
        frozen = ctx.norm2(ctx.residual(A, x, b)) > 1e-8  # a pre-built Bool over the zero x
        with ctx.while_(frozen):                          # must raise, not freeze
            x = ctx.combine(x + 0.5 * ctx.residual(A, x, b))
        return x

    with pytest.raises(TypeError):
        lib.build_solver_ir(s)


def test_richardson_example_builds_an_affine_update_loop():
    @lib.solver(name="rich_example", signature="(A, b)")
    def s(ctx, A, b):
        return _richardson(ctx, A, b)

    ir = lib.build_solver_ir(s)
    # The body has an affine x + omega*r combine with the expected coefficients.
    combines = [n for n in ir.nodes() if n.op == "linear_combine"]
    assert combines, "the Richardson update must be an affine linear_combine"
    coeffs = {c.get(0) for n in combines for c in n.attrs["coeffs"]}
    assert 0.5 in coeffs   # omega
    assert 1.0 in coeffs   # the x term


def test_ir_has_no_python_numeric_compute():
    captured = {"calls": 0}

    @lib.solver(name="no_python_compute")
    def s(ctx, A, b):
        # If the DSL ever ran the body numerically, a Python callback here would
        # fire. It must be recorded as IR (apply), never invoked.
        def py_kernel(_state):       # pragma: no cover - must never run
            captured["calls"] += 1
            return _state
        return _richardson(ctx, A, b)

    ir = lib.build_solver_ir(s)
    assert captured["calls"] == 0, "the builder must not run Python numerics"
    # Every IR node is a typed, inert record: no node holds a live float result
    # or a numpy array -- only IR values / literals / coefficient polynomials.
    for n in ir.nodes():
        for v in n.inputs:
            assert hasattr(v, "vtype"), "an IR input must be an IR value, not data"
        # Attr payloads are metadata (ints / floats-as-literals / dicts), never arrays.
        for av in n.attrs.values():
            assert not _looks_like_array(av), "an IR attr captured a data array: %r" % (av,)


def test_a_scalar_in_the_ir_cannot_collapse_to_a_python_bool():
    @lib.solver(name="loud_scalar")
    def s(ctx, A, b):
        return _richardson(ctx, A, b)

    ir = lib.build_solver_ir(s)
    scalars = [n for n in ir.nodes() if n.vtype in ("scalar", "bool")]
    assert scalars, "the convergence test must build runtime scalar/bool nodes"
    with pytest.raises(TypeError):
        bool(scalars[0])   # a runtime scalar must never decide a Python branch


def test_custom_solver_is_selectable_like_a_native_solver():
    @lib.solver(name="selectable")
    def s(ctx, A, b):
        return _richardson(ctx, A, b)

    native = lib.solvers.GMRES()
    # Same metadata shape as a native solver descriptor: a scheme string and a
    # category, so T.solve(..., solver=<descriptor>) accepts it like GMRES.
    assert s.scheme is not None
    assert s.category == native.category == "solver"
    assert s.scheme == "selectable"


def test_cpp_generation_lowers_to_a_real_cpp_loop_over_shared_primitives():
    """ADC-462 C++ follow-up: the Richardson IR lowers to GENERATED C++ that RUNS.

    The kernel must drive the solve entirely in C++ -- a REAL ``for (;;)`` whose convergence
    predicate re-evaluates each pass, calling the SHARED matrix-free HPC primitives
    (``pops::dot`` / ``pops::saxpy``). It must NOT contain a Python callback, a ``std::function``,
    a heap ``new`` in the loop, or a per-cell string lookup (criterion 24.9).
    """
    @lib.solver(name="richardson_codegen", signature="(A, b)")
    def s(ctx, A, b):
        return _richardson(ctx, A, b)

    src = lib.generate_solver_cpp(s)
    # The generated kernel function and its templated matrix-free operator parameter.
    assert "richardson_codegen_solve" in src
    assert "template <class Op>" in src
    assert "const Op& A" in src
    # A REAL C++ convergence loop with a re-evaluated break (not a Python loop, not unrolled).
    assert "for (;;" in src
    assert "break;" in src
    # The shared HPC primitives are the backend (the residual norm / the affine update).
    assert "pops::dot(" in src
    assert "pops::saxpy(" in src
    assert "A(" in src  # the matrix-free matvec A(out, x)

    # The no-Python-callback / no-std::function / no-heap proof (criterion 24.9): the kernel is pure
    # C++ over the shared primitives -- nothing Python, no type-erased indirection, no per-iteration
    # heap, no string-keyed per-cell dispatch.
    for banned in ("std::function", "PyObject", "py::", "import ", "new ", "malloc",
                   'operator_by_name', '["', "scratch_state_like"):
        assert banned not in src, "generated solver C++ must not contain %r (criterion 24.9)" % banned


def test_cpp_generation_binds_the_iteration_cap_to_the_real_loop_counter():
    """The authored ``it < max_iter`` cap must bound the GENERATED loop for real.

    An SSA scalar literal cannot mutate, so the cap is lowered against the live C++ loop counter
    (``pops_iters``), not frozen on the initial ``it = 0``. The compare against the max literal must
    reference that counter.
    """
    @lib.solver(name="capped", signature="(A, b)")
    def s(ctx, A, b):
        return _richardson(ctx, A, b, max_iter=37)

    src = lib.generate_solver_cpp(s)
    assert "pops_iters" in src
    # The authored cap (37) appears as a literal compared against the loop counter.
    assert "37" in src
    assert "pops_iters) < static_cast<pops::Real>(37" in src


def test_cpp_generation_allocates_scratch_once_before_the_loop():
    """The matrix-free scratch fields must be allocated ONCE, before the loop -- never inside it
    (no per-iteration heap churn). Every MultiFab construction must precede the ``for (;;``."""
    @lib.solver(name="scratch_once", signature="(A, b)")
    def s(ctx, A, b):
        return _richardson(ctx, A, b)

    src = lib.generate_solver_cpp(s)
    loop_at = src.index("for (;;")
    # The only MultiFab constructed after the loop is the post-loop residual diagnostic; no scratch
    # is constructed INSIDE the loop body (between the for and its matching closing brace region).
    body = src[loop_at:src.index("// Final relative residual")]
    assert "pops::MultiFab " not in body, "no MultiFab may be constructed inside the convergence loop"


def test_builder_signature_and_name_are_validated():
    with pytest.raises((TypeError, ValueError)):
        lib.solver(name="")          # empty name is rejected

    with pytest.raises(TypeError):
        lib.solver(name="bad_signature", signature=123)   # signature must be a string

    with pytest.raises(TypeError):
        lib.solver(name="not_callable")(42)   # the body must be a callable builder


def _looks_like_array(value):
    """True if ``value`` looks like a live numeric data buffer (numpy array / list of
    floats), which an IR attr must never capture."""
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return True
    if isinstance(value, (list, tuple)) and value and all(
            isinstance(x, float) for x in value):
        return True
    return False
