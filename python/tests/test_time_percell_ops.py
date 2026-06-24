#!/usr/bin/env python3
"""adc.time per-cell op completeness (epic ADC-399, ADC-434): field-vs-field ``cell_compare`` /
``where`` and a CUSTOM ``project``.

ADC-434 lifts two "later phase" guards on the per-cell ops:

  1. ``P.cell_compare(field, threshold, cmp)`` now accepts a FIELD threshold (a State/RHS/scalar_field),
     not only a Python float: the per-cell mask compares component 0 of @p field against component 0 of
     the threshold field (``maskA = fieldA(i,j,0) <cmp> rhsA(i,j,0)``). The constant-threshold path is
     unchanged (byte-identical lowering).
  2. ``P.project(state, projection=fn)`` now accepts a CUSTOM projection callable ``fn(P, U) -> U`` that
     BUILDS a per-cell map (an affine combine + ``cell_compare`` / ``where`` + named ``source`` /
     ``apply``) lowered through the existing per-cell kernels, the result copied back into the state in
     place. ``projection="block"`` (the native ``ctx.apply_projection``) is unchanged.

(A) Codegen (pure Python, always runs): a field-vs-field ``cell_compare`` builds with a second input and
    lowers to a per-cell kernel reading ``rhsA(i,j,0)``; the float path keeps one input and the literal
    cast; a custom projection records a LOCAL sub-block and lowers to the per-cell select kernels + an
    in-place ``lincomb`` (NOT ``ctx.apply_projection``); the native ``"block"`` projection still lowers
    to ``ctx.apply_projection``; non-local / bad projections are rejected.

(B) End-to-end parity (skips unless the full toolchain is present):
      - ``where(A > B, A, B)`` with a field-vs-field mask, stepped once, compared to an offline numpy
        per-cell select -> bit-exact.
      - a CUSTOM projection (a positivity floor ``U <- where(U < 0, 0, U)``) stepped once, compared to an
        offline ``np.where`` -> bit-exact; the native positivity projection is left untouched.
    Self-skips without numpy / _adc / a compiler / Kokkos / install_program (never faking the engine).
"""
import sys


def _adc_time():
    try:
        import adc.time as t
    except Exception as exc:  # noqa: BLE001 -- adc not importable here -> skip, never fake
        print("skip test_time_percell_ops (adc.time unavailable: %s)" % exc)
        sys.exit(0)
    return t


# ---- (A) codegen: pure Python, always runs ----
def test_cell_compare_field_threshold_builds(t):
    # A FIELD threshold (ADC-434): a second input, no literal value attr.
    P = t.Program("p")
    A = P.state("blk")
    B = P.linear_combine("B", 0.5 * A)
    m = P.cell_compare(A, B, ">", name="mask")
    assert m.vtype == "scalar_field", "a field-vs-field compare still returns a 1-component mask"
    assert m.attrs["value"] is None, "a field threshold carries no literal value (got %r)" % m.attrs
    assert [i.op for i in m.inputs] == ["state", "linear_combine"], \
        "a field-vs-field compare records (field, threshold) inputs (got %r)" % [i.op for i in m.inputs]


def test_cell_compare_float_threshold_unchanged(t):
    # The constant-threshold path is unchanged: one input, the literal value attr.
    P = t.Program("p")
    A = P.state("blk")
    m = P.cell_ge(A, 0.5, name="mask")
    assert m.attrs["value"] == 0.5 and len(m.inputs) == 1, m.attrs


def test_cell_compare_field_threshold_codegen(t):
    P = t.Program("p")
    A = P.state("blk")
    B = P.linear_combine("B", 0.25 * A)
    mask = P.cell_compare(A, B, ">", name="mask")
    out = P.where(mask, A, B, name="out")
    P.commit("blk", out)
    src = P.emit_cpp_program()
    assert "const adc::ConstArray4 rhsA" in src, "the field-rhs read handle must be bound\n%s" % src
    assert "fieldA(i, j, 0) > rhsA(i, j, 0)" in src, \
        "the kernel must compare component 0 of field vs component 0 of the rhs field\n%s" % src


def test_cell_compare_float_threshold_codegen_no_rhs_handle(t):
    # The float path must NOT bind a rhs handle (byte-identical to pre-ADC-434).
    P = t.Program("p")
    A = P.state("blk")
    half = P.linear_combine("half", 0.5 * A)
    mask = P.cell_ge(A, 0.5, name="mask")
    out = P.where(mask, A, half, name="out")
    P.commit("blk", out)
    src = P.emit_cpp_program()
    assert "rhsA" not in src, "the float-threshold kernel must not emit a rhs field handle\n%s" % src
    assert "static_cast<adc::Real>(0.5))" in src, "the literal threshold must be cast inline\n%s" % src


def test_cell_compare_rejects_block_mismatch(t):
    P = t.Program("p")
    A = P.state("a")
    B = P.state("b")
    try:
        P.cell_compare(A, B, ">")
    except ValueError as exc:
        assert "same block" in str(exc), str(exc)
    else:
        raise AssertionError("a field threshold from another block must be rejected")


def test_cell_compare_rejects_non_field_non_float(t):
    P = t.Program("p")
    A = P.state("blk")
    try:
        P.cell_compare(A, "0.5", ">")
    except TypeError as exc:
        assert "float threshold or a" in str(exc), str(exc)
    else:
        raise AssertionError("a non-field non-float threshold must be rejected")


def _floor_program(t, *, name="floor_proj"):
    """U <- project(U, floor-at-zero): a CUSTOM projection mapping U to where(U < 0, 0, U) per cell.
    Model-free (cell_compare + where + linear_combine only)."""
    P = t.Program(name)
    U = P.state("blk")

    def floor0(P, u):
        neg = P.cell_lt(u, 0.0, name="neg")        # 1 where u < 0
        zero = P.linear_combine("zero", 0.0 * u)   # the floor value (0)
        return P.where(neg, zero, u, name="floored")  # per-cell: 0 if u < 0 else u

    proj = P.project(state=U, projection=floor0)
    P.commit("blk", proj)
    return P


def test_custom_projection_records_local_subblock(t):
    P = t.Program("p")
    U = P.state("blk")

    def floor0(P, u):
        neg = P.cell_lt(u, 0.0)
        zero = P.linear_combine("zero", 0.0 * u)
        return P.where(neg, zero, u)

    proj = P.project(state=U, projection=floor0)
    assert proj.op == "project" and proj.attrs["projection"] == "custom", proj.attrs
    ops = [w.op for w in proj.attrs["proj_block"]]
    assert ops == ["state", "cell_compare", "linear_combine", "where"], ops
    assert proj.attrs["proj_result"].op == "where", "the result is the where output"


def test_custom_projection_codegen(t):
    P = _floor_program(t)
    src = P.emit_cpp_program()
    assert "ctx.apply_projection" not in src, \
        "a custom projection must NOT call the native apply_projection\n%s" % src
    assert "adc::for_each_cell" in src, "the projection lowers to per-cell select kernels\n%s" % src
    assert "? aA(i, j, c) : bA(i, j, c)" in src, "the where select kernel must be present\n%s" % src
    # the projected result is written back into the state in place (the final lincomb).
    assert "ctx.lincomb(u0, static_cast<adc::Real>(0), u0, static_cast<adc::Real>(1)," in src, \
        "the projected result must be copied back into the state in place\n%s" % src
    # device-clean: no heap / std::function in the kernel.
    assert "std::function" not in src and "new " not in src, "the kernel must be device-clean\n%s" % src


def test_custom_projection_validates(t):
    P = _floor_program(t)
    assert P.validate() is True, "the custom-projection Program must validate"
    assert P._ir_hash(), "the IR must serialize to a stable hash"


def test_block_projection_unchanged(t):
    # The native "block" projection is byte-identical to before: ctx.apply_projection(idx, state).
    P = t.Program("p")
    U = P.state("blk")
    pb = P.project(state=U)  # default projection="block"
    assert pb.attrs["projection"] == "block" and "proj_block" not in pb.attrs, pb.attrs
    P.commit("blk", pb)
    src = P.emit_cpp_program()
    assert "ctx.apply_projection(0," in src, "the block path must call apply_projection\n%s" % src


def test_custom_projection_rejects_non_local(t):
    # A non-local op (rhs / divergence / solve_fields) inside the projection is rejected.
    P = t.Program("p")
    U = P.state("blk")

    def bad(P, u):
        return P.rhs(state=u, flux=True)  # a non-local divergence in a per-cell map

    try:
        P.project(state=U, projection=bad)
    except ValueError as exc:
        assert "not LOCAL" in str(exc), str(exc)
    else:
        raise AssertionError("a non-local op in a custom projection must be rejected")


def test_custom_projection_rejects_non_state_result(t):
    P = t.Program("p")
    U = P.state("blk")

    def bad(P, u):
        return P.cell_lt(u, 0.0)  # a 1-component mask scalar_field, not the projected State

    try:
        P.project(state=U, projection=bad)
    except ValueError as exc:
        assert "must return the projected State" in str(exc), str(exc)
    else:
        raise AssertionError("a custom projection must return a State")


def test_project_rejects_bad_projection(t):
    P = t.Program("p")
    U = P.state("blk")
    try:
        P.project(state=U, projection="positivity")
    except NotImplementedError as exc:
        assert "'block'" in str(exc) and "callable" in str(exc), str(exc)
    else:
        raise AssertionError("an unknown string projection must be rejected")


# ---- (B) end-to-end parity: skips unless the full toolchain is present ----
def _passive_model(dsl, name):
    """A minimal 1-variable model with NO Poisson coupling: solve_fields is inert and the per-cell ops
    need no fields. A complete compilable block (flux + primitive + eigenvalue)."""
    m = dsl.Model(name)
    (rho,) = m.conservative_vars("rho")
    u = m.primitive("u", 0.0 * rho)
    m.primitive_vars(rho=rho, u=u)
    m.conservative_from([rho])
    m.flux(x=[0.0 * rho], y=[0.0 * rho])
    m.eigenvalues(x=[0.0 * rho], y=[0.0 * rho])
    return m


def _where_field_vs_field_program(t, *, name):
    """U <- where(U > half, U, half) where half = 0.5*U is a FIELD threshold (the mask compares the
    state against a derived field, not a constant)."""
    P = t.Program(name)
    U = P.state("blk")
    half = P.linear_combine("half", 0.5 * U)
    mask = P.cell_compare(U, half, ">", name="mask")  # field-vs-field: U > 0.5*U  <=>  U > 0
    out = P.where(mask, U, half, name="out")
    P.commit("blk", out)
    return P


def _run_section_b(t):
    try:
        import numpy as np

        import adc
    except Exception as exc:  # noqa: BLE001 -- numpy / _adc unavailable in this interpreter
        print("-- (B) skipped: adc/numpy unavailable: %s --" % exc)
        return

    from adc import dsl

    n = 8
    if not hasattr(adc.System(n=n, L=1.0, periodic=True), "install_program"):
        print("-- (B) skipped: _adc lacks the install_program binding (rebuild _adc) --")
        return

    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")

    # ---- (B1) where with a FIELD-vs-FIELD mask: out = where(rho0 > 0.5*rho0, rho0, 0.5*rho0) ----
    rho0 = 0.3 + 0.5 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)  # straddles 0 -> mask varies
    try:
        compiled = adc.compile_problem(model=_passive_model(dsl, "wff_prog"),
                                       time=_where_field_vs_field_program(t, name="wff_step"))
        compiled_model = _passive_model(dsl, "wff_block").compile(backend="production")
    except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
        print("-- (B) skipped: compile_problem could not build the .so: %s --" % str(exc)[:160])
        return

    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.add_equation("blk", compiled_model,
                     spatial=adc.FiniteVolume(limiter="none", riemann="rusanov"),
                     time=adc.Explicit(method="euler"))
    sim.set_state("blk", np.stack([rho0]))
    sim.install_program(compiled.so_path)
    sim.step(0.05)  # dt is irrelevant: the select is dt-free
    out = np.array(sim.get_state("blk"))[0]

    half = 0.5 * rho0
    mask = rho0 > half
    ref = np.where(mask, rho0, half)
    err = float(np.abs(out - ref).max())
    n_a, n_b = int(mask.sum()), int((~mask).sum())
    print("  where(field>field) parity: max|compiled - offline| = %.2e  cells(a=%d, b=%d)"
          % (err, n_a, n_b))
    assert err <= 1e-15, "compiled field-vs-field where == offline np.where (max|d| = %.2e)" % err
    assert n_a > 0 and n_b > 0, "the field-vs-field select must be non-vacuous (a=%d b=%d)" % (n_a, n_b)

    # ---- (B2) a CUSTOM projection: U <- where(U < 0, 0, U) (a positivity floor) ----
    rho1 = 0.4 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)  # straddles 0 -> some cells floored
    try:
        compiled2 = adc.compile_problem(model=_passive_model(dsl, "floor_prog"),
                                        time=_floor_program(t, name="floor_step"))
        compiled_model2 = _passive_model(dsl, "floor_block").compile(backend="production")
    except RuntimeError as exc:
        print("-- (B2) skipped: compile_problem could not build the .so: %s --" % str(exc)[:160])
        return

    sim2 = adc.System(n=n, L=1.0, periodic=True)
    sim2.add_equation("blk", compiled_model2,
                      spatial=adc.FiniteVolume(limiter="none", riemann="rusanov"),
                      time=adc.Explicit(method="euler"))
    sim2.set_state("blk", np.stack([rho1]))
    sim2.install_program(compiled2.so_path)
    sim2.step(0.05)
    out2 = np.array(sim2.get_state("blk"))[0]

    ref2 = np.where(rho1 < 0.0, 0.0, rho1)  # the IDENTICAL per-cell floor the custom projection runs
    err2 = float(np.abs(out2 - ref2).max())
    n_floored = int((rho1 < 0.0).sum())
    moved = float(np.abs(out2 - rho1).max())
    print("  custom-projection parity: max|compiled - offline floor| = %.2e  floored=%d  max|x - U0| = %.2e"
          % (err2, n_floored, moved))
    assert err2 <= 1e-15, "compiled custom projection == offline floor (max|d| = %.2e)" % err2
    assert n_floored > 0, "the floor must touch some cells (got %d)" % n_floored
    assert moved > 1e-6, "the projection must change the floored cells (max|d| = %.2e)" % moved


def _run():
    t = _adc_time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    print("PASS test_time_percell_ops (A: %d checks)" % len(fns))
    _run_section_b(t)


if __name__ == "__main__":
    _run()
