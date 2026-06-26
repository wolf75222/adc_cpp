#!/usr/bin/env python3
"""pops.time per-cell conditional select (epic ADC-399 spec section 17, ADC-418): ``P.where``.

``P.where(mask, a, b)`` is a PER-CELL select -- ``out(i,j,c) = mask ? a(i,j,c) : b(i,j,c)``
COMPONENT-WISE -- lowered to a condition INSIDE a Kokkos for_each_cell kernel (a ternary), NOT the
scalar runtime branch ``if_``. The 0/1 mask is built per cell with ``P.cell_ge`` / ``cell_gt`` /
``cell_lt`` / ``cell_le`` (a threshold on component 0 of a field).

(A) Codegen (pure Python, always runs): ``cell_ge`` + ``where`` build + lower to the select kernel
    (``for_each_cell`` + the ``maskA ? aA : bA`` ternary, component-wise over the runtime ncomp), and
    the validation (mismatched a/b ncomp / vtype / block rejected; a non-float threshold rejected; a
    mask whose ncomp is neither 1 nor a/b's ncomp rejected).

(B) End-to-end parity (skips unless the full toolchain is present): a per-cell select
    ``U <- where(U >= floor, U, 0.5*U)`` stepped once, compared to an offline numpy ``np.where`` doing
    the IDENTICAL per-cell select -> bit-exact. The IC straddles the threshold so SOME cells take a and
    SOME take b (non-vacuous). Self-skips without numpy / _pops / a compiler / Kokkos / install_program
    (never faking the engine).
"""
import sys


def _pops_time():
    try:
        import pops.time as t
    except Exception as exc:  # pops not importable here -> skip, never fake
        print("skip test_time_where (pops.time unavailable: %s)" % exc)
        sys.exit(0)
    return t


def _clamp_program(t, *, name="where_clamp", floor=0.5):
    """U <- where(U >= floor, U, 0.5*U): keep U where it is at/above the floor, halve it below.

    Uses only linear_combine + cell_ge + where, so the Program lowers with NO model (solve_fields is
    inert / absent); the select is decided per cell entirely in C++."""
    P = t.Program(name)
    U = P.state("blk")
    half = P.linear_combine("half", 0.5 * U)        # the 'b' branch: 0.5 * U
    mask = P.cell_ge(U, floor, name="mask")          # 1 where U >= floor, else 0
    clamped = P.where(mask, U, half, name="clamped")  # per-cell: U if mask else 0.5*U
    P.commit("blk", clamped)
    return P


# ---- (A) codegen: pure Python, always runs ----
def test_cell_ge_is_scalar_field(t):
    P = t.Program("p")
    U = P.state("blk")
    m = P.cell_ge(U, 0.5)
    assert m.vtype == "scalar_field", "cell_ge returns a 1-component mask scalar_field (got %r)" % m.vtype
    assert m.attrs["cmp"] == ">=" and m.attrs["value"] == 0.5, m.attrs
    assert P._ncomp(m) == 1, "the mask is 1-component"


def test_cell_compare_variants(t):
    P = t.Program("p")
    U = P.state("blk")
    assert P.cell_gt(U, 1.0).attrs["cmp"] == ">"
    assert P.cell_ge(U, 1.0).attrs["cmp"] == ">="
    assert P.cell_lt(U, 1.0).attrs["cmp"] == "<"
    assert P.cell_le(U, 1.0).attrs["cmp"] == "<="


def test_where_result_type(t):
    P = t.Program("p")
    U = P.state("blk")
    a = P.linear_combine("a", 1.0 * U)
    b = P.linear_combine("b", 0.5 * U)
    m = P.cell_ge(U, 0.5)
    w = P.where(m, a, b)
    assert w.vtype == "state", "where over states returns a State (got %r)" % w.vtype
    assert w.block == "blk", "where inherits a's block"
    assert [i.op for i in w.inputs] == ["cell_compare", "linear_combine", "linear_combine"], \
        "where inputs = (mask, a, b)"


def test_where_codegen(t):
    P = _clamp_program(t)
    src = P.emit_cpp_program()
    for frag in ("ctx.alloc_scalar_field(1, 1)",          # the mask field
                 "pops::for_each_cell",                      # the per-cell select kernel
                 "static_cast<pops::Real>(0.5))",            # the threshold in the compare kernel
                 "? static_cast<pops::Real>(1) : static_cast<pops::Real>(0)",  # 0/1 mask
                 "for (int c = 0; c < ncomp_; ++c)",        # component-wise select
                 "(mask_ncomp_ == 1) ? 0 : c",              # shared vs per-component mask
                 "? aA(i, j, c) : bA(i, j, c)"):            # the select ternary
        assert frag in src, "the generated select kernel must contain %r\n%s" % (frag, src)


def test_where_validates(t):
    P = _clamp_program(t)
    assert P.validate() is True, "the where Program must validate"
    assert P._ir_hash(), "the IR must serialize to a stable hash"


def test_where_rejects_mismatched_ncomp(t):
    # a / b ncomp must match (checked statically when both are scalar_fields).
    P = t.Program("p")
    a = P.scalar_field("a", ncomp=2)
    b = P.scalar_field("b", ncomp=3)
    m = P.scalar_field("m", ncomp=1)
    try:
        P.where(m, a, b)
    except ValueError as exc:
        assert "same ncomp" in str(exc), str(exc)
    else:
        raise AssertionError("where must reject a/b with mismatched ncomp")


def test_where_rejects_bad_mask_ncomp(t):
    # mask ncomp must be 1 or match a/b's ncomp.
    P = t.Program("p")
    a = P.scalar_field("a", ncomp=2)
    b = P.scalar_field("b", ncomp=2)
    m = P.scalar_field("m", ncomp=3)
    try:
        P.where(m, a, b)
    except ValueError as exc:
        assert "mask must be 1-component or match" in str(exc), str(exc)
    else:
        raise AssertionError("where must reject a mask whose ncomp is neither 1 nor a/b's ncomp")


def test_where_accepts_per_component_mask(t):
    # a mask with the SAME ncomp as a/b (a per-component mask) is allowed.
    P = t.Program("p")
    a = P.scalar_field("a", ncomp=2)
    b = P.scalar_field("b", ncomp=2)
    m = P.scalar_field("m", ncomp=2)
    w = P.where(m, a, b)
    assert w.vtype == "scalar_field" and P._ncomp(w) == 2, "where keeps a's ncomp"


def test_where_rejects_mismatched_vtype(t):
    P = t.Program("p")
    U = P.state("blk")
    sf = P.scalar_field("sf", ncomp=1)
    m = P.cell_ge(U, 0.5)
    try:
        P.where(m, U, sf)
    except ValueError as exc:
        assert "same value type" in str(exc), str(exc)
    else:
        raise AssertionError("where must reject a / b of different value types")


def test_cell_compare_rejects_non_float_threshold(t):
    P = t.Program("p")
    U = P.state("blk")
    try:
        P.cell_ge(U, U)  # a per-cell field threshold is a later phase
    except TypeError as exc:
        assert "float threshold" in str(exc), str(exc)
    else:
        raise AssertionError("cell_compare must reject a non-float threshold")


def test_cell_compare_rejects_bad_cmp(t):
    P = t.Program("p")
    U = P.state("blk")
    try:
        P.cell_compare(U, 0.5, "==")
    except ValueError as exc:
        assert "cmp must be one of" in str(exc), str(exc)
    else:
        raise AssertionError("cell_compare must reject an unsupported comparison")


# ---- (B) end-to-end parity: skips unless the full toolchain is present ----
def _run_section_b(t):
    try:
        import numpy as np

        import pops
    except Exception as exc:  # noqa: BLE001  -- numpy / _pops unavailable in this interpreter
        print("-- (B) skipped: pops/numpy unavailable: %s --" % exc)
        return None

    n = 8
    sim = pops.System(n=n, L=1.0, periodic=True)
    if not hasattr(sim, "install_program"):
        print("-- (B) skipped: _pops lacks the install_program binding (rebuild _pops) --")
        return None

    from pops import dsl

    # A minimal 1-variable model with NO Poisson coupling: solve_fields is inert and the select needs
    # no fields. A complete compilable block (flux + primitive + eigenvalue).
    def passive_model(name):
        m = dsl.Model(name)
        (rho,) = m.conservative_vars("rho")
        u = m.primitive("u", 0.0 * rho)  # passive advection at speed 0 (the Program never runs a rhs)
        m.primitive_vars(rho=rho, u=u)
        m.conservative_from([rho])
        m.flux(x=[0.0 * rho], y=[0.0 * rho])
        m.eigenvalues(x=[0.0 * rho], y=[0.0 * rho])
        return m

    floor = 0.5
    try:
        compiled = pops.compile_problem(
            model=passive_model("where_prog"),
            time=_clamp_program(t, name="where_step", floor=floor))
    except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
        print("-- (B) skipped: compile_problem could not build the .so: %s --" % str(exc)[:160])
        return None

    assert compiled.program_name == "where_step", "handle carries the program name"

    try:
        compiled_model = passive_model("where_block").compile(backend="production")
    except RuntimeError as exc:  # no compiler / no Kokkos visible
        print("-- (B) skipped: model compile could not build the .so: %s --" % str(exc)[:160])
        return None
    sim.add_equation("blk", compiled_model,
                     spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                     time=pops.Explicit(method="euler"))
    # An IC that STRADDLES the floor: a sine swinging through 0.5 so some cells are >= floor and some
    # are < floor (the select must genuinely vary per cell -- non-vacuous).
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho0 = 0.5 + 0.4 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("blk", np.stack([rho0]))

    sim.install_program(compiled.so_path)
    sim.step(0.05)  # dt is irrelevant: the select is dt-free
    out = np.array(sim.get_state("blk"))[0]

    # OFFLINE per-cell select: out = where(rho0 >= floor, rho0, 0.5*rho0). The IDENTICAL select the
    # compiled kernel runs cell by cell.
    mask = rho0 >= floor
    ref = np.where(mask, rho0, 0.5 * rho0)
    err = float(np.abs(out - ref).max())
    n_a = int(mask.sum())
    n_b = int((~mask).sum())
    moved = float(np.abs(out - rho0).max())
    print("  where parity: max|compiled - offline np.where| = %.2e  cells(a=%d, b=%d)  max|x - U0| = %.2e"
          % (err, n_a, n_b, moved))
    assert err <= 1e-15, "compiled where == offline np.where per cell (max|d| = %.2e)" % err
    assert n_a > 0 and n_b > 0, \
        "the select must be non-vacuous (some cells take a, some take b): got a=%d b=%d" % (n_a, n_b)
    assert moved > 1e-6, "the select must change the b-branch cells from U0 (max|d| = %.2e)" % moved
    return (err, n_a, n_b)


def _run():
    t = _pops_time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    print("PASS test_time_where (A: %d checks)" % len(fns))
    _run_section_b(t)


if __name__ == "__main__":
    _run()
