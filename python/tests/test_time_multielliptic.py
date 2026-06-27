#!/usr/bin/env python3
"""Named multi-elliptic-field runtime (m.elliptic_field), ADC-428 (epic ADC-399, completes ADC-419).

ADC-419 landed the IR + validation + hash for m.elliptic_field("phi2", rhs=, operator=, aux=[...]) but
P.solve_fields(field=name) raised NotImplementedError (the SECOND elliptic solve + its own aux channel
were unwired). ADC-428 wires the runtime on the production/system backend: a named field gets

  - its OWN RHS brick (a function of the conservative state, like m.elliptic_rhs),
  - a DEDICATED native elliptic solver instance (GeometricMG/FFT, reused -- not reimplemented),
  - its OWN aux output channel (the model's named aux_field slots, distinct from the shared phi/grad),

and P.solve_fields(field=name, state=U) lowers to ctx.solve_fields_from_state(field, block, U).

Section A (pure Python, always runs): the named solve_fields op lowers to the named ctx call (NOT the
default 2-arg one); the default solve_fields lowers byte-identically to before; unknown field / missing
model / aux-reading rhs / undeclared aux output are rejected with clear errors.

Section B (gated, self-skip): the OFFLINE REFERENCE is the engine's own default Poisson solve.

  - PARITY: a named field "phi2" with rhs = (the SAME RHS as the default Poisson coupling) solves the
    IDENTICAL elliptic problem with the SAME native solver, so its derived gradient (g2x/g2y) equals the
    default grad_x/grad_y. A program whose source reads phi2's gradient therefore steps the state
    bit-for-bit like a program whose source reads the default grad -> max|d| ~ round-off. This is a true
    second INDEPENDENT solve validated against the default one (no offline multigrid reimplementation).
  - DISTINCT RHS (linearity): a named field with rhs = 2 * (default RHS) produces phi2 = 2*phi (Poisson
    is linear), so g2x = 2*grad_x; the source reading 0.5*g2x reproduces the default-grad step -> the
    named field carries a genuinely DIFFERENT, correctly scaled field.
  - NO REGRESSION: a default-only model (no named field) stepped via a program is byte-identical to the
    same model stepped before this feature (the named code path is inert; asserted on the lowered C++).

Skips cleanly (exit 0) without numpy / _pops / a compiler / a visible Kokkos -- never fakes the engine.
"""
import sys


def _pops_mods():
    try:
        from pops.ir.ops import sqrt
        from pops.physics.facade import Model
        from pops import time as adctime
    except Exception as exc:  # pops not importable here -> skip, never fake
        print("skip test_time_multielliptic (pops unavailable: %s)" % exc)
        sys.exit(0)
    return Model, sqrt, adctime


Model, sqrt, adctime = _pops_mods()

fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


def raises(exc_types, fn):
    try:
        fn()
    except exc_types:
        return True
    except Exception:  # noqa: BLE001  -- wrong exception type is a failure
        return False
    return False


Q = -1.0  # charge sign (f = q * rho), like pops::ChargeDensity


# --- shared isothermal 2D fluid block (rho, mx, my; default Poisson f = q*rho) ---
def _block(m):
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    cs2 = m.param("cs2", 0.5)
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    p = m.primitive("p", cs2 * rho)
    m.primitive_vars(rho=rho, u=u, v=v, p=p)
    m.conservative_from([rho, rho * u, rho * v])
    cs = sqrt(cs2)
    m.eigenvalues(x=[u - cs, u, u + cs], y=[v - cs, v, v + cs])
    m.flux(x=[mx, mx * u + p, my * u], y=[my, mx * v, my * v + p])
    m.elliptic_rhs(Q * rho)  # default Poisson coupling: f = q * rho
    return rho, mx, my


def default_model(name="me_default"):
    """Default Poisson only; the source pushes momentum along the default electric field -grad phi."""
    m = Model(name)
    rho, mx, my = _block(m)
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    # S = (0, -rho*grad_x, -rho*grad_y): the standard electrostatic force on the momentum.
    m.source([0.0 * rho, -rho * gx, -rho * gy])
    return m


def named_model(name="me_named", scale=1.0, src_scale=1.0):
    """Default Poisson PLUS a named field 'phi2' with rhs = scale * (default RHS). The source reads
    phi2's OWN gradient (g2x/g2y, the named aux), multiplied by src_scale -- NOT the default grad."""
    m = Model(name)
    rho, mx, my = _block(m)
    # The named field's output aux components -- declared as model aux_field slots so a source can read
    # them and the runtime has a channel to write into.
    g2x = m.aux_field("g2x")
    g2y = m.aux_field("g2y")
    m.aux_field("phi2")  # declare the named field's potential aux slot (written C++-side, not read in this IR)
    m.elliptic_field("phi2", rhs=(scale * Q) * rho, aux=["phi2", "g2x", "g2y"])
    m.source([0.0 * rho, -src_scale * rho * g2x, -src_scale * rho * g2y])
    return m


# =================== Section A: pure Python ===================
print("== (A) m.elliptic_field lowering + validation ==")


def _prog(name, field=None):
    P = adctime.Program(name)
    U = P.state("plasma")
    if field is None:
        f = P.solve_fields(U)
    else:
        f = P.solve_fields("f_" + field, U, field=field)
    R = P.rhs(name="R", state=U, fields=f, flux=True)
    P.commit("plasma", P.linear_combine("U1", U + P.dt * R))
    return P


# default solve_fields lowers to the 2-arg ctx call (historical), named to the 3-arg ctx call.
src_default = _prog("me_def_prog").emit_cpp_program(model=default_model())
chk("ctx.solve_fields_from_state(0, " in src_default,
    "default solve_fields lowers to ctx.solve_fields_from_state(0, ...)")
chk('ctx.solve_fields_from_state("' not in src_default,
    "default solve_fields does NOT use the named (3-arg) overload")

src_named = _prog("me_nam_prog", field="phi2").emit_cpp_program(model=named_model())
chk('ctx.solve_fields_from_state("phi2", 0, ' in src_named,
    "named solve_fields lowers to ctx.solve_fields_from_state(\"phi2\", 0, ...)")

# The named brick + registration land in the native loader (production backend).
loader = named_model("me_nam_loader")._m.emit_cpp_native_loader(target="system")
chk("Ell_phi2" in loader, "the named elliptic RHS brick is emitted in the native loader")
chk('register_elliptic_field("phi2"' in loader, "the named field registers its aux components")
chk("set_block_elliptic_field" in loader and "make_poisson_rhs" in loader,
    "the named field attaches its RHS closure (make_poisson_rhs of the brick)")

# Validation: unknown field / missing model / aux-reading rhs / undeclared aux output / amr target.
chk(raises(ValueError, lambda: _prog("me_unknown", field="nope").emit_cpp_program(model=named_model())),
    "an unknown elliptic_field name in solve_fields raises ValueError")
chk(raises(NotImplementedError, lambda: _prog("me_nomodel", field="phi2").emit_cpp_program()),
    "a named solve_fields without a model raises NotImplementedError")


def _bad_rhs_aux():
    m = Model("me_badrhs")
    rho, mx, my = _block(m)
    m.elliptic_field("phi2", rhs=rho + m.aux("grad_x"))  # rhs reading aux -> rejected


chk(raises(ValueError, _bad_rhs_aux), "an elliptic_field rhs reading the aux channel is rejected")


def _bad_aux_out():
    m = Model("me_badaux")
    rho, mx, my = _block(m)
    m.elliptic_field("phi2", rhs=rho, aux=["never_declared"])  # output aux not an aux_field
    m._m._elliptic_field_registrations("Me_badauxGen")


chk(raises(ValueError, _bad_aux_out),
    "an elliptic_field whose aux output is not a declared aux_field is rejected")


def _amr_named():
    named_model("me_amr")._m.emit_cpp_native_loader(target="amr_system")


chk(raises(NotImplementedError, _amr_named),
    "a named elliptic field on target='amr_system' raises NotImplementedError (deferred)")


# The flat-ABI backends (aot: POPS_DEFINE_COMPILED_BLOCK; jit: extern "C" factory) emit the named RHS
# brick via the shared _emit_bricks but have NO hook to register the field on the System. Reject them
# loud at the EMIT boundary, not silently (a dropped field would only fail at runtime: "System: unknown
# named elliptic field"). Mirrors the target='amr_system' guard.
def _aot_named():
    named_model("me_aot")._m.emit_cpp_aot_source()


chk(raises(NotImplementedError, _aot_named),
    "a named elliptic field on backend='aot' raises NotImplementedError at emit (deferred)")


def _jit_named():
    named_model("me_jit")._m.emit_cpp_so_source()


chk(raises(NotImplementedError, _jit_named),
    "a named elliptic field on backend='jit' raises NotImplementedError at emit (deferred)")

# NO REGRESSION: a default-only model lowers IDENTICALLY whether or not the named feature exists. We
# assert the default program never emits the named (3-arg) ctx call (above) AND that adding a named
# field to a SECOND model leaves the default model's lowering untouched.
src_default2 = _prog("me_def_prog").emit_cpp_program(model=default_model())
chk(src_default == src_default2, "the default program lowers deterministically (no named-field leak)")


# =================== Section B: gated end-to-end parity ===================
print("== (B) named second elliptic solve == default Poisson solve (offline reference) ==")


def _skipB(msg):
    print("-- (B) skipped: %s --" % msg)
    print("%s test_time_multielliptic (A only)" % ("FAIL" if fails else "PASS"))
    sys.exit(1 if fails else 0)


try:
    import numpy as np

    import pops
except Exception as exc:  # noqa: BLE001
    _skipB("numpy/_pops unavailable: %s" % exc)

if not hasattr(pops.System(n=8, L=1.0, periodic=True), "install_program"):
    _skipB("_pops lacks the install_program binding (rebuild _pops)")

N = 16
DT = 0.005


def _ic():
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    mx = 0.2 * rho
    my = -0.1 * rho
    return np.stack([rho, mx, my])


def make_sim(model):
    sim = pops.System(n=N, L=1.0, periodic=True)
    try:
        compiled = model.compile(backend="production")
    except RuntimeError as exc:
        _skipB("model compile could not build the .so: %s" % str(exc)[:160])
    sim.add_equation("plasma", compiled,
                     spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                     time=pops.Explicit(method="euler"))
    sim.set_poisson("composite", "geometric_mg")  # f = sum of the per-block elliptic bricks
    sim.set_state("plasma", _ic())
    return sim


def step_program(model, prog):
    try:
        compiled = pops.compile_problem(model=model, time=prog)
    except RuntimeError as exc:
        _skipB("compile_problem could not build the .so: %s" % str(exc)[:160])
    sim = make_sim(model)
    sim.install_program(compiled.so_path)
    sim.step(DT)
    return np.array(sim.get_state("plasma"))


# REFERENCE: the default Poisson coupling, source reads the default grad.
U0 = _ic()
ref = step_program(default_model("me_ref"), _prog("me_ref_fe"))
chk(float(np.abs(ref - U0).max()) > 1e-9, "the default electrostatic source actually moved the state")

# PARITY: a named field with rhs == the default RHS solves the same problem with the same native
# solver, so g2x/g2y == grad_x/grad_y -> the named-field-driven step matches the default step.
got = step_program(named_model("me_par", scale=1.0, src_scale=1.0), _prog("me_par_fe", field="phi2"))
e_par = float(np.abs(got - ref).max())
print("  named(rhs=default) vs default Poisson: max|d| = %.2e" % e_par)
chk(e_par < 1e-12,
    "named second elliptic solve (same RHS) == default Poisson solve (max|d| = %.2e)" % e_par)

# DISTINCT RHS (linearity): named rhs = 2*default -> phi2 = 2*phi -> g2x = 2*grad_x; the source reads
# 0.5*g2x, recovering the default-grad step. Confirms the named field carries a genuinely different,
# correctly scaled field (not an alias of the shared aux).
got2 = step_program(named_model("me_lin", scale=2.0, src_scale=0.5),
                    _prog("me_lin_fe", field="phi2"))
e_lin = float(np.abs(got2 - ref).max())
print("  named(rhs=2*default, src=0.5*g2) vs default: max|d| = %.2e" % e_lin)
chk(e_lin < 1e-12,
    "named field with rhs=2*default and src=0.5*g2 reproduces the default step (max|d| = %.2e)"
    % e_lin)

print("%s test_time_multielliptic" % ("FAIL (%d)" % fails if fails else "PASS"))
sys.exit(1 if fails else 0)
