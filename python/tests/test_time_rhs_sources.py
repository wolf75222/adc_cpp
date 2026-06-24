#!/usr/bin/env python3
"""adc.time RHS source routing -- P.rhs flux/source selection (epic ADC-399 / ADC-425).

Spec criterion 17: a source is included in an RHS only when EXPLICITLY listed in ``sources``, never
summed implicitly. Before ADC-425 the codegen always lowered the default-flux RHS to ``ctx.rhs_into``
(= -div F + the model's default/composite source), so ``P.rhs(flux=True, sources=[])`` on a model with
a default ``m.source`` STILL added that source -- breaking the hyperbolic stage of a Lie/Strang/IMEX
split (which needs "flux but no source"). ADC-425 adds the flux-only runtime primitive
``ctx.neg_div_flux_default_into`` (= -div F WITHOUT the default source) and routes the codegen on
whether ``"default"`` is among the requested sources.

(A) Codegen / IR (pure Python, always runs):
    - sources=[] / ["default"] / ["named"] lower to DISTINCT IR (distinct _ir_hash) and DISTINCT C++
      (sources=[] -> neg_div_flux_default_into; sources=["default"] -> rhs_into).
    - sources=["default"] lowers to ctx.rhs_into (the historical path, unchanged).

(B) End-to-end probe (skips unless the full toolchain is present): a ZERO-FLUX scalar model with a
    default source S = c*rho. A one-step program U + dt*rhs(flux=True, sources=[]) must NOT move rho
    (flux is zero and the default source is excluded -> 0 cells change); the same with
    sources=["default"] applies the source (out - rho0 == dt*c*rho). A Lie split
    H = rhs(flux=True, sources=[]) ; S = rhs(flux=False, sources=["default"]) reproduces the offline
    single-source split. An existing default-source forward_euler (sources=["default"]) is unchanged.
    Self-skips if _adc lacks install_program, numpy/_adc is absent, or no compiler/Kokkos is visible --
    never faking the engine.

Run with python3 (PYTHONPATH = built adc package).
"""
import sys


def _skip(msg):
    print("skip test_time_rhs_sources (%s)" % msg)
    sys.exit(0)


try:
    import numpy as np

    import adc
    from adc import dsl
    from adc import time as adctime
except Exception as exc:  # noqa: BLE001  -- numpy or _adc unavailable in this interpreter
    _skip("adc/numpy unavailable: %s" % exc)

fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


def decay_model(name="rhs_decay", c=0.7):
    """Scalar 'rho' with NO flux and a default source S = c*rho (a constant decay/growth rate).

    A complete, compilable production block (zero flux + primitives + zero eigenvalues). flux == 0 so
    -div F == 0 exactly: the only dynamics is the default source, which lets the probe separate the
    flux from the source bit-exactly."""
    m = dsl.Model(name)
    (rho,) = m.conservative_vars("rho")
    zero = 0.0 * rho  # null expr (flux / eig do not wrap a bare float)
    m.flux(x=[zero], y=[zero])
    m.eigenvalues(x=[zero], y=[zero])
    m.primitive_vars(rho=rho)
    m.conservative_from([rho])
    m.source([c * rho])  # default/composite source S = c*rho
    return m


def decay_model_with_named(name="rhs_decay_named", c=0.7, d=0.5):
    """Same zero-flux scalar but with BOTH a default source S = c*rho and a NAMED source 'decay' =
    d*rho. Lets (A) check that named sources axpy onto either base (rhs_into or flux-only) without a
    double-count rejection (the ADC-425 routing makes them sound)."""
    m = dsl.Model(name)
    (rho,) = m.conservative_vars("rho")
    zero = 0.0 * rho
    m.flux(x=[zero], y=[zero])
    m.eigenvalues(x=[zero], y=[zero])
    m.primitive_vars(rho=rho)
    m.conservative_from([rho])
    m.source([c * rho])              # default/composite source
    m.source_term("decay", [d * rho])  # a distinct NAMED source
    return m


def one_step_program(name, sources, flux=True):
    """U^{n+1} = U + dt * rhs(flux=flux, sources=sources), committed on block 'plasma'."""
    P = adctime.Program(name)
    U = P.state("plasma")
    fields = P.solve_fields(U) if flux else None
    R = P.rhs(state=U, fields=fields, flux=flux, sources=list(sources))
    P.commit("plasma", P.linear_combine("%s_step" % name, U + P.dt * R))
    return P


# ---- (A) codegen / IR: pure Python, always runs ----
print("== (A) rhs source routing: IR + codegen ==")
m = decay_model()

src_default = one_step_program("p_default", ["default"]).emit_cpp_program(model=m)
src_empty = one_step_program("p_empty", []).emit_cpp_program(model=m)

# sources=["default"] keeps the historical ctx.rhs_into (flux + default source), unchanged.
chk("ctx.rhs_into(0," in src_default and "ctx.neg_div_flux_default_into(" not in src_default,
    "sources=['default'] lowers to ctx.rhs_into (flux + default source, historical path)")
# sources=[] (flux only) lowers to ctx.neg_div_flux_default_into -- NO default source.
chk("ctx.neg_div_flux_default_into(0," in src_empty and "ctx.rhs_into(" not in src_empty,
    "sources=[] lowers to ctx.neg_div_flux_default_into (flux only, NO default source)")

# Distinct IR for [] vs ["default"] vs a named source (the sources attr is in the IR hash). Same
# program name so only the sources attr differs.
h_empty = one_step_program("p", [])._ir_hash()
h_default = one_step_program("p", ["default"])._ir_hash()
h_named = one_step_program("p", ["decay"])._ir_hash()
chk(len({h_empty, h_default, h_named}) == 3,
    "sources=[] / ['default'] / ['decay'] produce 3 distinct IR hashes")

# sources=['default'] IR hash is deterministic across constructions (codegen-only change, IR untouched).
chk(one_step_program("p", ["default"])._ir_hash() == h_default,
    "sources=['default'] IR hash is stable (codegen-only change, IR untouched)")

# A named-source rhs on a model WITH a non-empty default source now lowers (no double-count): the base
# is rhs_into (default folded) iff 'default' is listed; the named source is axpy'd on top either way.
src_named_on_default = one_step_program("p_nd", ["default", "decay"]).emit_cpp_program(
    model=decay_model_with_named())
chk("ctx.rhs_into(0," in src_named_on_default,
    "sources=['default','decay'] uses rhs_into base (default folded) + named axpy (no double-count)")
src_named_only = one_step_program("p_no", ["decay"]).emit_cpp_program(model=decay_model_with_named())
chk("ctx.neg_div_flux_default_into(0," in src_named_only,
    "sources=['decay'] (no 'default') uses flux-only base + named axpy")


# ---- (B) end-to-end probe: skips unless the full toolchain is present ----
if not hasattr(adc.System(n=8, L=1.0, periodic=True), "install_program"):
    print("-- (B) skipped: _adc lacks the install_program binding (rebuild _adc) --")
    print("%s test_time_rhs_sources (A only)" % ("FAIL" if fails else "PASS"))
    sys.exit(1 if fails else 0)

print("== (B) end-to-end: flux-only vs default-source RHS ==")

DT = 0.05
C = 0.7
N = 16


def make_sim(prog_model_name):
    sim = adc.System(n=N, L=1.0, periodic=True)
    try:
        compiled_model = decay_model("decay_block", C).compile(backend="production")
    except RuntimeError as exc:  # no compiler / no Kokkos visible
        _skip("model compile could not build the .so: %s" % str(exc)[:160])
    sim.add_equation("plasma", compiled_model,
                     spatial=adc.FiniteVolume(limiter="none", riemann="rusanov"),
                     time=adc.Explicit(method="euler"))
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("plasma", np.stack([rho]))
    return sim, rho


def run_one_step(sources, flux=True):
    """Compile + install a one-step program, step once, return (out, rho0)."""
    pname = "p_%s_%s" % ("flux" if flux else "noflux", "_".join(sources) or "empty")
    try:
        compiled = adc.compile_problem(model=decay_model("decay_%s" % pname, C),
                                       time=one_step_program(pname, sources, flux=flux))
    except RuntimeError as exc:  # no compiler / no Kokkos / .so compile failed
        _skip("compile_problem could not build the .so: %s" % str(exc)[:160])
    sim, rho0 = make_sim(pname)
    sim.install_program(compiled.so_path)
    sim.step(DT)
    return np.array(sim.get_state("plasma"))[0], rho0


# The PROBE: U + dt*rhs(flux=True, sources=[]) on a zero-flux model WITH m.source([C*rho]).
# flux is zero and the default source is EXCLUDED -> rho must not move at all (0 cells change).
out_empty, rho0 = run_one_step([])
d_empty = float(np.abs(out_empty - rho0).max())
n_changed = int(np.count_nonzero(np.abs(out_empty - rho0) > 0.0))
print("  sources=[]        max|out-rho0| = %.3e  changed cells = %d" % (d_empty, n_changed))
chk(d_empty == 0.0 and n_changed == 0,
    "sources=[] is flux-only: rho unchanged (0 cells), default source NOT leaked")

# sources=["default"] applies the default source: out - rho0 == dt*C*rho0 (flux is zero).
out_default, rho0d = run_one_step(["default"])
ref = DT * C * rho0d
d_default = float(np.abs((out_default - rho0d) - ref).max())
print("  sources=['default'] max|(out-rho0) - dt*C*rho0| = %.3e" % d_default)
chk(d_default < 1e-12, "sources=['default'] applies the default source (out-rho0 == dt*C*rho)")
chk(float(np.abs(out_default - rho0d).max()) > 1e-6, "sources=['default'] actually moved rho")

# Lie split: H = flux(flux=True, sources=[]) then S = source(flux=False, sources=["default"]).
# On this model H is the identity (zero flux, no source) and S adds dt*C*rho -> equals the offline
# single-source split U + dt*C*U applied once.
def lie_split_program(name):
    P = adctime.Program(name)
    U = P.state("plasma")
    H = P.rhs(state=U, fields=P.solve_fields(U), flux=True, sources=[])  # flux only (== identity here)
    U1 = P.linear_combine("%s_H" % name, U + P.dt * H)
    S = P.rhs(state=U1, fields=None, flux=False, sources=["default"])    # default source on U1
    P.commit("plasma", P.linear_combine("%s_S" % name, U1 + P.dt * S))
    return P


try:
    compiled_lie = adc.compile_problem(model=decay_model("decay_lie", C), time=lie_split_program("lie"))
except RuntimeError as exc:
    _skip("compile_problem (lie) could not build the .so: %s" % str(exc)[:160])
sim_lie, rho0l = make_sim("lie")
sim_lie.install_program(compiled_lie.so_path)
sim_lie.step(DT)
out_lie = np.array(sim_lie.get_state("plasma"))[0]
# Offline single-source Lie split on the zero-flux model: H(dt) is identity, S(dt) adds dt*C*rho.
ref_lie = rho0l + DT * C * rho0l
d_lie = float(np.abs(out_lie - ref_lie).max())
print("  Lie split        max|out - offline| = %.3e" % d_lie)
chk(d_lie < 1e-12, "Lie split H(flux,sources=[]);S(source) == offline single-source split")

# NO-REGRESSION: a default-source forward_euler (sources=['default']) is the historical path. On the
# zero-flux model it is U + dt*C*rho -- identical to the sources=['default'] one-step above.
try:
    P_fe = adctime.Program("fe_default")
    adctime.std.forward_euler(P_fe, "plasma", sources=("default",))
    compiled_fe = adc.compile_problem(model=decay_model("decay_fe", C), time=P_fe)
except RuntimeError as exc:
    _skip("compile_problem (forward_euler) could not build the .so: %s" % str(exc)[:160])
sim_fe, rho0f = make_sim("fe")
sim_fe.install_program(compiled_fe.so_path)
sim_fe.step(DT)
out_fe = np.array(sim_fe.get_state("plasma"))[0]
d_fe = float(np.abs((out_fe - rho0f) - DT * C * rho0f).max())
print("  forward_euler    max|(out-rho0) - dt*C*rho0| = %.3e" % d_fe)
chk(d_fe < 1e-12, "forward_euler(sources=['default']) unchanged (out-rho0 == dt*C*rho)")

print("%s test_time_rhs_sources" % ("FAIL (%d)" % fails if fails else "PASS"))
sys.exit(1 if fails else 0)
