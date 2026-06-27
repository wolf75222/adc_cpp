#!/usr/bin/env python3
"""pops.time RHS flux toggle -- P.rhs(flux=False) is source-only (epic ADC-399 / ADC-430).

Sibling of ADC-425 (which fixed sources=[] on flux=True). The rhs codegen routed on ``sources`` but
IGNORED ``flux``: a source-only stage (flux=False) still emitted the ``-div F`` base. Masked because
Lie/Strang SOURCE stages were tested only on ZERO-flux models (-div F == 0); on a NON-zero-flux model a
flux=False stage WRONGLY re-added -div F. ADC-430 adds the source-only runtime primitive
``ctx.source_default_into`` (= S WITHOUT -div F, the exact mirror of ADC-425's
``ctx.neg_div_flux_default_into``) and routes a flux=False rhs to a zeroed base + the requested sources.

(A) Codegen / IR (pure Python, always runs):
    - flux=False, sources=["default"] lowers to ctx.source_default_into (NO ctx.rhs_into, NO
      ctx.neg_div_flux_default_into) -- this is THE BUG PROBE: before ADC-430 it emitted a flux base.
    - flux=False, sources=["s"] (named only) emits NO flux base at all (the zeroed scratch + the named
      axpy is the whole RHS).
    - flux=False, sources=[] emits NO base and NO source (the zero RHS).
    - flux=True paths are UNCHANGED (rhs_into / neg_div_flux_default_into), bit-identical to ADC-425.
    - flux=False + named fluxes is rejected (a source-only stage has no flux to divide).

(B) End-to-end probe (skips unless the full toolchain is present): a NON-zero-flux advection model
    (F_x = a*rho) WITH a default source S = c*rho. A one-step program U + dt*rhs(flux=False,
    sources=["default"]) must apply ONLY the source (out - rho0 == dt*c*rho0, the SAME as an offline
    source-only step), with the -div F NOT leaked -- before ADC-430 it also subtracted dt*div(a*rho).
    flux=True still includes -div F (a non-trivial transport, distinct from the source-only step). A Lie
    split H(flux=True,sources=[]) ; S(flux=False,sources=["default"]) on the non-zero-flux model matches
    the offline split (was double-flux). Self-skips if _pops lacks install_program, numpy/_pops is absent,
    or no compiler/Kokkos is visible -- never faking the engine.

Run with python3 (PYTHONPATH = built pops package).
"""
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
import sys


def _skip(msg):
    print("skip test_time_rhs_flux_false (%s)" % msg)
    sys.exit(0)


try:
    import numpy as np

    import pops
    from pops.physics.facade import Model
    from pops import time as adctime
except Exception as exc:  # noqa: BLE001  -- numpy or _pops unavailable in this interpreter
    _skip("pops/numpy unavailable: %s" % exc)

fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


def advect_model(name="adv_rhs", a=1.0, c=0.7):
    """Scalar 'rho' with a NON-zero x flux F_x = a*rho and a default source S = c*rho.

    A complete, compilable production block. F_x != 0 so -div F is NON-zero on a non-uniform field: this
    is what lets the probe distinguish a flux=False (source-only) stage from a flux=True one -- on the
    zero-flux model ADC-425 used, the two would be indistinguishable (the very masking ADC-430 fixes)."""
    m = Model(name)
    (rho,) = m.conservative_vars("rho")
    zero = 0.0 * rho
    m.flux(x=[a * rho], y=[zero])          # NON-zero flux F_x = a*rho
    m.eigenvalues(x=[a + zero], y=[zero])  # |lambda| = a (constant advection speed)
    m.primitive_vars(rho=rho)
    m.conservative_from([rho])
    m.source([c * rho])                    # default/composite source S = c*rho
    return m


def advect_model_with_named(name="adv_rhs_named", a=1.0, c=0.7, d=0.5):
    """Same non-zero-flux scalar with BOTH a default source S = c*rho and a NAMED source 'decay' = d*rho.
    Lets (A) check that a flux=False named-only rhs emits the named axpy and NO flux base."""
    m = Model(name)
    (rho,) = m.conservative_vars("rho")
    zero = 0.0 * rho
    m.flux(x=[a * rho], y=[zero])
    m.eigenvalues(x=[a + zero], y=[zero])
    m.primitive_vars(rho=rho)
    m.conservative_from([rho])
    m.source([c * rho])
    m.source_term("decay", [d * rho])
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
print("== (A) rhs flux=False routing: IR + codegen ==")
m = advect_model()

# THE BUG PROBE (codegen): flux=False, sources=["default"] must be SOURCE-ONLY -- no flux base.
src_noflux_default = one_step_program("p_nf_default", ["default"], flux=False).emit_cpp_program(model=m)
chk("ctx.source_default_into(0," in src_noflux_default,
    "flux=False, sources=['default'] lowers to ctx.source_default_into (source only)")
chk("ctx.rhs_into(" not in src_noflux_default
    and "ctx.neg_div_flux_default_into(" not in src_noflux_default,
    "flux=False, sources=['default'] emits NO flux base (the -div F leak is fixed)")


# MULTI-BLOCK bidx routing (ADC-426 x ADC-430): a flux=False source-only stage must route to ITS OWN
# block's runtime index, not a hardcoded 0. Block 'b' is declared second -> index 1, so its source-only
# RHS must lower to ctx.source_default_into(1, ...). (A naive ADC-425-era flux=False branch hardcoded 0,
# which would silently apply block b's source to block a under a multi-block Program.)
def two_block_noflux(name):
    P = adctime.Program(name)
    Ua = P.state("a")
    Ub = P.state("b")
    Ra = P.rhs(state=Ua, fields=None, flux=False, sources=["default"])
    Rb = P.rhs(state=Ub, fields=None, flux=False, sources=["default"])
    P.commit("a", P.linear_combine("%s_a" % name, Ua + P.dt * Ra))
    P.commit("b", P.linear_combine("%s_b" % name, Ub + P.dt * Rb))
    return P


src_2blk = two_block_noflux("p_2blk_nf").emit_cpp_program(model=m)
chk("ctx.source_default_into(0," in src_2blk and "ctx.source_default_into(1," in src_2blk,
    "flux=False source-only routes to its block's bidx (block b -> index 1, not a hardcoded 0)")

# flux=False, sources=[] -> the zero RHS: NO flux base, NO source.
src_noflux_empty = one_step_program("p_nf_empty", [], flux=False).emit_cpp_program(model=m)
chk("ctx.rhs_into(" not in src_noflux_empty
    and "ctx.neg_div_flux_default_into(" not in src_noflux_empty
    and "ctx.source_default_into(" not in src_noflux_empty,
    "flux=False, sources=[] is the zero RHS (no flux base, no source)")

# flux=False, sources=["decay"] (named only) -> NO flux base, the named source axpy is the whole RHS.
src_noflux_named = one_step_program("p_nf_named", ["decay"], flux=False).emit_cpp_program(
    model=advect_model_with_named())
chk("ctx.rhs_into(" not in src_noflux_named
    and "ctx.neg_div_flux_default_into(" not in src_noflux_named
    and "ctx.source_default_into(" not in src_noflux_named,
    "flux=False, sources=['decay'] emits NO flux base")
chk("ctx.axpy(" in src_noflux_named,
    "flux=False, sources=['decay'] axpys the named source onto the zeroed RHS")

# flux=True paths UNCHANGED (ADC-425 routing): rhs_into for default, flux-only for [].
src_flux_default = one_step_program("p_f_default", ["default"], flux=True).emit_cpp_program(model=m)
src_flux_empty = one_step_program("p_f_empty", [], flux=True).emit_cpp_program(model=m)
chk("ctx.rhs_into(0," in src_flux_default and "ctx.source_default_into(" not in src_flux_default,
    "flux=True, sources=['default'] still lowers to ctx.rhs_into (unchanged)")
chk("ctx.neg_div_flux_default_into(0," in src_flux_empty,
    "flux=True, sources=[] still lowers to ctx.neg_div_flux_default_into (unchanged)")

# flux=False distinct from flux=True in the IR (the flux attr is in the hash).
h_nf = one_step_program("p", ["default"], flux=False)._ir_hash()
h_f = one_step_program("p", ["default"], flux=True)._ir_hash()
chk(h_nf != h_f, "flux=False vs flux=True produce distinct IR hashes")

# flux=False + named fluxes is rejected (a source-only stage has no flux divergence).
def _noflux_named_fluxes():
    P = adctime.Program("p_bad")
    U = P.state("plasma")
    P.rhs(state=U, fields=None, flux=False, sources=["default"], fluxes=["fx"])
    P.commit("plasma", P.linear_combine("p_bad_step", U))
    return P


rejected = False
try:
    _noflux_named_fluxes().emit_cpp_program(model=advect_model())
except ValueError:
    rejected = True
chk(rejected, "flux=False with named fluxes is rejected (no flux base to divide)")


# ---- (B) end-to-end probe: skips unless the full toolchain is present ----
if not hasattr(pops.System(n=8, L=1.0, periodic=True), "install_program"):
    print("-- (B) skipped: _pops lacks the install_program binding (rebuild _pops) --")
    print("%s test_time_rhs_flux_false (A only)" % ("FAIL" if fails else "PASS"))
    sys.exit(1 if fails else 0)

print("== (B) end-to-end: source-only vs full RHS on a NON-zero-flux model ==")

DT = 0.02
A = 1.0
C = 0.7
N = 16


def make_sim(name):
    sim = pops.System(n=N, L=1.0, periodic=True)
    try:
        compiled_model = advect_model("adv_block_%s" % name, A, C).compile(backend="production")
    except RuntimeError as exc:  # no compiler / no Kokkos visible
        _skip("model compile could not build the .so: %s" % str(exc)[:160])
    sim.add_equation("plasma", compiled_model,
                     spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                     time=pops.Explicit(method="euler"))
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("plasma", np.stack([rho]))
    return sim, rho


def run_one_step(sources, flux):
    """Compile + install a one-step program, step once, return (out, rho0)."""
    tag = "%s_%s" % ("flux" if flux else "noflux", "_".join(sources) or "empty")
    try:
        compiled = pops.compile_problem(model=advect_model("adv_%s" % tag, A, C),
                                       time=one_step_program("p_%s" % tag, sources, flux=flux))
    except RuntimeError as exc:  # no compiler / no Kokkos / .so compile failed
        _skip("compile_problem could not build the .so: %s" % str(exc)[:160])
    sim, rho0 = make_sim(tag)
    sim.install_program(compiled.so_path)
    sim.step(DT)
    return np.array(sim.get_state("plasma"))[0], rho0


# THE PROBE: U + dt*rhs(flux=False, sources=["default"]) on a NON-zero-flux model. The flux must NOT
# leak: out - rho0 == dt*C*rho0 exactly (the offline source-only step), NOT dt*(C*rho - div(a*rho)).
out_src, rho0 = run_one_step(["default"], flux=False)
ref_src = DT * C * rho0
d_src = float(np.abs((out_src - rho0) - ref_src).max())
print("  flux=False,sources=['default'] max|(out-rho0) - dt*C*rho0| = %.3e" % d_src)
chk(d_src < 1e-12,
    "flux=False applies ONLY the source (out-rho0 == dt*C*rho0; -div F NOT leaked)")
chk(float(np.abs(out_src - rho0).max()) > 1e-6, "flux=False,sources=['default'] actually moved rho")

# flux=True,sources=["default"] INCLUDES -div F: it must DIFFER from the source-only step (the flux is
# non-trivial on this field) -- proves flux=False is not accidentally equal to flux=True.
out_full, rho0f = run_one_step(["default"], flux=True)
d_full_vs_src = float(np.abs((out_full - rho0f) - DT * C * rho0f).max())
print("  flux=True,sources=['default']  max|(out-rho0) - dt*C*rho0| = %.3e (should be >> 0: -div F)"
      % d_full_vs_src)
chk(d_full_vs_src > 1e-6,
    "flux=True includes -div F (differs from the source-only step -- flux is not dropped)")

# Lie split: H = rhs(flux=True, sources=[]) then S = rhs(flux=False, sources=["default"]) on the
# NON-zero-flux model. It must equal the offline split: U1 = U + dt*(-div F)(U); out = U1 + dt*C*U1.
# Offline reference for the H half = a flux-only one-step (ADC-425's flux-only primitive, validated).
def lie_split_program(name):
    P = adctime.Program(name)
    U = P.state("plasma")
    H = P.rhs(state=U, fields=P.solve_fields(U), flux=True, sources=[])   # flux only (-div F)
    U1 = P.linear_combine("%s_H" % name, U + P.dt * H)
    S = P.rhs(state=U1, fields=None, flux=False, sources=["default"])     # source only on U1
    P.commit("plasma", P.linear_combine("%s_S" % name, U1 + P.dt * S))
    return P


try:
    compiled_lie = pops.compile_problem(model=advect_model("adv_lie", A, C), time=lie_split_program("lie"))
except RuntimeError as exc:
    _skip("compile_problem (lie) could not build the .so: %s" % str(exc)[:160])
sim_lie, rho0l = make_sim("lie")
sim_lie.install_program(compiled_lie.so_path)
sim_lie.step(DT)
out_lie = np.array(sim_lie.get_state("plasma"))[0]
# Offline split: H half = the flux-only one-step (validated ADC-425 primitive); S half = + dt*C*U1.
u1_flux_only, rho0fo = run_one_step([], flux=True)  # U1 = U + dt*(-div F)(U), same IC
ref_lie = u1_flux_only + DT * C * u1_flux_only
d_lie = float(np.abs(out_lie - ref_lie).max())
print("  Lie split        max|out - offline(H flux; S source)| = %.3e" % d_lie)
chk(d_lie < 1e-12,
    "Lie split H(flux,sources=[]);S(flux=False,source) == offline split (no double-flux)")

print("%s test_time_rhs_flux_false" % ("FAIL (%d)" % fails if fails else "PASS"))
sys.exit(1 if fails else 0)
