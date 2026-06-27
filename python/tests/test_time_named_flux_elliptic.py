#!/usr/bin/env python3
"""Named fluxes (m.flux_term) and named elliptic fields (m.elliptic_field), epic ADC-399 (ADC-419).

Mirrors the named-source pattern (ADC-400 / ADC-403):

  - m.flux_term(name, x=, y=) declares an OPT-IN named physical flux (n_cons expressions per
    direction); name='default' is the backward-compatible alias of m.flux(...). A compiled time
    Program selects a SUM of named fluxes via ctx.rhs(..., fluxes=[name, ...]) and assembles -div of
    that sum; fluxes=['default'] (or no list) keeps the historical -div F (rhs_into), byte-identical.
  - m.elliptic_field(name, rhs=, operator=, aux=) declares an OPT-IN named elliptic field. The IR +
    validation + hash land here (ADC-419); the multi-elliptic SOLVE RUNTIME (a second elliptic operator
    + its own aux channel) is now wired (ADC-428), so solve_fields(field=name) lowers to the named ctx
    call. The end-to-end second-elliptic parity lives in python/tests/test_time_multielliptic.py.

Section A (pure Python, always runs): validation (dims / unique / unknown-name / collisions), the
model hash changes when a named flux / elliptic changes (and is byte-identical when none is declared),
and rhs(fluxes=['default']) lowers IDENTICALLY to the current default rhs.

Section B (gated, self-skip): a compiled program using NAMED fluxes that split the physical flux into
two pieces summing to the default -> stepping it equals stepping a single-named-flux ('whole') program
to ~1e-14 (the named fluxes sum to the same -div F).

Skips cleanly (exit 0) without numpy / _pops / a compiler / a visible Kokkos -- never fakes the engine.
"""
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
import sys


def _pops_mods():
    try:
        from pops.ir.ops import sqrt
        from pops.physics.facade import Model
        from pops import time as adctime
    except Exception as exc:  # pops not importable here -> skip, never fake
        print("skip test_time_named_flux_elliptic (pops unavailable: %s)" % exc)
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


# --- shared isothermal 2D fluid block (rho, mx, my; Poisson aux) ---
def _base_block(m):
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    cs2 = m.param("cs2", 0.5)
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    p = m.primitive("p", cs2 * rho)
    m.primitive_vars(rho=rho, u=u, v=v, p=p)
    m.conservative_from([rho, rho * u, rho * v])
    cs = sqrt(cs2)
    m.eigenvalues(x=[u - cs, u, u + cs], y=[v - cs, v, v + cs])
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    return rho, mx, my, u, v, p, gx, gy


def whole_flux_model(name="nf_whole"):
    """The physical flux as the model's DEFAULT (m.flux) AND as a single named flux 'whole'."""
    m = Model(name)
    rho, mx, my, u, v, p, gx, gy = _base_block(m)
    fx = [mx, mx * u + p, my * u]
    fy = [my, mx * v, my * v + p]
    m.flux(x=fx, y=fy)
    m.flux_term("whole", x=fx, y=fy)
    return m


def split_flux_model(name="nf_split"):
    """The SAME physical flux split into two named pieces 'conv' + 'press' that sum to it. The default
    m.flux is the same whole flux (so the model is otherwise identical to whole_flux_model)."""
    m = Model(name)
    rho, mx, my, u, v, p, gx, gy = _base_block(m)
    fx = [mx, mx * u + p, my * u]
    fy = [my, mx * v, my * v + p]
    m.flux(x=fx, y=fy)
    # conv carries the advective part, press the pressure part; conv + press == (fx, fy).
    m.flux_term("conv", x=[mx, mx * u, my * u], y=[my, mx * v, my * v])
    m.flux_term("press", x=[0.0 * rho, p, 0.0 * rho], y=[0.0 * rho, 0.0 * rho, p])
    return m


# =================== Section A: pure Python ===================
print("== (A) m.flux_term / m.elliptic_field validation + hash + codegen ==")


def _carrier():
    m = Model("nf")
    rho, mx, my, u, v, p, gx, gy = _base_block(m)
    m.flux(x=[mx, mx * u + p, my * u], y=[my, mx * v, my * v + p])
    return m, dict(rho=rho, mx=mx, my=my, u=u, v=v, p=p, gx=gx, gy=gy)


# --- flux_term validation ---
m, V = _carrier()
m.flux_term("conv", x=[V["mx"], V["mx"] * V["u"], V["my"] * V["u"]],
            y=[V["my"], V["mx"] * V["v"], V["my"] * V["v"]])
chk(m.check(), "flux_term valid (3 components per direction, check passes)")

m2, V2 = _carrier()
chk(raises(ValueError, lambda: m2.flux_term("bad", x=[V2["mx"], V2["mx"]], y=[V2["my"], V2["my"], V2["my"]])),
    "flux_term wrong x dimension rejected")
chk(raises(ValueError, lambda: m2.flux_term("bad", x=[V2["mx"], V2["mx"], V2["mx"]], y=[V2["my"], V2["my"]])),
    "flux_term wrong y dimension rejected")

m3, V3 = _carrier()
m3.flux_term("conv", x=[V3["mx"], V3["mx"], V3["mx"]], y=[V3["my"], V3["my"], V3["my"]])
chk(raises(ValueError, lambda: m3.flux_term("conv", x=[V3["mx"], V3["mx"], V3["mx"]],
                                            y=[V3["my"], V3["my"], V3["my"]])),
    "duplicate flux_term name rejected")
chk(raises(ValueError, lambda: m3.flux_term("1bad", x=[V3["mx"], V3["mx"], V3["mx"]],
                                            y=[V3["my"], V3["my"], V3["my"]])),
    "non-identifier flux_term name rejected")

# default alias: flux_term('default', ...) == m.flux(...), hash unchanged.
a = whole_flux_model("nf_alias_a")
b = Model("nf_alias_a")
rho, mx, my, u, v, p, gx, gy = _base_block(b)
b.flux_term("default", x=[mx, mx * u + p, my * u], y=[my, mx * v, my * v + p])
b.flux_term("whole", x=[mx, mx * u + p, my * u], y=[my, mx * v, my * v + p])
chk(a._m._model_hash() == b._m._model_hash(), "flux_term('default', ...) == m.flux(...) (same hash)")

# --- flux_term hash policy ---
base, Vb = _carrier()
base_hash = base._m._model_hash()
withnamed, Vn = _carrier()
withnamed.flux_term("conv", x=[Vn["mx"], Vn["mx"], Vn["mx"]], y=[Vn["my"], Vn["my"], Vn["my"]])
chk(base_hash != withnamed._m._model_hash(), "declaring a named flux changes the model hash")
changed, Vc = _carrier()
changed.flux_term("conv", x=[Vc["mx"], 2.0 * Vc["mx"], Vc["mx"]], y=[Vc["my"], Vc["my"], Vc["my"]])
chk(withnamed._m._model_hash() != changed._m._model_hash(),
    "changing a named-flux expression invalidates the cache")

# Golden: a model that never declares a named flux / elliptic field keeps a byte-identical hash. We
# assert it via a fresh carrier built two ways (no named extras) being equal -- and equal to the
# value before ANY flux_term touched the object.
plain1, _ = _carrier()
plain2, _ = _carrier()
chk(plain1._m._model_hash() == plain2._m._model_hash() == base_hash,
    "a model without named flux/elliptic keeps a stable (historical) hash")

# --- elliptic_field validation ---
e, Ve = _carrier()
e.elliptic_field("phi2", rhs=Ve["rho"], aux=["phi2", "g2x", "g2y"])
chk(e.check(), "elliptic_field valid (rhs + named aux, check passes)")

e2, V2e = _carrier()
chk(raises(ValueError, lambda: e2.elliptic_field("default", rhs=V2e["rho"])),
    "elliptic_field('default') rejected (default is m.elliptic_rhs)")
chk(raises(ValueError, lambda: e2.elliptic_field("phi2", rhs=V2e["rho"], operator="helmholtz")),
    "elliptic_field unsupported operator rejected")
chk(raises(ValueError, lambda: e2.elliptic_field("1bad", rhs=V2e["rho"])),
    "non-identifier elliptic_field name rejected")
chk(raises(ValueError, lambda: e2.elliptic_field("phi2", rhs=V2e["rho"], aux=[])),
    "elliptic_field with empty aux rejected")
e2.elliptic_field("phi2", rhs=V2e["rho"])
chk(raises(ValueError, lambda: e2.elliptic_field("phi2", rhs=V2e["rho"])),
    "duplicate elliptic_field name rejected")

# --- elliptic_field hash policy ---
eh1, Veh1 = _carrier()
eh1.elliptic_field("phi2", rhs=Veh1["rho"])
chk(eh1._m._model_hash() != base_hash, "declaring a named elliptic field changes the model hash")
eh2, Veh2 = _carrier()
eh2.elliptic_field("phi2", rhs=2.0 * Veh2["rho"])
chk(eh1._m._model_hash() != eh2._m._model_hash(),
    "changing a named-elliptic rhs invalidates the cache")


# --- codegen: rhs(fluxes=['default']) lowers IDENTICALLY to the current default rhs ---
def _fe_program(name, fluxes):
    P = adctime.Program(name)
    U = P.state("plasma")
    f = P.solve_fields(U)
    R = P.rhs(name="R", state=U, fields=f, flux=True, fluxes=fluxes)
    P.commit("plasma", P.linear_combine("U1", U + P.dt * R))
    return P


def _norm_ids(src):
    """Strip the program NAME / HASH literals (they differ by construction) so two bodies compare by
    their lowered ALGORITHM, not their identity strings."""
    import re
    src = re.sub(r'pops_program_name\(\) \{ return "[^"]*"', 'pops_program_name() { return "<n>"', src)
    src = re.sub(r'pops_program_hash\(\) \{ return "[^"]*"', 'pops_program_hash() { return "<h>"', src)
    return src


mdl = whole_flux_model("nf_codegen")
src_none = _norm_ids(_fe_program("nf_codegen", None).emit_cpp_program(model=mdl))
src_default = _norm_ids(_fe_program("nf_codegen", ["default"]).emit_cpp_program(model=mdl))
chk("ctx.rhs_into(0, " in src_none, "default rhs lowers via ctx.rhs_into")
chk("ctx.neg_div_flux_into(" not in src_none, "default rhs does NOT use the named-flux divergence path")
chk(src_none == src_default,
    "rhs(fluxes=['default']) lowers byte-identically to rhs(fluxes=None) (the historical default)")

# A NAMED flux lowers the new path (per-cell flux kernel + neg_div_flux_into), NOT rhs_into.
src_named = _fe_program("nf_named", ["whole"]).emit_cpp_program(model=mdl)
chk("ctx.neg_div_flux_into(" in src_named, "a named-flux rhs lowers via ctx.neg_div_flux_into")
chk("pops::for_each_cell(" in src_named, "the named flux is evaluated by a per-cell kernel")
chk("ctx.rhs_into(0, " not in src_named, "a named-flux rhs does NOT call rhs_into (distinct stencil)")

# Validation: unknown flux name -> clear error; mixing default + named -> clear error; named flux
# without a model -> NotImplementedError.
chk(raises(ValueError, lambda: _fe_program("nf_unknown", ["does_not_exist"]).emit_cpp_program(model=mdl)),
    "an unknown flux_term name in rhs raises ValueError")
chk(raises(ValueError, lambda: _fe_program("nf_mix", ["default", "whole"]).emit_cpp_program(model=mdl)),
    "mixing 'default' with a named flux raises ValueError")
chk(raises(NotImplementedError, lambda: _fe_program("nf_nomodel", ["whole"]).emit_cpp_program()),
    "a named-flux rhs without a model raises NotImplementedError")

# Two named fluxes summing to the whole flux lower to ONE kernel + one neg_div_flux_into (the SUM).
src_split = _fe_program("nf_split_prog", ["conv", "press"]).emit_cpp_program(model=split_flux_model())
chk(src_split.count("ctx.neg_div_flux_into(") == 1,
    "a multi-named-flux rhs assembles a single -div of the summed fluxes")

# --- elliptic_field: solve_fields(field=) now lowers to the named ctx call (ADC-428 wired the runtime;
#     the multi-elliptic SOLVE + aux channel are no longer deferred). The end-to-end parity lives in
#     python/tests/test_time_multielliptic.py; here we just assert the lowering + the error paths. ---
def _named_field_program(name="nf_ell"):
    P = adctime.Program(name)
    U = P.state("plasma")
    f = P.solve_fields("fields_phi2", U, field="phi2")
    R = P.rhs(name="R", state=U, fields=f, flux=True)
    P.commit("plasma", P.linear_combine("U1", U + P.dt * R))
    return P


ell_model, Vell = _carrier()
ell_model.elliptic_field("phi2", rhs=Vell["rho"], aux=["phi2", "g2x", "g2y"])
for a in ("phi2", "g2x", "g2y"):
    ell_model._m.aux_field(a)  # the named field's aux outputs need channel slots (ADC-428)
src_named_ell = _named_field_program().emit_cpp_program(model=ell_model)
chk('ctx.solve_fields_from_state("phi2", 0, ' in src_named_ell,
    "solve_fields(field=) lowers to the named ctx call (ADC-428 multi-elliptic runtime)")
# Unknown field name -> clear ValueError; missing model -> NotImplementedError (cannot validate).
chk(raises(ValueError,
           lambda: _named_field_program("nf_unknown_ell").emit_cpp_program(
               model=_carrier()[0])),
    "an unknown elliptic_field name in solve_fields raises ValueError")
chk(raises(NotImplementedError, lambda: _named_field_program("nf_nomodel_ell").emit_cpp_program()),
    "a named-elliptic solve_fields without a model raises NotImplementedError")
# An empty field name is rejected at construction (clear error).
chk(raises(ValueError, lambda: adctime.Program("x").solve_fields("f", adctime.Program("x").state("b"),
                                                                 field="")),
    "solve_fields with an empty field name raises ValueError")


# =================== Section B: gated end-to-end parity ===================
print("== (B) named-flux parity: split fluxes == whole flux (one FE step) ==")


def _skipB(msg):
    print("-- (B) skipped: %s --" % msg)
    print("%s test_time_named_flux_elliptic (A only)" % ("FAIL" if fails else "PASS"))
    sys.exit(1 if fails else 0)


try:
    import numpy as np

    import pops
except Exception as exc:  # noqa: BLE001
    _skipB("numpy/_pops unavailable: %s" % exc)

if not hasattr(pops.System(n=8, L=1.0, periodic=True), "install_program"):
    _skipB("_pops lacks the install_program binding (rebuild _pops)")

N = 16
DT = 0.01


def make_sim(model):
    sim = pops.System(n=N, L=1.0, periodic=True)
    try:
        compiled = model.compile(backend="production")
    except RuntimeError as exc:
        _skipB("model compile could not build the .so: %s" % str(exc)[:160])
    sim.add_equation("plasma", compiled,
                     spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                     time=pops.Explicit(method="euler"))
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    mx = 0.4 * rho
    my = -0.2 * rho
    U0 = np.stack([rho, mx, my])
    sim.set_state("plasma", U0)
    return sim, U0


def _flux_fe_program(name, fluxes):
    P = adctime.Program(name)
    U = P.state("plasma")
    R = P.rhs(name="R", state=U, flux=True, fluxes=fluxes)
    P.commit("plasma", P.linear_combine("U1", U + P.dt * R))
    return P


# whole: one named flux 'whole' = the physical flux. split: two named fluxes 'conv'+'press' summing
# to it. Both go through the SAME centered-FV neg_div_flux_into, so -div(conv)+-div(press) ==
# -div(whole) exactly (linearity) -> the stepped states must match to round-off.
try:
    compiled_whole = pops.compile_problem(model=whole_flux_model("nf_whole_prog"),
                                         time=_flux_fe_program("nf_whole_fe", ["whole"]))
    compiled_split = pops.compile_problem(model=split_flux_model("nf_split_prog2"),
                                         time=_flux_fe_program("nf_split_fe", ["conv", "press"]))
except RuntimeError as exc:
    _skipB("compile_problem could not build the .so: %s" % str(exc)[:160])

sim_w, U0 = make_sim(whole_flux_model("nf_whole_block"))
sim_w.install_program(compiled_whole.so_path)
sim_w.step(DT)
U_w = np.array(sim_w.get_state("plasma"))

sim_s, _ = make_sim(split_flux_model("nf_split_block"))
sim_s.install_program(compiled_split.so_path)
sim_s.step(DT)
U_s = np.array(sim_s.get_state("plasma"))

e_split = float(np.abs(U_w - U_s).max())
print("  split-vs-whole named-flux parity: max|d| = %.2e" % e_split)
chk(e_split < 1e-14, "split named fluxes (conv+press) step == whole named flux step (max|d| = %.2e)"
                     % e_split)
chk(float(np.abs(U_w - U0).max()) > 1e-6, "the named flux actually moved the state")

print("== elliptic_field SOLVE runtime is wired (ADC-428); end-to-end parity: test_time_multielliptic ==")
print("%s test_time_named_flux_elliptic" % ("FAIL (%d)" % fails if fails else "PASS"))
sys.exit(1 if fails else 0)
