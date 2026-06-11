#!/usr/bin/env python3
"""Vitesses d'onde signees EXPLICITES sans primitive 'p' : m.wave_speeds(x=, y=) -> riemann='hll'.

Avant : la brique DSL n'emettait wave_speeds que si une primitive 'p' (pression) etait declaree
(gate unique pression + vitesses signees) -- un modele sans pression (systeme de moments,
isotherme sans 'p'...) etait limite a Rusanov. Desormais m.wave_speeds(x=(smin, smax), y=(...))
emet wave_speeds(U, aux, dir, smin, smax) INDEPENDAMMENT de 'p', et, sans set_eigenvalues,
max_wave_speed (Rusanov / CFL) derive de max(|smin|, |smax|). Aucun changement C++ : le gate HLL
du coeur est deja requires { m.wave_speeds(...) } (block_builder.hpp).

Modele JOUET (PAS HyQMOM) : acoustique lineaire 2-var q1, q2 ; flux x = [a q2, a q1],
y = [b q2, b q1] ; spectre exact {-a, +a} en x et {-b, +b} en y -- vitesses signees CONSTANTES,
flux HLL analytique trivial a re-calculer en numpy.

On verifie :
 (1) facade + evaluateurs numpy : eval_wave_speeds = paire declaree ; max_wave_speed (interprete)
     = max(|smin|, |smax|) SANS eigenvalues ; check_model passe (coherence ws <-> mws).
 (2) [compilateur] le modele SANS 'p' compile (aot) et compiled.has_wave_speeds est vrai.
 (3) [compilateur] riemann='hll' construit et tourne : eval_rhs == divergence HLL re-calculee en
     numpy (Davis, sL/sR constants), atol 1e-13.
 (4) [compilateur] riemann='rusanov' sur le MEME modele (max_wave_speed emis depuis la paire) :
     eval_rhs == divergence Rusanov numpy, atol 1e-13.
 (5) gardes : ni eigenvalues ni wave_speeds -> emission refusee (message actionnable) ;
     eigenvalues sans 'p' ni wave_speeds -> hll toujours rejete par le gate C++ (historique).
 (6) retro-compat : un modele AVEC 'p' (sans paire explicite) emet toujours wave_speeds
     (has_wave_speeds vrai) -- chemin historique inchange.

S'auto-saute (exit 0) sans compilateur pour (2)-(4) ; (1)/(5)/(6 partiel) tournent toujours.
"""
import os
import sys
import tempfile

import numpy as np

import adc
from adc import dsl

fails = 0
INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))

A, B = 1.5, 0.7    # vitesses des ondes en x / y (constantes, asymetriques pour pieger un swap d'axe)
# Paire SIGNEE ASYMETRIQUE (smin != -smax) : avec une paire symetrique, le flux HLL se reduit
# ALGEBRIQUEMENT a Rusanov ((sR FL - sL FR + sL sR dU)/(sR-sL) == (FL+FR)/2 - (s/2) dU quand
# sL = -sR = -s) et le test ne discriminerait pas le chemin HLL (revue adverse). Les facteurs
# ci-dessous cassent la symetrie tout en MAJORANT le spectre vrai {-a, +a} (smin <= -a, smax >= +a
# n'est PAS requis par HLL : la paire declaree est le contrat, la reference numpy la recalcule).
WSX = (-0.5, 1.0)  # (smin_x, smax_x) = (-0.5*a, 1.0*a)
WSY = (-1.0, 0.25)  # (smin_y, smax_y) = (-1.0*b, 0.25*b)


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def err_msg(fn):
    try:
        fn(); return ""
    except Exception as ex:  # noqa: BLE001
        return str(ex)


def toy_model(name="acoustic2"):
    """Acoustique lineaire 2-var SANS pression : vitesses signees explicites, PAS d'eigenvalues
    (exerce aussi le fallback max_wave_speed = max(|smin|, |smax|))."""
    m = dsl.Model(name)
    q1, q2 = m.conservative_vars("q1", "q2")
    a = m.param("a", A)
    b = m.param("b", B)
    m.flux(x=[a * q2, a * q1], y=[b * q2, b * q1])
    m.wave_speeds(x=(WSX[0] * a, WSX[1] * a), y=(WSY[0] * b, WSY[1] * b))
    m.primitive_vars(q1, q2)
    m.conservative_from([q1, q2])
    return m


def toy_state(n):
    """Etat lisse non uniforme, different selon x et y (asymetrie volontaire)."""
    xx, yy = np.meshgrid((np.arange(n) + 0.5) / n, (np.arange(n) + 0.5) / n, indexing="xy")
    q1 = 1.0 + 0.3 * np.sin(2 * np.pi * xx) * np.cos(4 * np.pi * yy)
    q2 = 0.2 * np.cos(2 * np.pi * xx) + 0.1 * np.sin(2 * np.pi * yy)
    return np.stack([q1, q2], axis=0)


def phys_flux(U, d):
    a = A if d == 0 else B
    return np.stack([a * U[1], a * U[0]], axis=0)


def expected_rhs(U, n, riemann):
    """- div F avec flux d'interface HLL (Davis) ou Rusanov re-calcules en numpy. Axe x = dernier
    axe du tableau (n_vars, ny, nx), axe y = avant-dernier ; faces periodiques par np.roll."""
    dx = 1.0 / n
    rhs = np.zeros_like(U)
    for d, axis in ((0, 2), (1, 1)):
        base = A if d == 0 else B
        klo, khi = WSX if d == 0 else WSY
        sL, sR = klo * base, khi * base       # paire signee ASYMETRIQUE, constante des deux cotes
        alpha = max(abs(sL), abs(sR))         # rusanov : max(|smin|, |smax|) (= max_wave_speed emis)
        UL = U
        UR = np.roll(U, -1, axis=axis)        # voisin de droite le long de l'axe
        FL, FR = phys_flux(UL, d), phys_flux(UR, d)
        if riemann == "hll":
            F = (sR * FL - sL * FR + sL * sR * (UR - UL)) / (sR - sL)
        else:
            F = 0.5 * (FL + FR) - 0.5 * alpha * (UR - UL)
        rhs -= (F - np.roll(F, 1, axis=axis)) / dx
    return rhs


print("== (1) facade + evaluateurs numpy (sans compilateur) ==")
m = toy_model()
U0 = toy_state(8)
lo, hi = m.eval_wave_speeds(U0, {}, 0)
chk(np.allclose(lo, WSX[0] * A) and np.allclose(hi, WSX[1] * A),
    "eval_wave_speeds x = paire declaree (asymetrique)")
lo, hi = m.eval_wave_speeds(U0, {}, 1)
chk(np.allclose(lo, WSY[0] * B) and np.allclose(hi, WSY[1] * B),
    "eval_wave_speeds y = paire declaree (asymetrique)")
mws = m._m.max_wave_speed(U0.reshape(2, -1), {}, 0)
chk(abs(mws - max(abs(WSX[0]), abs(WSX[1])) * A) < 1e-14,
    "max_wave_speed interprete = max(|smin|, |smax|) SANS eigenvalues")
rep = m.check_model(samples=U0.reshape(2, -1))
chk(rep["ok"], "check_model passe (finitude + coherence ws <-> max_wave_speed)")

print("== (5) gardes (sans compilateur) ==")
m_none = dsl.Model("nospeed")
r1, r2 = m_none.conservative_vars("r1", "r2")
m_none.flux(x=[r2, r1], y=[r2, r1])
m_none.primitive_vars(r1, r2)
m_none.conservative_from([r1, r2])
msg = err_msg(lambda: m_none._m.emit_cpp_brick())
chk("set_wave_speeds" in msg and "set_eigenvalues" in msg,
    f"ni eigenvalues ni wave_speeds -> emission refusee, remede nomme ({msg[:60]}...)")

cxx = dsl._default_cxx(None)
if not cxx:
    print("pas de compilateur C++ : tests (2)-(4)/(6) sautes")
    sys.exit(1 if fails else 0)

tmp = tempfile.mkdtemp(prefix="adc_ws_nop_")

print("== (2) compile aot sans 'p' : wave_speeds emis ==")
compiled = toy_model().compile(os.path.join(tmp, "acoustic2.so"), INCLUDE, backend="aot")
chk(getattr(compiled, "has_wave_speeds", False), "compiled.has_wave_speeds (paire explicite, sans 'p')")

n = 32
dis = float(np.max(np.abs(expected_rhs(toy_state(n), n, "hll") - expected_rhs(toy_state(n), n, "rusanov"))))
chk(dis > 1e-3, f"les references HLL et Rusanov DIFFERENT (dmax = {dis:.3e}) : le test discrimine")
for label, riemann in (("(3) riemann='hll'", "hll"), ("(4) riemann='rusanov'", "rusanov")):
    print(f"== {label} : eval_rhs == reference numpy ==")
    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.add_equation("toy", model=compiled,
                     spatial=adc.FiniteVolume(limiter="none", riemann=riemann),
                     time=adc.Explicit())
    U = toy_state(n)
    sim.set_state("toy", U)
    rhs = np.array(sim.eval_rhs("toy"))
    ref = expected_rhs(U, n, riemann)
    d = float(np.max(np.abs(rhs - ref)))
    chk(d < 1e-13, f"eval_rhs {riemann} == divergence {riemann} numpy (dmax = {d:.2e})")

print("== (5b) eigenvalues sans 'p' ni paire : hll toujours rejete (historique) ==")
m_eig = dsl.Model("eigonly")
e1, e2 = m_eig.conservative_vars("e1", "e2")
ae = m_eig.param("a", A)
m_eig.flux(x=[ae * e2, ae * e1], y=[ae * e2, ae * e1])
m_eig.eigenvalues(x=[-1.0 * ae, 1.0 * ae], y=[-1.0 * ae, 1.0 * ae])
m_eig.primitive_vars(e1, e2)
m_eig.conservative_from([e1, e2])
c_eig = m_eig.compile(os.path.join(tmp, "eigonly.so"), INCLUDE, backend="aot")
chk(not getattr(c_eig, "has_wave_speeds", True), "has_wave_speeds faux (eigenvalues sans 'p')")
sim = adc.System(n=16, L=1.0, periodic=True)
msg = err_msg(lambda: sim.add_equation(
    "eig", model=c_eig, spatial=adc.FiniteVolume(limiter="none", riemann="hll"),
    time=adc.Explicit()))
chk("vitesses d'onde" in msg or "wave_speeds" in msg,
    f"hll rejete par le gate C++ avec remede ({msg[:60]}...)")

print("== (6) retro-compat : modele AVEC 'p' emet toujours wave_speeds ==")
m_p = dsl.Model("withp")
rho, mx, my = m_p.conservative_vars("rho", "m_x", "m_y",
                                    roles=["Density", "MomentumX", "MomentumY"])
u = m_p.primitive("u", mx / rho)
v = m_p.primitive("v", my / rho)
p = m_p.primitive("p", 1.0 * rho)  # isotherme theta = 1
cs = m_p.param("cs", 1.0)
m_p.flux(x=[mx, mx * u + p, mx * v], y=[my, my * u, my * v + p])
m_p.eigenvalues(x=[u - cs, u, u + cs], y=[v - cs, v, v + cs])
m_p.primitive_vars(rho, u, v)
m_p.conservative_from([rho, rho * u, rho * v])
c_p = m_p.compile(os.path.join(tmp, "withp.so"), INCLUDE, backend="aot")
chk(getattr(c_p, "has_wave_speeds", False), "has_wave_speeds vrai (chemin historique 'p' + eigenvalues)")
sim = adc.System(n=16, L=1.0, periodic=True)
msg = err_msg(lambda: sim.add_equation(
    "gasp", model=c_p, spatial=adc.FiniteVolume(limiter="none", riemann="hll"),
    time=adc.Explicit()))
chk(msg == "", f"hll accepte sur le modele avec 'p' (historique, message='{msg[:40]}')")

print("== (7) briques hybrides : flag has_wave_speeds propage (sans compilateur machine) ==")
# HyperbolicBrick.compile passe par emit_cpp_brick : le struct de brique emet wave_speeds selon
# les MEMES regles ('p' OU paire explicite). CompiledBrick.has_wave_speeds porte l'info jusqu'au
# CompiledModel hybride (HybridModel._compiled_model) -- sans quoi la garde precoce hll
# bloquerait a tort un hybride compressible (revue adverse).
bp = dsl.HyperbolicBrick("hybp")
hrho, hmx, hmy = bp.conservative_vars("rho", "m_x", "m_y",
                                      roles=["Density", "MomentumX", "MomentumY"])
hu = bp.primitive("u", hmx / hrho)
hv = bp.primitive("v", hmy / hrho)
hp = bp.primitive("p", 1.0 * hrho)
bp.flux(x=[hmx, hmx * hu + hp, hmx * hv], y=[hmy, hmy * hu, hmy * hv + hp])
bp.eigenvalues(x=[hu - 1.0, hu, hu + 1.0], y=[hv - 1.0, hv, hv + 1.0])
bp.primitive_vars(hrho, hu, hv)
bp.conservative_from([hrho, hrho * hu, hrho * hv])
cbp = bp.compile()
chk(cbp.has_wave_speeds and "wave_speeds" in cbp.struct_src,
    "brique AVEC 'p' : has_wave_speeds vrai, struct emet wave_speeds")

bn = dsl.HyperbolicBrick("hybn")
hc, = bn.conservative_vars("c")
bn.flux(x=[1.0 * hc], y=[0.0 * hc])
bn.eigenvalues(x=[1.0 + 0.0 * hc], y=[0.0 * hc])
bn.primitive_vars(hc)
bn.conservative_from([hc])
cbn = bn.compile()
chk((not cbn.has_wave_speeds) and "wave_speeds" not in cbn.struct_src,
    "brique SANS 'p' ni paire : has_wave_speeds faux, struct sans wave_speeds")

bw = dsl.HyperbolicBrick("hybw")
hq, = bw.conservative_vars("q")
bw.flux(x=[2.0 * hq], y=[0.5 * hq])
bw.wave_speeds(x=(0.0 * hq, 2.0 + 0.0 * hq), y=(0.0 * hq, 0.5 + 0.0 * hq))
bw.primitive_vars(hq)
bw.conservative_from([hq])
cbw = bw.compile()
chk(cbw.has_wave_speeds and "wave_speeds" in cbw.struct_src,
    "brique a PAIRE explicite sans 'p' : has_wave_speeds vrai (miroir HyperbolicBrick.wave_speeds)")

print("== (8) wave_speeds_value : formes mixtes (valeur propre constante + dependante de l'etat) ==")
mm = dsl.Model("mixshape")
w1, w2 = mm.conservative_vars("w1", "w2")
mm.flux(x=[w2, w1], y=[w2, w1])
mm.eigenvalues(x=[1.0 + 0.0 * w1, w1], y=[0.0 * w1, w2])  # melange scalaire-constant / tableau
mm.primitive_vars(w1, w2)
mm.conservative_from([w1, w2])
Um = toy_state(4).reshape(2, -1)
lo_m, hi_m = mm.eval_wave_speeds(Um, {}, 0)
chk(np.all(np.isfinite(lo_m)) and np.all(np.isfinite(hi_m)) and np.all(lo_m <= hi_m),
    "min/max sur valeurs propres a formes mixtes (broadcast, pas de crash np.stack)")

print("FAILS =", fails)
sys.exit(1 if fails else 0)
