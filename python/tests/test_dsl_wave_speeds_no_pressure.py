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

A, B = 1.5, 0.7  # vitesses des ondes en x / y (constantes, asymetriques pour pieger un swap d'axe)


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
    m.wave_speeds(x=(-1.0 * a, 1.0 * a), y=(-1.0 * b, 1.0 * b))
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
        s = A if d == 0 else B  # |smin| = |smax| = s des deux cotes (constantes)
        UL = U
        UR = np.roll(U, -1, axis=axis)   # voisin de droite le long de l'axe
        FL, FR = phys_flux(UL, d), phys_flux(UR, d)
        if riemann == "hll":
            sL, sR = -s, s  # min/max de Davis sur les deux cotes (sL < 0 < sR partout)
            F = (sR * FL - sL * FR + sL * sR * (UR - UL)) / (sR - sL)
        else:  # rusanov : alpha = max(mws_L, mws_R) = s
            F = 0.5 * (FL + FR) - 0.5 * s * (UR - UL)
        rhs -= (F - np.roll(F, 1, axis=axis)) / dx
    return rhs


print("== (1) facade + evaluateurs numpy (sans compilateur) ==")
m = toy_model()
U0 = toy_state(8)
lo, hi = m.eval_wave_speeds(U0, {}, 0)
chk(np.allclose(lo, -A) and np.allclose(hi, A), "eval_wave_speeds x = paire declaree (-a, +a)")
lo, hi = m.eval_wave_speeds(U0, {}, 1)
chk(np.allclose(lo, -B) and np.allclose(hi, B), "eval_wave_speeds y = paire declaree (-b, +b)")
mws = m._m.max_wave_speed(U0.reshape(2, -1), {}, 0)
chk(abs(mws - A) < 1e-14, "max_wave_speed interprete = max(|smin|, |smax|) SANS eigenvalues")
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

print("FAILS =", fails)
sys.exit(1 if fails else 0)
