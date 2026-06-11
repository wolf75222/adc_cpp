#!/usr/bin/env python3
"""Vitesses d'onde EXACTES par valeurs propres du jacobien de flux :
m.wave_speeds_from_jacobian(eig='numeric'|'fd', blocks=...).

Le modele fournit (ou laisse l'autodiff deriver via flux_jacobian) A = dF/dU ; le codegen emet
le remplissage des sous-blocs + adc::real_eig_minmax (QR de Francis sur pile, dense_eig.hpp,
repli Gershgorin = borne externe sure) ; smin/smax = extremes des spectres des blocs. Sans
set_eigenvalues, max_wave_speed = max(|smin|, |smax|) sur les memes blocs (Rusanov et HLL
partagent la meme verite). Aucun changement du coeur au-dela de dense_eig.hpp (ADC-86).

On verifie :
 (1) [interprete] autodiff + bloc plein sur un jouet LINEAIRE decouple (J = diag(a, b)) :
     eval_wave_speeds == (min(a,b), max(a,b)) exact ; max_wave_speed == max(|a|, |b|).
 (2) [interprete] jouet NON LINEAIRE (Burgers x2 : J = diag(q1, q2)) : vitesses PAR CELLULE.
 (3) [interprete] blocks en LISTES D'INDICES non contigus == bloc plein (systeme decouple).
 (4) [compilateur] le modele compile (aot), has_wave_speeds vrai, la source generee contient
     dense_eig.hpp + real_eig_minmax (numeric) ; en 'fd', le corps appelle flux(U, a, dir).
 (5) [compilateur] riemann='hll' via System : eval_rhs == reference numpy HLL avec les vitesses
     exactes PAR CELLULE (Davis) sur le jouet non lineaire, atol 1e-12.
 (6) [compilateur] eig='fd' == eig='numeric' sur le jouet non lineaire (rtol 1e-5, troncature).
 (7) gardes : exclusivite avec set_wave_speeds ; blocs invalides ; fd avec x= explicite refuse.
 (8) retro-compat : l'emission d'un modele historique (p + eigenvalues) est inchangee
     (aucune reference a dense_eig).

S'auto-saute (exit 0) sans compilateur pour (4)-(6).
"""
import os
import sys
import tempfile

import numpy as np

import adc
from adc import dsl

fails = 0
INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))
A, B = 1.5, -0.7


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


def linear_toy(blocks=None, eig="numeric"):
    """Advections decouplees : flux x = [a q1, b q2], y = [b q1, a q2] -> Jx = diag(a, b)."""
    m = dsl.Model("jaclin")
    q1, q2 = m.conservative_vars("q1", "q2")
    a = m.param("a", A)
    b = m.param("b", B)
    m.flux(x=[a * q1, b * q2], y=[b * q1, a * q2])
    m.wave_speeds_from_jacobian(eig=eig, blocks=blocks)
    m.primitive_vars(q1, q2)
    m.conservative_from([q1, q2])
    return m


def burgers_toy(eig="numeric"):
    """Non lineaire : flux x = [q1^2/2, q2^2/2] -> Jx = diag(q1, q2) ; y croise."""
    m = dsl.Model("jacburg" + eig)
    q1, q2 = m.conservative_vars("q1", "q2")
    m.flux(x=[0.5 * q1 * q1, 0.5 * q2 * q2], y=[0.5 * q2 * q2, 0.5 * q1 * q1])
    m.wave_speeds_from_jacobian(eig=eig)
    m.primitive_vars(q1, q2)
    m.conservative_from([q1, q2])
    return m


def toy_state(n):
    xx, yy = np.meshgrid((np.arange(n) + 0.5) / n, (np.arange(n) + 0.5) / n, indexing="xy")
    q1 = 1.0 + 0.3 * np.sin(2 * np.pi * xx) * np.cos(4 * np.pi * yy)
    q2 = -0.5 + 0.2 * np.cos(2 * np.pi * xx) + 0.1 * np.sin(2 * np.pi * yy)
    return np.stack([q1, q2], axis=0)


print("== (1) lineaire decouple, autodiff, bloc plein (interprete) ==")
m = linear_toy()
U0 = toy_state(6)
lo, hi = m.eval_wave_speeds(U0.reshape(2, -1), {}, 0)
chk(np.allclose(lo, min(A, B), atol=1e-12) and np.allclose(hi, max(A, B), atol=1e-12),
    "eval_wave_speeds x == (min(a,b), max(a,b)) exact (J = diag)")
mws = m._m.max_wave_speed(U0.reshape(2, -1), {}, 0)
chk(abs(mws - max(abs(A), abs(B))) < 1e-12, "max_wave_speed == max(|a|, |b|) (memes blocs)")
rep = m.check_model(samples=U0.reshape(2, -1))
chk(rep["ok"], "check_model passe (chemin jacobien evalue en numpy)")

print("== (2) non lineaire : vitesses PAR CELLULE (interprete) ==")
mb = burgers_toy()
Uf = U0.reshape(2, -1)
lo, hi = mb.eval_wave_speeds(Uf, {}, 0)
chk(np.allclose(lo, np.minimum(Uf[0], Uf[1]), atol=1e-12)
    and np.allclose(hi, np.maximum(Uf[0], Uf[1]), atol=1e-12),
    "eval_wave_speeds x == (min(q1,q2), max(q1,q2)) par cellule")

print("== (3) blocs en listes d'indices non contigus == bloc plein ==")
m_blocks = linear_toy(blocks=[[1], [0]])  # ordre permute, non contigu : systeme decouple
lo2, hi2 = m_blocks.eval_wave_speeds(Uf, {}, 0)
lo1, hi1 = linear_toy().eval_wave_speeds(Uf, {}, 0)
chk(np.allclose(lo2, lo1, atol=1e-14) and np.allclose(hi2, hi1, atol=1e-14),
    "blocks=[[1],[0]] == bloc plein sur un systeme decouple")

print("== (3b) blocs PAR DIRECTION (dict x/y) ==")
m_xy = linear_toy(blocks={"x": [[0], [1]], "y": [[1], [0]]})
lox, hix = m_xy.eval_wave_speeds(Uf, {}, 0)
loy, hiy = m_xy.eval_wave_speeds(Uf, {}, 1)
chk(np.allclose(lox, min(A, B)) and np.allclose(hix, max(A, B))
    and np.allclose(loy, min(A, B)) and np.allclose(hiy, max(A, B)),
    "blocks={'x': ..., 'y': ...} : partitions independantes par direction")

print("== (7) gardes ==")
m_g = dsl.Model("guard")
g1, g2 = m_g.conservative_vars("g1", "g2")
m_g.flux(x=[g1, g2], y=[g2, g1])
m_g.wave_speeds(x=(0.0 * g1, 1.0 + 0.0 * g1), y=(0.0 * g1, 1.0 + 0.0 * g1))
msg = err_msg(lambda: m_g.wave_speeds_from_jacobian())
chk("un seul fournisseur" in msg, f"exclusivite paire explicite / jacobien ({msg[:40]}...)")
m_g2 = dsl.Model("guard2")
h1, h2 = m_g2.conservative_vars("h1", "h2")
m_g2.flux(x=[h1, h2], y=[h2, h1])
msg = err_msg(lambda: m_g2.wave_speeds_from_jacobian(blocks=[[0, 0], [1]]))
chk("deux blocs" in msg or "present" in msg, f"indice duplique refuse ({msg[:40]}...)")
msg = err_msg(lambda: m_g2.wave_speeds_from_jacobian(eig="fd", x=[[h1]], y=[[h1]]))
chk("n'ont pas de sens" in msg, f"fd + jacobien explicite refuse ({msg[:40]}...)")

cxx = dsl._default_cxx(None)
if not cxx:
    print("pas de compilateur C++ : tests (4)-(6)/(8) sautes")
    sys.exit(1 if fails else 0)

tmp = tempfile.mkdtemp(prefix="adc_ws_jac_")

print("== (4) compile + marqueurs d'emission ==")
src = burgers_toy()._m.emit_cpp_brick()
chk("dense_eig.hpp" in src and "adc::real_eig_minmax" in src and "EigBounds" in src,
    "source generee (numeric) : include dense_eig + real_eig_minmax + EigBounds")
src_fd = burgers_toy(eig="fd")._m.emit_cpp_brick()
chk("flux(U, a, dir)" in src_fd and "flux(Up_, a, dir)" in src_fd,
    "source generee (fd) : jacobien par colonnes du flux compile")
c_burg = burgers_toy().compile(os.path.join(tmp, "jacburg.so"), INCLUDE, backend="aot")
chk(getattr(c_burg, "has_wave_speeds", False), "compiled.has_wave_speeds (jacobien, sans 'p')")

print("== (8) retro-compat : emission historique inchangee ==")
m_p = dsl.Model("histo")
rho, mx = m_p.conservative_vars("rho", "m_x", roles=["Density", "MomentumX"])
u = m_p.primitive("u", mx / rho)
pp = m_p.primitive("p", 1.0 * rho)
m_p.flux(x=[mx, mx * u + pp], y=[mx, mx * u])
m_p.eigenvalues(x=[u - 1.0, u + 1.0], y=[u - 1.0, u + 1.0])
m_p.primitive_vars(rho, u)
m_p.conservative_from([rho, rho * u])
src_h = m_p._m.emit_cpp_brick()
chk("dense_eig" not in src_h and "real_eig_minmax" not in src_h,
    "modele historique (p + eigenvalues) : aucune reference a dense_eig")

n = 24


def expected_rhs_hll(U, n):
    """Reference numpy : HLL de Davis avec les vitesses EXACTES par cellule du jouet Burgers
    (Jx = diag(q1, q2) -> smin/smax = min/max(q1, q2) ; en y flux croise -> J = adiag mais
    diag(d(q2^2/2)/dq2...) : flux y = [q2^2/2, q1^2/2] -> Jy a entrees (0,1) = q2 et (1,0) = q1
    hors diagonale -> valeurs propres +-sqrt(q1 q2)... NON : Jy = [[0, q2], [q1, 0]], spectre
    +-sqrt(q1 q2) si produit positif, complexe sinon -- real_eig_minmax rend les PARTIES REELLES.
    On compare donc le chemin x (diagonal exact) et on borne le y par la meme machinerie numpy
    (np.linalg.eigvals), pas par une formule a la main."""
    dx = 1.0 / n
    rhs = np.zeros_like(U)
    for d, axis in ((0, 2), (1, 1)):
        UL = U
        UR = np.roll(U, -1, axis=axis)
        if d == 0:
            FL = np.stack([0.5 * UL[0] ** 2, 0.5 * UL[1] ** 2])
            FR = np.stack([0.5 * UR[0] ** 2, 0.5 * UR[1] ** 2])
            loL, hiL = np.minimum(UL[0], UL[1]), np.maximum(UL[0], UL[1])
            loR, hiR = np.minimum(UR[0], UR[1]), np.maximum(UR[0], UR[1])
        else:
            FL = np.stack([0.5 * UL[1] ** 2, 0.5 * UL[0] ** 2])
            FR = np.stack([0.5 * UR[1] ** 2, 0.5 * UR[0] ** 2])

            def speeds(V):
                q1, q2 = V[0].ravel(), V[1].ravel()
                J = np.zeros((q1.size, 2, 2))
                J[:, 0, 1] = q2
                J[:, 1, 0] = q1
                lam = np.linalg.eigvals(J)
                return (lam.real.min(axis=1).reshape(V[0].shape),
                        lam.real.max(axis=1).reshape(V[0].shape))
            loL, hiL = speeds(UL)
            loR, hiR = speeds(UR)
        sL = np.minimum(loL, loR)
        sR = np.maximum(hiL, hiR)
        span = np.where(np.abs(sR - sL) > 1e-300, sR - sL, 1.0)
        Fhll = (sR * FL - sL * FR + sL * sR * (UR - UL)) / span
        F = np.where(sL >= 0, FL, np.where(sR <= 0, FR, Fhll))
        rhs -= (F - np.roll(F, 1, axis=axis)) / dx
    return rhs


print("== (5) riemann='hll' via System : vitesses exactes par cellule ==")
sim = adc.System(n=n, L=1.0, periodic=True)
sim.add_equation("burg", model=c_burg,
                 spatial=adc.FiniteVolume(limiter="none", riemann="hll"),
                 time=adc.Explicit())
U = toy_state(n)
sim.set_state("burg", U)
rhs = np.array(sim.eval_rhs("burg"))
ref = expected_rhs_hll(U, n)
d = float(np.max(np.abs(rhs - ref)))
chk(d < 1e-12, f"eval_rhs hll == reference numpy a vitesses exactes (dmax = {d:.2e})")

print("== (6) eig='fd' == eig='numeric' ==")
c_fd = burgers_toy(eig="fd").compile(os.path.join(tmp, "jacburg_fd.so"), INCLUDE, backend="aot")
sim_fd = adc.System(n=n, L=1.0, periodic=True)
sim_fd.add_equation("burg", model=c_fd,
                    spatial=adc.FiniteVolume(limiter="none", riemann="hll"),
                    time=adc.Explicit())
sim_fd.set_state("burg", U)
rhs_fd = np.array(sim_fd.eval_rhs("burg"))
rel = float(np.max(np.abs(rhs_fd - rhs)) / max(1.0, np.max(np.abs(rhs))))
chk(rel < 1e-5, f"fd == numeric (ecart relatif {rel:.2e}, troncature O(eps))")

print("FAILS =", fails)
sys.exit(1 if fails else 0)
