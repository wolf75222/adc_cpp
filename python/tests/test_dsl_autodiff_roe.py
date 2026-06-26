#!/usr/bin/env python3
"""Roe DSL hors roles fluides (Linear ADC-68, GENERICITY point 11, voie FOURNIE) : autodiff
symbolique du DSL + dissipation de Roe ecrite par l'utilisateur.

  (a) dsl.diff : differentiation symbolique de l'arbre Expr (linearite, produit, quotient, chaine
      pour pow/sqrt). Verifiee NUMERIQUEMENT (chemin d'evaluation reference des Expr, sans
      compilateur) sur des cas analytiques ; noeud non derivable / exposant dependant de la
      variable -> NotImplementedError ;
  (b) m.flux_jacobian(dir) : A = dF/dU auto-derive des flux declares == matrice analytique connue
      de l'isotherme 3-var (evaluation numerique point a point), pour les deux directions ;
  (c) m.roe_dissipation(x=, y=) reproduisant A LA MAIN le Roe isotherme (left/right des deux etats)
      == m.enable_roe() sur le MEME modele : compile (production) puis compare les trajectoires
      (~1e-12). C'est le test de bout en bout (AUTO-SKIP sans compilateur / en-tetes) ;
  (d) rejets explicites : dimensions fausses, variable hors left()/right(), enable_roe +
      roe_dissipation simultanes (les deux ordres), marqueur imbrique.

Les sections (a), (b), (d) sont PUREMENT symboliques (aucune compilation) ; seule (c) compile.
Invariants par assert ; imprime "OK test_dsl_autodiff_roe" en cas de succes.
"""
import os
import shutil
import sys
import tempfile

import numpy as np

import pops
from pops import dsl

fails = 0
INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))
CS2 = 0.5  # vitesse du son au carre (isotherme / pseudo-pression p = cs2 rho)


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


# === (a) dsl.diff : differentiation symbolique, verifiee numeriquement =========================
print("== (a) dsl.diff : regles de derivation sur cas analytiques (eval numpy) ==")
rng = np.random.default_rng(0)
xs = rng.uniform(-2.0, 2.0, 64)
ys = rng.uniform(-2.0, 2.0, 64)
xp = rng.uniform(0.2, 3.0, 64)  # echantillons strictement positifs (sqrt, puissances)

x = dsl.Var("x", "cons")
y = dsl.Var("y", "cons")

# d(x^2 + 3xy)/dx == 2x + 3y ; d/dy == 3x
e = x * x + 3.0 * x * y
dex = dsl.diff(e, x)
dey = dsl.diff(e, y)
chk(np.allclose(dex.eval({"x": xs, "y": ys}), 2.0 * xs + 3.0 * ys), "d(x^2+3xy)/dx == 2x+3y")
chk(np.allclose(dey.eval({"x": xs, "y": ys}), 3.0 * xs), "d(x^2+3xy)/dy == 3x")

# puissance : d(x^3)/dx == 3 x^2 ; exposant symbolique INDEPENDANT : d(x^y)/dx == y x^(y-1)
chk(np.allclose(dsl.diff(x ** 3, x).eval({"x": xp}), 3.0 * xp ** 2), "d(x^3)/dx == 3 x^2")
yp = rng.uniform(1.0, 3.0, 64)
chk(np.allclose(dsl.diff(x ** y, x).eval({"x": xp, "y": yp}), yp * xp ** (yp - 1.0)),
    "d(x^y)/dx == y x^(y-1) (exposant independant)")

# racine : d(sqrt(x))/dx == 1/(2 sqrt(x))
chk(np.allclose(dsl.diff(dsl.sqrt(x), x).eval({"x": xp}), 0.5 / np.sqrt(xp)),
    "d(sqrt(x))/dx == 1/(2 sqrt(x))")

# quotient : d(x/y)/dx == 1/y ; d/dy == -x/y^2
q = x / y
chk(np.allclose(dsl.diff(q, x).eval({"x": xs, "y": yp}), 1.0 / yp), "d(x/y)/dx == 1/y")
chk(np.allclose(dsl.diff(q, y).eval({"x": xs, "y": yp}), -xs / yp ** 2), "d(x/y)/dy == -x/y^2")

# primitive DEFINIE derivee par sa definition (chaine) : f = x*w, w = x/y. d(x*w)/dx = w + x*w'
# = w + x/y ; l'occurrence NON derivee de w reste un symbole (env doit fournir sa valeur w=x/y).
w = dsl.Var("w", "prim")
dfw = dsl.diff(x * w, x, defs={"w": x / y})
env_w = {"x": xs, "y": yp, "w": xs / yp}
chk(np.allclose(dfw.eval(env_w), xs / yp + xs / yp),
    "primitive definie : d(x*w)/dx avec w=x/y == w + x/y (derivee de la definition, chaine)")

# noeud non derivable -> NotImplementedError nommant le type
class _Weird(dsl.Expr):
    pass


try:
    dsl.diff(_Weird(), x)
    chk(False, "noeud inconnu aurait du lever NotImplementedError")
except NotImplementedError as ex:
    chk("Weird" in str(ex), f"noeud inconnu -> NotImplementedError ({str(ex)[:48]})")

# exposant DEPENDANT de la variable (x^x) -> NotImplementedError (logarithme absent du DSL)
try:
    dsl.diff(x ** x, x)
    chk(False, "x^x aurait du lever NotImplementedError")
except NotImplementedError as ex:
    chk("logarithm" in str(ex), f"d(x^x)/dx -> NotImplementedError ({str(ex)[:48]})")


# === (b) m.flux_jacobian : A = dF/dU de l'isotherme 3-var == matrice analytique =================
print("== (b) m.flux_jacobian : Jacobien de flux auto-derive == analytique connue ==")


def iso_model(name):
    """Isotherme 3-var (rho, mx, my), p = cs2 rho, flux et valeurs propres standard."""
    m = dsl.Model(name)
    rho, mx, my = m.conservative_vars("rho", "mx", "my",
                                      roles=["Density", "MomentumX", "MomentumY"])
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    m.primitive("p", CS2 * rho)
    c = dsl.sqrt(CS2)
    m.flux(x=[mx, mx * u + CS2 * rho, mx * v], y=[my, my * u, my * v + CS2 * rho])
    m.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    m.primitive_vars(rho, u, v)
    m.conservative_from([rho, rho * u, rho * v])
    m.elliptic_rhs(0.0 * rho)
    return m


m = iso_model("iso_jac")
N = 50
U = np.stack([rng.uniform(0.5, 2.0, N), rng.uniform(-1.0, 1.0, N), rng.uniform(-1.0, 1.0, N)])
uu, vv = U[1] / U[0], U[2] / U[0]
env = {"rho": U[0], "mx": U[1], "my": U[2], "u": uu, "v": vv, "p": CS2 * U[0]}
zero = np.zeros(N)
# F_x = [mx, mx^2/rho + cs2 rho, mx my/rho] -> A_x analytique :
Ax_ref = [[zero, zero + 1.0, zero],
          [-uu * uu + CS2, 2.0 * uu, zero],
          [-uu * vv, vv, uu]]
# F_y = [my, mx my/rho, my^2/rho + cs2 rho] -> A_y analytique :
Ay_ref = [[zero, zero, zero + 1.0],
          [-uu * vv, vv, uu],
          [-vv * vv + CS2, zero, 2.0 * vv]]
for tag, Aref, d in (("x", Ax_ref, 0), ("y", Ay_ref, 1)):
    A = m.flux_jacobian(d)
    ok = True
    for i in range(3):
        for j in range(3):
            val = np.broadcast_to(A[i][j].eval(env), (N,))
            if not np.allclose(val, Aref[i][j]):
                ok = False
    chk(ok, f"flux_jacobian({d!r}) == A_{tag} analytique (3x3, point a point)")

chk(len(m.flux_jacobian("x")) == 3 and len(m.flux_jacobian(0)) == 3,
    "flux_jacobian accepte 0/'x' et 1/'y'")
try:
    iso_model("iso_nodir").flux_jacobian(2)
    chk(False, "direction invalide aurait du lever")
except ValueError as ex:
    chk("direction" in str(ex), f"direction invalide rejetee ({str(ex)[:40]})")
try:
    mm = dsl.Model("noflux")
    mm.conservative_vars("a", "b", "c")
    mm.flux_jacobian(0)
    chk(False, "flux_jacobian sans flux aurait du lever")
except ValueError as ex:
    chk("set_flux" in str(ex), "flux_jacobian sans set_flux rejete")


# === (d) rejets explicites (purement symboliques) ==============================================
print("== (d) rejets : dimensions, variable hors left/right, conflit enable_roe ==")
L, R = dsl.left, dsl.right


def iso_bare():
    m = dsl.Model("iso_rej")
    rho, mx, my = m.conservative_vars("rho", "mx", "my",
                                      roles=["Density", "MomentumX", "MomentumY"])
    m.primitive("u", mx / rho)
    m.primitive("p", CS2 * rho)
    return m, rho, mx, my


m0, rho0, mx0, my0 = iso_bare()
try:
    m0.roe_dissipation(x=[L(rho0) - R(rho0)], y=[L(rho0), L(mx0), L(my0)])
    chk(False, "dimension fausse aurait du lever")
except ValueError as ex:
    chk("roe_dissipation" in str(ex), "dimensions fausses rejetees")

m0, rho0, mx0, my0 = iso_bare()
try:
    m0.roe_dissipation(x=[L(rho0) - R(rho0), mx0, L(my0)], y=[L(rho0), L(mx0), L(my0)])
    chk(False, "variable nue aurait du lever")
except ValueError as ex:
    chk("'mx'" in str(ex), "variable hors left()/right() rejetee")

m0, rho0, mx0, my0 = iso_bare()
try:
    m0.roe_dissipation(x=[L(R(rho0)), L(mx0), L(my0)], y=[L(rho0), L(mx0), L(my0)])
    chk(False, "marqueur imbrique aurait du lever")
except ValueError as ex:
    chk("left()/right()" in str(ex), "marqueur left()/right() imbrique rejete")

m0, rho0, mx0, my0 = iso_bare()
m0.enable_roe()
try:
    m0.roe_dissipation(x=[L(rho0), L(mx0), L(my0)], y=[L(rho0), L(mx0), L(my0)])
    chk(False, "enable_roe puis roe_dissipation aurait du lever")
except ValueError as ex:
    chk("one single provider" in str(ex), "enable_roe -> roe_dissipation rejete (un seul hook)")

m0, rho0, mx0, my0 = iso_bare()
m0.roe_dissipation(x=[L(rho0), L(mx0), L(my0)], y=[L(rho0), L(mx0), L(my0)])
try:
    m0.enable_roe()
    chk(False, "roe_dissipation puis enable_roe aurait du lever")
except ValueError as ex:
    chk("one single provider" in str(ex), "roe_dissipation -> enable_roe rejete (un seul hook)")


# === (c) bout en bout : roe_dissipation a la main == enable_roe (compile) =======================
cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
if not cxx or not os.path.isdir(INCLUDE):
    print("skip (c) test_dsl_autodiff_roe : compilateur ou en-tetes pops absents")
    if fails:
        print(f"FAIL test_dsl_autodiff_roe : {fails} echec(s)")
        sys.exit(1)
    print("OK test_dsl_autodiff_roe")
    sys.exit(0)


def iso_roe_hand(name):
    """Isotherme 3-var avec roe_dissipation FOURNIE a la main (algebre de Roe isotherme, sans
    Energy, transcrite en left()/right() des deux etats). Pas d'enable_roe : l'utilisateur fournit
    tout le hook."""
    m = dsl.Model(name)
    rho, mx, my = m.conservative_vars("rho", "mx", "my",
                                      roles=["Density", "MomentumX", "MomentumY"])
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    p = m.primitive("p", CS2 * rho)
    c0 = dsl.sqrt(CS2)
    m.flux(x=[mx, mx * u + CS2 * rho, mx * v], y=[my, my * u, my * v + CS2 * rho])
    m.eigenvalues(x=[u - c0, u, u + c0], y=[v - c0, v, v + c0])
    m.primitive_vars(rho, u, v)
    m.conservative_from([rho, rho * u, rho * v])
    m.elliptic_rhs(0.0 * rho)

    def dissipation(norm, tang):
        """Lignes (densite, normale, tangentielle) de d = |A_roe| dU, moyennes de Roe explicites."""
        sqL, sqR = L(dsl.sqrt(rho)), R(dsl.sqrt(rho))
        den = sqL + sqR
        rho_roe = sqL * sqR
        un = (sqL * L(norm) + sqR * R(norm)) / den
        ut = (sqL * L(tang) + sqR * R(tang)) / den
        c = (sqL * dsl.sqrt(L(p) / L(rho)) + sqR * dsl.sqrt(R(p) / R(rho))) / den
        c2 = c * c
        dr = R(rho) - L(rho)
        dp = R(p) - L(p)
        dun = R(norm) - L(norm)
        dut = R(tang) - L(tang)
        a1 = (dp - rho_roe * c * dun) / (2.0 * c2)
        a2 = dr - dp / c2
        a3 = rho_roe * dut
        a5 = (dp + rho_roe * c * dun) / (2.0 * c2)
        al1, al2, al5 = abs(un - c), abs(un), abs(un + c)
        d_dens = al1 * a1 + al2 * a2 + al5 * a5
        d_norm = al1 * a1 * (un - c) + al2 * a2 * un + al5 * a5 * (un + c)
        d_tang = al1 * a1 * ut + al2 * (a2 * ut + a3) + al5 * a5 * ut
        return d_dens, d_norm, d_tang

    dDx, dNx, dTx = dissipation(u, v)  # dir x : normale = u (indice 1), tangentielle = v (indice 2)
    dDy, dNy, dTy = dissipation(v, u)  # dir y : normale = v (indice 2), tangentielle = u (indice 1)
    m.roe_dissipation(x=[dDx, dNx, dTx], y=[dDy, dTy, dNy])
    return m


def iso_roe_roles(name):
    """Meme isotherme avec enable_roe() (capability generee depuis les ROLES) : la reference."""
    m = dsl.Model(name)
    rho, mx, my = m.conservative_vars("rho", "mx", "my",
                                      roles=["Density", "MomentumX", "MomentumY"])
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    m.primitive("p", CS2 * rho)
    c0 = dsl.sqrt(CS2)
    m.flux(x=[mx, mx * u + CS2 * rho, mx * v], y=[my, my * u, my * v + CS2 * rho])
    m.eigenvalues(x=[u - c0, u, u + c0], y=[v - c0, v, v + c0])
    m.primitive_vars(rho, u, v)
    m.conservative_from([rho, rho * u, rho * v])
    m.elliptic_rhs(0.0 * rho)
    m.enable_roe()
    return m


def gaussian(n, amp=0.4):
    xv = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xv, xv, indexing="xy")
    return 1.0 + amp * np.exp(-60.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))


print("== (c) bout en bout : roe_dissipation a la main == enable_roe (riemann='roe') ==")
tmp = tempfile.mkdtemp()
try:
    n = 24
    cm_hand = iso_roe_hand("iso_roe_hand").compile(os.path.join(tmp, "hand.so"), INCLUDE,
                                                   backend="production")
    cm_ref = iso_roe_roles("iso_roe_ref").compile(os.path.join(tmp, "ref.so"), INCLUDE,
                                                  backend="production")
    chk(getattr(cm_hand, "has_roe", False), "CompiledModel.has_roe = True (voie roe_dissipation)")
    chk(getattr(cm_ref, "has_roe", False), "CompiledModel.has_roe = True (voie enable_roe)")

    rho0 = gaussian(n)
    z = np.zeros((n, n))

    s_hand = pops.System(n=n, L=1.0, periodic=True)
    s_hand.set_poisson()
    s_hand.add_equation("f", model=cm_hand,
                        spatial=pops.FiniteVolume(limiter="minmod", riemann="roe"),
                        time=pops.Explicit())
    s_hand.set_primitive_state("f", rho=rho0, u=z + 0.1, v=z)

    s_ref = pops.System(n=n, L=1.0, periodic=True)
    s_ref.set_poisson()
    s_ref.add_equation("f", model=cm_ref,
                       spatial=pops.FiniteVolume(limiter="minmod", riemann="roe"),
                       time=pops.Explicit())
    s_ref.set_primitive_state("f", rho=rho0, u=z + 0.1, v=z)

    for _ in range(8):
        s_hand.step(2e-4)
        s_ref.step(2e-4)
    dh = np.asarray(s_hand.get_state("f"))
    dr = np.asarray(s_ref.get_state("f"))
    chk(np.all(np.isfinite(dh)), "etat fini (roe_dissipation fournie)")
    err = float(np.max(np.abs(dh - dr))) / float(np.max(np.abs(dr)))
    chk(err < 1e-12, f"8 pas : roe_dissipation a la main == enable_roe (ecart rel {err:.2e})")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

if fails:
    print(f"FAIL test_dsl_autodiff_roe : {fails} echec(s)")
    sys.exit(1)
print("OK test_dsl_autodiff_roe")
