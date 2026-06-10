"""AmrSystem.set_conservative_state : demarrer l'AMR depuis un ETAT CONSERVATIF COMPLET (rho, rho_u,
rho_v) au lieu de la densite seule (qty de mouvement = 0). C'est le Probleme 2 de la reproduction
Hoffart (arXiv:2510.11808) cote AMR : la vitesse de derive v0 du papier doit etre posee a l'init.

Le chemin C++ (build_amr_compiled / build_amr_block, seed du grossier) ecrit l'etat complet via
coupler_write_coarse_state puis le prolonge aux niveaux fins (injection constante de TOUTES les
composantes). Mono-bloc ET multi-blocs natif (vague 3) ; loaders .so multi-blocs = rejet explicite.

On verifie :
 (G) GARDES (sans compilateur) : ndim != 3 rejete au binding ; etat vide rejete ; taille non multiple
     de n*n rejetee ; multi-blocs natif ACCEPTE au build et tourne fini (cable vague 3).
 (A) NO-DEFAULT-CHANGE : set_conservative_state([rho, 0, 0]) == set_density([rho]) a la cellule pres
     (modele isotherme 3-var, AUCUNE energie -> pas de bruit gamma -> BIT-IDENTIQUE, dmax == 0).
 (D) CONSERVATION : depuis un etat porteur de qty de mouvement, la masse est conservee sur N pas.
 (E) LA QTY DE MOUVEMENT BOUGE LA DENSITE : un etat [rho, rho*u0, 0] (u0 > 0) advecte le pic de
     densite en +x, contrairement a l'init densite-seule (m=0) qui ne bouge pas au 1er pas. Verifie
     que la qty de mouvement du seed est bien posee ET prolongee, SANS lecture inverse de l'etat.

S'auto-saute (exit 0) pour (A)/(D)/(E) sans compilateur C++ ; les gardes (G) tournent toujours.
"""
import os
import shutil
import sys
import tempfile

import numpy as np

import adc
from adc import dsl

GAMMA = 1.4
INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def raises(fn):
    try:
        fn()
        return False
    except Exception:
        return True


def _euler_spec():
    """Bloc natif compressible (4 var) -- pas de compilateur requis (gardes pre-build)."""
    return adc.Model(state=adc.FluidState("compressible", gamma=GAMMA),
                     transport=adc.CompressibleFlux(), source=adc.NoSource(),
                     elliptic=adc.BackgroundDensity(alpha=0.0, n0=0.0))


def _bump(n):
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    return 1.0 + 0.5 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01)


def _amr(n, L=1.0):
    s = adc.AmrSystem(n=n, L=L, periodic=True)
    s.set_refinement(1.2)
    return s


# --- (G) GARDES (sans compilateur) -----------------------------------------------------------------
print("== (G) gardes de set_conservative_state ==")
n = 32
rho = _bump(n)

# ndim != 3 (densite 2D passee par erreur) -> rejet au binding.
s = _amr(n); s.add_block("gas", _euler_spec(), time=adc.Explicit())
chk(raises(lambda: s.set_conservative_state("gas", rho)),
    "(G) ndim==2 (densite) rejete (attendu (ncomp, n, n))")

# etat vide -> rejet.
s = _amr(n); s.add_block("gas", _euler_spec(), time=adc.Explicit())
chk(raises(lambda: s.set_conservative_state("gas", np.zeros((0, n, n)))),
    "(G) etat vide rejete")

# taille non multiple de n*n -> rejet (ncomp*n*(n-1) p.ex.).
s = _amr(n); s.add_block("gas", _euler_spec(), time=adc.Explicit())
bad = np.zeros((3, n, n - 1))
chk(raises(lambda: s.set_conservative_state("gas", bad)),
    "(G) taille non multiple de n*n rejetee")

# systeme deja construit -> rejet (poser l'etat avant le build).
s = _amr(n); s.add_block("gas", _euler_spec(), time=adc.Explicit())
s.set_density("gas", rho); s.step(1e-4)  # force le build
chk(raises(lambda: s.set_conservative_state("gas", np.stack([rho, 0 * rho, 0 * rho, rho]))),
    "(G) set_conservative_state apres build rejete")

# MULTI-BLOCS natif : CABLE depuis la vague 3 (l'etat complet seede le grossier via
# coupler_write_coarse_state) -- le build N'EST PLUS un rejet ; le pas tourne fini.
# (test_v3_features (C) prouve en plus que la qty de mouvement seedee advecte.)
s = _amr(n)
s.add_block("a", _euler_spec(), time=adc.Explicit())
s.add_block("b", _euler_spec(), time=adc.Explicit())
s.set_conservative_state("a", np.stack([rho, 0 * rho, 0 * rho, rho / (GAMMA - 1.0)]))
s.set_density("b", rho)
s.step(1e-4)
chk(np.all(np.isfinite(np.asarray(s.density("a")))),
    "(G) set_conservative_state + multi-blocs natif -> accepte au build, pas fini (vague 3)")

# --- (A)/(D)/(E) : necessitent un compilateur (modele isotherme 3-var compile production) -----------
cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
if not cxx or not os.path.isdir(INCLUDE):
    print("skip  (A)/(D)/(E) : compilateur ou en-tetes adc absents")
    print("test_amr_conservative_state : OK (gardes vertes)" if fails == 0 else f"{fails} ECHEC(S)")
    sys.exit(0 if fails == 0 else 1)


def _isothermal():
    """Isotherme 3-var (rho, rho_u, rho_v), c^2 = 0.5 : la qty de mouvement advecte la densite et il
    n'y a PAS d'energie -> seed conservatif == seed densite a la cellule pres (bit-identique)."""
    m = dsl.Model("iso3")
    rho_, rhou_, rhov_ = m.conservative_vars("rho", "rho_u", "rho_v")
    cs2 = 0.5
    u = rhou_ / rho_
    v = rhov_ / rho_
    pu, pv = m.primitive("u", u), m.primitive("v", v)
    m.flux(x=[rhou_, rhou_ * pu + cs2 * rho_, rhou_ * pv],
           y=[rhov_, rhov_ * pu, rhov_ * pv + cs2 * rho_])
    m.eigenvalues(x=[pu - dsl.sqrt(cs2), pu, pu + dsl.sqrt(cs2)],
                  y=[pv - dsl.sqrt(cs2), pv, pv + dsl.sqrt(cs2)])
    m.primitive_vars(rho_, pu, pv)
    m.conservative_from([rho_, rho_ * pu, rho_ * pv])
    m.elliptic_rhs(0.0 * rho_)  # transport pur (pas de Poisson) : isole l'effet du seed
    return m


tmp = tempfile.mkdtemp()
try:
    cm = _isothermal().compile(os.path.join(tmp, "iso3_amr.so"), INCLUDE,
                               backend="production", target="amr_system")

    def build(setter):
        s = _amr(n)
        s.add_equation("gas", cm, spatial=adc.Spatial(minmod=True, flux="rusanov",
                                                      recon="conservative"))
        setter(s)
        return s

    dt = 2e-4
    zero = np.zeros((n, n))

    # (A) NO-DEFAULT-CHANGE : [rho, 0, 0] (etat) == set_density(rho).
    Asd = build(lambda s: s.set_density("gas", rho))
    Acs = build(lambda s: s.set_conservative_state("gas", np.stack([rho, zero, zero])))
    for _ in range(10):
        Asd.step(dt); Acs.step(dt)
    dsd, dcs = np.array(Asd.density()), np.array(Acs.density())
    dmax = float(np.max(np.abs(dsd - dcs)))
    chk(float(np.max(np.abs(dsd))) > 1e-6, "(A) densite non triviale")
    chk(dmax == 0.0, "(A) set_conservative_state([rho,0,0]) BIT-IDENTIQUE a set_density([rho]) (dmax=%.1e)" % dmax)

    # (D) CONSERVATION : etat porteur de qty de mouvement, masse conservee sur N pas (periodique).
    u0 = 0.3
    Dm = build(lambda s: s.set_conservative_state("gas", np.stack([rho, u0 * rho, zero])))
    m0 = float(Dm.mass())
    for _ in range(20):
        Dm.step(dt)
    chk(np.isfinite(np.array(Dm.density())).all(), "(D) etat fini sur 20 pas")
    chk(abs(float(Dm.mass()) - m0) < 1e-12 * abs(m0), "(D) masse conservee (rel < 1e-12)")

    # (E) LA QTY DE MOUVEMENT BOUGE LA DENSITE : centroid_x(etat avec u0>0) > centroid_x(densite seule).
    def centroid_x(field):
        xs = (np.arange(field.shape[1]) + 0.5) / field.shape[1]
        w = field - field.min()  # retire le fond pour isoler le pic
        return float((w * xs[None, :]).sum() / w.sum())

    Em0 = build(lambda s: s.set_density("gas", rho))            # m = 0
    Emu = build(lambda s: s.set_conservative_state("gas", np.stack([rho, u0 * rho, zero])))  # m = rho*u0 ex
    for _ in range(40):
        Em0.step(dt); Emu.step(dt)
    cx0 = centroid_x(np.array(Em0.density()))
    cxu = centroid_x(np.array(Emu.density()))
    chk(cxu - cx0 > 1e-4,
        "(E) qty de mvt advecte le pic en +x : centroid(u0) %.5f > centroid(m=0) %.5f" % (cxu, cx0))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("test_amr_conservative_state : tout est vert" if fails == 0 else f"{fails} ECHEC(S)")
sys.exit(0 if fails == 0 else 1)
