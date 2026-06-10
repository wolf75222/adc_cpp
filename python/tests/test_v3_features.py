#!/usr/bin/env python3
"""Vague 3 (solde des restes de genericite) : couverture facade.

  (A) CoupledSource.frequency : la 'CFL de couplage' declaree borne le pas
      (dt == cfl/mu, raison 'coupled_source:<nom>') -- System, sans compilateur ; et un couplage
      REJETE ne laisse AUCUNE borne fantome (frequence enregistree apres validation, revue v3) ;
  (B) options Newton sur AMR : TRANSPORTEES en multi-blocs natif (run fini, fail_policy acceptee),
      REJET explicite en mono-bloc (iters=2 fige cote coupleur) et pour newton_diagnostics ;
  (C) set_conservative_state MULTI-BLOCS : l'etat complet (avec quantite de mouvement) seede le
      grossier (la masse et la dynamique different du seed densite au repos) ; et AmrSystem.write
      npz ecrit un champ PAR bloc par NOM (pas seulement le bloc 0, revue v3) ;
  (D, compilateur) enable_hllc : riemann='hllc' accepte sur un modele DSL 3-var NON Euler via la
      capability emise ; rejete sans elle ; source_jacobian : parite de trajectoire avec les FD ;
      garde CODEGEN : source_jacobian sans source leve a compile() (pas seulement check()).

Invariants par assert ; imprime "OK test_v3_features" en cas de succes.
"""
import os
import shutil
import sys
import tempfile

import numpy as np

import adc
from adc import dsl

fails = 0
INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def iso_model(charge=1.0):
    return adc.Model(state=adc.FluidState("isothermal", cs2=0.5),
                     transport=adc.IsothermalFlux(),
                     source=adc.PotentialForce(charge=charge),
                     elliptic=adc.ChargeDensity(charge=charge))


def gaussian(n):
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="xy")
    return 1.0 + 0.4 * np.exp(-60.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))


# --- (A) CoupledSource.frequency ---------------------------------------------------
print("== (A) CoupledSource.frequency : borne dt <= cfl/mu sur le macro-pas ==")
n = 16
sim = adc.System(n=n, L=1.0, periodic=True)
sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
sim.add_block("a", iso_model(+1.0), spatial=adc.FiniteVolume(limiter="minmod"))
sim.add_block("b", iso_model(-1.0), spatial=adc.FiniteVolume(limiter="minmod"))
sim.set_density("a", gaussian(n).ravel())
sim.set_density("b", gaussian(n).ravel())
src = dsl.CoupledSource("friction").frequency(500.0)  # mu = 500 -> dt = 0.4/500 = 8e-4 << transport
na = src.block("a").role("density")
k = src.param("k", 1e-3)
src.add_pair("a", "b", role="density", expr=k * na)
sim.add_coupling(src.compile())
dt = sim.step_cfl(0.4)
chk(abs(dt - 0.4 / 500.0) < 1e-15, f"dt = cfl/mu = 8e-4 ({dt:.3e})")
chk(sim.last_dt_bound() == "coupled_source:friction",
    f"borne active = coupled_source:friction (recu {sim.last_dt_bound()!r})")
# Pas de BORNE FANTOME (revue vague 3) : un couplage REJETE (role absent du bloc) ne doit laisser
# AUCUNE frequence enregistree -- sinon le pas serait bride par une physique inexistante.
ghost = dsl.CoupledSource("ghost").frequency(5000.0)  # 0.4/5000 = 8e-5 << 8e-4 si fantome
ng = ghost.block("a").role("density")
kg = ghost.param("kg", 1e-3)
ghost.add_pair("a", "b", role="energy", expr=kg * ng)  # isotherme : pas de role Energy -> rejet C++
try:
    sim.add_coupling(ghost.compile())
    chk(False, "couplage sur role absent aurait du lever")
except (RuntimeError, ValueError):
    pass
dt2 = sim.step_cfl(0.4)
chk(abs(dt2 - 0.4 / 500.0) < 1e-15 and sim.last_dt_bound() == "coupled_source:friction",
    f"couplage rejete = ZERO borne fantome (dt {dt2:.3e}, borne {sim.last_dt_bound()!r})")

# --- (B) options Newton sur AMR ------------------------------------------------------
print("== (B) AMR : options Newton transportees (multi-blocs), rejets mono/diagnostics ==")
amr = adc.AmrSystem(n=16, L=1.0, periodic=True, regrid_every=0)
amr.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
amr.set_refinement(1e30)
amr.add_block("e1", iso_model(+1.0), spatial=adc.FiniteVolume(limiter="minmod"),
              time=adc.IMEX(newton_max_iters=4, newton_fail_policy="warn"))
amr.add_block("e2", iso_model(-1.0), spatial=adc.FiniteVolume(limiter="minmod"),
              time=adc.Explicit())
amr.set_density("e1", gaussian(16).ravel())
amr.set_density("e2", gaussian(16).ravel())
amr.step(2e-3)
chk(np.all(np.isfinite(np.asarray(amr.density("e1")))),
    "multi-blocs : IMEX(newton_max_iters=4, fail_policy='warn') tourne fini")
mono = adc.AmrSystem(n=16, L=1.0, periodic=True, regrid_every=0)
mono.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
mono.set_refinement(1e30)
mono.add_block("e", iso_model(), spatial=adc.FiniteVolume(limiter="minmod"),
               time=adc.IMEX(newton_max_iters=4))
mono.set_density("e", gaussian(16).ravel())
try:
    mono.step(2e-3)  # build paresseux mono-bloc -> rejet explicite
    chk(False, "mono-bloc + options Newton aurait du lever au build")
except RuntimeError as e:
    chk("MULTI-BLOCS" in str(e), f"mono-bloc rejete : {str(e)[:70]}")
try:
    adc.AmrSystem(n=8, periodic=True, regrid_every=0).add_block(
        "e", iso_model(), time=adc.IMEX(newton_diagnostics=True))
    chk(False, "newton_diagnostics sur AMR aurait du lever")
except ValueError as e:
    chk("newton_report" in str(e) or "diagnostics" in str(e),
        f"diagnostics rejetes : {str(e)[:70]}")

# --- (C) set_conservative_state multi-blocs ------------------------------------------
print("== (C) set_conservative_state multi-blocs : etat complet seede (avec derive) ==")
amr3 = adc.AmrSystem(n=16, L=1.0, periodic=True, regrid_every=0)
amr3.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
amr3.set_refinement(1e30)
amr3.add_block("e1", iso_model(+1.0), spatial=adc.FiniteVolume(limiter="minmod"))
amr3.add_block("e2", iso_model(-1.0), spatial=adc.FiniteVolume(limiter="minmod"))
rho0 = gaussian(16)
u0 = 0.3 * np.ones((16, 16))
amr3.set_conservative_state("e1", np.stack([rho0, rho0 * u0, 0.0 * rho0]))
amr3.set_density("e2", rho0.ravel())
d_before = np.asarray(amr3.density("e1")).reshape(16, 16).copy()
amr3.step(2e-3)
d_after = np.asarray(amr3.density("e1")).reshape(16, 16)
chk(np.all(np.isfinite(d_after)), "multi-blocs + etat complet : pas fini")
# la derive u0=0.3 advecte la bosse -> la densite CHANGE des le 1er pas (un seed au repos ne
# bougerait quasiment pas en 1 pas) : preuve que la quantite de mouvement a ete seedee.
chk(float(np.max(np.abs(d_after - d_before))) > 1e-5,
    "la quantite de mouvement seedee advecte la densite (etat complet actif)")
# AmrSystem.write multi-blocs (revue vague 3) : un champ PAR bloc, par NOM (e1 ET e2) -- pas
# seulement le bloc 0 sous une cle generique.
wdir = tempfile.mkdtemp()
try:
    fnpz = amr3.write(os.path.join(wdir, "amr_out"), format="npz")
    with np.load(fnpz) as z:
        chk("density_e1" in z.files and "density_e2" in z.files and "phi" in z.files,
        f"AmrSystem.write npz multi-blocs : density_e1 + density_e2 + phi (cles {z.files})")
finally:
    shutil.rmtree(wdir, ignore_errors=True)

# --- (D) DSL : enable_hllc + source_jacobian (compilateur requis) ---------------------
cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
if not cxx or not os.path.isdir(INCLUDE):
    print("skip  (D) : compilateur ou en-tetes adc absents")
    if fails:
        print(f"FAIL test_v3_features : {fails} echec(s)")
        sys.exit(1)
    print("OK test_v3_features (A-C)")
    sys.exit(0)


def iso3_dsl(name, hllc=False, jac=False):
    m = dsl.Model(name)
    rho, mx, my = m.conservative_vars("rho", "mx", "my", roles=["Density", "MomentumX", "MomentumY"])
    cs2 = 0.5
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    m.primitive("p", cs2 * rho)
    c = dsl.sqrt(cs2)
    m.flux(x=[mx, mx * u + cs2 * rho, mx * v], y=[my, my * u, my * v + cs2 * rho])
    m.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    m.primitive_vars(rho, u, v)
    m.conservative_from([rho, rho * u, rho * v])
    m.elliptic_rhs(0.0 * rho)
    if hllc:
        m.enable_hllc()
    if jac:
        kk = 50.0
        m.source([0.0 * rho, -kk * mx, -kk * my])  # friction raide lineaire
        m.source_jacobian([[0.0 * rho, 0.0 * rho, 0.0 * rho],
                           [0.0 * rho, -kk + 0.0 * rho, 0.0 * rho],
                           [0.0 * rho, 0.0 * rho, -kk + 0.0 * rho]])
    return m


tmp = tempfile.mkdtemp()
try:
    print("== (D1) enable_hllc : riemann='hllc' sur 3-var NON Euler ==")
    cm_h = iso3_dsl("iso3_hllc", hllc=True).compile(os.path.join(tmp, "iso3_hllc.so"), INCLUDE,
                                                    backend="production")
    chk(getattr(cm_h, "has_hllc", False), "CompiledModel.has_hllc = True (capability emise)")
    sh = adc.System(n=24, L=1.0, periodic=True)
    sh.set_poisson()
    sh.add_equation("f", model=cm_h, spatial=adc.FiniteVolume(limiter="minmod", riemann="hllc"),
                    time=adc.Explicit())
    z = np.zeros((24, 24))
    sh.set_primitive_state("f", rho=gaussian(24), u=z, v=z)
    for _ in range(5):
        sh.step_cfl(0.3)
    chk(np.all(np.isfinite(np.asarray(sh.density("f")))),
        "HLLC capability sur 3-var : 5 pas finis (contact-resolving hors Euler)")
    cm_nh = iso3_dsl("iso3_nohllc").compile(os.path.join(tmp, "iso3_nohllc.so"), INCLUDE,
                                            backend="production")
    try:
        s2 = adc.System(n=16, L=1.0, periodic=True)
        s2.add_equation("f", model=cm_nh, spatial=adc.FiniteVolume(limiter="minmod",
                                                                   riemann="hllc"))
        chk(False, "hllc sans capability sur 3-var aurait du lever")
    except (ValueError, RuntimeError) as e:
        chk("hllc" in str(e), f"rejet sans capability : {str(e)[:70]}")

    print("== (D2) source_jacobian : meme trajectoire que les differences finies ==")
    cm_j = iso3_dsl("iso3_jac", jac=True).compile(os.path.join(tmp, "iso3_jac.so"), INCLUDE,
                                                  backend="production")
    cm_f = iso3_dsl("iso3_fd", jac=True)
    cm_f._m._src_jac = None  # meme modele, SANS jacobien emis -> FD historiques
    cm_f = cm_f.compile(os.path.join(tmp, "iso3_fd.so"), INCLUDE, backend="production")

    def run_imex(cm):
        s = adc.System(n=16, L=1.0, periodic=True)
        s.set_poisson()
        s.add_equation("f", model=cm, spatial=adc.FiniteVolume(limiter="minmod"),
                       time=adc.IMEX())
        z16 = np.zeros((16, 16))
        s.set_primitive_state("f", rho=gaussian(16), u=0.2 + z16, v=z16)
        for _ in range(4):
            s.step(1e-3)
        return np.asarray(s.get_state("f"))

    uj, uf = run_imex(cm_j), run_imex(cm_f)
    chk(np.allclose(uj, uf, rtol=1e-9, atol=1e-11),
        f"jacobien analytique ~ FD (ecart max {float(np.max(np.abs(uj - uf))):.2e})")
    chk(np.all(np.isfinite(uj)), "source raide (k*dt = 0.05*50) : etat fini")

    print("== (D3) garde CODEGEN : source_jacobian sans source -> erreur (pas de purge muette) ==")
    mg = iso3_dsl("iso3_guard", jac=True)
    mg._m._source = None  # jacobien declare, source retiree : compile() doit lever (pas check())
    try:
        mg.compile(os.path.join(tmp, "iso3_guard.so"), INCLUDE, backend="production")
        chk(False, "source_jacobian sans source aurait du lever au codegen")
    except ValueError as e:
        chk("sans source" in str(e), f"codegen leve : {str(e)[:70]}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

if fails:
    print(f"FAIL test_v3_features : {fails} echec(s)")
    sys.exit(1)
print("OK test_v3_features")
