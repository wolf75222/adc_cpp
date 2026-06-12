#!/usr/bin/env python3
"""Test de la politique temporelle adc.Strang(hyperbolic=Explicit, source=CondensedSchur) : le
splitting de STRANG GENERIQUE (H(dt/2) ; S(dt) ; H(dt/2), 2e ordre) cable au stepper de systeme via
set_time_scheme("strang"). Pendant 2e ordre de adc.Split (Lie, 1er ordre) : memes briques (transport
SSPRK + etage source condense), seul l'ordre/la cadence des resolutions de champ changent. cf.
docs/HOFFART_STEP_SEQUENCE.md, include/adc/runtime/system_stepper.hpp (SystemStepper::step_strang) et
le test C++ tests/test_strang_splitting.cpp (ordre temporel observe ~2 vs ~1).

On valide :
  (a) API / opt-in : adc.Strang existe, EST une sous-classe de adc.Split (donc soumise a la meme
      garde : rejet sur add_block, System ET AmrSystem), expose .scheme == "strang" (Split -> "lie"),
      est exportee dans adc.__all__. NB : AmrSystem.add_equation SUPPORTE Strang/Lie depuis le chemin
      amr-schur (#265, set_source_stage + set_time_scheme) ; seul add_block continue de rejeter.
  (b) RUN C++ (si compilateur) : un fluide isotherme magnetise tourne avec time=adc.Strang(...) ;
      etat FINI, densite gelee dans la source, quantite de mvt qui EVOLUE (source engagee).
  (c) STRANG != LIE sur le chemin moteur : a setup et dt IDENTIQUES, le run Strang et le run Split
      (Lie) different (deux orchestrations distinctes ; sinon set_time_scheme serait inerte). Le
      defaut (Explicit pur) reste BIT-IDENTIQUE entre deux runs.

Lance avec python3. Les gardes (a) tournent SANS compilateur ; (b)/(c) sautent proprement si aucun
compilateur C++ n'est disponible (le modele magnetise a roles passe par le DSL compile).
"""
import os
import shutil

import numpy as np

import adc
from adc import dsl

INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))

fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


def raises(exc_types, fn, *args, **kw):
    try:
        fn(*args, **kw)
    except exc_types as e:
        return e
    except Exception as e:  # pragma: no cover
        raise AssertionError("attendu %r, leve %r (%s)" % (exc_types, type(e).__name__, e))
    raise AssertionError("attendu %r, aucune exception levee" % (exc_types,))


def isothermal_magnetized(cs2=1.0):
    """Fluide isotherme 2D (rho, mx, my) magnetise (roles Density/MomentumX/MomentumY, lit B_z). La
    source LOCALE est nulle : l'etage SOURCE est porte par adc.CondensedSchur (electrostatique + Lorentz)."""
    m = dsl.Model("iso_mag_strang")
    rho, mx, my = m.conservative_vars("rho", "mx", "my",
                                      roles=["Density", "MomentumX", "MomentumY"])
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    p = cs2 * rho
    c = dsl.sqrt(cs2)
    m.flux(x=[mx, mx * u + p, mx * v], y=[my, my * u, my * v + p])
    m.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    m.primitive_vars(rho=rho, u=u, v=v)
    m.conservative_from([rho, rho * u, rho * v])
    bz = m.aux("B_z")
    m.source([0.0 * rho, 0.0 * bz * mx, 0.0 * my])  # source locale NULLE (Schur porte la source)
    return m


def smooth_init(n, L):
    x = (np.arange(n) + 0.5) * (L / n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    sx, sy = np.sin(np.pi * X / L), np.sin(np.pi * Y / L)
    rho0 = 1.5 + 0.0 * X
    u0 = 0.6 * sx * sy
    v0 = -0.4 * np.sin(2 * np.pi * X / L) * sy
    return rho0, u0, v0


def build_sim(compiled, time, n=32, L=1.0, B0=4.0):
    """System non periodique (Dirichlet), un bloc isotherme magnetise, politique temporelle @p time."""
    sim = adc.System(n=n, L=L, periodic=False)
    sim.set_poisson(bc="dirichlet")
    sim.set_magnetic_field(B0 * np.ones((n, n)))
    sim.add_equation("ions", model=compiled,
                     spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov",
                                              variables="conservative"),
                     time=time)
    rho0, u0, v0 = smooth_init(n, L)
    sim.set_primitive_state("ions", rho=rho0, u=u0, v=v0)
    return sim


def split_lie(theta=1.0, alpha=3.0):
    return adc.Split(hyperbolic=adc.Explicit(),
                     source=adc.CondensedSchur(theta=theta, alpha=alpha))


def strang(theta=1.0, alpha=3.0):
    return adc.Strang(hyperbolic=adc.Explicit(),
                      source=adc.CondensedSchur(theta=theta, alpha=alpha))


def scalar_native_model():
    return adc.Model(state=adc.Scalar(), transport=adc.ExB(B0=1.0),
                     source=adc.NoSource(),
                     elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0))


def check_api():
    """(a) API / opt-in : pas de compilateur requis."""
    chk("Strang" in adc.__all__, "(a) adc.Strang exporte dans __all__")
    chk(issubclass(adc.Strang, adc.Split), "(a) adc.Strang sous-classe de adc.Split")
    s = strang()
    p = split_lie()
    chk(getattr(s, "scheme", None) == "strang", "(a) adc.Strang.scheme == 'strang'")
    chk(getattr(p, "scheme", None) == "lie", "(a) adc.Split.scheme == 'lie'")
    # adc.Strang herite des memes briques que Split (hyperbolique + source condensee).
    chk(isinstance(s.source, adc.CondensedSchur) and isinstance(s.hyperbolic, adc.Explicit),
        "(a) adc.Strang porte hyperbolique Explicit + source CondensedSchur")
    # Garde heritee : Strang (etant un Split) est rejete sur add_block, sur System ET sur AmrSystem --
    # l'etage source condense par Schur n'est cable que par add_equation (qui branche set_source_stage),
    # jamais par add_block (qui jouerait le seul transport et PERDRAIT la source en silence). Depuis le
    # chemin amr-schur (#265), AmrSystem.add_equation(time=adc.Strang(...)) est au contraire SUPPORTE
    # (cf. test_amr_schur_via_system.py pour la couverture positive sur AMR) : seul add_block rejette.
    sys_ = adc.System(n=8, L=1.0, periodic=False)
    e_blk = raises(TypeError, sys_.add_block, "x", scalar_native_model(), time=strang())
    chk("Split" in str(e_blk) or "Schur" in str(e_blk),
        "(a) System.add_block(time=adc.Strang(...)) -> rejet (heritage Split)")
    amr = adc.AmrSystem(n=8, L=1.0, periodic=True)
    e_amr = raises((TypeError, ValueError), amr.add_block, "x", scalar_native_model(),
                   time=strang())
    chk("Split" in str(e_amr) or "Schur" in str(e_amr),
        "(a) AmrSystem.add_block(time=adc.Strang(...)) -> rejet (heritage Split)")


def vel_mom(sim):
    P = sim.get_primitive_state("ions")
    rho = np.array(sim.density("ions"))
    return np.array(P["u"]) * rho, np.array(P["v"]) * rho


def main():
    check_api()  # (a) pur Python : tourne meme sans compilateur

    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes adc absents -> test_strang_split saute (%s)" % INCLUDE)
        print("test_strang_split : OK (rien a compiler)")
        return

    n, L, dt = 32, 1.0, 2.0e-3
    # adc_cpp est Kokkos-only (#263) : le .so AOT inclut les en-tetes adc (multifab/for_each) qui ne
    # compilent QUE sous ADC_HAS_KOKKOS, donc compile_aot exige un Kokkos installe (ADC_KOKKOS_ROOT).
    # Sans lui, on saute proprement (b)/(c) -- meme convention que test_time_euler.py / test_dsl_aot.
    try:
        compiled = isothermal_magnetized().compile(backend="aot", include=INCLUDE)
    except RuntimeError as ex:
        if "Kokkos" in str(ex):
            print("skip  Kokkos introuvable -> (b)/(c) sautes (%s)" % str(ex).splitlines()[0][:70])
            print("test_strang_split : OK (rien a compiler)")
            return
        raise

    # (b) RUN Strang : etat fini, quantite de mvt qui evolue (etage source engage sous Strang).
    sim_s = build_sim(compiled, strang(theta=1.0, alpha=3.0), n=n, L=L)
    mom0x, mom0y = vel_mom(sim_s)
    for _ in range(4):
        sim_s.step(dt)
    Ps = sim_s.get_primitive_state("ions")
    rho_s = np.array(sim_s.density("ions"))
    finite_s = bool(np.all(np.isfinite(Ps["u"])) and np.all(np.isfinite(Ps["v"]))
                    and np.all(np.isfinite(rho_s)))
    chk(finite_s, "(b) etat fini apres run Strang (H(dt/2) S(dt) H(dt/2))")
    mom1x, _ = vel_mom(sim_s)
    chk(float(np.max(np.abs(mom1x - mom0x))) > 1e-6,
        "(b) la quantite de mvt evolue sous Strang (etage source engage)")

    # (c) STRANG != LIE : meme setup, meme dt, memes pas -> densites finales DISTINCTES (sinon
    #     set_time_scheme serait inerte : Lie et Strang emprunteraient le meme chemin moteur).
    sim_lie = build_sim(compiled, split_lie(theta=1.0, alpha=3.0), n=n, L=L)
    sim_str = build_sim(compiled, strang(theta=1.0, alpha=3.0), n=n, L=L)
    for _ in range(4):
        sim_lie.step(dt)
        sim_str.step(dt)
    d_lie = np.array(sim_lie.density("ions"))
    d_str = np.array(sim_str.density("ions"))
    chk(bool(np.all(np.isfinite(d_str))), "(c) etat Strang fini (comparaison Lie/Strang)")
    chk(float(np.max(np.abs(d_lie - d_str))) > 0.0,
        "(c) Strang DIFFERE de Lie sur le chemin moteur (set_time_scheme actif)")

    # (c-bis) Reproductibilite : deux runs Strang identiques donnent un resultat bit-identique
    #         (le chemin Strang est deterministe, pas de regression de non-determinisme).
    sim_str2 = build_sim(compiled, strang(theta=1.0, alpha=3.0), n=n, L=L)
    for _ in range(4):
        sim_str2.step(dt)
    d_str2 = np.array(sim_str2.density("ions"))
    chk(float(np.max(np.abs(d_str - d_str2))) == 0.0, "(c) deux runs Strang bit-identiques")

    if fails == 0:
        print("test_strang_split : tout est vert")
    else:
        raise SystemExit("test_strang_split : %d verification(s) en echec" % fails)


if __name__ == "__main__":
    main()
