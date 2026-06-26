"""Bloc DYNAMIQUE COUPLE de bout en bout : un modele euler_poisson ecrit en FORMULES (flux + source de
gravite + second membre elliptique) -> .so JIT (CompositeModel<hyperbolique, source, elliptique>) ->
sim.add_dynamic_block. On verifie que le bloc charge a l'execution est un VRAI bloc couple, pas un
simple transport :

(A) couplage elliptique : le elliptic_rhs genere est cable dans le Poisson de systeme et donne le MEME
    potentiel que le euler_poisson COMPILE (add_block + briques manuelles GravityCoupling). Le solve de
    Poisson ne depend pas du schema de flux, donc l'egalite isole le second membre par bloc.
(B) terme source : eval_rhs(bloc dynamique) == residu de flux (pops.PythonFlux, a_max GLOBAL, comme le
    chemin hote) + source_value(U, grad phi) evaluee depuis les MEMES formules. Isole la source (le flux
    a deja ete valide contre PythonFlux dans test_dsl_block). On ne compare PAS le flux au bloc compile :
    le bloc dynamique dissipe avec un a_max global, le compile avec des vitesses d'onde locales.
(C) le bloc couple tourne dans le System (transport + Poisson + force a chaque pas) en restant physique
    et en conservant la masse (la source de gravite n'agit pas sur la densite).
"""
import os
import shutil
import tempfile

import numpy as np

import pops
from pops import dsl

GAMMA = 1.4
INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def build_euler_poisson():
    """euler_poisson en formules : Euler compressible + force de gravite (g = -grad phi) + couplage
    self-consistant f = -(rho - 1) (GravityCoupling sign=-1, 4piG=1, rho0=1)."""
    e = dsl.HyperbolicModel("euler_poisson")
    rho, rhou, rhov, E = e.conservative_vars("rho", "rho_u", "rho_v", "E")
    u = e.primitive("u", rhou / rho)
    v = e.primitive("v", rhov / rho)
    p = e.primitive("p", (GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v)))
    H = (E + p) / rho
    c = dsl.sqrt(GAMMA * p / rho)
    e.set_flux(x=[rhou, rhou * u + p, rhou * v, rho * H * u],
               y=[rhov, rhov * u, rhov * v + p, rho * H * v])
    e.set_eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    e.set_primitive_state(rho, u, v, p)
    e.set_conservative_from([rho, rho * u, rho * v, p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)])
    # source de gravite : g = -grad phi ; S = (0, rho gx, rho gy, rho_u gx + rho_v gy) = GravityForce
    gx = e.aux("grad_x")
    gy = e.aux("grad_y")
    e.set_source([0.0, -rho * gx, -rho * gy, -(rhou * gx + rhov * gy)])
    # couplage : f = sign 4piG (rho - rho0), sign=-1 4piG=1 rho0=1 (GravityCoupling)
    e.set_elliptic_rhs(-1.0 * (rho - 1.0))
    return e


def main():
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes pops absents -> bloc dynamique couple saute")
        print("test_dsl_coupled : OK (rien a compiler)")
        return

    e = build_euler_poisson()
    n, L = 48, 1.0
    h = L / n
    tmp = tempfile.mkdtemp()
    try:
        so = e.compile_so(os.path.join(tmp, "euler_poisson.so"), INCLUDE)

        # etat initial : bulle de densite, vitesse nulle, pression uniforme
        xs = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(xs, xs)
        U = np.zeros((4, n, n))
        U[0] = 1.0 + 0.3 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)
        U[3] = 1.0 / (GAMMA - 1.0)  # E = p/(gamma-1), p = 1, energie cinetique nulle
        Uflat = U.reshape(-1).tolist()

        # --- bloc DYNAMIQUE (JIT) euler_poisson ---
        dyn = pops.System(n=n, L=L, periodic=True)
        dyn.add_dynamic_block("gas", so, names=["rho", "rho_u", "rho_v", "E"])
        dyn.set_poisson(rhs="charge_density", solver="geometric_mg")
        dyn.set_state("gas", Uflat)
        dyn.solve_fields()
        phi_dyn = np.array(dyn.potential()).reshape(n, n)
        R_dyn = np.array(dyn.eval_rhs("gas")).reshape(4, n, n)

        # --- euler_poisson COMPILE de reference (memes briques manuelles) ---
        spec = pops.Model(state=pops.FluidState("compressible", gamma=GAMMA),
                         transport=pops.CompressibleFlux(),
                         source=pops.GravityForce(),
                         elliptic=pops.GravityCoupling(sign=-1.0, four_pi_G=1.0, rho0=1.0))
        cmp = pops.System(n=n, L=L, periodic=True)
        cmp.add_block("gas", spec, spatial=pops.Spatial(none=True, flux="rusanov"),
                      time=pops.Explicit())
        cmp.set_poisson(rhs="charge_density", solver="geometric_mg")
        cmp.set_state("gas", Uflat)
        cmp.solve_fields()
        phi_cmp = np.array(cmp.potential()).reshape(n, n)

        # (A) le elliptic_rhs genere est cable dans le Poisson et == le bloc compile
        assert np.abs(phi_dyn).max() > 1e-6, "couplage Poisson inactif (phi nul)"
        dphi = float(np.max(np.abs(phi_dyn - phi_cmp)))
        assert dphi < 1e-7, "potentiel JIT != compile (ecart %.2e) -> elliptic_rhs mal cable" % dphi
        print("OK  elliptic_rhs JIT cable au Poisson, identique au compile (ecart phi %.1e)" % dphi)

        # (B) eval_rhs == residu de flux (PythonFlux, a_max global) + source(U, grad phi)
        pf = e.to_python_flux()  # flux seul, aux nul (Euler n'en depend pas)
        R_flux = pf.residual(U, h)
        gradx = (np.roll(phi_dyn, -1, axis=1) - np.roll(phi_dyn, 1, axis=1)) / (2 * h)
        grady = (np.roll(phi_dyn, -1, axis=0) - np.roll(phi_dyn, 1, axis=0)) / (2 * h)
        S = e.source_value(U, {"grad_x": gradx, "grad_y": grady})  # memes formules, interpretees
        assert float(np.max(np.abs(S[1]))) > 1e-3, "source de gravite triviale (grad phi nul)"
        dres = float(np.max(np.abs(R_dyn - (R_flux + S))))
        assert dres < 1e-9, "eval_rhs != flux + source (ecart %.2e)" % dres
        print("OK  eval_rhs(bloc dynamique) == flux + source de gravite (ecart max %.1e)" % dres)

        # (C) le bloc couple tourne dans le System en restant physique et conservatif
        mass0 = float(np.array(dyn.get_state("gas")).reshape(4, n, n)[0].sum())
        for _ in range(15):
            dyn.step_cfl(0.4)  # transport + Poisson + force, par pas
        U1 = np.array(dyn.get_state("gas")).reshape(4, n, n)
        drel = abs(float(U1[0].sum()) - mass0) / mass0
        assert np.isfinite(U1).all() and U1[0].min() > 0, "etat non physique"
        assert drel < 1e-9, "masse non conservee (drel=%.2e)" % drel
        assert float(np.abs(U1[1]).max()) > 1e-4, "la force de gravite n'a pas mis le gaz en mouvement"
        print("OK  bloc couple avance dans le System (15 pas, masse drel=%.1e)" % drel)
        print("test_dsl_coupled : tout est vert")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
