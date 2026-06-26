"""Chantier Aux extensible : T_e (2e champ extra, DERIVE) pilotable via le chemin JIT.

Symetrique de test_dsl_jit_bz, mais pour le champ T_e (composante aux 4) qui est DERIVE par le
System (T = p/rho d'un bloc fluide designe) et non fourni par l'utilisateur comme B_z. Un modele
DSL dynamique lit aux('T_e') -> brique GenSrc n_aux=5 -> .so JIT charge par add_dynamic_block : son
IModel expose n_aux()=5, le System elargit le canal aux partage et MARSHALE 5 composantes vers le
chemin hote (host_residual doit lire a.T_e ; sinon T_e=0 et le residu serait faux). Sans ce
marshaling, T_e marchait en natif (add_compiled_model) et en AOT mais PAS en JIT : garde-fou ici.
On verifie de bout en bout cote Python : eval_rhs du bloc dynamique = source = T_e * n.
"""
import os
import shutil
import tempfile

import numpy as np

import pops
from pops import dsl

INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def build_te_scalar():
    """Scalaire transporte sans flux, source = T_e * n (lit aux('T_e') -> n_aux=5)."""
    m = dsl.HyperbolicModel("tescalar")
    (nn,) = m.conservative_vars("n")
    zero = 0.0 * nn                        # expression nulle (set_flux n'enrobe pas les floats bruts)
    m.set_flux(x=[zero], y=[zero])         # flux nul -> R = source
    m.set_eigenvalues(x=[zero], y=[zero])  # pas de transport
    m.set_primitive_state(nn)
    m.set_conservative_from([nn])
    te = m.aux("T_e")
    m.set_source([te * nn])                # S = T_e n
    return m


def euler_gas(gamma=1.4):
    """Bloc fluide compressible (4 var) : source de T_e via T = p/rho."""
    return pops.Model(state=pops.FluidState(kind="compressible", gamma=gamma),
                     transport=pops.CompressibleFlux(), source=pops.NoSource(),
                     elliptic=pops.ChargeDensity(charge=1.0))


def main():
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes adc absents -> JIT T_e saute")
        print("test_dsl_jit_te : OK (rien a compiler)")
        return

    m = build_te_scalar()
    n, L, gamma, c = 32, 1.0, 1.4, 0.6
    tmp = tempfile.mkdtemp()
    try:
        so = m.compile_so(os.path.join(tmp, "tescalar.so"), INCLUDE)

        sim = pops.System(n=n, L=L, periodic=True)
        sim.add_dynamic_block("te", so, names=["n"])  # IModel.n_aux()=5 -> ensure_aux_width(5)
        sim.add_block("gas", model=euler_gas(gamma))  # bloc fluide source de T_e
        sim.set_poisson(rhs="charge_density", solver="geometric_mg")
        sim.set_density("te", np.ones((n, n)))

        # gas uniforme : rho=1, m=0, E=c/(gamma-1) -> p=(gamma-1)E=c, T=p/rho=c.
        E = c / (gamma - 1.0)
        z = np.zeros((n, n))
        sim.set_state("gas", np.stack([np.ones((n, n)), z, z, E * np.ones((n, n))]))
        sim.set_electron_temperature_from("gas")  # comp aux 4 = T = p/rho du bloc 'gas'
        sim.solve_fields()                          # apply_te -> T_e = c sur la composante 4

        # eval_rhs = -div F + S ; flux nul -> R = source = T_e n = c.
        R = np.array(sim.eval_rhs("te"))
        err = float(np.max(np.abs(R - c)))
        print("  JIT T_e : eval_rhs, max|R - T_e| = %.2e" % err)
        assert err < 1e-12, "le bloc dynamique ne lit pas T_e via le marshaling JIT (ecart %.2e)" % err

        # controle : gas E=0 -> p=0 -> T_e=0 -> residu nul.
        sim.set_state("gas", np.stack([np.ones((n, n)), z, z, z]))
        sim.solve_fields()
        R0 = np.array(sim.eval_rhs("te"))
        a0 = float(np.max(np.abs(R0)))
        print("  controle T_e=0 : max|R| = %.2e" % a0)
        assert a0 < 1e-12, "residu non nul a T_e=0 (%.2e)" % a0

        print("OK  modele DSL T_e (derive p/rho) pilote via le chemin JIT (add_dynamic_block)")
        print("test_dsl_jit_te : tout est vert")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
