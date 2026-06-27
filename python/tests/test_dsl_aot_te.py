"""Chantier Aux extensible : T_e (temperature electronique) lue via le chemin AOT COMPILE.

Pendant de test_dsl_aot_bz pour le 2e champ aux extra T_e, DERIVE de p/rho d'un bloc fluide. Un
modele DSL lisant aux('T_e') -> brique GenSrc n_aux=5 -> CompositeModel::n_aux=5 -> .so AOT
(compile_aot) charge par add_compiled_block : le .so expose pops_compiled_naux()=5, le System elargit
le canal aux partage (ensure_aux_width(5)) et marshale 5 composantes vers l'ABI extern "C" (load_aux<5>
lit a.T_e). Un bloc fluide COMPRESSIBLE fournit T = p/rho via set_electron_temperature_from. On
verifie de bout en bout cote Python : eval_rhs du bloc compile = source = T_e * n.
"""
import os
import shutil
import tempfile

import numpy as np

import pops
from pops.physics.model import HyperbolicModel

INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def build_te_scalar():
    """Scalaire sans flux, source S = T_e * n (lit aux('T_e') -> n_aux=5)."""
    m = HyperbolicModel("tescalaraot")
    (nn,) = m.conservative_vars("n")
    zero = 0.0 * nn                      # expression nulle (set_flux n'enrobe pas les floats bruts)
    m.set_flux(x=[zero], y=[zero])       # flux nul -> R = source
    m.set_eigenvalues(x=[zero], y=[zero])
    m.set_primitive_state(nn)
    m.set_conservative_from([nn])
    te = m.aux("T_e")
    m.set_source([te * nn])              # S = T_e n
    return m


def gas_model(gamma):
    """Bloc fluide compressible source de T_e (T = p/rho) ; pas de contribution au Poisson."""
    return pops.Model(state=pops.FluidState("compressible", gamma=gamma),
                     transport=pops.CompressibleFlux(), source=pops.NoSource(),
                     elliptic=pops.ChargeDensity(charge=0.0))


def main():
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes pops absents -> AOT T_e saute")
        print("test_dsl_aot_te : OK (rien a compiler)")
        return

    m = build_te_scalar()
    n, L, gamma = 32, 1.0, 1.4
    rho, p = 1.0, 3.0
    Te = p / rho  # T = p / rho = 3
    tmp = tempfile.mkdtemp()
    try:
        so = m.compile_or_jit(os.path.join(tmp, "tescalar_aot.so"), INCLUDE, mode="compile")

        sim = pops.System(n=n, L=L, periodic=True)
        sim.add_block("gas", model=gas_model(gamma),
                      spatial=pops.Spatial(flux="rusanov"), time=pops.Explicit())
        # add_compiled_block : pops_compiled_naux()=5 -> ensure_aux_width(5)
        sim.add_compiled_block("probe", so, limiter="none", riemann="rusanov",
                               recon="conservative", time="explicit", names=["n"])
        sim.set_poisson(rhs="charge_density", solver="geometric_mg")

        # etat du gaz : rho=1, qte de mvt nulle, E = p/(gamma-1) -> p=3, donc T = p/rho = 3.
        Ug = np.zeros((4, n, n))
        Ug[0] = rho
        Ug[3] = p / (gamma - 1.0)
        sim.set_state("gas", Ug.reshape(-1).tolist())
        sim.set_density("probe", np.ones((n, n)))
        sim.set_electron_temperature_from("gas")  # T_e <- p/rho du gaz, marshale au bloc compile
        sim.solve_fields()

        # eval_rhs = -div F + S ; flux nul -> R = source = T_e n = Te.
        R = np.array(sim.eval_rhs("probe"))
        err = float(np.max(np.abs(R - Te)))
        print("  AOT T_e : eval_rhs, max|R - T_e| = %.2e (T_e=%.3g)" % (err, Te))
        assert err < 1e-12, "le bloc compile ne lit pas T_e (ecart %.2e)" % err

        print("OK  modele DSL T_e (derive p/rho) lu via AOT (compile_aot + add_compiled_block)")
        print("test_dsl_aot_te : tout est vert")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
