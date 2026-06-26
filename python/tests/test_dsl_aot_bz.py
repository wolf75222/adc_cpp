"""Chantier Aux extensible : B_z pilotable 100% depuis Python via le chemin AOT COMPILE.

Pendant de test_dsl_jit_bz (chemin JIT add_dynamic_block) pour le chemin de PRODUCTION : un
modele DSL qui lit B_z (aux('B_z')) -> brique GenSrc avec n_aux=4 -> CompositeModel::n_aux=4 ->
.so AOT (compile_aot / compile_or_jit(mode='compile')) charge par add_compiled_block. Le .so
expose pops_compiled_naux()=4 ; le System elargit le canal aux partage (ensure_aux_width) et
marshale 4 composantes vers l'ABI extern "C", qui alloue son aux interne a 4 comp et la peuple
(load_aux<4> lit a.B_z dans assemble_rhs). set_magnetic_field peuple B_z.
On verifie de bout en bout cote Python : eval_rhs du bloc compile = source = B_z * n.
"""
import os
import shutil
import tempfile

import numpy as np

import pops
from pops import dsl

INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def build_bz_scalar():
    """Scalaire transporte sans flux, source magnetisee S = B_z * n (lit aux('B_z'))."""
    m = dsl.HyperbolicModel("bzscalaraot")
    (nn,) = m.conservative_vars("n")
    zero = 0.0 * nn                      # expression nulle (set_flux n'enrobe pas les floats bruts)
    m.set_flux(x=[zero], y=[zero])       # flux nul -> R = source
    m.set_eigenvalues(x=[zero], y=[zero])  # pas de transport
    m.set_primitive_state(nn)
    m.set_conservative_from([nn])
    bz = m.aux("B_z")
    m.set_source([bz * nn])              # S = B_z n
    return m


def main():
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes pops absents -> AOT B_z saute")
        print("test_dsl_aot_bz : OK (rien a compiler)")
        return

    m = build_bz_scalar()
    n, L, c = 32, 1.0, 0.7
    tmp = tempfile.mkdtemp()
    try:
        so = m.compile_or_jit(os.path.join(tmp, "bzscalar_aot.so"), INCLUDE, mode="compile")

        sim = pops.System(n=n, L=L, periodic=True)
        # add_compiled_block : pops_compiled_naux()=4 -> ensure_aux_width(4)
        sim.add_compiled_block("bz", so, limiter="none", riemann="rusanov",
                               recon="conservative", time="explicit", names=["n"])
        sim.set_poisson(rhs="charge_density", solver="geometric_mg")
        sim.set_density("bz", np.ones((n, n)))
        sim.set_magnetic_field(c * np.ones((n, n)))  # peuple le canal B_z partage
        sim.solve_fields()

        # eval_rhs = -div F + S ; flux nul -> R = source = B_z n = c
        R = np.array(sim.eval_rhs("bz"))
        err = float(np.max(np.abs(R - c)))
        print("  AOT B_z : eval_rhs, max|R - B_z| = %.2e" % err)
        assert err < 1e-12, "le bloc compile ne lit pas B_z (ecart %.2e)" % err

        # controle : B_z = 0 -> residu nul (la source ne contribue plus).
        sim.set_magnetic_field(np.zeros((n, n)))
        sim.solve_fields()
        R0 = np.array(sim.eval_rhs("bz"))
        a0 = float(np.max(np.abs(R0)))
        print("  controle B_z=0 : max|R| = %.2e" % a0)
        assert a0 < 1e-12, "residu non nul a B_z=0 (%.2e)" % a0

        print("OK  modele DSL B_z pilote 100%% depuis Python (compile_aot + add_compiled_block)")
        print("test_dsl_aot_bz : tout est vert")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
