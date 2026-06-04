"""block_names() voit TOUS les chemins d'ajout, pas seulement add_block.

Un integrateur ecrit en Python (adc.integrate.euler_step / ssprk2_step) itere sur
sim.block_names() ; un bloc charge a l'execution depuis un .so (add_dynamic_block, JIT ;
add_compiled_block, AOT) doit donc y figurer, sinon l'integrateur le SAUTE silencieusement.

On compose un bloc natif (add_block) puis on charge un bloc dynamique (formules Python ->
.so JIT -> add_dynamic_block) et on verifie que block_names() retourne les DEUX, dans l'ordre
d'ajout, et coherent avec n_species(). On verifie aussi que l'integrateur Python avance bien le
bloc dynamique (etat modifie). Gate sur compilateur + en-tetes adc : SKIP propre (exit 0) sinon,
mais on teste TOUJOURS la partie sans .so (un add_block seul est deja visible).
"""
import os
import shutil
import tempfile

import numpy as np

import adc
from test_dsl_brick import build_euler_brick, GAMMA

INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def euler_native():
    return adc.Model(state=adc.FluidState("compressible", gamma=GAMMA),
                     transport=adc.CompressibleFlux(), source=adc.NoSource(),
                     elliptic=adc.ChargeDensity(charge=1.0))


def main():
    # (0) sans .so : un add_block est deja vu par block_names (garde de base, sans compilateur).
    s0 = adc.System(n=16, L=1.0, periodic=True)
    s0.add_block("gas", model=euler_native(), spatial=adc.Spatial(minmod=True))
    assert list(s0.block_names()) == ["gas"], "add_block invisible dans block_names()"
    assert s0.n_species() == 1, "n_species != 1 pour un bloc"
    print("OK  block_names() voit un add_block natif")

    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes adc absents -> bloc dynamique saute")
        print("test_block_names_dynamic : OK (add_block seul)")
        return

    e = build_euler_brick()  # Euler en formules -> .so JIT chargeable
    n, L = 32, 1.0
    tmp = tempfile.mkdtemp()
    try:
        so = e.compile_so(os.path.join(tmp, "euler_model.so"), INCLUDE)

        sim = adc.System(n=n, L=L, periodic=True)
        sim.add_block("native", model=euler_native(), spatial=adc.Spatial(minmod=True))
        sim.add_dynamic_block("dyn", so, names=["rho", "rho_u", "rho_v", "E"])

        # (1) le bloc dynamique (.so) figure dans block_names, dans l'ordre d'ajout
        names = list(sim.block_names())
        assert names == ["native", "dyn"], \
            "block_names() ne voit pas le bloc dynamique : %s" % names
        assert sim.n_species() == 2, "n_species != 2 (natif + dynamique)"
        print("OK  block_names() == %s (add_block + add_dynamic_block)" % names)

        # (2) un integrateur Python qui itere sur block_names() avance BIEN le bloc dynamique
        xs = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(xs, xs)
        p0 = 1.0 + 0.4 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01)
        for nm in ("native", "dyn"):
            U = np.zeros((4, n, n))
            U[0] = 1.0
            U[3] = p0 / (GAMMA - 1.0)
            sim.set_state(nm, U.reshape(-1).tolist())
        before = np.array(sim.get_state("dyn")).reshape(4, n, n).copy()
        for _ in range(5):
            adc.integrate.euler_step(sim, 0.0005)  # names=None -> itere sur sim.block_names()
        after = np.array(sim.get_state("dyn")).reshape(4, n, n)
        assert np.isfinite(after).all(), "bloc dynamique : etat non fini apres integration Python"
        assert float(np.abs(after - before).max()) > 1e-9, \
            "bloc dynamique SAUTE par l'integrateur Python (etat inchange)"
        print("OK  l'integrateur Python (block_names()) avance bien le bloc dynamique")
        print("test_block_names_dynamic : tout est vert")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
