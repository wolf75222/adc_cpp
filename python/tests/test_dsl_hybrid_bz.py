"""Aux extensible (B_z) dans un composite HYBRIDE : une brique de SOURCE DSL lit aux('B_z') (n_aux=4)
tandis que le TRANSPORT et l'ELLIPTIQUE sont NATIFS. On verifie que la largeur du canal aux se PROPAGE
a travers le composite melange (CompositeModel::n_aux=4 -> le .so expose naux=4 -> ensure_aux_width(4))
et que la source hybride lit bien B_z.

Montage : transport natif ExB (scalaire, derive E x B) + source DSL S = B_z * n + elliptique native
ChargeDensity. Le terme B_z n'entre QUE par la source DSL ; le flux de derive ne depend que de grad phi
(inchange entre les deux mesures). Donc eval_rhs(B_z=c) - eval_rhs(B_z=0) = c * n, ce qui isole et
prouve la lecture de B_z dans le composite hybride. On verifie aussi que le transport natif contribue
(residu non trivial a B_z=0). Lance avec python3.
"""
import os
import shutil
import tempfile

import numpy as np

import pops
from pops import dsl

INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def build_bz_source():
    """Brique de source DSL S = B_z * n (scalaire), lit aux('B_z') -> n_aux=4."""
    s = dsl.SourceBrick("bz")
    (nn,) = s.conservative_vars("n")
    bz = s.aux("B_z")
    s.source([bz * nn])
    return s


def main():
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes adc absents -> hybride B_z saute")
        print("test_dsl_hybrid_bz : OK (rien a compiler)")
        return

    m = pops.CompositeModel(transport=pops.ExB(B0=1.0),
                           source=build_bz_source().compile(),
                           elliptic=pops.ChargeDensity(charge=1.0))
    assert m.n_vars == 1, "transport scalaire ExB attendu (1 variable)"
    assert m.n_aux == 4, "largeur aux B_z non propagee dans le composite (n_aux=%d)" % m.n_aux

    n, L, c = 32, 1.0, 0.7
    tmp = tempfile.mkdtemp()
    try:
        co = m.compile(backend="aot", so_path=os.path.join(tmp, "hybrid_bz.so"), include=INCLUDE)
        assert co.n_aux == 4

        xs = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(xs, xs)
        dens = 1.0 + 0.2 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.03)  # non uniforme -> derive

        sim = pops.System(n=n, L=L, periodic=True)
        sim.add_equation("bz", co, spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                         names=["n"])
        sim.set_poisson(rhs="charge_density", solver="geometric_mg")
        sim.set_density("bz", dens)

        sim.solve_fields()  # resout phi UNE fois (grad phi fige) ; B_z ne change pas la derive
        sim.set_magnetic_field(c * np.ones((n, n)))
        Rc = np.array(sim.eval_rhs("bz")).reshape(n, n)

        sim.set_magnetic_field(np.zeros((n, n)))  # meme grad phi -> seul le terme de source change
        R0 = np.array(sim.eval_rhs("bz")).reshape(n, n)

        # delta = source(B_z=c) - source(B_z=0) = c * n (le flux de derive est identique, grad phi fige).
        err = float(np.max(np.abs((Rc - R0) - c * dens)))
        assert err < 1e-12, "la source hybride ne lit pas B_z (ecart %.2e)" % err
        assert float(np.max(np.abs(R0))) > 1e-3, "le transport natif ExB ne contribue pas (setup)"
        print("OK  composite hybride : source DSL lit B_z (n_aux=4 propage), transport natif actif "
              "(max|R(c)-R(0)-c n| = %.1e)" % err)

        # T_e (2e champ aux extra, index 4) se propage par le MEME mecanisme que B_z : une source DSL
        # lisant aux('T_e') porte le composite a n_aux=5. Le marshaling hote/device est partage avec
        # B_z (deja prouve de bout en bout ci-dessus) ; on verifie ici la propagation au composite.
        s_te = dsl.SourceBrick("te")
        (nn_te,) = s_te.conservative_vars("n")
        s_te.source([s_te.aux("T_e") * nn_te])
        m_te = pops.CompositeModel(transport=pops.ExB(B0=1.0), source=s_te.compile(),
                                  elliptic=pops.ChargeDensity(charge=1.0))
        assert m_te.n_aux == 5, "largeur aux T_e non propagee dans le composite (n_aux=%d)" % m_te.n_aux
        print("OK  T_e (aux index 4) propage aussi : composite n_aux=5 (meme marshaling que B_z)")
        print("test_dsl_hybrid_bz : tout est vert")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
