"""Couplage inter-especes PAR ROLE sur un bloc HYBRIDE. Un couplage de collision (friction
k (u_a - u_b)) resout les composantes de quantite de mouvement par VariableRole (index_of(MomentumX/Y)),
pas par indice litteral. On verifie qu'un bloc HYBRIDE (transport DSL isotherme + NoSource natif +
ChargeDensity native, backend production) porte des roles corrects dans ses metadonnees .so et participe
donc au couplage EXACTEMENT comme un bloc 100% natif.

Montage : deux especes isothermes (electrons natif, ions HYBRIDE) avec des vitesses initiales
differentes + une collision electrons<->ions. On compare a l'oracle ou les ions sont 100% natifs
(pops.Model) : densite ET quantite de mouvement des deux blocs bit-identiques (< 1e-12) => les roles du
bloc hybride resolvent la friction au bon endroit. On verifie aussi que la collision agit reellement
(la qte de mvt des ions differe du cas sans collision) et que la qte de mvt totale est conservee.
Lance avec python3.
"""
import os
import shutil
import tempfile

import numpy as np

import pops
from pops import dsl
from test_dsl_hybrid import build_iso_transport, CS2

RATE = 3.0
INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def _states(n):
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    ne = 1.0 + 0.05 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.03)
    ni = 1.0 + 0.03 * np.cos(2 * np.pi * X)
    Ue = np.zeros((3, n, n)); Ue[0] = ne; Ue[1] = 0.20 * ne; Ue[2] = 0.00
    Ui = np.zeros((3, n, n)); Ui[0] = ni; Ui[1] = -0.10 * ni; Ui[2] = 0.05 * ni
    return Ue, Ui


def main():
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes adc absents")
        print("test_dsl_hybrid_coupling : OK (rien a compiler)")
        return

    n, L = 40, 1.0
    Ue, Ui = _states(n)
    Ueflat, Uiflat = Ue.reshape(-1).tolist(), Ui.reshape(-1).tolist()
    spatial = pops.FiniteVolume(limiter="minmod", riemann="rusanov", variables="conservative")

    spec_e = pops.Model(state=pops.FluidState("isothermal", cs2=CS2), transport=pops.IsothermalFlux(),
                       source=pops.NoSource(), elliptic=pops.ChargeDensity(charge=-1.0))
    spec_i = pops.Model(state=pops.FluidState("isothermal", cs2=CS2), transport=pops.IsothermalFlux(),
                       source=pops.NoSource(), elliptic=pops.ChargeDensity(charge=1.0))

    tmp = tempfile.mkdtemp()
    try:
        # Bloc ions HYBRIDE (production) : transport DSL isotherme + NoSource natif + ChargeDensity native.
        co_i = pops.CompositeModel(transport=build_iso_transport(CS2).compile(),
                                  source=pops.NoSource(),
                                  elliptic=pops.ChargeDensity(charge=1.0)).compile(
            backend="production", so_path=os.path.join(tmp, "ions_hybrid.so"), include=INCLUDE)

        def run(add_ions, collision=True):
            s = pops.System(n=n, L=L, periodic=True)
            s.add_block("electrons", spec_e, spatial=spatial, time=pops.Explicit())
            add_ions(s)
            s.set_poisson(rhs="charge_density", solver="geometric_mg")
            if collision:
                s.add_collision("electrons", "ions", RATE)
            s.set_state("electrons", Ueflat)
            s.set_state("ions", Uiflat)
            for _ in range(8):
                s.step_cfl(0.3)
            e = np.array(s.get_state("electrons")).reshape(3, n, n)
            i = np.array(s.get_state("ions")).reshape(3, n, n)
            return e, i

        hyb = lambda s: s.add_equation("ions", co_i, spatial=spatial)        # production : sans names=
        nat = lambda s: s.add_block("ions", spec_i, spatial=spatial, time=pops.Explicit())

        He, Hi = run(hyb, collision=True)
        Ne, Ni = run(nat, collision=True)
        de = float(np.max(np.abs(He - Ne)))
        di = float(np.max(np.abs(Hi - Ni)))
        assert de < 1e-12, "electrons : couplage hybride != natif (%.2e)" % de
        assert di < 1e-12, "ions : couplage hybride != natif (%.2e)" % di
        print("OK  collision par role sur bloc hybride == bloc natif (electrons %.1e, ions %.1e)"
              % (de, di))

        # La collision AGIT vraiment : la qte de mvt des ions differe du cas sans collision.
        _He0, Hi0 = run(hyb, collision=False)
        dmom = float(np.max(np.abs(Hi[1] - Hi0[1])))
        assert dmom > 1e-4, "la collision n'a pas modifie la qte de mvt des ions (%.2e)" % dmom
        # Quantite de mouvement TOTALE (electrons + ions) conservee par la friction interne.
        p0 = float(Ue[1].sum() + Ui[1].sum())
        p1 = float(He[1].sum() + Hi[1].sum())
        assert abs(p1 - p0) < 1e-9 * (abs(p0) + 1.0), "qte de mvt totale non conservee (%.2e)" % (p1 - p0)
        print("OK  collision active (dmom ions %.1e) + qte de mvt totale conservee" % dmom)
        print("test_dsl_hybrid_coupling : tout est vert")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
