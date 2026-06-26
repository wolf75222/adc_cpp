"""Prototype : composition HYBRIDE brique native + brique DSL DANS UN SEUL modele (Phase B).

Jusqu'ici on melangeait natif et DSL au niveau du SYSTEME (un bloc add_block + un bloc add_equation).
Ce test exerce le MELANGE dans UN modele, via pops.CompositeModel(transport=, source=, elliptic=), ou
chaque slot est soit une brique native (pops.*) soit une brique DSL partielle compilee
(pops.dsl.HyperbolicBrick/SourceBrick/EllipticBrick). Le composite est compile en UN .so (backend aot)
et branche via System.add_equation (add_compiled_block).

On verifie les DEUX SENS contre un oracle 100% natif (pops.Model -> add_block), meme physique
(isotherme cs2=1, force du potentiel qom, densite de charge q) :
  - sens 1 : transport DSL + source native + elliptique native ;
  - sens 2 : transport natif + source DSL + elliptique native.
Pour le meme etat et le meme schema (minmod + rusanov + conservatif), eval_rhs ET le potentiel du
bloc hybride doivent egaler ceux du bloc natif a la precision machine (< 1e-9). On verifie aussi que
le bloc hybride AVANCE dans le System (masse conservee en periodique). Lance avec python3.
"""
import os
import shutil
import tempfile

import numpy as np

import pops
from pops import dsl

CS2 = 1.0     # vitesse du son au carre (isotherme)
QOM = -1.0    # q/m de la force du potentiel (non trivial : exerce le cuisson du parametre natif)
Q = -1.0      # charge de la densite de charge (idem)
INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def build_iso_transport(cs2):
    """Brique hyperbolique DSL repliquant pops::IsothermalFlux{cs2} (3 variables)."""
    b = dsl.HyperbolicBrick("iso")
    rho, rho_u, rho_v = b.conservative_vars("rho", "rho_u", "rho_v")
    u = b.primitive("u", rho_u / rho)
    v = b.primitive("v", rho_v / rho)
    c = dsl.sqrt(cs2)
    b.flux(x=[rho_u, rho_u * u + cs2 * rho, rho_v * u],
           y=[rho_v, rho_u * v, rho_v * v + cs2 * rho])
    b.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    b.primitive_vars(rho, u, v)
    b.conservative_from([rho, rho * u, rho * v])
    return b


def build_force_source(qom):
    """Brique de source DSL repliquant pops::PotentialForce{qom} sur 3 variables : (q/m) rho E,
    E = -grad phi (pas de terme d'energie a 3 variables)."""
    s = dsl.SourceBrick("force")
    rho, rho_u, rho_v = s.conservative_vars("rho", "rho_u", "rho_v")
    gx = s.aux("grad_x")
    gy = s.aux("grad_y")
    s.source([0, qom * rho * (-gx), qom * rho * (-gy)])
    return s


def init_state(n):
    """Etat isotherme : bosse de densite + quantite de mouvement faible (flux non trivial)."""
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    U = np.zeros((3, n, n))
    U[0] = 1.0 + 0.3 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)
    U[1] = 0.10 * U[0]
    U[2] = -0.05 * U[0]
    return U


def main():
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes adc absents -> rien a compiler")
        print("test_dsl_hybrid : OK (rien a compiler)")
        return

    n, L = 48, 1.0
    U = init_state(n)
    Uflat = U.reshape(-1).tolist()
    names = ["rho", "rho_u", "rho_v"]
    spatial = pops.FiniteVolume(limiter="minmod", riemann="rusanov", variables="conservative")

    # Oracle 100% natif : CompositeModel<IsothermalFlux, PotentialForce, ChargeDensity> via add_block.
    spec = pops.Model(state=pops.FluidState("isothermal", cs2=CS2),
                     transport=pops.IsothermalFlux(),
                     source=pops.PotentialForce(charge=QOM),
                     elliptic=pops.ChargeDensity(charge=Q))

    def fields(setup):
        s = pops.System(n=n, L=L, periodic=True)
        setup(s)
        s.set_poisson(rhs="charge_density", solver="geometric_mg")
        s.set_state("gas", Uflat)
        s.solve_fields()
        R = np.array(s.eval_rhs("gas")).reshape(3, n, n)
        phi = np.array(s.potential()).reshape(n, n)
        return R, phi

    R_nat, phi_nat = fields(
        lambda s: s.add_block("gas", spec, spatial=spatial, time=pops.Explicit()))
    assert float(np.max(np.abs(R_nat))) > 1e-3, "residu natif trivial (setup du test)"

    tmp = tempfile.mkdtemp()
    try:
        # --- sens 1 : transport DSL + source native + elliptique native ---
        m1 = pops.CompositeModel(transport=build_iso_transport(CS2).compile(),
                                source=pops.PotentialForce(charge=QOM),
                                elliptic=pops.ChargeDensity(charge=Q))
        assert m1.n_vars == 3 and m1.n_aux == 3
        co1 = m1.compile(backend="aot", so_path=os.path.join(tmp, "h1.so"), include=INCLUDE)
        assert co1.adder == "add_compiled_block"
        R1, phi1 = fields(
            lambda s: s.add_equation("gas", co1, spatial=spatial, names=names))
        dphi1 = float(np.max(np.abs(phi1 - phi_nat)))
        dres1 = float(np.max(np.abs(R1 - R_nat)))
        assert dphi1 < 1e-9, "sens 1 : potentiel hybride != natif (%.2e)" % dphi1
        assert dres1 < 1e-9, "sens 1 : eval_rhs hybride != natif (%.2e)" % dres1
        print("OK  sens 1 (transport DSL + source/elliptic natifs) == natif (ecart %.1e)"
              % max(dphi1, dres1))

        # --- sens 2 : transport natif + source DSL + elliptique native ---
        m2 = pops.CompositeModel(transport=pops.IsothermalFlux(),
                                source=build_force_source(QOM).compile(),
                                elliptic=pops.ChargeDensity(charge=Q))
        assert m2.n_vars == 3
        co2 = m2.compile(backend="aot", so_path=os.path.join(tmp, "h2.so"), include=INCLUDE)
        R2, phi2 = fields(
            lambda s: s.add_equation("gas", co2, spatial=spatial, names=names))
        dphi2 = float(np.max(np.abs(phi2 - phi_nat)))
        dres2 = float(np.max(np.abs(R2 - R_nat)))
        assert dphi2 < 1e-9, "sens 2 : potentiel hybride != natif (%.2e)" % dphi2
        assert dres2 < 1e-9, "sens 2 : eval_rhs hybride != natif (%.2e)" % dres2
        print("OK  sens 2 (transport natif + source DSL + elliptic natif) == natif (ecart %.1e)"
              % max(dphi2, dres2))

        # --- (C) le bloc hybride AVANCE dans le System (masse conservee en periodique) ---
        adv = pops.System(n=n, L=L, periodic=True)
        adv.add_equation("gas", co1, spatial=spatial, names=names)
        adv.set_poisson(rhs="charge_density", solver="geometric_mg")
        adv.set_state("gas", Uflat)
        mass0 = float(np.array(adv.get_state("gas")).reshape(3, n, n)[0].sum())
        for _ in range(15):
            adv.step_cfl(0.4)
        U1 = np.array(adv.get_state("gas")).reshape(3, n, n)
        drel = abs(float(U1[0].sum()) - mass0) / mass0
        assert np.isfinite(U1).all() and U1[0].min() > 0, "etat non physique apres avance"
        assert drel < 1e-9, "masse non conservee (drel=%.2e)" % drel
        print("OK  bloc hybride avance dans le System (15 pas, masse drel=%.1e)" % drel)

        # --- (D) backend production (natif zero-copie via add_native_block) : MEME parite, sans names=
        # (les noms/roles viennent des metadonnees du .so). Chemin de prod (MPI par construction).
        co1p = m1.compile(backend="production", so_path=os.path.join(tmp, "h1p.so"), include=INCLUDE)
        assert co1p.adder == "add_native_block" and co1p.caps["mpi"] is True
        R1p, phi1p = fields(lambda s: s.add_equation("gas", co1p, spatial=spatial))
        dphi1p = float(np.max(np.abs(phi1p - phi_nat)))
        dres1p = float(np.max(np.abs(R1p - R_nat)))
        assert dphi1p < 1e-9, "production : potentiel hybride != natif (%.2e)" % dphi1p
        assert dres1p < 1e-9, "production : eval_rhs hybride != natif (%.2e)" % dres1p
        print("OK  backend production (natif zero-copie add_native_block) == natif (ecart %.1e)"
              % max(dphi1p, dres1p))

        # --- (E) backend prototype (JIT, dispatch virtuel hote Rusanov ordre 1) : tourne dans le System
        # (schema host order 1 != production, donc smoke physique : tourne, fini, masse conservee).
        co1j = m1.compile(backend="prototype", so_path=os.path.join(tmp, "h1j.so"), include=INCLUDE)
        assert co1j.adder == "add_dynamic_block"
        jit = pops.System(n=n, L=L, periodic=True)
        jit.add_equation("gas", co1j, spatial=pops.FiniteVolume(limiter="minmod", riemann="rusanov"),
                         names=names)
        jit.set_poisson(rhs="charge_density", solver="geometric_mg")
        jit.set_state("gas", Uflat)
        mass0j = float(np.array(jit.get_state("gas")).reshape(3, n, n)[0].sum())
        for _ in range(10):
            jit.step_cfl(0.4)
        Uj = np.array(jit.get_state("gas")).reshape(3, n, n)
        assert np.isfinite(Uj).all() and Uj[0].min() > 0, "JIT hybride non physique"
        assert abs(float(Uj[0].sum()) - mass0j) / mass0j < 1e-9, "JIT hybride : masse non conservee"
        print("OK  backend prototype (JIT virtuel) hybride : tourne dans le System (masse conservee)")

        # --- garde-fous : composition tout-native rejetee, slot mal place rejete ---
        try:
            pops.CompositeModel(transport=pops.IsothermalFlux(), source=pops.NoSource(),
                               elliptic=pops.ChargeDensity())
            raise AssertionError("composition tout-native acceptee a tort")
        except ValueError:
            pass
        try:
            pops.CompositeModel(transport=build_force_source(QOM).compile(),  # source dans le slot transport
                               source=pops.NoSource(), elliptic=pops.ChargeDensity())
            raise AssertionError("brique DSL mal placee acceptee a tort")
        except ValueError:
            pass
        print("OK  garde-fous (tout-native rejete, slot mal place rejete)")
        print("test_dsl_hybrid : tout est vert")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
