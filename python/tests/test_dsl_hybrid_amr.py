"""Composite HYBRIDE sur AMR (loader natif AMR hybride) : un modele melant transport NATIF (Euler) +
source DSL (gravite) + elliptique NATIVE (GravityCoupling) est compile en loader natif AMR via
pops.CompositeModel(...).compile(backend="production", target="amr_system"), puis branche dans une
AmrSystem (symbole pops_install_native_amr -> add_compiled_model(AmrSystem&), MEME hierarchie que
AmrSystem.add_block : reflux conservatif, regrid).

On verifie la PARITE FORTE contre l'oracle 100% natif (pops.Model -> add_block) : memes masse, densite
grossiere et nombre de patchs a la precision machine (< 1e-12 ; le solve MG elliptique accumule un bruit
FP d'ordre 1e-16). On verifie aussi les garde-fous (aot+amr_system refuse ; CompiledModel target=system
refuse sur AMR). Auto-saute sans compilateur. Lance avec python3.
"""
import os
import shutil
import tempfile

import numpy as np

import pops
from pops.physics.bricks import SourceBrick

GAMMA = 1.4
INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def build_gravity_source():
    """Source DSL repliquant pops::GravityForce sur 4 variables : rho g, g = -grad phi (travail inclus)."""
    s = SourceBrick("grav")
    rho, rho_u, rho_v, E = s.conservative_vars("rho", "rho_u", "rho_v", "E")
    gx = s.aux("grad_x")
    gy = s.aux("grad_y")
    s.source([0.0, -rho * gx, -rho * gy, -(rho_u * gx + rho_v * gy)])
    return s


def _bubble(n):
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    return (1.0 + 0.5 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)).reshape(-1)


def _amr(n, L, branch):
    cfg = pops.AmrSystemConfig()
    cfg.n = n
    cfg.L = L
    cfg.periodic = True
    cfg.regrid_every = 4
    s = pops.AmrSystem(cfg)
    branch(s)
    s.set_refinement(1.2)
    s.set_density("gas", _bubble(n))
    return s


def main():
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes pops absents")
        print("test_dsl_hybrid_amr : OK (rien a compiler)")
        return

    n, L, dt = 48, 1.0, 2e-4
    tmp = tempfile.mkdtemp()
    try:
        # Composite hybride : transport NATIF Euler + source DSL gravite + elliptique NATIVE.
        m = pops.CompositeModel(transport=pops.CompressibleFlux(),
                               source=build_gravity_source().compile(),
                               elliptic=pops.GravityCoupling(sign=-1.0, four_pi_G=1.0, rho0=1.0))
        assert m.n_vars == 4
        co = m.compile(backend="production", target="amr_system",
                       so_path=os.path.join(tmp, "hybrid_amr.so"), include=INCLUDE)
        assert co.adder == "add_native_block" and co.target == "amr_system"
        assert co.caps.get("amr") is True

        # Oracle 100% natif equivalent (Euler + GravityForce + GravityCoupling).
        spec = pops.Model(state=pops.FluidState("compressible", gamma=GAMMA),
                         transport=pops.CompressibleFlux(), source=pops.GravityForce(),
                         elliptic=pops.GravityCoupling(sign=-1.0, four_pi_G=1.0, rho0=1.0))

        def poisson(s):
            s.set_poisson("charge_density", "geometric_mg")

        spatial = pops.FiniteVolume(limiter="minmod", riemann="rusanov", variables="conservative")
        H = _amr(n, L, lambda s: (s.add_equation("gas", co, spatial=spatial), poisson(s)))
        N = _amr(n, L, lambda s: (s.add_block("gas", spec, spatial=spatial, time=pops.Explicit()),
                                  poisson(s)))
        assert H.n_patches() == N.n_patches(), "n_patches initial hybride != add_block"
        m0h, m0n = H.mass(), N.mass()
        assert abs(m0h - m0n) < 1e-12 * (abs(m0n) + 1.0), "masse initiale hybride != add_block"
        for _ in range(12):
            H.step(dt)
            N.step(dt)
        dh, dn = np.array(H.density()), np.array(N.density())
        assert float(np.max(np.abs(dn))) > 1e-6, "densite natif triviale"
        dmax = float(np.max(np.abs(dh - dn)))
        assert dmax < 1e-12, "AMR hybride : densite production != add_block (%.2e)" % dmax
        assert abs(H.mass() - N.mass()) < 1e-12 * (abs(N.mass()) + 1.0), "masse finale != add_block"
        assert H.n_patches() == N.n_patches(), "n_patches final != add_block (regrid different)"
        print("OK  composite hybride sur AMR : masse/densite/patchs == add_block (dmax=%.1e)" % dmax)

        # Garde-fou : aot + amr_system refuse a la compilation.
        try:
            m.compile(backend="aot", target="amr_system", so_path=os.path.join(tmp, "bad.so"),
                      include=INCLUDE)
            raise AssertionError("compile(aot, amr_system) accepte a tort")
        except ValueError as ex:
            assert "amr_system" in str(ex)

        # Garde-fou : un CompiledModel target='system' refuse sur AmrSystem.add_equation.
        co_sys = m.compile(backend="production", target="system",
                           so_path=os.path.join(tmp, "hybrid_sys.so"), include=INCLUDE)
        try:
            pops.AmrSystem(n=n, L=L, periodic=True).add_equation("gas", co_sys, spatial=spatial)
            raise AssertionError("AmrSystem.add_equation a accepte un CompiledModel target='system'")
        except ValueError as ex:
            assert "target='system'" in str(ex) or "amr_system" in str(ex)
        print("OK  garde-fous AMR (aot+amr_system refuse, target=system refuse sur AMR)")
        print("test_dsl_hybrid_amr : tout est vert")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
