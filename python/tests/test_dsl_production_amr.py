"""Backend "production" (NATIF) du DSL cote AMR (Plan Ideal etape 5 / DSL Phase D) : un modele ecrit
en formules est compile en un LOADER .so via compile(backend="production", target="amr_system"), qui
inline le gabarit en-tete pops::add_compiled_model(AmrSystem&, ...), puis branche dans une AmrSystem
via AmrSystem.add_native_block (symbole pops_install_native_amr, distinct du chemin System).

A la difference du chemin System (grille plate mono-niveau), le bloc est l'UNIQUE modele porte sur la
hierarchie AMR (AmrCouplerMP<Model> + reflux conservatif + regrid), MEME chemin que AmrSystem.add_block
(dispatch d'une ModelSpec, qui passe par detail::dispatch_amr_compiled = le pendant natif de
add_compiled_model). On verifie :

  1) PARITE STRICTE (transport pur, elliptic_rhs nul => zero bruit FP elliptique) : la densite
     grossiere apres plusieurs pas est BIT-IDENTIQUE (dmax == 0) entre le bloc "production" (loader
     .so -> add_native_block) et le bloc NATIF add_block (ModelSpec CompressibleFlux). C'est la parite
     attendue : le brique Euler generee a une arithmetique de flux bit-identique a pops::Euler natif, et
     les deux empruntent la MEME machinerie AMR (add_compiled_model(AmrSystem&)).
  2) PARITE FORTE (euler_poisson couple) : memes masse / n_patches / densite a la precision machine
     (< 1e-12), comme le test C++ test_amr_compiled_model.cpp (le solve elliptique MG accumule un bruit
     FP d'ordre 1e-16, donc < 1e-12 et non == 0 quand le couplage est actif).
  3) LIMITES AMR enforcees (AmrSystem est mono-bloc ; time cable a {explicit, imex}, imex = source
     raide implicite) : la facade AmrSystem.add_equation applique son garde-fou pression (hllc/roe sans
     primitive 'p' rejete), AVANT le C++. limiter="weno5"
     (WENO5-Z, 3 ghosts) est en revanche CABLE sur AMR (rusanov) : on prouve sa PARITE STRICTE
     (densite production == add_block, dmax == 0) et qu'il DIFFERE de minmod (reconstruction active).
  4) GARDE-FOUS de compilation : compile(target="amr_system") exige backend="production" ; un
     CompiledModel target="system" est refuse par AmrSystem.add_equation (loader sans pops_install_native_amr).
  5) GARDE-FOU ABI : un loader AMR a cle pops_native_abi_key falsifiee est rejete par add_native_block.

S'auto-saute (exit 0) sans compilateur C++ ou en-tetes pops (comme test_dsl_production).
"""
from pops.numerics.variables import Conservative, Primitive
from pops.numerics.riemann import HLLC, Roe
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.riemann import Rusanov
import os
import shutil
import subprocess
import tempfile

import numpy as np

import pops
from pops.codegen.loader import CompiledModel
from pops.ir.ops import sqrt
from pops.physics.facade import Model

GAMMA = 1.4
INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def _euler_formulas(m):
    """Pose les formules Euler compressible (flux + valeurs propres + conversions + gamma) sur la
    FACADE Model @p m (dont compile(...) rend un CompiledModel). Renvoie (rho, rho_u, rho_v, E)."""
    rho, rhou, rhov, E = m.conservative_vars("rho", "rho_u", "rho_v", "E")
    u = rhou / rho
    v = rhov / rho
    p = (GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v))
    pu, pv, pp = m.primitive("u", u), m.primitive("v", v), m.primitive("p", p)
    H = (E + pp) / rho
    c = sqrt(GAMMA * pp / rho)
    m.flux(x=[rhou, rhou * pu + pp, rhou * pv, rho * H * pu],
           y=[rhov, rhov * pu, rhov * pv + pp, rho * H * pv])
    m.eigenvalues(x=[pu - c, pu, pu + c], y=[pv - c, pv, pv + c])
    m.primitive_vars(rho, pu, pv, pp)
    m.conservative_from([rho, rho * pu, rho * pv, pp / (GAMMA - 1.0) + 0.5 * rho * (pu * pu + pv * pv)])
    m.gamma(GAMMA)
    return rho, rhou, rhov, E, pu, pv


def _build_euler_transport():
    """Euler PUR (transport seul) via la facade Model : pas de source, elliptic_rhs NUL. Le solve
    elliptique donne phi=0 des deux cotes (zero bruit FP), donc la parite transport est BIT-IDENTIQUE."""
    m = Model("euler_transport")
    rho, _rhou, _rhov, _E, _u, _v = _euler_formulas(m)
    m.elliptic_rhs(0.0 * rho)  # f = 0 : aucun couplage Poisson (isole le transport)
    return m


def _build_euler_poisson():
    """Euler compressible + force de gravite + couplage self-consistant f = -(rho - 1) (GravityForce +
    GravityCoupling sign=-1, 4piG=1, rho0=1) : un VRAI bloc couple sur AMR (facade Model)."""
    m = Model("euler_poisson")
    rho, rhou, rhov, _E, _u, _v = _euler_formulas(m)
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    m.source([0.0, -rho * gx, -rho * gy, -(rhou * gx + rhov * gy)])
    m.elliptic_rhs(-1.0 * (rho - 1.0))
    return m


def _bubble(n):
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    return (1.0 + 0.5 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)).reshape(-1)


def _amr(n, L, branch, refine=1.2):
    cfg = pops.AmrSystemConfig()
    cfg.n = n
    cfg.L = L
    cfg.periodic = True
    cfg.regrid_every = 4
    s = pops.AmrSystem(cfg)
    branch(s)
    s.set_refinement(refine)
    s.set_density("gas", _bubble(n))
    return s


def main():
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes pops absents")
        print("test_dsl_production_amr : OK (rien a compiler)")
        return

    n, L = 48, 1.0
    tmp = tempfile.mkdtemp()
    try:
        # --- (1) PARITE STRICTE : transport pur (elliptic_rhs = 0), dmax == 0 ---
        et = _build_euler_transport()
        cm_t = et.compile(os.path.join(tmp, "euler_transport_amr.so"), INCLUDE,
                          backend="production", target="amr_system")
        so_t = cm_t.so_path
        assert isinstance(cm_t, CompiledModel)
        assert cm_t.adder == "add_native_block" and cm_t.target == "amr_system"
        assert cm_t.caps.get("amr") is True, "production caps amr=True (Phase D)"
        spec_t = pops.Model(state=pops.FluidState("compressible", gamma=GAMMA),
                           transport=pops.CompressibleFlux(), source=pops.NoSource(),
                           elliptic=pops.BackgroundDensity(alpha=0.0, n0=0.0))

        A = _amr(n, L, lambda s: s._s.add_native_block(
            "gas", so_t, limiter="minmod", riemann="rusanov", recon="conservative",
            time="explicit", gamma=GAMMA, substeps=1))
        B = _amr(n, L, lambda s: s.add_block(
            "gas", spec_t, spatial=pops.Spatial(minmod=True, flux=Rusanov(), recon=Conservative()),
            time=pops.Explicit()))
        assert A.n_patches() == B.n_patches(), "n_patches initial production != add_block"
        dt = 2e-4
        for _ in range(12):
            A.step(dt)
            B.step(dt)
        da, db = np.array(A.density()), np.array(B.density())
        assert da.size == db.size and da.size > 0
        nrm = float(np.max(np.abs(db)))
        assert nrm > 1e-6, "densite natif triviale"
        dmax = float(np.max(np.abs(da - db)))
        assert dmax == 0.0, ("transport pur AMR : densite production != add_block (dmax %.2e, "
                             "attendu 0)" % dmax)
        assert A.n_patches() == B.n_patches(), "n_patches final production != add_block"
        print("OK  (1) transport pur AMR : densite production BIT-IDENTIQUE a add_block (dmax=0)")

        # --- (2) PARITE FORTE : euler_poisson couple, < 1e-12 (bruit FP du MG elliptique) ---
        ep = _build_euler_poisson()
        cm_p = ep.compile(os.path.join(tmp, "euler_poisson_amr.so"), INCLUDE,
                          backend="production", target="amr_system")
        so_p = cm_p.so_path
        spec_p = pops.Model(state=pops.FluidState("compressible", gamma=GAMMA),
                           transport=pops.CompressibleFlux(), source=pops.GravityForce(),
                           elliptic=pops.GravityCoupling(sign=-1.0, four_pi_G=1.0, rho0=1.0))

        def poisson(s):
            s.set_poisson("charge_density", "geometric_mg")

        C = _amr(n, L, lambda s: (s._s.add_native_block(
            "gas", so_p, limiter="minmod", riemann="rusanov", recon="conservative",
            time="explicit", gamma=GAMMA, substeps=1), poisson(s)))
        D = _amr(n, L, lambda s: (s.add_block(
            "gas", spec_p, spatial=pops.Spatial(minmod=True, flux=Rusanov(), recon=Conservative()),
            time=pops.Explicit()), poisson(s)))
        assert C.n_patches() == D.n_patches()
        m0c, m0d = C.mass(), D.mass()
        assert abs(m0c - m0d) < 1e-12 * (abs(m0d) + 1.0), "masse initiale production != add_block"
        for _ in range(12):
            C.step(dt)
            D.step(dt)
        dc, dd = np.array(C.density()), np.array(D.density())
        dmaxp = float(np.max(np.abs(dc - dd)))
        assert dmaxp < 1e-12, "euler_poisson AMR : densite production != add_block (%.2e)" % dmaxp
        assert abs(C.mass() - D.mass()) < 1e-12 * (abs(D.mass()) + 1.0), "masse finale != add_block"
        assert C.n_patches() == D.n_patches(), "n_patches final != add_block (regrid different)"
        print("OK  (2) euler_poisson AMR couple : masse/densite/patchs == add_block (dmax=%.1e)" % dmaxp)

        # --- (3) PARITE hllc/roe/primitive : la facade add_equation ACCEPTE et donne un resultat
        #     bit-identique a add_block (Gap 1 parite : le moteur AMR supporte ces schemas).
        #     Reutilise cm_t (transport pur, phi=0) : parite STRICTE dmax==0 (zero bruit FP),
        #     meme garantie que le test C++ test_amr_riemann_native. cm_t a une primitive 'p'
        #     (declaree dans _euler_formulas via _build_euler_transport) -> garde-fou pression OK.

        def parity_riemann(riem, recon, label):
            """add_equation(riemann, recon) BIT-IDENTIQUE a add_block (dmax==0)."""
            R = _amr(n, L, lambda s: s.add_equation(
                "gas", cm_t,
                spatial=pops.FiniteVolume(limiter=Minmod(), riemann=riem, variables=recon)))
            S = _amr(n, L, lambda s: s.add_block(
                "gas", spec_t,
                spatial=pops.Spatial(minmod=True, flux=riem, recon=recon),
                time=pops.Explicit()))
            for _ in range(12):
                R.step(dt)
                S.step(dt)
            dr, ds = np.array(R.density()), np.array(S.density())
            dmax = float(np.max(np.abs(dr - ds)))
            assert dmax == 0.0, ("%s: add_equation != add_block (dmax=%.2e)" % (label, dmax))
            assert np.isfinite(dr).all() and float(np.max(np.abs(ds))) > 1e-6
            print("OK  (3) %s : add_equation BIT-IDENTIQUE a add_block (dmax=%.0f)" % (label, dmax))

        parity_riemann(HLLC(), Conservative(), "hllc/conservative")
        parity_riemann(HLLC(), Primitive(),    "hllc/primitive")
        parity_riemann(Roe(),  Conservative(), "roe/conservative")
        parity_riemann(Roe(),  Primitive(),    "roe/primitive")

        # La garde-fou pressure reste active : un modele SANS primitive 'p' doit etre rejete.
        # Modele isotherme 3 variables (rho, rho_u, rho_v) avec primitives (rho, u, v) sans 'p' :
        # il compile, mais add_equation(flux=hllc) doit lever ValueError (pression requise).
        m_iso = Model("isothermal_no_p")
        rho_i, rhou_i, rhov_i = m_iso.conservative_vars("rho", "rho_u", "rho_v")
        cs2 = 0.5
        ui = rhou_i / rho_i
        vi = rhov_i / rho_i
        pui = m_iso.primitive("u", ui)
        pvi = m_iso.primitive("v", vi)
        m_iso.flux(x=[rhou_i, rhou_i * pui + cs2 * rho_i, rhou_i * pvi],
                   y=[rhov_i, rhov_i * pui, rhov_i * pvi + cs2 * rho_i])
        m_iso.eigenvalues(x=[pui - sqrt(cs2), pui, pui + sqrt(cs2)],
                          y=[pvi - sqrt(cs2), pvi, pvi + sqrt(cs2)])
        m_iso.primitive_vars(rho_i, pui, pvi)
        m_iso.conservative_from([rho_i, rho_i * pui, rho_i * pvi])
        m_iso.elliptic_rhs(0.0 * rho_i)
        cm_iso = m_iso.compile(os.path.join(tmp, "isothermal_amr.so"), INCLUDE,
                               backend="production", target="amr_system")
        assert "p" not in cm_iso.prim_names, "modele isotherme ne devrait pas avoir 'p'"
        raised = False
        try:
            s_nop = pops.AmrSystem(n=n, L=L, periodic=True)
            s_nop.add_equation("gas", cm_iso,
                               spatial=pops.Spatial(minmod=True, flux=HLLC()))
        except ValueError as ex:
            raised = True
            assert "hllc" in str(ex).lower()
        assert raised, "add_equation a accepte hllc sans primitive 'p'"
        print("OK  (3) garde-fou pression hllc/roe SANS primitive 'p' : rejet explicite")

        # --- (3w) WENO5 (3 ghosts) CABLE sur AMR : parite stricte production == add_block + != minmod.
        # Le coupleur alloue ses niveaux a Limiter::n_ghost (3) et le regrid herite n_grow() : le
        # stencil 5 points ne lit pas hors bornes (MEME mecanisme ghost que System, pas de duplication).
        Aw = _amr(n, L, lambda s: s._s.add_native_block(
            "gas", so_t, limiter="weno5", riemann="rusanov", recon="conservative",
            time="explicit", gamma=GAMMA, substeps=1))
        Bw = _amr(n, L, lambda s: s.add_block(
            "gas", spec_t, spatial=pops.Spatial(weno5=True, flux=Rusanov(), recon=Conservative()),
            time=pops.Explicit()))
        assert Aw.n_patches() == Bw.n_patches(), "weno5 : n_patches initial production != add_block"
        for _ in range(12):
            Aw.step(dt)
            Bw.step(dt)
        daw, dbw = np.array(Aw.density()), np.array(Bw.density())
        assert float(np.max(np.abs(dbw))) > 1e-6, "weno5 : densite natif triviale"
        dmaxw = float(np.max(np.abs(daw - dbw)))
        assert dmaxw == 0.0, ("weno5 AMR : densite production != add_block (dmax %.2e, attendu 0)"
                              % dmaxw)
        assert Aw.n_patches() == Bw.n_patches(), "weno5 : n_patches final production != add_block"
        # WENO5 (ordre 5) doit differer de minmod (ordre 2) sur le meme transport lisse : la
        # reconstruction WENO5 est bien active (sinon le branchement weno5 retomberait sur minmod).
        assert float(np.max(np.abs(daw - da))) > 1e-9, "weno5 == minmod (reconstruction inactive)"
        # WENO5 via la facade add_equation (pas seulement le binding bas niveau) : meme chemin nominal.
        # Reutilise cm_t (transport pur) : pas de Poisson, tourne sans set_poisson.
        Gw = pops.AmrSystem(n=n, L=L, periodic=True)
        Gw.add_equation("gas", cm_t,
                        spatial=pops.Spatial(weno5=True, flux=Rusanov(), recon=Conservative()))
        Gw.set_refinement(1.2)
        Gw.set_density("gas", _bubble(n))
        for _ in range(4):
            Gw.step(dt)
        assert np.isfinite(np.array(Gw.density())).all() and Gw.mass() > 1e-6
        print("OK  (3w) WENO5 AMR : production BIT-IDENTIQUE a add_block (dmax=%.1e), != minmod, "
              "add_equation(weno5) tourne" % dmaxw)

        # add_equation chemin nominal (rusanov + conservatif) accepte et tourne :
        E = pops.AmrSystem(n=n, L=L, periodic=True)
        E.set_poisson("charge_density", "geometric_mg")
        E.add_equation("gas", cm_t,
                       spatial=pops.Spatial(minmod=True, flux=Rusanov(), recon=Conservative()))
        E.set_refinement(1.2)
        E.set_density("gas", _bubble(n))
        for _ in range(4):
            E.step(dt)
        assert np.isfinite(np.array(E.density())).all() and E.mass() > 1e-6
        print("OK  (3b) AmrSystem.add_equation(production, rusanov) tourne et reste physique")

        # --- (4) GARDE-FOUS de compilation / dispatch ---
        raised = False
        try:
            ep.compile(os.path.join(tmp, "bad.so"), INCLUDE, backend="aot", target="amr_system")
        except ValueError as ex:
            raised = True
            assert "amr_system" in str(ex)
        assert raised, "compile(backend='aot', target='amr_system') aurait du lever"

        sys_cm = ep.compile(os.path.join(tmp, "ep_sys_cm.so"), INCLUDE,
                            backend="production", target="system")  # target System par defaut
        s = pops.AmrSystem(n=n, L=L, periodic=True)
        raised = False
        try:
            s.add_equation("gas", sys_cm,
                           spatial=pops.Spatial(minmod=True, flux=Rusanov(), recon=Conservative()))
        except ValueError as ex:
            raised = True
            assert "target='system'" in str(ex) or "amr_system" in str(ex)
        assert raised, "AmrSystem.add_equation a accepte un CompiledModel target='system'"
        print("OK  (4) compile(target=) garde-fous + CompiledModel target='system' refuse sur AMR")

        # --- (5) GARDE-FOU ABI : loader AMR a cle pops_native_abi_key falsifiee -> rejet ---
        bad_abi = _compile_wrong_abi(ep, os.path.join(tmp, "ep_amr_wrongabi.so"), cxx)
        s = pops.AmrSystem(n=n, L=L, periodic=True)
        raised = False
        try:
            s._s.add_native_block("gas", bad_abi, limiter="minmod", riemann="rusanov",
                                  recon="conservative", time="explicit", gamma=GAMMA)
        except RuntimeError as ex:
            raised = True
            assert "ABI" in str(ex), "message inattendu : %s" % ex
        assert raised, "add_native_block a accepte un loader AMR a cle d'ABI fausse (UB silencieux)"
        print("OK  (5) cle d'ABI divergente REJETEE par AmrSystem.add_native_block")

        print("test_dsl_production_amr : tout est vert")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _compile_wrong_abi(model, dst_so, cxx):
    """Compile le MEME loader natif AMR mais avec une signature d'en-tetes FAUSSE (-DPOPS_HEADER_SIG
    bidon) : le .so est valide mais sa cle d'ABI differe de celle du module -> rejet d'add_native_block.
    On regenere (pas de patch binaire : sur macOS ARM cela invaliderait la signature et tuerait le
    process). Renvoie le chemin du .so."""
    from pops.codegen.toolchain import pops_loader_build_flags
    # model est une facade Model : le HyperbolicModel backing (_m) porte emit_cpp_native_loader.
    src = model._m.emit_cpp_native_loader(target="amr_system")
    # adc_cpp est Kokkos-only : le loader inclut les en-tetes pops -> Kokkos + (macOS) -undefined
    # dynamic_lookup via pops_loader_build_flags. SIGNATURE D'EN-TETES FAUSSE conservee (le .so compile
    # mais doit etre REJETE a l'ABI par add_native_block).
    cc, kflags_c, kflags_l = pops_loader_build_flags(cxx)
    flags = ["-shared", "-fPIC", "-std=c++20", "-O2",
             "-DPOPS_HEADER_SIG=\"deadbeef_signature_volontairement_fausse\"", *kflags_c]
    with tempfile.TemporaryDirectory() as t:
        cpp = os.path.join(t, "wrong_amr.cpp")
        with open(cpp, "w") as f:
            f.write(src)
        subprocess.run([cc, *flags, "-I", INCLUDE, cpp, "-o", dst_so, *kflags_l], check=True)
    return dst_so


if __name__ == "__main__":
    main()
