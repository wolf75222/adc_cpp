"""ADC-369 : halos/ghosts auxiliaires CONFIGURABLES PAR CHAMP (pops.AuxHalo).

Un champ aux NOMME (m.aux_field) heritait jusqu'ici de la BC de ghost PARTAGEE (derivee de phi). Cette
PR laisse UN champ declarer sa propre politique de halo (foextrap / dirichlet), appliquee APRES le
remplissage partage a SA composante seulement, sur les faces NON PERIODIQUES (les faces periodiques --
domaine periodique, theta polaire -- gardent leur wrap).

On verifie (compilateur requis pour le bout-en-bout, auto-skip sinon) :
  (1) validation pure-Python de pops.AuxHalo ;
  (2) capabilities() annonce la politique de halo ;
  (3) System CARTESIEN non periodique : un flux lisant un aux nomme vx voit le ghost ; la BC dirichlet
      du champ change le residu AU BORD mais PAS a l'interieur, et 'foextrap' explicite == defaut
      (non-regression bit-identique du chemin partage) ;
  (4) polaire : le halo est accepte + applique (solve_fields_polar) sans casser le champ ;
  (5) AMR : le halo est accepte sur AmrSystem (mono + multi bloc) et n'empeche pas le pas.
"""
import os
import shutil
import tempfile

import numpy as np

import pops
from pops import dsl

INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def _have_compiler():
    return bool((shutil.which("c++") or shutil.which("g++") or shutil.which("clang++"))
                and os.path.isdir(INCLUDE))


def build_advect_vx():
    """Scalaire 'n', flux_x = vx*n ou vx est un aux NOMME, flux_y = 0. Le flux LIT vx (au bord, via le
    ghost de vx) -> le halo de vx est observable dans eval_rhs au bord."""
    m = dsl.Model("advectvx")
    (nn,) = m.conservative_vars("n")
    vx = m.aux_field("vx")
    zero = 0.0 * nn
    m.flux(x=[vx * nn], y=[zero])
    m.eigenvalues(x=[vx], y=[zero])
    m.primitive_vars(n=nn)
    m.conservative_from([nn])
    m.source([zero])
    return m


def test_auxhalo_validation():
    """pops.AuxHalo : kinds valides, valeur, rejet d'un kind inconnu. Pur Python."""
    assert pops.AuxHalo("foextrap").bc_type == 1
    d = pops.AuxHalo("dirichlet", value=2.5)
    assert d.bc_type == 2 and d.value == 2.5
    raised = False
    try:
        pops.AuxHalo("periodic")  # pas une politique par champ (gardee par le domaine)
    except ValueError:
        raised = True
    assert raised, "AuxHalo('periodic') devrait lever (foextrap/dirichlet uniquement)"
    print("OK  AuxHalo validation : foextrap/dirichlet, rejet kind inconnu")


def test_capabilities_halo():
    """capabilities()['aux']['named']['halo_policy'] annonce la politique par champ."""
    named = pops.capabilities()["aux"]["named"]
    hp = named["halo_policy"]
    assert set(hp["kinds"]) >= {"inherit", "foextrap", "dirichlet"}, hp["kinds"]
    assert "amr_coarse" in hp["backends"] and "system_polar" in hp["backends"], hp["backends"]
    print("OK  capabilities : halo_policy (foextrap/dirichlet, system+polar+amr)")


def _cart_rhs(compiled, n, vx2d, halo):
    sim = pops.System(n=n, L=1.0, periodic=False)
    sim.add_equation("a", model=compiled,
                     spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                     time=pops.Explicit())
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet")
    sim.set_density("a", np.ones((n, n)))
    sim.set_aux_field("a", "vx", vx2d, halo=halo)
    sim.solve_fields()
    # eval_rhs returns (n_vars, n*n) row-major; reshape the single scalar to (ny, nx) = (j, i).
    return np.array(sim.eval_rhs("a")).reshape(n, n)


def test_system_cartesian_halo():
    """Cartesien non periodique : la BC du champ vx change le residu au bord, pas a l'interieur ;
    'foextrap' explicite == defaut (non-regression du chemin partage)."""
    if not _have_compiler():
        print("skip  cartesian halo (compilateur absent)")
        return
    tmp = tempfile.mkdtemp()
    try:
        compiled = build_advect_vx().compile(os.path.join(tmp, "advx.so"), include=INCLUDE, backend="aot")
        n = 16
        x = (np.arange(n) + 0.5) / n
        vx2d = np.tile(x, (n, 1))  # vx[j, i] = x_i : varie en x -> foextrap != dirichlet au bord x

        R_def = _cart_rhs(compiled, n, vx2d, None)
        R_foe = _cart_rhs(compiled, n, vx2d, pops.AuxHalo("foextrap"))
        R_dir = _cart_rhs(compiled, n, vx2d, pops.AuxHalo("dirichlet", value=0.0))

        # (a) defaut == foextrap explicite : le defaut partage (phi non periodique) EST deja Foextrap.
        assert np.allclose(R_def, R_foe, atol=1e-12), \
            "halo='foextrap' devrait reproduire le defaut (max %.2e)" % float(np.max(np.abs(R_def - R_foe)))
        # (b) dirichlet change le residu AU BORD x (colonnes 0 et n-1).
        db = max(float(np.max(np.abs(R_dir[:, 0] - R_def[:, 0]))),
                 float(np.max(np.abs(R_dir[:, n - 1] - R_def[:, n - 1]))))
        assert db > 1e-6, "halo='dirichlet' devrait changer le residu au bord x (got %.2e)" % db
        # (c) interieur INCHANGE (le halo ne touche que les ghosts du bord).
        assert np.allclose(R_dir[:, 1:n - 1], R_def[:, 1:n - 1], atol=1e-12), \
            "le halo ne doit PAS changer l'interieur (max %.2e)" % float(
                np.max(np.abs(R_dir[:, 1:n - 1] - R_def[:, 1:n - 1])))
        print("OK  cartesian halo : dirichlet change le bord (%.2e), interieur intact, foextrap==defaut" % db)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_polar_halo():
    """Polaire : un flux lisant un aux nomme vx voit le ghost RADIAL ; le halo dirichlet du champ change
    le residu au bord radial, PAS a l'interieur ; theta reste periodique (residu fini). Le halo est
    applique par solve_fields_polar sur le canal aux PARTAGE (lu directement par le bloc polaire)."""
    if not _have_compiler():
        print("skip  polar halo (compilateur absent)")
        return
    tmp = tempfile.mkdtemp()
    try:
        compiled = build_advect_vx().compile(os.path.join(tmp, "avxp.so"), include=INCLUDE, backend="aot")
        nr, nth = 16, 16
        vx = np.tile((np.arange(nr) + 0.5) / nr, (nth, 1))  # varies in r (fast axis i)

        def rhs(halo):
            s = pops.System(mesh=pops.PolarMesh(r_min=0.3, r_max=1.0, nr=nr, ntheta=nth))
            s.add_equation("a", model=compiled,
                           spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"), time=pops.Explicit())
            s.set_density("a", np.ones((nth, nr)))
            s.set_aux_field("a", "vx", vx, halo=halo)
            s.solve_fields()
            return np.array(s.eval_rhs("a")).reshape(nth, nr)  # (theta=j, r=i)

        R_def = rhs(None)
        R_dir = rhs(pops.AuxHalo("dirichlet", value=0.0))
        assert np.all(np.isfinite(R_def)) and np.all(np.isfinite(R_dir)), "polar residual finite (theta periodic)"
        # the RADIAL boundary (r = column 0 and nr-1) changes; theta is periodic -> no theta boundary effect.
        rb = max(float(np.max(np.abs(R_dir[:, 0] - R_def[:, 0]))),
                 float(np.max(np.abs(R_dir[:, nr - 1] - R_def[:, nr - 1]))))
        assert rb > 1e-6, "polar : the radial dirichlet halo should change the radial boundary residual (%.2e)" % rb
        assert np.allclose(R_dir[:, 1:nr - 1], R_def[:, 1:nr - 1], atol=1e-12), \
            "polar : the halo must not change the radial interior (max %.2e)" % float(
                np.max(np.abs(R_dir[:, 1:nr - 1] - R_def[:, 1:nr - 1])))
        print("OK  polar halo : radial dirichlet changes the radial boundary (%.2e), interior intact, theta finite" % rb)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_amr_halo():
    """AMR : un flux lisant un aux nomme vx (compile production target=amr_system) voit le ghost via le
    canal aux PARTAGE du moteur. AmrSystem n'expose pas eval_rhs : on observe l'effet du halo sur la
    densite APRES un pas -- le halo dirichlet change la densite au bord, pas a l'interieur. Le halo est
    applique par AmrRuntime/AmrCouplerMP (apply_named_aux_bc) au niveau grossier."""
    if not _have_compiler():
        print("skip  AMR halo (compilateur absent)")
        return
    tmp = tempfile.mkdtemp()
    try:
        n = 16
        sp = pops.FiniteVolume(limiter="none", riemann="rusanov")
        compiled = build_advect_vx().compile(os.path.join(tmp, "avxa.so"), include=INCLUDE,
                                             backend="production", target="amr_system")
        vx = np.tile((np.arange(n) + 0.5) / n, (n, 1))

        def stepped_density(halo):
            s = pops.AmrSystem(n=n, L=1.0, periodic=False)
            s.add_equation("a", model=compiled, spatial=sp, time=pops.Explicit())
            s.set_poisson(bc="dirichlet")
            s.set_density("a", np.ones((n, n)))
            s.set_aux_field("a", "vx", vx, halo=halo)
            s.step(1e-3)
            return np.array(s.density("a")).reshape(n, n)

        D_def = stepped_density(None)
        D_dir = stepped_density(pops.AuxHalo("dirichlet", value=0.0))
        assert np.all(np.isfinite(D_def)) and np.all(np.isfinite(D_dir)), "AMR density finite with a halo"
        db = max(float(np.max(np.abs(D_dir[:, 0] - D_def[:, 0]))),
                 float(np.max(np.abs(D_dir[:, n - 1] - D_def[:, n - 1]))))
        assert db > 1e-6, "AMR : the dirichlet halo should change the post-step boundary density (%.2e)" % db
        # cells away from the x-boundary are untouched by the boundary ghost in one step.
        assert np.allclose(D_dir[:, 2:n - 2], D_def[:, 2:n - 2], atol=1e-12), \
            "AMR : the halo must not change the interior density (max %.2e)" % float(
                np.max(np.abs(D_dir[:, 2:n - 2] - D_def[:, 2:n - 2])))
        print("OK  AMR halo : dirichlet changes the post-step boundary density (%.2e), interior intact" % db)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_auxhalo_validation()
    test_capabilities_halo()
    test_system_cartesian_halo()
    test_polar_halo()
    test_amr_halo()
    print("test_aux_halo : OK")


if __name__ == "__main__":
    main()
