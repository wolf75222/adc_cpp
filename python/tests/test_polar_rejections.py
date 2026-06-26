"""Lot E - rejets explicites du chemin polaire (Phase 2b).

Verrouille les rejets HONETES du System polaire : chaque test EXIGE que l'appel leve une exception.
Un test echoue si le rejet est supprime (l'appel passe sans lever).

Rejets confirmes (lus dans python/system.cpp et include/pops/runtime/block_builder_polar.hpp) :
  R1 - transport non-ExB (ex. compressible) sur un System polaire :
       dispatch_transport_polar leve RuntimeError :
       "transport polaire '...' non supporte (Phase 2b : seul 'exb' ...)"
  R2 - flux Riemann NON cable en polaire ('hllc' / 'roe') sur un System polaire :
       make_block_polar leve RuntimeError :
       "System (polaire) : flux Riemann '...' non supporte ... HLLC/Roe supposent n_vars==4 ..."
       NB : 'hll' est desormais CABLE en polaire (fluide isotherme, gate model.wave_speeds) -- voir
       test_polar_hll.py ; sur un transport ExB SCALAIRE (pas de wave_speeds) 'hll' leve aussi un
       rejet clair (teste R2c ci-dessous).
  R3 - time=pops.IMEX() sur un System polaire :
       add_block leve RuntimeError :
       "System::add_block (polaire) : time='imex' non supporte ..."
  R4 - set_epsilon_field / set_epsilon_anisotropic_field / set_reaction_field puis step() :
       ensure_elliptic_polar leve RuntimeError :
       "System::set_poisson (polaire) : permittivite variable / anisotrope / reaction non supportee
        par le Poisson polaire direct (Phase 2b ...)"

Ce qui n'est PAS teste ici (deja couvert) :
  - PolarMesh(nr<3) -> voir test_polar_system.py : test_polar_rejects_nr_below_3.

Note sur PolarMesh : config.n = nr (cf. python/pops/__init__.py), donc set_epsilon_field
exige un tableau de taille nr*nr.
"""
import numpy as np
import pytest

import pops


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exb_model():
    """Bloc ExB scalaire standard : le seul transport valide en Phase 2b."""
    return pops.Model(
        state=pops.Scalar(),
        transport=pops.ExB(B0=1.0),
        source=pops.NoSource(),
        elliptic=pops.ChargeDensity(charge=1.0),
    )


def _compressible_model():
    """Bloc fluide compressible : transport NON supporte en polaire (Phase 2b).
    Le second membre elliptique est neutre (fond nul) ; seul le transport est teste ici."""
    return pops.Model(
        state=pops.FluidState(kind="compressible", gamma=1.4),
        transport=pops.CompressibleFlux(),
        source=pops.NoSource(),
        elliptic=pops.BackgroundDensity(alpha=0.0, n0=0.0),
    )


_NR, _NTH = 8, 8
_RMIN, _RMAX = 0.3, 1.0


def _make_polar_sim():
    """System polaire minimal (ExB scalaire, Rusanov, Explicit), sans bloc encore ajoute."""
    return pops.System(mesh=pops.PolarMesh(r_min=_RMIN, r_max=_RMAX, nr=_NR, ntheta=_NTH))


def _make_polar_sim_ready(solver="polar"):
    """System polaire minimal avec bloc ExB et densite initiale : pret pour step()."""
    sim = pops.System(mesh=pops.PolarMesh(r_min=_RMIN, r_max=_RMAX, nr=_NR, ntheta=_NTH))
    sim.add_block("ne", model=_exb_model(),
                  spatial=pops.Spatial(minmod=True), time=pops.Explicit())
    sim.set_poisson(rhs="charge_density", solver=solver, bc="dirichlet")
    sim.set_density("ne", [1.0] * (_NR * _NTH))
    return sim


# ---------------------------------------------------------------------------
# R1 : transport non-ExB (compressible) sur un System polaire
# ---------------------------------------------------------------------------

def test_polar_rejects_non_exb_transport():
    """R1 : add_block avec un transport compressible sur PolarMesh doit lever RuntimeError.

    Confirme (python/system.cpp, dispatch_transport_polar appele depuis add_block chemin polaire) :
      "transport polaire 'compressible' non supporte (Phase 2b : seul 'exb' ...)"
    """
    sim = _make_polar_sim()
    raised = False
    msg = ""
    try:
        sim.add_block("fluid", model=_compressible_model(),
                      spatial=pops.Spatial(minmod=True), time=pops.Explicit())
    except RuntimeError as e:
        raised = True
        msg = str(e)
    assert raised, (
        "add_block avec transport compressible sur PolarMesh aurait du lever RuntimeError"
    )
    # Sous-chaine stable du message (confirme dans block_builder_polar.hpp : dispatch_transport_polar)
    assert "compressible" in msg or "exb" in msg.lower(), (
        "message inattendu : %r" % msg
    )


# ---------------------------------------------------------------------------
# R2 : flux Riemann autre que 'rusanov' sur un System polaire
# ---------------------------------------------------------------------------

def test_polar_rejects_non_rusanov_flux():
    """R2 : add_block avec flux='hllc' sur PolarMesh doit lever RuntimeError.

    Confirme (include/pops/runtime/block_builder_polar.hpp : make_block_polar) :
      "System (polaire) : flux Riemann 'hllc' non supporte ..."
    """
    sim = _make_polar_sim()
    raised = False
    msg = ""
    try:
        sim.add_block("ne", model=_exb_model(),
                      spatial=pops.Spatial(flux="hllc"), time=pops.Explicit())
    except RuntimeError as e:
        raised = True
        msg = str(e)
    assert raised, "add_block avec flux='hllc' sur PolarMesh aurait du lever RuntimeError"
    assert "rusanov" in msg.lower() or "Riemann" in msg or "non supporte" in msg, (
        "message inattendu : %r" % msg
    )


def test_polar_rejects_roe_flux():
    """R2 variante : flux='roe' sur PolarMesh doit lever RuntimeError (meme garde-fou)."""
    sim = _make_polar_sim()
    raised = False
    try:
        sim.add_block("ne", model=_exb_model(),
                      spatial=pops.Spatial(flux="roe"), time=pops.Explicit())
    except RuntimeError:
        raised = True
    assert raised, "add_block avec flux='roe' sur PolarMesh aurait du lever RuntimeError"


def test_polar_rejects_hll_on_scalar_exb():
    """R2c : flux='hll' sur un transport ExB SCALAIRE doit lever RuntimeError.

    'hll' est cable en polaire (cf. test_polar_hll.py) mais GATE sur model.wave_speeds (ondes
    signees), exactement comme le cartesien (block_builder.hpp make_block branche 'hll'). L'ExB
    scalaire (ExBVelocityPolar, 1 var) ne declare PAS wave_speeds -> rejet CLAIR (pas un echec de
    compilation pour un modele scalaire). Le fluide isotherme polaire, lui, accepte 'hll'.
    """
    sim = _make_polar_sim()
    raised = False
    msg = ""
    try:
        sim.add_block("ne", model=_exb_model(),
                      spatial=pops.Spatial(flux="hll"), time=pops.Explicit())
    except RuntimeError as e:
        raised = True
        msg = str(e)
    assert raised, "add_block avec flux='hll' sur un ExB scalaire aurait du lever RuntimeError"
    assert "wave_speeds" in msg or "ondes signees" in msg or "rusanov" in msg.lower(), (
        "message inattendu : %r" % msg
    )


# ---------------------------------------------------------------------------
# R3 : time=pops.IMEX() sur un System polaire
# ---------------------------------------------------------------------------

def test_polar_rejects_imex_time():
    """R3 : add_block avec time=pops.IMEX() sur PolarMesh doit lever RuntimeError.

    Confirme (python/system.cpp, chemin polaire dans add_block) :
      "System::add_block (polaire) : time='imex' non supporte ..."
    """
    sim = _make_polar_sim()
    raised = False
    msg = ""
    try:
        sim.add_block("ne", model=_exb_model(),
                      spatial=pops.Spatial(minmod=True), time=pops.IMEX())
    except RuntimeError as e:
        raised = True
        msg = str(e)
    assert raised, "add_block avec IMEX sur PolarMesh aurait du lever RuntimeError"
    assert "imex" in msg.lower() or "non supporte" in msg or "polaire" in msg.lower(), (
        "message inattendu : %r" % msg
    )


# ---------------------------------------------------------------------------
# R4a : set_epsilon_field puis step() sur un System polaire doit lever
# ---------------------------------------------------------------------------

def test_polar_rejects_variable_epsilon_on_step():
    """R4a : permittivite variable (set_epsilon_field) + step() sur PolarMesh -> RuntimeError.

    La validation est PARESSEUSE (dans ensure_elliptic_polar, appelee par solve_fields_polar
    au premier step). set_epsilon_field reussit ; c'est step() qui leve.

    Confirme (python/system.cpp : ensure_elliptic_polar) :
      "System::set_poisson (polaire) : permittivite variable / anisotrope / reaction non supportee
       par le Poisson polaire direct (Phase 2b ...)"

    Note : config.n = nr pour PolarMesh -> set_epsilon_field attend nr*nr elements.
    Solver 'geometric_mg' est utilise pour que ensure_elliptic_polar soit atteint avec pell_=null
    et has_eps_field_=True ; les deux solveurs ('polar', 'geometric_mg') passent par
    ensure_elliptic_polar en chemin polaire, qui rejette has_eps_field_.
    """
    # Nouveau System (pell_ = null) avec solver geometric_mg : ensure_elliptic_polar rejettera
    sim = _make_polar_sim_ready(solver="geometric_mg")
    # config.n = nr = 8 -> set_epsilon_field attend 8*8 = 64 elements
    sim.set_epsilon_field([1.5] * (_NR * _NR))
    raised = False
    msg = ""
    try:
        sim.step(1e-3)
    except RuntimeError as e:
        raised = True
        msg = str(e)
    assert raised, (
        "step() avec set_epsilon_field sur PolarMesh aurait du lever RuntimeError "
        "(ensure_elliptic_polar rejette la permittivite variable)"
    )
    assert "Phase 2b" in msg, (
        "message inattendu : %r" % msg
    )


# ---------------------------------------------------------------------------
# R4b : set_epsilon_anisotropic_field puis step() sur un System polaire
# ---------------------------------------------------------------------------

def test_polar_rejects_anisotropic_epsilon_on_step():
    """R4b : permittivite anisotrope (set_epsilon_anisotropic_field) + step() -> RuntimeError.

    Confirme (python/system.cpp : ensure_elliptic_polar, meme throw que R4a).
    """
    sim = _make_polar_sim_ready(solver="geometric_mg")
    n2 = _NR * _NR  # config.n = nr
    sim.set_epsilon_anisotropic_field(
        np.array([1.2] * n2, dtype=float),
        np.array([0.8] * n2, dtype=float),
    )
    raised = False
    msg = ""
    try:
        sim.step(1e-3)
    except RuntimeError as e:
        raised = True
        msg = str(e)
    assert raised, (
        "step() avec set_epsilon_anisotropic_field sur PolarMesh aurait du lever RuntimeError"
    )
    assert "Phase 2b" in msg, (
        "message inattendu : %r" % msg
    )


# ---------------------------------------------------------------------------
# R4c : set_reaction_field (kappa) puis step() sur un System polaire
# ---------------------------------------------------------------------------

def test_polar_rejects_reaction_field_on_step():
    """R4c : terme de reaction kappa(x) (Poisson ecrane) + step() sur PolarMesh -> RuntimeError.

    Confirme (python/system.cpp : ensure_elliptic_polar, meme throw que R4a/b).
    """
    sim = _make_polar_sim_ready(solver="geometric_mg")
    n2 = _NR * _NR  # config.n = nr
    sim.set_reaction_field(np.array([0.5] * n2, dtype=float))
    raised = False
    msg = ""
    try:
        sim.step(1e-3)
    except RuntimeError as e:
        raised = True
        msg = str(e)
    assert raised, (
        "step() avec set_reaction_field sur PolarMesh aurait du lever RuntimeError"
    )
    assert "Phase 2b" in msg, (
        "message inattendu : %r" % msg
    )


if __name__ == "__main__":
    test_polar_rejects_non_exb_transport()
    print("OK R1 : transport non-ExB rejete")
    test_polar_rejects_non_rusanov_flux()
    print("OK R2 : flux Riemann non-rusanov (hllc) rejete")
    test_polar_rejects_roe_flux()
    print("OK R2b : flux Riemann non-rusanov (roe) rejete")
    test_polar_rejects_hll_on_scalar_exb()
    print("OK R2c : flux 'hll' sur ExB scalaire rejete (pas de wave_speeds)")
    test_polar_rejects_imex_time()
    print("OK R3 : IMEX sur polaire rejete")
    test_polar_rejects_variable_epsilon_on_step()
    print("OK R4a : permittivite variable rejetee a step()")
    test_polar_rejects_anisotropic_epsilon_on_step()
    print("OK R4b : permittivite anisotrope rejetee a step()")
    test_polar_rejects_reaction_field_on_step()
    print("OK R4c : kappa reaction rejete a step()")
    print("test_polar_rejections : OK")
