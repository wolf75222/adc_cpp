"""Lot E - verrouillage du rejet de role inconnu dans CoupledSource / add_coupled_source.

Deux points de rejet sont confirmes (lus dans python/pops/dsl.py et python/system.cpp) :

  (A) CoupledSource.compile() cote PYTHON : _role_canonical() leve ValueError pour un role
      dont le nom n'appartient pas a _ROLE_TO_CANONICAL (p.ex. 'massflux', 'bogus_role').
      Cela se produit au moment ou le champ (block, role) est cree via .block().role(name) OU
      via .add(role=name). La DSL traduit le nom CamelCase en canonique AVANT compile().

  (B) System.add_coupled_source() cote C++ : si un role CANONIQUE inconnu est fourni directement
      (contournement de la DSL), RuntimeError est levee avec le message
      "System::add_coupled_source : role '...' inconnu (bloc '...')".
      role_from_name(role) renvoie VariableRole::Custom pour tout role non reconnu, et le code
      leve explicitement.

  (C) System.add_coupled_source() cote C++ : un role CANONIQUE valide mais NON EXPOSE par le bloc
      cible leve desormais RuntimeError (fix du gap identifie a la revue Lot E). Avant, resolve()
      retombait SILENCIEUSEMENT sur la composante 0 (role_index avec fallback=0) et appliquait la
      source au mauvais champ. Le chemin DSL CoupledSource est maintenant STRICT (index_of(role) < 0
      -> throw "le bloc '...' n'expose pas le role '...'"). Les couplages NOMMES (add_collision /
      add_pair) conservent volontairement leur repli canonique via role_index et ne sont PAS touches.
"""
import pytest

import pops
from pops import dsl


# ---------------------------------------------------------------------------
# (A) Rejet DSL : role inconnu dans CoupledSource.block().role() et .add()
# ---------------------------------------------------------------------------

def test_coupled_source_rejects_unknown_role_at_field():
    """(A1) block().role('bogus') leve ValueError car 'bogus' n'est pas dans _ROLE_TO_CANONICAL."""
    src = dsl.CoupledSource("test")
    raised = False
    msg = ""
    try:
        src.block("electrons").role("bogus_role")
    except ValueError as e:
        raised = True
        msg = str(e)
    assert raised, (
        "CoupledSource.block().role('bogus_role') aurait du lever ValueError (role inconnu)"
    )
    # Message confirme : "CoupledSource : role 'bogus_role' inconnu (roles : ...)"
    assert "bogus_role" in msg, "message inattendu : %r" % msg


def test_coupled_source_rejects_unknown_role_at_add():
    """(A2) .add(role='bad') leve ValueError car 'bad' n'est pas un role canonique connu."""
    src = dsl.CoupledSource("test")
    ne = src.block("electrons").role("density")  # valide
    raised = False
    msg = ""
    try:
        src.add("electrons", role="not_a_role", expr=ne)
    except ValueError as e:
        raised = True
        msg = str(e)
    assert raised, ".add(role='not_a_role') aurait du lever ValueError"
    assert "not_a_role" in msg, "message inattendu : %r" % msg


def test_coupled_source_accepts_valid_role():
    """Contrepartie positive : un role valide ('density') est accepte sans lever."""
    src = dsl.CoupledSource("test_ok")
    ne = src.block("electrons").role("density")
    ng = src.block("neutrals").role("density")
    k = src.param("k", 1.0)
    src.add("electrons", role="density", expr=+k * ne * ng)
    src.add("neutrals", role="density", expr=-k * ne * ng)
    compiled = src.compile(backend="production")
    assert compiled.in_roles == ["density", "density"]
    assert compiled.out_roles == ["density", "density"]


# ---------------------------------------------------------------------------
# (B) Rejet C++ : role canonique inconnu passe directement a add_coupled_source
# ---------------------------------------------------------------------------

def _make_cartesian_system():
    """System cartesien minimal avec un bloc scalaire pour les tests C++ directs."""
    sim = pops.System(n=4, L=1.0, periodic=True)
    sim.add_block(
        "ne",
        model=pops.Model(
            state=pops.Scalar(),
            transport=pops.ExB(B0=1.0),
            source=pops.NoSource(),
            elliptic=pops.ChargeDensity(charge=1.0),
        ),
        spatial=pops.Spatial(none=True),
        time=pops.Explicit(),
    )
    sim.set_poisson(rhs="charge_density", solver="geometric_mg")
    sim.set_density("ne", [1.0] * (4 * 4))
    return sim


def test_add_coupled_source_rejects_unknown_role_direct():
    """(B) add_coupled_source C++ direct avec role inconnu -> RuntimeError.

    Contourne la DSL pour tester le garde-fou C++ lui-meme.
    role_from_name("bogus_canonical") renvoie VariableRole::Custom -> throw RuntimeError
    "System::add_coupled_source : role 'bogus_canonical' inconnu (bloc 'ne')".
    """
    sim = _make_cartesian_system()
    # ABI plate minimale : 1 entree (ne, role connu), 1 terme de sortie (ne, role INCONNU).
    # Programme bytecode trivial : PushReg(0) = reg 0 (l'entree).
    _CS_PUSHREG = 0
    raised = False
    msg = ""
    try:
        sim._s.add_coupled_source(
            in_blocks=["ne"],
            in_roles=["density"],           # valide (canonique connu)
            consts=[],
            out_blocks=["ne"],
            out_roles=["bogus_canonical"],  # INCONNU -> VariableRole::Custom -> throw
            prog_ops=[_CS_PUSHREG],
            prog_args=[0],
            prog_lens=[1],
        )
    except RuntimeError as e:
        raised = True
        msg = str(e)
    assert raised, (
        "add_coupled_source avec out_role='bogus_canonical' aurait du lever RuntimeError "
        "(role_from_name -> VariableRole::Custom)"
    )
    # Message confirme : "System::add_coupled_source : role 'bogus_canonical' inconnu (bloc 'ne')"
    assert "bogus_canonical" in msg, "message inattendu : %r" % msg


def test_add_coupled_source_rejects_role_not_exposed():
    """(C) Role CANONIQUE valide mais NON EXPOSE par le bloc -> RuntimeError (fix du gap Lot E).

    Le bloc scalaire 'ne' n'expose que la densite (composante 0) ; viser 'momentum_x' (role
    canonique reconnu par role_from_name, mais absent du descripteur du scalaire) doit lever au
    lieu de retomber silencieusement sur la composante 0.
    """
    sim = _make_cartesian_system()
    _CS_PUSHREG = 0
    raised = False
    msg = ""
    try:
        sim._s.add_coupled_source(
            in_blocks=["ne"],
            in_roles=["density"],            # valide ET expose
            consts=[],
            out_blocks=["ne"],
            out_roles=["momentum_x"],        # canonique reconnu MAIS non expose par un scalaire
            prog_ops=[_CS_PUSHREG],
            prog_args=[0],
            prog_lens=[1],
        )
    except RuntimeError as e:
        raised = True
        msg = str(e)
    assert raised, (
        "add_coupled_source ciblant un role valide NON EXPOSE aurait du lever (gap Lot E corrige : "
        "plus de repli silencieux sur comp=0)"
    )
    assert "expose" in msg or "momentum_x" in msg, "message inattendu : %r" % msg


if __name__ == "__main__":
    test_coupled_source_rejects_unknown_role_at_field()
    print("OK A1 : role inconnu dans .block().role() rejete")
    test_coupled_source_rejects_unknown_role_at_add()
    print("OK A2 : role inconnu dans .add() rejete")
    test_coupled_source_accepts_valid_role()
    print("OK contrepartie positive : role valide accepte")
    test_add_coupled_source_rejects_unknown_role_direct()
    print("OK B : role inconnu dans add_coupled_source C++ rejete")
    test_add_coupled_source_rejects_role_not_exposed()
    print("OK C : role valide non expose rejete (gap Lot E corrige)")
    print("test_dsl_coupled_role_error : OK")
