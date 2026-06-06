"""Lot E - verrouillage du rejet de role inconnu dans CoupledSource / add_coupled_source.

Deux points de rejet sont confirmes (lus dans python/adc/dsl.py et python/system.cpp) :

  (A) CoupledSource.compile() cote PYTHON : _role_canonical() leve ValueError pour un role
      dont le nom n'appartient pas a _ROLE_TO_CANONICAL (p.ex. 'massflux', 'bogus_role').
      Cela se produit au moment ou le champ (block, role) est cree via .block().role(name) OU
      via .add(role=name). La DSL traduit le nom CamelCase en canonique AVANT compile().

  (B) System.add_coupled_source() cote C++ : si un role CANONIQUE inconnu est fourni directement
      (contournement de la DSL), RuntimeError est levee avec le message
      "System::add_coupled_source : role '...' inconnu (bloc '...')".
      role_from_name(role) renvoie VariableRole::Custom pour tout role non reconnu, et le code
      leve explicitement.

NOTE IMPORTANTE - ce qui n'est PAS un rejet :
  Si un role syntaxiquement valide ('density', 'momentum_x', ...) est passe a un bloc qui ne
  l'expose PAS dans ses variables conservatives, add_coupled_source ne leve PAS : il utilise le
  fallback silencieux comp=0 (role_index avec fallback=0). Ce comportement est documente dans
  le code (commentaire "Resout role -> composante ... fallback comp 0 si le bloc ne renseigne
  pas le role") et ne constitue PAS un rejet honete. On ne teste donc PAS ce cas ici (gap
  volontairement non couvert : le fallback silencieux est le contrat actuel du code).
"""
import pytest

import adc
from adc import dsl


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
    assert "inconnu" in msg or "bogus_role" in msg, "message inattendu : %r" % msg


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
    assert "inconnu" in msg or "not_a_role" in msg, "message inattendu : %r" % msg


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
    sim = adc.System(n=4, L=1.0, periodic=True)
    sim.add_block(
        "ne",
        model=adc.Model(
            state=adc.Scalar(),
            transport=adc.ExB(B0=1.0),
            source=adc.NoSource(),
            elliptic=adc.ChargeDensity(charge=1.0),
        ),
        spatial=adc.Spatial(none=True),
        time=adc.Explicit(),
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
    assert "inconnu" in msg or "bogus_canonical" in msg, "message inattendu : %r" % msg


if __name__ == "__main__":
    test_coupled_source_rejects_unknown_role_at_field()
    print("OK A1 : role inconnu dans .block().role() rejete")
    test_coupled_source_rejects_unknown_role_at_add()
    print("OK A2 : role inconnu dans .add() rejete")
    test_coupled_source_accepts_valid_role()
    print("OK contrepartie positive : role valide accepte")
    test_add_coupled_source_rejects_unknown_role_direct()
    print("OK B : role inconnu dans add_coupled_source C++ rejete")
    print("test_dsl_coupled_role_error : OK")
