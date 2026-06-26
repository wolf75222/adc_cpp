#!/usr/bin/env python3
"""Revue #195 : AmrSystem.add_equation REJETTE explicitement la cadence multirate (stride > 1) et le
masque IMEX partiel (implicit_vars / implicit_roles) sur le CHEMIN PRODUCTION AMR (CompiledModel
backend='production', target='amr_system' -> AmrSystem.add_native_block).

POURQUOI un rejet et pas un fix : l'ABI PLATE du loader .so (symbole pops_install_native_amr,
add_native_block) ne transporte NI stride NI le masque IMEX. Passes par ce chemin, ils prendraient
leurs defauts EN SILENCE (stride=1, masque vide = backward-Euler plein) -> demi-fix silencieux interdit.
La facade les rejette donc clairement (meme esprit que System.add_equation, qui rejette stride>1 et le
masque sur les backends compiles .so : cf. python/tests/test_stride.py section 8).

Route explicite vers les chemins qui SUPPORTENT ces parametres :
  - AmrSystem.add_block (modele natif pops.Model(...)) : stride + masque cables (et FORWARDES par
    add_equation sur la branche ModelSpec) ;
  - add_compiled_model(AmrSystem&) en DIRECT (C++) : stride + masque exposes (exerce par le test C++
    tests/test_amr_multiblock_compiled.cpp, cas (F)).

Ce test est PUR PYTHON et NE DEPEND PAS d'un compilateur : la garde est purement Python et leve AVANT
tout dlopen du .so, donc un CompiledModel FACTICE (backend='production', target='amr_system', .so
inexistant) suffit (deterministe en CI minimale ; meme recette que test_stride.py).
"""

import sys

import pops
from pops import dsl

fails = 0


def chk(cond, label):
    global fails
    ok = "OK " if cond else "XX "
    print(f"  [{ok}] {label}")
    if not cond:
        fails += 1


def fake_production_amr():
    """CompiledModel FACTICE du chemin production AMR : backend='production', target='amr_system',
    adder='add_native_block', .so inexistant. La garde stride/masque de add_equation leve AVANT le
    dlopen du .so, donc le chemin n'est jamais reellement charge."""
    return dsl.CompiledModel(
        so_path="/inexistant_amr.so", backend="production", adder="add_native_block",
        cons_names=["rho", "rho_u", "rho_v", "E"],
        cons_roles=["Density", "MomentumX", "MomentumY", "Energy"],
        prim_names=["rho", "u", "v", "p"], n_vars=4, gamma=1.4, n_aux=3, params={}, caps={},
        abi_key="k", model_hash="h", cxx="c++", std="c++20", target="amr_system")


# ---- 1. stride > 1 via pops.IMEX(stride=...) -> ValueError claire -------------------------------
print("== production AMR : stride>1 (IMEX) rejete explicitement ==")
sim = pops.AmrSystem(n=16, periodic=True)
try:
    sim.add_equation("gas", fake_production_amr(), spatial=pops.FiniteVolume(),
                     time=pops.IMEX(stride=5))
    chk(False, "add_equation(time=IMEX(stride=5), production AMR) doit lever ValueError")
except ValueError as ex:
    chk("stride" in str(ex) and "production" in str(ex),
        "add_equation(stride=5, production AMR) leve une ValueError claire (stride/production)")

# stride > 1 aussi via pops.Explicit(stride=...) (la cadence n'est pas specifique a l'IMEX).
print("== production AMR : stride>1 (Explicit) rejete explicitement ==")
sim_e = pops.AmrSystem(n=16, periodic=True)
try:
    sim_e.add_equation("gas", fake_production_amr(), spatial=pops.FiniteVolume(),
                       time=pops.Explicit(stride=5))
    chk(False, "add_equation(time=Explicit(stride=5), production AMR) doit lever ValueError")
except ValueError as ex:
    chk("stride" in str(ex) and "production" in str(ex),
        "add_equation(Explicit(stride=5), production AMR) leve une ValueError claire")

# ---- 2. masque IMEX partiel (implicit_vars / implicit_roles) -> ValueError claire --------------
print("== production AMR : implicit_vars (masque IMEX partiel) rejete explicitement ==")
sim2 = pops.AmrSystem(n=16, periodic=True)
try:
    sim2.add_equation("gas", fake_production_amr(), spatial=pops.FiniteVolume(),
                      time=pops.IMEX(implicit_vars=["rho_u"]))
    chk(False, "add_equation(IMEX(implicit_vars=['rho_u']), production AMR) doit lever ValueError")
except ValueError as ex:
    chk("implicit" in str(ex) and "production" in str(ex),
        "add_equation(implicit_vars, production AMR) leve une ValueError claire (implicit/production)")

print("== production AMR : implicit_roles (masque IMEX partiel) rejete explicitement ==")
sim3 = pops.AmrSystem(n=16, periodic=True)
try:
    sim3.add_equation("gas", fake_production_amr(), spatial=pops.FiniteVolume(),
                      time=pops.IMEX(implicit_roles=["momentum_x"]))
    chk(False, "add_equation(IMEX(implicit_roles=['momentum_x']), production AMR) doit lever ValueError")
except ValueError as ex:
    chk("implicit" in str(ex) and "production" in str(ex),
        "add_equation(implicit_roles, production AMR) leve une ValueError claire")

# ---- 3. PAS DE FAUX POSITIF : stride=1 + masque vide NE doivent PAS lever de ValueError ---------
# Le defaut (Explicit, stride=1, masque vide) doit traverser la garde et echouer PLUS LOIN au dlopen
# du .so inexistant (RuntimeError) -- jamais une ValueError stride/masque. Idem IMEX sans masque.
print("== production AMR : stride=1 / masque vide NE levent PAS (pas de faux positif) ==")
sim_ok = pops.AmrSystem(n=16, periodic=True)
try:
    sim_ok.add_equation("gas", fake_production_amr(), spatial=pops.FiniteVolume(),
                        time=pops.Explicit())  # stride=1, masque vide : DEFAUT
    chk(False, "add_equation(Explicit(), production AMR) : attendu un echec au dlopen (.so inexistant)")
except ValueError as ex:
    chk(False, "add_equation(Explicit(), production AMR) ne doit PAS lever de ValueError (%s)" % ex)
except Exception:  # noqa: BLE001  RuntimeError du dlopen attendu : la garde a laisse passer
    chk(True, "add_equation(Explicit(), production AMR) : pas de rejet (echec au dlopen attendu)")

sim_ok_imex = pops.AmrSystem(n=16, periodic=True)
try:
    sim_ok_imex.add_equation("gas", fake_production_amr(), spatial=pops.FiniteVolume(),
                             time=pops.IMEX())  # stride=1, masque vide : IMEX plein backward-Euler
    chk(False, "add_equation(IMEX(), production AMR) : attendu un echec au dlopen (.so inexistant)")
except ValueError as ex:
    chk(False, "add_equation(IMEX(), production AMR) ne doit PAS lever de ValueError (%s)" % ex)
except Exception:  # noqa: BLE001  RuntimeError du dlopen attendu
    chk(True, "add_equation(IMEX(), production AMR) : pas de rejet (echec au dlopen attendu)")

# ---- Bilan ------------------------------------------------------------------------------------
print()
if fails == 0:
    print("OK test_amr_production_stride_reject (%d assertions)" % (
        sum(1 for line in open(__file__) if line.strip().startswith("chk("))))
else:
    print("ECHEC test_amr_production_stride_reject : %d assertion(s) en erreur" % fails)
    sys.exit(1)
