#!/usr/bin/env python3
"""Test du MASQUE IMPLICITE par bloc (System-pipeline P3) : pops.IMEX / pops.SourceImplicit acceptent
implicit_vars=[noms] et/ou implicit_roles=[roles physiques] pour choisir QUELLES variables conservees
sont traitees en implicite dans le pas de source IMEX (les autres restent explicites, Euler avant).

DECISION DE CONCEPTION : le masque vit cote POLITIQUE TEMPORELLE / BLOC (et NON le modele) -> le MEME
modele se reutilise avec des traitements implicites differents. Defaut (pas de masque) = defaut modele
(Model::is_implicit, ou tout implicite a defaut de trait) -> numeriquement BIT-IDENTIQUE a avant.

Modele d'epreuve : electron Euler compressible (4 var : rho, rho_u, rho_v, E ; roles density, momentum_x,
momentum_y, energy) avec source PotentialForce raide (force (q/m) rho E sur la qte de mvt + travail sur
l'energie). La source de l'energie depend de la qte de mvt -> le couplage rend le RESULTAT sensible au
choix du set implicite (energie implicite = qte de mvt AVANCEE ; energie explicite = qte de mvt d'entree).

Verifie :
  1. implicit_vars=["rho_u","rho_v"] -> seules ces composantes implicites ; resultat DIFFERENT du defaut
     (tout implicite) -> le masque est bien APPLIQUE (et non ignore).
  2. implicit_roles=["MomentumX","MomentumY"] resout aux MEMES indices que implicit_vars=["rho_u","rho_v"]
     -> bit-identique (role -> index correct). Idem set complet par roles == par noms.
  3. Defaut (aucun masque) BIT-IDENTIQUE a la valeur de reference sans argument de masque.
  4. Erreur CLAIRE si un nom (implicit_vars) ou un role (implicit_roles) est absent du bloc.
  5. implicit_vars / implicit_roles sur une politique non-IMEX (Explicit) ou sur un backend compile
     -> erreur explicite (pas d'ignore silencieux).
"""

import sys
import numpy as np
import pops

fails = 0


def chk(cond, label):
    global fails
    ok = "OK " if cond else "XX "
    print(f"  [{ok}] {label}")
    if not cond:
        fails += 1


def meshx(n):
    return (np.arange(n) + 0.5) / n


def electron_model():
    # Euler compressible (4 var) + force du potentiel RAIDE (charge forte) : source non triviale sur
    # qte de mvt (depend de rho) et energie (depend de la qte de mvt).
    return pops.Model(state=pops.FluidState("compressible", gamma=1.4),
                     transport=pops.CompressibleFlux(),
                     source=pops.PotentialForce(charge=-50.0),
                     elliptic=pops.ChargeDensity(charge=-1.0))


def run(policy, n=24, dt=0.002, nsteps=4):
    """Avance le bloc electron avec la politique temporelle @p policy ; renvoie l'etat final (4, n, n)."""
    s = pops.System(n=n, periodic=False)
    s.add_block("ne", electron_model(), spatial=pops.Spatial(minmod=True), time=policy)
    s.set_poisson(bc="dirichlet")
    xs = meshx(n)
    rho_e = 1.0 + 0.2 * np.cos(2 * np.pi * xs)[None, :] * np.ones((n, n))
    s.set_density("ne", rho_e)
    s.advance(dt, nsteps)
    return np.array(s.get_state("ne")).copy()


# ---- 0. attributs portes par la politique (masque cote bloc, pas modele) -------
print("== 0. IMEX / SourceImplicit portent le masque implicite ==")
p = pops.IMEX(substeps=2, implicit_vars=["rho_u", "rho_v"])
chk(p.implicit_vars == ["rho_u", "rho_v"], "IMEX.implicit_vars stocke les noms")
chk(p.implicit_roles == [], "IMEX.implicit_roles vide par defaut")
pr = pops.IMEX(implicit_roles=["MomentumX", "MomentumY", "Energy"])
# normalisation PascalCase -> cle stable snake_case (cf. role_from_name C++)
chk(pr.implicit_roles == ["momentum_x", "momentum_y", "energy"],
    "IMEX.implicit_roles normalise PascalCase -> snake_case stable")
si = pops.SourceImplicit(implicit_vars=["E"])
chk(si.implicit_vars == ["E"], "SourceImplicit.implicit_vars stocke les noms")
chk(pops.IMEX().implicit_vars == [] and pops.IMEX().implicit_roles == [],
    "IMEX() sans masque : listes vides (defaut)")

# ---- 1. le masque est APPLIQUE : implicit_vars change la numerique ------------
print("== 1. implicit_vars=['rho_u','rho_v'] : seules ces composantes implicites ==")
ref_full = run(pops.IMEX(substeps=2))                                  # defaut : tout implicite
masked_mom = run(pops.IMEX(substeps=2, implicit_vars=["rho_u", "rho_v"]))  # qte de mvt seule
diff = float(np.max(np.abs(masked_mom - ref_full)))
chk(diff > 0.0,
    "masque qte de mvt != defaut tout-implicite (le masque est applique, diff=%g)" % diff)

# ---- 2. implicit_roles resout aux MEMES indices que implicit_vars -------------
print("== 2. implicit_roles -> memes indices que implicit_vars (role -> index) ==")
masked_roles = run(pops.IMEX(substeps=2, implicit_roles=["MomentumX", "MomentumY"]))
d_role_name = float(np.max(np.abs(masked_roles - masked_mom)))
chk(d_role_name == 0.0,
    "implicit_roles=[MomentumX,MomentumY] == implicit_vars=[rho_u,rho_v] (bit-identique, diff=%g)"
    % d_role_name)

# set complet (qte de mvt + energie) par roles == par noms (3 composantes)
full3_names = run(pops.IMEX(substeps=2, implicit_vars=["rho_u", "rho_v", "E"]))
full3_roles = run(pops.IMEX(substeps=2, implicit_roles=["MomentumX", "MomentumY", "Energy"]))
d3 = float(np.max(np.abs(full3_roles - full3_names)))
chk(d3 == 0.0,
    "implicit_roles=[MomentumX,MomentumY,Energy] == implicit_vars=[rho_u,rho_v,E] (diff=%g)" % d3)
# et ce set (qui exclut rho dont la source est nulle) differe du masque qte-de-mvt-seule
chk(float(np.max(np.abs(full3_names - masked_mom))) > 0.0,
    "ajouter 'E' au set implicite change le resultat (energie implicite vs explicite)")

# ---- 3. defaut (aucun masque) BIT-IDENTIQUE -----------------------------------
print("== 3. defaut sans masque : bit-identique ==")
ref_full_b = run(pops.IMEX(substeps=2))
chk(float(np.max(np.abs(ref_full_b - ref_full))) == 0.0,
    "IMEX(substeps=2) sans masque : reproductible et bit-identique a la reference")
# SourceImplicit sans masque == IMEX sans masque (meme chemin C++)
si_ref = run(pops.SourceImplicit(substeps=2))
chk(float(np.max(np.abs(si_ref - ref_full))) == 0.0,
    "SourceImplicit(substeps=2) sans masque == IMEX(substeps=2) (bit-identique)")

# ---- 4. erreur claire si nom / role absent du bloc ----------------------------
print("== 4. erreur explicite sur un nom / role absent ==")
try:
    run(pops.IMEX(implicit_vars=["rho_w"]))   # 'rho_w' n'existe pas (le bloc a rho,rho_u,rho_v,E)
    chk(False, "implicit_vars=['rho_w'] doit lever (nom absent)")
except Exception as e:
    msg = str(e)
    chk("rho_w" in msg and "implicit_vars" in msg,
        "nom absent -> erreur mentionnant 'rho_w' et 'implicit_vars'")

try:
    run(pops.IMEX(implicit_roles=["Pressure"]))  # role absent des conservatives Euler
    chk(False, "implicit_roles=['Pressure'] doit lever (role absent)")
except Exception as e:
    msg = str(e)
    chk("implicit_roles" in msg,
        "role absent -> erreur mentionnant 'implicit_roles'")

try:
    run(pops.IMEX(implicit_roles=["NotARole"]))  # role inconnu (mappe sur Custom)
    chk(False, "implicit_roles=['NotARole'] doit lever (role inconnu)")
except Exception as e:
    chk("implicit_roles" in str(e),
        "role inconnu -> erreur explicite")

# ---- 5. masque interdit hors IMEX (explicite) ---------------------------------
print("== 5. masque rejete sur une politique non-IMEX ==")
# pops.Explicit ne porte pas implicit_vars ; on simule un appel direct add_block avec un masque
# sur une politique explicite : le C++ doit lever (le masque n'a de sens qu'en IMEX).
try:
    s = pops.System(n=16, periodic=False)
    s._s.add_block("ne", electron_model(), "minmod", "rusanov", "conservative",
                   "explicit", 1, True, 1, ["rho_u"], [])
    chk(False, "add_block(time='explicit', implicit_vars=['rho_u']) doit lever")
except Exception as e:
    chk("imex" in str(e).lower(),
        "masque sur time='explicit' -> erreur mentionnant imex")

# ---- Bilan -------------------------------------------------------------------
print()
n_chks = sum(1 for line in open(__file__) if line.strip().startswith("chk("))
if fails == 0:
    print("OK test_implicit_vars (%d assertions)" % n_chks)
else:
    print("ECHEC test_implicit_vars : %d assertion(s) en erreur" % fails)
    sys.exit(1)
