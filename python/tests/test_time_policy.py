#!/usr/bin/env python3
"""Test du nommage des politiques temporelles : SourceImplicit, IMEX, deprecation de Implicit.

Verifie :
  1. pops.SourceImplicit produit les memes numeriques (bit-identiques) que pops.IMEX et
     que l'ancien pops.Implicit -- les trois empruntes le meme chemin C++ (kind="imex",
     ImplicitSourceStepper / backward_euler_source).
  2. pops.Implicit leve bien un DeprecationWarning (avec le message attendu) et reste
     fonctionnel (pas de regression comportementale).
  3. pops.Explicit / pops.IMEX sont INCHANGES (bit-identiques par rapport aux tests existants).
  4. pops.SourceImplicit est exportee dans pops.__all__.
"""

import sys
import warnings
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


def diocotron_model(B0=1.0, alpha=1.0, n0=0.0):
    return pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=B0),
                     source=pops.NoSource(),
                     elliptic=pops.BackgroundDensity(alpha=alpha, n0=n0))


def electron_model():
    return pops.Model(state=pops.FluidState("compressible", gamma=1.4),
                     transport=pops.CompressibleFlux(),
                     source=pops.PotentialForce(charge=-1.0),
                     elliptic=pops.ChargeDensity(charge=-1.0))


# ---- 1. SourceImplicit est dans __all__ et a kind="imex" -----------------------
print("== 1. SourceImplicit : presence dans __all__, kind, attributs ==")
chk("SourceImplicit" in pops.__all__, "SourceImplicit est dans pops.__all__")
si = pops.SourceImplicit(substeps=3, stride=2)
chk(si.kind == "imex", "SourceImplicit.kind == 'imex' (meme chemin C++ que IMEX)")
chk(si.substeps == 3, "SourceImplicit.substeps correctement stocke")
chk(si.stride == 2, "SourceImplicit.stride correctement stocke")

imex_ref = pops.IMEX(substeps=3, stride=2)
chk(si.kind == imex_ref.kind, "SourceImplicit.kind == IMEX.kind")
chk(si.substeps == imex_ref.substeps, "SourceImplicit.substeps == IMEX.substeps")
chk(si.stride == imex_ref.stride, "SourceImplicit.stride == IMEX.stride")

# ---- 2. SourceImplicit : validation des entrees --------------------------------
print("== 2. SourceImplicit : validation des entrees ==")
try:
    pops.SourceImplicit(substeps=0)
    chk(False, "SourceImplicit(substeps=0) doit lever ValueError")
except ValueError:
    chk(True, "SourceImplicit(substeps=0) leve ValueError")

try:
    pops.SourceImplicit(stride=0)
    chk(False, "SourceImplicit(stride=0) doit lever ValueError")
except ValueError:
    chk(True, "SourceImplicit(stride=0) leve ValueError")

# ---- 3. Numeriques bit-identiques : SourceImplicit == IMEX == Implicit ----------
# On avance le meme etat initial avec les trois politiques sur un seul bloc
# (diocotron ExB, domaine non periodique, Poisson Dirichlet) et on verifie que le
# resultat final est bit-identique -- ce qui confirme que les trois empruntent le
# meme chemin C++ (ImplicitSourceStepper, backward_euler_source).
print("== 3. Numeriques bit-identiques : SourceImplicit == IMEX == Implicit ==")
n = 32
dt = 0.001
xs = meshx(n)
rho0 = 1.0 + 0.04 * np.cos(2 * np.pi * xs)[None, :] * np.ones((n, 1))

policies = {
    "SourceImplicit": pops.SourceImplicit(substeps=2),
    "IMEX": pops.IMEX(substeps=2),
}

# Implicit : on le capturera en supprimant le warning (retrocompatibilite)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    policies["Implicit(dt_ratio=2)"] = pops.Implicit(dt_ratio=2)

# Electron model (Euler compressible IMEX) pour exercer le chemin backward_euler_source
# (la source PotentialForce est raide ; le chemin imex est plus significatif qu'ExB/NoSource).
results = {}
for label, policy in policies.items():
    s = pops.System(n=n, periodic=False)
    s.add_block("ne", electron_model(),
                spatial=pops.Spatial(minmod=True), time=policy)
    s.set_poisson(bc="dirichlet")
    rho_e = 1.0 + 0.04 * np.cos(2 * np.pi * xs)[None, :] * np.ones((n, n))
    s.set_density("ne", rho_e)
    s.advance(dt, 4)
    results[label] = np.array(s.density("ne")).copy()

ref = results["IMEX"]
for label, arr in results.items():
    diff = float(np.max(np.abs(arr - ref)))
    chk(diff == 0.0,
        "%s vs IMEX : bit-identiques (diff=%g)" % (label, diff))

# ---- 4. pops.Implicit leve un DeprecationWarning --------------------------------
print("== 4. pops.Implicit emet un DeprecationWarning ==")

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = pops.Implicit()
    dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    chk(len(dep_warnings) >= 1,
        "pops.Implicit() leve au moins un DeprecationWarning")
    if dep_warnings:
        msg = str(dep_warnings[0].message)
        chk("SourceImplicit" in msg or "IMEX" in msg,
            "DeprecationWarning mentionne SourceImplicit ou IMEX")
        chk("Implicit" in msg,
            "DeprecationWarning mentionne Implicit (nom obsolete)")

# Retrocompatibilite : Implicit() retourne un objet fonctionnel (kind="imex")
chk(result.kind == "imex",
    "pops.Implicit() retourne un objet fonctionnel (kind='imex') apres le warning")

# Arguments historiques : dt_ratio, substeps, stride
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    chk(pops.Implicit(dt_ratio=4).substeps == 4,
        "pops.Implicit(dt_ratio=4) -> substeps=4 (retrocompatibilite)")
    chk(pops.Implicit(substeps=3).substeps == 3,
        "pops.Implicit(substeps=3) -> substeps=3 (retrocompatibilite)")
    chk(pops.Implicit(stride=2).stride == 2,
        "pops.Implicit(stride=2) -> stride=2 (retrocompatibilite)")

# ---- 5. pops.Explicit et pops.IMEX : comportement INCHANGE -----------------------
# On verifie juste que les attributs et le kind sont les bons (les tests numeriques
# sont couverts par test_bindings et test_stride ; on ne les reproduit pas ici).
print("== 5. pops.Explicit et pops.IMEX inchanges ==")
ex = pops.Explicit()
chk(ex.kind == "explicit", "Explicit().kind == 'explicit' (inchange)")
chk(ex.substeps == 1, "Explicit().substeps == 1 (defaut inchange)")
ex3 = pops.Explicit(method="ssprk3")
chk(ex3.kind == "ssprk3", "Explicit(ssprk3).kind == 'ssprk3' (inchange)")

imex = pops.IMEX()
chk(imex.kind == "imex", "IMEX().kind == 'imex' (inchange)")
chk(imex.substeps == 1, "IMEX().substeps == 1 (defaut inchange)")

# pops.Implicit PAS dans un bloc pops.IMEX / pops.Explicit : on s'assure que les deux
# n'emettent PAS de DeprecationWarning.
with warnings.catch_warnings(record=True) as w2:
    warnings.simplefilter("always")
    pops.Explicit(substeps=2)
    pops.IMEX(substeps=2)
    dep2 = [x for x in w2 if issubclass(x.category, DeprecationWarning)]
    chk(len(dep2) == 0,
        "Explicit() et IMEX() ne levent aucun DeprecationWarning")

# ---- Bilan -------------------------------------------------------------------
print()
n_chks = sum(1 for line in open(__file__) if line.strip().startswith("chk("))
if fails == 0:
    print("OK test_time_policy (%d assertions)" % n_chks)
else:
    print("ECHEC test_time_policy : %d assertion(s) en erreur" % fails)
    sys.exit(1)
