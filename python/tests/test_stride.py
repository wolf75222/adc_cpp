#!/usr/bin/env python3
"""Test de l'exposition du parametre `stride` dans les politiques temporelles Python.

Semantique verifiee = HOLD-THEN-CATCH-UP (rattrapage en FIN de fenetre) :
  - stride=M  : le bloc est TENU (etat INCHANGE) tant que (macro_step + 1) % M != 0, puis avance
                d'un pas EFFECTIF M*dt au macro-pas ou (macro_step + 1) % M == 0 (fin de fenetre).
                Le bloc reste ainsi temporellement coherent avec les blocs rapides (jamais avance
                "dans le futur" des le pas 0, contrairement a l'ancienne semantique de DEBUT de
                fenetre macro_step % M == 0).
  - stride=1  : comportement par defaut, BIT-IDENTIQUE a l'historique sans stride.
  - CFL       : step_cfl honore la cadence -- un bloc stride=M limiteur fait retourner un dt ~ dt/M
                par rapport a stride=1 (le facteur stride entre dans la condition stable par bloc).
  - AOT       : Explicit(stride>1) + backend='aot' (CompiledModel) leve une erreur CLAIRE
                (la cadence n'est pas cablee dans l'ABI du .so AOT -> pas d'ignore silencieux).
  - evolve=False : bloc GELE (etat inchange), mais toujours visible par le Poisson de systeme.
  - Multi-blocs (stride=1 + stride=3) : les cadences respectives sont respectees.
"""

import sys
import numpy as np
import adc

fails = 0


def chk(cond, label):
    global fails
    ok = "OK " if cond else "XX "
    print(f"  [{ok}] {label}")
    if not cond:
        fails += 1


def changed(a, b):
    """Renvoie True si les tableaux ne sont PAS bit-identiques (avance reelle detectee)."""
    return not np.array_equal(a, b)


def frozen(a, b):
    """Renvoie True si les tableaux sont bit-identiques (bloc tenu/gele : pas d'avance)."""
    return np.array_equal(a, b)


def meshx(n):
    return (np.arange(n) + 0.5) / n


def diocotron_model(n0=1.0):
    """Modele scalaire ExB avec fond neutralisant (alpha=1, n0). Domaine NON periodique
    (Dirichlet) : le second membre peut avoir une integrale non nulle."""
    return adc.Model(state=adc.Scalar(), transport=adc.ExB(B0=1.0),
                     source=adc.NoSource(),
                     elliptic=adc.BackgroundDensity(alpha=1.0, n0=n0))


dt = 0.001

# ---- 1. stride=1 : comportement bit-identique a l'historique -------------------
print("== stride=1 : parite bit-identique avec l'historique ==")
n = 32
xs = meshx(n)
rho0 = 1.0 + 0.02 * np.cos(2 * np.pi * xs)[None, :] * np.ones((n, 1))

s_ref = adc.System(n=n, periodic=False)
s_ref.add_block("ne", diocotron_model(), time=adc.Explicit(substeps=1))
s_ref.set_poisson(bc="dirichlet")
s_ref.set_density("ne", rho0)

s_s1 = adc.System(n=n, periodic=False)
s_s1.add_block("ne", diocotron_model(), time=adc.Explicit(substeps=1, stride=1))
s_s1.set_poisson(bc="dirichlet")
s_s1.set_density("ne", rho0)

for _ in range(5):
    s_ref.step(dt)
    s_s1.step(dt)

diff = np.max(np.abs(s_ref.density("ne") - s_s1.density("ne")))
chk(diff == 0.0, "stride=1 bit-identique a l'historique (diff=%g)" % diff)

# ---- 2. stride=M HOLD-THEN-CATCH-UP : tenu jusqu'a la FIN de la fenetre ---------
# Cadence M=3, semantique de fin de fenetre : (macro_step + 1) % 3 == 0.
#   macro_step 0 -> (0+1)%3=1 != 0 -> TENU
#   macro_step 1 -> (1+1)%3=2 != 0 -> TENU
#   macro_step 2 -> (2+1)%3=0      -> RATTRAPAGE (avance dt_eff=3*dt)
#   macro_step 3 -> TENU ; macro_step 4 -> TENU ; macro_step 5 -> RATTRAPAGE
print("== stride=M : HOLD-THEN-CATCH-UP (rattrapage en fin de fenetre) ==")
M = 3
n = 32
rho0 = 1.0 + 0.02 * np.cos(2 * np.pi * meshx(n))[None, :] * np.ones((n, 1))

sim = adc.System(n=n, periodic=False)
sim.add_block("ne", diocotron_model(), time=adc.Explicit(stride=M))
sim.set_poisson(bc="dirichlet")
sim.set_density("ne", rho0)

# macro_step 0 : (0+1)%3 != 0 -> TENU (etat inchange)
s_before_0 = sim.density("ne").copy()
sim.step(dt)
s_after_0 = sim.density("ne").copy()
chk(frozen(s_before_0, s_after_0),
    "stride=3, macro-pas 0 (debut de fenetre) : le bloc est TENU (etat inchange)")

# macro_step 1 : (1+1)%3 != 0 -> TENU
s_before_1 = sim.density("ne").copy()
sim.step(dt)
s_after_1 = sim.density("ne").copy()
chk(frozen(s_before_1, s_after_1),
    "stride=3, macro-pas 1 (milieu de fenetre) : le bloc est TENU (etat inchange)")

# macro_step 2 : (2+1)%3 == 0 -> RATTRAPAGE (avance)
s_before_2 = sim.density("ne").copy()
sim.step(dt)
s_after_2 = sim.density("ne").copy()
chk(changed(s_before_2, s_after_2),
    "stride=3, macro-pas 2 (fin de fenetre) : le bloc RATTRAPE (etat change)")

# macro_step 3, 4 : tenus ; macro_step 5 : rattrapage
s_before_3 = sim.density("ne").copy()
sim.step(dt)  # macro_step 3 -> TENU
chk(frozen(s_before_3, sim.density("ne")),
    "stride=3, macro-pas 3 (nouvelle fenetre) : le bloc est TENU")
s_before_4 = sim.density("ne").copy()
sim.step(dt)  # macro_step 4 -> TENU
chk(frozen(s_before_4, sim.density("ne")),
    "stride=3, macro-pas 4 : le bloc est TENU")
s_before_5 = sim.density("ne").copy()
sim.step(dt)  # macro_step 5 -> (5+1)%3==0 -> RATTRAPAGE
chk(changed(s_before_5, sim.density("ne")),
    "stride=3, macro-pas 5 (fin de fenetre) : le bloc RATTRAPE a nouveau")

# ---- 3. stride=M : la VALEUR rattrapee = avance directe d'un pas M*dt -----------
# Apres M macro-pas (le bloc n'avance qu'une fois, en fin de fenetre, avec dt_eff=M*dt),
# l'etat doit egaler bit-pres celui d'un bloc stride=1 qui a fait UN pas de M*dt.
# (Hold-then-catch-up : Poisson identique au moment du rattrapage car l'etat est fige avant.)
print("== stride=M : valeur du rattrapage = un pas direct M*dt ==")
M = 5
n = 32
rho_init = 1.0 + 0.02 * np.cos(2 * np.pi * meshx(n))[None, :] * np.ones((n, 1))

sim_slow = adc.System(n=n, periodic=False)
sim_slow.add_block("ne", diocotron_model(), time=adc.Explicit(stride=M))
sim_slow.set_poisson(bc="dirichlet")
sim_slow.set_density("ne", rho_init.copy())
for _ in range(M):
    sim_slow.step(dt)  # 5 pas : le bloc rattrape UNE fois (macro_step 4) avec dt_eff=5*dt

sim_direct = adc.System(n=n, periodic=False)
sim_direct.add_block("ne", diocotron_model(), time=adc.Explicit(stride=1))
sim_direct.set_poisson(bc="dirichlet")
sim_direct.set_density("ne", rho_init.copy())
sim_direct.step(M * dt)  # un seul pas de M*dt

diff_eff = np.max(np.abs(sim_slow.density("ne") - sim_direct.density("ne")))
chk(diff_eff < 1e-12,
    "stride=5 : valeur rattrapee = avance directe 5*dt (diff : %g)" % diff_eff)

# ---- 4. CFL : un bloc stride=M limiteur fait retourner dt ~ dt/M ----------------
# step_cfl doit honorer la cadence : dt_b <= cfl*h*substeps / (stride*w). Avec UN seul bloc,
# le dt retourne par stride=M vaut donc 1/M fois celui de stride=1 (meme w, meme h, meme cfl).
print("== CFL : step_cfl honore le facteur stride ==")
n = 32
rho_cfl = 1.0 + 0.05 * np.cos(2 * np.pi * meshx(n))[None, :] * np.ones((n, 1))
cfl = 0.4
M = 4

sim_cfl1 = adc.System(n=n, periodic=False)
sim_cfl1.add_block("ne", diocotron_model(), time=adc.Explicit(stride=1))
sim_cfl1.set_poisson(bc="dirichlet")
sim_cfl1.set_density("ne", rho_cfl.copy())
dt1 = sim_cfl1.step_cfl(cfl)

sim_cflM = adc.System(n=n, periodic=False)
sim_cflM.add_block("ne", diocotron_model(), time=adc.Explicit(stride=M))
sim_cflM.set_poisson(bc="dirichlet")
sim_cflM.set_density("ne", rho_cfl.copy())
dtM = sim_cflM.step_cfl(cfl)

# Au premier appel macro_step_=0 : solve_fields + max_speed identiques -> rapport exact 1/M.
ratio = dtM / dt1
chk(abs(ratio - 1.0 / M) < 1e-9,
    "step_cfl : stride=%d -> dt = dt(stride=1)/%d (ratio mesure %g, attendu %g)"
    % (M, M, ratio, 1.0 / M))

# ---- 5. evolve=False : bloc gele, toujours dans le Poisson ---------------------
print("== evolve=False : bloc gele, second membre Poisson preserve ==")
n = 32
ne_model = adc.Model(state=adc.Scalar(), transport=adc.ExB(B0=1.0),
                     source=adc.NoSource(),
                     elliptic=adc.ChargeDensity(charge=-1.0))
ni_model = adc.Model(state=adc.Scalar(), transport=adc.ExB(B0=1.0),
                     source=adc.NoSource(),
                     elliptic=adc.ChargeDensity(charge=1.0))
rho_e_ev = 1.0 + 0.02 * np.cos(2 * np.pi * meshx(n))[None, :] * np.ones((n, 1))
rho_bg_ev = 0.5 * np.ones((n, n))

sim_ev = adc.System(n=n, periodic=True)
sim_ev.add_block("ne", ne_model, time=adc.Explicit(), evolve=True)
sim_ev.add_block("ni", ni_model, time=adc.Explicit(), evolve=False)
sim_ev.set_poisson()
sim_ev.set_density("ne", rho_e_ev)
sim_ev.set_density("ni", rho_bg_ev)
sim_ev.solve_fields()
phi_with_ni = sim_ev.potential().copy()

sim_no_ni = adc.System(n=n, periodic=True)
sim_no_ni.add_block("ne", ne_model, time=adc.Explicit(), evolve=True)
sim_no_ni.set_poisson()
sim_no_ni.set_density("ne", rho_e_ev)
sim_no_ni.solve_fields()
phi_no_ni = sim_no_ni.potential()

chk(not np.array_equal(phi_with_ni, phi_no_ni),
    "evolve=False : le bloc gele contribue au Poisson (phi different sans ni)")

state_ni_before = sim_ev.density("ni").copy()
sim_ev.step(0.001)
state_ni_after = sim_ev.density("ni")
chk(frozen(state_ni_before, state_ni_after),
    "evolve=False : le bloc gele ne change pas apres step")

# ---- 6. Multi-blocs : stride=1 + stride=3, cadences respectees ----------------
# Fenetres de stride=3 : rattrapage aux macro-pas 2, 5, 8 ((k+1)%3==0).
print("== multi-blocs : stride=1 + stride=3 (rattrapage en fin de fenetre) ==")
n = 32
rho_a = 1.0 + 0.02 * np.cos(2 * np.pi * meshx(n))[None, :] * np.ones((n, 1))
rho_b = 1.0 + 0.01 * np.sin(2 * np.pi * meshx(n))[None, :] * np.ones((n, 1))

sim_mb = adc.System(n=n, periodic=False)
sim_mb.add_block("a", diocotron_model(), time=adc.Explicit(stride=1))
sim_mb.add_block("b", diocotron_model(), time=adc.Explicit(stride=3))
sim_mb.set_poisson(bc="dirichlet")
sim_mb.set_density("a", rho_a)
sim_mb.set_density("b", rho_b)

# macro_step 0 : a avance (stride 1) ; b tenu ((0+1)%3 != 0)
a0, b0 = sim_mb.density("a").copy(), sim_mb.density("b").copy()
sim_mb.step(dt)
a1, b1 = sim_mb.density("a").copy(), sim_mb.density("b").copy()
chk(changed(a0, a1), "multi-blocs, pas 0 : bloc a (stride=1) avance")
chk(frozen(b0, b1),  "multi-blocs, pas 0 : bloc b (stride=3) est TENU")

# macro_step 1 : a avance ; b tenu ((1+1)%3 != 0)
a1_, b1_ = sim_mb.density("a").copy(), sim_mb.density("b").copy()
sim_mb.step(dt)
a2, b2 = sim_mb.density("a").copy(), sim_mb.density("b").copy()
chk(changed(a1_, a2), "multi-blocs, pas 1 : bloc a (stride=1) avance")
chk(frozen(b1_, b2),  "multi-blocs, pas 1 : bloc b (stride=3) est TENU")

# macro_step 2 : a avance ; b RATTRAPE ((2+1)%3 == 0)
a2_, b2_ = sim_mb.density("a").copy(), sim_mb.density("b").copy()
sim_mb.step(dt)
a3, b3 = sim_mb.density("a").copy(), sim_mb.density("b").copy()
chk(changed(a2_, a3), "multi-blocs, pas 2 : bloc a (stride=1) avance")
chk(changed(b2_, b3), "multi-blocs, pas 2 : bloc b (stride=3) RATTRAPE")

# macro_step 3 : a avance ; b tenu (nouvelle fenetre)
a3_, b3_ = sim_mb.density("a").copy(), sim_mb.density("b").copy()
sim_mb.step(dt)
a4, b4 = sim_mb.density("a").copy(), sim_mb.density("b").copy()
chk(changed(a3_, a4), "multi-blocs, pas 3 : bloc a (stride=1) avance")
chk(frozen(b3_, b4),  "multi-blocs, pas 3 : bloc b (stride=3) est TENU")

# ---- 7. adc.IMEX avec stride (hold-then-catch-up) ------------------------------
# Cadence M=2 : rattrapage quand (macro_step+1)%2==0 -> macro_step 1, 3, ...
print("== IMEX avec stride (hold-then-catch-up) ==")
n = 32
rho_e = 1.0 + 0.02 * np.cos(2 * np.pi * meshx(n))[None, :] * np.ones((n, 1))

sim_imex = adc.System(n=n, periodic=False)
sim_imex.add_block("ne", diocotron_model(), time=adc.IMEX(substeps=2, stride=2))
sim_imex.set_poisson(bc="dirichlet")
sim_imex.set_density("ne", rho_e)

state_before = sim_imex.density("ne").copy()
sim_imex.step(dt)  # macro_step 0 : (0+1)%2 != 0 -> TENU
state_after_0 = sim_imex.density("ne").copy()
chk(frozen(state_before, state_after_0),
    "IMEX stride=2 substeps=2, pas 0 : le bloc est TENU")

state_before_1 = sim_imex.density("ne").copy()
sim_imex.step(dt)  # macro_step 1 : (1+1)%2 == 0 -> RATTRAPAGE
state_after_1 = sim_imex.density("ne").copy()
chk(changed(state_before_1, state_after_1),
    "IMEX stride=2 substeps=2, pas 1 : le bloc RATTRAPE")

# ---- 8. AOT : Explicit(stride>1) + backend='aot' leve une erreur claire --------
# Le bloc compile AOT (.so) ne transporte PAS la cadence dans son ABI extern "C" : System.add_equation
# doit REJETER stride > 1 plutot que de tourner a stride=1 en silence. La garde est purement Python et
# leve AVANT le dlopen du .so : un CompiledModel FACTICE (backend='aot', .so inexistant) suffit, donc
# le sous-test ne depend PAS d'un compilateur (deterministe en CI minimale).
print("== AOT : stride>1 + backend='aot' rejete explicitement ==")
fake_aot = adc.dsl.CompiledModel(
    so_path="/inexistant.so", backend="aot", adder="add_compiled_block",
    cons_names=["rho", "rho_u", "rho_v", "E"],
    cons_roles=["Density", "MomentumX", "MomentumY", "Energy"],
    prim_names=["rho", "u", "v", "p"], n_vars=4, gamma=1.4, n_aux=3, params={}, caps={},
    abi_key="k", model_hash="h", cxx="c++", std="c++20")

sim_aot = adc.System(n=16, periodic=True)
try:
    sim_aot.add_equation("gas", fake_aot, spatial=adc.FiniteVolume(), time=adc.Explicit(stride=2))
    chk(False, "add_equation(stride=2, backend='aot') doit lever ValueError")
except ValueError as ex:
    chk("stride" in str(ex) and "aot" in str(ex),
        "add_equation(stride=2, backend='aot') leve une ValueError claire (stride/aot)")

# stride override via add_equation(stride=) AUSSI rejete (couvre les deux sources de cadence).
sim_aot2 = adc.System(n=16, periodic=True)
try:
    sim_aot2.add_equation("gas", fake_aot, spatial=adc.FiniteVolume(), stride=3)
    chk(False, "add_equation(stride=3 override, backend='aot') doit lever ValueError")
except ValueError as ex:
    chk("stride" in str(ex) and "aot" in str(ex),
        "add_equation(stride= override, backend='aot') leve une ValueError claire")

# stride=1 (defaut) : la garde stride NE doit PAS lever (le .so inexistant echoue plus loin au dlopen,
# RuntimeError) -- on verifie juste que ce n'est PAS la ValueError de stride.
sim_aot_ok = adc.System(n=16, periodic=True)
try:
    sim_aot_ok.add_equation("gas", fake_aot, spatial=adc.FiniteVolume(), time=adc.Explicit(stride=1))
    chk(False, "add_equation(stride=1, backend='aot') : attendu un echec au dlopen (.so inexistant)")
except ValueError as ex:
    chk(False, "add_equation(stride=1, backend='aot') ne doit PAS lever de ValueError stride (%s)" % ex)
except Exception:  # noqa: BLE001  RuntimeError du dlopen attendu : la garde stride a laisse passer
    chk(True, "add_equation(stride=1, backend='aot') : pas de rejet stride (echec au dlopen attendu)")

# ---- 9. Validation des entrees ------------------------------------------------
print("== validation des entrees ==")
try:
    adc.Explicit(stride=0)
    chk(False, "Explicit(stride=0) doit lever ValueError")
except ValueError:
    chk(True, "Explicit(stride=0) leve ValueError")

try:
    adc.IMEX(stride=-1)
    chk(False, "IMEX(stride=-1) doit lever ValueError")
except ValueError:
    chk(True, "IMEX(stride=-1) leve ValueError")

# ---- Bilan -------------------------------------------------------------------
print()
if fails == 0:
    print("OK test_stride (%d assertions)" % (
        sum(1 for line in open(__file__) if line.strip().startswith("chk("))))
else:
    print("ECHEC test_stride : %d assertion(s) en erreur" % fails)
    sys.exit(1)
