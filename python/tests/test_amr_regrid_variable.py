"""ADC-296 : critere de regrid AMR configurable par nom de variable ou role physique, au-dela de la
composante 0 (cote Python / pybind). set_refinement(threshold, variable=, role=) resout le selecteur
PAR BLOC contre les variables conservatives du bloc (STRICT : un bloc sans le nom/role leve, jamais de
repli silencieux vers la composante 0) ; defaut (selecteur vide) = composante 0, bit-identique.

On monte deux blocs Euler compressibles (4 var : rho, rho_u, rho_v, E) sur la facade AmrSystem, on seede
le bloc 0 avec une DENSITE uniforme et une bosse d'ENERGIE (composante 3) en bas-gauche, et on verifie
que refiner sur l'energie deplace le patch fin vers la bosse, la ou le selecteur par defaut (densite
uniforme < seuil) garde le seed central : les layouts DIFFERENT, preuve que la composante lue a change.

Chemin natif (pops.Model) : aucun compilateur requis (cf. test_amr_conservative_state gardes pre-build).
"""
import numpy as np

import pops

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def raises(fn):
    try:
        fn()
        return False
    except Exception:
        return True


# Euler compressible pur (sans source), elliptique de fond trivial (alpha=0) : Poisson a second membre
# nul -> aucune contrainte de solvabilite periodique (le regrid tague sur le champ conservatif).
def _comp():
    return pops.Model(state=pops.FluidState("compressible", gamma=1.4),
                     transport=pops.CompressibleFlux(), source=pops.NoSource(),
                     elliptic=pops.BackgroundDensity(alpha=0.0, n0=0.0))


# Etat conservatif (rho, rho_u, rho_v, E), shape (4, n, n) attendu par set_conservative_state. Au repos
# (mom=0) ; la composante bump_comp est portee a bump_val dans la boite bas-gauche [lo, hi)^2 (hors du
# seed central [n/4, 3n/4)).
def _state(n, rho, E, bump_comp, bump_val, lo, hi):
    comps = [np.full((n, n), rho), np.zeros((n, n)), np.zeros((n, n)), np.full((n, n), E)]
    comps[bump_comp][lo:hi, lo:hi] = bump_val
    return np.stack(comps)


def _min_fine_corner(boxes):
    # boxes : liste de tuples (level, ilo, jlo, ihi, jhi). Plus petit coin sur les boites FINES.
    corners = [min(b[1], b[2]) for b in boxes if b[0] >= 1]
    return min(corners) if corners else 1 << 30


def _run(n, thr, variable, role, s0):
    sim = pops.AmrSystem(n=n, L=1.0, periodic=True, regrid_every=1)
    sim.add_block("gas0", _comp(), time=pops.Explicit())
    sim.add_block("gas1", _comp(), time=pops.Explicit())
    sim.set_poisson(bc="periodic")
    sim.set_refinement(thr, variable=variable, role=role)
    sim.set_conservative_state("gas0", s0)
    sim.set_conservative_state("gas1", _state(n, 1.0, 2.0, 0, 1.0, 0, 0))  # uniforme
    for _ in range(4):
        sim.step(1e-3)
    return sim.patch_boxes()


N = 64
# Bosse d'ENERGIE bas-gauche (boite [4, 20)^2, hors du seed central [16, 48)) ; densite UNIFORME (=1).
s_energy = _state(N, 1.0, 2.0, bump_comp=3, bump_val=12.0, lo=4, hi=20)

print("== (1) selecteur energie/nom suit la composante 3 ==")
default = _run(N, 6.0, "", "", s_energy)        # defaut comp 0 (densite=1 < 6 -> aucun tag)
by_role = _run(N, 6.0, "", "energy", s_energy)  # role energie -> comp 3
by_name = _run(N, 6.0, "E", "", s_energy)       # nom "E" -> comp 3

chk(_min_fine_corner(by_role) < _min_fine_corner(default), "role=energy : patch atteint le bas-gauche")
chk(_min_fine_corner(by_name) < _min_fine_corner(default), "variable=E : patch atteint le bas-gauche")
chk(by_role != default, "role=energy : layout different du defaut")
chk(by_name != default, "variable=E : layout different du defaut")

print("== (2) non-regression composante 0 : le defaut raffine encore la densite ==")
s_density = _state(N, 1.0, 2.0, bump_comp=0, bump_val=3.0, lo=4, hi=20)
dens_default = _run(N, 2.0, "", "", s_density)
chk(_min_fine_corner(dens_default) < 32, "defaut : raffine la bosse de densite (comp 0)")

print("== (3) erreurs strictes ==")
chk(raises(lambda: _run(N, 6.0, "", "temperature", s_energy)),
    "role absent du bloc -> leve au build (pas de repli comp 0)")


def _solo_with_selector():
    # Bloc UNIQUE + selecteur : le selecteur n'est cable que sur le moteur multi-blocs ; refuse au
    # build (pas de repli silencieux vers la composante 0).
    sim = pops.AmrSystem(n=N, L=1.0, periodic=True, regrid_every=1)
    sim.add_block("solo", _comp(), time=pops.Explicit())
    sim.set_poisson(bc="periodic")
    sim.set_refinement(6.0, role="energy")
    sim.set_density("solo", np.full((N, N), 1.0))
    sim.n_patches()  # declenche le build -> refus


chk(raises(_solo_with_selector), "bloc unique + selecteur -> leve au build (multi-blocs only)")
chk(raises(lambda: pops.AmrSystem(n=N, L=1.0, periodic=True, regrid_every=1)
           .set_refinement(6.0, variable="E", role="energy")),
    "variable ET role -> leve immediatement")

if fails == 0:
    print("test_amr_regrid_variable : OK")
import sys  # noqa: E402
sys.exit(1 if fails else 0)
