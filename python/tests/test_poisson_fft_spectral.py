#!/usr/bin/env python3
"""Poisson FFT a symbole spectral continu (solver='fft_spectral') -- ADC-175.

Le kind 'fft' historique diagonalise le stencil 5-points DISCRET (meme operateur que le MG,
exact a l'arrondi POUR CE STENCIL). 'fft_spectral' divise par le symbole CONTINU
-(kx^2 + ky^2) (frequences signees [0..N/2-1, -N/2..-1] * 2pi/L) : la convention des
references spectrales (poisson_fft.m de RIEMOM2D). Les deux ne different que par O(h^2).

L'oracle sinusoide rho = eps cos(2 pi x) -- a MOYENNE NULLE : les solveurs fft annulent le
mode k = 0 implicitement, le MG periodique exige un rhs deflate (un rhs a moyenne non nulle
le rend singulier, piege documente du diocotron) ; la moyenne nulle met les trois solveurs
sur le MEME probleme bien pose -- DISCRIMINE les trois solveurs : la solution continue
phi = -eps cos(2 pi x)/(2 pi)^2 est atteinte a ~1e-12 par le spectral (exact sur les
sinusoides) alors que fft discret et geometric_mg en different par (k dx)^2/12 ~ 3e-3
relatif a n = 32 -- la MEME mesure prouve le nouveau chemin ET le no-default-change.

On verifie :
 (1) spectral : || phi - phi_exact || / eps' < 1e-12 ; moyenne de phi nulle ;
 (2) fft discret et geometric_mg : ecart relatif au continu dans [1e-4, 1e-2] (fenetre
     O(h^2) attendue) et fft == MG a la tolerance MG pres (meme operateur) ;
 (3) convergence du spectral vs continu independante de n (n = 16 et 32 : toujours machine) ;
 (4) rejets : paroi refusee (message au kind effectif) ; solver inconnu liste fft_spectral ;
 (5) GHOSTS de phi (bug latent corrige ici) : la derivation grad phi de solve_fields lit les
     voisins, donc les ghosts du bord ; le solveur fft direct ne les remplissait pas (le MG
     les remplit en lissant). Sonde par le chemin SOURCE complet : eval_rhs(fft) ==
     eval_rhs(geometric_mg) au residu MG pres -- avant le fix, l'ecart au bord etait O(1).

Modele natif (isotherme + ChargeDensity) : aucun compilateur requis.
"""
import sys

import numpy as np

import pops

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def err_msg(fn):
    try:
        fn(); return ""
    except Exception as ex:  # noqa: BLE001
        return str(ex)


def solve_phi(n, solver, eps=1e-3):
    """System periodique, rho = eps cos(2 pi x) (moyenne nulle) -> phi par le solveur
    demande. Une densite negative est sans objet ici : seul solve_fields est appele."""
    sim = pops.System(n=n, L=1.0, periodic=True)
    sim.add_block("ions",
                  pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                            transport=pops.IsothermalFlux(),
                            source=pops.PotentialForce(charge=1.0),
                            elliptic=pops.ChargeDensity(charge=1.0)),
                  spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                  time=pops.Explicit())
    sim.set_poisson(rhs="charge_density", solver=solver, bc="periodic")
    x = (np.arange(n) + 0.5) / n
    rho = eps * np.cos(2.0 * np.pi * x)[None, :] * np.ones((n, n))
    sim.set_state("ions", np.stack([rho, np.zeros_like(rho), np.zeros_like(rho)]))
    sim.solve_fields()
    phi = np.array(sim.potential())
    phi_ex = -(eps * np.cos(2.0 * np.pi * x) / (2.0 * np.pi) ** 2)[None, :] * np.ones((n, n))
    scale = eps / (2.0 * np.pi) ** 2
    return np.abs(phi - phi_ex).max() / scale, abs(phi.mean()) / scale, phi


print("== (1) spectral == solution continue a la precision machine ==")
e_sp, m_sp, phi_sp = solve_phi(32, "fft_spectral")
chk(e_sp < 1e-12, f"fft_spectral vs analytique : err rel {e_sp:.2e} < 1e-12")
chk(m_sp < 1e-12, f"moyenne de phi nulle ({m_sp:.2e})")

print("== (2) fft discret / geometric_mg : fenetre O(h^2), no-default-change ==")
e_fft, _, phi_fft = solve_phi(32, "fft")
e_mg, _, phi_mg = solve_phi(32, "geometric_mg")
chk(1e-4 < e_fft < 1e-2, f"fft discret vs analytique : {e_fft:.2e} dans [1e-4, 1e-2] (O(h^2))")
chk(1e-4 < e_mg < 1e-2, f"geometric_mg vs analytique : {e_mg:.2e} dans [1e-4, 1e-2] (O(h^2))")
d = np.abs(phi_fft - phi_mg).max() / (1e-3 / (2.0 * np.pi) ** 2)
chk(d < 1e-5, f"fft == MG au residu MG pres ({d:.2e}) : meme operateur discret")
chk(e_sp < e_fft / 100, f"discrimination : spectral {e_sp:.1e} << discret {e_fft:.1e}")

print("== (3) spectral exact sur sinusoide independamment de n ==")
e16, _, _ = solve_phi(16, "fft_spectral")
chk(e16 < 1e-12, f"n=16 : err rel {e16:.2e} < 1e-12 (pas de terme O(h^2))")

print("== (4) rejets ==")
sim = pops.System(n=32, L=1.0, periodic=False)
sim.add_block("ions",
              pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                        transport=pops.IsothermalFlux(),
                        source=pops.PotentialForce(charge=1.0),
                        elliptic=pops.ChargeDensity(charge=1.0)),
              spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
              time=pops.Explicit())
sim.set_poisson(rhs="charge_density", solver="fft_spectral", bc="dirichlet",
                wall="circle", wall_radius=0.4)
msg = err_msg(sim.solve_fields)
chk("fft_spectral" in msg and "wall" in msg, f"paroi refusee au kind effectif ({msg[:60]}...)")
sim2 = pops.System(n=32, L=1.0, periodic=True)
sim2.add_block("ions",
               pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                         transport=pops.IsothermalFlux(),
                         source=pops.PotentialForce(charge=1.0),
                         elliptic=pops.ChargeDensity(charge=1.0)),
               spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
               time=pops.Explicit())
sim2.set_poisson(rhs="charge_density", solver="dct", bc="periodic")
msg = err_msg(sim2.solve_fields)
chk("fft_spectral" in msg, f"solver inconnu : la liste inclut fft_spectral ({msg[:60]}...)")

print("== (5) ghosts de phi : le chemin source complet fft == MG ==")


def rhs_with(solver):
    n = 32
    sim = pops.System(n=n, L=1.0, periodic=True)
    sim.add_block("ions",
                  pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                            transport=pops.IsothermalFlux(),
                            source=pops.PotentialForce(charge=1.0),
                            elliptic=pops.ChargeDensity(charge=1.0)),
                  spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                  time=pops.Explicit())
    sim.set_poisson(rhs="charge_density", solver=solver, bc="periodic")
    x = (np.arange(n) + 0.5) / n
    rho = 1e-3 * np.cos(2.0 * np.pi * x)[None, :] * np.ones((n, n)) + 1e-3 * np.sin(
        2.0 * np.pi * x)[:, None] * np.ones((n, n))
    sim.set_state("ions", np.stack([1.0 + rho, np.zeros_like(rho), np.zeros_like(rho)]))
    sim.solve_fields()
    return np.array(sim.eval_rhs("ions"))


for solver in ("fft", "fft_spectral"):
    dr = np.abs(rhs_with(solver) - rhs_with("geometric_mg")).max()
    tol = 1e-7 if solver == "fft" else 1e-4  # spectral != stencil discret : O(h^2) sur grad
    chk(dr < tol, f"eval_rhs({solver}) == eval_rhs(MG) a {dr:.2e} (< {tol:.0e}) : "
                  "les ghosts de phi sont remplis (source au bord saine)")

print("FAILS =", fails)
sys.exit(1 if fails else 0)
