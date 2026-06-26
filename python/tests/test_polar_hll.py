"""Chantier POLAIRE (audit 2026-06, section 3) : flux HLL cable sur l'anneau pour le fluide
isotherme polaire (IsothermalFluxPolar).

CE QUE VERROUILLE CE TEST :
  T1 - DEFAUT BIT-IDENTIQUE : un run polaire isotherme avec riemann='rusanov' (le defaut) est
       reproductible (deux constructions identiques -> etat bit-identique). On ne PROUVE pas ici la
       non-regression vis-a-vis d'avant le patch (impossible dans un seul process), mais le patch ne
       touche PAS la branche rusanov de make_block_polar (ajout d'une branche 'hll' SEPAREE) : le
       defaut reste strictement l'historique.
  T2 - HLL TOURNE FINI : le meme run avec riemann='hll' avance sans NaN/Inf (le flux signe
       assemble_rhs_polar<Limiter, HLLFlux> est device-clean, REUTILISE verbatim depuis le cartesien).
  T3 - HLL DIFFERE DE RUSANOV : HLL est moins diffusif que Rusanov (dissipation ~ |sR - sL| signee au
       lieu de 2 max|v| symetrique) -> l'etat final differe au-dela du bruit FP. C'est la preuve que
       le flux injecte est REELLEMENT HLL (et non un alias silencieux de Rusanov).

Le fluide isotherme polaire expose model.wave_speeds (herite d'IsothermalFlux) : c'est la condition
du gate 'hll' (identique au cartesien block_builder.hpp). Un transport ExB SCALAIRE ne la fournit pas
-> rejet, couvert par test_polar_rejections.test_polar_rejects_hll_on_scalar_exb.
"""
import math

import numpy as np

import pops

RMIN, RMAX = 0.3, 1.0


def iso_polar_model(cs2=1.0):
    """Fluide isotherme NATIF : sur l'anneau, le dispatch polaire (block_builder_polar) instancie
    IsothermalFluxPolar (roles Density / MomentumX (radial) / MomentumY (azimutal)). Second membre
    elliptique neutre (alpha=0) : on isole le TRANSPORT (le flux Riemann), pas le couplage Poisson."""
    return pops.Model(
        state=pops.FluidState(kind="isothermal", cs2=cs2),
        transport=pops.IsothermalFlux(),
        source=pops.NoSource(),
        elliptic=pops.BackgroundDensity(alpha=0.0, n0=0.0),
    )


def _annular_state(nr, nth):
    """Etat conservatif (3, ntheta, nr) = (rho, mom_r, mom_theta) lisse, > 0, avec une vitesse
    initiale module en theta (gradient azimutal -> le flux travaille). Layout set_state : composante
    lente, puis j=theta, puis i=r (numpy (ncomp, ntheta, nr) aplati en C-order), cf.
    test_polar_schur_via_system._initial_velocity_state."""
    dr = (RMAX - RMIN) / nr
    dth = 2.0 * math.pi / nth
    rho = np.empty((nth, nr), dtype=np.float64)
    mr = np.empty((nth, nr), dtype=np.float64)
    mth = np.empty((nth, nr), dtype=np.float64)
    for j in range(nth):
        th = (j + 0.5) * dth
        for i in range(nr):
            r = RMIN + (i + 0.5) * dr
            rr = (r - RMIN) / (RMAX - RMIN)
            h = math.sin(math.pi * rr)  # 0 aux bords radiaux (compatible paroi)
            rho[j, i] = 1.5 + 0.3 * math.cos(2.0 * th) * h
            vr = 0.5 * h * math.cos(2.0 * th)
            vth = -0.4 * h * math.sin(th)
            mr[j, i] = rho[j, i] * vr
            mth[j, i] = rho[j, i] * vth
    return np.stack([rho, mr, mth], axis=0)  # (3, ntheta, nr)


def _build(nr, nth, riemann, cs2=1.0):
    """System polaire isotherme avec le flux Riemann demande, etat initial pose, pret a stepper."""
    sim = pops.System(mesh=pops.PolarMesh(r_min=RMIN, r_max=RMAX, nr=nr, ntheta=nth))
    sim.set_poisson(rhs="charge_density", solver="polar", bc="dirichlet")
    sim.add_equation(
        "ions",
        model=iso_polar_model(cs2=cs2),
        spatial=pops.FiniteVolume(limiter="minmod", riemann=riemann, variables="conservative"),
        time=pops.Explicit(),
    )
    u0 = _annular_state(nr, nth)
    sim.set_density("ions", u0[0].ravel())     # pose rho (vitesse au repos)
    sim.set_state("ions", u0.ravel())          # injecte la vitesse initiale (mom_r, mom_theta)
    return sim


def _state3(sim, nr, nth):
    return np.array(sim.get_state("ions")).reshape(3, nth, nr)


def _run(sim, nr, nth, n_steps, dt):
    for _ in range(n_steps):
        sim.step(dt)
    return _state3(sim, nr, nth)


def test_polar_hll():
    nr, nth = 24, 24
    cs2 = 1.0
    h = min((RMAX - RMIN) / nr, RMIN * (2.0 * math.pi / nth))  # pas physique min polaire
    dt = 0.2 * h / math.sqrt(cs2)
    n_steps = 8

    # T1 : defaut rusanov reproductible (deux constructions identiques -> bit-identique).
    s_rus_a = _run(_build(nr, nth, "rusanov", cs2), nr, nth, n_steps, dt)
    s_rus_b = _run(_build(nr, nth, "rusanov", cs2), nr, nth, n_steps, dt)
    assert np.array_equal(s_rus_a, s_rus_b), "rusanov polaire : non reproductible (T1)"

    # T2 : hll tourne fini.
    s_hll = _run(_build(nr, nth, "hll", cs2), nr, nth, n_steps, dt)
    assert np.all(np.isfinite(s_hll)), "hll polaire : etat non fini (T2)"

    # T3 : hll differe de rusanov (au-dela du bruit FP) -> le flux injecte est bien HLL.
    diff = float(np.max(np.abs(s_hll - s_rus_a)))
    assert diff > 1e-8, (
        "hll polaire ne differe pas de rusanov (diff=%.3e) : le flux injecte serait un alias "
        "silencieux de Rusanov (T3)" % diff
    )


if __name__ == "__main__":
    test_polar_hll()
    print("test_polar_hll : OK (rusanov reproductible, hll fini et distinct)")
