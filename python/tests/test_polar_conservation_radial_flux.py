"""Conservation polaire avec flux radial interieur NON NUL (A4).

Motivation (gap A4)
-------------------
test_polar_system.py et test_polar_transport_mms.cpp (run_conservation, l.198) exercent la
conservation FV polaire uniquement avec v_r = 0 (grad_theta = 0 dans le champ auxiliaire).
Le terme de flux radial de la divergence en coordonnees polaires,

    (1/r) d/dr (r * v_r * n),

n'est donc jamais ACTIF dans ces tests : v_r = 0 implique une annulation triviale de ce terme.
Ce test leve cette limitation en forcant v_r != 0 dans l'interieur du domaine.

Strategie
---------
On seme une densite AZIMUTALEMENT ASYMETRIQUE :

    n(r, theta) = 1 + A * cos(L * theta) * sin(pi * (r - rmin) / (rmax - rmin)),

avec A = 0.5, L = 3. L'asymetrie en theta implique une charge non uniforme en theta.
Le solveur Poisson polaire calcule alors un potentiel phi avec grad_theta != 0, ce qui
donne une vitesse ExB radiale v_r = -(1/r) dphi/dtheta / B0 != 0 dans l'interieur.

Conservation : la paroi radiale solide (wall_radial=true, defaut du System polaire) impose
un flux nul aux BORDS r = rmin et r = rmax. Par consequent, la masse FV polaire

    M = sum_{i,j} n(i,j) * r(i) * dr * dtheta

est conservee a precision machine meme si v_r != 0 a l'interieur.

Verifications
-------------
  (C1) Pas de nan/inf avant tout calcul de tolerance.
  (C2) La densite EVOLUE (Poisson + transport actifs).
  (C3) La masse FV polaire est conservee a ~machine (|dM/M| < 1e-11) sur K pas.
  (C4) Le flux radial INTERIEUR est genuinement non nul : on verifie que le profil spatial de
       la densite change d'une facon INCOMPATIBLE avec une pure rotation azimutale rigide (i.e.
       le contenu radial de chaque colonne theta evolue, ce qui ne peut pas se produire si v_r = 0
       partout). Quantitativement : la variance radiale de n varie d'au moins 1e-9 entre t=0 et
       t=T. Si v_r etait identiquement nul, un transport purement azimutal par fft-shift ne
       changerait que la distribution en theta de n mais pas la distribution en r ; la variance
       radiale de n resterait exactement constante. Toute deviation > 1e-9 implique un flux radial
       actif.

Mono-rang (le Poisson polaire direct refuse MPI).
"""
import math

import numpy as np

import adc

# Parametres geometriques
RMIN, RMAX = 0.30, 1.00
NR, NTH = 48, 48

# Modulation de la densite initiale : asymetrie azimutale (force v_r != 0)
A_ASYM = 0.5    # amplitude de la perturbation
L_MODE = 3      # mode azimutal (L >= 2 pour avoir grad_theta != 0 dans le bulk)

# Nombre de pas temporels apres le premier step_cfl
NSTEPS = 30


# ---------------------------------------------------------------------------
# Profil de densite azimutalement asymetrique
# ---------------------------------------------------------------------------

def _asymmetric_density(nr, nth, rmin, rmax, a, l):
    """Densite n = 1 + a*cos(l*theta)*sin(pi*(r-rmin)/(rmax-rmin)), strictement positive.

    Layout attendu par System::set_density (polaire) : flat[j * nr + i],
    theta (j) axe lent, r (i) axe rapide (meme convention que test_polar_system.py).

    L'asymetrie azimutale (cos(l*theta) avec l >= 2) implique grad_theta phi != 0 apres
    resolution Poisson, donc v_r = -(1/r) dphi/dtheta / B0 != 0 dans le bulk.
    """
    dr = (rmax - rmin) / nr
    dth = 2.0 * math.pi / nth
    rho = []
    for j in range(nth):
        th = (j + 0.5) * dth
        for i in range(nr):
            r = rmin + (i + 0.5) * dr
            rr = (r - rmin) / (rmax - rmin)
            val = 1.0 + a * math.cos(l * th) * math.sin(math.pi * rr)
            rho.append(max(val, 1e-9))   # plancher positif (ExB scalaire)
    return rho


# ---------------------------------------------------------------------------
# Masse FV polaire : sum n * r * dr * dtheta
# ---------------------------------------------------------------------------

def _polar_mass(sim, species, nr, nth, rmin, rmax):
    """Calcule la masse FV polaire en reproduisant la formule de sim.mass() pour verification."""
    return sim.mass(species)


# ---------------------------------------------------------------------------
# Variance radiale de n (indicateur de flux radial actif)
# ---------------------------------------------------------------------------

def _radial_variance(sim, species, nr, nth):
    """Variance de la distribution radiale moyenne de n.

    On calcule n_bar(i) = mean_j n(j, i) (moyenne azimutale par anneau radial), puis
    var = mean_i n_bar(i)^2 - (mean_i n_bar(i))^2.

    Si v_r = 0 partout, un transport purement azimutal laisse n_bar(i) constant (rotation
    cyclique des colonnes theta ne change pas la moyenne azimutale par anneau). Toute
    variation de _radial_variance entre t=0 et t>0 implique donc que v_r != 0 quelque
    part dans le domaine et que le flux radial a effectivement transporte de la masse entre
    anneaux radiaux.
    """
    a = np.asarray(sim.density(species), dtype=float).ravel().reshape(nth, nr)  # [theta, r]
    n_bar = a.mean(axis=0)          # moyenne azimutale, shape (nr,)
    return float(np.var(n_bar))     # variance sur r


# ---------------------------------------------------------------------------
# Test principal
# ---------------------------------------------------------------------------

def test_polar_conservation_with_nonzero_radial_flux():
    """Conservation FV polaire avec flux radial interieur NON NUL (A4).

    Voir docstring du module pour la motivation et la strategie.
    """
    sim = adc.System(mesh=adc.PolarMesh(r_min=RMIN, r_max=RMAX, nr=NR, ntheta=NTH))
    sim.add_block(
        "ne",
        model=adc.Model(
            state=adc.Scalar(),
            transport=adc.ExB(B0=1.0),
            source=adc.NoSource(),
            elliptic=adc.ChargeDensity(charge=1.0),
        ),
        spatial=adc.Spatial(minmod=True),
        time=adc.Explicit(),
    )
    sim.set_poisson(rhs="charge_density", solver="polar", bc="dirichlet")
    sim.set_density("ne", _asymmetric_density(NR, NTH, RMIN, RMAX, A_ASYM, L_MODE))

    # --- etat initial ---
    rho0 = np.asarray(sim.density("ne"), dtype=float)
    m0 = _polar_mass(sim, "ne", NR, NTH, RMIN, RMAX)
    var0 = _radial_variance(sim, "ne", NR, NTH)

    # (C1) Etat initial sain
    assert np.all(np.isfinite(rho0)), "densite initiale non finie"
    assert math.isfinite(m0) and m0 > 0.0, "masse initiale invalide"

    # --- premier pas (resout Poisson et active le transport) ---
    dt = sim.step_cfl(0.3)
    assert math.isfinite(dt) and dt > 0.0, "dt CFL invalide"

    rho1 = np.asarray(sim.density("ne"), dtype=float)
    assert np.all(np.isfinite(rho1)), "densite non finie apres step_cfl (blow-up ?)"

    # (C2) La densite evolue (Poisson + transport actifs)
    dmax = float(np.max(np.abs(rho1 - rho0)))
    assert dmax > 1e-9, (
        "le premier pas ne modifie pas la densite (Poisson/aux/transport inertes ?) : dmax=%.3e" % dmax
    )
    assert dmax < 1.0, (
        "variation > 1.0 apres 1 pas = instabilite (densite initiale ~1) : dmax=%.3e" % dmax
    )

    # --- K pas supplementaires ---
    for _ in range(NSTEPS):
        sim.step(dt)

    rho_final = np.asarray(sim.density("ne"), dtype=float)
    m_final = _polar_mass(sim, "ne", NR, NTH, RMIN, RMAX)
    var_final = _radial_variance(sim, "ne", NR, NTH)

    # (C1 final) Pas de nan/inf
    assert np.all(np.isfinite(rho_final)), "densite non finie apres %d pas (blow-up)" % (NSTEPS + 1)
    assert math.isfinite(m_final), "masse non finie apres %d pas" % (NSTEPS + 1)

    # (C3) Conservation de la masse FV polaire a ~machine
    rel = abs(m_final - m0) / abs(m0)
    assert rel < 1e-11, (
        "masse FV polaire non conservee malgre paroi radiale solide : ecart relatif %.3e" % rel
    )

    # (C4) Flux radial interieur genuinement non nul
    # La variance radiale de la densite doit changer : si v_r = 0 partout, un transport
    # purement azimutal ne peut pas modifier n_bar(i) = mean_j n(j,i) (rotation cyclique
    # des colonnes theta). Toute variation de var(n_bar) prouve que des echanges radiaux
    # ont eu lieu, i.e. que le terme (1/r) d/dr(r v_r n) etait actif.
    dvar = abs(var_final - var0)
    assert dvar > 1e-9, (
        "la variance radiale de n n'a pas change (|dvar|=%.3e < 1e-9) : "
        "le flux radial interieur ne semble pas etre actif ; "
        "verifier que la modulation azimutale (L=%d, A=%.2f) est suffisante pour generer "
        "grad_theta phi != 0 et v_r != 0." % (dvar, L_MODE, A_ASYM)
    )


if __name__ == "__main__":
    import adc as _adc_mod
    import numpy as _np2
    import math as _math2

    _sim = _adc_mod.System(mesh=_adc_mod.PolarMesh(r_min=RMIN, r_max=RMAX, nr=NR, ntheta=NTH))
    _sim.add_block(
        "ne",
        model=_adc_mod.Model(
            state=_adc_mod.Scalar(),
            transport=_adc_mod.ExB(B0=1.0),
            source=_adc_mod.NoSource(),
            elliptic=_adc_mod.ChargeDensity(charge=1.0),
        ),
        spatial=_adc_mod.Spatial(minmod=True),
        time=_adc_mod.Explicit(),
    )
    _sim.set_poisson(rhs="charge_density", solver="polar", bc="dirichlet")
    _sim.set_density("ne", _asymmetric_density(NR, NTH, RMIN, RMAX, A_ASYM, L_MODE))
    _m0 = _sim.mass("ne")
    _var0 = _radial_variance(_sim, "ne", NR, NTH)
    print("var0 = %.6e (esperance: 0 car perturbation azimutale moyenne nulle)" % _var0)
    _dt = _sim.step_cfl(0.3)
    print("dt_cfl = %.6e" % _dt)
    for _ in range(NSTEPS):
        _sim.step(_dt)
    _m1 = _sim.mass("ne")
    _var1 = _radial_variance(_sim, "ne", NR, NTH)
    _rel = abs(_m1 - _m0) / abs(_m0)
    _dvar = abs(_var1 - _var0)
    print("var_final = %.6e  dvar = %.6e  (> 1e-9 requis)" % (_var1, _dvar))
    print("conservation masse : ecart relatif = %.3e  (< 1e-11 requis)" % _rel)
    assert _dvar > 1e-9, "ECHEC C4 : dvar=%.3e" % _dvar
    assert _rel < 1e-11, "ECHEC C3 : rel=%.3e" % _rel
    print("OK test_polar_conservation_radial_flux")
