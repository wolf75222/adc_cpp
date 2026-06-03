"""Second membre du Poisson de systeme GENERIQUE : f = somme_s elliptic_rhs_s(u_s).

Le second membre n'est PAS cable en dur sur la densite de charge (q n) : c'est la SOMME des briques
elliptiques portees par les blocs. Chaque bloc choisit sa brique via Model(elliptic=...) :
  - ChargeDensity     f = q n
  - BackgroundDensity f = alpha (n - n0)
  - GravityCoupling   f = sign 4piG (rho - rho0)
On le prouve sans toucher au solveur : pour le MEME operateur div(grad), poser une brique non-charge
sur un bloc revient EXACTEMENT a poser une brique de charge unite dont la densite vaut le second
membre de la brique. phi doit coincider a la precision du solveur.

Tests :
1. BackgroundDensity (non-charge) : f = alpha (n - n0). phi == reference charge unite f.
2. Bricks MIXTES sur deux blocs : f = q0 n0 + alpha (n1 - n0bg). phi == reference somme manuelle.
3. Facade EPM : add_elliptic_model(rhs=composite_rhs()) == set_poisson(rhs="composite").
4. Le token "composite" cote set_poisson est accepte (alias honnete de "charge_density").
"""
import numpy as np

import adc

PI = np.pi


def _scalar(elliptic):
    """Bloc scalaire (1 var) transporte par ExB : set_density n'ecrit que la densite (comp 0), ce
    qui isole le second membre du Poisson pour la brique elliptique choisie."""
    return adc.Model(state=adc.Scalar(), transport=adc.ExB(B0=1.0),
                     source=adc.NoSource(), elliptic=elliptic)


def _density(n):
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    return 1.0 + 0.3 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)


def _solve_single(elliptic, dens, n, rhs="composite"):
    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.add_block("blk", model=_scalar(elliptic), spatial=adc.Spatial(none=True))
    sim.set_poisson(rhs=rhs, solver="fft")
    sim.set_density("blk", dens.reshape(-1).tolist())
    sim.solve_fields()
    return np.array(sim.potential()).reshape(n, n)


def background_is_generic():
    """Brique BackgroundDensity : f = alpha (n - n0), distincte de q n. phi == reference charge."""
    n = 64
    alpha, n0 = 1.7, 0.4
    dens = _density(n)

    phi_bg = _solve_single(adc.BackgroundDensity(alpha=alpha, n0=n0), dens, n)

    # Reference : un bloc de charge unite (f = q n = n) dont la densite vaut DIRECTEMENT le second
    # membre de la brique de fond, alpha (n - n0). Meme operateur, meme f -> meme phi.
    f_ref = alpha * (dens - n0)
    phi_ref = _solve_single(adc.ChargeDensity(charge=1.0), f_ref, n, rhs="charge_density")

    scale = float(np.max(np.abs(phi_ref)))
    assert scale > 1e-6, "potentiel nul (second membre de fond trivial ?)"
    err = float(np.max(np.abs(phi_bg - phi_ref))) / scale
    assert err < 1e-10, "BackgroundDensity : phi != reference alpha(n-n0) (err %.2e)" % err

    # Garde-fou : le fond n'est PAS la densite de charge q n. phi(fond) doit differer de phi(charge n).
    phi_charge = _solve_single(adc.ChargeDensity(charge=1.0), dens, n, rhs="charge_density")
    diff = float(np.max(np.abs(phi_bg - phi_charge))) / scale
    assert diff > 1e-2, "fond == charge ? (le second membre serait fige sur q n)"
    print("OK  BackgroundDensity : f = alpha (n - n0) compose (err %.1e, ecart a q n %.1e)"
          % (err, diff))


def mixed_bricks_sum():
    """Deux blocs, briques DIFFERENTES : f = q0 n0 + alpha (n1 - n0bg). phi == somme manuelle."""
    n = 64
    q0, alpha, n0bg = -0.8, 1.3, 0.5
    d0, d1 = _density(n), 1.0 + 0.2 * np.cos(2 * PI * (np.arange(n) + 0.5) / n)[None, :] * np.ones((n, n))

    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.add_block("a", model=_scalar(adc.ChargeDensity(charge=q0)), spatial=adc.Spatial(none=True))
    sim.add_block("b", model=_scalar(adc.BackgroundDensity(alpha=alpha, n0=n0bg)),
                  spatial=adc.Spatial(none=True))
    sim.set_poisson(rhs="composite", solver="fft")
    sim.set_density("a", d0.reshape(-1).tolist())
    sim.set_density("b", d1.reshape(-1).tolist())
    sim.solve_fields()
    phi_mix = np.array(sim.potential()).reshape(n, n)

    # Reference : un seul bloc de charge unite dont la densite vaut la somme manuelle des deux briques.
    f_ref = q0 * d0 + alpha * (d1 - n0bg)
    phi_ref = _solve_single(adc.ChargeDensity(charge=1.0), f_ref, n, rhs="charge_density")

    scale = float(np.max(np.abs(phi_ref)))
    err = float(np.max(np.abs(phi_mix - phi_ref))) / scale
    assert err < 1e-10, "briques mixtes : phi != somme manuelle (err %.2e)" % err
    print("OK  briques mixtes (charge + fond) sommees : phi == reference (err %.1e)" % err)


def epm_facade_roundtrip():
    """add_elliptic_model(rhs=composite_rhs()) == set_poisson(rhs='composite') (bit-identique)."""
    n = 64
    alpha, n0 = 0.9, 0.3
    dens = _density(n)
    phi_setpoisson = _solve_single(adc.BackgroundDensity(alpha=alpha, n0=n0), dens, n)

    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.add_block("blk", model=_scalar(adc.BackgroundDensity(alpha=alpha, n0=n0)),
                  spatial=adc.Spatial(none=True))
    sim.add_elliptic_model("poisson",
                           adc.elliptic(operator=adc.div_eps_grad(1.0), rhs=adc.composite_rhs(),
                                        output=adc.electric_field_from_potential()),
                           solver=adc.EllipticSolver("fft"))
    sim.set_density("blk", dens.reshape(-1).tolist())
    sim.solve_fields()
    phi_epm = np.array(sim.potential()).reshape(n, n)

    scale = float(np.max(np.abs(phi_setpoisson)))
    err = float(np.max(np.abs(phi_epm - phi_setpoisson))) / scale
    assert err < 1e-12, "EPM composite_rhs() != set_poisson(rhs=composite) (err %.2e)" % err
    print("OK  EPM add_elliptic_model(composite_rhs()) == set_poisson(rhs=composite) (err %.1e)" % err)


def token_alias():
    """'composite' et 'charge_density' empruntent le MEME chemin : phi identique sur un bloc charge."""
    n = 64
    dens = _density(n)
    phi_charge = _solve_single(adc.ChargeDensity(charge=1.0), dens, n, rhs="charge_density")
    phi_comp = _solve_single(adc.ChargeDensity(charge=1.0), dens, n, rhs="composite")
    err = float(np.max(np.abs(phi_comp - phi_charge)))
    assert err == 0.0, "token 'composite' != 'charge_density' sur un bloc charge (err %.2e)" % err
    print("OK  token 'composite' alias bit-identique de 'charge_density' (bloc charge)")

    # Un token inconnu est refuse explicitement.
    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.add_block("blk", model=_scalar(adc.ChargeDensity(charge=1.0)), spatial=adc.Spatial(none=True))
    try:
        sim.set_poisson(rhs="bogus", solver="fft")
        sim.set_density("blk", dens.reshape(-1).tolist())
        sim.solve_fields()
        raise AssertionError("un token rhs inconnu aurait du etre refuse")
    except RuntimeError:
        print("OK  token rhs inconnu refuse explicitement")


def main():
    background_is_generic()
    mixed_bricks_sum()
    epm_facade_roundtrip()
    token_alias()
    print("test_poisson_composite : tout est vert")


if __name__ == "__main__":
    main()
