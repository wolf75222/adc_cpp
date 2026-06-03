"""Operateur elliptique a permittivite CONSTANTE : div(eps grad phi) = f <=> lap phi = f/eps.
Verifie la linearite en 1/eps (phi(eps=2) == phi(eps=1)/2) via set_poisson(epsilon=...) ET via l'EPM
add_elliptic_model(div_eps_grad(eps)). eps(x) variable / diffusion / projection restent un raffinement
(solveur a coefficients variables, non disponible) signale par NotImplementedError.
"""
import numpy as np

import adc


def _charge_model():
    return adc.Model(state=adc.FluidState(kind="compressible", gamma=1.4),
                     transport=adc.CompressibleFlux(), source=adc.NoSource(),
                     elliptic=adc.ChargeDensity(charge=1.0))


def _density(n):
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    return 1.0 + 0.3 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)


def phi_set_poisson(eps, n=64):
    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.add_block("gas", model=_charge_model(),
                  spatial=adc.Spatial(flux="rusanov"), time=adc.Explicit())
    sim.set_density("gas", _density(n).reshape(-1).tolist())
    sim.set_poisson(rhs="charge_density", solver="fft", epsilon=eps)
    sim.solve_fields()
    return np.array(sim.potential()).reshape(n, n)


def main():
    n = 64
    phi1 = phi_set_poisson(1.0, n)
    phi2 = phi_set_poisson(2.0, n)
    scale = float(np.max(np.abs(phi1)))
    assert scale > 1e-6, "potentiel nul (pas de charge ?)"
    err = float(np.max(np.abs(phi2 - 0.5 * phi1))) / scale
    assert err < 1e-10, "phi(eps=2) != phi(eps=1)/2 (err %.2e)" % err
    print("OK  set_poisson(epsilon) : phi(eps=2) == phi(eps=1)/2 (err %.1e)" % err)

    # meme resultat via l'EPM compose (div_eps_grad(2.0))
    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.add_block("gas", model=_charge_model(),
                  spatial=adc.Spatial(flux="rusanov"), time=adc.Explicit())
    sim.set_density("gas", _density(n).reshape(-1).tolist())
    sim.add_elliptic_model("poisson",
                           adc.elliptic(operator=adc.div_eps_grad(2.0), rhs=adc.charge_density(),
                                        output=adc.electric_field_from_potential()),
                           solver=adc.EllipticSolver("fft"))
    sim.solve_fields()
    phi_epm = np.array(sim.potential()).reshape(n, n)
    err2 = float(np.max(np.abs(phi_epm - phi2))) / scale
    assert err2 < 1e-12, "EPM div_eps_grad(2) != set_poisson(epsilon=2) (err %.2e)" % err2
    print("OK  add_elliptic_model(div_eps_grad(2)) == set_poisson(epsilon=2) (err %.1e)" % err2)
    print("test_poisson_eps : tout est vert")


if __name__ == "__main__":
    main()
