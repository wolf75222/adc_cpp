"""Operateur elliptique de Poisson de systeme, permittivite CONSTANTE et VARIABLE.

1. eps CONSTANT : div(eps grad phi) = f <=> lap phi = f/eps. Linearite en 1/eps
   (phi(eps=2) == phi(eps=1)/2) via set_poisson(epsilon=...) ET via l'EPM div_eps_grad(eps).
2. eps(x) VARIABLE : set_epsilon_field(champ n*n) cable GeometricMG::set_epsilon (operateur
   div(eps grad phi) a coefficient de face harmonique, sans mise a l'echelle 1/eps du rhs). On
   verifie une solution manufacturee LISSE (Dirichlet), la coherence (eps(x) modifie bien phi), la
   non-regression (champ uniforme=1 == operateur sans eps) et le refus avec le solveur 'fft'.
"""
from pops.numerics.riemann import Rusanov
import numpy as np

import pops

PI = np.pi


def _charge_model():
    return pops.Model(state=pops.FluidState(kind="compressible", gamma=1.4),
                     transport=pops.CompressibleFlux(), source=pops.NoSource(),
                     elliptic=pops.ChargeDensity(charge=1.0))


def _charge_scalar():
    """Bloc scalaire (1 var) de densite de charge unite : f = q n = n. set_density n'ecrit que la
    densite (comp 0), ce qui isole le second membre du Poisson pour une solution manufacturee."""
    return pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0),
                     source=pops.NoSource(), elliptic=pops.ChargeDensity(charge=1.0))


def _density(n):
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    return 1.0 + 0.3 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)


def phi_set_poisson(eps, n=64):
    sim = pops.System(n=n, L=1.0, periodic=True)
    sim.add_block("gas", model=_charge_model(),
                  spatial=pops.Spatial(flux=Rusanov()), time=pops.Explicit())
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
    sim = pops.System(n=n, L=1.0, periodic=True)
    sim.add_block("gas", model=_charge_model(),
                  spatial=pops.Spatial(flux=Rusanov()), time=pops.Explicit())
    sim.set_density("gas", _density(n).reshape(-1).tolist())
    sim.add_elliptic_model("poisson",
                           pops.elliptic(operator=pops.div_eps_grad(2.0), rhs=pops.charge_density(),
                                        output=pops.electric_field_from_potential()),
                           solver=pops.EllipticSolver("fft"))
    sim.solve_fields()
    phi_epm = np.array(sim.potential()).reshape(n, n)
    err2 = float(np.max(np.abs(phi_epm - phi2))) / scale
    assert err2 < 1e-12, "EPM div_eps_grad(2) != set_poisson(epsilon=2) (err %.2e)" % err2
    print("OK  add_elliptic_model(div_eps_grad(2)) == set_poisson(epsilon=2) (err %.1e)" % err2)

    variable_epsilon_tests()
    print("test_poisson_eps : tout est vert")


def variable_epsilon_tests():
    """eps(x) VARIABLE via set_epsilon_field : MMS, coherence, non-regression, refus 'fft'."""
    # MMS Dirichlet : phi = sin(pi x) sin(pi y) (nul au bord), eps = 1 + 0.5 x (lisse), et
    #   f = div(eps grad phi) = -(1 + 0.5 x) 2 pi^2 sin(pi x) sin(pi y) + 0.5 pi cos(pi x) sin(pi y).
    # Le bloc de charge (charge 1) pose f comme densite -> rhs du Poisson = f.
    n = 96
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs, indexing="xy")               # X[j,i]=x_i, Y[j,i]=y_j
    s_xy = np.sin(PI * X) * np.sin(PI * Y)
    phi_ex = s_xy
    eps = 1.0 + 0.5 * X
    f = -(1.0 + 0.5 * X) * 2.0 * PI ** 2 * s_xy + 0.5 * PI * np.cos(PI * X) * np.sin(PI * Y)

    def solve(eps_field, solver="geometric_mg"):
        s = pops.System(n=n, L=1.0, periodic=False)
        s.add_block("q", model=_charge_scalar(), spatial=pops.Spatial(none=True))
        s.set_poisson(rhs="charge_density", solver=solver, bc="dirichlet")
        s.set_density("q", f)
        if eps_field is not None:
            s.set_epsilon_field(eps_field)
        s.solve_fields()
        return np.array(s.potential()).reshape(n, n)

    amp = float(np.max(np.abs(phi_ex)))
    phi_var = solve(eps)
    err = float(np.max(np.abs(phi_var - phi_ex))) / amp
    assert err < 5e-3, "eps(x) MMS : phi != solution manufacturee (err rel %.2e)" % err
    print("OK  eps(x) variable (MMS Dirichlet) : phi == sin(pi x)sin(pi y) (err rel %.1e)" % err)

    # Coherence : eps(x) modifie REELLEMENT l'operateur (phi differe de la solution a eps=1).
    phi_const = solve(None)
    diff = float(np.max(np.abs(phi_var - phi_const))) / amp
    assert diff > 1e-2, "eps(x) sans effet (phi identique a eps=1 ?)"
    print("OK  eps(x) modifie l'operateur : ecart a eps=1 = %.1e" % diff)

    # Non-regression : un champ eps UNIFORME=1 redonne EXACTEMENT l'operateur sans eps.
    phi_unit = solve(np.ones((n, n)))
    gap = float(np.max(np.abs(phi_unit - phi_const))) / amp
    assert gap < 1e-9, "champ eps uniforme=1 != operateur sans eps (gap %.2e)" % gap
    print("OK  non-regression : champ eps uniforme=1 == operateur sans eps (gap %.1e)" % gap)

    # eps(x) variable + solveur 'fft' (coefficient constant) : refus explicite au solve.
    sp = pops.System(n=n, L=1.0, periodic=True)
    sp.add_block("q", model=_charge_scalar(), spatial=pops.Spatial(none=True))
    sp.set_poisson(rhs="charge_density", solver="fft")
    sp.set_density("q", f)
    sp.set_epsilon_field(eps)
    try:
        sp.solve_fields()
        raise AssertionError("fft + eps(x) variable aurait du lever une erreur")
    except RuntimeError:
        print("OK  eps(x) variable refuse avec solver='fft' (coefficient constant)")


if __name__ == "__main__":
    main()
