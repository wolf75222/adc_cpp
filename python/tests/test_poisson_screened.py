"""Operateur elliptique de Poisson de systeme ECRANTE / Helmholtz, cote runtime System.

set_reaction_field(kappa) cable GeometricMG::set_reaction : l'operateur passe de div(eps grad phi)=f
a div(eps grad phi) - kappa phi = f (Poisson de Debye ; kappa = 1/lambda_D^2). On verifie une
solution manufacturee LISSE (Dirichlet), la coherence (kappa modifie reellement phi), la
non-regression (kappa=0 == Poisson), le refus de kappa < 0 et le refus avec le solveur 'fft'.
"""
import numpy as np

import pops

PI = np.pi
KAPPA = 50.0  # 1/lambda_D^2 (ecrantage modere)


def _charge_scalar():
    """Bloc scalaire de densite de charge unite (f = q n = n) : isole le second membre du Poisson."""
    return pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0),
                     source=pops.NoSource(), elliptic=pops.ChargeDensity(charge=1.0))


def main():
    # MMS Dirichlet : phi = sin(pi x) sin(pi y) (nul au bord) ; operateur lap - kappa :
    #   f = lap phi - kappa phi = -(2 pi^2 + kappa) sin(pi x) sin(pi y).
    n = 96
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    s_xy = np.sin(PI * X) * np.sin(PI * Y)
    phi_ex = s_xy
    f = -(2.0 * PI ** 2 + KAPPA) * s_xy

    def solve(kappa_field, solver="geometric_mg"):
        s = pops.System(n=n, L=1.0, periodic=False)
        s.add_block("q", model=_charge_scalar(), spatial=pops.Spatial(none=True))
        s.set_poisson(rhs="charge_density", solver=solver, bc="dirichlet")
        s.set_density("q", f)
        if kappa_field is not None:
            s.set_reaction_field(kappa_field)
        s.solve_fields()
        return np.array(s.potential()).reshape(n, n)

    amp = float(np.max(np.abs(phi_ex)))

    # (A) MMS ecrante : phi == sin(pi x) sin(pi y).
    phi_k = solve(KAPPA * np.ones((n, n)))
    err = float(np.max(np.abs(phi_k - phi_ex))) / amp
    assert err < 5e-3, "kappa MMS : phi != solution manufacturee (err rel %.2e)" % err
    print("OK  Poisson ecrante (MMS Dirichlet) : phi == sin(pi x)sin(pi y) (err rel %.1e)" % err)

    # (B) coherence : kappa modifie REELLEMENT l'operateur (phi differe du Poisson pur).
    phi0 = solve(None)
    diff = float(np.max(np.abs(phi_k - phi0))) / amp
    assert diff > 1e-2, "kappa sans effet (phi identique au Poisson ?)"
    print("OK  kappa modifie l'operateur : ecart au Poisson pur = %.1e" % diff)

    # (C) non-regression : kappa=0 partout redonne EXACTEMENT le Poisson.
    phi_k0 = solve(np.zeros((n, n)))
    gap = float(np.max(np.abs(phi_k0 - phi0))) / amp
    assert gap < 1e-9, "kappa=0 != Poisson (gap %.2e)" % gap
    print("OK  non-regression : kappa=0 == Poisson (gap %.1e)" % gap)

    # (D) kappa < 0 refuse (operateur mal pose / multigrille non convergente).
    try:
        solve(-np.ones((n, n)))
        raise AssertionError("kappa < 0 aurait du lever une erreur")
    except RuntimeError:
        print("OK  kappa < 0 refuse")

    # (E) kappa + solveur 'fft' (Poisson pur) : refus explicite.
    sp = pops.System(n=n, L=1.0, periodic=True)
    sp.add_block("q", model=_charge_scalar(), spatial=pops.Spatial(none=True))
    sp.set_poisson(rhs="charge_density", solver="fft")
    sp.set_density("q", f)
    sp.set_reaction_field(KAPPA * np.ones((n, n)))
    try:
        sp.solve_fields()
        raise AssertionError("fft + kappa aurait du lever une erreur")
    except RuntimeError:
        print("OK  kappa refuse avec solver='fft' (Poisson pur)")

    print("test_poisson_screened : tout est vert")


if __name__ == "__main__":
    main()
