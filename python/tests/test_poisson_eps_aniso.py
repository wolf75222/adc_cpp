"""Operateur elliptique de Poisson de systeme, permittivite ANISOTROPE eps_x(x), eps_y(x).

set_epsilon_anisotropic_field(eps_x, eps_y) cable GeometricMG::set_epsilon_anisotropic : l'operateur
passe de div(eps grad phi) (eps scalaire) a div(diag(eps_x, eps_y) grad phi) ; les faces normales a x
portent eps_x, celles normales a y portent eps_y (coefficients de face harmoniques, ordre 2), sans
mise a l'echelle 1/eps du second membre. On verifie une solution manufacturee LISSE (Dirichlet), la
coherence (l'anisotropie modifie REELLEMENT phi vs l'isotrope), et le refus avec le solveur 'fft'.
"""
import numpy as np

import pops

PI = np.pi


def _charge_scalar():
    """Bloc scalaire (1 var) de densite de charge unite : f = q n = n. set_density n'ecrit que la
    densite (comp 0), ce qui isole le second membre du Poisson pour une solution manufacturee."""
    return pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0),
                     source=pops.NoSource(), elliptic=pops.ChargeDensity(charge=1.0))


def anisotropic_epsilon_tests():
    """eps anisotrope via set_epsilon_anisotropic_field : MMS Dirichlet, coherence, refus 'fft'."""
    # MMS Dirichlet : phi = sin(pi x) sin(pi y) (nul au bord), eps_x = 1 + 0.5 x, eps_y = 1 + 0.3 y
    # (lisses, > 0), et f = d/dx(eps_x dphi/dx) + d/dy(eps_y dphi/dy) calcule analytiquement :
    #   f = 0.5 pi cos(pi x) sin(pi y) + 0.3 pi sin(pi x) cos(pi y)
    #       - (2 + 0.5 x + 0.3 y) pi^2 sin(pi x) sin(pi y).
    # Le bloc de charge (charge 1) pose f comme densite -> rhs du Poisson = f.
    n = 96
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs, indexing="xy")               # X[j,i]=x_i, Y[j,i]=y_j
    sx, sy = np.sin(PI * X), np.sin(PI * Y)
    cx, cy = np.cos(PI * X), np.cos(PI * Y)
    phi_ex = sx * sy
    eps_x = 1.0 + 0.5 * X
    eps_y = 1.0 + 0.3 * Y
    f = (0.5 * PI * cx * sy + 0.3 * PI * sx * cy
         - (2.0 + 0.5 * X + 0.3 * Y) * PI ** 2 * sx * sy)

    def solve(eps_xy, solver="geometric_mg"):
        s = pops.System(n=n, L=1.0, periodic=False)
        s.add_block("q", model=_charge_scalar(), spatial=pops.Spatial(none=True))
        s.set_poisson(rhs="charge_density", solver=solver, bc="dirichlet")
        s.set_density("q", f)
        if eps_xy is not None:
            s.set_epsilon_anisotropic_field(eps_xy[0], eps_xy[1])
        s.solve_fields()
        return np.array(s.potential()).reshape(n, n)

    amp = float(np.max(np.abs(phi_ex)))
    phi_aniso = solve((eps_x, eps_y))
    err = float(np.max(np.abs(phi_aniso - phi_ex))) / amp
    assert err < 5e-3, "eps anisotrope MMS : phi != solution manufacturee (err rel %.2e)" % err
    print("OK  eps anisotrope (MMS Dirichlet) : phi == sin(pi x)sin(pi y) (err rel %.1e)" % err)

    # Coherence : l'anisotropie modifie REELLEMENT l'operateur. On compare a l'operateur ISOTROPE
    # eps = eps_x (set_epsilon_field) : meme rhs f, mais les faces y voient eps_x au lieu de eps_y,
    # donc phi doit differer franchement (eps_x et eps_y sont distincts).
    s_iso = pops.System(n=n, L=1.0, periodic=False)
    s_iso.add_block("q", model=_charge_scalar(), spatial=pops.Spatial(none=True))
    s_iso.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet")
    s_iso.set_density("q", f)
    s_iso.set_epsilon_field(eps_x)
    s_iso.solve_fields()
    phi_iso = np.array(s_iso.potential()).reshape(n, n)
    diff = float(np.max(np.abs(phi_aniso - phi_iso))) / amp
    assert diff > 1e-2, "anisotropie sans effet (phi identique a l'isotrope eps_x ?)"
    print("OK  anisotropie modifie l'operateur : ecart a l'isotrope eps_x = %.1e" % diff)

    # Non-regression : eps_x == eps_y redonne EXACTEMENT l'operateur isotrope eps = eps_x.
    phi_deg = solve((eps_x, eps_x.copy()))
    gap = float(np.max(np.abs(phi_deg - phi_iso))) / amp
    assert gap < 1e-9, "anisotropie degeneree eps_x==eps_y != isotrope eps_x (gap %.2e)" % gap
    print("OK  non-regression : eps_x == eps_y == isotrope eps_x (gap %.1e)" % gap)

    # eps anisotrope + solveur 'fft' (coefficient constant) : refus explicite au solve.
    sp = pops.System(n=n, L=1.0, periodic=True)
    sp.add_block("q", model=_charge_scalar(), spatial=pops.Spatial(none=True))
    sp.set_poisson(rhs="charge_density", solver="fft")
    sp.set_density("q", f)
    sp.set_epsilon_anisotropic_field(eps_x, eps_y)
    try:
        sp.solve_fields()
        raise AssertionError("fft + eps anisotrope aurait du lever une erreur")
    except RuntimeError:
        print("OK  eps anisotrope refuse avec solver='fft' (coefficient constant)")


def main():
    anisotropic_epsilon_tests()
    print("test_poisson_eps_aniso : tout est vert")


if __name__ == "__main__":
    main()
