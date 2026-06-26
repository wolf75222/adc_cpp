"""Tests du mini-DSL symbolique pops.dsl (interprete CPU).

Verifie : (1) le flux d'Euler ECRIT en formules == flux d'Euler de reference (numpy), composante
par composante ; (2) max_wave_speed coherent ; (3) check() detecte une variable non definie
(verification de dependances) ; (4) le modele declare en formules TOURNE via pops.PythonFlux
(masse conservee sur un domaine periodique). Pur Python (aucun binding) : lance avec python3.
"""
import numpy as np

from pops import dsl

GAMMA = 1.4


def build_euler():
    """Euler compressible 2D ecrit entierement en formules symboliques."""
    e = dsl.HyperbolicModel("euler")
    rho, rhou, rhov, E = e.conservative_vars("rho", "rho_u", "rho_v", "E")
    u = e.primitive("u", rhou / rho)
    v = e.primitive("v", rhov / rho)
    p = e.primitive("p", (GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v)))
    H = (E + p) / rho
    c = dsl.sqrt(GAMMA * p / rho)
    e.set_flux(x=[rhou, rhou * u + p, rhou * v, rho * H * u],
               y=[rhov, rhov * u, rhov * v + p, rho * H * v])
    e.set_eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    return e


def ref_flux(U, d):
    """Flux d'Euler de reference, ecrit directement en numpy (oracle)."""
    rho = U[0]
    uu = U[1] / rho
    vv = U[2] / rho
    p = (GAMMA - 1.0) * (U[3] - 0.5 * rho * (uu * uu + vv * vv))
    H = (U[3] + p) / rho
    if d == 0:
        return np.stack([U[1], U[1] * uu + p, U[1] * vv, rho * H * uu])
    return np.stack([U[2], U[2] * uu, U[2] * vv + p, rho * H * vv])


def main():
    e = build_euler()
    assert e.check()
    assert e.n_vars == 4

    # (1) flux symbolique == flux de reference sur un etat deterministe positif
    n = 12
    x = np.linspace(0.1, 0.9, n)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    rho = 1.0 + 0.3 * xx
    u = 0.2 * np.cos(2 * np.pi * yy)
    v = 0.1 * np.sin(2 * np.pi * xx)
    p = 1.0 + 0.2 * yy
    E = p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)
    U = np.stack([rho, rho * u, rho * v, E])
    for d in (0, 1):
        assert np.allclose(e.flux(U, {}, d), ref_flux(U, d), rtol=1e-12, atol=1e-12), \
            "flux symbolique != reference (dir %d)" % d
    print("OK  flux symbolique == flux d'Euler de reference (x et y)")

    # (2) max_wave_speed == max(|u| + c) en x
    c = np.sqrt(GAMMA * p / rho)
    assert abs(e.max_wave_speed(U, {}, 0) - float(np.max(np.abs(u) + c))) < 1e-12
    print("OK  max_wave_speed coherent")

    # (3) verification de dependances : reference d'une variable non declaree -> ValueError
    bad = dsl.HyperbolicModel("bad")
    (r,) = bad.conservative_vars("rho")
    ghost = dsl.Var("ghost", "aux")  # jamais declaree via aux()
    bad.set_flux(x=[r * ghost], y=[r])
    bad.set_eigenvalues(x=[r], y=[r])
    try:
        bad.check()
        raise SystemExit("ECHEC : check() aurait du lever ValueError")
    except ValueError:
        print("OK  check() detecte une variable non definie")

    # (4) le modele declare en formules TOURNE : masse conservee (bulle de pression, periodique)
    m, L = 48, 1.0
    h = L / m
    xs = (np.arange(m) + 0.5) / m
    gx, gy = np.meshgrid(xs, xs, indexing="xy")
    r2 = (gx - 0.5) ** 2 + (gy - 0.5) ** 2
    p0 = 1.0 + 0.3 * np.exp(-r2 / 0.01)
    U = np.zeros((4, m, m))
    U[0] = 1.0                      # rho uniforme
    U[3] = p0 / (GAMMA - 1.0)       # u = v = 0 -> E = p/(gamma-1)
    mass0 = float(U[0].sum())
    pf = e.to_python_flux()
    for _ in range(40):
        U = U + pf.cfl_dt(U, h, 0.4) * pf.residual(U, h)
    drel = abs(float(U[0].sum()) - mass0) / mass0
    assert np.isfinite(U).all() and U[0].min() > 0, "etat non physique"
    assert drel < 1e-9, "masse non conservee (drel = %.2e)" % drel
    print("OK  euler symbolique tourne via PythonFlux (masse conservee, drel %.1e)" % drel)

    print("test_dsl : tout est vert")


if __name__ == "__main__":
    main()
