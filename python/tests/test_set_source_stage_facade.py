#!/usr/bin/env python3
"""ADC-308 : methode PUBLIQUE pops.System.set_source_stage sur la facade.

Le moteur expose set_source_stage cote binding (_pops.System) et la facade
pops.System l'APPELLE en interne (add_block / add_equation, etage source condense
par Schur), mais elle ne la publiait PAS comme methode publique : un cas aval
devait atteindre l'objet prive (sim._s.set_source_stage(...)). On surface ici un
mince wrapper public, de meme signature que le binding.

Le test prouve DEUX choses, chacune non triviale :

  1) EXPLICITE -- pops.System.set_source_stage est une vraie methode DEFINIE SUR LA
     FACADE, pas un simple transfert implicite via __getattr__ vers _s. On le
     verifie avec inspect.getattr_static, qui NE declenche PAS __getattr__ : sur le
     code d'avant ADC-308 il leve AttributeError (le test ECHOUE), la garde est donc
     reelle et pas satisfaite par le seul forwarding historique.
  2) PARITE -- configurer l'etage source via la methode PUBLIQUE donne un etat
     BIT-IDENTIQUE au chemin PRIVE sim._s.set_source_stage(...), et l'etage tourne
     REELLEMENT (diff > 0 vs un bloc identique SANS etage source : pas un no-op).

Briques NATIVES uniquement (IsothermalFlux : roles Density / MomentumX / MomentumY,
exactement ceux exiges par set_source_stage). Aucun compilateur ni DSL : CI-safe.
Calque sur test_schur_via_system.py.
"""

import inspect
import sys

import numpy as np

try:
    import pops
except ImportError as e:  # module pas construit : on se saute proprement (comme les freres)
    print("skip  module adc absent (PYTHONPATH ?) : %s" % e)
    sys.exit(0)


def chk(cond, label):
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        raise AssertionError(label)


def iso_fluid_model(cs2=1.0, alpha=1.0):
    """Fluide isotherme natif : modele MINIMAL accepte par set_source_stage."""
    return pops.Model(
        state=pops.FluidState(kind="isothermal", cs2=cs2),
        transport=pops.IsothermalFlux(),
        source=pops.NoSource(),
        elliptic=pops.BackgroundDensity(alpha=alpha, n0=0.0),
    )


def smooth_profile(n, L):
    """Profils lisses, nuls aux bords (compatibles Dirichlet)."""
    x = (np.arange(n) + 0.5) * (L / n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho0 = 1.5 * np.ones((n, n))
    u0 = 0.5 * np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
    v0 = -0.3 * np.sin(2.0 * np.pi * X / L) * np.sin(np.pi * Y / L)
    return rho0, u0, v0


def build_block(n, L, B0, alpha, cs2):
    """System cartesien + 1 bloc isotherme natif en TRANSPORT PUR (pops.Explicit).

    L'etage source n'est PAS cable ici : l'appelant decide ensuite (facade publique,
    chemin _s, ou rien). B_z est pose AVANT (prerequis du terme de Lorentz condense).
    """
    sim = pops.System(n=n, L=L, periodic=False)
    sim.set_poisson(bc="dirichlet")
    sim.set_magnetic_field(B0 * np.ones((n, n)))
    sim.add_equation(
        "ions",
        model=iso_fluid_model(cs2=cs2, alpha=alpha),
        spatial=pops.FiniteVolume(limiter="minmod", riemann="rusanov",
                                 variables="conservative"),
        time=pops.Explicit(),
    )
    rho0, u0, v0 = smooth_profile(n, L)
    sim.set_primitive_state("ions", rho=rho0, u=u0, v=v0)
    return sim


def state(sim, n):
    return np.array(sim.get_state("ions")).reshape(3, n, n)


def main():
    n, L = 24, 1.0
    B0, alpha, cs2 = 4.0, 3.0, 1.0
    theta, dt, n_steps = 1.0, 5.0e-4, 5
    kind = "electrostatic_lorentz"

    # ------------------------------------------------------------------
    # (1) La methode publique existe EXPLICITEMENT sur la facade.
    #     getattr_static ne declenche PAS __getattr__ : sans le wrapper, AttributeError.
    # ------------------------------------------------------------------
    try:
        meth = inspect.getattr_static(pops.System, "set_source_stage")
    except AttributeError:
        meth = None
    chk(meth is not None,
        "(1a) pops.System.set_source_stage defini sur la facade (pas seulement __getattr__)")
    chk(callable(meth) and getattr(meth, "__qualname__", "") == "System.set_source_stage",
        "(1b) c'est bien la methode de la facade (qualname=%r)"
        % getattr(meth, "__qualname__", None))

    # ------------------------------------------------------------------
    # (2) Parite PUBLIC vs PRIVE, et l'etage source tourne reellement.
    # ------------------------------------------------------------------
    sim_pub = build_block(n, L, B0, alpha, cs2)
    sim_pub.set_source_stage("ions", kind, theta, alpha)       # FACADE PUBLIQUE (ADC-308)

    sim_priv = build_block(n, L, B0, alpha, cs2)
    sim_priv._s.set_source_stage("ions", kind, theta, alpha)   # chemin prive historique

    sim_none = build_block(n, L, B0, alpha, cs2)               # aucun etage source (temoin)

    for _ in range(n_steps):
        sim_pub.step(dt)
        sim_priv.step(dt)
        sim_none.step(dt)

    s_pub, s_priv, s_none = state(sim_pub, n), state(sim_priv, n), state(sim_none, n)

    if not np.all(np.isfinite(s_pub)):
        raise AssertionError("etat public non fini (nan/inf)")

    d_parity = float(np.max(np.abs(s_pub - s_priv)))
    chk(d_parity == 0.0,
        "(2a) public == prive BIT-IDENTIQUE (max|diff|=%.3e)" % d_parity)

    # Si set_source_stage etait un no-op silencieux, s_pub == s_none et diff == 0.
    d_active = float(np.max(np.abs(s_pub - s_none)))
    chk(d_active > 1e-6,
        "(2b) l'etage source a tourne via la facade (diff vs no-source=%.3e)" % d_active)

    print("test_set_source_stage_facade : tout est vert (4 verifications)")


if __name__ == "__main__":
    main()
