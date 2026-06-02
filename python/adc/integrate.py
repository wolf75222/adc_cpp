"""Integrateurs temporels ECRITS EN PYTHON, par-dessus les primitives de la lib.

Le tuteur voulait pouvoir implementer SON propre integrateur temporel — en Python
si on veut — sans que cela force un callback Python dans le hot path. C'est exactement
ce que permettent les primitives exposees par `System` :

    sim.solve_fields()          # Poisson + aux = grad(phi)        (calcul C++)
    R = sim.eval_rhs(name)      # residu -div F + S du bloc        (calcul C++, par CELLULE)
    U = sim.get_state(name)     # etat du bloc (ndarray ncomp,n,n)
    sim.set_state(name, U)      # ecrit l'etat

Le residu et Poisson restent calcules dans la lib compilee (par cellule) ; seul
l'ASSEMBLAGE des etages RK est en Python (par PAS). On peut donc ecrire n'importe
quel schema en temps — ici Euler avant et SSPRK2 — sans toucher au C++.

Bonus : ces integrateurs Python re-resolvent Poisson a CHAQUE etage RK (couplage
« per-stage », plus precis que le couplage fige par pas du step() compile).
"""


def euler_step(sim, dt, names=None):
    """Euler avant, ecrit en Python : U <- U + dt * RHS(U). Poisson resolu une fois."""
    names = names if names is not None else sim.block_names()
    sim.solve_fields()
    for n in names:
        sim.set_state(n, sim.get_state(n) + dt * sim.eval_rhs(n))


def ssprk2_step(sim, dt, names=None):
    """SSPRK2 (Heun fort-stable) ecrit en Python, Poisson re-resolu a chaque etage.

        U1 = U0 + dt R(U0)
        U  = 1/2 U0 + 1/2 (U1 + dt R(U1))

    Tous les blocs avancent ensemble, Poisson (qui les couple) etant re-resolu entre
    les deux etages — le couplage hyperbolique/elliptique « per-stage » du tuteur.
    """
    names = names if names is not None else sim.block_names()
    sim.solve_fields()
    U0 = {n: sim.get_state(n) for n in names}
    for n in names:                                   # etage 1 : U1 = U0 + dt R(U0)
        sim.set_state(n, U0[n] + dt * sim.eval_rhs(n))
    sim.solve_fields()                                # Poisson re-resolu (per-stage)
    for n in names:                                   # etage 2 : combinaison fort-stable
        U1 = sim.get_state(n)                         # = U0 + dt R(U0)
        sim.set_state(n, 0.5 * U0[n] + 0.5 * (U1 + dt * sim.eval_rhs(n)))
