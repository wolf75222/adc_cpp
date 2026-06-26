"""Time integrators written in Python, on top of the library primitives.

You can implement your own time integrator in Python without a Python callback
in the hot path, via the primitives exposed by `System`:

    sim.solve_fields()          # Poisson + aux = grad(phi)        (C++ computation)
    R = sim.eval_rhs(name)      # residual -div F + S of the block (C++ computation, per cell)
    U = sim.get_state(name)     # block state (ndarray ncomp,n,n)
    sim.set_state(name, U)      # writes the state

The residual and Poisson stay computed in the compiled library (per cell), only
the assembly of the RK stages is in Python (per step). You can thus write any
time scheme (here forward Euler and SSPRK2) without touching the C++. These
integrators re-solve Poisson at each RK stage (per-stage coupling, more accurate
than the per-step frozen coupling of the compiled step()).
"""


def euler_step(sim, dt, names=None):
    """Forward Euler, written in Python: U <- U + dt * RHS(U). Poisson solved once."""
    names = names if names is not None else sim.block_names()
    sim.solve_fields()
    for n in names:
        sim.set_state(n, sim.get_state(n) + dt * sim.eval_rhs(n))


def ssprk2_step(sim, dt, names=None):
    """SSPRK2 (strong-stable Heun) written in Python, Poisson re-solved at each stage.

        U1 = U0 + dt R(U0)
        U  = 1/2 U0 + 1/2 (U1 + dt R(U1))

    All blocks advance together, with Poisson (which couples them) re-solved
    between the two stages: per-stage hyperbolic/elliptic coupling.
    """
    names = names if names is not None else sim.block_names()
    sim.solve_fields()
    U0 = {n: sim.get_state(n) for n in names}
    for n in names:                                   # stage 1: U1 = U0 + dt R(U0)
        sim.set_state(n, U0[n] + dt * sim.eval_rhs(n))
    sim.solve_fields()                                # Poisson re-solved (per-stage)
    for n in names:                                   # stage 2: strong-stable combination
        U1 = sim.get_state(n)                         # = U0 + dt R(U0)
        sim.set_state(n, 0.5 * U0[n] + 0.5 * (U1 + dt * sim.eval_rhs(n)))
