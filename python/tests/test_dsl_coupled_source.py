"""Source COUPLEE GENERIQUE inter-especes par le DSL (pops.dsl.CoupledSource, P5 phase 1, splitting
EXPLICITE). On decrit un couplage d'IONISATION en FORMULES -- pas la brique nommee add_ionization --
et on verifie qu'il s'applique comme un etage operator-split APRES le transport, AUCUN callback Python
par cellule (le bytecode est interprete cote C++ dans le for_each_cell device) :

    d_t n_e = +k n_e n_g     (un electron apparait)
    d_t n_i = +k n_e n_g     (un ion apparait)
    d_t n_g = -k n_e n_g     (un neutre disparait)

Invariants verifies :
(A) API : pops.dsl.CoupledSource(...).block(...).role(...) / .param(...) / .add(...) / .compile(...)
    construit l'ABI plate (bytecode) et sim.add_coupling(...) la branche sur System.add_coupled_source.
(B) Numerique : densites SPATIALEMENT UNIFORMES -> le transport (flux + derive E x B) est exactement
    nul a chaque pas (champ uniforme), donc seules les sources couplees evoluent l'etat. On compare la
    trajectoire a une REFERENCE NumPy de l'ODE forward-Euler (les MEMES Expr, evaluees par
    CompiledCoupledSource.reference_terms) -- bit-pour-bit la meme recurrence.
(C) Conservation : n_e et n_i CROISSENT, n_g DECROIT ; n_i + n_g (masse lourde) est conserve, et
    n_e - n_i reste constant (chaque ionisation cree une paire e/i). La masse totale n_e+n_i+n_g n'est
    PAS conservee (une paire e/i est creee par neutre consomme) -- c'est l'invariant ATTENDU.
(D) Defaut : un System SANS add_coupling reste BIT-IDENTIQUE (aucune evolution des densites uniformes).
(E) Pas de callback Python par cellule : la source est compilee en bytecode (verifie sur l'objet) ;
    l'evolution se produit dans System.step sans rappel Python.
"""
import numpy as np

import pops
from pops.physics.multispecies import CoupledSource


def chk(cond, msg, fails):
    if not cond:
        print("FAIL", msg)
        fails[0] += 1
    return cond


def build_source(k):
    """Construit la source d'ionisation generique (3 especes) par le DSL et la compile."""
    src = CoupledSource("ionization")
    ne = src.block("electrons").role("density")
    ni = src.block("ions").role("density")
    ng = src.block("neutrals").role("density")
    kp = src.param("Kiz", k)
    src.add("electrons", role="density", expr=+kp * ne * ng)
    src.add("ions", role="density", expr=+kp * ne * ng)
    src.add("neutrals", role="density", expr=-kp * ne * ng)
    return src.compile(backend="production")


def density_block(alpha=1.0, n0=1.0):
    """Bloc scalaire (densite) transporte par la derive E x B. Avec une densite UNIFORME et le fond
    neutralisant cale dessus, le transport est exactement nul -> seules les sources couplees agissent."""
    return pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0),
                     source=pops.NoSource(), elliptic=pops.BackgroundDensity(alpha=alpha, n0=n0))


def make_system(n, ne0, ni0, ng0):
    sim = pops.System(n=n, L=1.0, periodic=True)
    # n0 = densite uniforme de chaque bloc : f = alpha (n - n0) = 0 a l'init (phi uniforme -> derive nulle)
    sim.add_block("electrons", model=density_block(n0=ne0), spatial=pops.Spatial(none=True))
    sim.add_block("ions", model=density_block(n0=ni0), spatial=pops.Spatial(none=True))
    sim.add_block("neutrals", model=density_block(n0=ng0), spatial=pops.Spatial(none=True))
    sim.set_poisson(rhs="charge_density", solver="geometric_mg")
    sim.set_density("electrons", np.full((n, n), ne0))
    sim.set_density("ions", np.full((n, n), ni0))
    sim.set_density("neutrals", np.full((n, n), ng0))
    return sim


def main():
    fails = [0]
    n = 16
    k = 0.7
    dt = 0.01
    nsteps = 25
    ne0, ni0, ng0 = 0.30, 0.10, 1.00

    compiled = build_source(k)

    # (A) ABI plate construite par le DSL : 3 entrees (densites), 1 constante (k), 3 termes de sortie.
    chk(compiled.in_blocks == ["electrons", "ions", "neutrals"], "in_blocks ordre/contenu", fails)
    chk(compiled.in_roles == ["density", "density", "density"], "in_roles canoniques", fails)
    chk(compiled.consts == [k], "constante k inlinee", fails)
    chk(compiled.out_blocks == ["electrons", "ions", "neutrals"], "out_blocks", fails)
    chk(len(compiled.prog_lens) == 3 and sum(compiled.prog_lens) == len(compiled.prog_ops),
        "programmes bytecode segmentes", fails)
    # (E) bytecode (pas de callback Python) : programmes non vides, pas de fonction Python embarquee.
    chk(all(L > 0 for L in compiled.prog_lens) and not hasattr(compiled, "callback"),
        "source compilee en bytecode (aucun callback par cellule)", fails)

    # --- (D) defaut : System SANS couplage reste a l'identique (densites uniformes inchangees) ---
    base = make_system(n, ne0, ni0, ng0)
    for _ in range(nsteps):
        base.step(dt)
    chk(np.allclose(base.density("electrons"), ne0, atol=1e-12), "defaut: n_e inchange (pas de couplage)", fails)
    chk(np.allclose(base.density("ions"), ni0, atol=1e-12), "defaut: n_i inchange", fails)
    chk(np.allclose(base.density("neutrals"), ng0, atol=1e-12), "defaut: n_g inchange", fails)

    # --- couplage GENERIQUE branche via sim.add_coupling(compiled) ---
    sim = make_system(n, ne0, ni0, ng0)
    sim.add_coupling(compiled)

    # --- REFERENCE NumPy : MEME recurrence forward-Euler que l'etage C++ (sources evaluees par les
    #     memes Expr ; transport nul car etat uniforme). Etat scalaire (densites uniformes). ---
    ne, ni, ng = ne0, ni0, ng0
    traj = []
    for _ in range(nsteps):
        fields = {("electrons", "density"): np.array([ne]),
                  ("ions", "density"): np.array([ni]),
                  ("neutrals", "density"): np.array([ng])}
        terms = {(b, r): float(dS[0]) for (b, r, dS) in compiled.reference_terms(fields)}
        ne = ne + dt * terms[("electrons", "density")]
        ni = ni + dt * terms[("ions", "density")]
        ng = ng + dt * terms[("neutrals", "density")]
        traj.append((ne, ni, ng))

    for s, (rne, rni, rng) in enumerate(traj):
        sim.step(dt)
        ge = sim.density("electrons")
        gi = sim.density("ions")
        gg = sim.density("neutrals")
        # densite reste SPATIALEMENT UNIFORME (la source l'est) -> transport nul, etat = scalaire
        chk(np.ptp(ge) < 1e-12 and np.ptp(gi) < 1e-12 and np.ptp(gg) < 1e-12,
            "etat reste uniforme (transport nul) au pas %d" % s, fails)
        # (B) trajectoire == reference NumPy de l'ODE (tolerance serree : meme recurrence)
        chk(abs(ge.mean() - rne) < 1e-10, "n_e == ref ODE au pas %d (%.12g vs %.12g)" % (s, ge.mean(), rne), fails)
        chk(abs(gi.mean() - rni) < 1e-10, "n_i == ref ODE au pas %d" % s, fails)
        chk(abs(gg.mean() - rng) < 1e-10, "n_g == ref ODE au pas %d" % s, fails)

    ge = sim.density("electrons").mean()
    gi = sim.density("ions").mean()
    gg = sim.density("neutrals").mean()

    # (C) sens physique : electrons et ions ont CRU, neutres ont DECRU.
    chk(ge > ne0 + 1e-6, "n_e a cru", fails)
    chk(gi > ni0 + 1e-6, "n_i a cru", fails)
    chk(gg < ng0 - 1e-6, "n_g a decru", fails)
    # invariants de conservation : n_i + n_g (masse lourde) constant ; n_e - n_i constant (paires e/i).
    chk(abs((gi + gg) - (ni0 + ng0)) < 1e-9, "n_i + n_g conserve (masse lourde)", fails)
    chk(abs((ge - gi) - (ne0 - ni0)) < 1e-9, "n_e - n_i conserve (creation de paires)", fails)

    if fails[0] == 0:
        print("test_dsl_coupled_source : OK")
    else:
        print("test_dsl_coupled_source : %d FAIL" % fails[0])
    return fails[0]


if __name__ == "__main__":
    import sys
    sys.exit(1 if main() else 0)
