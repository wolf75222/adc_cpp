"""Regression : un profil polaire INSTABLE ne doit JAMAIS faire crasher le processus au teardown.

Bug PRE-EXISTANT (signale a la revue de PR #176, present sur master propre, pas une regression) :
un System polaire stepant un profil INSTABLE (Gaussienne raide + BackgroundDensity n0=0) crashait au
teardown / a la premiere lecture hote (density()/potential()). Cause RACINE : les bindings (to_2d /
to_3d, python/bindings.cpp) remodelaient TOUT champ en CARRE (nx(), nx()), alors qu'un anneau polaire
fait nr*ntheta valeurs. Quand nr != ntheta le memcpy debordait le tampon numpy (nr*ntheta > nx()^2),
corrompant le tas ; avec des valeurs non finies (run instable) les octets debordes (motif NaN
0x7ff8...) ecrasaient les metadonnees de la free-list, d'ou l'abort au teardown. Le cas STABLE de
test_polar_system.py utilise nr == ntheta == 48, donc nx()^2 == nr*ntheta : aucun debordement, c'est
pourquoi il ne l'a jamais attrape.

Ce test exerce le scenario instable AVEC nr != ntheta (declencheur deterministe du debordement) dans un
SOUS-PROCESSUS : un crash dur (SIGSEGV/SIGABRT) tue tout le processus pytest et ne peut pas etre rattrape
par un try/except en proc ; on verifie donc que le sous-processus se termine SANS signal (soit
proprement, soit avec une exception Python claire), JAMAIS par un signal de crash.

Sur master non corrige : le sous-processus meurt par signal -> returncode < 0 -> le test ECHOUE.
Avec le fix (remodelage (ny, nx) = (ntheta, nr) cote bindings) : il se termine proprement -> PASSE.

Mono-rang (le Poisson polaire direct refuse MPI) : ce n'est pas un test MPI.
"""
import subprocess
import sys
import textwrap

# Scenario INSTABLE execute dans un sous-processus isole. nr != ntheta est le declencheur deterministe
# du debordement de tampon ; la Gaussienne raide + BackgroundDensity(n0=0) reproduit le run instable
# signale (le champ part en NaN/Inf), de sorte que les octets debordes portent un motif non fini.
_CHILD = textwrap.dedent(
    """
    import math
    import pops

    RMIN, RMAX, NR, NTH = 0.30, 1.00, 48, 64  # nr != ntheta : declencheur du debordement de tampon

    def steep_gaussian():
        # Layout attendu par set_density (polaire) : flat[j*NR + i] (theta lent, r rapide).
        dr = (RMAX - RMIN) / NR
        dth = 2.0 * math.pi / NTH
        r0 = 0.5 * (RMIN + RMAX)
        w = 0.01 * (RMAX - RMIN)  # tres etroite -> raide -> instable a gros dt
        rho = []
        for j in range(NTH):
            th = (j + 0.5) * dth
            for i in range(NR):
                r = RMIN + (i + 0.5) * dr
                g = math.exp(-((r - r0) ** 2) / (2.0 * w * w))
                rho.append(100.0 * g * (1.0 + 0.5 * math.cos(3.0 * th)))
        return rho

    sim = pops.System(mesh=pops.PolarMesh(r_min=RMIN, r_max=RMAX, nr=NR, ntheta=NTH))
    sim.add_block(
        "ne",
        model=pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0),
                        source=pops.NoSource(),
                        elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0)),
        spatial=pops.Spatial(minmod=True), time=pops.Explicit())
    sim.set_poisson(rhs="charge_density", solver="polar", bc="dirichlet")
    sim.set_density("ne", steep_gaussian())

    # Quelques pas a GROS dt fixe -> le profil raide devient instable (NaN/Inf), comme le run signale.
    for _ in range(20):
        sim.step(0.1)

    # Lectures hote : c'est ici que le remodelage carre (nx, nx) debordait le tampon numpy (nr*ntheta
    # valeurs > nx()^2). Avec le fix, le tableau fait (ntheta, nr) et le memcpy reste dans les bornes.
    d = sim.density("ne")
    p = sim.potential()
    assert d.shape == (NTH, NR), "density doit etre (ntheta, nr), recu %r" % (d.shape,)
    assert p.shape == (NTH, NR), "potential doit etre (ntheta, nr), recu %r" % (p.shape,)
    print("CHILD_OK")  # teardown a partir d'ici (destruction du System + du module)
    """
)


def test_unstable_polar_profile_does_not_crash_at_teardown():
    # On herite du PYTHONPATH courant (le module compile pops) : meme interpreteur que pytest.
    proc = subprocess.run([sys.executable, "-c", _CHILD], capture_output=True, text=True, timeout=120)

    # Un crash dur (SIGSEGV=11 -> rc -11 ; SIGABRT=6 -> rc -6 ; etc.) se manifeste par un returncode
    # NEGATIF (tue par signal). C'est l'echec qu'on verrouille : le run instable ne doit PAS corrompre
    # le tas ni crasher au teardown.
    assert proc.returncode >= 0, (
        "le profil polaire instable a fait crasher le processus par signal (returncode=%d) : "
        "debordement de tampon au remodelage du champ ?\nstdout:\n%s\nstderr:\n%s"
        % (proc.returncode, proc.stdout, proc.stderr)
    )
    # Le scenario lui-meme doit aller jusqu'au bout (instable mais SANS crash) : soit il imprime
    # CHILD_OK (tear down propre apres lectures hote), soit il leve une exception Python CLAIRE
    # (returncode 1, traceback) -- ce qui reste un echec gracieux, pas un crash memoire.
    assert proc.returncode == 0, (
        "le sous-processus ne s'est pas termine proprement (returncode=%d) :\nstdout:\n%s\nstderr:\n%s"
        % (proc.returncode, proc.stdout, proc.stderr)
    )
    assert "CHILD_OK" in proc.stdout, (
        "le scenario instable n'a pas atteint le teardown propre :\nstdout:\n%s\nstderr:\n%s"
        % (proc.stdout, proc.stderr)
    )


if __name__ == "__main__":
    test_unstable_polar_profile_does_not_crash_at_teardown()
    print("OK test_polar_teardown_stability")
