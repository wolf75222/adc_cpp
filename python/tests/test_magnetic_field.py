#!/usr/bin/env python3
"""Test du binding System.set_magnetic_field (chantier Aux extensible, increment 5).

set_magnetic_field expose, cote Python, le canal aux supplementaire B_z. Pour un modele de
BASE (qui ne lit pas B_z, p.ex. ExB qui ne lit que grad phi), le binding doit etre INERTE :
le canal aux reste a sa largeur de base et le run est strictement celui sans B_z. On verifie
donc (1) que le binding existe et accepte un champ n*n, (2) qu'il n'altere PAS un modele de
base (retro-compat). Le chemin ou un modele LIT vraiment B_z est valide en C++
(tests/test_aux_runtime_bz.cpp), faute de modele B_z accessible depuis Python avant la DSL.
"""
import sys

import numpy as np

import pops

n = 32


def diocotron():
    # modele de base : ExB lit grad phi, PAS B_z (n_aux = 3).
    return pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0),
                     source=pops.NoSource(),
                     elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0))


def make():
    s = pops.System(n=n, L=1.0, periodic=True)
    s.add_block("ne", model=diocotron(), spatial=pops.Spatial(minmod=True))
    s.set_poisson()
    rho = 1.0 + 0.05 * np.cos(2 * np.pi * np.arange(n) / n)[None, :] * np.ones((n, 1))
    s.set_density("ne", rho)
    return s


# (1) le binding existe et accepte un champ B_z n*n (constant ici).
sim = make()
sim.set_magnetic_field(0.5 * np.ones((n, n)))
for _ in range(5):
    sim.step_cfl(0.4)
a = np.array(sim.density("ne"))

# (2) reference SANS B_z : pour un modele de base, set_magnetic_field doit etre inerte.
ref = make()
for _ in range(5):
    ref.step_cfl(0.4)
b = np.array(ref.density("ne"))

err = float(np.max(np.abs(a - b)))
print("  set_magnetic_field inerte pour modele de base : max diff = %.2e" % err)
assert err < 1e-12, "set_magnetic_field altere un modele de base (non retro-compatible)"

print("OK test_magnetic_field")
