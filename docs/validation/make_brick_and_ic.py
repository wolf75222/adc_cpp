#!/usr/bin/env python3
"""Genere (1) hyqmom15_brick.hpp -- la brique C++ emise par la DSL depuis le modele VALIDE
(exact_speeds + sources + Poisson, params caves aux valeurs du scenario) et (2) ic_<n>.raw --
l'etat initial diocotron calcule par le python valide (diocotron_state), en binaire.

Usage : python make_brick_and_ic.py --n 256 [--out .]"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.environ.get("POPS_CASES",
                       os.path.join(os.path.dirname(os.path.dirname(HERE)), "adc_cases"))
sys.path.insert(0, os.path.join(CASES, "hyqmom15"))
sys.path.insert(0, CASES)

from run_diocotron import DEBYE, OMEGA_P, diocotron_state  # noqa: E402
from model import build_moment_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args()

    # PIEGE rho_background : il est CUIT dans la brique elliptique mais sa valeur (moyenne de
    # M00 du scenario discretise) DEPEND de n -- un fond faux rend le rhs de Poisson a moyenne
    # non nulle et fait deriver le MG periodique. On emet donc une brique elliptique PAR n
    # (Hyqmom15Ell<n>) ; flux/vitesses (Hyp) et source (Src) n'en dependent pas (partagees).
    rho_bgs = {}
    for n in args.ns:
        U0 = diocotron_state(n)
        rho_bgs[n] = float(U0[0].mean())
        ic = os.path.join(args.out, "ic_%d.raw" % n)
        np.ascontiguousarray(U0, dtype=np.float64).tofile(ic)
        print("IC %s : rho_bg=%.12g, %.1f Mo" % (ic, rho_bgs[n], os.path.getsize(ic) / 1e6))

    def build(rho_bg):
        return build_moment_model(name="hyqmom15_gpu", robust=False, with_sources=True,
                                  q_over_m=1.0, omega_c=0.0, debye=DEBYE,
                                  rho_background=rho_bg, omega_p=OMEGA_P, exact_speeds=True)

    m = build(rho_bgs[args.ns[0]])
    parts = [m._m.emit_cpp_brick(name="Hyqmom15Hyp"),
             m._m.emit_cpp_source(name="Hyqmom15Src")]
    for n in args.ns:
        parts.append(build(rho_bgs[n])._m.emit_cpp_elliptic(name="Hyqmom15Ell%d" % n))
    hpp = os.path.join(args.out, "hyqmom15_brick.hpp")
    with open(hpp, "w") as f:
        f.write("\n".join(parts))
    blob = "".join(parts)
    print("brique %s : %d lignes (Hyp/Src partagees + Ell par n)" % (hpp, blob.count(chr(10))))
    for marker in (["real_eig_minmax", "struct Hyqmom15Src"]
                   + ["struct Hyqmom15Ell%d" % n for n in args.ns]):
        print("  contient '%s' : %s" % (marker, marker in blob))


if __name__ == "__main__":
    main()
