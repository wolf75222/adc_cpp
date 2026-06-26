"""JIT reel (dlopen) : formules Python -> C++ genere -> .so compile a la volee -> CHARGE dans le
process Python (ctypes) -> appele. Contrairement a test_dsl_jit (qui compile un executable driver),
ici on charge la bibliotheque dans le processus courant et on appelle la fonction flux generee, qu'on
compare a l'interprete numpy (meme arbre, deux backends executes cote a cote).

NB : c'est le JIT du NOYAU genere (un .so charge + appele), pas le branchement dans le solveur
template compile (qui exigerait une interface de modele type-erased, cf. ARCHITECTURE_CIBLE.md).
"""
import ctypes
import os
import shutil
import subprocess
import tempfile

import numpy as np

from pops import dsl

GAMMA = 1.4


def build_euler():
    e = dsl.HyperbolicModel("euler")
    rho, rhou, rhov, E = e.conservative_vars("rho", "rho_u", "rho_v", "E")
    u = e.primitive("u", rhou / rho)
    v = e.primitive("v", rhov / rho)
    p = e.primitive("p", (GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v)))
    H = (E + p) / rho
    e.set_flux(x=[rhou, rhou * u + p, rhou * v, rho * H * u],
               y=[rhov, rhov * u, rhov * v + p, rho * H * v])
    return e


def main():
    e = build_euler()
    flux_src = e.emit_cpp(func="euler")   # template <class Real> void euler_flux(const Real*, Real*, int)
    # wrapper C ABI : instancie la version double, exposable a ctypes
    src = flux_src + ('\nextern "C" void gen_flux(const double* U, double* F, int dir) {\n'
                      '  euler_flux<double>(U, F, dir);\n}\n')

    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        print("skip  pas de compilateur C++ -> JIT dlopen saute")
        print("test_dsl_jitlib : OK (rien a compiler)")
        return

    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "gen.cpp")
        so = os.path.join(tmp, "libgen.so")
        with open(cpp, "w") as f:
            f.write(src)
        # compilation a la volee en bibliotheque partagee, puis chargement dans CE process
        subprocess.run([cxx, "-shared", "-fPIC", "-std=c++20", "-O2", cpp, "-o", so], check=True)
        lib = ctypes.CDLL(so)
        lib.gen_flux.restype = None
        lib.gen_flux.argtypes = [ctypes.POINTER(ctypes.c_double),
                                 ctypes.POINTER(ctypes.c_double), ctypes.c_int]

        dptr = ctypes.POINTER(ctypes.c_double)
        states = [(1.0, 0.2, -0.1, 2.5), (2.0, 0.5, 0.3, 6.0),
                  (0.5, -0.2, 0.1, 1.8), (1.5, 0.0, 0.0, 3.0)]
        maxdiff = 0.0
        for s in states:
            U = np.array(s, dtype=np.float64)
            for d in (0, 1):
                F = np.zeros(4, dtype=np.float64)
                lib.gen_flux(U.ctypes.data_as(dptr), F.ctypes.data_as(dptr), d)   # appel du .so JIT
                Fi = e.flux(U.reshape(4, 1, 1), {}, d).reshape(4)                 # interprete numpy
                maxdiff = max(maxdiff, float(np.max(np.abs(F - Fi))))

    assert maxdiff < 1e-12, "flux JIT (.so dlopen) != interprete numpy (ecart max %.2e)" % maxdiff
    print("OK  flux JIT compile en .so + charge (ctypes) == interprete numpy (ecart max %.1e)" % maxdiff)
    print("test_dsl_jitlib : tout est vert")


if __name__ == "__main__":
    main()
