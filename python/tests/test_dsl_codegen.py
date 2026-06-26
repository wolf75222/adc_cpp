"""Test du codegen C++ du mini-DSL (etape 2 : arbre symbolique -> C++ compilable).

Verifie : (1) emit_cpp() produit une source C++ plausible (signature, locals, assignations) ;
(2) si un compilateur C++ est present, la fonction flux GENEREE est compilee, executee sur des
etats deterministes, et son resultat compare a l'interprete numpy (meme arbre, deux backends).
Pur Python ; lance avec python3 (PYTHONPATH = paquet adc construit). Le compilateur est celui qui
a deja servi a batir _pops, donc disponible dans le job CI correspondant.
"""
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
    c = dsl.sqrt(GAMMA * p / rho)
    e.set_eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    return e


def main():
    e = build_euler()
    src = e.emit_cpp(func="euler")

    # (1) la source generee a la bonne forme (sans compilateur)
    assert "void euler_flux(const Real* U, Real* F, int dir)" in src, "signature attendue absente"
    assert "const Real rho = U[0];" in src and "const Real E = U[3];" in src, "locals cons absents"
    assert "const Real p = " in src and "const Real u = " in src, "primitives absentes"
    assert src.count("F[") == 8, "attendu 4 composantes x 2 directions"
    assert "std::pow" not in src, "Euler ne devrait pas produire de pow"
    print("OK  emit_cpp : source C++ generee (%d lignes)" % src.count("\n"))

    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        print("skip  pas de compilateur C++ -> verification numerique sautee")
        print("test_dsl_codegen : OK (forme de la source seulement)")
        return

    # (2) etats deterministes (rho > 0, p > 0) ; le main genere imprime F en pleine precision
    states = [(1.0, 0.2, -0.1, 2.5), (2.0, 0.5, 0.3, 6.0),
              (0.5, -0.2, 0.1, 1.8), (1.5, 0.0, 0.0, 3.0)]
    lits = ",".join("{%s}" % ",".join(repr(float(x)) for x in s) for s in states)
    main_cpp = src + (
        "#include <cstdio>\n"
        "int main() {\n"
        "  const double S[%d][4] = {%s};\n"
        "  for (int k = 0; k < %d; ++k) {\n"
        "    double F[4];\n"
        "    for (int d = 0; d < 2; ++d) {\n"
        "      euler_flux<double>(S[k], F, d);\n"
        "      printf(\"%%.17g %%.17g %%.17g %%.17g\\n\", F[0], F[1], F[2], F[3]);\n"
        "    }\n"
        "  }\n"
        "  return 0;\n"
        "}\n"
    ) % (len(states), lits, len(states))

    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "gen.cpp")
        exe = os.path.join(tmp, "gen")
        with open(cpp, "w") as f:
            f.write(main_cpp)
        subprocess.run([cxx, "-std=c++17", "-O2", cpp, "-o", exe], check=True)
        out = subprocess.run([exe], capture_output=True, text=True, check=True).stdout

    rows = [list(map(float, line.split())) for line in out.strip().splitlines()]
    k = 0
    for s in states:
        U = np.array(s, dtype=float).reshape(4, 1, 1)
        for d in (0, 1):
            f_interp = e.flux(U, {}, d).reshape(4)
            f_cpp = np.array(rows[k]); k += 1
            assert np.allclose(f_interp, f_cpp, rtol=1e-12, atol=1e-12), \
                "flux C++ != interprete (etat %s, dir %d) : %s vs %s" % (s, d, f_interp, f_cpp)
    print("OK  flux C++ genere == interprete numpy (%d etats x 2 directions, compile %s)"
          % (len(states), os.path.basename(cxx)))
    print("test_dsl_codegen : tout est vert")


if __name__ == "__main__":
    main()
