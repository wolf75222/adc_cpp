"""Test de l'elimination des sous-expressions communes (CSE) du codegen.

Verifie que cse=True et cse=False produisent le MEME resultat numerique (la CSE ne change pas la
valeur, seulement le nombre de calculs) mais que la version CSE factorise effectivement : locales
'cseK_' presentes, et moins d'appels std::sqrt (la vitesse du son calculee une fois). Reutilise la
brique Euler et le harnais de comparaison a pops::Euler de test_dsl_brick.
"""
import os
import shutil
import subprocess
import tempfile

import test_dsl_brick as B

INCLUDE = B.INCLUDE


def compile_run(brick):
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    prog = B.HARNESS % (brick, B.GAMMA)
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "b.cpp")
        exe = os.path.join(tmp, "b")
        with open(cpp, "w") as f:
            f.write(prog)
        subprocess.run([cxx, "-std=c++20", "-O2", "-I", INCLUDE, cpp, "-o", exe], check=True)
        return float(subprocess.run([exe], capture_output=True, text=True, check=True).stdout.strip())


def main():
    e = B.build_euler_brick()
    brick_cse = e.emit_cpp_brick(name="EulerGen", cse=True)
    brick_raw = e.emit_cpp_brick(name="EulerGen", cse=False)

    # (1) la CSE factorise reellement : locales cseK_ presentes seulement avec cse=True
    assert "cse0_" in brick_cse and "cse0_" not in brick_raw, "CSE non appliquee / fuite en cse=False"
    # (2) vitesse du son calculee une fois : moins de std::sqrt avec CSE
    assert brick_cse.count("std::sqrt") < brick_raw.count("std::sqrt"), \
        "CSE ne reduit pas les std::sqrt (%d vs %d)" % (brick_cse.count("std::sqrt"),
                                                        brick_raw.count("std::sqrt"))
    print("OK  CSE factorise : std::sqrt %d -> %d, cseK_ presents"
          % (brick_raw.count("std::sqrt"), brick_cse.count("std::sqrt")))

    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes adc absents -> comparaison numerique sautee")
        print("test_dsl_cse : OK (forme seulement)")
        return

    # (3) meme resultat numerique des deux cotes (et tous deux == pops::Euler)
    d_cse = compile_run(brick_cse)
    d_raw = compile_run(brick_raw)
    assert d_cse < 1e-12 and d_raw < 1e-12, "une version ne reproduit pas pops::Euler"
    print("OK  cse et non-cse == pops::Euler (ecarts %.1e / %.1e)" % (d_cse, d_raw))
    print("test_dsl_cse : tout est vert")


if __name__ == "__main__":
    main()
