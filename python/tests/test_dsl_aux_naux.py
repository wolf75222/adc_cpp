"""Test du chantier Aux extensible cote DSL (increment 6).

Quand une formule du modele symbolique lit un champ aux SUPPLEMENTAIRE (aux('B_z')), la brique
generee declare `static constexpr int n_aux = 4` : CompositeModel le propage (cf. test_aux_composite)
et le runtime System dimensionne/peuple le canal aux partage (cf. test_aux_runtime_bz). Un modele qui
ne lit que phi/grad_x/grad_y reste a la largeur de base (pas de n_aux emis -> bit-identique). On
verifie : (1) le helper aux_n_aux ; (2) l'emission conditionnelle de n_aux ; (3) la retro-compat ;
(4) la validation des noms ; (5) si un compilateur est present, que la brique B_z compile et lit B_z.
"""
import os
import shutil
import subprocess
import tempfile

from pops.physics.aux import aux_n_aux
from pops.physics.model import HyperbolicModel

INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def main():
    # (1) helper aux_n_aux : largeur canonique du canal aux.
    assert aux_n_aux([]) == 3
    assert aux_n_aux(["grad_x", "grad_y"]) == 3
    assert aux_n_aux(["B_z"]) == 4
    assert aux_n_aux(["grad_x", "B_z"]) == 4
    print("OK  aux_n_aux : base=3, B_z=4")

    # (2) une source qui lit B_z -> brique avec n_aux = 4.
    m = HyperbolicModel("mag")
    (nn,) = m.conservative_vars("n")
    bz = m.aux("B_z")
    m.set_source([bz * nn])  # S = B_z * n
    src = m.emit_cpp_source(name="GenBzSrc")
    assert "static constexpr int n_aux = 4;" in src, "n_aux=4 absent de la source B_z"
    assert "const pops::Real B_z = a.B_z;" in src, "lecture a.B_z absente"
    print("OK  emit_cpp_source(B_z) declare n_aux = 4")

    # (3) retro-compat : une source qui ne lit que grad n'emet PAS de n_aux.
    m2 = HyperbolicModel("plain")
    (n2,) = m2.conservative_vars("n")
    gx = m2.aux("grad_x")
    m2.set_source([gx * n2])
    src2 = m2.emit_cpp_source(name="GenPlainSrc")
    assert "n_aux" not in src2, "n_aux ne doit pas etre emis pour un modele de base"
    print("OK  emit_cpp_source(base) sans n_aux (bit-identique)")

    # (4) validation : un nom aux inconnu est rejete (doit etre une composante de pops::Aux).
    try:
        aux_n_aux(["n_e"])  # nom absent de la disposition canonique
        raise AssertionError("aux_n_aux aurait du lever ValueError sur un nom inconnu")
    except ValueError:
        pass
    print("OK  aux_n_aux rejette un nom aux inconnu")

    # (5) compile-check : la brique B_z compile et lit bien B_z (sur les vrais en-tetes pops).
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes pops absents -> compile-check sautee (%s)" % INCLUDE)
        print("test_dsl_aux_naux : OK (forme seulement)")
        return

    harness = (
        "#include <pops/core/state/state.hpp>\n"
        "#include <pops/core/foundation/types.hpp>\n"
        + src
        + "#include <cstdio>\n"
        "static_assert(pops_generated::GenBzSrc::n_aux == 4, \"n_aux propage\");\n"
        "int main() {\n"
        "  pops_generated::GenBzSrc g; pops::StateVec<1> u{}; u[0] = 2.0;\n"
        "  pops::Aux a{}; a.B_z = 0.5;\n"
        "  auto s = g.apply(u, a);\n"
        "  printf(\"%.17g\\n\", s[0]);\n"
        "  return 0;\n"
        "}\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "bz.cpp")
        exe = os.path.join(tmp, "bz")
        with open(cpp, "w") as f:
            f.write(harness)
        subprocess.run([cxx, "-std=c++20", "-O2", "-I", INCLUDE, cpp, "-o", exe], check=True)
        out = subprocess.run([exe], capture_output=True, text=True, check=True).stdout
    val = float(out.strip())
    assert abs(val - 1.0) < 1e-12, "GenBzSrc::apply ne lit pas B_z (S=%g, attendu 1.0)" % val
    print("OK  GenBzSrc compile et lit B_z (S = B_z * n = %.3g)" % val)
    print("test_dsl_aux_naux : tout est vert")


if __name__ == "__main__":
    main()
