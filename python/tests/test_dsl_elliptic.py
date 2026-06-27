"""Test du codegen elliptique (emit_cpp_elliptic) : meme mecanique que source / flux.

La brique de second membre generee (rhs(U)) doit reproduire pops::ChargeDensity ecrite a la main.
Pur Python ; gate sur compilateur + en-tetes pops, sinon skip propre.
"""
import os
import shutil
import subprocess
import tempfile

from pops.physics.model import HyperbolicModel

Q = -1.0
INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def build_charge():
    e = HyperbolicModel("charge")
    (rho,) = e.conservative_vars("rho")
    e.set_elliptic_rhs(Q * rho)   # densite de charge f = q n
    return e


HARNESS = r"""
#include <pops/physics/bricks/bricks.hpp>
%s
#include <cstdio>
#include <cmath>

int main() {
  pops::ChargeDensity ref; ref.q = %r;
  pops_generated::GenCharge gen;
  const double S[] = {0.0, 0.5, 1.0, 2.5, -0.3};
  double maxdiff = 0.0;
  for (int k=0;k<5;++k){
    pops::StateVec<1> u{}; u[0]=S[k];
    double d = std::fabs(ref.rhs(u) - gen.rhs(u));
    if (d>maxdiff) maxdiff=d;
  }
  printf("%%.17g\n", maxdiff);
  return 0;
}
"""


def main():
    e = build_charge()
    src = e.emit_cpp_elliptic(name="GenCharge")
    assert "struct GenCharge" in src and "rhs(const State& U)" in src, "forme du struct inattendue"
    print("OK  emit_cpp_elliptic : struct genere (%d lignes)" % src.count("\n"))

    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes pops absents -> verification sautee")
        print("test_dsl_elliptic : OK (forme du struct seulement)")
        return

    prog = HARNESS % (src, Q)
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "ell.cpp")
        exe = os.path.join(tmp, "ell")
        with open(cpp, "w") as f:
            f.write(prog)
        subprocess.run([cxx, "-std=c++20", "-O2", "-I", INCLUDE, cpp, "-o", exe], check=True)
        out = subprocess.run([exe], capture_output=True, text=True, check=True).stdout

    d = float(out.strip())
    assert d < 1e-12, "elliptique genere != pops::ChargeDensity (ecart max %.2e)" % d
    print("OK  GenCharge::rhs == pops::ChargeDensity{%g} (ecart max %.1e)" % (Q, d))
    print("test_dsl_elliptic : tout est vert")


if __name__ == "__main__":
    main()
