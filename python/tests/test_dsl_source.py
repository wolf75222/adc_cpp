"""Test de la brique de SOURCE generee (etape 2ter du DSL).

emit_cpp_source() produit un struct C++ expose apply(U, a) cense reproduire une brique de source
ECRITE A LA MAIN. Ce test : (1) construit le modele a 4 variables avec la source (q/m) rho E (forme
electrostatique), aux = grad_x/grad_y ; (2) genere la brique GenForce ; (3) si un compilateur et les
en-tetes adc sont presents, compile un programme qui inclut les vrais en-tetes adc et compare, sur des
etats ET des aux deterministes, GenForce::apply a pops::PotentialForce{-1.0}::apply composante par
composante. Le programme imprime l'ecart max, qu'on exige < 1e-12. Lance avec python3.
"""
import os
import shutil
import subprocess
import tempfile

from pops import dsl

QOM = -1.0
INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def build_force_model():
    """Modele a 4 var avec la source (q/m) rho E, E = -grad phi (aux grad_x/grad_y).

    Source par composante (layout (rho, rho u, rho v, E)) :
      S[0] = 0
      S[1] = qom * rho * Ex          avec Ex = -grad_x
      S[2] = qom * rho * Ey          avec Ey = -grad_y
      S[3] = qom * (rho_u Ex + rho_v Ey)   (travail sur l'energie)
    Identique a pops::PotentialForce{qom} sur 4 variables.
    """
    m = dsl.HyperbolicModel("force")
    rho, rho_u, rho_v, E = m.conservative_vars("rho", "rho_u", "rho_v", "E")
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    m.set_source([
        0,
        QOM * rho * (-gx),
        QOM * rho * (-gy),
        QOM * (rho_u * (-gx) + rho_v * (-gy)),
    ])
    return m


HARNESS = r"""
#include <pops/physics/bricks/bricks.hpp>
%s
#include <cstdio>
#include <cmath>

int main() {
  pops_generated::GenForce gen;
  pops::PotentialForce ref{%r};

  // etats deterministes (rho > 0) et aux deterministes (gradients varies, signes mixtes).
  const double S[][4] = {{1.0,0.2,-0.1,2.5},{2.0,0.5,0.3,6.0},
                         {0.5,-0.2,0.1,1.8},{1.5,0.0,0.0,3.0},{3.0,-1.2,0.7,9.0}};
  const double G[][2] = {{-0.3,0.7},{0.4,-0.9},{1.1,0.2},{0.0,-0.5},{-0.6,-0.6}};
  const int ns = sizeof(S)/sizeof(S[0]);
  const int ng = sizeof(G)/sizeof(G[0]);

  double maxdiff = 0.0;
  auto upd = [&](double a, double b){ double d = std::fabs(a-b); if (d>maxdiff) maxdiff=d; };
  for (int k=0;k<ns;++k){
    pops::StateVec<4> u{}; for(int i=0;i<4;++i) u[i]=S[k][i];
    for (int j=0;j<ng;++j){
      pops::Aux a{}; a.grad_x = G[j][0]; a.grad_y = G[j][1];
      auto sg = gen.apply(u, a);
      auto sr = ref.apply(u, a);
      for(int i=0;i<4;++i) upd(sg[i], sr[i]);
    }
  }
  printf("%%.17g\n", maxdiff);
  return 0;
}
"""


def main():
    m = build_force_model()
    struct = m.emit_cpp_source(name="GenForce")

    # (1) forme de la brique (sans compilateur)
    assert "struct GenForce {" in struct
    for token in ("apply(const pops::StateVec<4>&", "const pops::Aux& a",
                  "const pops::Real grad_x = a.grad_x;", "const pops::Real grad_y = a.grad_y;",
                  "pops::StateVec<4> S{};"):
        assert token in struct, "membre attendu absent : %s" % token
    print("OK  emit_cpp_source : struct genere (%d lignes)" % struct.count("\n"))

    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes adc absents -> verification sautee (%s)" % INCLUDE)
        print("test_dsl_source : OK (forme du struct seulement)")
        return

    prog = HARNESS % (struct, QOM)
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "source.cpp")
        exe = os.path.join(tmp, "source")
        with open(cpp, "w") as f:
            f.write(prog)
        # le coeur adc est header-only et propre en C++20 ; -I include suffit.
        subprocess.run([cxx, "-std=c++20", "-O2", "-I", INCLUDE, cpp, "-o", exe], check=True)
        out = subprocess.run([exe], capture_output=True, text=True, check=True).stdout

    maxdiff = float(out.strip())
    assert maxdiff < 1e-12, "source generee != pops::PotentialForce (ecart max %.2e)" % maxdiff
    print("OK  GenForce::apply == pops::PotentialForce{%.1f} (ecart max %.1e)" % (QOM, maxdiff))
    print("test_dsl_source : tout est vert")


if __name__ == "__main__":
    main()
