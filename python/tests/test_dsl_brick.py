"""Test de l'emballage en BRIQUE compilee (etape 2bis du DSL).

emit_cpp_brick() produit un struct C++ cense satisfaire le concept adc::HyperbolicModel. Ce test :
(1) genere la brique pour Euler ; (2) si un compilateur + les en-tetes adc sont presents, compile un
programme qui inclut les vrais en-tetes adc, AFFIRME static_assert(adc::HyperbolicModel<brique>), et
compare chaque methode (flux, max_wave_speed, to_primitive, to_conservative) a la brique ECRITE A LA
MAIN adc::Euler sur des etats deterministes. La compilation echoue si le concept n'est pas satisfait ;
le programme imprime l'ecart max, qu'on exige < 1e-12. Lance avec python3.
"""
import os
import shutil
import subprocess
import tempfile

from adc import dsl

GAMMA = 1.4
INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def build_euler_brick():
    """Euler en formules + layout primitif + inverse cons<-prim, pret pour emit_cpp_brick."""
    e = dsl.HyperbolicModel("euler")
    rho, rhou, rhov, E = e.conservative_vars("rho", "rho_u", "rho_v", "E")
    u = e.primitive("u", rhou / rho)
    v = e.primitive("v", rhov / rho)
    p = e.primitive("p", (GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v)))
    H = (E + p) / rho
    c = dsl.sqrt(GAMMA * p / rho)
    e.set_flux(x=[rhou, rhou * u + p, rhou * v, rho * H * u],
               y=[rhov, rhov * u, rhov * v + p, rho * H * v])
    e.set_eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    # layout de Prim = (rho, u, v, p) ; inverse explicite pour to_conservative
    e.set_primitive_state(rho, u, v, p)
    e.set_conservative_from([rho, rho * u, rho * v, p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)])
    return e


HARNESS = r"""
#include <adc/model/euler.hpp>
#include <adc/core/physical_model.hpp>
%s
#include <cstdio>
#include <cmath>

static_assert(adc::HyperbolicModel<adc::Euler>, "oracle Euler non conforme (setup du test)");
static_assert(adc::HyperbolicModel<adc_generated::EulerGen>, "brique generee non conforme au concept");

int main() {
  adc::Euler ref; ref.gamma = %r;
  adc_generated::EulerGen gen;
  adc::Aux aux{};
  const double S[][4] = {{1.0,0.2,-0.1,2.5},{2.0,0.5,0.3,6.0},{0.5,-0.2,0.1,1.8},{1.5,0.0,0.0,3.0}};
  const int n = sizeof(S)/sizeof(S[0]);
  double maxdiff = 0.0;
  auto upd = [&](double a, double b){ double d = std::fabs(a-b); if (d>maxdiff) maxdiff=d; };
  for (int k=0;k<n;++k){
    adc::StateVec<4> u{}; for(int i=0;i<4;++i) u[i]=S[k][i];
    for (int dir=0; dir<2; ++dir){
      auto fr = ref.flux(u,aux,dir); auto fg = gen.flux(u,aux,dir);
      for(int i=0;i<4;++i) upd(fr[i], fg[i]);
      upd(ref.max_wave_speed(u,aux,dir), gen.max_wave_speed(u,aux,dir));
    }
    auto pr = ref.to_primitive(u); auto pg = gen.to_primitive(u);
    for(int i=0;i<4;++i) upd(pr[i], pg[i]);
    auto ur = ref.to_conservative(pr); auto ug = gen.to_conservative(pg);
    for(int i=0;i<4;++i) upd(ur[i], ug[i]);
  }
  printf("%%.17g\n", maxdiff);
  return 0;
}
"""


def main():
    e = build_euler_brick()
    brick = e.emit_cpp_brick(name="EulerGen")

    # (1) forme de la brique (sans compilateur)
    assert "struct EulerGen {" in brick
    for m in ("State flux(", "max_wave_speed(", "to_primitive(", "to_conservative(",
              "conservative_vars()", "primitive_vars()", "using State", "using Prim"):
        assert m in brick, "membre attendu absent : %s" % m
    print("OK  emit_cpp_brick : struct genere (%d lignes)" % brick.count("\n"))

    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes adc absents -> verification sautee (%s)" % INCLUDE)
        print("test_dsl_brick : OK (forme du struct seulement)")
        return

    prog = HARNESS % (brick, GAMMA)
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "brick.cpp")
        exe = os.path.join(tmp, "brick")
        with open(cpp, "w") as f:
            f.write(prog)
        # le coeur adc est propre en C++20 (concepts) ; -I include suffit (header-only).
        subprocess.run([cxx, "-std=c++20", "-O2", "-I", INCLUDE, cpp, "-o", exe], check=True)
        out = subprocess.run([exe], capture_output=True, text=True, check=True).stdout

    maxdiff = float(out.strip())
    assert maxdiff < 1e-12, "brique generee != adc::Euler (ecart max %.2e)" % maxdiff
    print("OK  static_assert(HyperbolicModel<EulerGen>) + brique == adc::Euler (ecart max %.1e)"
          % maxdiff)
    print("test_dsl_brick : tout est vert")


if __name__ == "__main__":
    main()
