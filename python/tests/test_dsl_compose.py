"""Test de COMPOSITION de la brique generee (etape 2bis du DSL, suite).

emit_cpp_brick() produit un struct hyperbolique ; ce test verifie qu'il se COMPOSE comme n'importe
quelle brique manuelle. On l'insere dans un pops::CompositeModel<EulerGen, NoSource, ChargeDensity> et
on exige : (1) static_assert(pops::PhysicalModel<Gen>) et static_assert(pops::HyperbolicModel<EulerGen>)
(la composition compile et satisfait le contrat du modele physique) ; (2) sur des etats deterministes
(rho>0, p>0) et dir 0/1, le compose genere egale le compose ECRIT A LA MAIN (Euler oracle) sur flux,
max_wave_speed et elliptic_rhs. La compilation echoue si un concept n'est pas satisfait ; le programme
imprime l'ecart max, qu'on exige < 1e-12. Lance avec python3.
"""
import os
import shutil
import subprocess
import tempfile

from pops import dsl

GAMMA = 1.4
INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def build_euler_brick():
    """Euler en formules + layout primitif + inverse cons<-prim, pret pour emit_cpp_brick.

    Identique au montage de test_dsl_brick.py : layout de Prim = (rho, u, v, p), inverse explicite
    fourni via set_conservative_from (le DSL ne sait pas inverser symboliquement)."""
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
    e.set_primitive_state(rho, u, v, p)
    e.set_conservative_from([rho, rho * u, rho * v, p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)])
    return e


HARNESS = r"""
#include <pops/physics/fluids/euler.hpp>
#include <pops/physics/bricks/bricks.hpp>
#include <pops/core/model/physical_model.hpp>
%s
#include <cstdio>
#include <cmath>

// La brique generee doit etre un modele hyperbolique conforme...
static_assert(pops::HyperbolicModel<pops_generated::EulerGen>, "brique generee non conforme au concept");

// ...et se composer en un PhysicalModel complet (hyperbolique + source + elliptique).
using Gen = pops::CompositeModel<pops_generated::EulerGen, pops::NoSource, pops::ChargeDensity>;
using Ref = pops::CompositeModel<pops::Euler,              pops::NoSource, pops::ChargeDensity>;
static_assert(pops::PhysicalModel<Gen>, "compose genere non conforme au concept PhysicalModel");
static_assert(pops::PhysicalModel<Ref>, "compose oracle non conforme (setup du test)");

int main() {
  Gen gen;                       // EulerGen inline gamma dans ses formules (pas de membre gamma).
  Ref ref;  ref.hyp.gamma = %r;   // on aligne l'oracle ; q par defaut = 1 (ChargeDensity) des deux cotes.
  pops::Aux aux{};
  const double S[][4] = {{1.0,0.2,-0.1,2.5},{2.0,0.5,0.3,6.0},{0.5,-0.2,0.1,1.8},{1.5,0.0,0.0,3.0}};
  const int n = sizeof(S)/sizeof(S[0]);
  double maxdiff = 0.0;
  auto upd = [&](double a, double b){ double d = std::fabs(a-b); if (d>maxdiff) maxdiff=d; };
  for (int k=0;k<n;++k){
    pops::StateVec<4> u{}; for(int i=0;i<4;++i) u[i]=S[k][i];
    for (int dir=0; dir<2; ++dir){
      auto fr = ref.flux(u,aux,dir); auto fg = gen.flux(u,aux,dir);
      for(int i=0;i<4;++i) upd(fr[i], fg[i]);
      upd(ref.max_wave_speed(u,aux,dir), gen.max_wave_speed(u,aux,dir));
    }
    upd(ref.elliptic_rhs(u), gen.elliptic_rhs(u));
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
        print("test_dsl_compose : OK (forme du struct seulement)")
        return

    prog = HARNESS % (brick, GAMMA)
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "compose.cpp")
        exe = os.path.join(tmp, "compose")
        with open(cpp, "w") as f:
            f.write(prog)
        # le coeur adc est propre en C++20 (concepts) ; -I include suffit (header-only).
        subprocess.run([cxx, "-std=c++20", "-O2", "-I", INCLUDE, cpp, "-o", exe], check=True)
        out = subprocess.run([exe], capture_output=True, text=True, check=True).stdout

    maxdiff = float(out.strip())
    assert maxdiff < 1e-12, "compose genere != compose oracle (ecart max %.2e)" % maxdiff
    print("OK  static_assert(PhysicalModel<Gen>) + CompositeModel(EulerGen) == CompositeModel(Euler)"
          " (ecart max %.1e)" % maxdiff)
    print("test_dsl_compose : tout est vert")


if __name__ == "__main__":
    main()
