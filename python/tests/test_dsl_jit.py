"""Test JIT-lite END-TO-END du DSL (le coeur de la demonstration).

Chaine complete : formules Python -> .hpp genere (emit_cpp_brick) -> g++ compile un DRIVER
volumes-finis -> tourne IDENTIQUE a la brique ecrite a la main pops::Euler. On ne se contente plus
de comparer methode par methode (cf. test_dsl_brick.py) : on FAIT TOURNER un vrai pas de schema.

Le driver calcule, sur une petite grille periodique n x n, UNE passe de residu Rusanov
res = -div(F*) (flux numerique de Rusanov en x et en y), pour DEUX modeles composites :
  - Gen = CompositeModel<pops_generated::EulerGen, NoSource, ChargeDensity> (brique GENEREE par le DSL)
  - Ref = CompositeModel<pops::Euler,            NoSource, ChargeDensity> (oracle ECRIT A LA MAIN)
a partir d'un MEME champ initial deterministe (rho=1, vitesse nulle, bulle de pression). Il imprime
l'ecart max sur tout le residu, qu'on exige < 1e-10. Gate sur compilateur + en-tetes pops (SKIP propre
sinon). Lance avec python3.
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

    Identique a la construction de test_dsl_brick.py : meme modele symbolique, donc la brique
    generee Est censee reproduire pops::Euler au bit pres (memes formules)."""
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


# Driver volumes-finis : maillage periodique n x n, indexation k = j*n + i, voisins par modulo.
# Champ initial deterministe : rho=1, vitesse nulle, bulle de pression gaussienne centree, energie
# E = p/(gamma-1) (vitesse nulle, pas de terme cinetique). Une passe de residu Rusanov pour Gen et
# Ref ; on compare le residu cellule par cellule, composante par composante.
DRIVER = r"""
#include <pops/physics/fluids/euler.hpp>
#include <pops/physics/bricks/bricks.hpp>
#include <pops/core/model/physical_model.hpp>
%s
#include <vector>
#include <cmath>
#include <cstdio>

using Gen = pops::CompositeModel<pops_generated::EulerGen, pops::NoSource, pops::ChargeDensity>;
using Ref = pops::CompositeModel<pops::Euler,            pops::NoSource, pops::ChargeDensity>;

static_assert(pops::HyperbolicModel<pops::Euler>,                "oracle Euler non conforme (setup)");
static_assert(pops::HyperbolicModel<pops_generated::EulerGen>,   "brique generee non conforme");
static_assert(pops::PhysicalModel<Gen>,                         "Gen compose non conforme");
static_assert(pops::PhysicalModel<Ref>,                         "Ref compose non conforme");

static constexpr int n = 24;
static constexpr double GAMMA = %r;
static constexpr double h = 1.0 / n;

// Index periodique k = j*n + i (i = x rapide, j = y lent), voisins par modulo.
static inline int idx(int i, int j) { return ((j %% n + n) %% n) * n + ((i %% n + n) %% n); }

// Flux numerique de Rusanov sur la face entre les cellules UL (gauche/bas) et UR (droite/haut),
// dans la direction dir : a = max(max_wave_speed des 2 cellules), face = 0.5*(FL+FR) - 0.5*a*(UR-UL).
template <class Model>
static pops::StateVec<4> rusanov(const Model& m, const pops::StateVec<4>& UL,
                                const pops::StateVec<4>& UR, const pops::Aux& aux, int dir) {
  const auto FL = m.flux(UL, aux, dir);
  const auto FR = m.flux(UR, aux, dir);
  const double aL = m.max_wave_speed(UL, aux, dir);
  const double aR = m.max_wave_speed(UR, aux, dir);
  const double a = aL > aR ? aL : aR;
  pops::StateVec<4> f{};
  for (int c = 0; c < 4; ++c) f[c] = 0.5 * (FL[c] + FR[c]) - 0.5 * a * (UR[c] - UL[c]);
  return f;
}

// Une passe de residu -div(F*) sur toute la grille (sortie dans res, taille n*n) pour le modele m.
template <class Model>
static void residual(const Model& m, const std::vector<pops::StateVec<4>>& U,
                     std::vector<pops::StateVec<4>>& res) {
  const pops::Aux aux{};  // Euler n'utilise pas aux
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const pops::StateVec<4>& Uc = U[idx(i, j)];
      // faces x : i-1/2 (entre i-1 et i) et i+1/2 (entre i et i+1)
      const auto Fxm = rusanov(m, U[idx(i - 1, j)], Uc,             aux, 0);
      const auto Fxp = rusanov(m, Uc,               U[idx(i + 1, j)], aux, 0);
      // faces y : j-1/2 (entre j-1 et j) et j+1/2 (entre j et j+1)
      const auto Fym = rusanov(m, U[idx(i, j - 1)], Uc,             aux, 1);
      const auto Fyp = rusanov(m, Uc,               U[idx(i, j + 1)], aux, 1);
      pops::StateVec<4>& r = res[idx(i, j)];
      for (int c = 0; c < 4; ++c)
        r[c] = -(Fxp[c] - Fxm[c]) / h - (Fyp[c] - Fym[c]) / h;
    }
  }
}

int main() {
  // Champ initial deterministe : rho=1, vitesse nulle, bulle de pression gaussienne centree.
  std::vector<pops::StateVec<4>> U(n * n);
  const double scale = double(n);  // largeur de la bulle (en cellules^2)
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const double dx = i - n / 2.0, dy = j - n / 2.0;
      const double p = 1.0 + 0.3 * std::exp(-(dx * dx + dy * dy) / scale);
      pops::StateVec<4>& u = U[idx(i, j)];
      u[0] = 1.0;            // rho
      u[1] = 0.0;            // rho u
      u[2] = 0.0;            // rho v
      u[3] = p / (GAMMA - 1.0);  // E (vitesse nulle : pas de terme cinetique)
    }
  }

  Gen gen; gen.hyp = pops_generated::EulerGen{};
  Ref ref; ref.hyp.gamma = GAMMA;

  std::vector<pops::StateVec<4>> rg(n * n), rr(n * n);
  residual(gen, U, rg);
  residual(ref, U, rr);

  double maxdiff = 0.0;
  for (int k = 0; k < n * n; ++k)
    for (int c = 0; c < 4; ++c) {
      const double d = std::fabs(rg[k][c] - rr[k][c]);
      if (d > maxdiff) maxdiff = d;
    }
  printf("%%.17g\n", maxdiff);
  return 0;
}
"""


def main():
    e = build_euler_brick()
    brick = e.emit_cpp_brick(name="EulerGen")

    # (1) forme minimale de la brique (verifiable sans compilateur)
    assert "struct EulerGen {" in brick, "struct EulerGen attendu dans la brique generee"
    print("OK  emit_cpp_brick : struct EulerGen genere (%d lignes)" % brick.count("\n"))

    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes pops absents -> JIT-lite saute (%s)" % INCLUDE)
        print("test_dsl_jit : OK (forme du struct seulement)")
        return

    prog = DRIVER % (brick, GAMMA)
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "jit_driver.cpp")
        exe = os.path.join(tmp, "jit_driver")
        with open(cpp, "w") as f:
            f.write(prog)
        # header-only : -I include suffit ; C++20 pour les concepts (PhysicalModel / HyperbolicModel).
        subprocess.run([cxx, "-std=c++20", "-O2", "-I", INCLUDE, cpp, "-o", exe], check=True)
        out = subprocess.run([exe], capture_output=True, text=True, check=True).stdout

    maxdiff = float(out.strip())
    assert maxdiff < 1e-10, \
        "residu Rusanov de la brique generee != pops::Euler (ecart max %.3e)" % maxdiff
    print("OK  jit-lite : residu Rusanov -div(F*) IDENTIQUE Gen vs pops::Euler (ecart max %.3e, %dx%d)"
          % (maxdiff, 24, 24))
    print("test_dsl_jit : tout est vert")


if __name__ == "__main__":
    main()
