// Test du flux de Roe (Euler 2D) + eigenvalues(). Deux verifications rigoureuses :
//  (1) consistance : F*(U, U) == flux(U) (dissipation nulle a etat constant) ;
//  (2) propriete de Roe via l'amont SUPERSONIQUE : si toutes les valeurs propres sont de meme signe
//      (ecoulement supersonique), F* doit valoir EXACTEMENT le flux amont. C'est equivalent a
//      F_R - F_L = A_roe (U_R - U_L) : ca ne passe que si la decomposition en ondes est correcte.
#include <adc/model/euler.hpp>
#include <adc/operator/numerical_flux.hpp>

#include <cmath>
#include <cstdio>

using State = adc::StateVec<4>;

static State cons(double rho, double u, double v, double p, double gamma) {
  State U{};
  U[0] = rho;
  U[1] = rho * u;
  U[2] = rho * v;
  U[3] = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v);
  return U;
}

static double maxdiff(const State& a, const State& b) {
  double m = 0;
  for (int c = 0; c < 4; ++c) m = std::fmax(m, std::fabs(a[c] - b[c]));
  return m;
}

int main() {
  adc::Euler e;
  e.gamma = 1.4;
  adc::RoeFlux roe;
  adc::Aux a{};

  // (1) consistance a etat constant, deux etats subsoniques, x et y
  for (const State U : {cons(1.2, 0.3, -0.1, 1.5, 1.4), cons(0.7, -0.2, 0.4, 0.9, 1.4)})
    for (int dir = 0; dir < 2; ++dir) {
      const double d = maxdiff(roe(e, U, a, U, a, dir), e.flux(U, a, dir));
      if (d > 1e-12) {
        std::printf("FAIL consistance Roe (dir %d) : %.3e\n", dir, d);
        return 1;
      }
    }
  std::printf("OK  Roe consistant : F*(U,U) == flux(U)\n");

  // (2) supersonique +x : un >> c pour L ET R -> F* == flux amont (gauche), exact
  {
    const State UL = cons(1.0, 8.0, 0.5, 1.0, 1.4);
    const State UR = cons(1.5, 12.0, -0.3, 1.3, 1.4);  // un ~ 10 >> c ~ 1.2
    const double d = maxdiff(roe(e, UL, a, UR, a, 0), e.flux(UL, a, 0));
    if (d > 1e-9) {
      std::printf("FAIL Roe supersonique +x (devrait valoir le flux amont gauche) : %.3e\n", d);
      return 1;
    }
  }
  // supersonique -x : un << -c -> F* == flux amont (droit)
  {
    const State UL = cons(1.2, -12.0, 0.0, 1.1, 1.4);
    const State UR = cons(0.9, -9.0, 0.4, 0.8, 1.4);
    const double d = maxdiff(roe(e, UL, a, UR, a, 0), e.flux(UR, a, 0));
    if (d > 1e-9) {
      std::printf("FAIL Roe supersonique -x (devrait valoir le flux amont droit) : %.3e\n", d);
      return 1;
    }
  }
  // supersonique +y
  {
    const State UL = cons(1.0, 0.2, 9.0, 1.0, 1.4);
    const State UR = cons(1.4, -0.1, 13.0, 1.2, 1.4);
    const double d = maxdiff(roe(e, UL, a, UR, a, 1), e.flux(UL, a, 1));
    if (d > 1e-9) {
      std::printf("FAIL Roe supersonique +y : %.3e\n", d);
      return 1;
    }
  }
  std::printf("OK  Roe : propriete amont supersonique (F* == flux amont) -> decomposition correcte\n");

  // (3) eigenvalues() : (vn-c, vn, vn, vn+c)
  {
    const State U = cons(1.0, 0.5, -0.2, 1.0, 1.4);
    const double c = std::sqrt(1.4 * 1.0 / 1.0);
    const State ev = e.eigenvalues(U, a, 0);
    if (std::fabs(ev[0] - (0.5 - c)) > 1e-12 || std::fabs(ev[1] - 0.5) > 1e-12 ||
        std::fabs(ev[2] - 0.5) > 1e-12 || std::fabs(ev[3] - (0.5 + c)) > 1e-12) {
      std::printf("FAIL eigenvalues Euler\n");
      return 1;
    }
  }
  std::printf("OK  eigenvalues() = (v_dir - c, v_dir, v_dir, v_dir + c)\n");
  std::printf("test_roe_flux : tout est vert\n");
  return 0;
}
