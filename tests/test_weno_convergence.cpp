// WENO5-Z : la reconstruction de la valeur de face d'une fonction LISSE depuis ses moyennes
// de cellule est d'ordre 5 (les poids non lineaires WENO-Z tendent vers les poids lineaires
// optimaux 1/10, 6/10, 3/10 en zone reguliere). On verifie l'ordre mesure >= 4.5 et la
// preservation des constantes. Brique de la voie haute precision vers le taux analytique 0.911.

#include <pops/numerics/fv/reconstruction.hpp>

#include <cmath>
#include <cstdio>
#include <initializer_list>

using namespace pops;
static constexpr double kPi = 3.14159265358979323846;

// moyenne de cellule de f(x) = sin(2 pi x) sur [a, b] (primitive exacte).
static double favg(double a, double b) {
  return (std::cos(2 * kPi * a) - std::cos(2 * kPi * b)) / (2 * kPi * (b - a));
}

int main() {
  // 1. preservation des constantes : weno5z(c,c,c,c,c) == c (poids sommes a 1).
  const double c = 3.14;
  if (std::fabs(weno5z(c, c, c, c, c) - c) > 1e-13) {
    std::printf("FAIL constante : %.3e\n", std::fabs(weno5z(c, c, c, c, c) - c));
    return 1;
  }

  // 2. ordre 5 sur une fonction lisse (erreur de face ~ dx^5).
  double prev = 0, last_order = 0;
  for (int N : {32, 64, 128, 256, 512}) {
    const double dx = 1.0 / N;
    double emax = 0;
    for (int i = 3; i < N - 3; ++i) {
      const double xc = (i + 0.5) * dx;
      const double rec =
          weno5z(favg(xc - 2.5 * dx, xc - 1.5 * dx), favg(xc - 1.5 * dx, xc - 0.5 * dx),
                 favg(xc - 0.5 * dx, xc + 0.5 * dx), favg(xc + 0.5 * dx, xc + 1.5 * dx),
                 favg(xc + 1.5 * dx, xc + 2.5 * dx));
      const double exact = std::sin(2 * kPi * (xc + 0.5 * dx));  // valeur a la face x = xc + dx/2
      emax = std::fmax(emax, std::fabs(rec - exact));
    }
    last_order = prev > 0 ? std::log(prev / emax) / std::log(2.0) : 0;
    std::printf("N=%4d  err_inf=%.3e  ordre=%.2f\n", N, emax, last_order);
    prev = emax;
  }
  if (last_order < 4.5) {
    std::printf("FAIL : ordre WENO5 mesure %.2f < 4.5\n", last_order);
    return 1;
  }
  std::printf("OK test_weno_convergence (ordre %.2f, constantes preservees)\n", last_order);
  return 0;
}
