#pragma once

#include <memory>
#include <vector>

// Facade compilee du solveur Euler-Poisson auto-gravitant (cf. diocotron_solver.hpp
// pour le rationale du split include/ header-only + src/ compile). Instancie
// Coupler<EulerPoisson> dans src/euler_poisson_solver.cpp.

namespace adc {

struct EulerPoissonConfig {
  int n = 128;
  double L = 1.0;
  double gamma = 5.0 / 3.0;
  double four_pi_G = 20.0;  // lap phi = four_pi_G (rho - rho0)
  double rho0 = 1.0;        // fond neutralisant
  double p0 = 1.0;
  double eps = 1e-3;        // amplitude de la perturbation de Jeans
};

class EulerPoissonSolver {
 public:
  explicit EulerPoissonSolver(const EulerPoissonConfig& cfg);
  ~EulerPoissonSolver();
  EulerPoissonSolver(EulerPoissonSolver&&) noexcept;
  EulerPoissonSolver& operator=(EulerPoissonSolver&&) noexcept;

  void step(double dt);
  double mass() const;
  double energy() const;
  double total_momentum(int dir) const;  // dir = 0 (x) ou 1 (y) ; conserve ~ 0
  double time() const;
  int nx() const;
  std::vector<double> density() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace adc
