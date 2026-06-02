#pragma once

#include <memory>
#include <vector>

// Facade compilee du solveur deux-fluides isotherme 2D asymptotic-preserving
// (cf. diocotron_solver.hpp pour le rationale include/ header-only + src/ compile).
// Instancie TwoFluidAP2D<GeometricMG> dans src/two_fluid_ap_solver.cpp : transport
// Rusanov des deux especes + Lorentz implicite + Poisson reformule par multigrille.
// L'elliptique multigrille est entierement on-device, donc cette facade se compile
// telle quelle pour le GPU des que le projet est configure -DADC_USE_KOKKOS=ON (le
// backend est herite de la cible `adc`, pas rajoute ici).

namespace adc {

struct TwoFluidAPConfig {
  int n = 64;
  double L = 6.283185307179586;  // 2*pi
  double cse2 = 1.0;             // vitesse du son electronique au carre
  double csi2 = 0.04;            // ion
  double omega_pe = 5.0;         // frequence plasma electronique
  double omega_pi = 1.0;         // ion
  bool stabilize = true;         // schema AP (Poisson reformule) ; false = non stabilise
  double eps = 1e-3;             // amplitude de la perturbation cosinus initiale
  bool upwind_continuity = false;  // flux de masse Rusanov (anti-Gibbs sur fronts raides)
                                   // au lieu de la continuite centree par defaut
  double omega_ce = 0.0;           // frequence cyclotron electronique (B hors-plan ; 0 = pas de B)
  double omega_ci = 0.0;           // frequence cyclotron ionique
};

class TwoFluidAPSolver {
 public:
  explicit TwoFluidAPSolver(const TwoFluidAPConfig& cfg = {});
  ~TwoFluidAPSolver();
  TwoFluidAPSolver(TwoFluidAPSolver&&) noexcept;
  TwoFluidAPSolver& operator=(TwoFluidAPSolver&&) noexcept;

  void step(double dt);
  void advance(double dt, int nsteps);

  int nx() const;
  double mass_e() const;            // masse electronique totale (conservee)
  double mass_i() const;            // masse ionique totale
  double max_charge() const;        // max|n_i - n_e| (quasi-neutralite)
  double max_dev() const;           // max|n_e - n0| (borne en regime AP raide)
  std::vector<double> density_e() const;  // n_e (n x n), row-major, copiee
  std::vector<double> density_i() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace adc
