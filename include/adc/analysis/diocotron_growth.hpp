#pragma once

// En-tete d'analyse host-only (Eigen). Garde interne symetrique a hdf5_writer.hpp :
// vide si Eigen n'est pas configure, donc safe a inclure n'importe ou.
#ifdef ADC_HAS_EIGEN

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <cmath>
#include <vector>

// Taux de croissance lineaire de l'instabilite diocotron pour une colonne creuse
// (anneau de charge r in [a, b]) dans une cavite conductrice de rayon Rw, par
// resolution du probleme aux valeurs propres radial de Petri (arXiv:astro-ph/
// 0611936, eq. 20-21), equivalent a Davidson-Felice :
//
//   omega L_m phi = m Omega L_m phi + q_m phi,   phi(0) = phi(Rw) = 0,
//   L_m = (1/r) d_r(r d_r) - m^2/r^2,   q_m = (m/r) d_r(rho),
//   Omega(r) = -(1/r^2) int_0^r rho r' dr'.       (unites eps0 = B = 1)
//
// Discretise en differences finies sur une grille radiale, le probleme devient
// generalise A phi = omega L phi ; on resout M = L^{-1} A (Eigen, host) et on
// prend gamma = max Im(omega). C'EST LE SEUL ENDROIT OU EIGEN EST UTILISE, cote
// host uniquement (hors hot path, conformement a la decision d'architecture).
//
// Profil lisse par des tanh de largeur w (w -> 0 = profil net). La valeur
// renvoyee est normalisee a la convention du papier (omega_D = rho_bar/(2 pi)),
// verifiee en reproduisant gamma_3 ~ 0.772, gamma_4 ~ 0.911, gamma_5 ~ 0.683
// pour (a, b, Rw) = (6, 8, 16).

namespace adc {

inline double diocotron_density(double r, double a, double b, double rhobar,
                                double w) {
  if (w <= 0) return (r > a && r < b) ? rhobar : 0.0;
  return 0.5 * rhobar * (std::tanh((r - a) / w) - std::tanh((r - b) / w));
}

inline double diocotron_growth_rate(int m, double a, double b, double Rw,
                                    double rhobar = 1.0, double w = 0.05,
                                    int N = 1200) {
  const double h = Rw / N;
  std::vector<double> r(N + 1), rho(N + 1), C(N + 1), Om(N + 1);
  for (int i = 0; i <= N; ++i) {
    r[i] = i * h;
    rho[i] = diocotron_density(r[i], a, b, rhobar, w);
  }
  C[0] = 0;
  for (int i = 1; i <= N; ++i)
    C[i] = C[i - 1] + 0.5 * (rho[i] * r[i] + rho[i - 1] * r[i - 1]) * h;
  for (int i = 0; i <= N; ++i) Om[i] = (i == 0) ? 0.0 : -C[i] / (r[i] * r[i]);

  const int n = N - 1;  // noeuds interieurs (phi=0 aux bords)
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(n, n);
  Eigen::VectorXd Q(n), Omv(n);
  for (int k = 0; k < n; ++k) {
    const int i = k + 1;
    const double ri = r[i];
    if (k > 0) L(k, k - 1) = 1.0 / (h * h) - 1.0 / (2 * h * ri);
    L(k, k) = -2.0 / (h * h) - double(m * m) / (ri * ri);
    if (k < n - 1) L(k, k + 1) = 1.0 / (h * h) + 1.0 / (2 * h * ri);
    Q(k) = (double(m) / ri) * ((rho[i + 1] - rho[i - 1]) / (2 * h));
    Omv(k) = Om[i];
  }

  Eigen::MatrixXd A = (double(m) * Omv).asDiagonal() * L;
  for (int k = 0; k < n; ++k) A(k, k) += Q(k);

  const Eigen::MatrixXd M = L.partialPivLu().solve(A);  // M = L^{-1} A
  Eigen::EigenSolver<Eigen::MatrixXd> es(M, /*computeEigenvectors=*/false);
  double gmax = 0;
  for (int k = 0; k < n; ++k)
    gmax = std::max(gmax, es.eigenvalues()(k).imag());

  // convention du papier : gamma / omega_D avec omega_D = rho_bar/(2 pi)
  return 2.0 * M_PI * gmax / rhobar;
}

}  // namespace adc

#endif  // ADC_HAS_EIGEN
