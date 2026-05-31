#pragma once

#include <memory>
#include <vector>

// Facade COMPILEE du solveur diocotron sur AMR multi-patch. Meme esprit que
// DiocotronSolver (PIMPL, API stable sans template, bindable pybind11), mais la
// hierarchie est raffinee : un niveau grossier + des patchs fins reconstruits a la
// volee par regrid Berger-Rigoutsos autour de la bande de charge. Le pas couple passe
// par AmrCouplerMP -> advance_amr (sous-cyclage + reflux conservatif). Le backend
// (serie/OpenMP/Kokkos) est celui de la cible adc, comme pour les autres facades.
//
// Cas couvert : bande de charge periodique (cat's eye / Kelvin-Helmholtz). C'est le
// cas ou l'AMR montre son interet : le bord cisaille concentre le raffinement.

namespace adc {

struct DiocotronAmrConfig {
  int n = 128;            // cellules du niveau GROSSIER (domaine n x n)
  double L = 1.0;         // taille physique du domaine carre
  double B0 = 1.0;        // champ magnetique hors-plan
  double alpha = 1.0;     // constante de couplage Poisson
  double band_amp = 1.0;  // amplitude A de la bande de charge
  double band_width = 0.05;  // demi-largeur w (gaussienne)
  int    band_mode = 2;      // mode seme en x
  double band_disp = 0.02;   // deplacement eta de l'axe de la bande
  double refine_frac = 0.15; // seuil de raffinement : tag si n_e > n_i0 + refine_frac
  int regrid_every = 20;     // re-raffine tous les N pas (0 = jamais apres l'init)
};

class DiocotronAmrSolver {
 public:
  explicit DiocotronAmrSolver(const DiocotronAmrConfig& cfg);
  ~DiocotronAmrSolver();
  DiocotronAmrSolver(DiocotronAmrSolver&&) noexcept;
  DiocotronAmrSolver& operator=(DiocotronAmrSolver&&) noexcept;

  void step(double dt);            // un pas AMR couple (regrid periodique inclus)
  void step_cfl(double cfl);       // pas stable : dt = cfl*dx_grossier/v_derive
  double max_drift_speed() const;
  double dx() const;               // pas d'espace grossier L/n
  double mass() const;             // masse sur le grossier (conservee a l'arrondi)
  double time() const;
  int nx() const;                  // n (niveau grossier)
  int n_patches() const;           // nombre de patchs fins courants (apres regrid)
  std::vector<double> density() const;  // n_e du niveau GROSSIER (row-major, n x n)

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace adc
