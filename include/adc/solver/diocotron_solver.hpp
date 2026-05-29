#pragma once

#include <memory>
#include <vector>

// Facade COMPILEE du solveur diocotron : un type concret, non templatise, dont les
// methodes sont DEFINIES dans src/diocotron_solver.cpp. Le PIMPL (Impl) instancie
// toute la pile template (Coupler<Diocotron> + multigrille + MUSCL/SSPRK2) UNE
// SEULE FOIS dans l'unite de compilation de la lib. Avantages :
//   - une vraie librairie compilee (libadc) au lieu de tout en-tete,
//   - API stable et sans template, ideale pour pybind11 et pour les applications,
//   - le backend (serie/OpenMP/Kokkos) est herite de la cible `adc` : compiler le
//     projet -DADC_USE_KOKKOS=ON suffit a faire tourner cette facade sur GPU.
// Le coeur generique (concepts, for_each_cell GPU) reste header-only dans include/.
//
// La facade couvre les TROIS mises en place canoniques du diocotron, choisies par
// `DiocotronConfig::ic`. Les exemples examples/diocotron*.cpp ne sont plus que des
// pilotes minces (CI + diagnostics + I/O) au-dessus de cette facade.

namespace adc {

// Condition initiale (et regime de bord associe).
enum class DiocotronIC {
  Smooth,  // sin*sin lisse, BC periodique (defaut, retro-compatible)
  Band,    // bande de charge periodique (enroulement "cat's eye" / KH)
  Ring,    // anneau/colonne, BC Dirichlet phi + Foextrap U (+ paroi optionnelle)
};

struct DiocotronConfig {
  int n = 128;          // cellules par direction (domaine n x n)
  double L = 1.0;       // taille physique du domaine carre
  double B0 = 1.0;      // champ magnetique hors-plan
  double n_i0 = 1.0;    // fond ionique (Smooth) ; DERIVE pour Band, = 0 pour Ring
  double alpha = 1.0;   // constante de couplage Poisson
  double eps = 0.2;     // amplitude de la perturbation lisse (Smooth)
  // true : Poisson resolu a chaque etage RK (precis). false : une fois par pas
  // (OncePerStep) -> ~2.6x plus rapide. Voir docs/PERFORMANCE.md.
  bool poisson_per_stage = true;

  DiocotronIC ic = DiocotronIC::Smooth;
  // --- bande de charge periodique (ic = Band) ---
  double band_amp = 1.0;     // amplitude A de la bande
  double band_width = 0.05;  // demi-largeur w (gaussienne)
  int    band_mode = 2;      // mode seme en x
  double band_disp = 0.02;   // deplacement eta de l'axe de la bande
  // --- anneau / colonne (ic = Ring) ---
  double ring_r0 = 0.15, ring_r1 = 0.20;  // rayons interne / externe
  double ring_delta = 0.1;   // amplitude de la perturbation azimutale
  int    ring_mode = 3;      // mode azimutal l
  double ring_floor = 1e-3;  // densite plancher hors anneau
  double wall_radius = 0.0;  // > 0 : paroi conductrice circulaire (embedded boundary)
};

class DiocotronSolver {
 public:
  explicit DiocotronSolver(const DiocotronConfig& cfg);
  ~DiocotronSolver();
  DiocotronSolver(DiocotronSolver&&) noexcept;
  DiocotronSolver& operator=(DiocotronSolver&&) noexcept;

  void step(double dt);            // un pas SSPRK2 couple (Poisson par etage)
  void step_cfl(double cfl);       // pas stable automatique : dt = cfl*dx/v_derive
  double max_drift_speed() const;  // max |grad phi| / B0 (vitesse de derive E x B)
  double dx() const;               // pas d'espace L/n
  double mass() const;             // somme de la densite (invariant conserve)
  double time() const;
  int nx() const;
  std::vector<double> density() const;    // n_e aux cellules valides (row-major)
  std::vector<double> potential() const;  // phi aux cellules valides (row-major)

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace adc
