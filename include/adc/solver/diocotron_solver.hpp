#pragma once

#include <memory>
#include <vector>

// Facade COMPILEE du solveur diocotron : un type concret, non templatise, dont les
// methodes sont DEFINIES dans src/diocotron_solver.cpp. Le PIMPL (Impl) instancie
// toute la pile template (Coupler<Diocotron> + multigrille + MUSCL/SSPRK2) UNE
// SEULE FOIS dans l'unite de compilation de la lib. Avantages :
//   - une vraie librairie compilee (libadc) au lieu de tout en-tete,
//   - API stable et sans template, ideale pour pybind11 et pour les applications,
//   - temps de compilation des exemples/tests reduits (la pile est deja compilee).
// Le coeur generique (concepts PhysicalModel/NumericalFlux/..., for_each_cell GPU)
// reste header-only dans include/ : ces templates DOIVENT etre visibles a
// l'instanciation. src/ contient les instanciations concretes "pretes a l'emploi".

namespace adc {

struct DiocotronConfig {
  int n = 128;          // cellules par direction (domaine n x n)
  double L = 1.0;       // taille physique du domaine carre
  double B0 = 1.0;      // champ magnetique hors-plan
  double n_i0 = 1.0;    // densite ionique de fond (neutralisante)
  double alpha = 1.0;   // constante de couplage Poisson
  double eps = 0.2;     // amplitude de la perturbation initiale lisse
};

class DiocotronSolver {
 public:
  explicit DiocotronSolver(const DiocotronConfig& cfg);
  ~DiocotronSolver();
  DiocotronSolver(DiocotronSolver&&) noexcept;
  DiocotronSolver& operator=(DiocotronSolver&&) noexcept;

  void step(double dt);                 // un pas SSPRK2 couple (Poisson par etage)
  double mass() const;                  // somme de la densite (invariant conserve)
  double time() const;
  int nx() const;
  std::vector<double> density() const;  // densite aux cellules valides (row-major)

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace adc
