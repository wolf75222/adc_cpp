#pragma once

#include <adc/elliptic/poisson_fft.hpp>
#include <adc/integrator/amr_reflux.hpp>  // advance_fab_1c, xface_box, yface_box
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/parallel/comm.hpp>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

// Stepper couple periodique E x B, DISTRIBUE et PRET A L'EMPLOI. Encapsule la
// boucle qui etait reecrite a la main dans chaque exemple/test :
//
//   rho = alpha (n_e - n_i0)  ->  Poisson spectral distribue (FFT, bandes)
//   ->  aux = (phi, grad phi) avec halos  ->  advance (Rusanov, derive E x B).
//
// Decomposition en bandes (1 box par rang), layout du solveur FFT ; Nx, Ny
// puissances de 2 divisibles par n_ranks(). Generique sur le modele (tout modele
// portant alpha, n_i0, B0 et un flux de derive E x B, p.ex. Diocotron).
//
// L'exemple/test ne reimplemente plus rien : il construit le stepper, remplit
// state() (densite initiale), puis boucle step(dt). C'est l'esprit "composant"
// (cf. Simulation de MUFFIN), mais en compile-time/template (zero virtuel,
// GPU-ready).

namespace adc {

template <class Model>
class SpectralExBStepper {
 public:
  SpectralExBStepper(const Model& model, int Nx, int Ny, double Lx, double Ly)
      : model_(model),
        Nx_(Nx),
        Ny_(Ny),
        dx_(Lx / Nx),
        dy_(Ly / Ny),
        dom_(Box2D::from_extents(Nx, Ny)),
        np_(n_ranks()),
        me_(my_rank()),
        nyl_(Ny / np_),
        y0_(me_ * nyl_),
        solver_(Nx, Ny, Lx, Ly),
        rho_(static_cast<std::size_t>(nyl_) * Nx) {
    std::vector<Box2D> slabs;
    for (int r = 0; r < np_; ++r)
      slabs.push_back(Box2D{{0, r * nyl_}, {Nx - 1, (r + 1) * nyl_ - 1}});
    ba_ = BoxArray(std::move(slabs));
    dm_ = DistributionMapping(np_, np_);  // box r -> rang r
    U_ = MultiFab(ba_, dm_, 1, 1);
    Uphi_ = MultiFab(ba_, dm_, 1, 1);
    Uaux_ = MultiFab(ba_, dm_, 3, 1);
  }

  // --- acces a l'etat (la bande locale de ce rang) ---
  MultiFab& state() { return U_; }
  const MultiFab& state() const { return U_; }
  Fab2D& local() { return U_.fab(0); }  // le fab de la bande locale
  const Box2D& domain() const { return dom_; }
  int y_begin() const { return y0_; }
  int ny_local() const { return nyl_; }
  int nx() const { return Nx_; }
  double dx() const { return dx_; }
  double dy() const { return dy_; }

  // Poisson + aux + halos sans avancer (pour fixer dt / diagnostiquer).
  void solve_aux() {
    const ConstArray4 u = U_.fab(0).const_array();
    for (int jl = 0; jl < nyl_; ++jl)
      for (int i = 0; i < Nx_; ++i)
        rho_[jl * Nx_ + i] = model_.alpha * (u(i, y0_ + jl) - model_.n_i0);
    solver_.solve(rho_, phi_);
    Array4 p = Uphi_.fab(0).array();
    for (int jl = 0; jl < nyl_; ++jl)
      for (int i = 0; i < Nx_; ++i) p(i, y0_ + jl) = phi_[jl * Nx_ + i];
    fill_boundary(Uphi_, dom_, Periodicity{true, true});
    const ConstArray4 pc = Uphi_.fab(0).const_array();
    Array4 a = Uaux_.fab(0).array();
    for (int j = y0_; j < y0_ + nyl_; ++j)
      for (int i = 0; i < Nx_; ++i) {
        a(i, j, 0) = pc(i, j);
        a(i, j, 1) = (pc(i + 1, j) - pc(i - 1, j)) / (2 * dx_);
        a(i, j, 2) = (pc(i, j + 1) - pc(i, j - 1)) / (2 * dy_);
      }
    fill_boundary(Uaux_, dom_, Periodicity{true, true});
  }

  // un pas couple complet.
  void step(double dt) {
    solve_aux();
    fill_boundary(U_, dom_, Periodicity{true, true});
    Fab2D fx(xface_box(U_.box(0)), 1, 0), fy(yface_box(U_.box(0)), 1, 0);
    advance_fab_1c(model_, U_.fab(0), Uaux_.fab(0), dx_, dy_, dt, fx, fy);
  }

  // vitesse de derive max (all-reduce), pour la CFL.
  double max_drift_speed() const {
    device_fence();  // GPU : barriere avant lecture hote apres advance_fab_1c (device)
    const ConstArray4 a = Uaux_.fab(0).const_array();
    double v = 0;
    for (int j = y0_; j < y0_ + nyl_; ++j)
      for (int i = 0; i < Nx_; ++i)
        v = std::max(v, std::hypot(a(i, j, 1), a(i, j, 2)) / model_.B0);
    return std::max(all_reduce_max(v), 1e-12);
  }

  // masse totale (all-reduce).
  double mass() const {
    device_fence();  // GPU : barriere avant lecture hote apres advance_fab_1c (device)
    const ConstArray4 u = U_.fab(0).const_array();
    double s = 0;
    for (int j = y0_; j < y0_ + nyl_; ++j)
      for (int i = 0; i < Nx_; ++i) s += u(i, j);
    return all_reduce_sum(s) * dx_ * dy_;
  }

 private:
  Model model_;
  int Nx_, Ny_;
  double dx_, dy_;
  Box2D dom_;
  int np_, me_, nyl_, y0_;
  BoxArray ba_;
  DistributionMapping dm_;
  MultiFab U_, Uphi_, Uaux_;
  PoissonFFT solver_;
  std::vector<double> rho_, phi_;
};

}  // namespace adc
