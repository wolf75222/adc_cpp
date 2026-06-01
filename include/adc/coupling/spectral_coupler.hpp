#pragma once

#include <adc/elliptic/poisson_fft_solver.hpp>  // DistributedFFTSolver (enveloppe PoissonFFT)
#include <adc/integrator/amr_reflux.hpp>  // advance_fab_1c, xface_box, yface_box
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/for_each.hpp>  // for_each_cell_reduce_sum, ADC_HD, device_fence
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
class SpectralCoupler {
 public:
  SpectralCoupler(const Model& model, int Nx, int Ny, double Lx, double Ly)
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
        fft_(Geometry{Box2D::from_extents(Nx, Ny), 0.0, Lx, 0.0, Ly}) {
    std::vector<Box2D> slabs;
    for (int r = 0; r < np_; ++r)
      slabs.push_back(Box2D{{0, r * nyl_}, {Nx - 1, (r + 1) * nyl_ - 1}});
    ba_ = BoxArray(std::move(slabs));
    dm_ = DistributionMapping(np_, np_);  // box r -> rang r
    U_ = MultiFab(ba_, dm_, 1, 1);
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
    // rho dans le rhs du solveur FFT distribue (meme decoupage en bandes que U_, donc
    // bit-identique a l'ancien chemin inline qui appelait PoissonFFT directement).
    const ConstArray4 u = U_.fab(0).const_array();
    Array4 r = fft_.rhs().fab(0).array();
    // f = model.elliptic_rhs(U) : on appelle le PhysicalModel (comme le Coupler normal)
    // au lieu de coder la formule diocotron alpha*(u - n_i0) en dur dans le coeur.
    // Pour le diocotron c'est exactement la meme expression -> bit-identique.
    for (int j = y0_; j < y0_ + nyl_; ++j)
      for (int i = 0; i < Nx_; ++i)
        r(i, j) = model_.elliptic_rhs(typename Model::State{u(i, j)});
    fft_.solve();
    fill_boundary(fft_.phi(), dom_, Periodicity{true, true});
    const ConstArray4 pc = fft_.phi().fab(0).const_array();
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

  // vitesse d'onde max (all-reduce), pour la CFL. GENERALISE (TODO 4.3) : via
  // model.max_wave_speed au lieu du diocotron /B0 code en dur -> le coupleur spectral
  // n'est plus lie au diocotron. NB : pour le diocotron c'est max(|gx|,|gy|)/B0 (par
  // direction) au lieu de hypot(gx,gy)/B0 ; le dt CFL change donc legerement (a re-valider
  // cote physique, ce n'est PAS bit-identique a l'ancien diagnostic).
  double max_drift_speed() const {
    device_fence();  // GPU : barriere avant lecture hote apres advance_fab_1c (device)
    const ConstArray4 u = U_.fab(0).const_array();
    const ConstArray4 a = Uaux_.fab(0).const_array();
    const Model m = model_;
    double v = 0;
    for (int j = y0_; j < y0_ + nyl_; ++j)
      for (int i = 0; i < Nx_; ++i) {
        typename Model::State s{};
        s[0] = u(i, j, 0);
        const typename Model::Aux ax{a(i, j, 0), a(i, j, 1), a(i, j, 2)};
        const Real wx = m.max_wave_speed(s, ax, 0);
        const Real wy = m.max_wave_speed(s, ax, 1);
        v = std::max(v, static_cast<double>(wx > wy ? wx : wy));
      }
    return std::max(all_reduce_max(v), 1e-12);
  }

  // masse totale (all-reduce).
  double mass() const {
    // seam reducteur (bande locale = 1 fab -> bit-identique a l'ancienne somme hote en
    // serie/OpenMP) ; parallel_reduce absorbe la barriere, plus de device_fence en tete.
    const ConstArray4 u = U_.fab(0).const_array();
    const Real s = for_each_cell_reduce_sum(U_.box(0), [u] ADC_HD(int i, int j) { return u(i, j); });
    return all_reduce_sum(static_cast<double>(s)) * dx_ * dy_;
  }

 private:
  Model model_;
  int Nx_, Ny_;
  double dx_, dy_;
  Box2D dom_;
  int np_, me_, nyl_, y0_;
  BoxArray ba_;
  DistributionMapping dm_;
  MultiFab U_, Uaux_;
  DistributedFFTSolver fft_;
};

}  // namespace adc
