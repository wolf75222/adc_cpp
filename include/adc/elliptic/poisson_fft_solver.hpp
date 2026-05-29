#pragma once

#include <adc/elliptic/elliptic_solver.hpp>
#include <adc/elliptic/poisson_fft.hpp>
#include <adc/elliptic/poisson_operator.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/parallel/comm.hpp>

#include <cassert>
#include <functional>
#include <vector>

// Backend EllipticSolver DIRECT par FFT spectrale (CL periodiques). Resout le MEME
// Laplacien discret 5 points que GeometricMG (memes valeurs propres
// lambda = (2cos(2 pi kx/Nx)-2)/dx^2 + idem y), mais en UNE transformee au lieu
// d'iterer -> bien moins cher quand l'elliptique domine le run (cf. PERFORMANCE.md).
//
// Memes signature de constructeur et interface que GeometricMG, donc interchangeable
// comme parametre Elliptic du Coupler : Coupler<Model, PoissonFFTSolver>.
//
// Portee : mono-rang, boite unique couvrant le domaine (le cas par defaut du Coupler
// et des facades). Le periodique distribue tuiles <-> bandes FFT est porte par
// SpectralExBStepper (qui gere la redistribution).

namespace adc {

class PoissonFFTSolver {
 public:
  PoissonFFTSolver(const Geometry& geom, const BoxArray& ba,
                   const BCRec& = BCRec{}, std::function<bool(Real, Real)> = {})
      : geom_(geom),
        dm_(ba.size(), n_ranks()),
        phi_(ba, dm_, 1, 1),
        rhs_(ba, dm_, 1, 0),
        res_(ba, dm_, 1, 0),
        fft_(geom.domain.nx(), geom.domain.ny(), geom.xhi - geom.xlo,
             geom.yhi - geom.ylo) {
    assert(n_ranks() == 1 && ba.size() == 1 &&
           "PoissonFFTSolver : mono-rang / boite unique (sinon SpectralExBStepper)");
  }

  MultiFab& rhs() { return rhs_; }
  MultiFab& phi() { return phi_; }
  const Geometry& geom() const { return geom_; }

  // Resout lap(phi) = rhs en place (direct, une FFT directe + inverse).
  void solve() {
    const int Nx = geom_.domain.nx(), Ny = geom_.domain.ny();
    const ConstArray4 f = rhs_.fab(0).const_array();
    const Box2D v = rhs_.box(0);
    std::vector<double> rho(static_cast<std::size_t>(Nx) * Ny), phil;
    for (int j = 0; j < Ny; ++j)
      for (int i = 0; i < Nx; ++i)
        rho[static_cast<std::size_t>(j) * Nx + i] = f(v.lo[0] + i, v.lo[1] + j);
    fft_.solve(rho, phil);  // mode k=0 (constante) mis a zero -> phi de moyenne nulle
    Array4 p = phi_.fab(0).array();
    for (int j = 0; j < Ny; ++j)
      for (int i = 0; i < Nx; ++i)
        p(v.lo[0] + i, v.lo[1] + j) = phil[static_cast<std::size_t>(j) * Nx + i];
  }

  // Residu discret ||lap(phi) - rhs|| (~ arrondi pour ce solveur direct).
  Real residual() {
    BCRec bc;  // periodique
    poisson_residual(phi_, rhs_, geom_, bc, res_);
    return norm_inf(res_);
  }

 private:
  Geometry geom_;
  DistributionMapping dm_;
  MultiFab phi_, rhs_, res_;
  PoissonFFT fft_;
};

static_assert(EllipticSolver<PoissonFFTSolver>,
              "PoissonFFTSolver doit modeler EllipticSolver");

}  // namespace adc
