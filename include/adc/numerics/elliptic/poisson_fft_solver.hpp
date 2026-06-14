#pragma once

#include <adc/numerics/elliptic/elliptic_solver.hpp>
#include <adc/numerics/elliptic/poisson_fft.hpp>
#include <adc/numerics/elliptic/poisson_operator.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/parallel/comm.hpp>

#include <cassert>
#include <functional>
#include <stdexcept>
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
// et des facades). Pour le periodique DISTRIBUE, voir DistributedFFTSolver ci-dessous (bandes,
// EllipticSolver autonome) ; SpectralCoupler l'enveloppe avec le transport E x B.

namespace adc {

class PoissonFFTSolver {
 public:
  /// @p spectral : symbole du Laplacien (false = stencil 5-points discret, defaut bit-identique ;
  /// true = symbole continu -(kx^2+ky^2), fidelite aux references spectrales -- cf. PoissonFFT).
  PoissonFFTSolver(const Geometry& geom, const BoxArray& ba,
                   const BCRec& = BCRec{}, std::function<bool(Real, Real)> = {},
                   bool spectral = false)
      : geom_(geom),
        dm_(ba.size(), n_ranks()),
        phi_(ba, dm_, 1, 1),
        rhs_(ba, dm_, 1, 0),
        res_(ba, dm_, 1, 0),
        fft_(geom.domain.nx(), geom.domain.ny(), geom.xhi - geom.xlo,
             geom.yhi - geom.ylo, spectral) {
    // Garde-fou DUR (actif en Release, NDEBUG ne le retire PAS) : ce solveur direct est mono-rang /
    // boite unique. Sous DistributionMapping de systeme a n_ranks()>1, certains rangs n'ont aucune
    // box locale (local_size()==0) et solve() dereferencerait fab(0) inexistant -> SIGSEGV. L'ancien
    // assert disparaissait en Release et la protection s'evanouissait en silence. On leve sur TOUS les
    // rangs (chacun construit l'objet), donc pas d'interblocage. Pour le periodique distribue :
    // DistributedFFTSolver (bandes, MPI_Alltoall).
    if (n_ranks() != 1)
      throw std::runtime_error(
          "solveur fft non supporte en MPI (n_ranks>1) : utiliser geometric_mg ou le solveur fft "
          "distribue (DistributedFFTSolver)");
    if (ba.size() != 1)
      throw std::runtime_error(
          "PoissonFFTSolver : boite unique requise (ba.size()==1) ; pour un domaine multi-box "
          "distribue, utiliser DistributedFFTSolver ou geometric_mg");
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
    // GHOSTS de phi : la derivation de l'aux (grad phi centre, solve_fields) lit les voisins
    // i±1/j±1, donc les ghosts du bord de domaine. GeometricMG les remplit en lissant ; ce
    // solveur direct ecrit SEULEMENT les cellules valides -> sans cet echange, le gradient du
    // bord lirait des ghosts perimes (bug latent revele par une source electrique branchee sur
    // 'fft' : Ex faux sur l'anneau de bord). Periodique pur par construction (gardes du ctor).
    fill_boundary(phi_, geom_.domain, Periodicity{true, true});
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

// Variante DISTRIBUEE du meme backend FFT. Le domaine periodique est decoupe en BANDES (slabs,
// 1 box par rang, le layout natif du solveur FFT) ; PoissonFFT fait la transposee parallele
// (MPI_Alltoall) en interne. C'est un EllipticSolver AUTONOME : utilisable comme
// Coupler<Model, DistributedFFTSolver>, au lieu d'enfermer la FFT distribuee dans SpectralCoupler.
// Contrainte : Ny divisible par n_ranks(), Nx/Ny puissances de 2. En serie (n_ranks()==1) une
// seule bande couvre le domaine, identique a PoissonFFTSolver.
class DistributedFFTSolver {
 public:
  DistributedFFTSolver(const Geometry& geom, const BCRec& = BCRec{},
                       std::function<bool(Real, Real)> = {})
      : geom_(geom),
        Nx_(geom.domain.nx()),
        nyl_(geom.domain.ny() / n_ranks()),
        fft_(geom.domain.nx(), geom.domain.ny(), geom.xhi - geom.xlo,
             geom.yhi - geom.ylo) {
    const int np = n_ranks();
    // Garde-fou DUR (actif en Release) : nyl_ = Ny / np est deja calcule dans la liste d'init.
    // Si Ny n'est pas divisible par np, les bandes seraient mal dimensionnees et solve() lirait
    // hors bornes. L'ancien assert disparaissait sous NDEBUG. Aligne sur PoissonFFTSolver (throw).
    if (geom.domain.ny() % np != 0)
      throw std::runtime_error(
          "DistributedFFTSolver : Ny doit etre divisible par n_ranks()");
    std::vector<Box2D> slabs;
    for (int r = 0; r < np; ++r)
      slabs.push_back(Box2D{{0, r * nyl_}, {Nx_ - 1, (r + 1) * nyl_ - 1}});
    BoxArray ba(std::move(slabs));
    dm_ = DistributionMapping(np, np);  // bande r -> rang r
    phi_ = MultiFab(ba, dm_, 1, 1);
    rhs_ = MultiFab(ba, dm_, 1, 0);
    res_ = MultiFab(ba, dm_, 1, 0);
  }

  MultiFab& rhs() { return rhs_; }
  MultiFab& phi() { return phi_; }
  const Geometry& geom() const { return geom_; }

  // Resout lap(phi) = rhs en place : bande locale -> tableau plat -> PoissonFFT (transposee MPI
  // interne) -> reecrit la bande locale. Mode k=0 mis a zero (phi de moyenne nulle).
  void solve() {
    const ConstArray4 f = rhs_.fab(0).const_array();
    const Box2D v = rhs_.box(0);  // bande locale [0..Nx-1] x [y0..y0+nyl-1]
    std::vector<double> rho(static_cast<std::size_t>(nyl_) * Nx_), phil;
    for (int jl = 0; jl < nyl_; ++jl)
      for (int i = 0; i < Nx_; ++i)
        rho[static_cast<std::size_t>(jl) * Nx_ + i] = f(v.lo[0] + i, v.lo[1] + jl);
    fft_.solve(rho, phil);
    Array4 p = phi_.fab(0).array();
    for (int jl = 0; jl < nyl_; ++jl)
      for (int i = 0; i < Nx_; ++i)
        p(v.lo[0] + i, v.lo[1] + jl) = phil[static_cast<std::size_t>(jl) * Nx_ + i];
  }

  // Residu discret ||lap(phi) - rhs|| reduit sur tous les rangs (~arrondi : solve direct exact).
  Real residual() {
    BCRec bc;  // periodique
    fill_boundary(phi_, geom_.domain, Periodicity{true, true});  // halos inter-bandes (MPI)
    poisson_residual(phi_, rhs_, geom_, bc, res_);
    return all_reduce_max(norm_inf(res_));
  }

 private:
  Geometry geom_;
  int Nx_, nyl_;
  DistributionMapping dm_;
  MultiFab phi_, rhs_, res_;
  PoissonFFT fft_;
};

static_assert(EllipticSolver<DistributedFFTSolver>,
              "DistributedFFTSolver doit modeler EllipticSolver");

}  // namespace adc
