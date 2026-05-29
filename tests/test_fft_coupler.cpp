// Valide le 2e backend du concept EllipticSolver : PoissonFFTSolver (direct,
// spectral) comme backend du Coupler, a la place de GeometricMG (iteratif).
//   1. PoissonFFTSolver seul : resout le Laplacien discret 5 points a l'arrondi.
//   2. Coupler<EulerPoisson, PoissonFFTSolver> vs Coupler<EulerPoisson> (MG) :
//      memes resultats (meme operateur discret), donc le backend est interchangeable.
//
// N puissance de 2 (FFT radix-2). Mono-rang.

#include <adc/coupling/coupler.hpp>
#include <adc/elliptic/poisson_fft_solver.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/euler_poisson.hpp>
#include <adc/operator/reconstruction.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  const int N = 64;  // puissance de 2
  const double L = 1.0;
  Box2D dom = Box2D::from_extents(N, N);
  Geometry geom{dom, 0.0, L, 0.0, L};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, 1);

  // --- 1. PoissonFFTSolver seul : residu a l'arrondi ---
  {
    BCRec bc;
    PoissonFFTSolver fft(geom, ba, bc);
    Array4 f = fft.rhs().fab(0).array();
    const Box2D v = fft.rhs().box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        f(i, j) = std::sin(2 * kPi * geom.x_cell(i)) *
                  std::sin(2 * kPi * geom.y_cell(j));  // moyenne nulle
    fft.solve();
    const double r = fft.residual();
    std::printf("PoissonFFTSolver : residu discret = %.3e\n", r);
    chk(r < 1e-10, "fft_resout_le_laplacien_discret");
  }

  // --- 2. Coupler MG vs Coupler FFT : memes resultats ---
  {
    const double g = 5.0 / 3, fpg = 20, rho0 = 1, p0 = 1, eps = 1e-3;
    const double k = 2 * kPi / L, cs2 = g * p0 / rho0;
    EulerPoisson model;
    model.hydro.gamma = g;
    model.four_pi_G = fpg;
    model.rho0 = rho0;
    BCRec bcU, bcPhi;

    auto initU = [&](MultiFab& U) {
      Array4 u = U.fab(0).array();
      const Box2D v = U.box(0);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
          const double x = geom.x_cell(i);
          const double dr = eps * rho0 * std::cos(k * x), r = rho0 + dr,
                       p = p0 + cs2 * dr;
          u(i, j, 0) = r;
          u(i, j, 1) = 0;
          u(i, j, 2) = 0;
          u(i, j, 3) = p / (g - 1);
        }
    };
    const double dt = 0.4 * (L / N) / (std::sqrt(cs2) + 0.1);
    const int K = 5;

    Coupler<EulerPoisson> cmg(model, geom, ba, bcU, bcPhi);             // MG
    Coupler<EulerPoisson, PoissonFFTSolver> cff(model, geom, ba, bcU, bcPhi);  // FFT
    MultiFab Umg(ba, dm, 4, 2), Uff(ba, dm, 4, 2);
    initU(Umg);
    initU(Uff);
    for (int s = 0; s < K; ++s) cmg.advance<Minmod>(Umg, dt);
    for (int s = 0; s < K; ++s) cff.advance<Minmod>(Uff, dt);

    double maxdiff = 0;
    const ConstArray4 a = Umg.fab(0).const_array();
    const ConstArray4 b = Uff.fab(0).const_array();
    for (int j = 0; j < N; ++j)
      for (int i = 0; i < N; ++i)
        for (int c = 0; c < 4; ++c)
          maxdiff = std::fmax(maxdiff, std::fabs(a(i, j, c) - b(i, j, c)));
    std::printf("Coupler MG vs FFT : maxdiff(U) apres %d pas = %.3e\n", K, maxdiff);
    chk(maxdiff < 1e-5, "mg_et_fft_donnent_le_meme_resultat");
  }

  if (fails == 0) std::printf("OK test_fft_coupler\n");
  return fails == 0 ? 0 : 1;
}
