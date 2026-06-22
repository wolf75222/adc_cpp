// Non-regression du solveur de Poisson FFT (PoissonFFTSolver) sur une grille dont la
// taille n'est PAS une puissance de 2 (n=48). Le moteur radix-2 ne traite que n=2^k :
// sur n=48 sa butterfly debordait le buffer (heap overflow, confirme AddressSanitizer),
// rendant un potentiel CORROMPU, non deterministe et aberrant (~1e+277) la ou
// GeometricMG restait sain. Le repli DFT directe (poisson_fft.hpp) corrige la cause.
//
// On valide sur un domaine PERIODIQUE, second membre a MOYENNE NON NULLE (le scenario qui
// divergeait : le solveur doit retirer le mode k=0 pour la solvabilite) :
//   A. phi fini et borne (avant le fix : non fini / ~1e+277) ;
//   A. FFT inverse EXACTEMENT le Laplacien 5-points (residu ||lap(phi) - (f-<f>)|| ~ arrondi) ;
//   B. sur second membre a moyenne NULLE (bien pose), phi(FFT) == phi(GeometricMG) a une
//      constante additive pres (jauge) : les deux inversent le meme stencil discret.

#include <adc/numerics/elliptic/mg/geometric_mg.hpp>
#include <adc/numerics/elliptic/poisson/poisson_fft_solver.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/storage/fab2d.hpp>
#include <adc/mesh/execution/for_each.hpp>
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/mf_arith.hpp>
#include <adc/mesh/storage/multifab.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int n = 48;  // NON puissance de 2 (entre 32 et 64) : declenchait le heap overflow radix-2
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);
  BCRec bc;  // periodique par defaut

  auto gauss = [&](double x, double y) {
    return std::exp(-((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) / 0.02);
  };

  // --- A. moyenne NON nulle : phi fini/borne + residu exact du Laplacien 5-points ---
  {
    PoissonFFTSolver fft(geom, ba, bc);
    Array4 f = fft.rhs().fab(0).array();
    for_each_cell(dom, [f, geom, gauss](int i, int j) {
      f(i, j, 0) = 1.0 + 0.3 * gauss(geom.x_cell(i), geom.y_cell(j));
    });
    fft.solve();

    bool finite = true;
    double maxabs = 0;
    Fab2D& p = fft.phi().fab(0);
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
        const double v = p(i, j, 0);
        finite = finite && std::isfinite(v);
        maxabs = std::max(maxabs, std::fabs(v));
      }
    chk(finite, "fft_phi_fini");
    chk(maxabs < 1e3, "fft_phi_borne");  // valeur reelle ~3e-3 ; le bug donnait ~1e+277

    // Le solveur met le mode k=0 a zero, donc resout lap(phi) = f - <f>. On retire <f> du
    // rhs en place, residual() rend alors ||(f-<f>) - lap(phi)|| ~ arrondi (FFT = inverse exact).
    const double mean = sum(fft.rhs()) / static_cast<double>(dom.num_cells());
    Array4 fr = fft.rhs().fab(0).array();
    for_each_cell(dom, [fr, mean](int i, int j) { fr(i, j, 0) -= mean; });
    const double res = fft.residual();
    std::printf("FFT n=%d <f> non nul : max|phi|=%.3e  residu 5-pts=%.2e\n", n, maxabs, res);
    chk(res < 1e-9, "fft_residu_5pts_arrondi");
  }

  // --- B. moyenne NULLE : phi(FFT) == phi(GeometricMG) a une constante pres ---
  {
    PoissonFFTSolver fft(geom, ba, bc);
    GeometricMG mg(geom, ba, bc);
    Array4 ff = fft.rhs().fab(0).array();
    Array4 fm = mg.rhs().fab(0).array();
    for_each_cell(dom, [ff, fm, geom, gauss](int i, int j) {
      const double g = gauss(geom.x_cell(i), geom.y_cell(j));
      ff(i, j, 0) = g;
      fm(i, j, 0) = g;
    });
    const double gmean = sum(fft.rhs()) / static_cast<double>(dom.num_cells());
    for_each_cell(dom, [ff, fm, gmean](int i, int j) {
      ff(i, j, 0) -= gmean;
      fm(i, j, 0) -= gmean;
    });

    fft.solve();
    mg.phi().set_val(0.0);
    mg.solve(Real(1e-12), 200);

    Fab2D& pf = fft.phi().fab(0);
    Fab2D& pm = mg.phi().fab(0);
    const double mf = sum(fft.phi()) / static_cast<double>(dom.num_cells());  // jauge
    const double mm = sum(mg.phi()) / static_cast<double>(dom.num_cells());
    double dmax = 0, ref = 0;
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
        const double a = pf(i, j, 0) - mf, b = pm(i, j, 0) - mm;
        dmax = std::max(dmax, std::fabs(a - b));
        ref = std::max(ref, std::fabs(b));
      }
    std::printf("FFT vs MG (<f>=0) : max|dphi|=%.2e  relatif=%.2e\n", dmax, dmax / ref);
    chk(dmax / ref < 1e-6, "fft_egale_mg_au_gradient_pres");
  }

  if (fails == 0)
    std::printf("OK test_poisson_fft\n");
  return fails == 0 ? 0 : 1;
}
