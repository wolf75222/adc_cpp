// Identite STRUCTURELLE des backends elliptiques : GeometricMG (iteratif) et
// PoissonFFTSolver (spectral direct) inversent le MEME operateur discret. On le prouve non
// par "MG ~ FFT" (cf. test_fft_coupler) mais en appliquant a CHAQUE solution le MEME
// operateur canonique partage `poisson_residual` (le Laplacien 5 points de
// elliptic/poisson_operator.hpp, l'EllipticOperator de l'archi) : les deux residus tombent a
// l'arrondi. C'est l'OperatorSpec partage rendu verifiable, pas seulement documentaire.

#include <adc/numerics/elliptic/geometric_mg.hpp>
#include <adc/numerics/elliptic/poisson_fft_solver.hpp>
#include <adc/numerics/elliptic/poisson_operator.hpp>  // poisson_residual : l'operateur canonique
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>

#include "test_harness.hpp"  // adc::test::Checker + kPi partages

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;
using adc::test::kPi;

int main() {
  adc::test::Checker chk;  // style terse : n'imprime que les echecs (FAIL <libelle>)

  const int N = 64;  // puissance de 2 (FFT)
  Box2D dom = Box2D::from_extents(N, N);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, 1);
  BCRec bc;  // periodique : second membre a moyenne nulle ci-dessous

  auto fr = [&](int i, int j) {
    return std::sin(2 * kPi * geom.x_cell(i)) * std::sin(2 * kPi * geom.y_cell(j));
  };

  // residu du Laplacien canonique applique a la solution du solveur donne.
  auto residual_under_shared_op = [&](auto& solver) {
    Array4 f = solver.rhs().fab(0).array();
    const Box2D v = solver.rhs().box(0);
    for_each_cell(v, [=] ADC_HD(int i, int j) { f(i, j) = fr(i, j); });
    solver.phi().set_val(0.0);
    solver.solve();
    MultiFab res(ba, dm, 1, 0);
    poisson_residual(solver.phi(), solver.rhs(), geom, bc, res);  // operateur PARTAGE
    return static_cast<double>(norm_inf(res));
  };

  GeometricMG mg(geom, ba, bc);
  PoissonFFTSolver fft(geom, ba, bc);
  const double rmg = residual_under_shared_op(mg);
  const double rfft = residual_under_shared_op(fft);

  // les deux solutions coincident a la jauge pres (solution periodique definie a une cte).
  auto demean = [&](MultiFab& m) {
    const Real avg = sum(m) / static_cast<Real>(dom.num_cells());
    Array4 a = m.fab(0).array();
    for_each_cell(dom, [=] ADC_HD(int i, int j) { a(i, j) -= avg; });
  };
  demean(mg.phi());
  demean(fft.phi());
  double maxdiff = 0;
  const ConstArray4 pm = mg.phi().fab(0).const_array(), pf = fft.phi().fab(0).const_array();
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i)
      maxdiff = std::fmax(maxdiff, std::fabs(pm(i, j) - pf(i, j)));

  std::printf("operateur partage : residu(MG)=%.3e residu(FFT)=%.3e | maxdiff(phi)=%.3e\n",
              rmg, rfft, maxdiff);

  chk(rmg < 1e-9, "MG_inverse_l_operateur_canonique");
  chk(rfft < 1e-9, "FFT_inverse_l_operateur_canonique");
  chk(maxdiff < 1e-9, "MG_et_FFT_meme_solution");

  if (chk.fails() == 0) std::printf("OK test_elliptic_operator\n");
  return chk.failed();
}
