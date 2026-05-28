// Multigrille geometrique : convergence rapide et quasi independante du maillage
// sur des solutions manufacturees (Dirichlet et periodique), precision O(dx^2).

#include <adc/elliptic/geometric_mg.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

static constexpr double kPi = 3.14159265358979323846;

// Resout lap(phi)=f pour phi_ex donne, renvoie (cycles, erreur_inf).
template <class PhiEx, class RhsF>
static void solve_case(int n, const BCRec& bc, bool periodic, PhiEx phi_ex,
                       RhsF rhs_f, int& cycles, double& err) {
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);

  GeometricMG mg(geom, ba, bc);
  Array4 af = mg.rhs().fab(0).array();
  for_each_cell(dom, [af, geom, rhs_f](int i, int j) {
    af(i, j, 0) = rhs_f(geom.x_cell(i), geom.y_cell(j));
  });
  mg.phi().set_val(0.0);

  const Real r0 = mg.current_residual();
  Real rn = r0;
  cycles = 0;
  while (rn > 1e-9 * r0 && cycles < 50) {
    mg.vcycle();
    rn = mg.current_residual();
    ++cycles;
  }

  // pour le cas periodique, la solution est definie a une constante pres
  Fab2D& p = mg.phi().fab(0);
  if (periodic) {
    Real mean = sum(mg.phi()) / static_cast<Real>(dom.num_cells());
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) p(i, j, 0) -= mean;
  }
  err = 0;
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
      err = std::max(err, std::fabs(p(i, j, 0) -
                                    phi_ex(geom.x_cell(i), geom.y_cell(j))));
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  // --- Dirichlet : phi = sin(pi x) sin(pi y), lap phi = -2 pi^2 phi ---
  {
    BCRec bc;
    bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;
    auto pe = [](double x, double y) {
      return std::sin(kPi * x) * std::sin(kPi * y);
    };
    auto fr = [&](double x, double y) { return -2 * kPi * kPi * pe(x, y); };

    int c32 = 0, c64 = 0;
    double e32 = 0, e64 = 0;
    solve_case(32, bc, false, pe, fr, c32, e32);
    solve_case(64, bc, false, pe, fr, c64, e64);
    std::printf("Dirichlet : c32=%d e32=%.2e | c64=%d e64=%.2e\n", c32, e32, c64,
                e64);
    chk(c64 <= 25, "dir_converged_fast");
    chk(std::abs(c64 - c32) <= 5, "dir_mesh_independent");
    chk(e64 < 5e-3, "dir_accurate");
    chk(e64 < e32, "dir_second_order");  // erreur baisse en raffinant
  }

  // --- periodique : phi = sin(2 pi x) sin(2 pi y), lap phi = -8 pi^2 phi ---
  {
    BCRec bc;  // periodique par defaut sur les 4 faces
    auto pe = [](double x, double y) {
      return std::sin(2 * kPi * x) * std::sin(2 * kPi * y);
    };
    auto fr = [&](double x, double y) { return -8 * kPi * kPi * pe(x, y); };

    int c64 = 0;
    double e64 = 0;
    solve_case(64, bc, true, pe, fr, c64, e64);
    std::printf("Periodique : c64=%d e64=%.2e\n", c64, e64);
    chk(c64 <= 30, "per_converged");
    chk(e64 < 5e-3, "per_accurate");
  }

  if (fails == 0) std::printf("OK test_geometric_mg\n");
  return fails == 0 ? 0 : 1;
}
