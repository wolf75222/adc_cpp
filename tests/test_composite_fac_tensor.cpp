// Solveur elliptique COMPOSITE FAC a TENSEUR PLEIN (use_cross_terms) : test MMS. C'est l'operateur
// condense de Schur a B_z != 0 -> A = I + c rho B^-1 = diag(eps,eps) + [[0,a_xy],[a_yx,0]] avec
// a_yx = -a_xy (antisymetrique), NON auto-adjoint. Phase 3b du Schur composite.
//
// Solution manufacturee u = sin(3 pi x) sin(3 pi y), coefficients lisses :
//   eps  = 1 + 0.3 sin(2 pi x) sin(2 pi y)   (> 0)
//   a_xy = 0.2 sin(2 pi x) sin(2 pi y),  a_yx = -a_xy.
// Avec a_yx = -a_xy, le terme (a_xy + a_yx) u_xy s'annule et il reste, pour le second membre :
//   f = div(A grad u) = eps Lap u + eps_x u_x + eps_y u_y + a_xy_x u_y - a_xy_y u_x.
// On compare, zone interieure du patch, le COMPOSITE au COARSE-ONLY (GeometricMG tenseur plein) : le
// patch fin doit REDUIRE l'erreur (phi + grad phi) -- la FAC tient avec les termes croises (explicites,
// petits pour le pas Schur ou c = theta^2 dt^2 alpha).

#include <adc/numerics/elliptic/composite_fac_poisson.hpp>

#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/elliptic/geometric_mg.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

static double u_exact(double x, double y) { return std::sin(3 * kPi * x) * std::sin(3 * kPi * y); }
static double eps_xy(double x, double y) { return 1.0 + 0.3 * std::sin(2 * kPi * x) * std::sin(2 * kPi * y); }
static double axy_xy(double x, double y) { return 0.2 * std::sin(2 * kPi * x) * std::sin(2 * kPi * y); }
static double ayx_xy(double x, double y) { return -axy_xy(x, y); }
static double f_rhs(double x, double y) {
  const double u = u_exact(x, y);
  const double ux = 3 * kPi * std::cos(3 * kPi * x) * std::sin(3 * kPi * y);
  const double uy = 3 * kPi * std::sin(3 * kPi * x) * std::cos(3 * kPi * y);
  const double ex = 0.6 * kPi * std::cos(2 * kPi * x) * std::sin(2 * kPi * y);
  const double ey = 0.6 * kPi * std::sin(2 * kPi * x) * std::cos(2 * kPi * y);
  const double axyx = 0.4 * kPi * std::cos(2 * kPi * x) * std::sin(2 * kPi * y);
  const double axyy = 0.4 * kPi * std::sin(2 * kPi * x) * std::cos(2 * kPi * y);
  return eps_xy(x, y) * (-18.0 * kPi * kPi * u) + ex * ux + ey * uy + axyx * uy - axyy * ux;
}

template <class Setter>
static void fill(MultiFab& m, const Geometry& g, Setter s) {
  for (int li = 0; li < m.local_size(); ++li) {
    Array4 a = m.fab(li).array();
    const Box2D b = m.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) a(i, j, 0) = s(g.x_cell(i), g.y_cell(j));
  }
}

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank();
  long fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { if (me == 0) std::printf("FAIL %s\n", w); ++fails; }
  };

  const int n = 48, r = 2;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom_c{dom, 0.0, 1.0, 0.0, 1.0};
  Geometry geom_f = geom_c.refine(r);
  BoxArray ba_c = BoxArray::from_domain(dom, n);
  DistributionMapping dm_c(ba_c.size(), n_ranks());
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;
  const int Ic0 = n / 4, Ic1 = 3 * n / 4 - 1;
  Box2D fine_box{{r * Ic0, r * Ic0}, {r * Ic1 + r - 1, r * Ic1 + r - 1}};

  // --- coarse-only tenseur plein (GeometricMG.set_epsilon + set_cross_terms) ---
  MultiFab eps_c(ba_c, dm_c, 1, 1), axy_c(ba_c, dm_c, 1, 1), ayx_c(ba_c, dm_c, 1, 1);
  fill(eps_c, geom_c, eps_xy); fill(axy_c, geom_c, axy_xy); fill(ayx_c, geom_c, ayx_xy);
  device_fence();
  fill_ghosts(eps_c, dom, BCRec{}); fill_ghosts(axy_c, dom, BCRec{}); fill_ghosts(ayx_c, dom, BCRec{});
  GeometricMG mg0(geom_c, ba_c, bc, {}, /*replicated=*/true);
  mg0.set_epsilon(eps_c);
  mg0.set_cross_terms(axy_c, ayx_c);
  fill(mg0.rhs(), geom_c, f_rhs);
  mg0.phi().set_val(0.0);
  mg0.solve(1e-12, 300);
  device_fence();

  // --- composite tenseur plein ---
  CompositeFacPoisson fac(geom_c, ba_c, bc, fine_box, r);
  fill(fac.eps_coarse(), geom_c, eps_xy); fill(fac.eps_fine(), geom_f, eps_xy);
  fill(fac.a_xy_coarse(), geom_c, axy_xy); fill(fac.a_xy_fine(), geom_f, axy_xy);
  fill(fac.a_yx_coarse(), geom_c, ayx_xy); fill(fac.a_yx_fine(), geom_f, ayx_xy);
  fac.use_variable_coefficient(true);
  fac.use_cross_terms(true);
  fill(fac.rhs_coarse(), geom_c, f_rhs);
  fill(fac.rhs_fine(), geom_f, f_rhs);
  const Real rfac = fac.solve(/*max_iters=*/40, /*fine_sweeps=*/80, /*tol=*/1e-10);
  device_fence();

  chk(std::isfinite(rfac) && rfac < 1e-6, "(convergence) FAC tenseur plein converge (residu -> 0)");

  const int guard = 3;
  const int iIc0 = Ic0 + guard, iIc1 = Ic1 - guard;
  const ConstArray4 PC0 = mg0.phi().fab(0).const_array();
  const ConstArray4 PF = fac.phi_fine().fab(0).const_array();
  const double dxf = geom_f.dx();
  double e_coarse = 0, e_comp = 0, eg_optA = 0, eg_comp = 0;
  for (int J = iIc0; J <= iIc1; ++J)
    for (int I = iIc0; I <= iIc1; ++I) {
      const double gxc = (PC0(I + 1, J, 0) - PC0(I - 1, J, 0)) / (2 * geom_c.dx());
      const double gyc = (PC0(I, J + 1, 0) - PC0(I, J - 1, 0)) / (2 * geom_c.dy());
      for (int tj = 0; tj < r; ++tj)
        for (int ti = 0; ti < r; ++ti) {
          const int iff = r * I + ti, jff = r * J + tj;
          const double xf = geom_f.x_cell(iff), yf = geom_f.y_cell(jff);
          const double ue = u_exact(xf, yf);
          e_coarse = std::fmax(e_coarse, std::fabs(detail::fac_bilerp_coarse(PC0, iff, jff, r) - ue));
          e_comp = std::fmax(e_comp, std::fabs(PF(iff, jff, 0) - ue));
          const double gxa = 3 * kPi * std::cos(3 * kPi * xf) * std::sin(3 * kPi * yf);
          const double gya = 3 * kPi * std::sin(3 * kPi * xf) * std::cos(3 * kPi * yf);
          const double gxf = (PF(iff + 1, jff, 0) - PF(iff - 1, jff, 0)) / (2 * dxf);
          const double gyf = (PF(iff, jff + 1, 0) - PF(iff, jff - 1, 0)) / (2 * dxf);
          eg_optA = std::fmax(eg_optA, std::fmax(std::fabs(gxc - gxa), std::fabs(gyc - gya)));
          eg_comp = std::fmax(eg_comp, std::fmax(std::fabs(gxf - gxa), std::fabs(gyf - gya)));
        }
    }
  e_coarse = all_reduce_max(e_coarse); e_comp = all_reduce_max(e_comp);
  eg_optA = all_reduce_max(eg_optA); eg_comp = all_reduce_max(eg_comp);

  if (me == 0)
    std::printf("  tenseur plein : phi e_coarse=%.3e e_comp=%.3e (x%.2f)  grad e_optA=%.3e e_comp=%.3e (x%.2f)\n",
                e_coarse, e_comp, e_coarse / std::fmax(e_comp, 1e-30), eg_optA, eg_comp,
                eg_optA / std::fmax(eg_comp, 1e-30));

  chk(std::isfinite(e_comp) && std::isfinite(eg_comp), "erreurs finies");
  chk(e_comp < 0.6 * e_coarse, "(fidelite phi) patch fin plus precis que coarse-only (tenseur plein)");
  chk(eg_comp < 0.5 * eg_optA, "(fidelite grad phi) composite >> injection Option A (tenseur plein)");

  fails = static_cast<long>(all_reduce_max(static_cast<double>(fails)));
  if (me == 0 && fails == 0) std::printf("OK test_composite_fac_tensor\n");
  comm_finalize();
  return fails == 0 ? 0 : 1;
}
