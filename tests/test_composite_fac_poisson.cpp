// Solveur Poisson COMPOSITE FAC 2 niveaux (CompositeFacPoisson) : test MMS / convergence.
// Cf. include/adc/numerics/elliptic/composite_fac_poisson.hpp.
//
// On resout Lap phi = f sur un domaine [0,1]^2 carre, Dirichlet phi = 0 au bord, avec une solution
// MANUFACTUREE u_exact = sin(3 pi x) sin(3 pi y) (nulle au bord), f = Lap u = -18 pi^2 u. On compare,
// dans la zone INTERIEURE du patch fin (loin du bord C-F pour eviter la contamination) :
//   - le potentiel COARSE-ONLY (un seul solve grossier), interpole bilineairement aux centres fins ;
//   - le potentiel COMPOSITE (solve grossier + patch fin couple par FAC), au meme centres fins.
// CRITERE (le verrou de fidelite AMR, demande par le user) : le patch fin doit donner un phi PLUS
// PRECIS que le coarse-only -- e_composite < e_coarse dans la zone raffinee -- et la difference doit
// suivre le facteur de raffinement (~ (H/h)^2 = 4 pour une solution lisse). On verifie aussi que
// l'iteration FAC CONVERGE (residu composite -> petit) et que phi_composite != phi_coarse (le patch
// change effectivement la solution, pas un no-op).
//
// Serie (Kokkos OFF). Le grossier est mono-box replique ; la FAC mono-rang est validee ici.

#include <adc/numerics/elliptic/composite_fac_poisson.hpp>

#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/elliptic/geometric_mg.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

static double u_exact(double x, double y) {
  return std::sin(3.0 * kPi * x) * std::sin(3.0 * kPi * y);
}
static double f_rhs(double x, double y) {  // Lap u = -(9+9) pi^2 u
  return -18.0 * kPi * kPi * u_exact(x, y);
}

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank();
  long fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { if (me == 0) std::printf("FAIL %s\n", w); ++fails; }
  };

  const int n = 48;                 // grossier
  const int r = 2;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom_c{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba_c = BoxArray::from_domain(dom, n);  // mono-box couvrant le domaine
  DistributionMapping dm_c(ba_c.size(), n_ranks());
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;  // phi = 0 au bord

  // Patch fin sur la moitie centrale grossiere [n/4, 3n/4) -> box fine [2*n/4, 2*3n/4 - 1].
  const int Ic0 = n / 4, Ic1 = 3 * n / 4 - 1;  // empreinte grossiere du patch
  Box2D fine_box{{r * Ic0, r * Ic0}, {r * Ic1 + r - 1, r * Ic1 + r - 1}};
  Geometry geom_f = geom_c.refine(r);

  // --- second membre f aux centres (grossier et fin) ---
  MultiFab f_c(ba_c, dm_c, 1, 0);
  for (int li = 0; li < f_c.local_size(); ++li) {
    Array4 a = f_c.fab(li).array();
    const Box2D b = f_c.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        a(i, j, 0) = f_rhs(geom_c.x_cell(i), geom_c.y_cell(j));
  }

  // --- (1) COARSE-ONLY : un seul solve grossier ---
  GeometricMG mg0(geom_c, ba_c, bc, {}, /*replicated=*/true);
  for (int li = 0; li < mg0.rhs().local_size(); ++li) {
    Array4 a = mg0.rhs().fab(li).array();
    const ConstArray4 s = f_c.fab(li).const_array();
    const Box2D b = mg0.rhs().box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) a(i, j, 0) = s(i, j, 0);
  }
  mg0.phi().set_val(0.0);
  mg0.solve(1e-12, 100);
  device_fence();

  // --- (2) COMPOSITE FAC ---
  CompositeFacPoisson fac(geom_c, ba_c, bc, fine_box, r);
  // rhs grossier
  for (int li = 0; li < fac.rhs_coarse().local_size(); ++li) {
    Array4 a = fac.rhs_coarse().fab(li).array();
    const ConstArray4 s = f_c.fab(li).const_array();
    const Box2D b = fac.rhs_coarse().box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) a(i, j, 0) = s(i, j, 0);
  }
  // rhs fin
  {
    Array4 a = fac.rhs_fine().fab(0).array();
    const Box2D b = fac.rhs_fine().box(0);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        a(i, j, 0) = f_rhs(geom_f.x_cell(i), geom_f.y_cell(j));
  }
  const Real rfac = fac.solve(/*max_iters=*/40, /*fine_sweeps=*/80, /*tol=*/1e-10);
  device_fence();

  chk(std::isfinite(rfac), "FAC residu fini");
  // residu composite reduit (relatif a l'echelle de f ~ 18pi^2 ~ 178).
  chk(rfac < 1e-2, "FAC converge (residu composite petit)");

  // --- comparaison dans la zone INTERIEURE du patch (retrecie de 'guard' cellules grossieres) ---
  const int guard = 3;  // marge en cellules GROSSIERES pour eviter la contamination du bord C-F
  const int iIc0 = Ic0 + guard, iIc1 = Ic1 - guard;
  const ConstArray4 PC0 = mg0.phi().fab(0).const_array();  // coarse-only
  const ConstArray4 PF = fac.phi_fine().fab(0).const_array();

  const double dxc = geom_c.dx(), dxf = geom_f.dx();
  double e_coarse = 0, e_comp = 0, diff_cf = 0;       // erreur sur phi
  double eg_optA = 0, eg_comp = 0;                     // erreur sur grad phi (la quantite physique ExB)
  for (int J = iIc0; J <= iIc1; ++J)
    for (int I = iIc0; I <= iIc1; ++I) {
      // grad phi GROSSIER au centre (I,J) : centre, = ce qu'Option A injecte (constant par morceaux) aux fins.
      const double gxc = (PC0(I + 1, J, 0) - PC0(I - 1, J, 0)) / (2 * dxc);
      const double gyc = (PC0(I, J + 1, 0) - PC0(I, J - 1, 0)) / (2 * dxc);
      for (int tj = 0; tj < r; ++tj)
        for (int ti = 0; ti < r; ++ti) {
          const int iff = r * I + ti, jff = r * J + tj;
          const double xf = geom_f.x_cell(iff), yf = geom_f.y_cell(jff);
          const double ue = u_exact(xf, yf);
          // phi : coarse-only interpole bilineairement aux centres fins ; composite = phi_f.
          const double pc = detail::fac_bilerp_coarse(PC0, iff, jff, r);
          const double pf = PF(iff, jff, 0);
          e_coarse = std::fmax(e_coarse, std::fabs(pc - ue));
          e_comp = std::fmax(e_comp, std::fabs(pf - ue));
          diff_cf = std::fmax(diff_cf, std::fabs(pf - pc));
          // grad phi : Option A (grossier constant par morceaux) vs composite (diff centree FINE).
          const double gxa = 3.0 * kPi * std::cos(3 * kPi * xf) * std::sin(3 * kPi * yf);
          const double gya = 3.0 * kPi * std::sin(3 * kPi * xf) * std::cos(3 * kPi * yf);
          const double gxf = (PF(iff + 1, jff, 0) - PF(iff - 1, jff, 0)) / (2 * dxf);
          const double gyf = (PF(iff, jff + 1, 0) - PF(iff, jff - 1, 0)) / (2 * dxf);
          eg_optA = std::fmax(eg_optA, std::fmax(std::fabs(gxc - gxa), std::fabs(gyc - gya)));
          eg_comp = std::fmax(eg_comp, std::fmax(std::fabs(gxf - gxa), std::fabs(gyf - gya)));
        }
    }
  e_coarse = all_reduce_max(e_coarse);
  e_comp = all_reduce_max(e_comp);
  diff_cf = all_reduce_max(diff_cf);
  eg_optA = all_reduce_max(eg_optA);
  eg_comp = all_reduce_max(eg_comp);

  if (me == 0)
    std::printf("  phi: e_coarse=%.3e e_composite=%.3e (x%.2f)  gradphi: e_optionA=%.3e e_composite=%.3e (x%.2f)  rfac=%.2e\n",
                e_coarse, e_comp, e_coarse / std::fmax(e_comp, 1e-30), eg_optA, eg_comp,
                eg_optA / std::fmax(eg_comp, 1e-30), rfac);

  chk(std::isfinite(e_comp) && std::isfinite(e_coarse), "erreurs finies");
  chk(rfac < 1e-6, "(convergence) l'iteration FAC converge (residu composite -> 0)");
  // CRITERE PRINCIPAL (fidelite) : le patch fin REDUIT l'erreur elliptique dans la zone raffinee.
  chk(e_comp < 0.6 * e_coarse, "(fidelite phi) patch fin plus precis que coarse-only (e_comp < 0.6 e_coarse)");
  // grad phi (la quantite physique de la derive ExB) : composite NETTEMENT meilleur que l'injection
  // Option A (grad grossier constant par morceaux). C'est ce que le couplage elliptique raffine gagne.
  chk(eg_comp < 0.5 * eg_optA, "(fidelite grad phi) composite plus precis qu'injection Option A (eg_comp < 0.5 eg_optA)");
  // le patch CHANGE effectivement la solution (pas un no-op / pas une simple injection coarse).
  chk(diff_cf > 1e-4, "le patch fin change la solution (composite != coarse interpole)");

  fails = static_cast<long>(all_reduce_max(static_cast<double>(fails)));
  if (me == 0 && fails == 0) std::printf("OK test_composite_fac_poisson\n");
  comm_finalize();
  return fails == 0 ? 0 : 1;
}
