// Reflux multi-patch 2-niveaux DISTRIBUE avec replication grossiere, via mpirun -np N.
//
// Le grossier mono-box est REPLIQUE : chaque rang en detient une copie (DistributionMapping
// par-rang) initialisee de facon deterministe, donc identique. Les 4 patchs fins, eux, sont
// repartis. average_down + reflux remontent par buffers grossiers + all_reduce, puis chaque
// rang applique a sa copie. On execute le MEME pas dans deux repartitions des patchs fins,
// les deux par tous les rangs (collectives appariees) :
//   DIST : 4 patchs round-robin sur les rangs.
//   REF  : 4 patchs tous sur le rang 0 (equivalent np=1 ; les autres rangs participent a vide).
// Le grossier etant replique, chaque rang detient les deux resultats : comparaison bit a bit.
// DIST == REF == np=1 prouve que la distribution des patchs ne change pas le resultat.

#include <adc/integrator/amr_reflux_mf.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank();
  long fails = 0;

  const int nc = 32;
  Box2D dom = Box2D::from_extents(nc, nc);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const double dxc = geom.dx(), dyc = geom.dy();
  BoxArray bac(std::vector<Box2D>{dom});
  DistributionMapping dmc(std::vector<int>(1, me));  // grossier REPLIQUE : chaque rang a box 0

  Diocotron model;
  model.B0 = 1.0;
  model.n_i0 = 1.0;
  const double gx = 0.5, gy = -0.3;  // aux uniforme -> advection pure
  auto ne0 = [&](double x, double y) {
    return 1.0 + 0.3 * std::sin(2 * kPi * x) * std::cos(2 * kPi * y);
  };
  const double dt = 0.2 * dxc / std::hypot(gx, gy);

  // region fine [8..23]^2 decoupee en 2x2 quadrants (4 patchs, ratio 2).
  const int I0 = 8, I1 = 23, J0 = 8, J1 = 23, MI = 15, MJ = 15;
  std::vector<Box2D> faces = {
      {{2 * I0, 2 * J0}, {2 * MI + 1, 2 * MJ + 1}},
      {{2 * (MI + 1), 2 * J0}, {2 * I1 + 1, 2 * MJ + 1}},
      {{2 * I0, 2 * (MJ + 1)}, {2 * MI + 1, 2 * J1 + 1}},
      {{2 * (MI + 1), 2 * (MJ + 1)}, {2 * I1 + 1, 2 * J1 + 1}}};
  BoxArray baf(faces);

  auto fill = [&](MultiFab& U, double dx) {
    for (int li = 0; li < U.local_size(); ++li) {
      Array4 u = U.fab(li).array();
      const Box2D b = U.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          u(i, j, 0) = ne0((i + 0.5) * dx, (j + 0.5) * dx);
    }
  };
  auto fill_aux = [&](MultiFab& a) {
    for (int li = 0; li < a.local_size(); ++li) {
      Array4 ar = a.fab(li).array();
      const Box2D g = a.fab(li).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
          ar(i, j, 0) = 0; ar(i, j, 1) = gx; ar(i, j, 2) = gy;
        }
    }
  };
  // masse grossiere locale : le grossier est replique (identique sur chaque rang), donc
  // une somme LOCALE donne la vraie masse, sans all_reduce (qui multiplierait par np).
  auto local_mass = [&](const MultiFab& U) {
    const ConstArray4 c = U.fab(0).const_array();
    double m = 0;
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) m += c(i, j, 0);
    return m;
  };

  auto run = [&](const DistributionMapping& dmf) {
    MultiFab Uc(bac, dmc, 1, 1), Uf(baf, dmf, 1, 1);
    MultiFab axc(bac, dmc, 3, 1), axf(baf, dmf, 3, 1);
    fill(Uc, dxc); fill(Uf, dxc / 2); fill_aux(axc); fill_aux(axf);
    const double m0 = local_mass(Uc);
    amr_step_2level_multipatch<NoSlope, RusanovFlux>(model, Uc, dom, dxc, dyc, Uf, axc,
                                                     axf, dt);
    return std::make_pair(std::move(Uc), std::fabs(local_mass(Uc) - m0));
  };

  auto [UcDist, drift] = run(DistributionMapping(static_cast<int>(faces.size()), n_ranks()));
  auto [UcRef, driftRef] = run(DistributionMapping(std::vector<int>(faces.size(), 0)));

  // grossier replique : chaque rang detient les deux champs -> comparaison bit a bit.
  double maxdiff = 0;
  const ConstArray4 ud = UcDist.fab(0).const_array(), ur = UcRef.fab(0).const_array();
  for (int j = 0; j < nc; ++j)
    for (int i = 0; i < nc; ++i)
      maxdiff = std::fmax(maxdiff, std::fabs(ud(i, j, 0) - ur(i, j, 0)));
  maxdiff = all_reduce_max(maxdiff);
  drift = all_reduce_max(drift);

  if (me == 0)
    std::printf("reflux multipatch distribue (np=%d) : max|Uc_dist - Uc_ref| = %.3e, "
                "derive masse = %.3e\n", n_ranks(), maxdiff, drift);

  if (maxdiff > 1e-14) { if (me == 0) std::printf("FAIL distribution_invariante\n"); ++fails; }
  if (drift > 1e-12) { if (me == 0) std::printf("FAIL masse_conservee\n"); ++fails; }

  fails = all_reduce_sum(fails);
  if (fails == 0 && me == 0) std::printf("OK test_mpi_amr_multipatch\n");
  comm_finalize();
  return fails == 0 ? 0 : 1;
}
