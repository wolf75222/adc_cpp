// Capstone distribue (lance via mpirun -np N) : un pas d'advection complet sur
// un MultiFab reparti doit egaler, cellule par cellule, le meme pas calcule en
// serie sur un Fab unique couvrant tout le domaine. Ca exerce toute la pile :
//   distribution SFC -> fill_boundary cross-rang -> advance par fab.
//
// La reference serie est recalculee identiquement sur chaque rang (peu couteux,
// deterministe) ; chaque rang compare ses fabs locaux a cette reference. Comme
// l'arithmetique est rigoureusement la meme (memes voisins fournis par les
// ghosts, meme flux de Rusanov, meme dt), l'accord doit etre a l'arrondi.

#include <adc/integrator/amr_reflux.hpp>  // advance_fab_1c, fill_periodic_fab, *face_box
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/parallel/comm.hpp>
#include <adc/parallel/load_balance.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank(), np = n_ranks();
  long fails = 0;

  constexpr double kPi = 3.14159265358979323846;
  const int L = 64, ng = 1;
  Box2D dom = Box2D::from_extents(L, L);
  const double dx = 1.0 / L, dy = 1.0 / L, dt = 0.3 * dx;

  Diocotron m;
  m.B0 = 1.0;
  // aux constant -> vitesse E x B uniforme : vx = -gy/B0 = 1, vy = gx/B0 = 0.3.
  auto fill_aux_const = [&](Fab2D& a) {
    const Box2D g = a.grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
        a(i, j, 0) = 0.0;
        a(i, j, 1) = 0.3;   // gx
        a(i, j, 2) = -1.0;  // gy
      }
  };
  auto u0 = [&](int i, int j) {
    return 1.0 + 0.5 * std::sin(2 * kPi * (i + 0.5) / L) *
                     std::sin(2 * kPi * (j + 0.5) / L);
  };

  // --- reference serie : Fab unique sur tout le domaine ---
  Fab2D Uref(dom, 1, 1), aref(dom, 3, 1);
  fill_aux_const(aref);
  for (int j = 0; j < L; ++j)
    for (int i = 0; i < L; ++i) Uref(i, j) = u0(i, j);
  fill_periodic_fab(Uref, dom);
  {
    Fab2D fxr(xface_box(dom), 1, 0), fyr(yface_box(dom), 1, 0);
    advance_fab_1c(m, Uref, aref, dx, dy, dt, fxr, fyr);
  }

  // --- distribue : MultiFab reparti SFC ---
  BoxArray ba = BoxArray::from_domain(dom, 16);  // 4x4 = 16 boxes
  DistributionMapping dm = make_sfc_distribution(ba, np);
  MultiFab U(ba, dm, 1, ng), Aux(ba, dm, 3, ng);
  for (int li = 0; li < U.local_size(); ++li) {
    Fab2D& F = U.fab(li);
    const Box2D b = F.box();
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) F(i, j) = u0(i, j);
    fill_aux_const(Aux.fab(li));
  }
  fill_boundary(U, dom, Periodicity{true, true});  // ghosts cross-rang
  for (int li = 0; li < U.local_size(); ++li) {
    Fab2D fxf(xface_box(U.box(li)), 1, 0), fyf(yface_box(U.box(li)), 1, 0);
    advance_fab_1c(m, U.fab(li), Aux.fab(li), dx, dy, dt, fxf, fyf);
  }

  // --- comparaison cellule par cellule contre la reference serie ---
  double maxdiff = 0;
  for (int li = 0; li < U.local_size(); ++li) {
    const Fab2D& F = U.fab(li);
    const Box2D b = F.box();
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        const double d = std::fabs(F(i, j) - Uref(i, j));
        maxdiff = std::max(maxdiff, d);
        if (d > 1e-11) ++fails;
      }
  }

  const long gfails = all_reduce_sum(fails);
  const double gmax = all_reduce_max(maxdiff);
  if (me == 0) {
    std::printf("np=%d  maxdiff(distribue vs serie)=%.3e\n", np, gmax);
    if (gfails == 0)
      std::printf("OK test_mpi_advect (np=%d)\n", np);
    else
      std::printf("FAIL test_mpi_advect : %ld cellules divergent (np=%d)\n",
                  gfails, np);
  }
  comm_finalize();
  return gfails == 0 ? 0 : 1;
}
