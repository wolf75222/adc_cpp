// Etape A du solveur complet GPU : l'operateur spatial assemble_rhs (MUSCL avec
// limiteur Minmod + flux de Rusanov + source, le RHS de l'integrateur SSPRK)
// tourne sur GPU. assemble_rhs utilise DEJA for_each_cell ; il suffit que son
// lambda et la chaine de reconstruction soient device-callable (ADC_HD) et que le
// MultiFab soit en memoire unifiee. On valide contre une reference SERIE qui
// appelle EXACTEMENT les memes fonctions (reconstruct/rusanov/source, host path).

#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/operator/reconstruction.hpp>
#include <adc/operator/spatial_operator.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const int N = 128, ng = 2;  // Minmod : 2 ghosts
    Box2D dom = Box2D::from_extents(N, N);
    Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
    const double dx = geom.dx(), dy = geom.dy();
    BoxArray ba(std::vector<Box2D>{dom});
    DistributionMapping dm(1, 1);

    Diocotron model;
    model.B0 = 1.0;
    MultiFab U(ba, dm, 1, ng), aux(ba, dm, 3, ng);
    MultiFab Rg(ba, dm, 1, 0), Rr(ba, dm, 1, 0);

    auto u0 = [&](int i, int j) {
      constexpr double pi = 3.14159265358979323846;
      return 1.0 + 0.5 * std::sin(2 * pi * (i + 0.5) / N) *
                       std::sin(2 * pi * (j + 0.5) / N);
    };
    auto wrap = [&](int x) { return (x % N + N) % N; };
    {
      Fab2D& f = U.fab(0);
      const Box2D g = f.grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i) f(i, j) = u0(wrap(i), wrap(j));
      Fab2D& a = aux.fab(0);
      const Box2D ga = a.grown_box();
      for (int j = ga.lo[1]; j <= ga.hi[1]; ++j)
        for (int i = ga.lo[0]; i <= ga.hi[0]; ++i) {
          a(i, j, 0) = 0.0;
          a(i, j, 1) = 0.3;   // gx -> vy
          a(i, j, 2) = -1.0;  // gy -> vx
        }
    }

    // RHS sur GPU (for_each_cell -> Cuda) avec reconstruction MUSCL Minmod.
    assemble_rhs<Minmod>(model, U, aux, geom, Rg);
    Kokkos::fence();

    // reference SERIE : memes fonctions ADC_HD, iteration sequentielle hote.
    {
      const ConstArray4 u = U.fab(0).const_array();
      const ConstArray4 ax = aux.fab(0).const_array();
      Array4 r = Rr.fab(0).array();
      const Box2D v = Rr.box(0);
      const Minmod lim{};
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
          const Aux Ac = load_aux(ax, i, j);
          const Aux Axm = load_aux(ax, i - 1, j), Axp = load_aux(ax, i + 1, j);
          const Aux Aym = load_aux(ax, i, j - 1), Ayp = load_aux(ax, i, j + 1);
          const auto Lxm = reconstruct<Diocotron>(u, i - 1, j, 0, +1, lim);
          const auto Rxm = reconstruct<Diocotron>(u, i, j, 0, -1, lim);
          const auto Lxp = reconstruct<Diocotron>(u, i, j, 0, +1, lim);
          const auto Rxp = reconstruct<Diocotron>(u, i + 1, j, 0, -1, lim);
          const auto Fxm = rusanov_flux(model, Lxm, Axm, Rxm, Ac, 0);
          const auto Fxp = rusanov_flux(model, Lxp, Ac, Rxp, Axp, 0);
          const auto Lym = reconstruct<Diocotron>(u, i, j - 1, 1, +1, lim);
          const auto Rym = reconstruct<Diocotron>(u, i, j, 1, -1, lim);
          const auto Lyp = reconstruct<Diocotron>(u, i, j, 1, +1, lim);
          const auto Ryp = reconstruct<Diocotron>(u, i, j + 1, 1, -1, lim);
          const auto Fym = rusanov_flux(model, Lym, Aym, Rym, Ac, 1);
          const auto Fyp = rusanov_flux(model, Lyp, Ac, Ryp, Ayp, 1);
          const auto S = model.source(load_state<Diocotron>(u, i, j), Ac);
          r(i, j) = S[0] - (Fxp[0] - Fxm[0]) / dx - (Fyp[0] - Fym[0]) / dy;
        }
    }

    double maxdiff = 0;
    const ConstArray4 rg = Rg.fab(0).const_array();
    const ConstArray4 rr = Rr.fab(0).const_array();
    for (int j = 0; j < N; ++j)
      for (int i = 0; i < N; ++i)
        maxdiff = std::fmax(maxdiff, std::fabs(rg(i, j) - rr(i, j)));

    std::printf("exec=%s  N=%d  assemble_rhs<Minmod> sur GPU  maxdiff=%.3e\n",
                Kokkos::DefaultExecutionSpace::name(), N, maxdiff);
    if (maxdiff < 1e-12)
      std::printf("OK rhs_kokkos\n");
    else {
      std::printf("FAIL rhs_kokkos\n");
      rc = 1;
    }
  }
  Kokkos::finalize();
  return rc;
}
