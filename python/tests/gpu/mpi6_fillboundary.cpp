// Phase 6 (multi-GPU) : echange de halos DISTRIBUE sur device. Identique a
// tests/test_mpi_fillboundary.cpp (remplissage par valeur periodique repliee + verif des ghosts)
// mais sous Kokkos+CUDA : MultiFab en memoire unifiee, fill_boundary fait le chemin local (for_each
// device) + le chemin MPI cross-rang (OpenMPI CUDA-aware). Avec np>1 sur plusieurs GH200, les ghosts
// viennent de fabs distants -> valide le transfert device-to-device. device_fence() avant la lecture
// hote (kernels async + MPI). Invariant au nombre de rangs : OK en np=1/2/4.
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/parallel/comm.hpp>
#include <adc/parallel/load_balance.hpp>

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>

using namespace adc;

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  long gfails = 0;
  {
    const int me = my_rank(), np = n_ranks();
    const int L = 64, ng = 1, ncomp = 2;
    Box2D dom = Box2D::from_extents(L, L);
    auto wrap = [&](int x) { return ((x % L) + L) % L; };
    auto val = [&](int i, int j, int c) {
      return double(wrap(i)) + 0.001 * double(wrap(j)) + 100.0 * c;
    };
    BoxArray ba = BoxArray::from_domain(dom, 16);  // 16 boxes reparties sur np rangs
    DistributionMapping dm = make_sfc_distribution(ba, np);
    MultiFab mf(ba, dm, ncomp, ng);

    for (int li = 0; li < mf.local_size(); ++li) {  // remplir les cellules valides (hote, unifiee)
      Fab2D& F = mf.fab(li);
      const Box2D b = F.box();
      for (int c = 0; c < ncomp; ++c)
        for (int j = b.lo[1]; j <= b.hi[1]; ++j)
          for (int i = b.lo[0]; i <= b.hi[0]; ++i)
            F(i, j, c) = val(i, j, c);
    }

    fill_boundary(mf, dom, Periodicity{true, true});  // local for_each + MPI cross-rang
    device_fence();                                   // avant lecture hote des ghosts

    long fails = 0;
    for (int li = 0; li < mf.local_size(); ++li) {
      const Fab2D& F = mf.fab(li);
      const Box2D g = F.box().grow(ng);
      for (int c = 0; c < ncomp; ++c)
        for (int j = g.lo[1]; j <= g.hi[1]; ++j)
          for (int i = g.lo[0]; i <= g.hi[0]; ++i)
            if (std::fabs(F(i, j, c) - val(i, j, c)) > 1e-12)
              ++fails;
    }
    gfails = all_reduce_sum(fails);
#if defined(ADC_HAS_KOKKOS)
    const char* space = Kokkos::DefaultExecutionSpace::name();
#else
    const char* space = "Serial(host)";
#endif
    if (me == 0)
      std::printf("%s test_mpi6_fillboundary  np=%d boxes=%d exec=%s  (gfails=%ld)\n",
                  gfails == 0 ? "OK" : "FAIL", np, ba.size(), space, gfails);
  }
  Kokkos::finalize();
  comm_finalize();
  return gfails == 0 ? 0 : 1;
}
