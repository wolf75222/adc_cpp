// Etape C : Arena (pool memoire unifiee). Les etages SSPRK et les niveaux de la
// multigrille allouent/liberent en boucle des Fabs temporaires de memes tailles.
// Sans pool, chacun fait un cudaMallocManaged (lent). ManagedArena recycle les
// blocs (free-list par taille) : apres le rodage du premier pas, les pas suivants
// ne declenchent plus AUCUN cudaMallocManaged.
//
// On avance un diocotron couple periodique et on lit arena_stats() : misses =
// nombre de cudaMallocManaged reels, hits = allocations servies par le pool. On
// verifie que les misses cessent de croitre apres le premier pas et que les hits
// dominent (le pool recycle). Meme source CPU (sans pool, demo informatif) et GPU.

#include <adc/core/allocator.hpp>
#include <adc/coupling/coupler.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/operator/reconstruction.hpp>

#ifdef ADC_HAS_KOKKOS
#include <Kokkos_Core.hpp>
#endif

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

int main(int argc, char** argv) {
#ifdef ADC_HAS_KOKKOS
  Kokkos::initialize(argc, argv);
  const char* exec = Kokkos::DefaultExecutionSpace::name();
#else
  (void)argc;
  (void)argv;
  const char* exec = "serial-cpu";
#endif
  int rc = 0;
  {
    const int N = 128, ng = 2;
    Box2D dom = Box2D::from_extents(N, N);
    Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
    BoxArray ba(std::vector<Box2D>{dom});
    DistributionMapping dm(1, 1);

    Diocotron model;
    BCRec bcU, bcPhi;
    Coupler<Diocotron> cpl(model, geom, ba, bcU, bcPhi);

    MultiFab U(ba, dm, 1, ng);
    {
      Array4 u = U.fab(0).array();
      const Box2D g = U.fab(0).grown_box();
      auto wrap = [&](int x) { return (x % N + N) % N; };
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
          const int ii = wrap(i), jj = wrap(j);
          const double x = (ii + 0.5) / N, y = (jj + 0.5) / N;
          u(i, j, 0) = 1.0 + 0.3 * std::sin(2 * kPi * x) * std::sin(2 * kPi * y);
        }
    }

    const double dt = 0.02;
    const int K = 12;

    cpl.advance<Minmod>(U, dt);             // pas 1 : rodage du pool
    const ArenaStats s1 = arena_stats();
    for (int s = 1; s < K; ++s) cpl.advance<Minmod>(U, dt);  // pas 2..K
    const ArenaStats sK = arena_stats();

    const long new_mallocs = sK.misses - s1.misses;  // cudaMallocManaged apres pas 1
    const long hits_after = sK.hits - s1.hits;        // recyclages sur pas 2..K
    std::printf(
        "exec=%s  N=%d  pas=%d\n"
        "  apres pas 1 : misses=%ld hits=%ld fences=%ld reserve=%.1f Mo\n"
        "  apres pas %d : misses=%ld hits=%ld fences=%ld\n"
        "  -> nouveaux cudaMallocManaged sur pas 2..%d = %ld ; recyclages = %ld\n",
        exec, N, K, s1.misses, s1.hits, s1.fences, sK.reserved_bytes / 1.0e6, K,
        sK.misses, sK.hits, sK.fences, K, new_mallocs, hits_after);

#ifdef ADC_HAS_KOKKOS
    // pool actif : aucun nouveau cudaMallocManaged apres le rodage, recyclage massif
    const bool ok = (new_mallocs == 0) && (hits_after > sK.misses);
    if (ok)
      std::printf("OK arena_kokkos (pool stable : 0 malloc apres rodage)\n");
    else {
      std::printf("FAIL arena_kokkos (new_mallocs=%ld hits_after=%ld)\n",
                  new_mallocs, hits_after);
      rc = 1;
    }
#else
    std::printf("OK arena_kokkos (CPU : pas de pool, demo informatif)\n");
#endif
  }
#ifdef ADC_HAS_KOKKOS
  Kokkos::finalize();
#endif
  return rc;
}
