// Phase 3 du portage runtime : un solve POISSON (GeometricMG) sur la vraie pile pops. Toute la boucle
// V-cycle (smoother red-black GS, residu, restriction average_down, prolongation interpolate, norme
// via reduce) est deja en for_each -> device. Seul le setup (masque/coefs cut-cell, ici absent) est
// hote. On resout lap phi = f (Dirichlet phi=0 au bord) et on compare CPU vs GPU. Portable seriel/Kokkos.
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>

#include <cmath>
#include <cstdio>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

int main(int argc, char** argv) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::initialize(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  int cycles = 0;
  double sum_phi = 0, max_phi = 0;
  {
    const int n = 128;
    const double L = 1.0;
    Geometry geom{Box2D::from_extents(n, n), 0.0, L, 0.0, L};
    BoxArray ba(std::vector<Box2D>{Box2D::from_extents(n, n)});
    BCRec bc;  // phi = 0 au bord (Dirichlet homogene)
    bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;

    GeometricMG mg(geom, ba, bc);
    mg.phi().set_val(0);
    mg.rhs().set_val(0);
    {  // second membre : une bulle gaussienne centree (ecriture hote, memoire unifiee)
      Array4 r = mg.rhs().fab(0).array();
      const Box2D v = mg.rhs().box(0);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
          const double x = (i + 0.5) / n - 0.5, y = (j + 0.5) / n - 0.5;
          r(i, j, 0) = std::exp(-(x * x + y * y) / 0.01);
        }
    }

    cycles = mg.solve(1e-10, 200);  // V-cycles : kernels device, reduce device pour le critere
    device_fence();

    ConstArray4 p = mg.phi().fab(0).const_array();
    const Box2D v = mg.phi().box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        sum_phi += p(i, j, 0);
        max_phi = std::fmax(max_phi, std::fabs(p(i, j, 0)));
      }
  }
#if defined(POPS_HAS_KOKKOS)
  const char* space = Kokkos::DefaultExecutionSpace::name();
#else
  const char* space = "Serial(host)";
#endif
  std::printf("exec=%s n=128 cycles=%d  sum(phi)=%.12f  max|phi|=%.12f\n", space, cycles, sum_phi,
              max_phi);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::finalize();
#endif
  return 0;
}
