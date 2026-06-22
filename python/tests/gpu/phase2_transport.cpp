// Phase 1 du portage runtime GPU : un pas de transport Euler via la VRAIE machinerie adc
// (MultiFab + fill_boundary + assemble_rhs + for_each_cell), portable seriel <-> Kokkos.
// Sous un build Kokkos+CUDA, les Fab sont en memoire unifiee (ManagedAllocator) et assemble_rhs
// dispatche via for_each_cell -> Kokkos::parallel_for sur le device. On imprime un checksum
// (masse, energie) pour comparer CPU vs GPU. Compiler seriel (sans -DADC_HAS_KOKKOS) ou Kokkos+CUDA.
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/numerical_flux.hpp>
#include <adc/numerics/reconstruction.hpp>
#include <adc/numerics/spatial_operator.hpp>
#include <adc/physics/bricks/bricks.hpp>  // Euler + CompositeModel + NoSource + ChargeDensity

#include <cmath>
#include <cstdio>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace adc;

int main(int argc, char** argv) {
#if defined(ADC_HAS_KOKKOS)
  Kokkos::initialize(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  {
    const int n = 64;
    const double L = 1.0, h = L / n, gamma = 1.4;
    Geometry geom{Box2D::from_extents(n, n), 0.0, L, 0.0, L};
    BoxArray ba(std::vector<Box2D>{Box2D::from_extents(n, n)});
    DistributionMapping dm(1, 1);
    Box2D dom = Box2D::from_extents(n, n);
    BCRec bc;  // sortie libre (Foextrap) sur les 4 bords -> exerce fill_physical_bc (device)
    bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Foextrap;

    MultiFab U(ba, dm, 4, 2), R(ba, dm, 4, 0), aux(ba, dm, 3, 1);
    U.set_val(0);
    aux.set_val(0);

    // condition initiale : bulle de pression (ecriture hote ; memoire unifiee sous CUDA)
    {
      Array4 a = U.fab(0).array();
      const Box2D v = U.box(0);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
          const double x = (i + 0.5) / n - 0.5, y = (j + 0.5) / n - 0.5;
          const double p = 1.0 + 0.4 * std::exp(-(x * x + y * y) / 0.01);
          a(i, j, 0) = 1.0;
          a(i, j, 1) = 0.0;
          a(i, j, 2) = 0.0;
          a(i, j, 3) = p / (gamma - 1.0);
        }
    }

    // assemble_rhs attend un PhysicalModel complet (flux + source) : on compose Euler + NoSource
    // (+ une brique elliptique inutilisee ici, pas de Poisson en phase 1).
    using Model = CompositeModel<Euler, NoSource, ChargeDensity>;
    Model model;
    model.hyp.gamma = gamma;
    const int steps = 80;
    const double dt = 0.4 * h / 3.0;  // pas fixe stable (max wave speed ~ c ~ 1.2)

    for (int s = 0; s < steps; ++s) {
      fill_ghosts(U, dom,
                  bc);  // periodique (copy_shifted) + bords physiques (fill_physical_bc), device
      assemble_rhs<NoSlope, RusanovFlux>(model, U, aux, geom, R, false);  // for_each -> GPU
      Array4 uu = U.fab(0).array();
      Array4 rr = R.fab(0).array();
      for_each_cell(U.box(0), [=] ADC_HD(int i, int j) {  // U += dt R, sur le device
        for (int c = 0; c < 4; ++c)
          uu(i, j, c) += dt * rr(i, j, c);
      });
    }
    device_fence();

    ConstArray4 a = U.fab(0).const_array();
    const Box2D v = U.box(0);
    double mass = 0, energy = 0, rmin = 1e30, rmax = -1e30;
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        mass += a(i, j, 0);
        energy += a(i, j, 3);
        rmin = std::fmin(rmin, a(i, j, 0));
        rmax = std::fmax(rmax, a(i, j, 0));
      }
#if defined(ADC_HAS_KOKKOS)
    const char* space = Kokkos::DefaultExecutionSpace::name();
#else
    const char* space = "Serial(host)";
#endif
    std::printf("exec=%s n=%d steps=%d  mass=%.12f  energy=%.12f  rho[min,max]=[%.6f,%.6f]\n",
                space, n, steps, mass, energy, rmin, rmax);
  }
#if defined(ADC_HAS_KOKKOS)
  Kokkos::finalize();
#endif
  return 0;
}
