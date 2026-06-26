// Phase 4 du portage runtime : le KERNEL de couplage inter-especes (ionisation) en for_each_cell,
// sur la vraie infra MultiFab (memoire unifiee sous CUDA). Reproduit la math de System::add_ionization
// (n_g -= dt k n_e n_g ; n_i += ; n_e +=) et verifie la conservation n_i + n_g. Portable seriel/Kokkos.
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>

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
  {
    const int n = 64;
    BoxArray ba(std::vector<Box2D>{Box2D::from_extents(n, n)});
    DistributionMapping dm(1, 1);
    MultiFab Ue(ba, dm, 1, 0), Ui(ba, dm, 1, 0), Ug(ba, dm, 1, 0);
    Ue.set_val(0.5);  // n_e
    Ug.set_val(1.0);  // n_g (neutres)
    Ui.set_val(0.0);  // n_i

    const Real k = 0.3, dt = 0.01;
    const int steps = 50;
    for (int s = 0; s < steps; ++s) {
      Array4 ue = Ue.fab(0).array();
      Array4 ui = Ui.fab(0).array();
      Array4 ug = Ug.fab(0).array();
      for_each_cell(Ue.box(0), [=] POPS_HD(int i, int j) {  // ionisation, sur device
        const Real dn = dt * k * ue(i, j, 0) * ug(i, j, 0);
        ug(i, j, 0) -= dn;
        ui(i, j, 0) += dn;
        ue(i, j, 0) += dn;
      });
    }
    device_fence();

    ConstArray4 ae = Ue.fab(0).const_array(), ai = Ui.fab(0).const_array(),
                ag = Ug.fab(0).const_array();
    const Box2D v = Ue.box(0);
    double ne = 0, ni = 0, ng = 0;
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        ne += ae(i, j, 0);
        ni += ai(i, j, 0);
        ng += ag(i, j, 0);
      }
#if defined(POPS_HAS_KOKKOS)
    const char* space = Kokkos::DefaultExecutionSpace::name();
#else
    const char* space = "Serial(host)";
#endif
    std::printf("exec=%s steps=%d  n_e=%.10f  n_i=%.10f  n_g=%.10f  (n_i+n_g)=%.10f\n", space,
                steps, ne, ni, ng, ni + ng);
  }
#if defined(POPS_HAS_KOKKOS)
  Kokkos::finalize();
#endif
  return 0;
}
