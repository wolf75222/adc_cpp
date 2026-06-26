"""Genere un harnais KOKKOS : la brique EulerGen GENEREE tourne dans un Kokkos::parallel_for
(backend CUDA sur GH200), comparee a pops::Euler sur hote. Placeholder __BRICK__."""
import sys
sys.path.insert(0, "python/tests")
from test_dsl_brick import build_euler_brick

brick = build_euler_brick().emit_cpp_brick(name="EulerGen")  # CSE active

HARNESS = r"""// harnais Kokkos : brique generee via Kokkos::parallel_for (CUDA/GH200) vs pops::Euler hote.
#include <Kokkos_Core.hpp>
#include <pops/physics/fluids/euler.hpp>
__BRICK__
#include <cstdio>
#include <cmath>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  double md = 0.0;
  {
    const int n = 4;
    double hU[16] = {1.0, 0.2, -0.1, 2.5,  2.0, 0.5, 0.3, 6.0,
                     0.5, -0.2, 0.1, 1.8,  1.5, 0.0, 0.0, 3.0};
    Kokkos::View<double*> dU("U", 16), dFx("Fx", 16), dFy("Fy", 16);
    auto hUv = Kokkos::create_mirror_view(dU);
    for (int i = 0; i < 16; ++i) hUv(i) = hU[i];
    Kokkos::deep_copy(dU, hUv);

    Kokkos::parallel_for("flux", n, KOKKOS_LAMBDA(int t) {
      pops::StateVec<4> u{};
      for (int i = 0; i < 4; ++i) u[i] = dU(t * 4 + i);
      pops::Aux a{};
      pops_generated::EulerGen gen;                  // brique GENEREE, sur le device via Kokkos
      auto fx = gen.flux(u, a, 0);
      auto fy = gen.flux(u, a, 1);
      for (int i = 0; i < 4; ++i) { dFx(t * 4 + i) = fx[i]; dFy(t * 4 + i) = fy[i]; }
    });
    Kokkos::fence();

    auto hFx = Kokkos::create_mirror_view(dFx);
    auto hFy = Kokkos::create_mirror_view(dFy);
    Kokkos::deep_copy(hFx, dFx);
    Kokkos::deep_copy(hFy, dFy);

    pops::Euler ref; ref.gamma = 1.4; pops::Aux a{};
    for (int t = 0; t < n; ++t) {
      pops::StateVec<4> u{};
      for (int i = 0; i < 4; ++i) u[i] = hU[t * 4 + i];
      auto rx = ref.flux(u, a, 0); auto ry = ref.flux(u, a, 1);
      for (int i = 0; i < 4; ++i) {
        md = fmax(md, fabs(rx[i] - hFx(t * 4 + i)));
        md = fmax(md, fabs(ry[i] - hFy(t * 4 + i)));
      }
    }
    printf("exec_space=%s  maxdiff(Kokkos EulerGen vs hote pops::Euler)=%.3e\n",
           Kokkos::DefaultExecutionSpace::name(), md);
  }
  Kokkos::finalize();
  return md < 1e-12 ? 0 : 1;
}
"""

open("/tmp/kokkos_euler.cpp", "w").write(HARNESS.replace("__BRICK__", brick))
print("kokkos_euler.cpp genere")
