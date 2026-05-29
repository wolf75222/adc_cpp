// Etape 2 du portage GPU : un pas d'advection sur le VRAI Fab2D de la
// bibliotheque, alloue en memoire UNIFIEE (cudaMallocManaged via fab_allocator),
// execute par for_each_cell (backend Kokkos -> Cuda) DIRECTEMENT sur les vues
// Array4 du Fab. Aucun buffer device manuel, aucun deep_copy : le meme Fab sert
// au code hote (remplissage, lecture, reference CPU) et au kernel device, grace a
// la coherence Grace+Hopper du GH200. C'est ce qui permettra aux operateurs
// existants (assemble_rhs, coupleurs) de tourner sur GPU sans reecriture.
//
// Build : examples/gpu/CMakeLists.txt (ADC_HAS_KOKKOS + nvcc_wrapper).

#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;

ADC_HD inline double rus(double uL, double uR, double v) {
  const double a = v < 0 ? -v : v;
  return 0.5 * v * (uL + uR) - 0.5 * a * (uR - uL);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const int N = 256, ng = 1;
    const double dx = 1.0 / N, dy = 1.0 / N, dt = 0.3 * dx, vx = 1.0, vy = 0.3;
    Box2D dom = Box2D::from_extents(N, N);

    // Fab2D en memoire unifiee (fab_allocator = cudaMallocManaged sous CUDA).
    Fab2D U(dom, 1, ng), Un(dom, 1, 0);
    auto u0 = [&](int i, int j) {
      constexpr double pi = 3.14159265358979323846;
      return 1.0 + 0.5 * std::sin(2 * pi * (i + 0.5) / N) *
                       std::sin(2 * pi * (j + 0.5) / N);
    };
    for (int j = 0; j < N; ++j)
      for (int i = 0; i < N; ++i) U(i, j) = u0(i, j);  // ecriture hote
    auto wrap = [&](int x) { return (x % N + N) % N; };
    for (int j = -ng; j < N + ng; ++j)
      for (int i = -ng; i < N + ng; ++i)
        if (i < 0 || i >= N || j < 0 || j >= N) U(i, j) = u0(wrap(i), wrap(j));

    // reference CPU (sur le meme Fab, acces hote).
    std::vector<double> ref(static_cast<std::size_t>(N) * N);
    for (int j = 0; j < N; ++j)
      for (int i = 0; i < N; ++i) {
        const double fxL = rus(U(i - 1, j), U(i, j), vx);
        const double fxR = rus(U(i, j), U(i + 1, j), vx);
        const double fyB = rus(U(i, j - 1), U(i, j), vy);
        const double fyT = rus(U(i, j), U(i, j + 1), vy);
        ref[j * N + i] = U(i, j) - dt * ((fxR - fxL) / dx + (fyT - fyB) / dy);
      }

    // GPU : for_each_cell directement sur les vues Array4 du Fab (memoire
    // unifiee accessible device). Pas de copie host<->device.
    const ConstArray4 u = U.const_array();
    Array4 un = Un.array();
    for_each_cell(Box2D{{0, 0}, {N - 1, N - 1}}, [=] ADC_HD(int i, int j) {
      const double fxL = rus(u(i - 1, j), u(i, j), vx);
      const double fxR = rus(u(i, j), u(i + 1, j), vx);
      const double fyB = rus(u(i, j - 1), u(i, j), vy);
      const double fyT = rus(u(i, j), u(i, j + 1), vy);
      un(i, j) = u(i, j) - dt * ((fxR - fxL) / dx + (fyT - fyB) / dy);
    });
    Kokkos::fence();

    // comparaison (lecture hote du meme Fab, coherent).
    double maxdiff = 0;
    for (int j = 0; j < N; ++j)
      for (int i = 0; i < N; ++i)
        maxdiff = std::fmax(maxdiff, std::fabs(Un(i, j) - ref[j * N + i]));

    std::printf("exec=%s  N=%d  Fab2D=memoire unifiee  maxdiff(GPU vs CPU)=%.3e\n",
                Kokkos::DefaultExecutionSpace::name(), N, maxdiff);
    if (maxdiff < 1e-12)
      std::printf("OK advect_fab_kokkos\n");
    else {
      std::printf("FAIL advect_fab_kokkos\n");
      rc = 1;
    }
  }
  Kokkos::finalize();
  return rc;
}
