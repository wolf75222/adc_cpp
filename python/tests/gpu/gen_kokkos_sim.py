"""Genere un CAS COMPLET Euler 2D (time-stepping CFL, Rusanov, periodique) qui tourne sur GPU a
travers le seam Kokkos d'adc (for_each_cell / for_each_cell_reduce_*). On simule avec la brique
GENEREE EulerGen ET avec adc::Euler, et on compare les champs finaux (+ conservation de la masse).
Placeholder __BRICK__."""
import sys
sys.path.insert(0, "python/tests")
from test_dsl_brick import build_euler_brick

brick = build_euler_brick().emit_cpp_brick(name="EulerGen")

HARNESS = r"""// CAS COMPLET Euler 2D sur GPU via le seam Kokkos d'adc (for_each_cell).
#define ADC_HAS_KOKKOS 1
#include <Kokkos_Core.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/physics/fluids/euler.hpp>
__BRICK__
#include <cstdio>
#include <cmath>

using State = adc::StateVec<4>;
static constexpr double GAMMA = 1.4;

// Avance @p U de @p steps pas (Rusanov ordre 1, periodique, CFL) via for_each_cell (Kokkos -> GPU).
template <class Model>
void run(Model model, int n, int steps, double h, double cfl, Kokkos::View<State**> U) {
  adc::Box2D box{{0, 0}, {n - 1, n - 1}};
  Kokkos::View<State**> R("R", n, n);
  for (int s = 0; s < steps; ++s) {
    double amax = adc::for_each_cell_reduce_max(box, KOKKOS_LAMBDA(int i, int j) {
      adc::Aux a{}; State u = U(i, j);
      double sx = model.max_wave_speed(u, a, 0), sy = model.max_wave_speed(u, a, 1);
      return sx > sy ? sx : sy;
    });
    double dt = cfl * h / amax;
    adc::for_each_cell(box, KOKKOS_LAMBDA(int i, int j) {
      adc::Aux a{};
      State Uc = U(i, j);
      State Uxp = U((i + 1) % n, j), Uxm = U((i + n - 1) % n, j);
      State Uyp = U(i, (j + 1) % n), Uym = U(i, (j + n - 1) % n);
      State Fxr, Fxl, Fyr, Fyl;
      { State FL = model.flux(Uc, a, 0), FR = model.flux(Uxp, a, 0);
        double al = fmax(model.max_wave_speed(Uc, a, 0), model.max_wave_speed(Uxp, a, 0));
        for (int k = 0; k < 4; ++k) Fxr[k] = 0.5 * (FL[k] + FR[k]) - 0.5 * al * (Uxp[k] - Uc[k]); }
      { State FL = model.flux(Uxm, a, 0), FR = model.flux(Uc, a, 0);
        double al = fmax(model.max_wave_speed(Uxm, a, 0), model.max_wave_speed(Uc, a, 0));
        for (int k = 0; k < 4; ++k) Fxl[k] = 0.5 * (FL[k] + FR[k]) - 0.5 * al * (Uc[k] - Uxm[k]); }
      { State FL = model.flux(Uc, a, 1), FR = model.flux(Uyp, a, 1);
        double al = fmax(model.max_wave_speed(Uc, a, 1), model.max_wave_speed(Uyp, a, 1));
        for (int k = 0; k < 4; ++k) Fyr[k] = 0.5 * (FL[k] + FR[k]) - 0.5 * al * (Uyp[k] - Uc[k]); }
      { State FL = model.flux(Uym, a, 1), FR = model.flux(Uc, a, 1);
        double al = fmax(model.max_wave_speed(Uym, a, 1), model.max_wave_speed(Uc, a, 1));
        for (int k = 0; k < 4; ++k) Fyl[k] = 0.5 * (FL[k] + FR[k]) - 0.5 * al * (Uc[k] - Uym[k]); }
      State r;
      for (int k = 0; k < 4; ++k) r[k] = -((Fxr[k] - Fxl[k]) + (Fyr[k] - Fyl[k])) / h;
      R(i, j) = r;
    });
    adc::for_each_cell(box, KOKKOS_LAMBDA(int i, int j) {
      State u = U(i, j), r = R(i, j);
      for (int k = 0; k < 4; ++k) u[k] += dt * r[k];
      U(i, j) = u;
    });
  }
  adc::device_fence();
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 1;
  {
    const int n = 64, steps = 80;
    const double L = 1.0, h = L / n, cfl = 0.4;
    adc::Box2D box{{0, 0}, {n - 1, n - 1}};
    Kokkos::View<State**> Ug("Ug", n, n), Ur("Ur", n, n);
    auto hg = Kokkos::create_mirror_view(Ug);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        double x = (i + 0.5) / n - 0.5, y = (j + 0.5) / n - 0.5;
        double p = 1.0 + 0.4 * std::exp(-(x * x + y * y) / 0.01);
        State s{}; s[0] = 1.0; s[1] = 0.0; s[2] = 0.0; s[3] = p / (GAMMA - 1.0);
        hg(i, j) = s;
      }
    Kokkos::deep_copy(Ug, hg);
    Kokkos::deep_copy(Ur, hg);

    double mass0 = adc::for_each_cell_reduce_sum(box, KOKKOS_LAMBDA(int i, int j) { return Ug(i, j)[0]; });
    run(adc_generated::EulerGen{}, n, steps, h, cfl, Ug);  // brique GENEREE, sur GPU
    adc::Euler ref; ref.gamma = GAMMA;
    run(ref, n, steps, h, cfl, Ur);                         // oracle, meme boucle, sur GPU
    double mass1 = adc::for_each_cell_reduce_sum(box, KOKKOS_LAMBDA(int i, int j) { return Ug(i, j)[0]; });

    auto hgg = Kokkos::create_mirror_view(Ug), hrr = Kokkos::create_mirror_view(Ur);
    Kokkos::deep_copy(hgg, Ug); Kokkos::deep_copy(hrr, Ur);
    double maxdiff = 0.0, rmin = 1e30, rmax = -1e30;
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        rmin = fmin(rmin, hgg(i, j)[0]); rmax = fmax(rmax, hgg(i, j)[0]);
        for (int k = 0; k < 4; ++k) maxdiff = fmax(maxdiff, fabs(hgg(i, j)[k] - hrr(i, j)[k]));
      }
    double drel = fabs(mass1 - mass0) / mass0;
    std::printf("exec=%s  n=%d steps=%d  mass_drel=%.3e  rho[min,max]=[%.4f,%.4f]"
                "  maxdiff(EulerGen vs adc::Euler, GPU)=%.3e\n",
                Kokkos::DefaultExecutionSpace::name(), n, steps, drel, rmin, rmax, maxdiff);
    bool moved = (rmax - rmin) > 1e-3;   // dynamique non triviale
    rc = (drel < 1e-9 && maxdiff < 1e-12 && rmin > 0.0 && moved) ? 0 : 1;
  }
  Kokkos::finalize();
  return rc;
}
"""

open("/tmp/kokkos_euler_sim.cpp", "w").write(HARNESS.replace("__BRICK__", brick))
print("kokkos_euler_sim.cpp genere")
