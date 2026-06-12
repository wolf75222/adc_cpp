// Harnais HOTE (serie / Kokkos Serial) du critere d'arret mixte rel/abs du GeometricMG.
// Resout lap(phi) = f (bulle gaussienne, Dirichlet homogene) DEUX fois et controle :
//   (a) defaut abs_tol=0 : le 2e solve sur etat inchange cycle EXACTEMENT comme avant le patch
//       (le critere relatif force a sur-resoudre un etat deja converge -> beaucoup de cycles) ;
//   (b) abs_tol>0 : le 2e solve sort SANS cycler (early-exit, residu initial sous le plancher) ;
//   (c) le 1er solve (chemin defaut) est BIT-IDENTIQUE avec et sans le patch.
// (a) et (c) se verifient en compilant CE fichier contre le header AVANT patch (sans -DHARNESS_ABS)
// et APRES patch (avec -DHARNESS_ABS) : la ligne "BASE ..." doit etre identique caractere pour
// caractere. Compilation type (Kokkos OpenMP/Serial conda) :
//   c++ -std=c++20 -O2 -DADC_HAS_KOKKOS [-DHARNESS_ABS] -Xpreprocessor -fopenmp \
//       -I <libomp>/include -I <worktree>/include -I $ADC_KOKKOS_ROOT/include \
//       mg_cold_tolerance.cpp -L $ADC_KOKKOS_ROOT/lib -lkokkoscore -lkokkossimd -ldl \
//       -L <libomp>/lib -lomp -o mg_cold_tolerance
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/elliptic/geometric_mg.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace adc;

namespace {
void fill_rhs(GeometricMG& mg, int n) {  // bulle gaussienne centree (ecriture hote, memoire unifiee)
  Array4 r = mg.rhs().fab(0).array();
  const Box2D v = mg.rhs().box(0);
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
      const double x = (i + 0.5) / n - 0.5, y = (j + 0.5) / n - 0.5;
      r(i, j, 0) = std::exp(-(x * x + y * y) / 0.01);
    }
}

double sum_phi(GeometricMG& mg) {
  ConstArray4 p = mg.phi().fab(0).const_array();
  const Box2D v = mg.phi().box(0);
  double s = 0;
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) s += p(i, j, 0);
  return s;
}
}  // namespace

int main(int argc, char** argv) {
#if defined(ADC_HAS_KOKKOS)
  Kokkos::initialize(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  int rc = 0;
  {
    const int n = 64;
    const double L = 1.0;
    Geometry geom{Box2D::from_extents(n, n), 0.0, L, 0.0, L};
    BoxArray ba(std::vector<Box2D>{Box2D::from_extents(n, n)});
    BCRec bc;  // phi = 0 au bord (Dirichlet homogene)
    bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;

    const double rel = 1e-8;
    const int maxc = 50;

    // bloc DEFAUT (abs_tol=0) : solve #1 (converge), puis solve #2 sur etat INCHANGE (warm start
    // deja converge). Le critere relatif rapporte au nouveau r0 (minuscule) force a re-cycler.
    GeometricMG mg(geom, ba, bc);
    mg.phi().set_val(0);
    mg.rhs().set_val(0);
    fill_rhs(mg, n);
    const int c1 = mg.solve(rel, maxc);  // 2 args : compile contre baseline ET patch
    device_fence();
    const double s1 = sum_phi(mg);
    const double res1 = mg.residual();
    const int c2_default = mg.solve(rel, maxc);  // etat inchange -> sur-resolution (cout a froid)
    device_fence();
    const double s2 = sum_phi(mg);
    std::printf("BASE c1=%d sum1=%.17g res1=%.17g c2_default=%d sum2=%.17g\n",
                c1, s1, res1, c2_default, s2);

#ifdef HARNESS_ABS
    // bloc PLANCHER ABSOLU : MG frais, solve #1 (chemin defaut, doit egaler le bloc ci-dessus),
    // puis solve #2 avec abs_tol > residu converge -> early-exit (0 cycle), sans toucher au solve #1.
    GeometricMG mg2(geom, ba, bc);
    mg2.phi().set_val(0);
    mg2.rhs().set_val(0);
    fill_rhs(mg2, n);
    const int c1b = mg2.solve(rel, maxc);  // defaut (abs_tol=0)
    device_fence();
    const double s1b = sum_phi(mg2);
    const double res1b = mg2.residual();
    const double abs_floor = 1e-6;  // > residu converge (~rel*r0) et << r0 initial (~||rhs||)
    const int c2_abs = mg2.solve(rel, maxc, abs_floor);  // 3 args -> early-exit attendu
    device_fence();
    std::printf("ABS  c1b=%d sum1b=%.17g res1b=%.17g abs_floor=%.3g c2_abs=%d\n",
                c1b, s1b, res1b, abs_floor, c2_abs);

    if (c1b != c1 || s1b != s1) { std::printf("FAIL solve#1 differe du bloc defaut\n"); rc = 1; }
    if (!(res1b <= abs_floor)) { std::printf("FAIL plancher mal choisi (res1b > abs_floor)\n"); rc = 1; }
    if (c2_abs != 0) { std::printf("FAIL early-exit non declenche (c2_abs=%d)\n", c2_abs); rc = 1; }
    // abs_tol par membre : meme early-exit via set_abs_tol + solve() sans argument (chemin coupleur).
    mg2.set_abs_tol(abs_floor);
    const int c3_member = mg2.solve(rel, maxc);  // 2 args mais membre abs_tol_ pose -> NON early-exit
    // (le 2-args ignore le membre ; on verifie plutot la voie solve() no-arg) :
    (void)c3_member;
    GeometricMG mg3(geom, ba, bc);
    mg3.phi().set_val(0);
    mg3.rhs().set_val(0);
    fill_rhs(mg3, n);
    mg3.solve();  // no-arg, abs_tol_ = 0 -> resout normalement
    device_fence();
    const double res3 = mg3.residual();
    mg3.set_abs_tol(abs_floor);
    mg3.solve();  // no-arg, abs_tol_ pose, etat converge -> early-exit (residu inchange)
    device_fence();
    const double res3b = mg3.residual();
    std::printf("NOARG res_after_solve=%.17g res_after_noarg_with_floor=%.17g (egaux => early-exit)\n",
                res3, res3b);
    if (res3 != res3b) { std::printf("FAIL solve() no-arg a cycle malgre le plancher\n"); rc = 1; }
#endif
  }
#if defined(ADC_HAS_KOKKOS)
  Kokkos::finalize();
#endif
  if (rc == 0) std::printf("[OK]\n");
  return rc;
}
