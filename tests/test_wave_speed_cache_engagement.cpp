// Cache des vitesses d'onde HLL (opt-in) : PREUVE D'ENGAGEMENT cote coeur C++. Le test Python
// (python/tests/test_wave_speed_cache.py) verifie ON == OFF + les gardes, mais sa bit-identite
// reussirait TRIVIALEMENT si le chemin cache devenait un no-op silencieux (repli par face). Ce test
// instrumente model.wave_speeds par un COMPTEUR et exerce make_block(none, hll) OFF puis ON :
//   (1) bit-exactitude : NoSlope + recon conservatif -> cache ON == OFF a 0 ulp apres N pas SSPRK2 ;
//   (2) engagement      : le chemin cache appelle wave_speeds UNE fois par cellule (pre-passe) au lieu
//       de par face -> calls_on < calls_off STRICTEMENT. Si le wiring ws_cache ou la garde HLLFlux
//       cassait (cache -> no-op), calls_on == calls_off et CE test echoue (le test Python, lui, non).
// Header-only (adc::adc seul), aucun modele physique : le compteur vit dans une Kokkos::View
// device-accessible (atomic), portable Serial / OpenMP / Cuda.

#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/layout/distribution_mapping.hpp>
#include <adc/mesh/execution/for_each.hpp>
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/multifab.hpp>
#include <adc/mesh/boundary/physical_bc.hpp>
#include <adc/numerics/spatial_operator.hpp>
#include <adc/runtime/builders/block/block_builder.hpp>

#include <Kokkos_Core.hpp>  // Kokkos::View / atomic_add / deep_copy (compteur d'appels device-accessible)

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

// Compteur d'appels a wave_speeds : Kokkos::View<long long> (memoire de l'espace d'execution par
// defaut, accessible dans le kernel device). atomic_increment -> correct sous Serial/OpenMP/Cuda.
using Counter = Kokkos::View<long long>;

// Modele isotherme 3-var (rho, mx, my), valide hyperbolique, dont wave_speeds INCREMENTE un compteur
// (et porte un cout factice DETERMINISTE busy >= 0 pour le volet timing, sans alterer lo/hi). Sert a
// PROUVER que le cache reduit le nombre d'appels (engagement) et reste bit-exact.
struct CountingIsothermal {
  static constexpr int n_vars = 3;
  using State = StateVec<3>;
  using Aux = adc::Aux;
  Real c0 = Real(1);
  int busy = 0;
  Counter calls;  // handle capture par valeur dans le kernel (donnees partagees)

  ADC_HD State flux(const State& u, const Aux&, int dir) const {
    const Real rho = u[0];
    const Real vx = u[1] / rho, vy = u[2] / rho;
    const Real p = c0 * c0 * rho;
    State F{};
    if (dir == 0) {
      F[0] = u[1];
      F[1] = u[1] * vx + p;
      F[2] = u[2] * vx;
    } else {
      F[0] = u[2];
      F[1] = u[1] * vy;
      F[2] = u[2] * vy + p;
    }
    return F;
  }
  ADC_HD Real max_wave_speed(const State& u, const Aux&, int dir) const {
    const Real v = (dir == 0 ? u[1] : u[2]) / u[0];
    const Real av = v < 0 ? -v : v;
    return av + c0;
  }
  ADC_HD void wave_speeds(const State& u, const Aux&, int dir, Real& lo, Real& hi) const {
    Kokkos::atomic_add(&calls(), 1LL);
    const Real v = (dir == 0 ? u[1] : u[2]) / u[0];
    Real acc = Real(0);
    for (int k = 0; k < busy; ++k)
      acc += std::sin(v + Real(k)) * std::cos(v - Real(k));
    const Real c = c0 + (acc - acc);  // acc-acc == 0 exact : lo/hi bit-stables quel que soit busy
    lo = v - c;
    hi = v + c;
  }
  ADC_HD State source(const State&, const Aux&) const { return State{}; }
};

static void init_state(MultiFab& U, const Geometry& geom, const Box2D& dom) {
  Array4 a = U.fab(0).array();
  for_each_cell(dom, [a, geom](int i, int j) {
    const double x = geom.x_cell(i), y = geom.y_cell(j);
    const double rho = 1.0 + 0.3 * std::sin(2 * kPi * x) * std::cos(2 * kPi * y);
    a(i, j, 0) = rho;
    a(i, j, 1) = 0.2 * rho * std::sin(2 * kPi * x);
    a(i, j, 2) = -0.15 * rho * std::cos(2 * kPi * y);
  });
}

// Compte les bits qui different entre deux MultiFab sur la boite valide (memcmp par valeur double).
static long long count_diff_bits(const MultiFab& A, const MultiFab& B, const Box2D& dom) {
  long long ndiff = 0;
  const ConstArray4 a = A.fab(0).const_array(), b = B.fab(0).const_array();
  for (int c = 0; c < 3; ++c)
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
        const double va = a(i, j, c), vb = b(i, j, c);
        if (std::memcmp(&va, &vb, sizeof(double)) != 0)
          ++ndiff;
      }
  return ndiff;
}

static long long read_counter(const Counter& c) {
  device_fence();
  auto h = Kokkos::create_mirror_view(c);
  Kokkos::deep_copy(h, c);
  return h();
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    std::printf("%s %s\n", c ? "[OK]  " : "[FAIL]", w);
    if (!c)
      ++fails;
  };

  const int n = 48;
  const double L = 1.0;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, L, 0.0, L};
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), n_ranks());
  BCRec bc;  // periodique
  MultiFab aux(ba, dm, 3,
               1);  // construit AVANT la View : initialise Kokkos via l'allocateur unifie
  aux.set_val(0.0);
  const GridContext ctx{dom, bc, geom, &aux};

  // ----- (1) bit-exactitude + (2) engagement (comptage), wave_speeds bon marche -----
  {
    Counter calls("ws_calls");
    CountingIsothermal model{Real(1), /*busy=*/0, calls};
    BlockClosures off = make_block(model, "none", "hll", ctx, false, false, "ssprk2", {}, {},
                                   nullptr, Real(0), /*wave_speed_cache=*/false);
    BlockClosures on = make_block(model, "none", "hll", ctx, false, false, "ssprk2", {}, {},
                                  nullptr, Real(0), /*wave_speed_cache=*/true);

    MultiFab Uoff(ba, dm, 3, 1), Uon(ba, dm, 3, 1), U0(ba, dm, 3, 1);
    init_state(Uoff, geom, dom);
    init_state(Uon, geom, dom);
    init_state(U0, geom, dom);

    const double dt = 0.2 * (L / n) / 2.0;  // CFL prudente (max |v|+c ~ 1.4)
    const int nsteps = 25;

    Kokkos::deep_copy(calls, 0LL);
    for (int s = 0; s < nsteps; ++s)
      off.advance(Uoff, dt, 1);
    const long long calls_off = read_counter(calls);

    Kokkos::deep_copy(calls, 0LL);
    for (int s = 0; s < nsteps; ++s)
      on.advance(Uon, dt, 1);
    const long long calls_on = read_counter(calls);

    device_fence();
    const long long ndiff = count_diff_bits(Uoff, Uon, dom);
    const long long evolved = count_diff_bits(Uoff, U0, dom);
    const double ratio = calls_on > 0 ? double(calls_off) / double(calls_on) : 0.0;
    std::printf("  n=%d nsteps=%d : ndiff_bits=%lld evolved_bits=%lld\n", n, nsteps, ndiff,
                evolved);
    std::printf("  wave_speeds calls : OFF=%lld ON=%lld  ratio=%.2fx\n", calls_off, calls_on,
                ratio);
    chk(evolved > 0, "l'etat a reellement evolue (test non creux)");
    chk(ndiff == 0, "bit-exact NoSlope+HLL : cache ON == OFF (0 ulp)");
    // PREUVE D'ENGAGEMENT : le cache pre-calcule wave_speeds par cellule, le chemin par face le rappelle
    // pour chaque face -> strictement moins d'appels. calls_on == calls_off signalerait un cache no-op.
    chk(calls_on < calls_off, "cache ENGAGE : moins d'appels wave_speeds que le chemin par face");
    chk(calls_off > 0 && calls_on > 0, "les deux chemins evaluent reellement wave_speeds");
  }

  // ----- (3) gain de temps (DIAGNOSTIC, non asserte) : wave_speeds couteux -----
  {
    Counter calls("ws_calls_costly");
    CountingIsothermal model{Real(1), /*busy=*/100, calls};  // emule moments + factorisations
    BlockClosures off = make_block(model, "none", "hll", ctx, false, false, "ssprk2", {}, {},
                                   nullptr, Real(0), false);
    BlockClosures on = make_block(model, "none", "hll", ctx, false, false, "ssprk2", {}, {},
                                  nullptr, Real(0), true);
    MultiFab Uoff(ba, dm, 3, 1), Uon(ba, dm, 3, 1);
    init_state(Uoff, geom, dom);
    init_state(Uon, geom, dom);
    const double dt = 0.2 * (L / n) / 2.0;
    const int nsteps = 10;

    auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < nsteps; ++s)
      off.advance(Uoff, dt, 1);
    device_fence();
    auto t1 = std::chrono::steady_clock::now();
    for (int s = 0; s < nsteps; ++s)
      on.advance(Uon, dt, 1);
    device_fence();
    auto t2 = std::chrono::steady_clock::now();
    const double ms_off = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double ms_on = std::chrono::duration<double, std::milli>(t2 - t1).count();
    const long long ndiff = count_diff_bits(Uoff, Uon, dom);
    std::printf(
        "  [diag] n=%d nsteps=%d busy=100 : OFF=%.1f ms ON=%.1f ms speedup=%.2fx ndiff_bits=%lld\n",
        n, nsteps, ms_off, ms_on, ms_on > 0 ? ms_off / ms_on : 0.0, ndiff);
    chk(ndiff == 0, "bit-exact (cas wave_speeds couteux) : cache ON == OFF");
  }

  std::printf(fails == 0 ? "\nALL OK\n" : "\n%d FAIL\n", fails);
  return fails == 0 ? 0 : 1;
}
