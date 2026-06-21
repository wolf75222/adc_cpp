// Chantier "Aux extensible", increment 5 : le runtime System (ce que pilote adc_cases) cable B_z.
// Un bloc COMPILE qui lit B_z (CompositeModel a source magnetisee, n_aux=4) elargit le canal aux
// PARTAGE du System (ensure_aux_width via add_compiled_model) ; set_magnetic_field(...) peuple la
// composante B_z. Le residu de production (eval_rhs = -div F + S sur les vrais MultiFab) voit alors
// B_z. On verifie le chemin de bout en bout cote runtime :
//   modele MagModel = CompositeModel<ExBVelocity, BzSource, NoEll>, flux ExB (grad=0 -> nul),
//   source S = B_z u, elliptic_rhs nul (phi=0) -> eval_rhs = B_z u = c (densite 1, B_z constant c).

#include <adc/physics/composite.hpp>
#include <adc/physics/hyperbolic.hpp>  // ExBVelocity
#include <adc/runtime/dsl_block.hpp>   // add_compiled_model
#include <adc/runtime/system.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace adc;

// Source magnetisee : S = B_z u (composante 0). Declare n_aux=4 -> le compose expose n_aux=4.
struct BzSource {
  static constexpr int n_aux = 4;
  template <class State>
  ADC_HD State apply(const State& u, const Aux& a) const {
    State s{};
    s[0] = a.B_z * u[0];
    return s;
  }
};
struct NoEll {
  template <class State>
  ADC_HD Real rhs(const State&) const {
    return Real(0);
  }
};

using MagModel = CompositeModel<ExBVelocity, BzSource, NoEll>;
static_assert(MagModel::n_aux == 4, "le compose lit B_z");

int main(int argc, char** argv) {
#if defined(ADC_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);  // Kokkos init AVANT la 1ere allocation (ctor System)
#else
  (void)argc;
  (void)argv;
#endif
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int n = 32;
  const double c = 0.7;
  std::vector<double> ones(static_cast<std::size_t>(n) * n, 1.0);
  std::vector<double> bz(static_cast<std::size_t>(n) * n, c);  // B_z = c partout

  SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;

  System sys(cfg);
  // add_compiled_model elargit le canal aux a aux_comps<MagModel> (=4) via ensure_aux_width.
  add_compiled_model(sys, "a", MagModel{}, "minmod", "rusanov", "conservative", "explicit");
  sys.set_poisson("charge_density", "geometric_mg");
  sys.set_density("a", ones);
  sys.set_magnetic_field(bz);  // peuple la composante B_z du canal partage
  sys.solve_fields();          // phi=0 (elliptic_rhs nul) ; B_z preserve

  // eval_rhs = -div F + S. flux ExB(grad=0)=0 -> R = source = B_z u = c.
  const std::vector<double> R = sys.eval_rhs("a");
  double maxerr = 0;
  for (double r : R)
    maxerr = std::fmax(maxerr, std::fabs(r - c));
  std::printf("  runtime System : eval_rhs, max|R - B_z| = %.2e (n_aux=%d)\n", maxerr,
              MagModel::n_aux);
  chk(maxerr < 1e-12, "runtime_system_reads_Bz");

  // Controle : B_z = 0 -> residu nul (la source ne contribue plus).
  std::vector<double> zero(static_cast<std::size_t>(n) * n, 0.0);
  sys.set_magnetic_field(zero);
  sys.solve_fields();
  const std::vector<double> R0 = sys.eval_rhs("a");
  double maxabs = 0;
  for (double r : R0)
    maxabs = std::fmax(maxabs, std::fabs(r));
  std::printf("  controle B_z=0 : ||R||_inf = %.2e\n", maxabs);
  chk(maxabs < 1e-12, "runtime_system_Bz_zero");

  if (fails == 0)
    std::printf("OK test_aux_runtime_bz\n");
  return fails ? 1 : 0;
}
