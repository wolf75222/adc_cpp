// Chantier "Aux extensible", increment 8 : T_e, 2e champ aux supplementaire, peuple par DERIVATION
// (et non fourni par l'utilisateur comme B_z). Un bloc fluide COMPRESSIBLE fournit T = p/rho ; un
// bloc qui declare lire aux('T_e') (n_aux=5) le lit. Valide la generalisation du canal aux a un 2e
// champ (composante 4) et la population derivee cote System (set_electron_temperature_from + apply_te
// recalcule a chaque solve_fields). Chemin de production : add_compiled_model + eval_rhs.

#include <pops/physics/composition/composite.hpp>
#include <pops/physics/fluids/euler.hpp>       // Euler (bloc fluide source de T_e)
#include <pops/physics/bricks/hyperbolic.hpp>  // ExBVelocity
#include <pops/physics/bricks/source.hpp>      // NoSource
#include <pops/runtime/builders/compiled/dsl_block.hpp>   // add_compiled_model
#include <pops/runtime/system.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

// Source qui lit T_e : S = T_e u (composante 0). n_aux=5 -> le compose expose n_aux=5.
struct TeSource {
  static constexpr int n_aux = 5;
  template <class State>
  POPS_HD State apply(const State& u, const Aux& a) const {
    State s{};
    s[0] = a.T_e * u[0];
    return s;
  }
};
struct NoEll {
  template <class State>
  POPS_HD Real rhs(const State&) const {
    return Real(0);
  }
};

using ProbeModel = CompositeModel<ExBVelocity, TeSource, NoEll>;  // lit T_e
using GasModel = CompositeModel<Euler, NoSource, NoEll>;          // fournit p/rho
static_assert(ProbeModel::n_aux == 5, "le probe lit T_e (composante aux 4)");

int main(int argc, char** argv) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
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

  const int n = 16;
  const double gamma = 1.4, rho_gas = 1.0, p_gas = 3.0;
  const double Te = p_gas / rho_gas;  // T = p / rho = 3

  SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;

  System sys(cfg);
  add_compiled_model(sys, "gas", GasModel{Euler{gamma}, NoSource{}, NoEll{}}, "minmod", "rusanov",
                     "conservative", "explicit", gamma);
  add_compiled_model(sys, "probe", ProbeModel{}, "minmod", "rusanov", "conservative", "explicit");
  sys.set_poisson("charge_density", "geometric_mg");

  // etat du gaz : rho=1, qte de mvt nulle, E = p/(gamma-1) -> p=3, donc T = p/rho = 3.
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  std::vector<double> Ug(4 * nn, 0.0);
  for (std::size_t k = 0; k < nn; ++k) {
    Ug[0 * nn + k] = rho_gas;
    Ug[3 * nn + k] = p_gas / (gamma - 1.0);
  }
  sys.set_state("gas", Ug);
  sys.set_density("probe", std::vector<double>(nn, 1.0));
  sys.set_electron_temperature_from("gas");  // T_e <- p/rho du gaz, recalcule a chaque solve
  sys.solve_fields();

  // eval_rhs(probe) = -div F + S ; flux ExB(grad=0)=0 -> R = source = T_e * n = Te.
  const std::vector<double> R = sys.eval_rhs("probe");
  double err = 0;
  for (double r : R)
    err = std::fmax(err, std::fabs(r - Te));
  std::printf("  T_e derive p/rho : eval_rhs(probe) max|R - T_e| = %.2e (T_e=%.3g)\n", err, Te);
  chk(err < 1e-12, "te_derived_and_read");

  if (fails == 0)
    std::printf("OK test_aux_te\n");
  return fails ? 1 : 0;
}
