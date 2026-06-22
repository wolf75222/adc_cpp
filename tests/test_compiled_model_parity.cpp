// Parite du chemin AOT "compile" (add_compiled_model, modele connu a la COMPILATION) avec le bloc
// NATIF (add_block, dispatch d'une ModelSpec). Pour le MEME modele euler_poisson et le MEME schema,
// le bloc compile doit tourner EXACTEMENT le chemin de production (fill_boundary + assemble_rhs sur
// les vrais MultiFab du System, sans marshaling) : on exige un residu eval_rhs ET un potentiel
// BIT-IDENTIQUES au bloc natif. C'est ce qui donne au modele genere par le DSL la parite Kokkos + MPI
// du bloc natif (les deux passent par le meme make_block / install_block / fill_boundary).
#include <adc/physics/bricks/bricks.hpp>  // CompositeModel, GravityForce, GravityCoupling
#include <adc/physics/fluids/euler.hpp>   // Euler (= CompressibleFlux)
#include <adc/runtime/builders/dsl_block.hpp>
#include <adc/runtime/model_spec.hpp>
#include <adc/runtime/system.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace adc;

int main(int argc, char** argv) {
#if defined(ADC_HAS_KOKKOS)
  // Sous Kokkos, l'allocateur unifie (kokkos_malloc<SharedSpace>) exige Kokkos initialise AVANT la
  // 1ere allocation (le ctor de System alloue l'aux) : ScopeGuard (RAII) avant toute construction.
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  const int n = 48;
  const double L = 1.0;
  std::vector<double> rho(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n - 0.5, y = (j + 0.5) / n - 0.5;
      rho[static_cast<std::size_t>(j) * n + i] = 1.0 + 0.3 * std::exp(-(x * x + y * y) / 0.02);
    }

  SystemConfig cfg;
  cfg.n = n;
  cfg.L = L;
  cfg.periodic = true;

  // (A) bloc COMPILE AOT : modele connu a la compilation, branche sur la grille reelle du System.
  System A(cfg);
  using Model = CompositeModel<Euler, GravityForce, GravityCoupling>;
  add_compiled_model(A, "gas", Model{Euler{1.4}, GravityForce{}, GravityCoupling{-1.0, 1.0, 1.0}},
                     "minmod", "rusanov", "conservative", "explicit", /*gamma=*/1.4);
  A.set_poisson("charge_density", "geometric_mg");
  A.set_density("gas", rho);
  A.solve_fields();
  const std::vector<double> Ra = A.eval_rhs("gas");
  const std::vector<double> pa = A.potential();

  // (B) bloc NATIF : meme modele euler_poisson via dispatch d'une ModelSpec, meme schema.
  System B(cfg);
  ModelSpec spec;
  spec.transport = "compressible";
  spec.source = "gravity";
  spec.elliptic = "gravity";
  spec.gamma = 1.4;
  spec.sign = -1.0;
  spec.four_pi_G = 1.0;
  spec.rho0 = 1.0;
  B.add_block("gas", spec, "minmod", "rusanov", "conservative", "explicit", 1, true);
  B.set_poisson("charge_density", "geometric_mg");
  B.set_density("gas", rho);
  B.solve_fields();
  const std::vector<double> Rb = B.eval_rhs("gas");
  const std::vector<double> pb = B.potential();

  double dres = 0, dphi = 0, nrm = 0;
  for (std::size_t k = 0; k < Ra.size(); ++k) {
    dres = std::fmax(dres, std::fabs(Ra[k] - Rb[k]));
    nrm = std::fmax(nrm, std::fabs(Rb[k]));
  }
  for (std::size_t k = 0; k < pa.size(); ++k)
    dphi = std::fmax(dphi, std::fabs(pa[k] - pb[k]));

  int fails = 0;
  if (!(nrm > 1e-6)) {
    std::printf("FAIL residu natif trivial\n");
    ++fails;
  }
  if (!(dres < 1e-12)) {
    std::printf("FAIL eval_rhs AOT != natif (ecart %.3e)\n", dres);
    ++fails;
  }
  if (!(dphi < 1e-12)) {
    std::printf("FAIL potentiel AOT != natif (ecart %.3e)\n", dphi);
    ++fails;
  }
  if (fails == 0)
    std::printf(
        "OK test_compiled_model_parity (add_compiled_model == add_block ; dres=%.1e dphi=%.1e)\n",
        dres, dphi);
  return fails ? 1 : 0;
}
