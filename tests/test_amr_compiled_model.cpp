// Parite du chemin COMPILE add_compiled_model(AmrSystem, ...) avec le chemin NATIF add_block (dispatch
// d'une ModelSpec) cote AmrSystem. Pour le MEME modele euler_poisson (CompressibleFlux = Euler + force
// de gravite + couplage GravityCoupling) et le MEME schema (minmod x rusanov, conservatif, explicite),
// le bloc compile doit tourner EXACTEMENT le chemin de production de l'AMR (AmrCouplerMP<Model> + reflux
// conservatif + regrid) : memes masse, nombre de patchs et densite grossiere BIT-IDENTIQUES apres
// plusieurs pas. C'est ce qui donne au modele genere par le DSL la machinerie AMR du bloc natif.
//
// dispatch_model construit CompositeModel<CompressibleFlux, GravityForce, GravityCoupling> ; on passe
// au chemin compile le MEME type (CompressibleFlux == Euler, cf. hyperbolic.hpp) -> parite exacte.
#include <adc/physics/bricks.hpp>  // CompositeModel, GravityForce, GravityCoupling
#include <adc/physics/euler.hpp>   // Euler (= CompressibleFlux)
#include <adc/runtime/amr_dsl_block.hpp>
#include <adc/runtime/amr_system.hpp>
#include <adc/runtime/model_spec.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace adc;

static std::vector<double> bubble(int n) {  // bulle de densite lisse, periodique
  std::vector<double> rho(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n - 0.5, y = (j + 0.5) / n - 0.5;
      rho[static_cast<std::size_t>(j) * n + i] = 1.0 + 0.5 * std::exp(-(x * x + y * y) / 0.02);
    }
  return rho;
}

int main(int argc, char** argv) {
#if defined(ADC_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  const int n = 64;
  const std::vector<double> rho = bubble(n);

  AmrSystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;
  cfg.regrid_every = 4;

  // (A) bloc COMPILE : modele connu a la compilation, branche par add_compiled_model.
  AmrSystem A(cfg);
  using Model = CompositeModel<Euler, GravityForce, GravityCoupling>;
  add_compiled_model(A, "gas", Model{Euler{1.4}, GravityForce{}, GravityCoupling{-1.0, 1.0, 1.0}},
                     "minmod", "rusanov", "conservative", "explicit", /*gamma=*/1.4);
  A.set_poisson("charge_density", "geometric_mg");
  A.set_refinement(1.2);  // raffine la bulle
  A.set_density("gas", rho);

  // (B) bloc NATIF : meme modele via dispatch d'une ModelSpec, meme schema.
  AmrSystem B(cfg);
  ModelSpec spec;
  spec.transport = "compressible";
  spec.source = "gravity";
  spec.elliptic = "gravity";
  spec.gamma = 1.4;
  spec.sign = -1.0;
  spec.four_pi_G = 1.0;
  spec.rho0 = 1.0;
  B.add_block("gas", spec, "minmod", "rusanov", "conservative", "explicit", 1);
  B.set_poisson("charge_density", "geometric_mg");
  B.set_refinement(1.2);
  B.set_density("gas", rho);

  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  // memes patchs au depart (meme critere, meme champ) : le build compile produit la meme hierarchie.
  chk(A.n_patches() == B.n_patches(), "n_patches initial AOT == natif");
  const double m0 = A.mass();
  chk(std::fabs(m0 - B.mass()) < 1e-12 * (std::fabs(m0) + 1.0), "masse initiale AOT == natif");

  // quelques pas (regrid inclus) puis parite stricte de la densite grossiere et de la masse.
  const double dt = 1e-3;
  for (int s = 0; s < 12; ++s) {
    A.step(dt);
    B.step(dt);
  }
  const std::vector<double> da = A.density(), db = B.density();
  chk(da.size() == db.size() && !da.empty(), "densite non vide, memes tailles");
  double dmax = 0, nrm = 0;
  for (std::size_t k = 0; k < da.size() && k < db.size(); ++k) {
    dmax = std::fmax(dmax, std::fabs(da[k] - db[k]));
    nrm = std::fmax(nrm, std::fabs(db[k]));
  }
  chk(nrm > 1e-6, "densite natif non triviale");
  chk(dmax < 1e-12, "densite grossiere AOT == natif (bit-identique)");

  const double mA = A.mass(), mB = B.mass();
  chk(std::fabs(mA - mB) < 1e-12 * (std::fabs(mB) + 1.0), "masse finale AOT == natif");
  chk(A.n_patches() == B.n_patches(), "n_patches final AOT == natif (regrid identique)");

  if (fails == 0)
    std::printf(
        "OK test_amr_compiled_model (add_compiled_model == add_block sur AMR ; dmax=%.1e)\n", dmax);
  return fails ? 1 : 0;
}
