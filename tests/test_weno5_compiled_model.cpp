// WENO5 sur le chemin COMPILE NATIF (add_compiled_model, le gabarit en-tete qu'inline le loader .so
// de production / add_native_block). Le verrou : add_compiled_model alloue desormais l'etat du bloc a
// block_n_ghost(limiter) APRES install_block (3 ghosts pour weno5), comme add_block (PR #88) ; sans
// cette largeur, assemble_rhs lirait hors bornes le stencil 5 points de WENO5.
//
// On exige :
//  (1) PARITE STRICTE weno5 : add_compiled_model("gas", ..., "weno5") == add_block(..., "weno5") sur
//      eval_rhs ET potentiel (BIT-IDENTIQUE : MEME make_block / install_block / fill_boundary, donc
//      meme code, meme allocation 3 ghosts) ; le residu n'est pas trivial.
//  (2) NO-DEFAULT-CHANGE : pour none et minmod (<= 2 ghosts), add_compiled_model reste BIT-IDENTIQUE
//      a add_block -- la reallocation set_block_ghosts est un NO-OP (U a deja 2 ghosts), donc
//      l'allocation et le resultat sont inchanges vs avant ce chantier.
#include <adc/physics/bricks.hpp>  // CompositeModel, GravityForce, GravityCoupling
#include <adc/physics/euler.hpp>   // Euler (= CompressibleFlux)
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

namespace {

// Densite lisse (bulle gaussienne) : transport regulier ou WENO5 est pleinement actif.
std::vector<double> smooth_rho(int n) {
  std::vector<double> rho(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n - 0.5, y = (j + 0.5) / n - 0.5;
      rho[static_cast<std::size_t>(j) * n + i] = 1.0 + 0.3 * std::exp(-(x * x + y * y) / 0.02);
    }
  return rho;
}

// Residu + potentiel du bloc COMPILE NATIF (add_compiled_model) pour le schema (limiter, rusanov).
void run_compiled(int n, double L, const std::vector<double>& rho, const char* limiter,
                  std::vector<double>& R, std::vector<double>& phi) {
  SystemConfig cfg;
  cfg.n = n;
  cfg.L = L;
  cfg.periodic = true;
  System sys(cfg);
  using Model = CompositeModel<Euler, GravityForce, GravityCoupling>;
  add_compiled_model(sys, "gas", Model{Euler{1.4}, GravityForce{}, GravityCoupling{-1.0, 1.0, 1.0}},
                     limiter, "rusanov", "conservative", "explicit", /*gamma=*/1.4);
  sys.set_poisson("charge_density", "geometric_mg");
  sys.set_density("gas", rho);
  sys.solve_fields();
  R = sys.eval_rhs("gas");
  phi = sys.potential();
}

// Reference NATIVE (add_block, dispatch d'une ModelSpec euler_poisson) pour le MEME schema.
void run_native(int n, double L, const std::vector<double>& rho, const char* limiter,
                std::vector<double>& R, std::vector<double>& phi) {
  SystemConfig cfg;
  cfg.n = n;
  cfg.L = L;
  cfg.periodic = true;
  System sys(cfg);
  ModelSpec spec;
  spec.transport = "compressible";
  spec.source = "gravity";
  spec.elliptic = "gravity";
  spec.gamma = 1.4;
  spec.sign = -1.0;
  spec.four_pi_G = 1.0;
  spec.rho0 = 1.0;
  sys.add_block("gas", spec, limiter, "rusanov", "conservative", "explicit", 1, true);
  sys.set_poisson("charge_density", "geometric_mg");
  sys.set_density("gas", rho);
  sys.solve_fields();
  R = sys.eval_rhs("gas");
  phi = sys.potential();
}

// Compare compile-natif vs natif pour @p limiter : residu non trivial + ecarts max BIT-IDENTIQUES.
int compare(int n, double L, const std::vector<double>& rho, const char* limiter) {
  std::vector<double> Rc, pc, Rn, pn;
  run_compiled(n, L, rho, limiter, Rc, pc);
  run_native(n, L, rho, limiter, Rn, pn);

  double dres = 0, dphi = 0, nrm = 0;
  for (std::size_t k = 0; k < Rc.size(); ++k) {
    dres = std::fmax(dres, std::fabs(Rc[k] - Rn[k]));
    nrm = std::fmax(nrm, std::fabs(Rn[k]));
  }
  for (std::size_t k = 0; k < pc.size(); ++k)
    dphi = std::fmax(dphi, std::fabs(pc[k] - pn[k]));

  int fails = 0;
  if (!(nrm > 1e-6)) {
    std::printf("FAIL [%s] residu natif trivial\n", limiter);
    ++fails;
  }
  // Meme chemin compile (make_block + install_block + meme allocation ghost) => bit-identique.
  if (!(dres == 0.0)) {
    std::printf("FAIL [%s] eval_rhs compile != natif (ecart %.3e, attendu 0)\n", limiter, dres);
    ++fails;
  }
  if (!(dphi == 0.0)) {
    std::printf("FAIL [%s] potentiel compile != natif (ecart %.3e, attendu 0)\n", limiter, dphi);
    ++fails;
  }
  if (fails == 0)
    std::printf("OK [%s] add_compiled_model == add_block (dres=%.1e dphi=%.1e)\n", limiter, dres,
                dphi);
  return fails;
}

}  // namespace

int main(int argc, char** argv) {
#if defined(ADC_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  const int n = 48;
  const double L = 1.0;
  const std::vector<double> rho = smooth_rho(n);

  int fails = 0;
  // (1) WENO5 (3 ghosts) : le verrou de ce chantier. add_compiled_model reallue a 3 ghosts apres
  //     install_block -> parite STRICTE avec add_block (qui fait de meme via set_block_ghosts).
  fails += compare(n, L, rho, "weno5");
  // (2) NO-DEFAULT-CHANGE : none (1 ghost) et minmod (2 ghosts) <= 2 ghosts alloues par install_block
  //     -> set_block_ghosts est un no-op, allocation et resultat INCHANGES (bit-identiques) vs avant.
  fails += compare(n, L, rho, "none");
  fails += compare(n, L, rho, "minmod");

  if (fails == 0)
    std::printf("OK test_weno5_compiled_model (weno5 + no-default-change none/minmod)\n");
  return fails ? 1 : 0;
}
