// AmrSystem::potential() : le getter qui expose phi du NIVEAU GROSSIER (base) en (n*n) row-major,
// pendant raffine de System::potential(). Sans raffinement (seuil enorme -> un seul niveau grossier
// mono-box couvrant tout le domaine), AmrSystem resout EXACTEMENT le meme Poisson discret que System
// avec solver='geometric_mg' (meme operateur lap(phi)=f, meme rhs f = elliptic_rhs(U), meme BC, meme
// box). On verifie donc :
//   (1) forme (n*n), valeurs FINIES, champ NON TRIVIAL (variation spatiale reelle) ;
//   (2) Poisson periodique a source NEUTRE (alpha (n - n0), integrale nulle) -> phi de moyenne ~0 ;
//   (3) PARITE avec System::potential() (geometric_mg) sur le MEME modele / densite : meme phi a la
//       tolerance MG pres (les deux passent par GeometricMG sur le meme grossier mono-box) ;
//   (4) apres quelques pas (regrid inclus), potential() reste fini et non trivial (rafraichi).
// Le modele est un transport ExB pur + fond neutralisant (briques exb / none / background), proche du
// scenario diocotron qui echantillonne phi sur un cercle median (FFT azimutale).
#include <adc/runtime/amr_system.hpp>
#include <adc/runtime/model_spec.hpp>
#include <adc/runtime/system.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace adc;

// Bulle de densite lisse autour du centre, periodique. Moyenne retiree pour neutraliser la source
// (fond background n0 = moyenne) : Poisson periodique exige une integrale de second membre nulle.
static std::vector<double> blob(int n, double& mean_out) {
  std::vector<double> rho(static_cast<std::size_t>(n) * n);
  double s = 0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n - 0.5, y = (j + 0.5) / n - 0.5;
      const double v = std::exp(-(x * x + y * y) / 0.01);
      rho[static_cast<std::size_t>(j) * n + i] = v;
      s += v;
    }
  mean_out = s / (static_cast<double>(n) * n);
  return rho;
}

static ModelSpec exb_background(double n0) {
  ModelSpec spec;
  spec.transport = "exb";        // derive E x B (a divergence nulle)
  spec.source = "none";          // pas de force source
  spec.elliptic = "background";  // f = alpha (n - n0) : fond neutralisant
  spec.B0 = 1.0;
  spec.alpha = 1.0;
  spec.n0 = n0;  // fond = moyenne -> source d'integrale nulle (Poisson periodique)
  return spec;
}

int main(int argc, char** argv) {
#if defined(ADC_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  const int n = 64;
  double n0 = 0;
  const std::vector<double> rho = blob(n, n0);

  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  // --- AmrSystem SANS raffinement : un seul niveau grossier mono-box couvrant tout le domaine ---
  AmrSystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;
  cfg.regrid_every = 0;  // pas de re-raffinement apres l'init (seuil enorme de toute facon)

  AmrSystem amr(cfg);
  amr.add_block("phi_test", exb_background(n0), "minmod", "rusanov", "conservative", "explicit", 1);
  amr.set_poisson("charge_density", "geometric_mg", "auto");
  amr.set_refinement(1e30);  // jamais : un seul niveau (base couvre tout le domaine)
  amr.set_density("phi_test", rho);

  const std::vector<double> pa = amr.potential();

  // (1) forme + valeurs finies + non trivial
  chk(static_cast<int>(pa.size()) == n * n, "potential() rend n*n valeurs");
  bool all_finite = true;
  double pmin = pa.empty() ? 0 : pa[0], pmax = pa.empty() ? 0 : pa[0], psum = 0;
  for (double v : pa) {
    if (!std::isfinite(v))
      all_finite = false;
    pmin = std::fmin(pmin, v);
    pmax = std::fmax(pmax, v);
    psum += v;
  }
  chk(all_finite, "potential() : toutes les valeurs sont finies");
  chk((pmax - pmin) > 1e-6, "potential() : champ non trivial (variation spatiale)");

  // (2) Poisson periodique a source neutre -> phi defini a une constante pres, moyenne ~ 0
  const double pmean = psum / (static_cast<double>(n) * n);
  chk(std::fabs(pmean) < 1e-6 * (pmax - pmin) + 1e-9, "potential() : moyenne ~0 (source neutre)");

  // --- System (solver geometric_mg) sur le MEME modele/densite : oracle de parite ---
  SystemConfig scfg;
  scfg.n = n;
  scfg.L = 1.0;
  scfg.periodic = true;
  System sys(scfg);
  sys.add_block("phi_test", exb_background(n0), "minmod", "rusanov", "conservative", "explicit", 1);
  sys.set_poisson("charge_density", "geometric_mg", "auto");
  sys.set_density("phi_test", rho);
  sys.solve_fields();
  const std::vector<double> ps = sys.potential();
  chk(ps.size() == pa.size(), "System.potential() meme taille qu'AmrSystem.potential()");

  // (3) parite a une constante additive pres (phi periodique defini modulo une constante) : on
  // compare apres recentrage sur la moyenne. Tolerance MG : meme operateur, meme rhs, meme box ->
  // l'ecart vient des iterations MG (rel_tol), pas du modele. Borne large mais discriminante.
  double smean = 0;
  for (double v : ps)
    smean += v;
  smean /= (static_cast<double>(n) * n);
  double dmax = 0, ref = 0;
  for (std::size_t k = 0; k < pa.size() && k < ps.size(); ++k) {
    dmax = std::fmax(dmax, std::fabs((pa[k] - pmean) - (ps[k] - smean)));
    ref = std::fmax(ref, std::fabs(ps[k] - smean));
  }
  chk(ref > 1e-6, "System phi non trivial (oracle valide)");
  chk(dmax < 1e-3 * (ref + 1e-12),
      "AmrSystem.potential() == System.potential() (geometric_mg) a la tolerance MG pres");

  // (4) apres quelques pas (transport ExB + regrid), potential() reste fini et non trivial
  amr.advance(1e-3, 8);
  const std::vector<double> pa2 = amr.potential();
  chk(static_cast<int>(pa2.size()) == n * n, "potential() apres advance : n*n");
  bool finite2 = true;
  double p2min = pa2[0], p2max = pa2[0];
  for (double v : pa2) {
    if (!std::isfinite(v))
      finite2 = false;
    p2min = std::fmin(p2min, v);
    p2max = std::fmax(p2max, v);
  }
  chk(finite2, "potential() apres advance : valeurs finies");
  chk((p2max - p2min) > 1e-6, "potential() apres advance : champ non trivial");

  if (fails == 0)
    std::printf("OK test_amr_potential (phi grossier expose ; parite System dmax/ref=%.1e/%.1e)\n",
                dmax, ref);
  return fails ? 1 : 0;
}
