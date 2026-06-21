// ADC-324 regression : le patch fin SEED du chemin AMR compile (build_amr_compiled, partage par
// add_compiled_model ET le bloc natif add_block mono-bloc) n'est alloue QUE quand le raffinement est
// reellement configure (set_refinement avec un seuil fini). Sans set_refinement, refine_threshold
// reste au sentinel 1e30 "pas de raffinement" : la hierarchie est alors MONO-NIVEAU (n_patches()==0),
// comme le chemin amr-schur, donc le transport grossier se distribue proprement sous MPI. Avant ce
// correctif un patch fin central (une SEULE boite non decoupee sur la dmap grossiere -> rang 0)
// persistait (n_patches()==1) meme sans raffinement : a np=4 le rang 0 portait ses boites grossieres
// PLUS tout le patch fin, et le flux aux vitesses exactes ne scalait pas (cf. ADC-324, mesure ROMEO).
//
// Quand le raffinement EST configure (set_refinement(seuil fini)), le seed est alloue et le regrid de
// build chope + distribue exactement comme avant : n_patches()>=1, chemin raffine INCHANGE (la parite
// bit-a-bit du chemin raffine est verrouillee par test_amr_compiled_model / test_amr_riemann_native).
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

static std::vector<double> bubble(int n) {  // bulle de densite lisse (pic 1.5 > 1.2), periodique
  std::vector<double> rho(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n - 0.5, y = (j + 0.5) / n - 0.5;
      rho[static_cast<std::size_t>(j) * n + i] = 1.0 + 0.5 * std::exp(-(x * x + y * y) / 0.02);
    }
  return rho;
}

using Model = CompositeModel<Euler, GravityForce, GravityCoupling>;

int main(int argc, char** argv) {
#if defined(ADC_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  const int n = 64;
  const std::vector<double> rho = bubble(n);

  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  AmrSystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;

  // (A) SANS set_refinement (refine_threshold == 1e30) sur la config amr_scale (regrid_every=0) :
  //     hierarchie MONO-NIVEAU, aucun patch fin seed -> n_patches() == 0.
  {
    AmrSystemConfig c = cfg;
    c.regrid_every = 0;
    AmrSystem A(c);
    add_compiled_model(A, "gas", Model{Euler{1.4}, GravityForce{}, GravityCoupling{-1.0, 1.0, 1.0}},
                       "minmod", "rusanov", "conservative", "explicit", /*gamma=*/1.4);
    A.set_poisson("charge_density", "geometric_mg");
    A.set_density("gas", rho);
    chk(A.n_patches() == 0, "no set_refinement -> n_patches()==0 (compile, mono-niveau)");
    // le mono-niveau reste un solveur valide : il avance et conserve la masse (FV periodique).
    const double m0 = A.mass();
    for (int s = 0; s < 8; ++s)
      A.step(1e-3);
    const std::vector<double> d = A.density();
    double nrm = 0;
    for (double v : d)
      nrm = std::fmax(nrm, std::fabs(v));
    chk(!d.empty() && nrm > 1e-6, "no set_refinement : densite non triviale apres pas");
    chk(std::fabs(A.mass() - m0) < 1e-9 * (std::fabs(m0) + 1.0),
        "no set_refinement : masse conservee (mono-niveau)");
    chk(A.n_patches() == 0, "no set_refinement : reste mono-niveau apres pas");
  }

  // (B) AVEC set_refinement(1.2) (seuil fini, bulle a 1.5 > 1.2) : le seed est alloue et le regrid de
  //     build chope -> n_patches() >= 1 ; le raffinement reste actif au fil des pas (regrid_every>0).
  {
    AmrSystemConfig c = cfg;
    c.regrid_every = 4;
    AmrSystem B(c);
    add_compiled_model(B, "gas", Model{Euler{1.4}, GravityForce{}, GravityCoupling{-1.0, 1.0, 1.0}},
                       "minmod", "rusanov", "conservative", "explicit", /*gamma=*/1.4);
    B.set_poisson("charge_density", "geometric_mg");
    B.set_refinement(1.2);
    B.set_density("gas", rho);
    chk(B.n_patches() >= 1, "set_refinement(1.2) -> n_patches()>=1 (seed alloue + regrid)");
    for (int s = 0; s < 8; ++s)
      B.step(1e-3);
    chk(B.n_patches() >= 1, "set_refinement(1.2) : raffinement actif apres pas");
  }

  // (C) chemin NATIF (add_block via ModelSpec) : il PARTAGE build_amr_compiled, donc la meme garde
  //     s'applique -> sans set_refinement, mono-niveau (n_patches()==0).
  {
    AmrSystemConfig c = cfg;
    c.regrid_every = 0;
    AmrSystem C(c);
    ModelSpec spec;
    spec.transport = "compressible";
    spec.source = "gravity";
    spec.elliptic = "gravity";
    spec.gamma = 1.4;
    spec.sign = -1.0;
    spec.four_pi_G = 1.0;
    spec.rho0 = 1.0;
    C.add_block("gas", spec, "minmod", "rusanov", "conservative", "explicit", 1);
    C.set_poisson("charge_density", "geometric_mg");
    C.set_density("gas", rho);
    chk(C.n_patches() == 0, "no set_refinement -> n_patches()==0 (natif, builder partage)");
  }

  if (fails == 0)
    std::printf(
        "OK test_amr_seed_no_refine (seed alloue seulement si set_refinement ; "
        "no-refine -> mono-niveau n_patches==0)\n");
  return fails ? 1 : 0;
}
