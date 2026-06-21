// Contrat mono-bloc de la facade AmrSystem : les parametres NON cables doivent etre REFUSES
// explicitement (std::runtime_error), plus de no-op silencieux. Avant ce nettoyage, set_poisson
// stockait rhs/solver sans jamais les valider (on pouvait croire que solver='fft' tournait sur la
// hierarchie alors qu'AmrCouplerMP cable toujours GeometricMG), et add_block acceptait n'importe
// quel time. Ce test verrouille les refus -- et, depuis le cablage IMEX (source raide implicite),
// l'ACCEPTATION de time='imex' (seul un time hors {explicit, imex} est refuse). Il compile
// python/amr_system.cpp avec le test, la classe AmrSystem etant la facade des bindings.

#include <adc/runtime/amr_system.hpp>
#include <adc/runtime/model_spec.hpp>

#include "test_harness.hpp"  // adc::test::Checker (style verbose) + raises partages

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace adc;
using adc::test::raises;  // true si l'appelable leve std::runtime_error (le refus attendu)

// Bloc ExB scalaire minimal valide (diocotron-like), pour exercer les chemins de refus.
static ModelSpec exb_spec() {
  ModelSpec s;
  s.transport = "exb";
  s.source = "none";
  s.elliptic = "charge";
  return s;
}

int main(int argc, char** argv) {
#if defined(ADC_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  adc::test::Checker checker{adc::test::Checker::Style::Verbose};  // imprime [OK ]/[XX ] par ligne
  auto chk = [&](bool c, const char* w) { checker(c, w); };

  AmrSystemConfig cfg;
  cfg.n = 16;
  cfg.L = 1.0;
  cfg.periodic = true;

  // --- set_poisson : refus immediat de solver/rhs hors du domaine cable ---------------------
  chk(raises([&] {
        AmrSystem s(cfg);
        s.set_poisson("charge_density", "fft");
      }),
      "set_poisson refuse solver='fft' (seul geometric_mg est cable sur AMR)");
  chk(raises([&] {
        AmrSystem s(cfg);
        s.set_poisson("charge_density", "inconnu");
      }),
      "set_poisson refuse un solver inconnu");
  chk(raises([&] {
        AmrSystem s(cfg);
        s.set_poisson("densite_bidon", "geometric_mg");
      }),
      "set_poisson refuse un rhs hors {charge_density, composite}");

  // Les valeurs supportees passent sans lever.
  chk(!raises([&] {
    AmrSystem s(cfg);
    s.set_poisson("charge_density", "geometric_mg");
  }),
      "set_poisson accepte charge_density + geometric_mg");
  chk(!raises([&] {
    AmrSystem s(cfg);
    s.set_poisson("composite", "geometric_mg");
  }),
      "set_poisson accepte rhs='composite'");

  // --- set_poisson : bc/wall valides au build (poisson_bc/wall_active), donc au 1er mass() ---
  chk(raises([&] {
        AmrSystem s(cfg);
        s.add_block("ne", exb_spec(), "none", "rusanov", "conservative", "explicit", 1);
        s.set_poisson("charge_density", "geometric_mg", "bc_bidon");
        (void)s.mass();  // declenche ensure_built -> poisson_bc()
      }),
      "bc inconnu refuse au build");
  chk(raises([&] {
        AmrSystem s(cfg);
        s.add_block("ne", exb_spec(), "none", "rusanov", "conservative", "explicit", 1);
        s.set_poisson("charge_density", "geometric_mg", "auto", "mur_bidon");
        (void)s.mass();  // declenche ensure_built -> wall_active()
      }),
      "wall inconnu refuse au build");

  // --- add_block : time={explicit, imex} ACCEPTE, tout autre traitement REFUSE -----------------
  // time='imex' est desormais cable sur AMR (source raide implicite via backward_euler_source ;
  // transport explicite porte par le reflux). On verrouille donc qu'il est ACCEPTE, et qu'un
  // traitement GENUINEMENT inconnu reste refuse tot.
  chk(!raises([&] {
    AmrSystem s(cfg);
    s.add_block("ne", exb_spec(), "none", "rusanov", "conservative", "imex", 1);
  }),
      "add_block accepte time='imex' (IMEX cable sur AMR)");
  chk(raises([&] {
        AmrSystem s(cfg);
        s.add_block("ne", exb_spec(), "none", "rusanov", "conservative", "time_bidon", 1);
      }),
      "add_block refuse un time hors {explicit, imex}");
  chk(raises([&] {
        AmrSystem s(cfg);
        s.add_block("ne", exb_spec(), "none", "rusanov", "recon_bidon", "explicit", 1);
      }),
      "add_block refuse un recon hors {conservative, primitive}");
  chk(raises([&] {
        AmrSystem s(cfg);
        s.add_block("ne", exb_spec(), "none", "rusanov", "conservative", "explicit", 0);
      }),
      "add_block refuse substeps < 1");

  // --- multi-blocs (capstone PR1) : un 2e bloc natif est desormais ACCEPTE -------------------
  // Bascule sur le moteur runtime AmrRuntime (hierarchie partagee, Poisson somme). On verifie que
  // l'ajout passe sans lever ; la physique (evolution, masse, Poisson somme) est verrouillee par
  // test_amr_system_twoblock.
  chk(!raises([&] {
    AmrSystemConfig c2 = cfg;
    c2.regrid_every = 0;  // multi-blocs PR1 : hierarchie FIGEE
    AmrSystem s(c2);
    s.add_block("ne", exb_spec(), "none", "rusanov", "conservative", "explicit", 1);
    s.add_block("ni", exb_spec(), "minmod", "rusanov", "conservative", "explicit", 1);
  }),
      "add_block accepte un second bloc (multi-blocs, hierarchie partagee)");

  // --- DEVERROUILLAGE (capstone Phase 2, C.6) : multi-blocs + regrid_every > 0 est ACCEPTE ----
  // L'ancien REFUS (la hierarchie multi-blocs etait FIGEE) est leve : AmrRuntime porte le regrid
  // d'union des tags (set_regrid + set_block_tag_predicate cables dans build_multi). ensure_built
  // (1er mass()) construit le moteur avec la cadence active au lieu de lever ; le regrid d'union et
  // le mouvement effectif de la hierarchie sont verrouilles par test_amr_multiblock_regrid_union.
  chk(!raises([&] {
    AmrSystemConfig c2 = cfg;
    c2.regrid_every = 5;  // > 0
    AmrSystem s(c2);
    s.add_block("ne", exb_spec(), "none", "rusanov", "conservative", "explicit", 1);
    s.add_block("ni", exb_spec(), "minmod", "rusanov", "conservative", "explicit", 1);
    (void)s.mass("ne");  // declenche ensure_built -> moteur multi-blocs avec regrid d'union actif
  }),
      "multi-blocs + regrid_every > 0 ACCEPTE (regrid d'union des tags, deverrouillage Phase 2)");

  // --- mono-bloc + regrid_every > 0 reste AUTORISE (chemin AmrCouplerMP, regrid intact) -------
  chk(!raises([&] {
    AmrSystemConfig c2 = cfg;
    c2.regrid_every = 5;
    AmrSystem s(c2);
    s.add_block("ne", exb_spec(), "none", "rusanov", "conservative", "explicit", 1);
    (void)s.mass();  // ensure_built : mono-bloc avec regrid, pas de refus
  }),
      "mono-bloc + regrid_every > 0 reste autorise (regrid AmrCouplerMP intact)");

  if (checker.fails() == 0)
    std::printf("OK test_amr_system_contract (refus explicite des parametres non cables)\n");
  else
    std::printf("FAIL test_amr_system_contract : %d echec(s)\n", checker.fails());
  return checker.failed();
}
