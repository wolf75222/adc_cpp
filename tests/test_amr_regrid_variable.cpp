// ADC-296 : critere de regrid AMR configurable par NOM de variable ou ROLE physique, au-dela de la
// seule composante 0. La couche moteur (TagPredicate per-bloc, union des tags) est deja generique ;
// seule la facade epinglait `a(i, j, 0) > seuil`. set_refinement(seuil, variable, role) resout, PAR
// BLOC, le selecteur en une composante (STRICT : un bloc sans le nom/role demande leve une erreur au
// build, jamais de repli silencieux vers la composante 0), defaut (selecteur vide) = composante 0
// bit-identique.
//
// On verrouille DEUX niveaux :
//   (1) le resolveur detail::resolve_selected_component (logique pure, deterministe), y compris le cas
//       d'acceptation cle "densite NON situee a la composante 0" ;
//   (2) la facade multi-blocs : refiner sur l'energie (composante 3 de l'Euler compressible) deplace le
//       patch fin vers la bosse d'energie, la ou le selecteur par defaut (densite uniforme) garde le
//       seed central -> les deux layouts DIFFERENT, preuve que la composante lue a change.
#include <adc/physics/euler.hpp>          // Euler::conservative_vars (rho, rho_u, rho_v, E)
#include <adc/core/variables.hpp>         // VariableSet, VariableRole, VariableKind
#include <adc/mesh/patch_box.hpp>         // PatchBox (signature index-espace des patchs fins)
#include <adc/runtime/amr_system.hpp>     // AmrSystem, AmrSystemConfig
#include <adc/runtime/builders/model_factory.hpp>  // detail::resolve_selected_component (ADC-296)
#include <adc/runtime/model_spec.hpp>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace adc;

template <class F>
static bool raises(F&& f) {
  try {
    f();
  } catch (const std::runtime_error&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

// Euler compressible pur (sans source), elliptique de fond trivial (alpha=0) : meme spec que
// test_amr_riemann_native / test_amr_weno5_native -> Poisson a second membre nul, aucune contrainte de
// solvabilite periodique a gerer (le regrid tague sur le champ conservatif, independant de phi).
static ModelSpec comp_spec() {
  ModelSpec s;
  s.transport = "compressible";
  s.source = "none";
  s.elliptic = "background";
  s.gamma = 1.4;
  s.alpha = 0.0;
  s.n0 = 0.0;
  return s;
}

// Etat conservatif (rho, rho_u, rho_v, E) component-major U[c*nn + j*n + i], au repos (mom=0), avec la
// composante @p bump_comp portee a @p bump_val dans la boite grossiere bas-gauche [lo, hi)^2 (en dehors
// du seed fin central [n/4, 3n/4)). Energie/pression positives au repos (u=0 -> E > 0 suffit).
static std::vector<double> make_state(int n, double rho, double E, int bump_comp, double bump_val,
                                      int lo, int hi) {
  const std::size_t nn = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  std::vector<double> U(4 * nn, 0.0);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const std::size_t k = static_cast<std::size_t>(j) * n + i;
      U[0 * nn + k] = rho;
      U[3 * nn + k] = E;
      if (i >= lo && i < hi && j >= lo && j < hi)
        U[static_cast<std::size_t>(bump_comp) * nn + k] = bump_val;
    }
  return U;
}

// Plus petit coin (min(ilo, jlo)) sur les boites FINES (level >= 1) : un patch suivant la bosse
// bas-gauche a un coin << au seed central (dont le coin fin vaut n/2). INT_MAX si aucune boite fine.
static int min_fine_corner(const std::vector<PatchBox>& boxes) {
  int m = INT_MAX;
  for (const auto& b : boxes)
    if (b.level >= 1)
      m = std::min(m, std::min(b.ilo, b.jlo));
  return m;
}

static bool same_boxes(const std::vector<PatchBox>& a, const std::vector<PatchBox>& b) {
  if (a.size() != b.size())
    return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].level != b[i].level || a[i].ilo != b[i].ilo || a[i].jlo != b[i].jlo ||
        a[i].ihi != b[i].ihi || a[i].jhi != b[i].jhi)
      return false;
  }
  return true;
}

// Monte un systeme multi-blocs compressible (regrid d'union actif), seed les deux blocs, avance et
// renvoie la signature des patchs fins. Bloc 0 porte l'etat d'interet ; bloc 1 est uniforme (jamais
// tague) : l'union est donc pilotee par la composante selectionnee du bloc 0.
static std::vector<PatchBox> run_case(int N, double thr, const std::string& variable,
                                      const std::string& role, const std::vector<double>& s0) {
  AmrSystemConfig cfg;
  cfg.n = N;
  cfg.L = 1.0;
  cfg.periodic = true;
  cfg.regrid_every = 1;  // regrid a chaque pas -> le patch suit le champ tague des le 1er pas
  AmrSystem sim(cfg);
  sim.add_block("gas0", comp_spec(), "minmod", "rusanov", "conservative", "explicit", 1);
  sim.add_block("gas1", comp_spec(), "minmod", "rusanov", "conservative", "explicit", 1);
  sim.set_poisson("charge_density", "geometric_mg", "periodic");
  sim.set_refinement(thr, variable, role);
  sim.set_conservative_state("gas0", s0);
  sim.set_conservative_state("gas1", make_state(N, 1.0, 2.0, 0, 1.0, 0, 0));  // uniforme
  for (int s = 0; s < 4; ++s)
    sim.step(1e-3);
  return sim.patch_boxes();
}

int main(int argc, char** argv) {
#if defined(ADC_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    std::printf("  [%s] %s\n", c ? "OK " : "XX ", w);
    if (!c)
      ++fails;
  };

  // ============================================================================================
  // (1) RESOLVEUR PUR detail::resolve_selected_component : nom XOR role -> composante, STRICT.
  // ============================================================================================
  {
    // Layout Euler canonique : rho(0) rho_u(1) rho_v(2) E(3).
    const VariableSet cv = Euler::conservative_vars();
    chk(detail::resolve_selected_component("set_refinement", "gas", cv, "", "") == -1,
        "resolver_empty_selector_means_default");
    chk(detail::resolve_selected_component("set_refinement", "gas", cv, "E", "") == 3,
        "resolver_name_E_is_comp3");
    chk(detail::resolve_selected_component("set_refinement", "gas", cv, "rho", "") == 0,
        "resolver_name_rho_is_comp0");
    chk(detail::resolve_selected_component("set_refinement", "gas", cv, "", "energy") == 3,
        "resolver_role_energy_is_comp3");
    chk(detail::resolve_selected_component("set_refinement", "gas", cv, "", "density") == 0,
        "resolver_role_density_is_comp0");
    chk(detail::resolve_selected_component("set_refinement", "gas", cv, "", "momentum_x") == 1,
        "resolver_role_momentum_x_is_comp1");
    chk(raises(
            [&] { detail::resolve_selected_component("set_refinement", "gas", cv, "bogus", ""); }),
        "resolver_unknown_name_throws");
    chk(raises([&] {
          detail::resolve_selected_component("set_refinement", "gas", cv, "", "temperature");
        }),
        "resolver_absent_role_throws");
    chk(raises([&] {
          detail::resolve_selected_component("set_refinement", "gas", cv, "E", "energy");
        }),
        "resolver_name_and_role_both_set_throws");

    // CAS D'ACCEPTATION CLE : densite NON situee a la composante 0. Le selecteur la retrouve par role
    // OU par nom, sans supposer l'index 0 (ce que l'ancien `a(i, j, 0)` ne pouvait pas).
    const VariableSet weird{VariableKind::Conservative,
                            {"phi_aux", "mx", "rho", "E"},
                            4,
                            {VariableRole::Scalar, VariableRole::MomentumX, VariableRole::Density,
                             VariableRole::Energy}};
    chk(detail::resolve_selected_component("set_refinement", "weird", weird, "", "density") == 2,
        "resolver_density_not_at_comp0_role");
    chk(detail::resolve_selected_component("set_refinement", "weird", weird, "rho", "") == 2,
        "resolver_density_not_at_comp0_name");
  }

  // ============================================================================================
  // (2) FACADE MULTI-BLOCS : refiner sur l'energie (comp 3) suit la bosse, le defaut (densite) non.
  // ============================================================================================
  const int N = 64;
  // Bosse d'ENERGIE en bas-gauche (boite grossiere [4, 20)^2, hors du seed central [16, 48)) ; densite
  // UNIFORME (=1). E base=2, bosse=12, seuil=6 : seule l'energie depasse, et seulement en bas-gauche.
  const std::vector<double> s_energy =
      make_state(N, 1.0, 2.0, /*bump_comp=*/3, /*bump_val=*/12.0, 4, 20);

  const std::vector<PatchBox> def =
      run_case(N, 6.0, "", "", s_energy);  // defaut comp 0 (densite=1<6)
  const std::vector<PatchBox> byrole = run_case(N, 6.0, "", "energy", s_energy);
  const std::vector<PatchBox> byname = run_case(N, 6.0, "E", "", s_energy);

  // Le defaut (densite uniforme < seuil) ne tague rien -> regrid no-op -> seed central conserve
  // (coin fin = n/2 = 32). Refiner sur l'energie deplace le patch vers la bosse bas-gauche (coin << 32).
  chk(min_fine_corner(byrole) < min_fine_corner(def), "role_energy_patch_reaches_lower_left");
  chk(min_fine_corner(byname) < min_fine_corner(def), "name_E_patch_reaches_lower_left");
  chk(!same_boxes(byrole, def), "role_energy_layout_differs_from_default");
  chk(!same_boxes(byname, def), "name_E_layout_differs_from_default");

  // NON-REGRESSION composante 0 : sur une bosse de DENSITE (comp 0) en bas-gauche, le selecteur par
  // defaut raffine bien la densite (le chemin historique reste fonctionnel).
  const std::vector<double> s_density =
      make_state(N, 1.0, 2.0, /*bump_comp=*/0, /*bump_val=*/3.0, 4, 20);
  const std::vector<PatchBox> dens_def = run_case(N, 2.0, "", "", s_density);
  chk(min_fine_corner(dens_def) < 32, "default_still_refines_on_density_comp0");

  // ============================================================================================
  // (3) ERREURS STRICTES : role absent du bloc -> erreur au build ; nom+role -> erreur immediate.
  // ============================================================================================
  chk(raises([&] { run_case(N, 6.0, "", "temperature", s_energy); }),
      "absent_role_throws_at_build_no_silent_comp0");
  // SINGLE-BLOCK : le selecteur n'est cable que sur le moteur multi-blocs ; un bloc unique + selecteur
  // est REFUSE au build (pas de repli silencieux vers la composante 0), comme le masque IMEX mono-bloc.
  chk(raises([&] {
        AmrSystemConfig cfg;
        cfg.n = N;
        cfg.L = 1.0;
        cfg.regrid_every = 1;
        AmrSystem sim(cfg);
        sim.add_block("solo", comp_spec(), "minmod", "rusanov", "conservative", "explicit", 1);
        sim.set_poisson("charge_density", "geometric_mg", "periodic");
        sim.set_refinement(6.0, "", "energy");  // bloc unique + role -> refus au build
        sim.set_density("solo", std::vector<double>(static_cast<std::size_t>(N) * N, 1.0));
        (void)sim.n_patches();  // declenche ensure_built -> refus
      }),
      "single_block_selector_rejected_at_build");
  chk(raises([&] {
        AmrSystemConfig cfg;
        cfg.n = N;
        cfg.L = 1.0;
        cfg.regrid_every = 1;
        AmrSystem sim(cfg);
        sim.add_block("gas0", comp_spec(), "minmod", "rusanov", "conservative", "explicit", 1);
        sim.add_block("gas1", comp_spec(), "minmod", "rusanov", "conservative", "explicit", 1);
        sim.set_refinement(6.0, "E", "energy");  // nom ET role -> ambiguite refusee
      }),
      "name_and_role_both_set_throws_at_facade");

  if (fails == 0)
    std::printf("OK test_amr_regrid_variable\n");
  else
    std::printf("FAIL test_amr_regrid_variable : %d echec(s)\n", fails);
  return fails == 0 ? 0 : 1;
}
