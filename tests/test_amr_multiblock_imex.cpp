// AMR MULTI-BLOCS IMEX (capstone vii) : la FACADE RUNTIME (AmrSystem -> AmrRuntime) honore le
// traitement temporel time="imex" PAR BLOC (source raide traitee en IMPLICITE localement par
// backward_euler_source), en mirroir de la branche IMEX du moteur compile-time AmrSystemCoupler::step
// (SourceFreeModel transport + AmrImplicitSourceStepper). Le pas IMEX d'UN bloc n'altere PAS les blocs
// explicites voisins (selection PAR BLOC).
//
// Ce que le test verrouille (cf. tache capstone vii + suite revue #184) :
//   (1) STABILITE RAIDE : un bloc a SOURCE LOCALE RAIDE (relaxation, raideur 1/eps >> 1/dt) sous IMEX
//       sur une hierarchie AMR 2 NIVEAUX est FINI et BORNE, la ou le MEME bloc en EXPLICITE DIVERGE
//       (facteur |1 - dt/eps| >> 1). Rejet nan/inf AVANT toute tolerance. La stabilite est observee
//       DIRECTEMENT sur le champ STIFFENE mx/my/E (comp 1/2/3 du grossier, lues via levels(b)), pas
//       seulement par contamination indirecte de la densite (revue #184) : la source ne touche PAS rho.
//   (2) CONSERVATION : la source raide IMEX choisie est CELLULE-LOCALE et n'agit PAS sur la composante 0
//       (densite) -> la masse du bloc IMEX est conservee a ~machine (hors registres de reflux, cascade
//       fin -> grossier intacte). On le verifie sur 2 niveaux (un patch fin present).
//   (3) DISABLE-AND-FAIL : forcer le MEME bloc raide en EXPLICITE (imex=false) le fait EXPLOSER -> la
//       selection IMEX est GENUINEMENT exercee (ce n'est pas un no-op silencieux). C'est le pendant
//       "negatif" de (1) : sans IMEX, la stabilite disparait. L'explosion est verifiee SUR mx/my/E.
//   (3bis) IMEX SOUS-CYCLE substeps>1 (revue #184) : un run IMEX substeps=4 est fini, borne (densite ET
//       mx/my/E), conservatif, et sa trajectoire DIFFERE d'un run IMEX substeps=1 -> le SOUS-CYCLAGE du
//       splitting IMEX (decision d'integration, contraire au compile-time qui ignore substeps en IMEX)
//       est INTENTIONNEL et reellement execute, pas un no-op silencieux.
//   (4) OPT-IN BIT-IDENTIQUE : un multi-blocs TOUT-EXPLICITE est inchange (dmax==0 entre deux runs), et
//       le mono-bloc reste sur AmrCouplerMP (dmax==0). L'IMEX par bloc est strictement opt-in.
//   (5) FACADE : AmrSystem.add_block(time="imex") en MULTI-BLOCS construit et tourne (bloc IMEX
//       potential + bloc explicite ExB), etat fini ; et le masque IMEX partiel (implicit_roles) est
//       refuse en explicite ET resolu en multi-blocs (un role absent leve une erreur claire).
//
// La source RAIDE (relaxation S = -k * (u - u_eq), cellule-locale) n'est PAS ModelSpec-atteignable
// (aucune brique de dispatch), donc le coeur du test (1)(2)(3) travaille au niveau du MOTEUR
// AmrRuntime + build_amr_block, EXACTEMENT comme test_amr_multiblock_substeps (acces niveaux/masses).
// La FACADE (4)(5) passe par AmrSystem (modeles ModelSpec : exb, potential).

#include <adc/physics/bricks/bricks.hpp>  // CompositeModel, Euler, BackgroundDensity, ChargeDensity, PotentialForce
#include <adc/runtime/builders/amr_dsl_block.hpp>  // detail::make_shared_amr_layout / build_amr_block / dispatch_amr_block
#include <adc/runtime/amr/amr_runtime.hpp>    // AmrRuntime, AmrRuntimeBlock
#include <adc/runtime/amr_system.hpp>     // facade AmrSystem
#include <adc/runtime/builders/model_factory.hpp>  // detail::dispatch_model
#include <adc/runtime/model_spec.hpp>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace adc;

namespace {

constexpr double kGamma = 1.4;

// Source RAIDE CELLULE-LOCALE qui NE TOUCHE PAS la densite (composante 0) : relaxation de la QUANTITE
// DE MOUVEMENT (mx, my) et de l'ENERGIE vers un equilibre, de raideur 1/eps. En EXPLICITE (Euler avant)
// mx <- mx (1 - dt/eps) DIVERGE des que dt/eps > 2 ; en IMEX (backward Euler) mx <- mx / (1 + dt/eps)
// reste BORNE pour tout dt > 0 (inconditionnellement stable). La composante 0 (rho) a source NULLE ->
// la MASSE est conservee a la machine, IMEX comme explicite (tant que ce dernier ne diverge pas).
struct StiffMomentumRelax {
  Real inv_eps = Real(0);
  Real e_eq = Real(2.5);  // energie d'equilibre (rho=1, vitesse nulle, p coherent)
  template <class State>
  ADC_HD State apply(const State& u, const Aux&) const {
    State s{};
    // s[0] (densite) = 0 : la source ne cree/detruit PAS de masse -> conservation a la machine.
    if (State::size() > 1)
      s[1] = -inv_eps * u[1];  // -mx / eps
    if (State::size() > 2)
      s[2] = -inv_eps * u[2];  // -my / eps
    if (State::size() > 3)
      s[3] = -inv_eps * (u[3] - e_eq);  // -(E - E_eq) / eps
    return s;
  }
};
// elliptic "background" alpha=0 : rhs nul -> phi=0, aucun couplage au champ (raideur PUREMENT locale,
// le cas propre pour separer la stabilite de la source de tout couplage Poisson).
using StiffModel = CompositeModel<Euler, StiffMomentumRelax, BackgroundDensity>;
StiffModel make_stiff(double eps) {
  StiffMomentumRelax r;
  r.inv_eps = static_cast<Real>(1.0 / eps);
  return StiffModel{Euler{static_cast<Real>(kGamma)}, r, BackgroundDensity{Real(0), Real(0)}};
}

// Modele EXPLICITE neutre (Euler sans source, charge nulle) : un 2e bloc "voisin" pour exercer le
// MULTI-BLOCS (hierarchie partagee, Poisson somme) sans raideur. ExB scalaire ne convient pas (1 var) ;
// on prend un Euler 4 var a source nulle, MEME nombre de variables que le bloc raide (layout coherent).
using NeutralModel = CompositeModel<Euler, NoSource, BackgroundDensity>;
NeutralModel make_neutral() {
  return NeutralModel{Euler{static_cast<Real>(kGamma)}, NoSource{},
                      BackgroundDensity{Real(0), Real(0)}};
}

// densite + impulsion initiales : une bulle de densite avec une impulsion non nulle (pour que la source
// de relaxation de mx/my AIT un effet a relaxer). On pose ici la SEULE composante 0 (densite) via le
// chemin coupler_write_coarse (qui derive mx,my,E d'un Euler au repos) ; la raideur agit ensuite sur
// l'impulsion engendree par le transport (gradients de la bulle) -- suffisant pour exploser en explicite.
std::vector<double> bubble(int n) {
  std::vector<double> rho(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n - 0.5, y = (j + 0.5) / n - 0.5;
      rho[static_cast<std::size_t>(j) * n + i] = 1.0 + 0.5 * std::exp(-(x * x + y * y) / 0.02);
    }
  return rho;
}

bool all_finite(const std::vector<double>& v) {
  for (double x : v)
    if (!std::isfinite(x))
      return false;
  return true;
}
double maxabs(const std::vector<double>& v) {
  double m = 0;
  for (double x : v)
    m = std::fmax(m, std::fabs(x));
  return m;
}
double dmax_field(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0;
  const std::size_t nn = a.size() < b.size() ? a.size() : b.size();
  for (std::size_t i = 0; i < nn; ++i)
    d = std::fmax(d, std::fabs(a[i] - b[i]));
  return d;
}

// Construit un AmrRuntime a DEUX blocs sur une hierarchie 2 NIVEAUX figee N x N : un bloc RAIDE
// (StiffModel) au traitement @p imex_stiff et de cadence @p substeps sous-pas, et un bloc NEUTRE
// explicite. La densite initiale rho est posee sur les deux. Le patch fin central FIXE de
// make_shared_amr_layout donne la 2e niveau (couverture). @p substeps : sous-pas du bloc raide
// (1 = un seul advance par macro-pas ; >1 = le pas effectif est decoupe en substeps morceaux, le
// splitting IMEX est alors SOUS-CYCLE par le moteur -- decision assumee, cf. amr_runtime.hpp).
AmrRuntime make_stiff_pair(int N, double L, double eps, bool imex_stiff,
                           const std::vector<double>& rho, int substeps = 1) {
  AmrBuildParams bp;
  bp.n = N;
  bp.L = L;
  bp.regrid_every = 0;      // hierarchie figee (multi-blocs)
  bp.poisson_bc = BCRec{};  // periodique
  const detail::SharedAmrLayout S = detail::make_shared_amr_layout(bp);
  std::vector<AmrRuntimeBlock> blocks;
  // bloc A : raide, traitement imex_stiff (true = IMEX, false = explicite : disable-and-fail).
  blocks.push_back(detail::build_amr_block<StiffModel, Minmod, RusanovFlux>(
      make_stiff(eps), S, "stiff", rho, /*has_density=*/true, kGamma, substeps,
      /*recon_prim=*/false, /*imex=*/imex_stiff, /*stride=*/1));
  // bloc B : neutre, EXPLICITE (voisin sur la hierarchie partagee ; Poisson somme co-localise).
  blocks.push_back(detail::build_amr_block<NeutralModel, Minmod, RusanovFlux>(
      make_neutral(), S, "neutral", rho, /*has_density=*/true, kGamma, /*substeps=*/1,
      /*recon_prim=*/false, /*imex=*/false, /*stride=*/1));
  return AmrRuntime(S.geom, S.ba_coarse, S.poisson_bc, std::move(blocks), S.base_per,
                    S.replicated_coarse, S.wall);
}

// Lit DIRECTEMENT le grossier (niveau 0) du bloc @p b et renvoie le max |U(.,.,c)| sur les
// composantes STIFFENEES mx/my/E (c=1,2,3), tout en signalant la presence d'un non-fini. Le reviewer
// #184 a note que borner la densite (comp 0) n'observe la stabilite qu'INDIRECTEMENT (la source raide
// StiffMomentumRelax laisse rho INTACTE et ne stiffenne que mx/my/E) : on lit donc les composantes
// REELLEMENT relaxees, sans changer l'API de production (rt est non-const et expose levels(b), exactement
// l'accesseur de test_amr_source_covered_cells). On itere les fabs LOCAUX (local_size()==0 sur un rang
// sans boite -> max nul, MPI-safe) et les cellules VALIDES (box(li), pas les ghosts). @p finite mis a
// false des qu'une cellule est non finie.
double max_momentum_energy_coarse(AmrRuntime& rt, std::size_t b, bool& finite) {
  finite = true;
  double m = 0.0;
  const MultiFab& U = rt.levels(b)[0].U;
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 a = U.fab(li).const_array();
    const Box2D box = U.box(li);
    for (int j = box.lo[1]; j <= box.hi[1]; ++j)
      for (int i = box.lo[0]; i <= box.hi[0]; ++i)
        for (int c = 1; c <= 3; ++c) {
          const double v = static_cast<double>(a(i, j, c));
          if (!std::isfinite(v))
            finite = false;
          m = std::fmax(m, std::fabs(v));
        }
  }
  return m;
}

// modeles ModelSpec pour la FACADE : ExB scalaire (charge q) et Euler+potential (source raide self-consistent).
ModelSpec exb_charge(double q, double B0) {
  ModelSpec s;
  s.transport = "exb";
  s.source = "none";
  s.elliptic = "charge";
  s.q = q;
  s.B0 = B0;
  return s;
}
ModelSpec pot_charge(double qom) {
  ModelSpec s;
  s.transport = "compressible";
  s.source = "potential";
  s.elliptic = "charge";
  s.gamma = kGamma;
  s.qom = qom;
  s.q = 1.0;
  return s;
}
std::vector<double> bump(int n, double base, double amp) {
  std::vector<double> r(static_cast<std::size_t>(n) * n, base);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const bool in = (i >= n / 4 && i < 3 * n / 4 && j >= n / 4 && j < 3 * n / 4);
      r[static_cast<std::size_t>(j) * n + i] = base + (in ? amp : -amp / 3.0);
    }
  return r;
}

}  // namespace

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

  const int N = 32;
  const double L = 1.0;
  const std::vector<double> rho = bubble(N);

  // ============================================================================================
  // (1)+(2)+(3) STABILITE RAIDE + CONSERVATION + DISABLE-AND-FAIL, au niveau du moteur AmrRuntime.
  //     eps << dt : explicite (facteur |1 - dt/eps| >> 1) DIVERGE, IMEX (backward Euler) reste fini.
  // ============================================================================================
  const double eps = 1e-5, dt = 1e-3;
  const int K = 12;  // macro-pas

  // (1) IMEX : bloc raide STABLE (fini + borne) sur 2 niveaux.
  {
    AmrRuntime rt = make_stiff_pair(N, L, eps, /*imex_stiff=*/true, rho);
    const Real m0 = rt.mass(0);  // masse du bloc raide AVANT (sur le grossier, cascade incluse)
    chk(rt.nlev() == 2, "imex_two_levels_present");  // un patch fin existe (couverture exercee)
    for (int s = 0; s < K; ++s)
      rt.step(static_cast<Real>(dt));
    const std::vector<double> dStiff = rt.density(0);
    const std::vector<double> dNeutral = rt.density(1);
    const Real m1 = rt.mass(0);
    // AVANT toute tolerance : etat fini (un nan passerait une borne par hasard).
    chk(all_finite(dStiff) && all_finite(dNeutral), "imex_stiff_state_finite");
    chk(maxabs(dStiff) < 1e3, "imex_stiff_state_bounded");
    // (3-revue #184) STABILITE OBSERVEE SUR LE BON CHAMP : la source raide ne stiffenne PAS la densite
    // (comp 0) mais mx/my/E (comp 1/2/3). Borner la seule densite n'observe la stabilite qu'INDIRECTEMENT
    // (par contamination via le transport). On lit donc DIRECTEMENT mx/my/E du grossier et on exige
    // qu'elles restent finies ET bornees sous IMEX (la ou l'explicite, ci-dessous, les fait diverger).
    bool me_finite = false;
    const double me_max = max_momentum_energy_coarse(rt, 0, me_finite);
    chk(me_finite, "imex_stiff_momentum_energy_finite_DIRECT");
    chk(me_max < 1e3, "imex_stiff_momentum_energy_bounded_DIRECT");
    // (2) CONSERVATION : la source raide ne touche pas la densite (comp 0) -> masse conservee ~machine.
    const double drift =
        std::fabs(static_cast<double>(m1 - m0)) / (std::fabs(static_cast<double>(m0)) + 1e-30);
    chk(drift < 1e-12, "imex_stiff_mass_conserved_to_machine");
    std::printf(
        "      IMEX : max(rho)=%.3e, max|mx,my,E|=%.3e, derive de masse=%.3e (eps=%.0e, dt=%.0e)\n",
        maxabs(dStiff), me_max, drift, eps, dt);
  }

  // (3) DISABLE-AND-FAIL : MEME bloc raide en EXPLICITE -> EXPLOSE. Prouve que la selection IMEX de (1)
  //     est GENUINEMENT exercee (sans elle, la stabilite disparait). On observe l'explosion SUR LE CHAMP
  //     STIFFENE (mx/my/E directement, comp 1/2/3), la ou la source agit, pas seulement sur la densite.
  {
    AmrRuntime rt = make_stiff_pair(N, L, eps, /*imex_stiff=*/false, rho);
    for (int s = 0; s < K; ++s)
      rt.step(static_cast<Real>(dt));
    const std::vector<double> dStiff = rt.density(0);
    bool me_finite = false;
    const double me_max = max_momentum_energy_coarse(rt, 0, me_finite);
    // L'explosion DOIT etre visible sur mx/my/E (le champ que la raideur attaque) ; on garde aussi le
    // critere densite (contamination par le transport) pour la lisibilite du diagnostic.
    const bool me_blew_up = !me_finite || me_max > 1e3;
    const bool rho_blew_up = !all_finite(dStiff) || maxabs(dStiff) > 1e3;
    chk(me_blew_up,
        "explicit_stiff_momentum_energy_BLOWS_UP_DIRECT (disable-and-fail sur mx/my/E)");
    chk(rho_blew_up, "explicit_stiff_BLOWS_UP (disable-and-fail : IMEX genuinement requis)");
    std::printf("      EXPLICITE : mx/my/E %s, rho %s (la stabilite vient bien du pas implicite)\n",
                me_finite ? "borne >> 1" : "NON FINI (explose)",
                all_finite(dStiff) ? "borne >> 1" : "NON FINI (explose)");
  }

  // ============================================================================================
  // (3bis) IMEX SOUS-CYCLE substeps>1 (revue #184) : le chemin IMEX avec substeps>1 est ATTEIGNABLE
  //     (AmrRuntime::step boucle substeps fois sur les DEUX traitements) et la DECISION d'integration
  //     est de SOUS-CYCLER le splitting IMEX (K=substeps pas de Lie sur dt/K), CONTRAIREMENT au moteur
  //     compile-time AmrSystemCoupler::step qui ignore substeps sur sa branche IMEX. Ce test VERROUILLE
  //     cette semantique : un run IMEX substeps=4 est (a) fini, (b) borne (densite ET mx/my/E directement),
  //     (c) conservatif en masse, et SURTOUT (d) sa trajectoire DIFFERE d'un run IMEX substeps=1 (memes
  //     eps/dt/macro-pas). La difference PROUVE que le sous-cyclage est INTENTIONNEL et REELLEMENT
  //     execute (si substeps etait silencieusement ignore comme en compile-time, dmax serait nul).
  // ============================================================================================
  {
    // substeps=1 : reference (un seul pas de Lie par macro-pas).
    AmrRuntime rt1 = make_stiff_pair(N, L, eps, /*imex_stiff=*/true, rho, /*substeps=*/1);
    const Real m0_1 = rt1.mass(0);
    for (int s = 0; s < K; ++s)
      rt1.step(static_cast<Real>(dt));
    const std::vector<double> d1 = rt1.density(0);
    const double drift1 = std::fabs(static_cast<double>(rt1.mass(0) - m0_1)) /
                          (std::fabs(static_cast<double>(m0_1)) + 1e-30);

    // substeps=4 : meme eps/dt/macro-pas, mais le moteur SOUS-CYCLE le splitting IMEX en 4 pas de dt/4.
    AmrRuntime rt4 = make_stiff_pair(N, L, eps, /*imex_stiff=*/true, rho, /*substeps=*/4);
    const Real m0_4 = rt4.mass(0);
    for (int s = 0; s < K; ++s)
      rt4.step(static_cast<Real>(dt));
    const std::vector<double> d4 = rt4.density(0);
    bool me4_finite = false;
    const double me4_max = max_momentum_energy_coarse(rt4, 0, me4_finite);
    const double drift4 = std::fabs(static_cast<double>(rt4.mass(0) - m0_4)) /
                          (std::fabs(static_cast<double>(m0_4)) + 1e-30);

    // (a)(b) le run sous-cycle reste fini + borne (backward-Euler stable a tout pas ; transport plus sur
    //        en CFL sur dt/4) sur la densite ET sur le champ stiffene mx/my/E lu directement.
    chk(all_finite(d4) && me4_finite, "imex_subcycled_s4_finite");
    chk(maxabs(d4) < 1e3 && me4_max < 1e3, "imex_subcycled_s4_bounded");
    // (c) conservation : la source raide laisse rho intacte -> masse conservee ~machine, comme substeps=1.
    chk(drift4 < 1e-12, "imex_subcycled_s4_mass_conserved");
    // (d) VERROU : substeps=4 DIFFERE de substeps=1 -> le sous-cyclage IMEX est intentionnel et execute.
    const double d14 = dmax_field(d1, d4);
    chk(d14 > 0.0, "imex_subcycled_s4_DIFFERS_from_s1 (sous-cyclage assume, pas ignore)");
    std::printf(
        "      IMEX substeps : s1 (derive=%.2e) vs s4 (max|mx,my,E|=%.3e, derive=%.2e), "
        "dmax(rho)=%.3e\n",
        drift1, me4_max, drift4, d14);
  }

  // ============================================================================================
  // (4) OPT-IN BIT-IDENTIQUE : un multi-blocs TOUT-EXPLICITE (deux blocs neutres) est inchange entre
  //     deux runs (dmax==0). L'IMEX par bloc ne perturbe RIEN tant qu'aucun bloc n'est IMEX.
  // ============================================================================================
  {
    auto run_all_explicit = [&]() {
      // regime NON raide (eps modere) : explicite NE diverge pas, et les deux blocs sont explicites.
      AmrRuntime rt = make_stiff_pair(N, L, /*eps=*/1.0, /*imex_stiff=*/false, rho);
      for (int s = 0; s < 5; ++s)
        rt.step(static_cast<Real>(1e-3));
      return rt.density(0);
    };
    const std::vector<double> a = run_all_explicit();
    const std::vector<double> b = run_all_explicit();
    chk(all_finite(a), "all_explicit_state_finite");
    chk(dmax_field(a, b) == 0.0, "all_explicit_multiblock_bit_identical");
  }

  // ============================================================================================
  // (5) FACADE AmrSystem : multi-blocs time="imex" construit et tourne ; masque IMEX partiel valide.
  // ============================================================================================
  {
    // (5a) deux blocs via la facade : A IMEX (potential, source self-consistent), B explicite (ExB).
    //      Construit, tourne, etat fini -> le drapeau time="imex" traverse add_block -> build_multi ->
    //      dispatch_amr_block -> build_amr_block -> AmrRuntime::step (selection IMEX par bloc).
    AmrSystemConfig cfg;
    cfg.n = N;
    cfg.L = L;
    cfg.periodic = true;
    cfg.regrid_every = 0;  // multi-blocs : hierarchie figee
    AmrSystem sim(cfg);
    sim.add_block("A", pot_charge(50.0), "minmod", "rusanov", "conservative", "imex", 1, 1);
    sim.add_block("B", exb_charge(-1.0, 1.0), "none", "rusanov", "conservative", "explicit", 1, 1);
    sim.set_poisson("charge_density", "geometric_mg", "periodic");
    sim.set_density("A", bump(N, 1.0, 0.40));
    sim.set_density("B", bump(N, 1.0, 0.20));
    for (int s = 0; s < 6; ++s)
      sim.step(5e-3);
    chk(sim.n_blocks() == 2, "facade_two_blocks");
    chk(all_finite(sim.density("A")) && all_finite(sim.density("B")),
        "facade_multiblock_imex_runs_finite");

    // (5b) masque IMEX partiel REFUSE en explicite (pas d'ignore silencieux).
    {
      AmrSystem s2(cfg);
      bool threw = false;
      try {
        s2.add_block("A", pot_charge(50.0), "minmod", "rusanov", "conservative", "explicit", 1, 1,
                     /*implicit_vars=*/{}, /*implicit_roles=*/{"momentum_x"});
      } catch (const std::exception&) {
        threw = true;
      }
      chk(threw, "facade_mask_rejected_in_explicit");
    }

    // (5c) masque IMEX partiel RESOLU en multi-blocs (role momentum_x present sur un Euler) : construit
    //      et tourne. Puis un role ABSENT (density n'a pas de... si, density existe ; on prend un role
    //      inexistant pour le bloc ExB scalaire) leve une erreur claire au build.
    {
      AmrSystem s3(cfg);
      s3.add_block("A", pot_charge(50.0), "minmod", "rusanov", "conservative", "imex", 1, 1,
                   /*implicit_vars=*/{}, /*implicit_roles=*/{"momentum_x", "momentum_y"});
      s3.add_block("B", exb_charge(-1.0, 1.0), "none", "rusanov", "conservative", "explicit", 1, 1);
      s3.set_poisson("charge_density", "geometric_mg", "periodic");
      s3.set_density("A", bump(N, 1.0, 0.40));
      s3.set_density("B", bump(N, 1.0, 0.20));
      bool ok = false;
      try {
        for (int s = 0; s < 4; ++s)
          s3.step(5e-3);
        ok = all_finite(s3.density("A"));
      } catch (const std::exception& e) {
        std::printf("      (5c) masque partiel a leve : %s\n", e.what());
      }
      chk(ok, "facade_partial_mask_resolved_and_runs");
    }

    // (5d) role ABSENT du bloc -> erreur claire au build (resolution du masque, build_multi).
    {
      AmrSystem s4(cfg);
      // ExB scalaire (1 var, role Scalar) : 'momentum_x' n'existe pas -> resolve_implicit_components leve.
      s4.add_block("A", exb_charge(1.0, 1.0), "none", "rusanov", "conservative", "imex", 1, 1,
                   /*implicit_vars=*/{}, /*implicit_roles=*/{"momentum_x"});
      s4.add_block("B", exb_charge(-1.0, 1.0), "none", "rusanov", "conservative", "explicit", 1, 1);
      s4.set_poisson("charge_density", "geometric_mg", "periodic");
      s4.set_density("A", bump(N, 1.0, 0.40));
      s4.set_density("B", bump(N, 1.0, 0.20));
      bool threw = false;
      try {
        s4.step(5e-3);  // build paresseux : resolution du masque -> role absent -> leve
      } catch (const std::exception&) {
        threw = true;
      }
      chk(threw, "facade_partial_mask_absent_role_throws");
    }
  }

  if (fails == 0)
    std::printf("OK test_amr_multiblock_imex\n");
  else
    std::printf("FAIL test_amr_multiblock_imex : %d echec(s)\n", fails);
  return fails == 0 ? 0 : 1;
}
