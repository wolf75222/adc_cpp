#pragma once

#include <adc/coupling/amr_condensed_schur_source_stepper.hpp>  // etage source condense GLOBAL (amr-schur)
#include <adc/coupling/amr_coupler_mp.hpp>  // AmrCouplerMP, AmrLevelMP
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/refinement.hpp>  // coarsen_index
#include <adc/numerics/numerical_flux.hpp>
#include <adc/numerics/reconstruction.hpp>
#include <adc/numerics/spatial_operator.hpp>  // SourceFreeModel (demi-pas explicite IMEX, transport seul)
#include <adc/numerics/time/implicit_stepper.hpp>  // backward_euler_source + ImplicitMask (source raide IMEX)
#include <adc/parallel/comm.hpp>  // n_ranks
#include <adc/runtime/amr_runtime.hpp>  // AmrRuntimeBlock (registre multi-blocs type-erase)
#include <adc/runtime/amr_system.hpp>
#include <adc/runtime/block_builder.hpp>  // detail::make_poisson_rhs (rhs += elliptic_rhs(U))
#include <adc/runtime/dispatch_tags.hpp>  // registry UNIQUE des tags (validate_limiter/riemann)

#include <algorithm>  // std::find, std::sort (resolution du masque IMEX partiel d'un bloc compile)
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/// @file
/// @brief add_compiled_model cote AmrSystem : branche un modele COMPILE (un CompositeModel, genere
///        par le DSL ou ecrit a la main, connu a la COMPILATION) comme un bloc d'une hierarchie AMR,
///        EXACTEMENT le chemin de production de AmrSystem::add_block mais SANS passer par la dispatch
///        ModelSpec (le modele est deja un type concret). UN seul bloc compile -> chemin AmrCouplerMP
///        mono-bloc historique (bit-identique) ; PLUSIEURS blocs compiles ou MELANGE compile + natif
///        (capstone v, DSL production multi-bloc) -> moteur runtime AmrRuntime sur la hierarchie
///        partagee, le bloc compile y etant materialise comme un AmrRuntimeBlock type-erase.
///
/// Pendant raffine de add_compiled_model(System&, ...) (dsl_block.hpp). La machinerie de build du
/// coupleur AMR (AmrCouplerMP<Model> + reflux conservatif + regrid) est instanciee ICI, depuis l'unite
/// de traduction APPELANTE, sur le type Model concret -- comme block_builder.hpp pour le System plat.
/// Les fermetures type-erasees entrent dans AmrSystem par AmrSystem::set_compiled_block (methode
/// non-template) qui fige DEUX builders : le mono-bloc (detail::build_amr_compiled / dispatch_amr_compiled,
/// PARTAGE avec le chemin ModelSpec natif d'add_block une fois le type resolu par detail::dispatch_model)
/// ET le multi-blocs (detail::dispatch_amr_block, PARTAGE lui aussi avec add_block en multi-blocs natif).

namespace adc {

/// Paquet (limiteur, flux Riemann) attendu par AmrCouplerMP::step<Disc>. Unique definition : le
/// chemin natif d'amr_system.cpp passe par ce meme header (plus de DiscLF duplique cote .cpp).
template <class L, class F>
struct AmrDiscLF {
  using Limiter = L;
  using NumericalFlux = F;
};

namespace detail {

/// Remplit le champ B_z GROSSIER (composante 0, n*n row-major en indices GLOBAUX) depuis @p field.
/// Pendant scalaire de coupler_write_coarse (parcours de boites identique, mono-box replique ET
/// multi-box reparti) : B_z est exige par l'etage source condense par Schur (terme de Lorentz).
inline void amr_write_coarse_bz(MultiFab& bz, const std::vector<double>& field, int n) {
  if (static_cast<int>(field.size()) != n * n)
    throw std::runtime_error(
        "AMR amr-schur : champ B_z de taille != n*n (appeler set_magnetic_field avant le 1er pas)");
  device_fence();
  for (int li = 0; li < bz.local_size(); ++li) {
    Array4 b = bz.fab(li).array();
    const Box2D v = bz.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        b(i, j, 0) = field[static_cast<std::size_t>(j) * n + i];
  }
}

/// Un ETAGE SOURCE condense GLOBAL sur la hierarchie du coupleur mono-bloc. Seed le warm-start phi^n
/// (= aux0 composante 0, soit le solve Poisson grossier du dernier update()), puis joue l'etage
/// condense (AmrCondensedSchurSourceStepper) qui assemble/resout son PROPRE operateur condense sur le
/// grossier et reconstruit la vitesse (rho gelee, mom/E mis a jour). En mono-niveau (aucun patch fin)
/// c'est bit-pour-bit l'etage uniforme #126.
template <class Coupler>
void amr_schur_source(Coupler& cpl, AmrCondensedSchurSourceStepper& schur, MultiFab& bz_coarse,
                      MultiFab& phi_coarse, double theta, double dt) {
  device_fence();
  for (int li = 0; li < phi_coarse.local_size(); ++li)
    for_each_cell(phi_coarse.box(li), CopyComp0Kernel{phi_coarse.fab(li).array(),
                                                      cpl.aux0().fab(li).const_array()});
  schur.step(cpl.levels(), phi_coarse, bz_coarse, /*c_bz=*/0, static_cast<Real>(theta),
             static_cast<Real>(dt));
}

/// Construit le coupleur AMR pour un Model compose + (Limiter, Flux) concrets et remplit les hooks
/// type-erased. Deux niveaux : grossier + un patch fin seed central, remodele par le regrid. C'est le
/// pendant header de AmrSystem::Impl::build, instancie depuis la TU appelante sur le type Model. Les
/// helpers de grossier (layout, write/read/inject) sont PARTAGES avec le chemin natif via
/// amr_coupler_mp.hpp (detail::coupler_*), donc replique et reparti suivent exactement la meme logique.
template <class Model, class Limiter, class Flux>
AmrCompiledHooks build_amr_compiled(const Model& model, const AmrBuildParams& bp) {
  using Coupler = AmrCouplerMP<Model>;
  const int nc = Model::n_vars;
  const Geometry g{Box2D::from_extents(bp.n, bp.n), 0.0, bp.L, 0.0, bp.L};
  const double dxc = bp.L / bp.n, dxf = dxc / 2;
  // Niveau 0 (grossier) : layout decide par la politique d'ownership (replique mono-box par defaut,
  // multi-box reparti si bp.distribute_coarse). En replique, dmap = my_rank() partout (la box vit
  // sur chaque rang ; un round-robin la poserait sur le seul rang 0 -> fab hors bornes ailleurs,
  // segfault sous np>1). Le seed fin part sur la MEME dmap que le grossier ; le regrid initial le
  // RECONSTRUIT puis le REPARTIT round-robin (DistributionMapping(nfine, n_ranks())) -> distribution
  // multi-GPU des patchs fins. En reparti, le grossier se distribue AUSSI (strong-scaling AMR).
  const auto [bac, dm] =
      coupler_make_coarse_layout(bp.n, bp.distribute_coarse, bp.coarse_max_grid);
  const int ng = Limiter::n_ghost;  // stencil du limiteur (1 NoSlope, 2 MUSCL) : parite du schema
  MultiFab Uc(bac, dm, nc, ng);
  Uc.set_val(Real(0));
  std::vector<AmrLevelMP> levels;
  levels.push_back({std::move(Uc), nullptr, dxc, dxc});
  // Niveau 1 (patch fin seed central, remodele par le regrid) : chemin explicit/imex SEULEMENT. Le
  // chemin amr-schur (Etape 2/3) tourne MONO-NIVEAU -- l'etage source condense ne porte pas encore le
  // multi-niveau (cf. AmrCondensedSchurSourceStepper, garde sur le nombre de patchs fins). On n'alloue
  // donc PAS de niveau fin pour bp.schur (sinon le seed declencherait cette garde des le 1er pas). Le
  // multi-niveau amr-schur (reconstruction fine + cascade + Schur/Poisson composite) est l'Etape 4.
  if (!bp.schur) {
    const int I0 = bp.n / 4, I1 = 3 * bp.n / 4 - 1, J0 = bp.n / 4, J1 = 3 * bp.n / 4 - 1;
    Box2D fb{{2 * I0, 2 * J0}, {2 * I1 + 1, 2 * J1 + 1}};
    BoxArray baf(std::vector<Box2D>{fb});
    MultiFab Uf(baf, dm, nc, ng);
    Uf.set_val(Real(0));
    levels.push_back({std::move(Uf), nullptr, dxf, dxf});
  }

  auto cpl = std::make_shared<Coupler>(model, g, bac, bp.poisson_bc, std::move(levels), bp.wall,
                                       !bp.distribute_coarse);
  // Seed du grossier : etat conservatif COMPLET (prioritaire, set_conservative_state) sinon densite
  // seule (historique). coupler_inject_coarse_to_fine_mb prolonge TOUTES les composantes (boucle
  // k<nc), donc la qty de mouvement du seed se propage librement aux niveaux fins -- aucun changement
  // de prolongation. has_state==false -> chemin densite bit-identique (NO-DEFAULT-CHANGE).
  if (bp.has_state)
    coupler_write_coarse_state(cpl->coarse(), bp.state, bp.n, nc);
  else if (bp.has_density)
    coupler_write_coarse(cpl->coarse(), bp.density, bp.n, nc, bp.gamma);
  auto& Lv = cpl->levels();
  for (std::size_t k = 1; k < Lv.size(); ++k)
    coupler_inject_coarse_to_fine_mb(cpl->coarse(), Lv[k].U, !bp.distribute_coarse);

  const double thr = bp.refine_threshold;
  auto crit = [thr](const ConstArray4& a, int i, int j) { return a(i, j, 0) > thr; };
  if (cpl->levels().size() > 1) cpl->regrid(crit);  // pas de regrid sur une hierarchie mono-niveau (amr-schur)
  cpl->update();

  AmrCompiledHooks h;
  h.coupler_holder = cpl;  // duree de vie : les fermetures capturent cpl (shared_ptr)
  const int sub = bp.substeps;
  const bool rprim = bp.recon_prim;
  const bool imex = bp.imex;  // source raide implicite (backward_euler) plutot qu'Euler avant
  const int regrid_every = bp.regrid_every;
  // OPTIONS NEWTON de la source IMEX mono-bloc (vague 3) : threadees a cpl->step -> advance_amr ->
  // backward_euler_source. DEFAUT {} (newton_options non pose) = constantes historiques (2 iters) ->
  // chemin (2a) bit-identique. Capturees PAR VALEUR (POD) dans la fermeture h.step.
  const NewtonOptions nopts = bp.newton_options;
  // METHODE TEMPORELLE mono-bloc : entier de l'ABI plate (bp.time_method) -> AmrTimeMethod, threade a
  // cpl->step -> advance_amr. 0 (defaut / loader .so anterieur) = kEuler historique, bit-identique.
  const AmrTimeMethod tmethod =
      bp.time_method == 1 ? AmrTimeMethod::kSsprk3 : AmrTimeMethod::kEuler;
  auto step_state = std::make_shared<int>(0);  // compteur de pas partage par la fermeture
  if (bp.schur) {
    // CHEMIN amr-schur : etage source condense GLOBAL (electrostatique/Lorentz) au lieu de la source
    // explicit/imex LOCALE. L'etage est construit sur le GROSSIER en COMPOSANT l'etage uniforme #126
    // (roles Density/MomentumX/MomentumY du Model -> erreur claire ICI si absents). B_z grossier exige
    // (set_magnetic_field). Le modele doit etre SOURCE-FREE (brique source NoSource) : advance_transport
    // ne joue alors AUCUNE source (la source est l'etage condense seul), miroir du chemin uniforme ou le
    // bloc est ajoute avec son seul transport (time.hyperbolic) + l'etage source condense separe.
    //
    // ATTENTION : OPTION A = INTERMEDIAIRE. L'etage condense resout l'elliptique sur le GROSSIER (comme le Poisson
    // AMR compute_aux/solve_fields), puis le grad phi est injecte (constant par morceaux) aux fins : les
    // patchs fins raffinent le TRANSPORT mais PAS le couplage elliptique. Pour une reproduction papier/AMR
    // FIDELE il faudra un Schur/Poisson COMPOSITE multi-niveau (elliptique condense resolu a la finesse
    // des patchs, MG composite croisant les niveaux) -- infrastructure absente aujourd'hui (GeometricMG
    // coarsen UNE grille, != hierarchie AMR). C'est le verrou de fidelite, a faire APRES la parite mono-niveau.
    // RESOLUTION des descripteurs de champs (audit vague 3, parite System::set_source_stage) :
    // "" = role canonique (historique, bit-identique) ; sinon nom de ROLE stable puis nom de
    // VARIABLE du bloc. Echec = erreur explicite au build (jamais d'ignore silencieux).
    const VariableSet schur_vs = Model::conservative_vars();
    auto resolve_schur = [&schur_vs](const std::string& spec, VariableRole canonical,
                                     const char* label) -> int {
      if (spec.empty()) {
        const int idx = schur_vs.index_of(canonical);
        if (idx < 0)
          throw std::runtime_error(std::string("AmrSystem::set_source_stage : le bloc n'expose pas "
                                               "le role ") + label +
                                   " (declarer les roles, ou passer un descripteur explicite)");
        return idx;
      }
      const VariableRole r = role_from_name(spec);
      if (r != VariableRole::Custom) {
        const int idx = schur_vs.index_of(r);
        if (idx < 0)
          throw std::runtime_error("AmrSystem::set_source_stage : role '" + spec + "' absent (" +
                                   label + ")");
        return idx;
      }
      for (std::size_t i = 0; i < schur_vs.names.size(); ++i)
        if (schur_vs.names[i] == spec) return static_cast<int>(i);
      throw std::runtime_error("AmrSystem::set_source_stage : '" + spec +
                               "' n'est ni un role stable ni une variable du bloc (" + label + ")");
    };
    const int sc_rho = resolve_schur(bp.schur_density, VariableRole::Density, "Density");
    const int sc_mx = resolve_schur(bp.schur_momentum_x, VariableRole::MomentumX, "MomentumX");
    const int sc_my = resolve_schur(bp.schur_momentum_y, VariableRole::MomentumY, "MomentumY");
    const int sc_E = (bp.schur_energy == "none")
                         ? -1
                         : (bp.schur_energy.empty()
                                ? schur_vs.index_of(VariableRole::Energy)
                                : resolve_schur(bp.schur_energy, VariableRole::Energy, "Energy"));
    auto schur = std::make_shared<AmrCondensedSchurSourceStepper>(
        schur_vs, sc_rho, sc_mx, sc_my, sc_E, g, bac, bp.poisson_bc,
        static_cast<Real>(bp.schur_alpha));
    if (bp.schur_krylov_tol > 0.0 || bp.schur_krylov_max_iters > 0)
      schur->set_krylov(bp.schur_krylov_tol > 0.0 ? static_cast<Real>(bp.schur_krylov_tol)
                                                  : Real(1e-10),
                        bp.schur_krylov_max_iters > 0 ? bp.schur_krylov_max_iters : 400);
    auto bz_coarse = std::make_shared<MultiFab>(bac, dm, 1, 1);
    amr_write_coarse_bz(*bz_coarse, bp.bz_field, bp.n);
    auto phi_coarse = std::make_shared<MultiFab>(bac, dm, 1, 1);
    phi_coarse->set_val(Real(0));
    const double theta = bp.schur_theta;
    const bool strang = bp.schur_strang;
    h.step = [cpl, crit, sub, rprim, regrid_every, step_state, schur, bz_coarse, phi_coarse, theta,
              strang](double dt) {
      // amr-schur Etape 2/3 : hierarchie MONO-NIVEAU (l'etage condense ne porte pas le multi-niveau).
      // On ne regrille donc PAS (un regrid creerait un patch fin -> garde multi-niveau de l'etage). Le
      // regrid amr-schur viendra avec le Schur/Poisson composite (Etape 4). cf. levels().size() > 1.
      if (regrid_every > 0 && *step_state > 0 && *step_state % regrid_every == 0 &&
          cpl->levels().size() > 1)
        cpl->regrid(crit);
      const double h2 = dt / sub;
      for (int s = 0; s < sub; ++s) {
        if (strang) {
          // STRANG (2e ordre) : H(dt/2) ; S(dt) ; H(dt/2), avec update() (= sync_down + Poisson
          // grossier + grad inject, le pendant AMR de solve_fields) RE-RESOLU AVANT chaque etage qui
          // consomme phi -- exactement SystemStepper::step_strang (3 solves : tete, pre-source, post-source).
          cpl->update();
          cpl->template advance_transport<AmrDiscLF<Limiter, Flux>>(Real(0.5) * h2, rprim);
          cpl->update();
          amr_schur_source(*cpl, *schur, *bz_coarse, *phi_coarse, theta, h2);
          cpl->update();
          cpl->template advance_transport<AmrDiscLF<Limiter, Flux>>(Real(0.5) * h2, rprim);
        } else {
          // LIE (Godunov, 1er ordre) : H(dt) ; S(dt). Un seul update() en tete (l'etage source lit le
          // phi de tete), miroir de SystemStepper::step Lie (un seul solve_fields, transport, source).
          cpl->update();
          cpl->template advance_transport<AmrDiscLF<Limiter, Flux>>(h2, rprim);
          amr_schur_source(*cpl, *schur, *bz_coarse, *phi_coarse, theta, h2);
        }
      }
      ++*step_state;
    };
  } else {
    h.step = [cpl, crit, sub, rprim, imex, regrid_every, step_state, nopts, tmethod](double dt) {
      if (regrid_every > 0 && *step_state > 0 && *step_state % regrid_every == 0) cpl->regrid(crit);
      const double h2 = dt / sub;
      // OPTIONS NEWTON threadees au coupleur (mono-bloc) : nopts={} par defaut => iters=2 historique,
      // bit-identique ; nopts non-defaut (set_density + adc.IMEX(newton_*)) pilote le Newton local.
      // tmethod (kEuler defaut) selectionne SSPRK3 si demande (time='ssprk3') ; kEuler bit-identique.
      for (int s = 0; s < sub; ++s)
        cpl->template step<AmrDiscLF<Limiter, Flux>>(h2, rprim, imex, nopts, tmethod);
      ++*step_state;
    };
  }
  // RESTAURATION de la PHASE DE CADENCE (IO v1, parite System::set_clock) : AmrSystem::set_clock pose
  // le compteur de macro-pas du mono-bloc (la cadence regrid lit *step_state) au restart. Partage le
  // MEME step_state que la fermeture step ci-dessus -> la phase regrid reprend exactement. Sans appel,
  // *step_state reste a 0 (defaut, bit-identique).
  h.set_macro_step = [step_state](int s) { *step_state = s; };
  // VITESSE DE CFL : lambda* (trait HasStabilitySpeed) si declare, sinon max_wave_speed du coupleur
  // (fallback historique bit-identique) -- MEME politique que System/make_max_speed, evaluee sur le
  // GROSSIER (la CFL mono-bloc AMR vit au pas grossier).
  if constexpr (HasStabilitySpeed<Model>) {
    h.max_speed = [cpl, model] {
      return static_cast<double>(max_stability_speed_mf(model, cpl->coarse(), cpl->aux0()));
    };
  } else {
    h.max_speed = [cpl] { return static_cast<double>(cpl->max_wave_speed()); };
  }
  // BORNES DE PAS OPTIONNELLES (StabilityPolicy AMR mono-bloc) : memes reductions que System,
  // hooks laisses VIDES sans trait (AmrSystem::step_cfl garde alors la formule historique).
  if constexpr (HasSourceFrequency<Model>) {
    h.source_frequency = [cpl, model] {
      return static_cast<double>(max_source_frequency_mf(model, cpl->coarse(), cpl->aux0()));
    };
  }
  if constexpr (HasStabilityDt<Model>) {
    h.stability_dt = [cpl, model] {
      return static_cast<double>(min_stability_dt_mf(model, cpl->coarse(), cpl->aux0()));
    };
  }
  h.mass = [cpl] { return static_cast<double>(cpl->mass()); };
  h.n_patches = [cpl] {
    auto& L = cpl->levels();
    return L.size() >= 2 ? static_cast<int>(L[1].U.box_array().size()) : 0;
  };
  // Empreintes index-space des patchs fins (pendant mono-bloc de AmrRuntime::patch_boxes). Capture le
  // MEME cpl que les autres hooks (aucun nouveau souci de duree de vie), lit le BoxArray deja
  // materialise -> query entre les pas, zero cout chemin chaud (h.step intouche).
  h.patch_boxes = [cpl] {
    auto& L = cpl->levels();
    std::vector<adc::PatchBox> out;
    for (std::size_t k = 1; k < L.size(); ++k) {
      const auto& bxs = L[k].U.box_array().boxes();
      for (const adc::Box2D& b : bxs)
        out.push_back(adc::PatchBox{static_cast<int>(k), b.lo[0], b.lo[1], b.hi[0], b.hi[1]});
    }
    return out;
  };
  // CHECKPOINT / RESTART AMR mono-rang (ADC-65) : etat conservatif COMPLET par niveau + phi
  // (warm-start) + imposition de la hierarchie fine sauvee. Capturent le MEME cpl (shared_ptr) que
  // les autres hooks (aucun nouveau souci de duree de vie). Mono-rang : les accesseurs du coupleur
  // bouclent sur local_size() (pas de gather) -- la facade rejette np>1 / multi-blocs en amont. Ces
  // hooks sont des QUERIES/SETTERS entre les pas : zero cout chemin chaud (h.step intouche).
  h.n_levels = [cpl] { return cpl->nlev(); };
  h.n_vars = [] { return Model::n_vars; };
  h.level_state = [cpl](int k) { return cpl->level_state(k); };
  h.set_level_state = [cpl](int k, const std::vector<double>& s) { cpl->set_level_state(k, s); };
  h.level_potential = [cpl](int k) { return cpl->level_potential(k); };
  h.set_level_potential = [cpl](int k, const std::vector<double>& p) {
    cpl->set_level_potential(k, p);
  };
  h.set_hierarchy = [cpl](const std::vector<adc::PatchBox>& boxes) {
    // Mono-bloc : tous les patchs vivent au niveau 1 -> on filtre level == 1 et on convertit en Box2D
    // (coins INCLUSIFS, espace d'indices du niveau fin), puis on impose ce BoxArray au coupleur.
    std::vector<adc::Box2D> fb;
    for (const adc::PatchBox& b : boxes)
      if (b.level == 1) fb.push_back(adc::Box2D{{b.ilo, b.jlo}, {b.ihi, b.jhi}});
    cpl->set_hierarchy(fb);
  };
  const int nn = bp.n;
  const bool repl = !bp.distribute_coarse;
  h.density = [cpl, nn, repl] { return coupler_read_coarse(cpl->coarse(), nn, repl); };
  // phi du grossier : on rafraichit (update() = sync_down + compute_aux, donc solve Poisson grossier)
  // puis on lit aux0 composante 0. Pendant de System::potential() qui appelle ensure_elliptic : la
  // valeur est courante meme si aucun step n'a encore tourne. update() est deja appele a chaque step,
  // donc le surcout n'existe que sur un appel hors boucle (diagnostic).
  h.potential = [cpl, nn, repl] {
    cpl->update();
    return coupler_read_coarse_phi(cpl->aux0(), nn, repl);
  };
  return h;
}

/// Layout PARTAGE d'une hierarchie AMR multi-blocs (PR1 capstone), fige a la construction. Tous les
/// blocs allouent leurs niveaux sur EXACTEMENT ce layout (meme BoxArray + DistributionMapping +
/// dx/dy par niveau) -> same_layout_or_throw passe par construction. PR1 : grossier + UN patch fin
/// central FIXE (la placement par union des tags est une PR ulterieure). On expose les BoxArrays /
/// dmaps / dx/dy par niveau, le grossier (Geometry + ba) pour le Poisson, et la politique
/// d'ownership. build_amr_block alloue le bloc dessus.
struct SharedAmrLayout {
  Geometry geom;                       // geometrie du niveau grossier (Poisson)
  BoxArray ba_coarse;                  // BoxArray du grossier
  DistributionMapping dm_coarse;       // DistributionMapping du grossier
  std::vector<BoxArray> ba;            // [niveau] BoxArray partage (grossier + fins)
  std::vector<DistributionMapping> dm; // [niveau] DistributionMapping partage
  std::vector<Real> dx, dy;            // [niveau] pas d'espace
  bool replicated_coarse = true;       // ownership du niveau 0
  BCRec poisson_bc;                    // CL du Poisson grossier
  std::function<bool(Real, Real)> wall;// predicat paroi conductrice (vide = aucune)
  int n = 128;                         // cellules du grossier par direction
  Periodicity base_per{true, true};    // periodicite du domaine de base

  int nlev() const { return static_cast<int>(ba.size()); }
};

/// Construit le layout PARTAGE (PR1) : grossier (selon la politique d'ownership) + UN patch fin
/// central FIXE (le seed de build_amr_compiled, AVANT son regrid). Identique a la geometrie du
/// chemin mono-bloc, mais SANS regrid initial (multi-blocs PR1 = hierarchie figee). Tous les blocs
/// s'y posent ensuite via build_amr_block.
inline SharedAmrLayout make_shared_amr_layout(const AmrBuildParams& bp) {
  SharedAmrLayout S;
  S.geom = Geometry{Box2D::from_extents(bp.n, bp.n), 0.0, bp.L, 0.0, bp.L};
  S.n = bp.n;
  S.replicated_coarse = !bp.distribute_coarse;
  S.poisson_bc = bp.poisson_bc;
  S.wall = bp.wall;
  const double dxc = bp.L / bp.n, dxf = dxc / 2;
  const auto [bac, dmc] =
      detail::coupler_make_coarse_layout(bp.n, bp.distribute_coarse, bp.coarse_max_grid);
  S.ba_coarse = bac;
  S.dm_coarse = dmc;
  // Patch fin central FIXE : memes empreintes que build_amr_compiled (cellules grossieres
  // [n/4 .. 3n/4-1]^2, raffinees x2). REPARTI round-robin DistributionMapping(nfine, n_ranks()),
  // EXACTEMENT comme le regrid du chemin mono-bloc (amr_regrid_finest) : les patchs fins se
  // distribuent sur les rangs (un par GPU). C'est INDISPENSABLE sous MPI : sur le grossier REPLIQUE,
  // si le fin etait pose sur la meme dmap repliquee ({my_rank()}), CHAQUE rang detiendrait une copie
  // de la box fine et le reflux (all_reduce_sum_inplace des registres de flux) sommerait la MEME
  // contribution n_ranks() fois -> masse sur-comptee (croit avec np). En serie (np=1) la dmap
  // round-robin pose la box sur le rang 0, identique a {my_rank()} : bit-identique.
  const int I0 = bp.n / 4, I1 = 3 * bp.n / 4 - 1, J0 = bp.n / 4, J1 = 3 * bp.n / 4 - 1;
  const Box2D fb{{2 * I0, 2 * J0}, {2 * I1 + 1, 2 * J1 + 1}};
  BoxArray baf(std::vector<Box2D>{fb});
  DistributionMapping dmf(baf.size(), n_ranks());  // fin reparti round-robin (un patch par rang)
  S.ba = {bac, baf};
  S.dm = {dmc, dmf};
  S.dx = {dxc, dxf};
  S.dy = {dxc, dxf};
  return S;
}

/// Construit UN bloc AMR type-erase (AmrRuntimeBlock) sur le layout PARTAGE @p S, pour un Model
/// compose + (Limiter, Flux) concrets. Pendant multi-blocs de build_amr_compiled : alloue la pile de
/// niveaux du bloc sur le MEME BoxArray/dmap que tous les autres (garantit same_layout_or_throw),
/// pose la densite initiale (composante 0) + injection coarse->fine, et CAPTURE le schema concret
/// dans les fermetures (advance via advance_amr<Limiter, Flux>, add_elliptic_rhs via PoissonRhs).
/// Le noyau reste COMPILE ; seule la liste de blocs est type-erasee (calque AMR de make_block /
/// PoissonRhs cote System plat). @p density (vide = grossier a zero), @p substeps sous-pas du bloc,
/// @p stride cadence hold-then-catch-up du bloc (1 = chaque macro-pas). substeps et stride sont
/// portes par AmrRuntime::step (la fermeture advance ne fait qu'UN advance_amr) : ils ne touchent
/// donc PAS la capture du schema, juste les champs substeps/stride de l'AmrRuntimeBlock.
///
/// TRAITEMENT TEMPOREL (capstone vii) : @p imex selectionne le traitement de la SOURCE. On peuple
/// DEUX fermetures distinctes posees sur l'AmrRuntimeBlock et AmrRuntime::step choisit (b.imex) :
///   - advance : transport AMR + source EXPLICITE (Euler avant) -- chemin historique inchange ;
///   - imex_advance : transport AMR SOURCE-FREE + source raide IMPLICITE backward_euler_source par
///     niveau (masque @p implicit_components pour l'IMEX partiel) + cascade. La SEMANTIQUE du splitting
///     calque la branche IMEX de AmrSystemCoupler::step (SourceFreeModel + AmrImplicitSourceStepper), et
///     A substeps=1 lui est IDENTIQUE. Cette fermeture fait UN pas de Lie ; AmrRuntime::step l'appelle
///     substeps fois (sur le pas effectif / substeps), donc pour substeps>1 le runtime SOUS-CYCLE le
///     splitting IMEX la ou le compile-time l'applique une seule fois sur le pas effectif. Divergence
///     ASSUMEE et saine (cf. SEMANTIQUE IMEX SOUS substeps dans amr_runtime.hpp).
/// @p implicit_components : indices des composantes traitees en IMPLICITE (IMEX partiel, porte par le
/// BLOC, prioritaire sur le defaut modele) ; VIDE (defaut) -> masque inactif -> backward-Euler plein
/// (toutes les composantes implicites), comportement bit-identique a l'IMEX sans masque. Ignore si imex==false.
template <class Model, class Limiter, class Flux>
AmrRuntimeBlock build_amr_block(const Model& model, const SharedAmrLayout& S,
                                const std::string& name, const std::vector<double>& density,
                                bool has_density, double gamma, int substeps, bool recon_prim,
                                bool imex, int stride = 1,
                                const std::vector<int>& implicit_components = {},
                                const NewtonOptions& nopts = {},
                                const std::vector<double>* state = nullptr,
                                bool newton_diagnostics = false,
                                AmrTimeMethod time_method = AmrTimeMethod::kEuler) {
  const int nc = Model::n_vars;
  const int ng = Limiter::n_ghost;  // stencil du limiteur (parite du schema, comme build_amr_compiled)
  const int nlev = S.nlev();
  auto levels = std::make_shared<std::vector<AmrLevelMP>>();
  levels->reserve(nlev);
  for (int k = 0; k < nlev; ++k) {
    MultiFab U(S.ba[k], S.dm[k], nc, ng);
    U.set_val(Real(0));
    levels->push_back(AmrLevelMP{std::move(U), nullptr, S.dx[k], S.dy[k]});
  }
  // Seed du grossier + injection piecewise-constante vers les fins, exactement comme
  // build_amr_compiled : ETAT CONSERVATIF COMPLET (set_conservative_state, vague 3 : desormais
  // cable en multi-blocs, prioritaire) sinon densite (composante 0, reste au repos) sinon zero.
  if (state && !state->empty())
    detail::coupler_write_coarse_state((*levels)[0].U, *state, S.n, nc);
  else if (has_density)
    detail::coupler_write_coarse((*levels)[0].U, density, S.n, nc, gamma);
  for (int k = 1; k < nlev; ++k)
    detail::coupler_inject_coarse_to_fine_mb((*levels)[0].U, (*levels)[k].U, S.replicated_coarse);

  AmrRuntimeBlock b;
  b.name = name;
  b.ncomp = nc;
  b.gamma = gamma;
  b.substeps = substeps;
  b.stride = stride;
  b.imex = imex;  // traitement temporel du bloc : selectionne advance vs imex_advance dans step()
  b.aux_ncomp = aux_comps<Model>();  // largeur aux LUE par le modele (B_z/T_e -> > kAuxBaseComps)
  b.cons_vars = Model::conservative_vars();  // noms + ROLES : resolution role -> comp des sources couplees
  b.levels = levels;

  const bool rprim = recon_prim;
  // advance : UN sous-pas de transport AMR du bloc (Berger-Oliger + reflux + average_down
  // conservatifs) de taille dt, avec SON schema (Limiter, Flux) sur SA pile de niveaux, source en
  // EULER AVANT (imex=false toujours ici : le chemin IMEX vit dans imex_advance, selectionne par
  // step()). La boucle de sous-pas (substeps) et la cadence stride sont PORTEES par AmrRuntime::step,
  // pas par cette fermeture : ainsi la semantique multirate est UNE seule fois dans le moteur (mirroir
  // de AmrSystemCoupler::step) et reste neutralisable / testable la-bas. FONCTEUR implicite :
  // advance_amr<Limiter, Flux> est une fonction template nommee (pas de lambda etendue cross-TU) ;
  // on la capture dans une std::function depuis CETTE TU (recette device-clean #64/#97).
  // tmethod (kEuler defaut) selectionne SSPRK3 (time='ssprk3') pour le transport explicite du bloc ;
  // kEuler -> Euler avant historique, bit-identique. La source explicite reste portee par advance_amr.
  b.advance = [model, rprim, time_method](std::vector<AmrLevelMP>& L, const Box2D& dom, Real dt,
                                          Periodicity per, bool repl) {
    advance_amr<Limiter, Flux>(model, L, dom, dt, per, repl, rprim, /*imex=*/false, NewtonOptions{},
                               time_method);
  };
  // imex_advance (capstone vii) : UN pas de Lie [transport source-free ; source implicite] dont la
  // SEMANTIQUE calque la branche IMEX de AmrSystemCoupler::step (SourceFreeModel + AmrImplicitSourceStepper),
  // peuple SEULEMENT si imex. (1) transport EXPLICITE sur le modele SOURCE-FREE (SourceFreeModel<Model> :
  // flux/CFL du modele, source nulle) par le MEME moteur AMR (reflux conservatif) ; (2) source raide
  // IMPLICITE backward_euler_source A CHAQUE NIVEAU (Newton local), avec le masque @p implicit_components
  // porte par le BLOC (IMEX partiel) ; (3) cascade fin -> grossier (mf_average_down_mb) pour la coherence
  // des cellules grossieres couvertes. AmrRuntime::step appelle cette fermeture substeps fois : a
  // substeps=1 c'est exactement la branche IMEX compile-time, pour substeps>1 le runtime SOUS-CYCLE le
  // splitting (decision assumee, cf. SEMANTIQUE IMEX SOUS substeps dans amr_runtime.hpp).
  // On CAPTURE le masque dans un ImplicitMask<Model::n_vars> (POD device-clean) une fois ici (la
  // largeur n_vars n'est connue qu'au build, le masque est inactif si implicit_components est vide ->
  // backward-Euler plein, bit-identique a l'IMEX sans masque). SourceFreeModel<Model> est un type
  // concret instancie DANS cette TU : son advance_amr<Limiter, Flux> reste compile (pas de lambda
  // etendue cross-TU), capture dans la std::function de signature identique a advance. La reconstruction
  // du demi-pas source-free reste CONSERVATIVE (recon_prim=false) : MEME choix que AmrSystemCoupler::step
  // (qui appelle advance_amr sur SourceFreeModel avec le defaut), et SourceFreeModel n'expose de toute
  // facon pas les variables primitives (cf. son en-tete). Le bloc EXPLICITE, lui, garde recon_prim=rprim.
  if (imex) {
    ImplicitMask<Model::n_vars> mask;
    for (int c : implicit_components)
      if (c >= 0 && c < Model::n_vars) { mask.active = true; mask.flag[c] = true; }
    // DIAGNOSTICS NEWTON (OPT-IN, vague 3) : on alloue le rapport AGREGE du bloc dans un shared_ptr
    // (adresse STABLE meme apres deplacement de l'AmrRuntimeBlock dans le registre du moteur) et on
    // capture son pointeur nu dans la fermeture imex_advance. newton_diagnostics==false (defaut) ->
    // nreport=nullptr -> backward_euler_source chemin RAPIDE, bit-identique. Le RESET du rapport est a
    // la charge d'AmrRuntime::step (tete d'avance du bloc), comme System::AdvanceImex.
    std::shared_ptr<NewtonReport> nrep;
    if (newton_diagnostics) {
      nrep = std::make_shared<NewtonReport>();
      b.newton_diagnostics = true;
      b.newton_report = nrep;
    }
    NewtonReport* nreport = nrep.get();  // nul sans diagnostics ; adresse stable sinon
    b.imex_advance = [model, mask, nopts, nreport](std::vector<AmrLevelMP>& L, const Box2D& dom,
                                                   Real dt, Periodicity per, bool repl) {
      // (1) transport explicite source-free (-div F seul), reflux porte la conservation hyperbolique.
      advance_amr<Limiter, Flux>(SourceFreeModel<Model>{model}, L, dom, dt, per, repl,
                                 /*recon_prim=*/false, /*imex=*/false);
      // (2) source raide implicite backward-Euler PAR NIVEAU (Newton local, masque de bloc). Le rapport
      // nreport (nul sans diagnostics) AGREGE sur les niveaux : backward_euler_source fait son propre
      // max/somme + all_reduce MPI dans *nreport (pas de reset ici -> il cumule aussi sur les sous-pas,
      // step() ayant reset en tete d'avance). nreport==nullptr -> chemin rapide bit-identique.
      const int nlev_l = static_cast<int>(L.size());
      for (int k = 0; k < nlev_l; ++k)
        backward_euler_source<Model>(model, *L[k].aux, L[k].U, dt, nopts, mask, nreport);
      // (3) INVARIANT DE COUVERTURE (cf. AmrImplicitSourceStepper) : la source implicite a ete resolue
      // niveau par niveau, donc une cellule grossiere COUVERTE porterait une source grossiere fantome
      // au lieu de la moyenne 2x2 de ses enfants. Cascade fin -> grossier pour la coherence (la masse,
      // somme du seul grossier, ne compte alors pas la source du patch en double). Mono-niveau : boucle
      // vide -> bit-identique. La source restant CELLULE-LOCALE (hors flux de face), elle n'entre PAS
      // dans les registres de reflux : la conservation aux interfaces grossier-fin reste intacte.
      for (int k = nlev_l - 1; k >= 1; --k) mf_average_down_mb(L[k].U, L[k - 1].U);
    };
  }
  // contribution du bloc au RHS de Poisson SOMME : rhs += elliptic_rhs(U) sur le grossier (boucle
  // hote pure). MEME foncteur que System plat (make_poisson_rhs -> detail::PoissonRhs) -> chaque
  // bloc accumule (+=) aux MEMES cellules du grossier partage (co-localisation par cellule).
  b.add_elliptic_rhs = make_poisson_rhs(model);
  // VITESSE DE CFL du bloc : MEME politique que System (make_max_speed) -- lambda* de stabilite
  // (trait HasStabilitySpeed) si le modele le declare, sinon max_wave_speed (fallback historique,
  // bit-identique). Les solveurs de Riemann lisent toujours max_wave_speed.
  if constexpr (HasStabilitySpeed<Model>) {
    b.max_speed = [model](const MultiFab& U, const MultiFab& aux) {
      return max_stability_speed_mf(model, U, aux);
    };
  } else {
    b.max_speed = [model](const MultiFab& U, const MultiFab& aux) {
      return max_wave_speed_mf(model, U, aux);
    };
  }
  // BORNES DE PAS OPTIONNELLES (StabilityPolicy AMR) : memes reductions que System
  // (max_source_frequency_mf / min_stability_dt_mf), evaluees par AmrRuntime::step_cfl sur le
  // GROSSIER. Fermetures laissees VIDES quand le modele ne declare pas le trait (bit-identique).
  if constexpr (HasSourceFrequency<Model>) {
    b.source_frequency = [model](const MultiFab& U, const MultiFab& aux) {
      return max_source_frequency_mf(model, U, aux);
    };
  }
  if constexpr (HasStabilityDt<Model>) {
    b.stability_dt = [model](const MultiFab& U, const MultiFab& aux) {
      return min_stability_dt_mf(model, U, aux);
    };
  }
  const Geometry g = S.geom;
  const bool repl = S.replicated_coarse;
  b.mass = [levels, g, repl] {
    const MultiFab& U = (*levels)[0].U;
    const Real dV = g.dx() * g.dy();
    Real M = 0;
    for (int li = 0; li < U.local_size(); ++li) {
      const ConstArray4 u = U.fab(li).const_array();
      M += for_each_cell_reduce_sum(
          U.box(li), [u, dV] ADC_HD(int i, int j) { return u(i, j, 0) * dV; });
    }
    return repl ? M : all_reduce_sum(M);
  };
  const int nn = S.n;
  b.density = [levels, nn, repl] {
    return detail::coupler_read_coarse((*levels)[0].U, nn, repl);
  };
  b.potential = [nn, repl](const MultiFab& aux0) {
    return detail::coupler_read_coarse_phi(aux0, nn, repl);
  };
  return b;
}

/// Dispatch du schema spatial (limiteur x flux Riemann) -> build_amr_block. MEMES gardes que
/// dispatch_amr_compiled (hllc/roe exigent un transport compressible a 4 variables + pression).
/// Pendant multi-blocs de dispatch_amr_compiled. @p implicit_components : masque IMEX partiel porte
/// par le bloc (indices des composantes implicites ; vide = backward-Euler plein), thread a build_amr_block.
template <class Model>
AmrRuntimeBlock dispatch_amr_block(const Model& m, const std::string& lim, const std::string& riem,
                                   const SharedAmrLayout& S, const std::string& name,
                                   const std::vector<double>& density, bool has_density,
                                   double gamma, int substeps, bool recon_prim, bool imex,
                                   int stride = 1,
                                   const std::vector<int>& implicit_components = {},
                                   const NewtonOptions& nopts = {},
                                   const std::vector<double>* state = nullptr,
                                   bool newton_diagnostics = false,
                                   AmrTimeMethod time_method = AmrTimeMethod::kEuler) {
  // VALIDATION CENTRALISEE (registry dispatch_tags.hpp) AVANT le dispatch : memes tags acceptes /
  // rejetes qu'avant, messages identiques. Le dispatch template if/else qui suit est INCHANGE ; les
  // gardes de capabilite (hllc/roe : Euler 2D ou capability) restent des `if constexpr` PAR MODELE.
  validate_riemann(riem, /*polar=*/false, "add_block(AmrSystem, multi-blocs)");
  validate_limiter(lim, "add_block(AmrSystem, multi-blocs)");
  if (riem == "rusanov") {
    if (lim == "none")
      return build_amr_block<Model, NoSlope, RusanovFlux>(m, S, name, density, has_density, gamma,
                                                          substeps, recon_prim, imex, stride, implicit_components, nopts, state, newton_diagnostics, time_method);
    if (lim == "minmod")
      return build_amr_block<Model, Minmod, RusanovFlux>(m, S, name, density, has_density, gamma,
                                                        substeps, recon_prim, imex, stride, implicit_components, nopts, state, newton_diagnostics, time_method);
    if (lim == "vanleer")
      return build_amr_block<Model, VanLeer, RusanovFlux>(m, S, name, density, has_density, gamma,
                                                         substeps, recon_prim, imex, stride, implicit_components, nopts, state, newton_diagnostics, time_method);
    if (lim == "weno5")
      return build_amr_block<Model, Weno5, RusanovFlux>(m, S, name, density, has_density, gamma,
                                                       substeps, recon_prim, imex, stride, implicit_components, nopts, state, newton_diagnostics, time_method);
    throw_registry_dispatch_mismatch("add_block(AmrSystem, multi-blocs)", "limiteur", lim);
  }
  if (riem == "hll") {
    // HLL : 2 ondes signees, generique des que le modele expose wave_speeds (PAS de pression ni
    // n_vars == 4 exiges) -- MEME garde que System::make_block (alignement de surface System/AMR).
    if constexpr (requires(const Model mm, typename Model::State s, Aux a, Real r) {
                    mm.wave_speeds(s, a, 0, r, r);
                  }) {
      if (lim == "none")
        return build_amr_block<Model, NoSlope, HLLFlux>(m, S, name, density, has_density, gamma,
                                                        substeps, recon_prim, imex, stride, implicit_components, nopts, state, newton_diagnostics, time_method);
      if (lim == "minmod")
        return build_amr_block<Model, Minmod, HLLFlux>(m, S, name, density, has_density, gamma,
                                                       substeps, recon_prim, imex, stride, implicit_components, nopts, state, newton_diagnostics, time_method);
      if (lim == "vanleer")
        return build_amr_block<Model, VanLeer, HLLFlux>(m, S, name, density, has_density, gamma,
                                                        substeps, recon_prim, imex, stride, implicit_components, nopts, state, newton_diagnostics, time_method);
      if (lim == "weno5")
        return build_amr_block<Model, Weno5, HLLFlux>(m, S, name, density, has_density, gamma,
                                                      substeps, recon_prim, imex, stride, implicit_components, nopts, state, newton_diagnostics, time_method);
      throw_registry_dispatch_mismatch("add_block(AmrSystem, multi-blocs)", "limiteur", lim);
    } else {
      throw std::runtime_error("add_block(AmrSystem, multi-blocs) : flux 'hll' exige des vitesses "
                               "d'onde signees (model.wave_speeds) ; ce transport -> 'rusanov'");
    }
  }
  if (riem == "hllc") {
    // Capability HasHLLCStructure OU chemin canonique Euler 2D -- meme gate que System::make_block.
    if constexpr (HasHLLCStructure<Model> ||
                  (Model::n_vars == 4 &&
                   requires(const Model mm, typename Model::State s) { mm.pressure(s); })) {
      if (lim == "none")
        return build_amr_block<Model, NoSlope, HLLCFlux>(m, S, name, density, has_density, gamma,
                                                        substeps, recon_prim, imex, stride, implicit_components, nopts, state, newton_diagnostics, time_method);
      if (lim == "minmod")
        return build_amr_block<Model, Minmod, HLLCFlux>(m, S, name, density, has_density, gamma,
                                                      substeps, recon_prim, imex, stride, implicit_components, nopts, state, newton_diagnostics, time_method);
      if (lim == "vanleer")
        return build_amr_block<Model, VanLeer, HLLCFlux>(m, S, name, density, has_density, gamma,
                                                       substeps, recon_prim, imex, stride, implicit_components, nopts, state, newton_diagnostics, time_method);
      // weno5 : PARITE avec System::make_block (qui route hllc+weno5). Avant ce chantier la branche
      // hllc AMR n'avait pas de cas weno5 -> "limiter inconnu" la ou System buildait : divergence de
      // table corrigee (build_amr_block supporte Weno5, deja cable sur rusanov/hll).
      if (lim == "weno5")
        return build_amr_block<Model, Weno5, HLLCFlux>(m, S, name, density, has_density, gamma,
                                                       substeps, recon_prim, imex, stride, implicit_components, nopts, state, newton_diagnostics, time_method);
      throw_registry_dispatch_mismatch("add_block(AmrSystem, multi-blocs)", "limiteur", lim);
    } else {
      throw std::runtime_error("add_block(AmrSystem, multi-blocs) : flux 'hllc' exige un transport "
                               "compressible (4 variables + pression)");
    }
  }
  if (riem == "roe") {
    // Capability HasRoeDissipation OU chemin canonique Euler 2D -- meme gate que System::make_block.
    if constexpr (HasRoeDissipation<Model> ||
                  (Model::n_vars == 4 &&
                   requires(const Model mm, typename Model::State s) { mm.pressure(s); })) {
      if (lim == "none")
        return build_amr_block<Model, NoSlope, RoeFlux>(m, S, name, density, has_density, gamma,
                                                       substeps, recon_prim, imex, stride, implicit_components, nopts, state, newton_diagnostics, time_method);
      if (lim == "minmod")
        return build_amr_block<Model, Minmod, RoeFlux>(m, S, name, density, has_density, gamma,
                                                     substeps, recon_prim, imex, stride, implicit_components, nopts, state, newton_diagnostics, time_method);
      if (lim == "vanleer")
        return build_amr_block<Model, VanLeer, RoeFlux>(m, S, name, density, has_density, gamma,
                                                      substeps, recon_prim, imex, stride, implicit_components, nopts, state, newton_diagnostics, time_method);
      // weno5 : PARITE avec System::make_block (qui route roe+weno5). Meme correction de divergence
      // de table que la branche hllc ci-dessus.
      if (lim == "weno5")
        return build_amr_block<Model, Weno5, RoeFlux>(m, S, name, density, has_density, gamma,
                                                      substeps, recon_prim, imex, stride, implicit_components, nopts, state, newton_diagnostics, time_method);
      throw_registry_dispatch_mismatch("add_block(AmrSystem, multi-blocs)", "limiteur", lim);
    } else {
      throw std::runtime_error("add_block(AmrSystem, multi-blocs) : flux 'roe' exige un transport "
                               "compressible (4 variables + pression)");
    }
  }
  throw_registry_dispatch_mismatch("add_block(AmrSystem, multi-blocs)", "flux", riem);
}

/// Dispatch du schema spatial (limiteur x flux Riemann) -> build_amr_compiled. Memes gardes que
/// AmrSystem::add_block (hllc/roe exigent un transport compressible a 4 variables + pression).
template <class Model>
AmrCompiledHooks dispatch_amr_compiled(const Model& m, const std::string& lim,
                                       const std::string& riem, const AmrBuildParams& bp) {
  // VALIDATION CENTRALISEE (registry dispatch_tags.hpp) AVANT le dispatch : memes tags acceptes /
  // rejetes qu'avant. Dispatch template if/else INCHANGE ; gardes de capabilite hllc/roe par modele.
  validate_riemann(riem, /*polar=*/false, "add_compiled_model(AmrSystem)");
  validate_limiter(lim, "add_compiled_model(AmrSystem)");
  if (riem == "rusanov") {
    if (lim == "none") return build_amr_compiled<Model, NoSlope, RusanovFlux>(m, bp);
    if (lim == "minmod") return build_amr_compiled<Model, Minmod, RusanovFlux>(m, bp);
    if (lim == "vanleer") return build_amr_compiled<Model, VanLeer, RusanovFlux>(m, bp);
    // WENO5-Z (3 ghosts) : meme mecanisme que System (block_n_ghost(limiter)). Ici les niveaux du
    // coupleur sont alloues a Limiter::n_ghost (build_amr_compiled : ng = Weno5::n_ghost = 3) et le
    // regrid HERITE n_grow() (amr_regrid_finest : ngf = L[fk].U.n_grow()), donc le stencil 5 points
    // ne lit pas hors bornes. Cable sur AMR au MEME titre que none/minmod (rusanov uniquement).
    if (lim == "weno5") return build_amr_compiled<Model, Weno5, RusanovFlux>(m, bp);
    throw_registry_dispatch_mismatch("add_compiled_model(AmrSystem)", "limiteur", lim);
  }
  if (riem == "hll") {
    // HLL : generique des que le modele expose wave_speeds (le DSL les emet des qu'une primitive
    // 'p' est declaree) -- MEME garde que System::make_block (alignement de surface System/AMR).
    if constexpr (requires(const Model mm, typename Model::State s, Aux a, Real r) {
                    mm.wave_speeds(s, a, 0, r, r);
                  }) {
      if (lim == "none") return build_amr_compiled<Model, NoSlope, HLLFlux>(m, bp);
      if (lim == "minmod") return build_amr_compiled<Model, Minmod, HLLFlux>(m, bp);
      if (lim == "vanleer") return build_amr_compiled<Model, VanLeer, HLLFlux>(m, bp);
      if (lim == "weno5") return build_amr_compiled<Model, Weno5, HLLFlux>(m, bp);
      throw_registry_dispatch_mismatch("add_compiled_model(AmrSystem)", "limiteur", lim);
    } else {
      throw std::runtime_error("add_compiled_model(AmrSystem) : flux 'hll' exige des vitesses "
                               "d'onde signees (model.wave_speeds : declarer une primitive 'p') ; "
                               "ce transport -> 'rusanov'");
    }
  }
  if (riem == "hllc") {
    // Capability HasHLLCStructure OU chemin canonique Euler 2D -- meme gate que System::make_block.
    if constexpr (HasHLLCStructure<Model> ||
                  (Model::n_vars == 4 &&
                   requires(const Model mm, typename Model::State s) { mm.pressure(s); })) {
      if (lim == "none") return build_amr_compiled<Model, NoSlope, HLLCFlux>(m, bp);
      if (lim == "minmod") return build_amr_compiled<Model, Minmod, HLLCFlux>(m, bp);
      if (lim == "vanleer") return build_amr_compiled<Model, VanLeer, HLLCFlux>(m, bp);
      // weno5 : PARITE avec System::make_block (qui route hllc+weno5). Avant ce chantier la branche
      // hllc du chemin compile n'avait pas de cas weno5 (build_amr_compiled supporte pourtant Weno5,
      // deja cable sur rusanov/hll) -> "limiter inconnu" la ou System buildait : divergence corrigee.
      if (lim == "weno5") return build_amr_compiled<Model, Weno5, HLLCFlux>(m, bp);
      throw_registry_dispatch_mismatch("add_compiled_model(AmrSystem)", "limiteur", lim);
    } else {
      throw std::runtime_error("add_compiled_model(AmrSystem) : flux 'hllc' exige un transport "
                               "compressible (4 variables + pression)");
    }
  }
  if (riem == "roe") {
    // Capability HasRoeDissipation OU chemin canonique Euler 2D -- meme gate que System::make_block.
    if constexpr (HasRoeDissipation<Model> ||
                  (Model::n_vars == 4 &&
                   requires(const Model mm, typename Model::State s) { mm.pressure(s); })) {
      if (lim == "none") return build_amr_compiled<Model, NoSlope, RoeFlux>(m, bp);
      if (lim == "minmod") return build_amr_compiled<Model, Minmod, RoeFlux>(m, bp);
      if (lim == "vanleer") return build_amr_compiled<Model, VanLeer, RoeFlux>(m, bp);
      // weno5 : PARITE avec System::make_block (qui route roe+weno5). Meme correction de divergence
      // de table que la branche hllc ci-dessus.
      if (lim == "weno5") return build_amr_compiled<Model, Weno5, RoeFlux>(m, bp);
      throw_registry_dispatch_mismatch("add_compiled_model(AmrSystem)", "limiteur", lim);
    } else {
      throw std::runtime_error("add_compiled_model(AmrSystem) : flux 'roe' exige un transport "
                               "compressible (4 variables + pression)");
    }
  }
  throw_registry_dispatch_mismatch("add_compiled_model(AmrSystem)", "flux", riem);
}

}  // namespace detail

/// Resout le MASQUE IMEX partiel (implicit_vars / implicit_roles) d'un bloc COMPILE en indices de
/// composantes conservees, contre le descripteur conservatif @p cons du Model CONCRET (connu ici).
/// MEME logique stricte que resolve_implicit_components d'amr_system.cpp (nom/role absent -> erreur ;
/// indices uniques tries) -- repliquee ici car ce header ne depend pas du .cpp de la facade. VIDE en
/// entree -> vide -> masque inactif (backward-Euler plein). Utilisee par le builder runtime multi-blocs.
inline std::vector<int> resolve_implicit_components_compiled(
    const std::string& block, const VariableSet& cons, const std::vector<std::string>& names,
    const std::vector<std::string>& roles) {
  std::vector<int> out;
  auto push_unique = [&out](int c) {
    if (std::find(out.begin(), out.end(), c) == out.end()) out.push_back(c);
  };
  for (const std::string& nm : names) {
    int idx = -1;
    for (int i = 0; i < static_cast<int>(cons.names.size()); ++i)
      if (cons.names[i] == nm) { idx = i; break; }
    if (idx < 0)
      throw std::runtime_error("add_compiled_model(AmrSystem) : implicit_vars : variable '" + nm +
                               "' absente du bloc '" + block + "'");
    push_unique(idx);
  }
  for (const std::string& rn : roles) {
    const VariableRole role = role_from_name(rn);
    const int idx = cons.index_of(role);
    if (role == VariableRole::Custom || idx < 0)
      throw std::runtime_error("add_compiled_model(AmrSystem) : implicit_roles : role '" + rn +
                               "' absent du bloc '" + block + "'");
    push_unique(idx);
  }
  std::sort(out.begin(), out.end());
  return out;
}

/// Branche @p model (CompositeModel concret) comme un bloc AMR de @p sys, avec le schema demande. Le
/// build est DIFFERE (comme add_block) : les fermetures capturees sont invoquees au premier
/// step/mass/density via ensure_built(), apres set_refinement / set_poisson / set_density.
///
/// MONO-BLOC (un seul add_compiled_model) : chemin AmrCouplerMP<Model> historique (mono_builder),
/// bit-identique. MULTI-BLOCS (>= 2 blocs, compiles et/ou natifs melanges ; capstone v) : le bloc est
/// materialise comme AmrRuntimeBlock type-erase sur le layout PARTAGE par le multi_builder, exactement
/// comme add_block natif. On fige les DEUX builders ici (la facade choisit le routage a ensure_built).
/// @p time : "explicit" (source en Euler avant) ou "imex" (source raide implicite via
/// backward_euler_source, transport explicite porte par le reflux). Tout autre traitement est refuse.
/// @p stride : cadence HOLD-THEN-CATCH-UP du bloc en multi-blocs (1 = chaque macro-pas).
/// @p implicit_vars / @p implicit_roles : masque IMEX partiel du bloc (multi-blocs ; exige time=imex).
/// @throws std::runtime_error si le systeme est deja construit ou si time/recon hors domaine.
template <class Model>
void add_compiled_model(AmrSystem& sys, const std::string& name, Model model,
                        const std::string& limiter = "minmod",
                        const std::string& riemann = "rusanov",
                        const std::string& recon = "conservative",
                        const std::string& time = "explicit", double gamma = 1.4, int substeps = 1,
                        int stride = 1, const std::vector<std::string>& implicit_vars = {},
                        const std::vector<std::string>& implicit_roles = {}) {
  if (substeps < 1) throw std::runtime_error("add_compiled_model(AmrSystem) : substeps >= 1");
  // SSPRK3 N'EST PAS transporte par le chemin COMPILE : ni le mono_builder ni le multi_builder ne
  // figent AmrBuildParams::time_method / ne passent AmrTimeMethod a dispatch_amr_block (l'ABI plate du
  // loader .so ne marshale pas la methode). Rejet EXPLICITE plutot qu'un repli kEuler silencieux ; un
  // bloc SSPRK3 doit etre NATIF (AmrSystem::add_block / dispatch_amr_block, qui le thread).
  if (time == "ssprk3")
    throw std::runtime_error("add_compiled_model(AmrSystem) : time='ssprk3' non transporte par le "
                             "chemin compile (.so) ; utiliser un bloc natif adc.Model(...).");
  if (time != "explicit" && time != "imex")
    throw std::runtime_error("add_compiled_model(AmrSystem) : time '" + time +
                             "' inconnu (explicit|imex)");
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error("add_compiled_model(AmrSystem) : recon inconnu '" + recon +
                             "' (conservative|primitive)");
  const bool recon_prim = (recon == "primitive");
  const bool imex = (time == "imex");
  // (1) Builder MONO-BLOC : capture le Model concret + le schema, materialise l'AmrCouplerMP au build
  // paresseux (parametres refine/poisson/density figes a ce moment-la). Chemin historique, intouche.
  auto mono_builder = [model, limiter, riemann, recon_prim, imex](const AmrBuildParams& bp) {
    AmrBuildParams p = bp;
    p.recon_prim = recon_prim;
    p.imex = imex;
    return detail::dispatch_amr_compiled(model, limiter, riemann, p);
  };
  // (2) Builder MULTI-BLOCS : capture le MEME Model/schema concrets, materialise l'AmrRuntimeBlock du
  // bloc sur le layout PARTAGE (commun a tous les blocs, cree une fois a ensure_built). Resout LUI-MEME
  // le masque IMEX partiel contre cons_vars du Model concret (connu ici), puis appelle dispatch_amr_block
  // -- EXACTEMENT le chemin natif d'add_block, seul le point de resolution du type differe (ici a
  // l'ajout, la-bas d'une ModelSpec au build). FONCTEUR sans lambda etendue cross-TU dans le noyau :
  // dispatch_amr_block capture advance_amr<Limiter, Flux> (fonction template nommee), recette
  // device-clean #64/#97 ; la lambda externe ne fait qu'orchestrer (pas de kernel device en son corps).
  auto multi_builder = [model, limiter, riemann](
                           const detail::SharedAmrLayout& S, const std::string& bname,
                           const std::vector<double>& density, bool has_density, double bgamma,
                           int bsub, bool brecon_prim, bool bimex, int bstride,
                           const std::vector<std::string>& ivars,
                           const std::vector<std::string>& iroles) {
    const std::vector<int> impl_components =
        bimex ? resolve_implicit_components_compiled(bname, Model::conservative_vars(), ivars, iroles)
              : std::vector<int>{};
    return detail::dispatch_amr_block(model, limiter, riemann, S, bname, density, has_density, bgamma,
                                      bsub, brecon_prim, bimex, bstride, impl_components);
  };
  sys.set_compiled_block(Model::n_vars, gamma, substeps, std::move(mono_builder),
                         std::move(multi_builder), name, recon_prim, imex, stride, implicit_vars,
                         implicit_roles);
}

}  // namespace adc
