#pragma once

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

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/// @file
/// @brief add_compiled_model cote AmrSystem : branche un modele COMPILE (un CompositeModel, genere
///        par le DSL ou ecrit a la main, connu a la COMPILATION) comme l'unique bloc d'une hierarchie
///        AMR, EXACTEMENT le chemin de production de AmrSystem::add_block mais SANS passer par la
///        dispatch ModelSpec (le modele est deja un type concret).
///
/// Pendant raffine de add_compiled_model(System&, ...) (dsl_block.hpp). La machinerie de build du
/// coupleur AMR (AmrCouplerMP<Model> + reflux conservatif + regrid) est instanciee ICI, depuis l'unite
/// de traduction APPELANTE, sur le type Model concret -- comme block_builder.hpp pour le System plat.
/// Le coupleur, type-erased en std::function (step / mass / max_speed / n_patches / density), entre
/// dans AmrSystem par AmrSystem::set_compiled_block (methode non-template). Le MEME builder partage
/// (detail::build_amr_compiled / dispatch_amr_compiled) sert AUSSI le chemin ModelSpec natif d'add_block
/// (amr_system.cpp), une fois le type Model concret resolu par detail::dispatch_model : un seul build.

namespace adc {

/// Paquet (limiteur, flux Riemann) attendu par AmrCouplerMP::step<Disc>. Unique definition : le
/// chemin natif d'amr_system.cpp passe par ce meme header (plus de DiscLF duplique cote .cpp).
template <class L, class F>
struct AmrDiscLF {
  using Limiter = L;
  using NumericalFlux = F;
};

namespace detail {

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
  const int I0 = bp.n / 4, I1 = 3 * bp.n / 4 - 1, J0 = bp.n / 4, J1 = 3 * bp.n / 4 - 1;
  Box2D fb{{2 * I0, 2 * J0}, {2 * I1 + 1, 2 * J1 + 1}};
  BoxArray baf(std::vector<Box2D>{fb});
  MultiFab Uf(baf, dm, nc, ng);
  Uf.set_val(Real(0));
  std::vector<AmrLevelMP> levels;
  levels.push_back({std::move(Uc), nullptr, dxc, dxc});
  levels.push_back({std::move(Uf), nullptr, dxf, dxf});

  auto cpl = std::make_shared<Coupler>(model, g, bac, bp.poisson_bc, std::move(levels), bp.wall,
                                       !bp.distribute_coarse);
  if (bp.has_density) coupler_write_coarse(cpl->coarse(), bp.density, bp.n, nc, bp.gamma);
  auto& Lv = cpl->levels();
  for (std::size_t k = 1; k < Lv.size(); ++k)
    coupler_inject_coarse_to_fine_mb(cpl->coarse(), Lv[k].U, !bp.distribute_coarse);

  const double thr = bp.refine_threshold;
  auto crit = [thr](const ConstArray4& a, int i, int j) { return a(i, j, 0) > thr; };
  cpl->regrid(crit);
  cpl->update();

  AmrCompiledHooks h;
  h.coupler_holder = cpl;  // duree de vie : les fermetures capturent cpl (shared_ptr)
  const int sub = bp.substeps;
  const bool rprim = bp.recon_prim;
  const bool imex = bp.imex;  // source raide implicite (backward_euler) plutot qu'Euler avant
  const int regrid_every = bp.regrid_every;
  auto step_state = std::make_shared<int>(0);  // compteur de pas partage par la fermeture
  h.step = [cpl, crit, sub, rprim, imex, regrid_every, step_state](double dt) {
    if (regrid_every > 0 && *step_state > 0 && *step_state % regrid_every == 0) cpl->regrid(crit);
    const double h2 = dt / sub;
    for (int s = 0; s < sub; ++s) cpl->template step<AmrDiscLF<Limiter, Flux>>(h2, rprim, imex);
    ++*step_state;
  };
  h.max_speed = [cpl] { return static_cast<double>(cpl->max_wave_speed()); };
  h.mass = [cpl] { return static_cast<double>(cpl->mass()); };
  h.n_patches = [cpl] {
    auto& L = cpl->levels();
    return L.size() >= 2 ? static_cast<int>(L[1].U.box_array().size()) : 0;
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
                                const std::vector<int>& implicit_components = {}) {
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
  // densite initiale (composante 0) sur le grossier + injection piecewise-constante vers les fins,
  // exactement comme build_amr_compiled. Sans densite : grossier a zero (bloc neutre / fond).
  if (has_density)
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
  b.advance = [model, rprim](std::vector<AmrLevelMP>& L, const Box2D& dom, Real dt,
                             Periodicity per, bool repl) {
    advance_amr<Limiter, Flux>(model, L, dom, dt, per, repl, rprim, /*imex=*/false);
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
    b.imex_advance = [model, mask](std::vector<AmrLevelMP>& L, const Box2D& dom, Real dt,
                                   Periodicity per, bool repl) {
      // (1) transport explicite source-free (-div F seul), reflux porte la conservation hyperbolique.
      advance_amr<Limiter, Flux>(SourceFreeModel<Model>{model}, L, dom, dt, per, repl,
                                 /*recon_prim=*/false, /*imex=*/false);
      // (2) source raide implicite backward-Euler PAR NIVEAU (Newton local, masque de bloc).
      const int nlev_l = static_cast<int>(L.size());
      for (int k = 0; k < nlev_l; ++k)
        backward_euler_source<Model>(model, *L[k].aux, L[k].U, dt, /*iters=*/2, mask);
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
  b.max_speed = [model](const MultiFab& U, const MultiFab& aux) {
    return max_wave_speed_mf(model, U, aux);
  };
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
                                   const std::vector<int>& implicit_components = {}) {
  if (riem == "rusanov") {
    if (lim == "none")
      return build_amr_block<Model, NoSlope, RusanovFlux>(m, S, name, density, has_density, gamma,
                                                          substeps, recon_prim, imex, stride, implicit_components);
    if (lim == "minmod")
      return build_amr_block<Model, Minmod, RusanovFlux>(m, S, name, density, has_density, gamma,
                                                        substeps, recon_prim, imex, stride, implicit_components);
    if (lim == "vanleer")
      return build_amr_block<Model, VanLeer, RusanovFlux>(m, S, name, density, has_density, gamma,
                                                         substeps, recon_prim, imex, stride, implicit_components);
    if (lim == "weno5")
      return build_amr_block<Model, Weno5, RusanovFlux>(m, S, name, density, has_density, gamma,
                                                       substeps, recon_prim, imex, stride, implicit_components);
    throw std::runtime_error("add_block(AmrSystem, multi-blocs) : limiter inconnu '" + lim + "'");
  }
  if (riem == "hllc") {
    if constexpr (Model::n_vars == 4 &&
                  requires(const Model mm, typename Model::State s) { mm.pressure(s); }) {
      if (lim == "none")
        return build_amr_block<Model, NoSlope, HLLCFlux>(m, S, name, density, has_density, gamma,
                                                        substeps, recon_prim, imex, stride, implicit_components);
      if (lim == "minmod")
        return build_amr_block<Model, Minmod, HLLCFlux>(m, S, name, density, has_density, gamma,
                                                      substeps, recon_prim, imex, stride, implicit_components);
      if (lim == "vanleer")
        return build_amr_block<Model, VanLeer, HLLCFlux>(m, S, name, density, has_density, gamma,
                                                       substeps, recon_prim, imex, stride, implicit_components);
      throw std::runtime_error("add_block(AmrSystem, multi-blocs) : limiter inconnu '" + lim + "'");
    } else {
      throw std::runtime_error("add_block(AmrSystem, multi-blocs) : flux 'hllc' exige un transport "
                               "compressible (4 variables + pression)");
    }
  }
  if (riem == "roe") {
    if constexpr (Model::n_vars == 4 &&
                  requires(const Model mm, typename Model::State s) { mm.pressure(s); }) {
      if (lim == "none")
        return build_amr_block<Model, NoSlope, RoeFlux>(m, S, name, density, has_density, gamma,
                                                       substeps, recon_prim, imex, stride, implicit_components);
      if (lim == "minmod")
        return build_amr_block<Model, Minmod, RoeFlux>(m, S, name, density, has_density, gamma,
                                                     substeps, recon_prim, imex, stride, implicit_components);
      if (lim == "vanleer")
        return build_amr_block<Model, VanLeer, RoeFlux>(m, S, name, density, has_density, gamma,
                                                      substeps, recon_prim, imex, stride, implicit_components);
      throw std::runtime_error("add_block(AmrSystem, multi-blocs) : limiter inconnu '" + lim + "'");
    } else {
      throw std::runtime_error("add_block(AmrSystem, multi-blocs) : flux 'roe' exige un transport "
                               "compressible (4 variables + pression)");
    }
  }
  throw std::runtime_error("add_block(AmrSystem, multi-blocs) : flux Riemann inconnu '" + riem +
                           "' (rusanov|hllc|roe)");
}

/// Dispatch du schema spatial (limiteur x flux Riemann) -> build_amr_compiled. Memes gardes que
/// AmrSystem::add_block (hllc/roe exigent un transport compressible a 4 variables + pression).
template <class Model>
AmrCompiledHooks dispatch_amr_compiled(const Model& m, const std::string& lim,
                                       const std::string& riem, const AmrBuildParams& bp) {
  if (riem == "rusanov") {
    if (lim == "none") return build_amr_compiled<Model, NoSlope, RusanovFlux>(m, bp);
    if (lim == "minmod") return build_amr_compiled<Model, Minmod, RusanovFlux>(m, bp);
    if (lim == "vanleer") return build_amr_compiled<Model, VanLeer, RusanovFlux>(m, bp);
    // WENO5-Z (3 ghosts) : meme mecanisme que System (block_n_ghost(limiter)). Ici les niveaux du
    // coupleur sont alloues a Limiter::n_ghost (build_amr_compiled : ng = Weno5::n_ghost = 3) et le
    // regrid HERITE n_grow() (amr_regrid_finest : ngf = L[fk].U.n_grow()), donc le stencil 5 points
    // ne lit pas hors bornes. Cable sur AMR au MEME titre que none/minmod (rusanov uniquement).
    if (lim == "weno5") return build_amr_compiled<Model, Weno5, RusanovFlux>(m, bp);
    throw std::runtime_error("add_compiled_model(AmrSystem) : limiter inconnu '" + lim + "'");
  }
  if (riem == "hllc") {
    if constexpr (Model::n_vars == 4 &&
                  requires(const Model mm, typename Model::State s) { mm.pressure(s); }) {
      if (lim == "none") return build_amr_compiled<Model, NoSlope, HLLCFlux>(m, bp);
      if (lim == "minmod") return build_amr_compiled<Model, Minmod, HLLCFlux>(m, bp);
      if (lim == "vanleer") return build_amr_compiled<Model, VanLeer, HLLCFlux>(m, bp);
      throw std::runtime_error("add_compiled_model(AmrSystem) : limiter inconnu '" + lim + "'");
    } else {
      throw std::runtime_error("add_compiled_model(AmrSystem) : flux 'hllc' exige un transport "
                               "compressible (4 variables + pression)");
    }
  }
  if (riem == "roe") {
    if constexpr (Model::n_vars == 4 &&
                  requires(const Model mm, typename Model::State s) { mm.pressure(s); }) {
      if (lim == "none") return build_amr_compiled<Model, NoSlope, RoeFlux>(m, bp);
      if (lim == "minmod") return build_amr_compiled<Model, Minmod, RoeFlux>(m, bp);
      if (lim == "vanleer") return build_amr_compiled<Model, VanLeer, RoeFlux>(m, bp);
      throw std::runtime_error("add_compiled_model(AmrSystem) : limiter inconnu '" + lim + "'");
    } else {
      throw std::runtime_error("add_compiled_model(AmrSystem) : flux 'roe' exige un transport "
                               "compressible (4 variables + pression)");
    }
  }
  throw std::runtime_error("add_compiled_model(AmrSystem) : flux Riemann inconnu '" + riem +
                           "' (rusanov|hllc|roe)");
}

}  // namespace detail

/// Branche @p model (CompositeModel concret) comme l'unique bloc AMR de @p sys, avec le schema demande.
/// Le build du coupleur est DIFFERE (comme add_block) : la fermeture capturee est invoquee au premier
/// step/mass/density via ensure_built(), apres set_refinement / set_poisson / set_density.
/// @p time : "explicit" (source en Euler avant) ou "imex" (source raide implicite via
/// backward_euler_source, transport explicite porte par le reflux). Tout autre traitement est refuse.
/// @throws std::runtime_error si un bloc est deja defini ou si time n'est pas dans {explicit, imex}.
template <class Model>
void add_compiled_model(AmrSystem& sys, const std::string& name, Model model,
                        const std::string& limiter = "minmod",
                        const std::string& riemann = "rusanov",
                        const std::string& recon = "conservative",
                        const std::string& time = "explicit", double gamma = 1.4, int substeps = 1) {
  (void)name;
  if (substeps < 1) throw std::runtime_error("add_compiled_model(AmrSystem) : substeps >= 1");
  if (time != "explicit" && time != "imex")
    throw std::runtime_error("add_compiled_model(AmrSystem) : time '" + time +
                             "' inconnu (explicit|imex)");
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error("add_compiled_model(AmrSystem) : recon inconnu '" + recon +
                             "' (conservative|primitive)");
  const bool recon_prim = (recon == "primitive");
  const bool imex = (time == "imex");
  // Builder type-erase : capture le Model concret + le schema, materialise le coupleur au build
  // paresseux (avec les parametres refine/poisson/density figes a ce moment-la).
  auto builder = [model, limiter, riemann, recon_prim, imex](const AmrBuildParams& bp) {
    AmrBuildParams p = bp;
    p.recon_prim = recon_prim;
    p.imex = imex;
    return detail::dispatch_amr_compiled(model, limiter, riemann, p);
  };
  sys.set_compiled_block(Model::n_vars, gamma, substeps, std::move(builder));
}

}  // namespace adc
