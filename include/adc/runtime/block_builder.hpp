#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/numerical_flux.hpp>
#include <adc/numerics/reconstruction.hpp>
#include <adc/numerics/spatial_operator.hpp>
#include <adc/numerics/spatial_operator_eb.hpp>  // assemble_rhs_eb (cut-cell EB) + detail::DiscLevelSet (T5-PR2)
#include <adc/numerics/time/implicit_stepper.hpp>
#include <adc/numerics/time/time_steppers.hpp>
#include <adc/runtime/dispatch_tags.hpp>  // registry UNIQUE des tags (validate_limiter/riemann, limiter_n_ghost)
#include <adc/runtime/grid_context.hpp>  // GridContext + BlockClosures (en-tete leger partage)
#include <adc/runtime/wall_predicate.hpp>  // detail::DiscDomain (level set device-callable du disque)

#include <cmath>  // std::sqrt (coefficients ARS(2,2,2) : gamma = 1 - 1/sqrt(2), hote)
#include <functional>
#include <memory>       // std::shared_ptr (scratch partage du cache de vitesses d'onde HLL, opt-in)
#include <stdexcept>
#include <string>
#include <type_traits>  // std::is_same_v (engagement du cache uniquement pour le flux HLL)
#include <utility>
#include <vector>

/// @file
/// @brief Construit les fermetures d'un bloc (avance en temps + residu + contribution Poisson) a
///        partir d'un modele COMPILE (CompositeModel) et d'un contexte de grille.
///
/// Ce code etait dans System::Impl ; il est extrait en en-tete pour que le MEME chemin template
/// (assemble_rhs<Limiter, Flux>, inlinable et device-ready) soit instanciable depuis une UNITE DE
/// TRADUCTION EXTERNE. C'est la brique qui permettra a un modele genere par le DSL d'etre compile
/// AOT (ahead-of-time) puis branche dans le System par le chemin de PRODUCTION (flux HLLC/Roe,
/// ordre 2, GPU), et non plus seulement par le chemin hote virtuel du bloc dynamique.
///
/// Le System reste l'unique proprietaire du maillage et de l'aux ; GridContext n'en porte que des
/// copies immuables (domaine, CL, geometrie) et un POINTEUR non possedant vers l'aux (adresse stable,
/// duree de vie superieure au bloc).

namespace adc {

// GridContext et BlockClosures : definis dans adc/runtime/grid_context.hpp (en-tete leger, inclus
// aussi par system.hpp pour exposer grid_context() / install_block() sans tirer la numerique).

namespace detail {
/// Foncteur residu -div F + S (fill_ghosts puis assemble_rhs), passe AUX TimeStepper comme RhsEval.
/// FONCTEUR NOMME (et non lambda) : c'est lui que take_step recoit et qui declenche l'instanciation
/// d'assemble_rhs<Limiter, Flux> (et de son AssembleRhsKernel device). Premiere-instancie depuis une
/// TU EXTERNE (add_compiled_model), une lambda a cette place fait buter nvcc sur l'emission du kernel
/// device imbrique (Heisenbug : OK Serial + compute-sanitizer, segfault a l'execution Cuda). Une
/// classe a un contexte d'instanciation stable -> codegen device robuste. Corps identique a l'ancienne
/// lambda -> residu bit-identique a add_block sur CPU (et, vise, sur device).
template <class Limiter, class Flux, class Model>
struct BlockRhsEval {
  Model model;
  const GridContext* ctx;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< limiteur de positivite Zhang-Shu (<= 0 : inactif, bit-identique)
  /// Scratch des vitesses d'onde par cellule (cache HLL, opt-in). nullptr (defaut) -> chemin par face
  /// strictement inchange. Non nul UNIQUEMENT pour le flux HLL (cf. build_block) : la branche cachee
  /// n'est instanciee que pour Flux == HLLFlux, donc model.wave_speeds y est toujours present.
  std::shared_ptr<MultiFab> ws_cache;
  void operator()(MultiFab& U, MultiFab& R) const {
    fill_ghosts(U, ctx->dom, ctx->bc);
    if constexpr (std::is_same_v<Flux, HLLFlux>) {
      if (ws_cache) {
        // Re-alloue le scratch au layout courant (4 composantes, 1 ghost) : couvre une regrid AMR ou
        // un premier appel (shared_ptr vers un MultiFab vide). Sinon reutilise l'allocation existante.
        if (ws_cache->local_size() != U.local_size() || ws_cache->ncomp() != 4)
          *ws_cache = MultiFab(U.box_array(), U.dmap(), 4, 1);
        assemble_rhs_hll_cached<Limiter>(model, U, *ctx->aux, ctx->geom, R, *ws_cache, recon_prim,
                                         pos_floor);
        return;
      }
    }
    assemble_rhs<Limiter, Flux>(model, U, *ctx->aux, ctx->geom, R, recon_prim, pos_floor);
  }
};

/// Avance EXPLICITE : n sous-pas du stepper @c Stepper (SSPRK2 par defaut, SSPRK3 optionnel) sur le
/// residu transport+source. Le schema RK est un parametre de template (FONCTEUR NOMME du coeur :
/// SSPRK2Step / SSPRK3Step) -> meme contrat device-clean que SSPRK2Step. SSPRK2 reproduit a l'identique
/// l'avance historique (bit-identique).
template <class Limiter, class Flux, class Model, class Stepper = SSPRK2Step>
struct AdvanceExplicit {
  Model m;
  GridContext ctx;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< limiteur de positivite Zhang-Shu (<= 0 : inactif, bit-identique)
  std::shared_ptr<MultiFab> ws_cache;  ///< cache de vitesses d'onde HLL (opt-in) ; nullptr -> par face
  void operator()(MultiFab& U, Real dt, int n) const {
    const Real h = dt / static_cast<Real>(n);
    const BlockRhsEval<Limiter, Flux, Model> rhs{m, &ctx, recon_prim, pos_floor, ws_cache};
    for (int s = 0; s < n; ++s) Stepper{}.take_step(rhs, U, h);
  }
};

/// Avance IMEX : par sous-pas, demi-pas EXPLICITE (transport sans source) + source IMPLICITE raide.
/// @c mask : masque implicite PORTE PAR LE BLOC (override du defaut modele is_implicit), resolu une
/// fois a l'ajout du bloc contre ses noms/roles de variables. Masque inactif (defaut) -> backward_euler
/// retombe sur model_is_implicit -> avance bit-identique a l'historique.
template <class Limiter, class Flux, class Model>
struct AdvanceImex {
  Model m;
  GridContext ctx;
  bool recon_prim;
  ImplicitMask<Model::n_vars> mask{};
  NewtonOptions nopts{};            // options Newton du bloc (defauts = historique : 2 iters, 1e-7)
  NewtonReport* nreport = nullptr;  // diagnostics OPT-IN (adresse stable, possedee par System::Impl)
  Real pos_floor = Real(0);         ///< limiteur de positivite Zhang-Shu (<= 0 : inactif)
  void operator()(MultiFab& U, Real dt, int n) const {
    const Real h = dt / static_cast<Real>(n);
    const BlockRhsEval<Limiter, Flux, SourceFreeModel<Model>> rhs{SourceFreeModel<Model>{m}, &ctx,
                                                                  recon_prim, pos_floor};
    if (nreport) nreport->reset();  // rapport AGREGE sur les n sous-pas de CETTE avance
    for (int s = 0; s < n; ++s) {
      ForwardEuler{}.take_step(rhs, U, h);     // demi-pas explicite : transport sans source
      backward_euler_source(m, *ctx.aux, U, h, nopts, mask, nreport);  // source implicite (rappel raide)
    }
  }
};

/// Avance IMEX-RK ARS(2,2,2) (Ascher, Ruuth, Spiteri 1997 ; " Implicit-explicit Runge-Kutta methods
/// for time-dependent partial differential equations ", Appl. Numer. Math. 25) : transport explicite
/// (L = -div F) couple a la source raide implicite (backward-Euler LOCAL par cellule), ORDRE 2. C'est
/// une famille DISTINCTE et PARALLELE a AdvanceImex (qui reste le backward-Euler d'ordre 1 par defaut,
/// INTOUCHE et bit-identique). Coefficients : gamma = 1 - 1/sqrt(2), delta = 1 - 1/(2 gamma).
///
/// Tableaux (stiffly accurate, partie implicite SDIRK ; c = [0, gamma, 1] pour les deux) :
///   explicite : A_E = [[0, 0, 0], [gamma, 0, 0], [delta, 1-delta, 0]],  b_E = [delta, 1-delta, 0]
///   implicite : A_I = [[0, 0, 0], [0, gamma, 0], [0, 1-gamma, gamma]],  b_I = [0, 1-gamma, gamma]
///
/// b_E == derniere ligne de A_E et b_I == derniere ligne de A_I -> schema STIFFLY ACCURATE -> la
/// solution finale EST le dernier etage (U^{n+1} = U^(3)), aucune recombinaison finale. Avec L = le
/// transport SourceFreeModel et S = la source du modele complet, la recurrence par etage est :
///   U^(1) = U^n                                        (1re ligne de A_I nulle : aucun solve, S^(1) inutilise)
///   L1    = L(U^n)
///   U^(2) = U^n + dt*gamma*L1 + dt*gamma*S(U^(2))       (solve implicite : backward_euler_source au pas
///                                                        dt*gamma sur la base U^n + dt*gamma*L1)
///   L2    = L(U^(2))
///   U^(3) = U^n + dt*delta*L1 + dt*(1-delta)*L2 + dt*(1-gamma)*S^(2) + dt*gamma*S(U^(3))
///   U^{n+1} = U^(3)
/// Le terme dt*gamma*S^(2) n'est PAS reevalue : par construction du solve d'etage 2,
/// dt*gamma*S^(2) = U^(2) - base2 (l'increment du solve), donc dt*(1-gamma)*S^(2) = ((1-gamma)/gamma) *
/// (U^(2) - base2). AUCUN noyau de source supplementaire : on REUTILISE BlockRhsEval<SourceFreeModel>
/// (transport, MEME mecanisme que le demi-pas explicite d'AdvanceImex), backward_euler_source (solve
/// implicite local) et saxpy/lincomb (etages). Device-clean (aucun nouveau kernel).
///
/// SOURCE PLEINEMENT IMPLICITE : le masque IMEX partiel n'est PAS cable ici (la relation de coherence
/// dt*gamma*S^(2) = U^(2) - base2 suppose un solve d'etage homogene ; un traitement forward-backward
/// par composante melangerait les tableaux explicite/implicite). System::add_block rejette donc
/// implicit_vars/implicit_roles avec time='imexrk_ars222'. Les options Newton (nopts) sont, elles,
/// transportees : elles parametrent les DEUX solves implicites d'etage.
///
/// Les MultiFab d'etage sont alloues UNE FOIS par avance (hors boucle de sous-pas) : Un (U^n), L1/L2
/// (residus de transport), base2 (base de l'etage 2, relue a l'etage 3) sans ghost (lus en cellules
/// valides) ; work (etat d'etage) avec les ghosts de U (il passe au residu de transport).
template <class Limiter, class Flux, class Model>
struct AdvanceImexRkArs222 {
  Model m;
  GridContext ctx;
  bool recon_prim;
  NewtonOptions nopts{};            // options Newton des solves d'etage (defauts = historique)
  NewtonReport* nreport = nullptr;  // diagnostics OPT-IN (adresse stable, possedee par System::Impl)
  Real pos_floor = Real(0);         ///< limiteur de positivite Zhang-Shu (<= 0 : inactif)
  void operator()(MultiFab& U, Real dt, int n) const {
    const Real h = dt / static_cast<Real>(n);
    const Real gamma = Real(1) - Real(1) / std::sqrt(Real(2));
    const Real delta = Real(1) - Real(1) / (Real(2) * gamma);
    const Real cS2 = (Real(1) - gamma) / gamma;  // facteur de (U^(2) - base2) a l'etage 3
    const ImplicitMask<Model::n_vars> mask{};    // source PLEINEMENT implicite (masque inactif)
    // Residu de transport SANS source (L = -div F) : MEME mecanisme que le demi-pas explicite d'AdvanceImex.
    const BlockRhsEval<Limiter, Flux, SourceFreeModel<Model>> rhs{SourceFreeModel<Model>{m}, &ctx,
                                                                  recon_prim, pos_floor};
    const int nc = U.ncomp();
    MultiFab Un(U.box_array(), U.dmap(), nc, 0);     // U^n
    MultiFab L1(U.box_array(), U.dmap(), nc, 0);     // L(U^n)
    MultiFab L2(U.box_array(), U.dmap(), nc, 0);     // L(U^(2))
    MultiFab base2(U.box_array(), U.dmap(), nc, 0);  // U^n + dt*gamma*L1
    MultiFab work(U.box_array(), U.dmap(), nc, U.n_grow());  // etat d'etage (passe au transport)
    if (nreport) nreport->reset();  // rapport AGREGE sur les sous-pas ET les 2 solves d'etage
    for (int s = 0; s < n; ++s) {
      // Etage 1 : U^(1) = U^n ; L1 = L(U^n).
      lincomb(Un, Real(1), U, Real(0), U);            // Un = U^n (cellules valides)
      rhs(U, L1);                                      // L1 = L(U^n)  (fill_ghosts(U) + assemble_rhs)
      // Etage 2 : U^(2) = base2 + dt*gamma*S(U^(2)),  base2 = U^n + dt*gamma*L1.
      lincomb(base2, Real(1), Un, h * gamma, L1);      // base2 = U^n + dt*gamma*L1
      lincomb(work, Real(1), base2, Real(0), base2);   // work = base2
      backward_euler_source(m, *ctx.aux, work, h * gamma, nopts, mask, nreport);  // work = U^(2)
      rhs(work, L2);                                   // L2 = L(U^(2))
      // Etage 3 : U <- base3 = U^n + dt*delta*L1 + dt*(1-delta)*L2 + ((1-gamma)/gamma)*(U^(2) - base2).
      lincomb(U, Real(1), Un, h * delta, L1);          // U = U^n + dt*delta*L1
      saxpy(U, h * (Real(1) - delta), L2);             // + dt*(1-delta)*L2
      saxpy(U, cS2, work);                             // + ((1-gamma)/gamma)*U^(2)
      saxpy(U, -cS2, base2);                           // - ((1-gamma)/gamma)*base2  -> U = base3
      backward_euler_source(m, *ctx.aux, U, h * gamma, nopts, mask, nreport);  // U = U^(3) = U^{n+1}
    }
  }
};

/// Residu fige (fill_ghosts + assemble_rhs) installe comme rhs_into du bloc.
/// Foncteur du diagnostic dt_hotspot (ADC-182) : cellule dominante de la CFL du bloc.
/// HOTE (les reductions internes sont device) ; nomme, comme MaxSpeed.
template <class Model>
struct HotspotFn {
  Model m;
  GridContext ctx;
  void operator()(const MultiFab& U, Real& w, int& i, int& j) const {
    max_wave_speed_hotspot_mf(m, U, *ctx.aux, ctx.dom.nx(), w, i, j);
  }
};

template <class Limiter, class Flux, class Model>
struct RhsInto {
  Model m;
  GridContext ctx;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< limiteur de positivite Zhang-Shu (<= 0 : inactif, bit-identique)
  std::shared_ptr<MultiFab> ws_cache;  ///< cache de vitesses d'onde HLL (opt-in) ; nullptr -> par face
  void operator()(MultiFab& U, MultiFab& R) const {
    // Delegue a BlockRhsEval (fill_ghosts + assemble_rhs OU chemin cache) : source unique du residu.
    BlockRhsEval<Limiter, Flux, Model>{m, &ctx, recon_prim, pos_floor, ws_cache}(U, R);
  }
};

// ============================================================================
// ROUTAGE DISQUE (chantier T5-PR3) : evaluateurs de residu DISQUE + avances qui les portent.
// ============================================================================
// Le residu de transport d'un bloc passe par BlockRhsEval (assemble_rhs, plein cartesien). Les deux
// evaluateurs ci-dessous SUBSTITUENT l'operateur disque a assemble_rhs, en lisant la geometrie du
// System PAR POINTEUR (adresse stable d'un membre de Impl) au moment du pas -- l'ordre add_block /
// set_disc_domain est donc indifferent. Foncteurs NOMMES (meme contrat device que BlockRhsEval).

/// Residu de transport MASQUE (mode Staircase) : fill_ghosts puis assemble_rhs_masked sur le masque
/// 0/1 cellule-centre du System (lu via @c mask, pointeur vers Impl::disc_mask_, adresse stable). Le
/// masque a le MEME layout que U (memes ba/dm, 1 ghost). Cellule inactive -> residu 0 ; face vers une
/// inactive -> flux normal nul (paroi FV). Le flux / la reconstruction sont REUTILISES verbatim.
template <class Limiter, class Flux, class Model>
struct BlockRhsEvalMasked {
  Model model;
  const GridContext* ctx;
  const MultiFab* mask;  // Impl::disc_mask_ (NON possede ; adresse stable)
  bool recon_prim;
  Real pos_floor = Real(0);  ///< limiteur de positivite Zhang-Shu (<= 0 : inactif, bit-identique)
  void operator()(MultiFab& U, MultiFab& R) const {
    fill_ghosts(U, ctx->dom, ctx->bc);
    assemble_rhs_masked<Limiter, Flux>(model, U, *ctx->aux, *mask, ctx->geom, R, recon_prim,
                                       pos_floor);
  }
};

/// Residu de transport CUT-CELL / EB (mode CutCell) : fill_ghosts puis assemble_rhs_eb sur le level
/// set du disque du System (lu via @c disc, pointeur vers Impl::disc_, adresse stable). Le level set
/// device-callable est construit ICI sur l'HOTE (detail::disc_level_set(*disc) -> DiscLevelSet, un
/// FONCTEUR NOMME capturant trois doubles PAR VALEUR) et passe PAR VALEUR a assemble_rhs_eb : le noyau
/// device ne recoit donc PAS de std::function, il reste device-clean (cf. spatial_operator_eb.hpp).
template <class Limiter, class Flux, class Model>
struct BlockRhsEvalEb {
  Model model;
  const GridContext* ctx;
  const DiscDomain* disc;  // Impl::disc_ (NON possede ; adresse stable)
  bool recon_prim;
  Real pos_floor = Real(0);  ///< limiteur de positivite Zhang-Shu (<= 0 : inactif, bit-identique)
  void operator()(MultiFab& U, MultiFab& R) const {
    fill_ghosts(U, ctx->dom, ctx->bc);
    assemble_rhs_eb<Limiter, Flux>(model, U, *ctx->aux, disc_level_set(*disc), ctx->geom, R,
                                   recon_prim, kEbKappaMin, pos_floor);
  }
};

/// Avance EXPLICITE MASQUEE : n sous-pas du stepper @c Stepper sur le residu de transport MASQUE.
/// Mime AdvanceExplicit a l'identique (meme math RK, meme limiteur / flux) : seul le residu change.
template <class Limiter, class Flux, class Model, class Stepper = SSPRK2Step>
struct AdvanceExplicitMasked {
  Model m;
  GridContext ctx;
  const MultiFab* mask;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< limiteur de positivite Zhang-Shu (<= 0 : inactif, bit-identique)
  void operator()(MultiFab& U, Real dt, int n) const {
    const Real h = dt / static_cast<Real>(n);
    const BlockRhsEvalMasked<Limiter, Flux, Model> rhs{m, &ctx, mask, recon_prim, pos_floor};
    for (int s = 0; s < n; ++s) Stepper{}.take_step(rhs, U, h);
  }
};

/// Avance EXPLICITE CUT-CELL / EB : n sous-pas du stepper @c Stepper sur le residu de transport EB.
/// Mime AdvanceExplicit a l'identique : seul le residu (assemble_rhs_eb) change.
template <class Limiter, class Flux, class Model, class Stepper = SSPRK2Step>
struct AdvanceExplicitEb {
  Model m;
  GridContext ctx;
  const DiscDomain* disc;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< limiteur de positivite Zhang-Shu (<= 0 : inactif, bit-identique)
  void operator()(MultiFab& U, Real dt, int n) const {
    const Real h = dt / static_cast<Real>(n);
    const BlockRhsEvalEb<Limiter, Flux, Model> rhs{m, &ctx, disc, recon_prim, pos_floor};
    for (int s = 0; s < n; ++s) Stepper{}.take_step(rhs, U, h);
  }
};

/// Avance IMEX MASQUEE : demi-pas EXPLICITE MASQUE (transport sans source) + source IMPLICITE raide.
/// Mime AdvanceImex : le transport (forward-Euler) lit le residu MASQUE sans source ; la source
/// implicite (backward_euler_source) est INCHANGEE (locale par cellule, hors frontiere disque). Une
/// cellule inactive a un residu de transport nul puis subit la source locale -- comme T2 / EB, seule
/// la FRONTIERE de transport est fermee, la source reste cellule-locale.
template <class Limiter, class Flux, class Model>
struct AdvanceImexMasked {
  Model m;
  GridContext ctx;
  const MultiFab* mask;
  bool recon_prim;
  ImplicitMask<Model::n_vars> mask_impl{};
  NewtonOptions nopts{};
  NewtonReport* nreport = nullptr;
  Real pos_floor = Real(0);  ///< limiteur de positivite Zhang-Shu (<= 0 : inactif)
  void operator()(MultiFab& U, Real dt, int n) const {
    const Real h = dt / static_cast<Real>(n);
    const BlockRhsEvalMasked<Limiter, Flux, SourceFreeModel<Model>> rhs{
        SourceFreeModel<Model>{m}, &ctx, mask, recon_prim, pos_floor};
    if (nreport) nreport->reset();
    for (int s = 0; s < n; ++s) {
      ForwardEuler{}.take_step(rhs, U, h);
      backward_euler_source(m, *ctx.aux, U, h, nopts, mask_impl, nreport);
    }
  }
};

/// Avance IMEX CUT-CELL / EB : demi-pas EXPLICITE EB (transport sans source) + source IMPLICITE raide.
/// Mime AdvanceImex : transport via assemble_rhs_eb, source implicite inchangee (cellule-locale).
template <class Limiter, class Flux, class Model>
struct AdvanceImexEb {
  Model m;
  GridContext ctx;
  const DiscDomain* disc;
  bool recon_prim;
  ImplicitMask<Model::n_vars> mask_impl{};
  NewtonOptions nopts{};
  NewtonReport* nreport = nullptr;
  Real pos_floor = Real(0);  ///< limiteur de positivite Zhang-Shu (<= 0 : inactif)
  void operator()(MultiFab& U, Real dt, int n) const {
    const Real h = dt / static_cast<Real>(n);
    const BlockRhsEvalEb<Limiter, Flux, SourceFreeModel<Model>> rhs{SourceFreeModel<Model>{m}, &ctx,
                                                                    disc, recon_prim, pos_floor};
    if (nreport) nreport->reset();
    for (int s = 0; s < n; ++s) {
      ForwardEuler{}.take_step(rhs, U, h);
      backward_euler_source(m, *ctx.aux, U, h, nopts, mask_impl, nreport);
    }
  }
};
}  // namespace detail

/// Construit le masque implicite POD device-clean d'un modele a N variables a partir d'une liste
/// d'indices de composantes (vide -> masque INACTIF -> defaut modele, bit-identique). Tout indice
/// hors [0, N) est ignore ici (la validation/le message clair vit cote System::add_block, qui resout
/// les noms/roles en indices et leve sur un nom/role absent).
template <int N>
ImplicitMask<N> make_implicit_mask(const std::vector<int>& implicit_components) {
  ImplicitMask<N> mask;
  if (implicit_components.empty()) return mask;  // inactif : defaut modele
  mask.active = true;
  for (int c : implicit_components)
    if (c >= 0 && c < N) mask.flag[c] = true;
  return mask;
}

/// Fermetures (avance + residu) pour un schema spatial (Limiter x Flux) fige. La math RK vient des
/// TimeStepper du coeur : en explicite SSPRK2 (defaut), SSPRK3 ou ForwardEuler ("euler", ordre 1,
/// fidelite aux references premier ordre -- validation, jamais defaut) selon @p method ; ForwardEuler +
/// backward_euler_source en IMEX. Les fermetures sont des FONCTEURS NOMMES (cf. namespace detail) et
/// non des lambdas : le chemin add_compiled_model (premiere instanciation depuis une TU externe)
/// s'emet alors proprement sous nvcc. @p method ne joue QUE sur l'avance explicite (l'IMEX garde son
/// demi-pas ForwardEuler + source implicite) ; "ssprk2" reproduit l'avance historique (bit-identique).
/// En IMEX (@p imex), @p method "imexrk_ars222" selectionne la famille IMEX-RK ARS(2,2,2) (ordre 2,
/// avance PARALLELE a AdvanceImex, cartesien plein seul) ; toute autre valeur garde l'IMEX historique
/// backward-Euler (ordre 1, bit-identique).
/// @p implicit_components : indices des variables conservees a traiter en IMPLICITE dans la source IMEX
/// (masque PORTE PAR LE BLOC, override du defaut modele). VIDE (defaut) -> masque inactif -> defaut
/// modele is_implicit -> bit-identique. Sans effet hors IMEX (l'explicite n'a pas de pas implicite).
/// Les avances de transport DISQUE optionnelles (advance_masked / advance_eb) sont fabriquees quand
/// @p ctx porte la geometrie disque du System (ctx.disc_mask / ctx.disc, chantier T5-PR3) ; sinon elles
/// restent vides et le stepper retombe sur advance (bit-identique). Adresses STABLES de membres de Impl,
/// lues PAR POINTEUR au pas -> l'ordre add_block / set_disc_domain est indifferent. Les avances disque
/// MIMENT advance (meme RK / IMEX, meme limiteur / flux) ; seul le residu de transport est aiguille
/// (assemble_rhs_masked / _eb).
template <class Limiter, class Flux, class Model>
BlockClosures build_block(const Model& m, const GridContext& ctx, bool imex, bool recon_prim,
                          const std::string& method = "ssprk2",
                          const std::vector<int>& implicit_components = {},
                          const NewtonOptions& newton_opts = {},
                          NewtonReport* newton_report = nullptr, Real pos_floor = Real(0),
                          bool wave_speed_cache = false) {
  const MultiFab* disc_mask = ctx.disc_mask;
  const detail::DiscDomain* disc = ctx.disc;
  BlockClosures bc;
  const ImplicitMask<Model::n_vars> impl_mask = make_implicit_mask<Model::n_vars>(implicit_components);
  // Scratch PARTAGE du cache de vitesses d'onde HLL (opt-in) : un seul MultiFab pour l'avance explicite
  // et rhs_into (jamais appelees en concurrence). nullptr quand l'option est OFF -> BlockRhsEval garde
  // le chemin par face (bit-identique). Alloue au layout reel au premier appel (cf. BlockRhsEval).
  std::shared_ptr<MultiFab> ws_cache =
      wave_speed_cache ? std::make_shared<MultiFab>() : std::shared_ptr<MultiFab>{};
  if (imex) {
    if (method == "imexrk_ars222") {
      // FAMILLE IMEX-RK, schema ARS(2,2,2) (ordre 2) : avance PARALLELE a AdvanceImex, source PLEINEMENT
      // implicite (impl_mask ignore : la facade rejette deja un masque partiel avec ce schema). CARTESIEN
      // PLEIN UNIQUEMENT : on ne fabrique PAS d'avance disque (advance_masked / advance_eb restent vides)
      // -> un mode geometrie disque sur ce bloc leve une erreur EXPLICITE au pas
      // (SystemStepper::advance_transport_n), jamais un cartesien silencieux.
      bc.advance = detail::AdvanceImexRkArs222<Limiter, Flux, Model>{m, ctx, recon_prim, newton_opts,
                                                                     newton_report, pos_floor};
    } else {
      // IMEX historique (backward-Euler local, ordre 1) : INTOUCHE, bit-identique.
      bc.advance = detail::AdvanceImex<Limiter, Flux, Model>{m, ctx, recon_prim, impl_mask,
                                                             newton_opts, newton_report, pos_floor};
      if (disc_mask)
        bc.advance_masked = detail::AdvanceImexMasked<Limiter, Flux, Model>{
            m, ctx, disc_mask, recon_prim, impl_mask, newton_opts, newton_report, pos_floor};
      if (disc)
        bc.advance_eb = detail::AdvanceImexEb<Limiter, Flux, Model>{
            m, ctx, disc, recon_prim, impl_mask, newton_opts, newton_report, pos_floor};
    }
  } else if (method == "euler") {
    bc.advance = detail::AdvanceExplicit<Limiter, Flux, Model, ForwardEuler>{m, ctx, recon_prim,
                                                                             pos_floor, ws_cache};
    if (disc_mask)
      bc.advance_masked = detail::AdvanceExplicitMasked<Limiter, Flux, Model, ForwardEuler>{
          m, ctx, disc_mask, recon_prim, pos_floor};
    if (disc)
      bc.advance_eb = detail::AdvanceExplicitEb<Limiter, Flux, Model, ForwardEuler>{
          m, ctx, disc, recon_prim, pos_floor};
  } else if (method == "ssprk3") {
    bc.advance = detail::AdvanceExplicit<Limiter, Flux, Model, SSPRK3Step>{m, ctx, recon_prim,
                                                                           pos_floor, ws_cache};
    if (disc_mask)
      bc.advance_masked = detail::AdvanceExplicitMasked<Limiter, Flux, Model, SSPRK3Step>{
          m, ctx, disc_mask, recon_prim, pos_floor};
    if (disc)
      bc.advance_eb = detail::AdvanceExplicitEb<Limiter, Flux, Model, SSPRK3Step>{
          m, ctx, disc, recon_prim, pos_floor};
  } else if (method == "ssprk2") {
    bc.advance = detail::AdvanceExplicit<Limiter, Flux, Model, SSPRK2Step>{m, ctx, recon_prim,
                                                                           pos_floor, ws_cache};
    if (disc_mask)
      bc.advance_masked = detail::AdvanceExplicitMasked<Limiter, Flux, Model, SSPRK2Step>{
          m, ctx, disc_mask, recon_prim, pos_floor};
    if (disc)
      bc.advance_eb = detail::AdvanceExplicitEb<Limiter, Flux, Model, SSPRK2Step>{
          m, ctx, disc, recon_prim, pos_floor};
  } else {
    throw std::runtime_error("System : methode temporelle explicite inconnue '" + method +
                             "' (euler|ssprk2|ssprk3)");
  }
  bc.rhs_into = detail::RhsInto<Limiter, Flux, Model>{m, ctx, recon_prim, pos_floor, ws_cache};
  bc.hotspot = detail::HotspotFn<Model>{m, ctx};  // diagnostic dt_hotspot (ADC-182), hors chemin chaud
  return bc;
}

/// Dispatch du schema spatial (limiteur x flux Riemann) -> fermetures compilees. HLLC / Roe gardes
/// par requires : exigent un transport a 4 variables exposant pressure (sinon erreur explicite).
/// "weno5" = reconstruction WENO5-Z (ordre 5, stencil 5 points, 3 ghosts) ; spatial_operator route
/// sur weno5z quand Limiter::n_ghost >= 3 (l'appelant doit allouer 3 ghosts, cf. block_n_ghost).
/// @p method choisit l'avance EXPLICITE (ssprk2 par defaut, ssprk3 | euler optionnels) ; sans effet en IMEX.
/// @p implicit_components : masque implicite IMEX porte par le bloc (indices ; vide = defaut modele,
/// bit-identique). cf. build_block.
template <class Model>
BlockClosures make_block(const Model& m, const std::string& lim, const std::string& riem,
                         const GridContext& ctx, bool imex, bool recon_prim,
                         const std::string& method = "ssprk2",
                         const std::vector<int>& implicit_components = {},
                         const NewtonOptions& newton_opts = {},
                         NewtonReport* newton_report = nullptr, Real pos_floor = Real(0),
                         bool wave_speed_cache = false) {
  // VALIDATION CENTRALISEE (registry dispatch_tags.hpp) AVANT le dispatch : memes acceptations /
  // rejets de tags qu'avant, messages identiques (validate_* reprend la formulation historique). Le
  // dispatch if/else qui suit est INCHANGE (Limiter / Flux sont des types compile-time) ; ses throws
  // finaux "limiter/flux inconnu" deviennent inatteignables -> remplaces par une garde d'incoherence
  // registry/dispatch. Les gardes de CAPABILITE (hll/hllc/roe sur un modele sans onde / sans pression)
  // restent des `if constexpr` PAR MODELE ci-dessous, avec leurs messages "exige ..." inchanges.
  validate_riemann(riem, /*polar=*/false, "System");
  validate_limiter(lim, "System");
  if (riem == "rusanov") {
    if (lim == "none") return build_block<NoSlope, RusanovFlux>(m, ctx, imex, recon_prim, method, implicit_components, newton_opts, newton_report, pos_floor);
    if (lim == "minmod") return build_block<Minmod, RusanovFlux>(m, ctx, imex, recon_prim, method, implicit_components, newton_opts, newton_report, pos_floor);
    if (lim == "vanleer") return build_block<VanLeer, RusanovFlux>(m, ctx, imex, recon_prim, method, implicit_components, newton_opts, newton_report, pos_floor);
    if (lim == "weno5") return build_block<Weno5, RusanovFlux>(m, ctx, imex, recon_prim, method, implicit_components, newton_opts, newton_report, pos_floor);
    throw_registry_dispatch_mismatch("System", "limiteur", lim);
  }
  if (riem == "hll") {
    // HLL (Harten-Lax-van Leer, 2 ondes) : moins diffusif que Rusanov (dissipation ~ |sR-sL| signee au
    // lieu de 2*max|v| symetrique), mais ne demande PAS de pression (contrairement a HLLC/Roe) -- seulement
    // des vitesses d'onde SIGNEES model.wave_speeds. Disponible des qu'un modele expose ses valeurs propres
    // signees (le DSL emet wave_speeds des qu'une primitive 'p' est declaree, meme isotherme froid p=0 ->
    // c=0 -> HLL degenere en upwind, toujours moins diffusif que Rusanov au contact). N'EXIGE PAS n_vars==4
    // ni une pression : utilisable par le modele isotherme 3-var (rho, m_x, m_y) du diocotron Hoffart, la
    // ou hllc/roe sont rejetes. Gate sur la presence de wave_speeds (sinon erreur CLAIRE, pas un echec de
    // compilation pour un modele scalaire sans onde signee, p.ex. transport ExB).
    if constexpr (requires(const Model mm, typename Model::State s, Aux a, Real r) {
                    mm.wave_speeds(s, a, 0, r, r);
                  }) {
      // wave_speed_cache (opt-in) forwarde UNIQUEMENT ici : le cache de vitesses d'onde ne s'engage que
      // pour le flux HLL (BlockRhsEval garde par Flux == HLLFlux). rusanov/hllc/roe l'ignorent.
      if (lim == "none") return build_block<NoSlope, HLLFlux>(m, ctx, imex, recon_prim, method, implicit_components, newton_opts, newton_report, pos_floor, wave_speed_cache);
      if (lim == "minmod") return build_block<Minmod, HLLFlux>(m, ctx, imex, recon_prim, method, implicit_components, newton_opts, newton_report, pos_floor, wave_speed_cache);
      if (lim == "vanleer") return build_block<VanLeer, HLLFlux>(m, ctx, imex, recon_prim, method, implicit_components, newton_opts, newton_report, pos_floor, wave_speed_cache);
      if (lim == "weno5") return build_block<Weno5, HLLFlux>(m, ctx, imex, recon_prim, method, implicit_components, newton_opts, newton_report, pos_floor, wave_speed_cache);
      throw_registry_dispatch_mismatch("System", "limiteur", lim);
    } else {
      throw std::runtime_error("System : flux 'hll' exige des vitesses d'onde signees "
                               "(model.wave_speeds : declarer une primitive 'p' / des eigenvalues) ; "
                               "ce transport -> 'rusanov'");
    }
  }
  if (riem == "hllc") {
    // CHEMINS HLLC : (a) capability HasHLLCStructure (le modele fournit contact_speed +
    // hllc_star_state -> algorithme contact-resolving GENERIQUE, aucun layout assume), OU
    // (b) chemin CANONIQUE Euler 2D (n_vars == 4 + pressure, implementation historique
    // bit-identique). Sans l'un des deux, rejet explicite avec le remede capability.
    if constexpr (HasHLLCStructure<Model> ||
                  (Model::n_vars == 4 &&
                   requires(const Model mm, typename Model::State s) { mm.pressure(s); })) {
      if (lim == "none") return build_block<NoSlope, HLLCFlux>(m, ctx, imex, recon_prim, method, implicit_components, newton_opts, newton_report, pos_floor);
      if (lim == "minmod") return build_block<Minmod, HLLCFlux>(m, ctx, imex, recon_prim, method, implicit_components, newton_opts, newton_report, pos_floor);
      if (lim == "vanleer") return build_block<VanLeer, HLLCFlux>(m, ctx, imex, recon_prim, method, implicit_components, newton_opts, newton_report, pos_floor);
      if (lim == "weno5") return build_block<Weno5, HLLCFlux>(m, ctx, imex, recon_prim, method, implicit_components, newton_opts, newton_report, pos_floor);
      throw_registry_dispatch_mismatch("System", "limiteur", lim);
    } else {
      throw std::runtime_error("System : flux 'hllc' exige un transport compressible Euler 2D "
                               "(4 variables + pression) OU la capability HLLC du modele "
                               "(pressure + wave_speeds + contact_speed + hllc_star_state, cf. "
                               "HasHLLCStructure) ; ce transport -> 'hll'/'rusanov'");
    }
  }
  if (riem == "roe") {
    // CHEMINS ROE : (a) capability HasRoeDissipation (le modele fournit sa dissipation de Roe
    // complete d = |A_roe| dU -> solveur Roe-like GENERIQUE), OU (b) chemin CANONIQUE Euler 2D
    // gaz parfait (historique bit-identique). Sans l'un des deux, rejet explicite.
    if constexpr (HasRoeDissipation<Model> ||
                  (Model::n_vars == 4 &&
                   requires(const Model mm, typename Model::State s) { mm.pressure(s); })) {
      if (lim == "none") return build_block<NoSlope, RoeFlux>(m, ctx, imex, recon_prim, method, implicit_components, newton_opts, newton_report, pos_floor);
      if (lim == "minmod") return build_block<Minmod, RoeFlux>(m, ctx, imex, recon_prim, method, implicit_components, newton_opts, newton_report, pos_floor);
      if (lim == "vanleer") return build_block<VanLeer, RoeFlux>(m, ctx, imex, recon_prim, method, implicit_components, newton_opts, newton_report, pos_floor);
      if (lim == "weno5") return build_block<Weno5, RoeFlux>(m, ctx, imex, recon_prim, method, implicit_components, newton_opts, newton_report, pos_floor);
      throw_registry_dispatch_mismatch("System", "limiteur", lim);
    } else {
      throw std::runtime_error("System : flux 'roe' exige un transport compressible Euler 2D "
                               "(4 variables + pression) OU la capability Roe du modele "
                               "(roe_dissipation, cf. HasRoeDissipation) ; ce transport -> "
                               "'hll'/'rusanov'");
    }
  }
  throw_registry_dispatch_mismatch("System", "flux", riem);
}

/// Nombre de ghosts requis par le schema spatial @p lim (source unique : Limiter::n_ghost). Sert a
/// l'allocation du MultiFab d'etat d'un bloc, pour que le stencil large de WENO5 (5 points, 3 ghosts)
/// ne lise pas hors bornes -- cf. comment AmrSystem alloue avec Limiter::n_ghost (PR #22). Defaut 2
/// (MUSCL) pour un limiteur inconnu : c'est l'allocation historique, donc bit-identique.
inline int block_n_ghost(const std::string& lim) {
  // Source UNIQUE : limiter_n_ghost(lim) (registry dispatch_tags.hpp). Le defaut 2 (MUSCL) pour un
  // limiteur inconnu est porte par le registry -> meme allocation historique, bit-identique. Les
  // static_assert ci-dessous (cette TU voit ET le registry ET les types) garantissent que la table
  // kLimiters ne derive jamais des constantes ::n_ghost reelles.
  static_assert(limiter_n_ghost_ct("none") == NoSlope::n_ghost, "kLimiters[none].n_ghost derive");
  static_assert(limiter_n_ghost_ct("minmod") == Minmod::n_ghost, "kLimiters[minmod].n_ghost derive");
  static_assert(limiter_n_ghost_ct("vanleer") == VanLeer::n_ghost, "kLimiters[vanleer].n_ghost derive");
  static_assert(limiter_n_ghost_ct("weno5") == Weno5::n_ghost, "kLimiters[weno5].n_ghost derive");
  return limiter_n_ghost(lim);
}

namespace detail {
/// Foncteur vitesse d'onde max du bloc (max_wave_speed_mf, reduction par le seam). FONCTEUR NOMME :
/// max_wave_speed_mf instancie MaxWaveSpeedKernel (deja un foncteur device) ; l'envelopper dans une
/// classe nommee plutot qu'une lambda preserve le contexte d'instanciation cross-TU sous nvcc.
template <class Model>
struct MaxSpeed {
  Model m;
  GridContext ctx;
  Real operator()(const MultiFab& U) const { return max_wave_speed_mf(m, U, *ctx.aux); }
};


/// Foncteur vitesse de STABILITE max du bloc (trait HasStabilitySpeed) : remplace MaxSpeed dans la
/// CFL quand le modele declare stability_speed (les solveurs de Riemann gardent max_wave_speed).
template <class Model>
struct MaxStabilitySpeed {
  Model m;
  GridContext ctx;
  Real operator()(const MultiFab& U) const { return max_stability_speed_mf(m, U, *ctx.aux); }
};

/// Foncteur frequence de source max du bloc (trait HasSourceFrequency, borne dt <= cfl/mu sans h).
template <class Model>
struct MaxSourceFreq {
  Model m;
  GridContext ctx;
  Real operator()(const MultiFab& U) const { return max_source_frequency_mf(m, U, *ctx.aux); }
};

/// Foncteur pas admissible min du bloc (trait HasStabilityDt ; 0 = aucune cellule ne contraint).
template <class Model>
struct MinStabilityDt {
  Model m;
  GridContext ctx;
  Real operator()(const MultiFab& U) const { return min_stability_dt_mf(m, U, *ctx.aux); }
};

/// Foncteur contribution Poisson : rhs += elliptic_rhs(U) (boucle HOTE pure, pas de kernel device).
template <class Model>
struct PoissonRhs {
  Model m;
  void operator()(const MultiFab& U, MultiFab& rhs) const {
    for (int li = 0; li < rhs.local_size(); ++li) {
      Array4 r = rhs.fab(li).array();
      const ConstArray4 u = U.fab(li).const_array();
      const Box2D b = rhs.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          r(i, j) += m.elliptic_rhs(load_state<Model>(u, i, j));
    }
  }
};
}  // namespace detail

/// Fermeture de la vitesse utilisee par le pas CFL du bloc. Si le modele declare le trait OPTIONNEL
/// stability_speed (HasStabilitySpeed), c'est ELLE qui pilote la CFL (lambda* de stabilite) ; sinon
/// fallback STRICT sur max_wave_speed (comportement historique, bit-identique). Les solveurs de
/// Riemann lisent toujours max_wave_speed : ce choix ne change que la politique de pas.
template <class Model>
std::function<Real(const MultiFab&)> make_max_speed(const Model& m, const GridContext& ctx) {
  if constexpr (HasStabilitySpeed<Model>)
    return detail::MaxStabilitySpeed<Model>{m, ctx};
  else
    return detail::MaxSpeed<Model>{m, ctx};
}

/// Fermeture de la frequence de source max du bloc (borne dt <= cfl * substeps / (stride * mu)).
/// VIDE (std::function nulle) si le modele ne declare pas le trait -> le stepper l'ignore
/// (comportement historique).
template <class Model>
std::function<Real(const MultiFab&)> make_source_frequency(const Model& m, const GridContext& ctx) {
  if constexpr (HasSourceFrequency<Model>)
    return detail::MaxSourceFreq<Model>{m, ctx};
  else
    return {};
}

/// Fermeture du pas admissible min du bloc (borne dt <= stability_dt * substeps / stride, SANS
/// cfl). VIDE si le modele ne declare pas le trait -> ignoree par le stepper (historique).
template <class Model>
std::function<Real(const MultiFab&)> make_stability_dt(const Model& m, const GridContext& ctx) {
  if constexpr (HasStabilityDt<Model>)
    return detail::MinStabilityDt<Model>{m, ctx};
  else
    return {};
}

/// Contribution du bloc au second membre de Poisson : rhs += elliptic_rhs(U) (boucle hote).
template <class Model>
std::function<void(const MultiFab&, MultiFab&)> make_poisson_rhs(const Model& m) {
  return detail::PoissonRhs<Model>{m};
}

/// Conversions PONCTUELLES (une cellule) cons <-> prim du MODELE, type-erasees sur des tableaux de
/// Model::n_vars doubles. Premiere = primitif -> conservatif (M.to_conservative, init depuis les
/// primitives), seconde = conservatif -> primitif (M.to_primitive, diagnostic). Capture le modele par
/// valeur (fige a l'ajout du bloc). Pour un modele SANS conversion (scalaire pur, pas de brique
/// hyperbolique) les deux sont l'IDENTITE -- exact pour un transport scalaire (prim == cons).
/// Model::Prim partage la largeur Model::n_vars de State (contrat HyperbolicPhysicalModel), donc les
/// tableaux plats s'alignent composante a composante. Partage par add_block (natif) et
/// add_compiled_model (compile) : la MEME conversion sert les deux chemins.
template <class Model>
std::pair<std::function<void(const double*, double*)>,
          std::function<void(const double*, double*)>>
make_cell_convert(const Model& m) {
  constexpr int NV = Model::n_vars;
  if constexpr (HasPrimitiveVars<Model>) {
    auto p2c = [m](const double* in, double* out) {
      typename Model::Prim p{};
      for (int c = 0; c < NV; ++c) p[c] = static_cast<Real>(in[c]);
      const typename Model::State u = m.to_conservative(p);
      for (int c = 0; c < NV; ++c) out[c] = static_cast<double>(u[c]);
    };
    auto c2p = [m](const double* in, double* out) {
      typename Model::State u{};
      for (int c = 0; c < NV; ++c) u[c] = static_cast<Real>(in[c]);
      const typename Model::Prim p = m.to_primitive(u);
      for (int c = 0; c < NV; ++c) out[c] = static_cast<double>(p[c]);
    };
    return {std::function<void(const double*, double*)>(p2c),
            std::function<void(const double*, double*)>(c2p)};
  } else {
    auto id = [](const double* in, double* out) {
      for (int c = 0; c < NV; ++c) out[c] = in[c];
    };
    return {std::function<void(const double*, double*)>(id),
            std::function<void(const double*, double*)>(id)};
  }
}

}  // namespace adc
