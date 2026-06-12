/// @file
/// @brief Operateur spatial cartesien : assemble R(U, aux) = -div F + S sur les cellules d'un niveau.
///
/// C'est la fleche "PDE -> systeme d'ODE" de la methode des lignes. L'integrateur en temps
/// (time/) ne connait que R ; il ignore tout de la geometrie et du schema de reconstruction.
///
/// Fonctions et types exposes :
///   - DiffusiveModel        : concept optionnel (model.diffusivity() -> nu) ; le flux Fickien
///                             F = -nu grad U est ajoute au flux hyperbolique si presente.
///   - SourceFreeModel<M>    : adaptateur qui zero la source (demi-pas explicite IMEX).
///   - load_state<Model>     : lecture de l'etat conserve depuis un Array4 (ADC_HD).
///   - load_aux<NComp>       : lecture de l'auxiliaire (phi, grad, champs extra) (ADC_HD).
///   - max_wave_speed_mf     : max CFL sur toute la MultiFab (reduction + all_reduce MPI).
///   - rusanov_flux          : compat libre, delegue a RusanovFlux{}.
///   - reconstruct<>         : valeur de face depuis le stencil MUSCL ou WENO5 (ADC_HD).
///   - compute_face_fluxes<> : flux aux faces (pour reflux AMR).
///   - assemble_rhs<>        : residu R = -div Fhat + S (+ diffusion) sur la boite.
///
/// INVARIANT : l'operateur cartesien est STRICTEMENT INTOUCHE par l'operateur polaire
/// (spatial_operator_polar.hpp) ; un run sur maillage cartesien est bit-identique.

#pragma once

#include <adc/core/state.hpp>
#include <adc/core/physical_model.hpp>  // HasPrimitiveVars : reconstruction primitive optionnelle
#include <adc/core/types.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/numerics/numerical_flux.hpp>
#include <adc/numerics/reconstruction.hpp>

#include <algorithm>
#include <concepts>
#include <stdexcept>  // positivity_comp : modele sans role Density -> erreur claire

// Operateur spatial : assemble le residu R(U, aux) = -div F(U, aux) + S(U, aux)
// sur les cellules valides d'un niveau. Fleche "PDE -> systeme d'ODE" de la
// methode des lignes ; l'integrateur ne connait que R.
//
// Flux numerique de Rusanov (local Lax-Friedrichs) sur des etats reconstruits :
//   Fhat = 1/2 (F(UL) + F(UR)) - 1/2 alpha (UR - UL),  alpha = max vitesse d'onde
//
// La reconstruction est un parametre de template (limiteur) :
//   - NoSlope  : premier ordre (UL, UR = valeurs de cellule), 1 ghost
//   - Minmod / VanLeer : MUSCL ordre 2, pente limitee par composante, 2 ghosts
//
// La physique entre uniquement par le PhysicalModel (flux, max_wave_speed,
// source). aux (phi, grad phi) n'est PAS reconstruit (champ lisse issu de
// l'elliptique) : on prend la valeur de cellule de chaque cote.
// Convention aux : composantes [0]=phi, [1]=d phi/dx, [2]=d phi/dy.

namespace adc {

// aux_comps<Model>() (largeur du canal aux d'un modele) vit desormais dans le header contrat
// adc/core/physical_model.hpp (inclus ci-dessus) pour que CompositeModel puisse le propager.

/// DiffusiveModel : concept optionnel pour les modeles avec diffusion scalaire isotrope.
///
/// Un modele satisfait DiffusiveModel si et seulement si m.diffusivity() -> Real (nu >= 0).
/// Le flux Fickien F = -nu grad U est AJOUTE au flux hyperbolique dans assemble_rhs /
/// compute_face_fluxes. La divergence donne exactement +nu Lap(U).
/// INVARIANT : un modele sans diffusivity() ne change pas d'un bit (chemin hyperbolique
/// strictement inchange -- le if constexpr est faux, zero codegen supplementaire).
// Modele DIFFUSIF (optionnel) : fournit une diffusivite scalaire isotrope nu. Le
// tuteur : "la diffusion, c'est comme un flux de plus". Le flux Fickien F = -nu grad U
// ajoute au flux hyperbolique donne, apres divergence (-div F), exactement +nu Lap(U).
// On l'implemente comme un terme additif au residu, GARDE par ce trait : un modele
// sans diffusivity() ne change pas d'un bit (chemin hyperbolique inchange).
template <class M>
concept DiffusiveModel = requires(const M m) {
  { m.diffusivity() } -> std::convertible_to<Real>;
};

/// SourceFreeModel<M> : adaptateur qui annule la source de M (demi-pas explicite IMEX).
///
/// Meme flux et max_wave_speed que M, mais source() renvoie toujours l'etat nul.
/// Sert au demi-pas EXPLICITE d'un schema IMEX (transport seul, -div F) ; la source raide
/// est traitee implicitement par backward_euler_source. N'expose pas diffusivity() pour ne
/// pas casser les modeles non diffusifs. Transparent au contrat HLL/HLLC : forwarde
/// pressure et wave_speeds uniquement si M les expose (clause requires).
// Modele "sans source" : meme flux et vitesse d'onde que M, mais source nulle. Sert au
// demi-pas EXPLICITE d'un schema IMEX (transport seul, -div F), la source raide etant
// traitee implicitement a part (backward_euler_source). Note : n'expose pas diffusivity()
// (le forwarder inconditionnellement casserait les modeles non diffusifs) -> un bloc IMEX
// diffusif perdrait son flux Fickien dans le demi-pas explicite ; raffinement a part.
template <class M>
struct SourceFreeModel {
  using State = typename M::State;
  using Aux = typename M::Aux;
  static constexpr int n_vars = M::n_vars;
  static constexpr int n_aux = aux_comps<M>();  // transparent a la largeur aux du modele enveloppe
  M m;
  ADC_HD State flux(const State& u, const Aux& a, int dir) const { return m.flux(u, a, dir); }
  ADC_HD Real max_wave_speed(const State& u, const Aux& a, int dir) const {
    return m.max_wave_speed(u, a, dir);
  }
  ADC_HD State source(const State&, const Aux&) const { return State{}; }
  ADC_HD Real elliptic_rhs(const State& u) const { return m.elliptic_rhs(u); }
  // SourceFreeModel n'expose pas les variables primitives : le demi-pas explicite IMEX qui
  // l'utilise reconstruit donc en conservatif (le chemin explicite direct, lui, dispose des
  // conversions du modele compose et peut reconstruire en primitif).
  // Transparent au contrat HLL/HLLC : ne forwarde pression et vitesses signees QUE si M
  // les expose (clause requires), pour qu'un demi-pas IMEX puisse rester en flux HLLC.
  ADC_HD Real pressure(const State& u) const
    requires requires(const M& mm, const State& s) { mm.pressure(s); }
  {
    return m.pressure(u);
  }
  ADC_HD void wave_speeds(const State& u, const Aux& a, int dir, Real& smin, Real& smax) const
    requires requires(const M& mm, const State& s, const Aux& aa, int d, Real& lo, Real& hi) {
      mm.wave_speeds(s, aa, d, lo, hi);
    }
  {
    m.wave_speeds(u, a, dir, smin, smax);
  }
  // Forward de l'introspection VariableSet (HOTE) : laisse positivity_comp resoudre le role Density
  // a travers le demi-pas explicite IMEX. Conditionnel (requires), comme pressure / wave_speeds.
  static VariableSet conservative_vars()
    requires requires { M::conservative_vars(); }
  {
    return M::conservative_vars();
  }
};

/// load_state<Model> : lit Model::n_vars scalaires a (i,j) depuis un Array4.
///
/// Retourne un StateVec<n_vars> initialise depuis les composantes 0..n_vars-1 du canal.
/// ADC_HD, zero allocation. Ne lit PAS les composantes au-dela de n_vars.
template <class Model>
ADC_HD inline typename Model::State load_state(const ConstArray4& a, int i,
                                              int j) {
  typename Model::State u;
  for (int c = 0; c < Model::n_vars; ++c) u[c] = a(i, j, c);
  return u;
}

/// load_aux<NComp> : lit NComp composantes de l'auxiliaire depuis un Array4 en (i,j).
///
/// Les 3 premieres composantes (phi, grad_x, grad_y) sont le contrat de base.
/// Les composantes >= 3 (B_z, T_e...) sont lues seulement si NComp > leur indice canonique
/// (guard if constexpr -> zero codegen pour NComp = kAuxBaseComps = 3 : bit-identique).
/// Les champs extra sont gouvernes par ADC_AUX_FIELDS (state.hpp) : ajouter un champ =>
/// 1 ligne dans ADC_AUX_FIELDS, pas dans ce chemin. ADC_HD.
// Charge l'auxiliaire de cellule depuis le canal aux. NComp = nombre de composantes lues
// (cf. aux_comps<Model>()). Les trois premieres sont toujours phi/grad_x/grad_y ; les
// suivantes, optionnelles, alimentent les champs extra de Aux dans l'ordre canonique.
// NComp = kAuxBaseComps (defaut) reproduit a l'identique l'ancien comportement : les
// champs extra restent a 0 et aucune composante au-dela de 2 n'est touchee.
//
// Les champs extra sont charges depuis la SOURCE UNIQUE ADC_AUX_FIELDS (state.hpp) : chaque
// X(name, idx) genere `if constexpr (NComp > idx) x.name = a(i,j,idx);`, exactement la
// sequence ecrite a la main auparavant. Ajouter un champ extra => 1 ligne dans ADC_AUX_FIELDS
// suffit pour que ce chemin de lecture device le couvre (et le marshaling hote, genere de la
// meme table). NComp = kAuxBaseComps : toutes les gardes sont fausses -> bit-identique.
template <int NComp = kAuxBaseComps>
ADC_HD inline Aux load_aux(const ConstArray4& a, int i, int j) {
  Aux x{a(i, j, 0), a(i, j, 1), a(i, j, 2)};
#define ADC_AUX_LOAD(name, idx) \
  if constexpr (NComp > (idx)) x.name = a(i, j, idx);
  ADC_AUX_FIELDS(ADC_AUX_LOAD)
#undef ADC_AUX_LOAD
  // Champs aux NOMMES (ADC-70 phase 1) : composantes a partir de kAuxNamedBase (= 5). Charges
  // SEULEMENT si le modele declare n_aux > kAuxNamedBase (sinon if constexpr faux -> aucun codegen,
  // NComp = kAuxBaseComps reste strictement bit-identique). La borne n_extra est connue a la
  // compilation (NComp template) : la boucle est deroulee et clampee a kAuxMaxExtra (taille de
  // x.extra) -- jamais d'acces hors du tableau C, device-clean.
  if constexpr (NComp > kAuxNamedBase) {
    constexpr int n_extra =
        (NComp - kAuxNamedBase) < kAuxMaxExtra ? (NComp - kAuxNamedBase) : kAuxMaxExtra;
    for (int k = 0; k < n_extra; ++k) x.extra[k] = a(i, j, kAuxNamedBase + k);
  }
  return x;
}

namespace detail {
// Noyau reducteur de la vitesse d'onde max d'une cellule (max sur les deux directions). FONCTEUR
// NOMME (et non lambda etendue) : emission device ROBUSTE quand le noyau Model-template est
// instancie depuis une TU EXTERNE (add_compiled_model, via la std::function de make_max_speed).
// Passe directement a reduce_max_cell -> aucune lambda etendue. Corps identique a l'ancienne
// lambda -> resultat bit-identique (meme Kokkos::Max, meme boucle hote).
/// MaxWaveSpeedKernel<Model> : foncteur de reduction device pour max_wave_speed_mf.
///
/// Accumule le max des vitesses d'onde dans les deux directions en cellule (i,j).
/// Foncteur nomme (et non lambda etendue) : emission device robuste depuis une TU externe
/// (add_compiled_model). Corps bit-identique a l'ancienne lambda. ADC_HD.
template <class Model>
struct MaxWaveSpeedKernel {
  Model model;
  ConstArray4 u, a;
  ADC_HD void operator()(int i, int j, Real& acc) const {
    const auto s = load_state<Model>(u, i, j);
    const Aux ax = load_aux<aux_comps<Model>()>(a, i, j);
    const Real wx = model.max_wave_speed(s, ax, 0);
    const Real wy = model.max_wave_speed(s, ax, 1);
    const Real w = wx > wy ? wx : wy;
    if (w > acc) acc = w;
  }
};
}  // namespace detail

/// max_wave_speed_mf : max global de la vitesse d'onde sur toute la MultiFab (CFL).
///
/// Reduce sur toutes les boites locales puis all_reduce_max sur tous les rangs MPI.
/// Sans l'all_reduce, chaque rang ne voit que ses boites et step_cfl calcule un dt
/// different par rang (desynchronisation / divergence). En serie all_reduce_max est l'identite.
/// Pour un modele sans transport (max_wave_speed = 0 partout) -> renvoie 0 (pas non contraint).
// Vitesse d'onde maximale d'un champ : max sur les cellules valides et les deux directions
// de model.max_wave_speed(U, aux, dir). Sert au choix CFL du pas (dt = cfl*h/w_max). Pour un
// modele sans transport (flux nul, w=0) -> 0, donc ne contraint pas le pas. Reduction par le
// seam (vraie reduction device sous Kokkos, boucle hote sinon).
//
// COLLECTIF SOUS MPI : on agrege par all_reduce_max sur TOUS les rangs (meme convention que
// AmrCouplerMp::max_wave_speed et GeometricMG::current_residual). Sans cet all-reduce, chaque rang
// ne voit que le max de SES boites : step_cfl / step_adaptive choisissent alors un dt DIFFERENT par
// rang (le rang dont le max local est plus faible prend un pas trop grand) et la simulation diverge
// ou desynchronise les rangs. En serie all_reduce_max est l'identite (comportement inchange).
template <class Model>
inline Real max_wave_speed_mf(const Model& model, const MultiFab& U,
                              const MultiFab& aux) {
  Real m = 0;
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    m = std::max(m, reduce_max_cell(U.box(li), detail::MaxWaveSpeedKernel<Model>{model, u, a}));
  }
  return static_cast<Real>(all_reduce_max(static_cast<double>(m)));
}

namespace detail {
/// Localisation de la cellule DOMINANTE de la CFL (diagnostic dt_hotspot, ADC-182) : scan
/// d'EGALITE du w recalcule -- meme foncteur et memes donnees que MaxWaveSpeedKernel, donc
/// bit-egal au max retourne par max_wave_speed_mf -- qui encode l'indice GLOBAL j*nx + i en
/// Real (exact tant que nx*ny < 2^53) et reduit au MIN (premiere cellule en ordre
/// lexicographique : deterministe). Foncteur NOMME (instanciation cross-TU sous nvcc).
template <class Model>
struct WaveSpeedMatchKernel {
  Model model;
  ConstArray4 u, a;
  Real target;
  Real nx;  // stride d'encodage (nx du DOMAINE, indices globaux)
  ADC_HD void operator()(int i, int j, Real& acc) const {
    const auto s = load_state<Model>(u, i, j);
    const Aux ax = load_aux<aux_comps<Model>()>(a, i, j);
    const Real wx = model.max_wave_speed(s, ax, 0);
    const Real wy = model.max_wave_speed(s, ax, 1);
    const Real w = wx > wy ? wx : wy;
    if (w == target) {
      const Real idx = static_cast<Real>(j) * nx + static_cast<Real>(i);
      if (idx < acc) acc = idx;
    }
  }
};
}  // namespace detail

/// Diagnostic dt_hotspot (ADC-182) : la cellule (indices GLOBAUX) qui domine la borne CFL de
/// transport du bloc, et sa vitesse w = max(wx, wy). A LA DEMANDE uniquement -- deux passes
/// completes (max puis localisation par egalite bit-exacte), step_cfl n'y touche pas
/// (bit-identique). MPI : all_reduce du max puis all_reduce_min de l'indice encode (+inf chez
/// les rangs non detenteurs). @p nx : largeur du domaine (encodage j*nx + i).
template <class Model>
inline void max_wave_speed_hotspot_mf(const Model& model, const MultiFab& U,
                                      const MultiFab& aux, int nx,
                                      Real& w_out, int& i_out, int& j_out) {
  const Real w = max_wave_speed_mf(model, U, aux);
  Real best = std::numeric_limits<Real>::infinity();
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    best = std::min(best, reduce_min_cell(U.box(li), detail::WaveSpeedMatchKernel<Model>{
                                              model, u, a, w, static_cast<Real>(nx)}));
  }
  best = static_cast<Real>(all_reduce_min(static_cast<double>(best)));
  w_out = w;
  // identite de Kokkos::Min = max_real (finie) : un rang/une boite sans cellule egalant le
  // max laisse cette valeur -> on ne decode que si un indice REEL a ete encode.
  if (best >= Real(0) && best < std::numeric_limits<Real>::max() * Real(0.5)) {
    const long long idx = static_cast<long long>(best);
    i_out = static_cast<int>(idx % nx);
    j_out = static_cast<int>(idx / nx);
  } else {  // domaine vide / etat degenere : pas de cellule (w peut etre 0)
    i_out = -1;
    j_out = -1;
  }
}

// ============================================================================
// REDUCTIONS DES BORNES DE PAS OPTIONNELLES (audit 2026-06, chantier step_cfl).
// Pendants de max_wave_speed_mf pour les traits HasStabilitySpeed / HasSourceFrequency /
// HasStabilityDt (cf. core/physical_model.hpp). Memes conventions : reduction par le seam
// (device sous Kokkos), all_reduce MPI (sans quoi chaque rang choisirait un dt different).
// Instanciees UNIQUEMENT pour un modele declarant le trait (if constexpr cote block_builder) :
// zero codegen, zero cout pour un modele historique.
// ============================================================================

namespace detail {
/// StabilitySpeedKernel : max cellules/directions de model.stability_speed (remplace
/// MaxWaveSpeedKernel quand le trait est declare). Foncteur nomme (device-clean cross-TU).
template <class Model>
struct StabilitySpeedKernel {
  Model model;
  ConstArray4 u, a;
  ADC_HD void operator()(int i, int j, Real& acc) const {
    const auto s = load_state<Model>(u, i, j);
    const Aux ax = load_aux<aux_comps<Model>()>(a, i, j);
    const Real wx = model.stability_speed(s, ax, 0);
    const Real wy = model.stability_speed(s, ax, 1);
    const Real w = wx > wy ? wx : wy;
    if (w > acc) acc = w;
  }
};

/// SourceFrequencyKernel : max cellules de model.source_frequency (mu >= 0, 1/s).
template <class Model>
struct SourceFrequencyKernel {
  Model model;
  ConstArray4 u, a;
  ADC_HD void operator()(int i, int j, Real& acc) const {
    const auto s = load_state<Model>(u, i, j);
    const Aux ax = load_aux<aux_comps<Model>()>(a, i, j);
    const Real mu = model.source_frequency(s, ax);
    if (mu > acc) acc = mu;
  }
};

/// InvStabilityDtKernel : max cellules de 1/model.stability_dt. On reduit l'INVERSE (une
/// frequence) parce que le seam ne fournit qu'une reduction MAX initialisee a 0 (reduce_max_cell) :
/// min(dt) == 1/max(1/dt) pour des dt > 0. Un stability_dt <= 0 ou non fini est ignore (ne
/// contraint pas) -- le modele signale "pas de borne ici" en retournant +inf.
template <class Model>
struct InvStabilityDtKernel {
  Model model;
  ConstArray4 u, a;
  ADC_HD void operator()(int i, int j, Real& acc) const {
    const auto s = load_state<Model>(u, i, j);
    const Aux ax = load_aux<aux_comps<Model>()>(a, i, j);
    const Real db = model.stability_dt(s, ax);
    if (db > Real(0)) {
      const Real inv = Real(1) / db;
      if (inv > acc) acc = inv;
    }
  }
};
}  // namespace detail

/// Max global de la vitesse de STABILITE (trait HasStabilitySpeed) -- pendant de max_wave_speed_mf.
template <class Model>
inline Real max_stability_speed_mf(const Model& model, const MultiFab& U, const MultiFab& aux) {
  Real m = 0;
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    m = std::max(m, reduce_max_cell(U.box(li), detail::StabilitySpeedKernel<Model>{model, u, a}));
  }
  return static_cast<Real>(all_reduce_max(static_cast<double>(m)));
}

/// Max global de la frequence de source (trait HasSourceFrequency). 0 si la source ne contraint pas.
template <class Model>
inline Real max_source_frequency_mf(const Model& model, const MultiFab& U, const MultiFab& aux) {
  Real m = 0;
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    m = std::max(m, reduce_max_cell(U.box(li), detail::SourceFrequencyKernel<Model>{model, u, a}));
  }
  return static_cast<Real>(all_reduce_max(static_cast<double>(m)));
}

/// Min global du pas admissible declare (trait HasStabilityDt), via max(1/dt) (cf.
/// InvStabilityDtKernel). @return 0 si AUCUNE cellule ne contraint (le bloc n'impose pas de borne).
template <class Model>
inline Real min_stability_dt_mf(const Model& model, const MultiFab& U, const MultiFab& aux) {
  Real inv = 0;
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    inv = std::max(inv, reduce_max_cell(U.box(li), detail::InvStabilityDtKernel<Model>{model, u, a}));
  }
  inv = static_cast<Real>(all_reduce_max(static_cast<double>(inv)));
  return inv > Real(0) ? Real(1) / inv : Real(0);
}

/// rusanov_flux : compat libre, delegue a RusanovFlux{} (politique de numerical_flux.hpp).
///
/// Conserve pour les references serie (demos GPU, tests unitaires) qui apellent rusanov_flux
/// directement. Preferer RusanovFlux{} passe en template pour les nouveaux appels. ADC_HD.
// Compat : flux de Rusanov en fonction libre, delegue a la politique RusanovFlux
// (operator/numerical_flux.hpp). Conserve pour les references serie (demos GPU,
// tests) qui appellent rusanov_flux directement.
template <class Model>
ADC_HD inline typename Model::State rusanov_flux(const Model& m,
                                          const typename Model::State& UL,
                                          const Aux& AL,
                                          const typename Model::State& UR,
                                          const Aux& AR, int dir) {
  return RusanovFlux{}(m, UL, AL, UR, AR, dir);
}

/// reconstruct<Model,Limiter> : valeur de face a (i,j) extrapolee dans la direction dir.
///
/// sgn = +1 -> face +dir de (i,j) ; sgn = -1 -> face -dir. Reconstruit en variables
/// PRIMITIVES si prim == true ET si Model expose HasPrimitiveVars (positivite de rho et p
/// pour Euler) ; sinon en conservatif. L'etat rendu est TOUJOURS conservatif.
/// NoSlope (n_ghost == 1) : pente nulle, prim sans effet -- chemin conservatif pur.
/// INVARIANT : fonction PONCTUELLE, ne boucle PAS sur la grille. ADC_HD.
// Valeur de cellule (i,j) extrapolee vers sa face +dir (sgn=+1) ou -dir (sgn=-1).
// Reconstruit en variables PRIMITIVES si prim==true et que le modele expose les conversions
// (plus robuste pour Euler : positivite de rho et p) ; sinon en conservatif. L'etat rendu est
// TOUJOURS conservatif (consomme par le flux numerique). NoSlope (n_ghost==1) : pas de pente,
// prim sans effet -> chemin conservatif.
template <class Model, class Limiter>
ADC_HD inline typename Model::State reconstruct(const Model& model, const ConstArray4& u,
                                                int i, int j, int dir, Real sgn,
                                                const Limiter& lim, bool prim) {
  if constexpr (HasPrimitiveVars<Model> && Limiter::n_ghost >= 2) {
    if (prim) {  // convertir le stencil U->P, limiter sur P, reconvertir P->U
      using Prim = typename Model::Prim;
      const Prim P0 = model.to_primitive(load_state<Model>(u, i, j));
      Prim Pf{};
      if constexpr (Limiter::n_ghost == 2) {
        const Prim Pm = model.to_primitive(
            load_state<Model>(u, dir == 0 ? i - 1 : i, dir == 0 ? j : j - 1));
        const Prim Pp = model.to_primitive(
            load_state<Model>(u, dir == 0 ? i + 1 : i, dir == 0 ? j : j + 1));
        for (int c = 0; c < Model::n_vars; ++c)
          Pf[c] = P0[c] + sgn * Real(0.5) * lim(P0[c] - Pm[c], Pp[c] - P0[c]);
      } else {  // WENO5 sur le stencil 5 points en primitif
        const int d = (sgn > Real(0)) ? 1 : -1;
        const Prim Pm2 = model.to_primitive(
            load_state<Model>(u, dir == 0 ? i - 2 * d : i, dir == 0 ? j : j - 2 * d));
        const Prim Pm1 = model.to_primitive(
            load_state<Model>(u, dir == 0 ? i - d : i, dir == 0 ? j : j - d));
        const Prim Pp1 = model.to_primitive(
            load_state<Model>(u, dir == 0 ? i + d : i, dir == 0 ? j : j + d));
        const Prim Pp2 = model.to_primitive(
            load_state<Model>(u, dir == 0 ? i + 2 * d : i, dir == 0 ? j : j + 2 * d));
        for (int c = 0; c < Model::n_vars; ++c)
          Pf[c] = weno5z(Pm2[c], Pm1[c], P0[c], Pp1[c], Pp2[c]);
      }
      return model.to_conservative(Pf);
    }
  }
  (void)model;
  (void)prim;
  typename Model::State s = load_state<Model>(u, i, j);
  if constexpr (Limiter::n_ghost == 2) {
    // MUSCL : pente limitee par composante (ordre 2).
    for (int c = 0; c < Model::n_vars; ++c) {
      const Real am = (dir == 0) ? u(i, j, c) - u(i - 1, j, c)
                                 : u(i, j, c) - u(i, j - 1, c);
      const Real ap = (dir == 0) ? u(i + 1, j, c) - u(i, j, c)
                                 : u(i, j + 1, c) - u(i, j, c);
      s[c] += sgn * Real(0.5) * lim(am, ap);
    }
  } else if constexpr (Limiter::n_ghost >= 3) {
    // WENO5 (ordre 5) : valeur de face depuis un stencil 5 points oriente par sgn
    // (sgn>0 -> face +dir ; sgn<0 -> face -dir, stencil renverse). lim inutilise.
    (void)lim;
    const int d = (sgn > Real(0)) ? 1 : -1;
    for (int c = 0; c < Model::n_vars; ++c) {
      if (dir == 0)
        s[c] = weno5z(u(i - 2 * d, j, c), u(i - d, j, c), u(i, j, c),
                      u(i + d, j, c), u(i + 2 * d, j, c));
      else
        s[c] = weno5z(u(i, j - 2 * d, c), u(i, j - d, c), u(i, j, c),
                      u(i, j + d, c), u(i, j + 2 * d, c));
    }
  }
  return s;
}

/// zhang_shu_scale : limiteur de POSITIVITE sur un etat de face reconstruit -- REPLI A L'ORDRE 1
/// LOCAL (variante robuste-au-vide du scaling de Zhang & Shu, JCP 2010).
///
/// Si la composante @p pos_comp (role Density) de l'etat de face @p s passe sous @p floor, l'etat
/// de face ENTIER est remplace par la moyenne de sa cellule SOURCE u(i,j,.) (pente nulle locale).
/// POURQUOI pas le theta-scaling colineaire du papier (s <- ubar + theta (s - ubar), theta tel que
/// rho_face = floor) : en variables CONSERVATIVES au bord du QUASI-VIDE (fond 1e-6 du diocotron
/// Hoffart), il pose rho_face = floor en laissant un moment de face O(moyenne) -> la VITESSE de
/// face v = m/rho diverge (~1e6) -> la vitesse d'onde Rusanov explose alors que dt a ete choisi
/// AVANT sur les vitesses de cellule -> blow-up immediat (mesure : NaN au pas 2 du cas Hoffart,
/// quel que soit le floor). Le papier couple son limiteur a la borne CFL recalculee ; ici le repli
/// a la moyenne borne la vitesse de face par CONSTRUCTION (v_face = v_cellule), reste conservatif
/// (la moyenne n'est pas touchee), positif des que la moyenne l'est, et ne degrade l'ordre que sur
/// les faces fautives (WENO5 intact partout ailleurs).
/// Inactif si floor <= 0 (chemin bit-identique) ou si la face est deja >= floor. Motivation : WENO5
/// sous-shoote au saut top-hat contraste 1e6 -> rho de face negatif -> 1/rho et la source Lorentz
/// detonent -> NaN (adc_cases ADC-62/ADC-74, ticket ADC-76). Fonction PONCTUELLE device-clean. ADC_HD.
template <class Model>
ADC_HD inline void zhang_shu_scale(typename Model::State& s, const ConstArray4& u,
                                   int i, int j, Real floor, int pos_comp) {
  if (!(floor > Real(0))) return;            // opt-in strict : floor <= 0 -> aucun effet
  if (!(s[pos_comp] < floor)) return;        // face deja au-dessus du plancher
  for (int c = 0; c < Model::n_vars; ++c) s[c] = u(i, j, c);  // repli ordre 1 : face = moyenne
}

/// reconstruct_pp : reconstruct + limiteur de positivite zhang_shu_scale sur l'etat rendu.
///
/// (i, j) est la cellule SOURCE de la reconstruction : c'est vers SA moyenne que l'etat de face est
/// ramene. pos_floor <= 0 -> strictement identique a reconstruct (court-circuit). ADC_HD.
template <class Model, class Limiter>
ADC_HD inline typename Model::State reconstruct_pp(const Model& model, const ConstArray4& u,
                                                   int i, int j, int dir, Real sgn,
                                                   const Limiter& lim, bool prim,
                                                   Real pos_floor, int pos_comp) {
  typename Model::State s = reconstruct<Model>(model, u, i, j, dir, sgn, lim, prim);
  zhang_shu_scale<Model>(s, u, i, j, pos_floor, pos_comp);
  return s;
}

namespace detail {
/// Composante du role Density pour le limiteur de positivite (HOTE, resolu une fois par appel
/// d'operateur spatial, jamais par cellule). pos_floor <= 0 -> 0 (jamais lu, le scaling est
/// court-circuite dans zhang_shu_scale). Un modele sans introspection VariableSet ou sans role
/// Density ne peut pas demander la positivite : erreur claire plutot qu'un scaling muet d'une
/// composante arbitraire.
template <class Model>
inline int positivity_comp(Real pos_floor) {
  if (!(pos_floor > Real(0))) return 0;
  if constexpr (requires { Model::conservative_vars(); }) {
    const int c = Model::conservative_vars().index_of(VariableRole::Density);
    if (c >= 0) return c;
    throw std::runtime_error(
        "positivity_floor > 0 : le modele n'expose pas le role Density (cible du scaling)");
  } else {
    throw std::runtime_error(
        "positivity_floor > 0 : modele sans introspection VariableSet (conservative_vars)");
  }
}
}  // namespace detail

/// xface_box / yface_box : boites de face normales a x (resp. y) associees a une boite de cellules.
///
/// xface_box(v) : nx+1 x ny (i dans [lo..hi+1], j dans [lo..hi]).
/// yface_box(v) : nx x ny+1 (i dans [lo..hi], j dans [lo..hi+1]).
/// Sert a dimensionner les MultiFab Fx, Fy recus par compute_face_fluxes.
// Boites de FACE associees a une boite de cellules (faces normales a x : nx+1 x ny ;
// normales a y : nx x ny+1). Sert a dimensionner les MultiFab de flux de face.
inline Box2D xface_box(const Box2D& v) {
  return Box2D{{v.lo[0], v.lo[1]}, {v.hi[0] + 1, v.hi[1]}};
}
inline Box2D yface_box(const Box2D& v) {
  return Box2D{{v.lo[0], v.lo[1]}, {v.hi[0], v.hi[1] + 1}};
}

// compute_face_fluxes : ecrit les flux numeriques aux FACES (Fx aux faces normales
// a x, Fy a y), AVANT divergence. C'est la brique dont le reflux AMR a besoin (il
// accumule les flux fins et soustrait le flux grossier aux interfaces coarse-fine ;
// assemble_rhs, lui, calcule directement -div F et jette les flux de face).
//
// Conventions : Fx(i,j) = flux a la face entre les cellules (i-1,j) et (i,j), i dans
// [lo..hi+1]. Fy(i,j) = flux entre (i,j-1) et (i,j), j dans [lo..hi+1]. Memes
// reconstruction (Limiter) et flux numerique (NumericalFlux) qu'assemble_rhs, donc
//   r(i,j) = S - (Fx(i+1,j)-Fx(i,j))/dx - (Fy(i,j+1)-Fy(i,j))/dy
// redonne EXACTEMENT le residu d'assemble_rhs. Fx, Fy dimensionnes par l'appelant
// (boites xface_box/yface_box, ncomp = Model::n_vars, 0 ghost). Device-callable.
//
// DIFFUSION sur AMR (TODO 4) : pour un DiffusiveModel, on ajoute le flux de FACE
// Fickien F_diff = -nu (u_R - u_L)/h (gradient centre au face, valeurs de cellule).
// Sa divergence -(Fx(i+1)-Fx(i))/dx redonne EXACTEMENT +nu Lap(u) d'assemble_rhs,
// mais traite en FLUX : le reflux AMR le voit donc, et la diffusion reste
// conservative aux interfaces coarse-fine (sinon un Laplacien direct serait ignore
// par le reflux). dx/dy = pas du NIVEAU (passes par l'appelant ; 0 par defaut, non
// lus pour un modele non diffusif -> chemin hyperbolique strictement bit-identique).
namespace detail {
// Noyaux de FLUX DE FACE (x puis y). FONCTEURS NOMMES (et non lambdas etendues) : emission device
// ROBUSTE quand le noyau Model-template est instancie depuis une TU EXTERNE (chemin reflux AMR d'un
// bloc add_compiled_model). Corps identique aux anciennes lambdas (le terme Fickien reste garde par
// DiffusiveModel) -> flux de face bit-identique. dx/dy ne sont lus que par la branche diffusive
// (membre inutilise, sans codegen, pour un modele non diffusif) : plus besoin du (void)dx hors
// if-constexpr qu'imposait la capture d'une lambda.
/// FaceFluxXKernel : noyau device de flux a la face radiale x (entre i-1 et i).
///
/// Reconstruit les etats L (cellule i-1, face +x) et R (cellule i, face -x), calcule
/// le flux numerique, ecrit dans fx(i,j). Ajoute le flux Fickien si DiffusiveModel.
/// Foncteur nomme (device-clean cross-TU). ADC_HD.
template <class Limiter, class NumericalFlux, class Model>
struct FaceFluxXKernel {
  Model model;
  ConstArray4 u, ax;
  Array4 fx;
  Real dx;
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< limiteur de positivite Zhang-Shu (<= 0 : inactif, bit-identique)
  int pos_comp = 0;          ///< composante du role Density (resolue par l'appelant hote)
  ADC_HD void operator()(int i, int j) const {
    const auto L = reconstruct_pp<Model>(model, u, i - 1, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rr = reconstruct_pp<Model>(model, u, i, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto F = nflux(model, L, load_aux<aux_comps<Model>()>(ax, i - 1, j), Rr,
                         load_aux<aux_comps<Model>()>(ax, i, j), 0);
    for (int c = 0; c < Model::n_vars; ++c) fx(i, j, c) = F[c];
    if constexpr (DiffusiveModel<Model>) {
      const Real nu = model.diffusivity();
      for (int c = 0; c < Model::n_vars; ++c)
        fx(i, j, c) += -nu * (u(i, j, c) - u(i - 1, j, c)) / dx;
    }
  }
};
/// FaceFluxYKernel : noyau device de flux a la face y (entre j-1 et j).
///
/// Analogue de FaceFluxXKernel dans la direction j. Foncteur nomme. ADC_HD.
template <class Limiter, class NumericalFlux, class Model>
struct FaceFluxYKernel {
  Model model;
  ConstArray4 u, ax;
  Array4 fy;
  Real dy;
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< limiteur de positivite Zhang-Shu (<= 0 : inactif, bit-identique)
  int pos_comp = 0;          ///< composante du role Density (resolue par l'appelant hote)
  ADC_HD void operator()(int i, int j) const {
    const auto L = reconstruct_pp<Model>(model, u, i, j - 1, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rr = reconstruct_pp<Model>(model, u, i, j, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto F = nflux(model, L, load_aux<aux_comps<Model>()>(ax, i, j - 1), Rr,
                         load_aux<aux_comps<Model>()>(ax, i, j), 1);
    for (int c = 0; c < Model::n_vars; ++c) fy(i, j, c) = F[c];
    if constexpr (DiffusiveModel<Model>) {
      const Real nu = model.diffusivity();
      for (int c = 0; c < Model::n_vars; ++c)
        fy(i, j, c) += -nu * (u(i, j, c) - u(i, j - 1, c)) / dy;
    }
  }
};
}  // namespace detail

/// compute_face_fluxes<Limiter,NumericalFlux> : ecrit les flux aux faces AVANT divergence.
///
/// Fx(i,j) = flux a la face entre (i-1,j) et (i,j), i dans [lo..hi+1].
/// Fy(i,j) = flux entre (i,j-1) et (i,j), j dans [lo..hi+1].
/// Brique necessaire au reflux AMR : assemble_rhs calcule directement -div F et jette les
/// flux de face, mais le reflux doit les voir pour corriger les interfaces coarse-fine.
/// Pour un DiffusiveModel, le flux Fickien F_diff = -nu (u_R-u_L)/h est ajoute (sa
/// divergence reproduit EXACTEMENT +nu Lap(u) d'assemble_rhs, et reste visible du reflux).
/// dx=0, dy=0 par defaut : non lus pour un modele non diffusif (bit-identique hyperbolique).
// compute_face_fluxes : ecrit les flux numeriques aux FACES (Fx aux faces normales
// a x, Fy a y), AVANT divergence. C'est la brique dont le reflux AMR a besoin (il
// accumule les flux fins et soustrait le flux grossier aux interfaces coarse-fine ;
// assemble_rhs, lui, calcule directement -div F et jette les flux de face).
//
// Conventions : Fx(i,j) = flux a la face entre les cellules (i-1,j) et (i,j), i dans
// [lo..hi+1]. Fy(i,j) = flux entre (i,j-1) et (i,j), j dans [lo..hi+1]. Memes
// reconstruction (Limiter) et flux numerique (NumericalFlux) qu'assemble_rhs, donc
//   r(i,j) = S - (Fx(i+1,j)-Fx(i,j))/dx - (Fy(i,j+1)-Fy(i,j))/dy
// redonne EXACTEMENT le residu d'assemble_rhs. Fx, Fy dimensionnes par l'appelant
// (boites xface_box/yface_box, ncomp = Model::n_vars, 0 ghost). Device-callable.
//
// DIFFUSION sur AMR (TODO 4) : pour un DiffusiveModel, on ajoute le flux de FACE
// Fickien F_diff = -nu (u_R - u_L)/h (gradient centre au face, valeurs de cellule).
// Sa divergence -(Fx(i+1)-Fx(i))/dx redonne EXACTEMENT +nu Lap(u) d'assemble_rhs,
// mais traite en FLUX : le reflux AMR le voit donc, et la diffusion reste
// conservative aux interfaces coarse-fine (sinon un Laplacien direct serait ignore
// par le reflux). dx/dy = pas du NIVEAU (passes par l'appelant ; 0 par defaut, non
// lus pour un modele non diffusif -> chemin hyperbolique strictement bit-identique).
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void compute_face_fluxes(const Model& model, const MultiFab& U, const MultiFab& aux,
                         MultiFab& Fx, MultiFab& Fy, Real dx = 0, Real dy = 0,
                         bool recon_prim = false, Real pos_floor = Real(0)) {
  const Limiter lim{};
  const NumericalFlux nflux{};
  const int pos_comp = detail::positivity_comp<Model>(pos_floor);
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    Array4 fx = Fx.fab(li).array();
    Array4 fy = Fy.fab(li).array();
    const Box2D v = U.box(li);
    for_each_cell(xface_box(v), detail::FaceFluxXKernel<Limiter, NumericalFlux, Model>{
                                    model, u, ax, fx, dx, lim, nflux, recon_prim, pos_floor, pos_comp});
    for_each_cell(yface_box(v), detail::FaceFluxYKernel<Limiter, NumericalFlux, Model>{
                                    model, u, ax, fy, dy, lim, nflux, recon_prim, pos_floor, pos_comp});
  }
}

namespace detail {
// Noyau device d'assemble_rhs : R = -div Fhat + S (+ Fickien si diffusif) en (i, j). FONCTEUR NOMME
// (et non lambda etendue) : c'est le point CLE du chemin AOT "parite native" (add_compiled_model).
// nvcc n'emet pas fiablement le kernel device d'une lambda etendue Model-template premiere-instanciee
// depuis une TU EXTERNE a travers le nesting std::function / lambda-hote de block_builder : le test
// passe sur Serial et sous compute-sanitizer mais segfaute a l'execution sur Cuda (Heisenbug). Une
// classe device-callable n'a pas ces restrictions de contexte d'instanciation. Corps IDENTIQUE a
// l'ancienne lambda -> residu BIT-IDENTIQUE a add_block sur CPU (et, vise, sur device).
/// AssembleRhsKernel<Limiter,NumericalFlux,Model> : noyau device du residu central d'assemble_rhs.
///
/// Calcule R(i,j) = S - (Fxp-Fxm)/dx - (Fyp-Fym)/dy (+ terme Fickien si DiffusiveModel).
/// Foncteur nomme : point cle de la parite native AOT (add_compiled_model via TU externe).
/// Corps bit-identique a l'ancienne lambda. ADC_HD.
template <class Limiter, class NumericalFlux, class Model>
struct AssembleRhsKernel {
  Model model;
  ConstArray4 u, ax;
  Array4 r;
  Real dx, dy;
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< limiteur de positivite Zhang-Shu (<= 0 : inactif, bit-identique)
  int pos_comp = 0;          ///< composante du role Density (resolue par l'appelant hote)
  ADC_HD void operator()(int i, int j) const {
    const Aux Ac = load_aux<aux_comps<Model>()>(ax, i, j);
    const Aux Axm = load_aux<aux_comps<Model>()>(ax, i - 1, j);
    const Aux Axp = load_aux<aux_comps<Model>()>(ax, i + 1, j);
    const Aux Aym = load_aux<aux_comps<Model>()>(ax, i, j - 1);
    const Aux Ayp = load_aux<aux_comps<Model>()>(ax, i, j + 1);

    // faces x : reconstruction des etats de part et d'autre de chaque face
    const auto Lxm = reconstruct_pp<Model>(model, u, i - 1, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rxm = reconstruct_pp<Model>(model, u, i, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto Lxp = reconstruct_pp<Model>(model, u, i, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rxp = reconstruct_pp<Model>(model, u, i + 1, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto Fxm = nflux(model, Lxm, Axm, Rxm, Ac, 0);
    const auto Fxp = nflux(model, Lxp, Ac, Rxp, Axp, 0);

    // faces y
    const auto Lym = reconstruct_pp<Model>(model, u, i, j - 1, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rym = reconstruct_pp<Model>(model, u, i, j, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto Lyp = reconstruct_pp<Model>(model, u, i, j, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Ryp = reconstruct_pp<Model>(model, u, i, j + 1, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto Fym = nflux(model, Lym, Aym, Rym, Ac, 1);
    const auto Fyp = nflux(model, Lyp, Ac, Ryp, Ayp, 1);

    const auto S = model.source(load_state<Model>(u, i, j), Ac);
    for (int c = 0; c < Model::n_vars; ++c)
      r(i, j, c) = S[c] - (Fxp[c] - Fxm[c]) / dx - (Fyp[c] - Fym[c]) / dy;

    // Terme parabolique (Fickien) : +nu Lap(U), differences centrees a 5 points.
    // Garde par DiffusiveModel : aucun effet (ni codegen) pour un modele non diffusif.
    if constexpr (DiffusiveModel<Model>) {
      const Real nu = model.diffusivity();
      const Real idx2 = Real(1) / (dx * dx), idy2 = Real(1) / (dy * dy);
      for (int c = 0; c < Model::n_vars; ++c)
        r(i, j, c) += nu * ((u(i + 1, j, c) - 2 * u(i, j, c) + u(i - 1, j, c)) * idx2 +
                            (u(i, j + 1, c) - 2 * u(i, j, c) + u(i, j - 1, c)) * idy2);
    }
  }
};
}  // namespace detail

/// assemble_rhs<Limiter,NumericalFlux> : residu R = -div Fhat + S sur toutes les boites.
///
/// Point d'entree principal de l'operateur spatial cartesien. Le limiteur (reconstruction)
/// ET le flux numerique sont des parametres de template choisis a la compilation (defaut :
/// NoSlope + RusanovFlux). recon_prim = true active la reconstruction en variables primitives
/// si le modele expose HasPrimitiveVars. Pour le terme diffusif, voir DiffusiveModel.
/// INVARIANT : l'operateur ne modifie pas U, aux -- il ecrit uniquement R. Pas de ghost fill.
// assemble_rhs<Limiter, NumericalFlux> : R = -div Fhat + S. Le limiteur (pente de
// reconstruction) ET le flux numerique sont des parametres de template, par defaut
// MUSCL au choix de l'appelant + Rusanov. Tous deux device-callable (ADC_HD).
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void assemble_rhs(const Model& model, const MultiFab& U, const MultiFab& aux,
                  const Geometry& geom, MultiFab& R, bool recon_prim = false,
                  Real pos_floor = Real(0)) {
  const Real dx = geom.dx(), dy = geom.dy();
  const Limiter lim{};
  const NumericalFlux nflux{};
  const int pos_comp = detail::positivity_comp<Model>(pos_floor);
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    Array4 r = R.fab(li).array();
    const Box2D v = R.box(li);
    for_each_cell(v, detail::AssembleRhsKernel<Limiter, NumericalFlux, Model>{
                         model, u, ax, r, dx, dy, lim, nflux, recon_prim, pos_floor, pos_comp});
  }
}

// ============================================================================
// MASQUE DE DOMAINE (chantier T2, conservatif, OPT-IN -- chemin par defaut intouche)
// ============================================================================
// Le masque rend le transport FV conscient d'un sous-domaine ACTIF (p.ex. le disque du papier).
// Convention : mask(i, j) >= 0.5 -> cellule ACTIVE, sinon INACTIVE. Une face est OUVERTE (flux
// normal calcule) si ses DEUX cellules adjacentes sont actives ; elle est FERMEE (flux normal mis
// a ZERO) si l'une au moins est inactive. Mettre a zero le flux normal aux faces active/inactive
// rend le pas CONSERVATIF sur le sous-domaine actif : aucune masse ne traverse la frontiere, donc
// la masse totale sur les cellules actives est conservee a la machine (flux telescopiques internes,
// flux de bord nuls). C'est le pendant FV du mur conducteur (qui n'agit que sur l'elliptique).
//
// Le residu n'est ecrit QUE sur les cellules actives ; une cellule inactive garde son residu a 0
// (l'appelant ne l'avance pas). Ce header NE wire PAS ce chemin dans System::step : il fournit la
// brique mask-aware, exercee directement par les tests et, a terme, derriere l'opt-in disque.

namespace detail {
/// Indicateur d'activite d'une cellule depuis un masque 0/1 cellule-centre (>= 0.5 -> actif).
ADC_HD inline bool mask_active(const ConstArray4& mask, int i, int j) {
  return mask(i, j, 0) >= Real(0.5);
}

/// AssembleRhsMaskedKernel : variante de AssembleRhsKernel CONSCIENTE d'un masque de domaine.
///
/// Cellule inactive -> residu 0 (non avancee par l'appelant). Cellule active -> R = -div Fhat + S,
/// MAIS le flux normal d'une face dont la cellule voisine est INACTIVE est mis a ZERO (paroi
/// FV : zero flux normal a la frontiere active/inactive) -> conservation de masse sur le
/// sous-domaine actif. Foncteur nomme (meme contrat device que AssembleRhsKernel). ADC_HD.
///
/// NB : sans terme diffusif (modeles transport-seul vises par le disque) ; un DiffusiveModel garde
/// son Laplacien NON masque ici (raffinement separe -- le masque conservatif cible le flux
/// hyperbolique, cf. le chantier "bords d'anneau").
template <class Limiter, class NumericalFlux, class Model>
struct AssembleRhsMaskedKernel {
  Model model;
  ConstArray4 u, ax, mask;
  Array4 r;
  Real dx, dy;
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< limiteur de positivite Zhang-Shu (<= 0 : inactif, bit-identique)
  int pos_comp = 0;          ///< composante du role Density (resolue par l'appelant hote)
  ADC_HD void operator()(int i, int j) const {
    if (!mask_active(mask, i, j)) {  // cellule hors sous-domaine actif : residu nul, non avancee
      for (int c = 0; c < Model::n_vars; ++c) r(i, j, c) = Real(0);
      return;
    }
    const Aux Ac = load_aux<aux_comps<Model>()>(ax, i, j);
    const Aux Axm = load_aux<aux_comps<Model>()>(ax, i - 1, j);
    const Aux Axp = load_aux<aux_comps<Model>()>(ax, i + 1, j);
    const Aux Aym = load_aux<aux_comps<Model>()>(ax, i, j - 1);
    const Aux Ayp = load_aux<aux_comps<Model>()>(ax, i, j + 1);

    // faces x : reconstruction de part et d'autre, flux numerique, PUIS porte du masque (face fermee
    // -> flux normal nul) -- une cellule voisine inactive ferme la face entre elle et (i, j).
    const auto Lxm = reconstruct_pp<Model>(model, u, i - 1, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rxm = reconstruct_pp<Model>(model, u, i, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto Lxp = reconstruct_pp<Model>(model, u, i, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rxp = reconstruct_pp<Model>(model, u, i + 1, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    auto Fxm = nflux(model, Lxm, Axm, Rxm, Ac, 0);
    auto Fxp = nflux(model, Lxp, Ac, Rxp, Axp, 0);
    if (!mask_active(mask, i - 1, j)) Fxm = typename Model::State{};
    if (!mask_active(mask, i + 1, j)) Fxp = typename Model::State{};

    // faces y
    const auto Lym = reconstruct_pp<Model>(model, u, i, j - 1, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rym = reconstruct_pp<Model>(model, u, i, j, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto Lyp = reconstruct_pp<Model>(model, u, i, j, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Ryp = reconstruct_pp<Model>(model, u, i, j + 1, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    auto Fym = nflux(model, Lym, Aym, Rym, Ac, 1);
    auto Fyp = nflux(model, Lyp, Ac, Ryp, Ayp, 1);
    if (!mask_active(mask, i, j - 1)) Fym = typename Model::State{};
    if (!mask_active(mask, i, j + 1)) Fyp = typename Model::State{};

    const auto S = model.source(load_state<Model>(u, i, j), Ac);
    for (int c = 0; c < Model::n_vars; ++c)
      r(i, j, c) = S[c] - (Fxp[c] - Fxm[c]) / dx - (Fyp[c] - Fym[c]) / dy;
  }
};
}  // namespace detail

/// assemble_rhs_masked<Limiter,NumericalFlux> : residu R = -div Fhat + S RESTREINT a un masque de
/// domaine 0/1 cellule-centre (OPT-IN, chantier T2). Sur une cellule inactive R = 0 (non avancee) ;
/// sur une cellule active, le flux normal d'une face dont la voisine est inactive est mis a zero
/// (paroi FV). Resultat : la masse sur le sous-domaine actif est CONSERVEE a la machine (aucun flux
/// ne traverse la frontiere) -- propriete validee par le test conservation du chantier disque.
///
/// @p mask doit avoir le MEME layout que @p U (meme BoxArray / DistributionMapping) et porter au
/// moins 1 ghost (lecture des voisins i-1/i+1/j-1/j+1 jusqu'au bord). Ce point d'entree est SEPARE
/// d'assemble_rhs : le chemin par defaut (System::step) reste strictement bit-identique tant qu'il
/// n'appelle PAS cette surcharge.
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void assemble_rhs_masked(const Model& model, const MultiFab& U, const MultiFab& aux,
                         const MultiFab& mask, const Geometry& geom, MultiFab& R,
                         bool recon_prim = false, Real pos_floor = Real(0)) {
  const Real dx = geom.dx(), dy = geom.dy();
  const Limiter lim{};
  const NumericalFlux nflux{};
  const int pos_comp = detail::positivity_comp<Model>(pos_floor);
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    const ConstArray4 mk = mask.fab(li).const_array();
    Array4 r = R.fab(li).array();
    const Box2D v = R.box(li);
    for_each_cell(v, detail::AssembleRhsMaskedKernel<Limiter, NumericalFlux, Model>{
                         model, u, ax, mk, r, dx, dy, lim, nflux, recon_prim, pos_floor, pos_comp});
  }
}

}  // namespace adc
