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

// Modele DIFFUSIF (optionnel) : fournit une diffusivite scalaire isotrope nu. Le
// tuteur : "la diffusion, c'est comme un flux de plus". Le flux Fickien F = -nu grad U
// ajoute au flux hyperbolique donne, apres divergence (-div F), exactement +nu Lap(U).
// On l'implemente comme un terme additif au residu, GARDE par ce trait : un modele
// sans diffusivity() ne change pas d'un bit (chemin hyperbolique inchange).
template <class M>
concept DiffusiveModel = requires(const M m) {
  { m.diffusivity() } -> std::convertible_to<Real>;
};

// Modele « sans source » : meme flux et vitesse d'onde que M, mais source nulle. Sert au
// demi-pas EXPLICITE d'un schema IMEX (transport seul, −div F), la source raide etant
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
};

template <class Model>
ADC_HD inline typename Model::State load_state(const ConstArray4& a, int i,
                                              int j) {
  typename Model::State u;
  for (int c = 0; c < Model::n_vars; ++c) u[c] = a(i, j, c);
  return u;
}

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
  return x;
}

namespace detail {
// Noyau reducteur de la vitesse d'onde max d'une cellule (max sur les deux directions). FONCTEUR
// NOMME (et non lambda etendue) : emission device ROBUSTE quand le noyau Model-template est
// instancie depuis une TU EXTERNE (add_compiled_model, via la std::function de make_max_speed).
// Passe directement a reduce_max_cell -> aucune lambda etendue. Corps identique a l'ancienne
// lambda -> resultat bit-identique (meme Kokkos::Max, meme boucle hote).
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
template <class Limiter, class NumericalFlux, class Model>
struct FaceFluxXKernel {
  Model model;
  ConstArray4 u, ax;
  Array4 fx;
  Real dx;
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  ADC_HD void operator()(int i, int j) const {
    const auto L = reconstruct<Model>(model, u, i - 1, j, 0, +1, lim, recon_prim);
    const auto Rr = reconstruct<Model>(model, u, i, j, 0, -1, lim, recon_prim);
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
template <class Limiter, class NumericalFlux, class Model>
struct FaceFluxYKernel {
  Model model;
  ConstArray4 u, ax;
  Array4 fy;
  Real dy;
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  ADC_HD void operator()(int i, int j) const {
    const auto L = reconstruct<Model>(model, u, i, j - 1, 1, +1, lim, recon_prim);
    const auto Rr = reconstruct<Model>(model, u, i, j, 1, -1, lim, recon_prim);
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

template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void compute_face_fluxes(const Model& model, const MultiFab& U, const MultiFab& aux,
                         MultiFab& Fx, MultiFab& Fy, Real dx = 0, Real dy = 0,
                         bool recon_prim = false) {
  const Limiter lim{};
  const NumericalFlux nflux{};
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    Array4 fx = Fx.fab(li).array();
    Array4 fy = Fy.fab(li).array();
    const Box2D v = U.box(li);
    for_each_cell(xface_box(v), detail::FaceFluxXKernel<Limiter, NumericalFlux, Model>{
                                    model, u, ax, fx, dx, lim, nflux, recon_prim});
    for_each_cell(yface_box(v), detail::FaceFluxYKernel<Limiter, NumericalFlux, Model>{
                                    model, u, ax, fy, dy, lim, nflux, recon_prim});
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
template <class Limiter, class NumericalFlux, class Model>
struct AssembleRhsKernel {
  Model model;
  ConstArray4 u, ax;
  Array4 r;
  Real dx, dy;
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  ADC_HD void operator()(int i, int j) const {
    const Aux Ac = load_aux<aux_comps<Model>()>(ax, i, j);
    const Aux Axm = load_aux<aux_comps<Model>()>(ax, i - 1, j);
    const Aux Axp = load_aux<aux_comps<Model>()>(ax, i + 1, j);
    const Aux Aym = load_aux<aux_comps<Model>()>(ax, i, j - 1);
    const Aux Ayp = load_aux<aux_comps<Model>()>(ax, i, j + 1);

    // faces x : reconstruction des etats de part et d'autre de chaque face
    const auto Lxm = reconstruct<Model>(model, u, i - 1, j, 0, +1, lim, recon_prim);
    const auto Rxm = reconstruct<Model>(model, u, i, j, 0, -1, lim, recon_prim);
    const auto Lxp = reconstruct<Model>(model, u, i, j, 0, +1, lim, recon_prim);
    const auto Rxp = reconstruct<Model>(model, u, i + 1, j, 0, -1, lim, recon_prim);
    const auto Fxm = nflux(model, Lxm, Axm, Rxm, Ac, 0);
    const auto Fxp = nflux(model, Lxp, Ac, Rxp, Axp, 0);

    // faces y
    const auto Lym = reconstruct<Model>(model, u, i, j - 1, 1, +1, lim, recon_prim);
    const auto Rym = reconstruct<Model>(model, u, i, j, 1, -1, lim, recon_prim);
    const auto Lyp = reconstruct<Model>(model, u, i, j, 1, +1, lim, recon_prim);
    const auto Ryp = reconstruct<Model>(model, u, i, j + 1, 1, -1, lim, recon_prim);
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

// assemble_rhs<Limiter, NumericalFlux> : R = -div Fhat + S. Le limiteur (pente de
// reconstruction) ET le flux numerique sont des parametres de template, par defaut
// MUSCL au choix de l'appelant + Rusanov. Tous deux device-callable (ADC_HD).
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void assemble_rhs(const Model& model, const MultiFab& U, const MultiFab& aux,
                  const Geometry& geom, MultiFab& R, bool recon_prim = false) {
  const Real dx = geom.dx(), dy = geom.dy();
  const Limiter lim{};
  const NumericalFlux nflux{};
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    Array4 r = R.fab(li).array();
    const Box2D v = R.box(li);
    for_each_cell(v, detail::AssembleRhsKernel<Limiter, NumericalFlux, Model>{
                         model, u, ax, r, dx, dy, lim, nflux, recon_prim});
  }
}

}  // namespace adc
