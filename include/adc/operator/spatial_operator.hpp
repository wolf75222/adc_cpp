#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/operator/numerical_flux.hpp>
#include <adc/operator/reconstruction.hpp>

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
  M m;
  ADC_HD State flux(const State& u, const Aux& a, int dir) const { return m.flux(u, a, dir); }
  ADC_HD Real max_wave_speed(const State& u, const Aux& a, int dir) const {
    return m.max_wave_speed(u, a, dir);
  }
  ADC_HD State source(const State&, const Aux&) const { return State{}; }
  ADC_HD Real elliptic_rhs(const State& u) const { return m.elliptic_rhs(u); }
};

template <class Model>
ADC_HD inline typename Model::State load_state(const ConstArray4& a, int i,
                                              int j) {
  typename Model::State u;
  for (int c = 0; c < Model::n_vars; ++c) u[c] = a(i, j, c);
  return u;
}

ADC_HD inline Aux load_aux(const ConstArray4& a, int i, int j) {
  return Aux{a(i, j, 0), a(i, j, 1), a(i, j, 2)};
}

// Vitesse d'onde maximale d'un champ : max sur les cellules valides et les deux directions
// de model.max_wave_speed(U, aux, dir). Sert au choix CFL du pas (dt = cfl*h/w_max). Pour un
// modele sans transport (flux nul, w=0) -> 0, donc ne contraint pas le pas. Reduction par le
// seam (vraie reduction device sous Kokkos, boucle hote sinon).
template <class Model>
inline Real max_wave_speed_mf(const Model& model, const MultiFab& U,
                              const MultiFab& aux) {
  Real m = 0;
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    const Model mm = model;
    m = std::max(m, for_each_cell_reduce_max(U.box(li), [u, a, mm] ADC_HD(int i, int j) {
                      const auto s = load_state<Model>(u, i, j);
                      const Aux ax = load_aux(a, i, j);
                      const Real wx = mm.max_wave_speed(s, ax, 0);
                      const Real wy = mm.max_wave_speed(s, ax, 1);
                      return wx > wy ? wx : wy;
                    }));
  }
  return m;
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
// Pour NoSlope (n_ghost==1) : pas de pente, aucune lecture de voisin.
template <class Model, class Limiter>
ADC_HD inline typename Model::State reconstruct(const ConstArray4& u, int i,
                                                int j, int dir, Real sgn,
                                                const Limiter& lim) {
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
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void compute_face_fluxes(const Model& model, const MultiFab& U, const MultiFab& aux,
                         MultiFab& Fx, MultiFab& Fy, Real dx = 0, Real dy = 0) {
  const Limiter lim{};
  const NumericalFlux nflux{};
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    Array4 fx = Fx.fab(li).array();
    Array4 fy = Fy.fab(li).array();
    const Box2D v = U.box(li);
    for_each_cell(xface_box(v), [=] ADC_HD(int i, int j) {
      const auto L = reconstruct<Model>(u, i - 1, j, 0, +1, lim);
      const auto Rr = reconstruct<Model>(u, i, j, 0, -1, lim);
      const auto F = nflux(model, L, load_aux(ax, i - 1, j), Rr, load_aux(ax, i, j), 0);
      for (int c = 0; c < Model::n_vars; ++c) fx(i, j, c) = F[c];
      if constexpr (DiffusiveModel<Model>) {
        const Real nu = model.diffusivity();
        for (int c = 0; c < Model::n_vars; ++c)
          fx(i, j, c) += -nu * (u(i, j, c) - u(i - 1, j, c)) / dx;
      }
    });
    for_each_cell(yface_box(v), [=] ADC_HD(int i, int j) {
      const auto L = reconstruct<Model>(u, i, j - 1, 1, +1, lim);
      const auto Rr = reconstruct<Model>(u, i, j, 1, -1, lim);
      const auto F = nflux(model, L, load_aux(ax, i, j - 1), Rr, load_aux(ax, i, j), 1);
      for (int c = 0; c < Model::n_vars; ++c) fy(i, j, c) = F[c];
      if constexpr (DiffusiveModel<Model>) {
        const Real nu = model.diffusivity();
        for (int c = 0; c < Model::n_vars; ++c)
          fy(i, j, c) += -nu * (u(i, j, c) - u(i, j - 1, c)) / dy;
      }
    });
  }
}

// assemble_rhs<Limiter, NumericalFlux> : R = -div Fhat + S. Le limiteur (pente de
// reconstruction) ET le flux numerique sont des parametres de template, par defaut
// MUSCL au choix de l'appelant + Rusanov. Tous deux device-callable (ADC_HD).
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void assemble_rhs(const Model& model, const MultiFab& U, const MultiFab& aux,
                  const Geometry& geom, MultiFab& R) {
  const Real dx = geom.dx(), dy = geom.dy();
  const Limiter lim{};
  const NumericalFlux nflux{};
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    Array4 r = R.fab(li).array();
    const Box2D v = R.box(li);
    for_each_cell(v, [=] ADC_HD(int i, int j) {
      const Aux Ac = load_aux(ax, i, j);
      const Aux Axm = load_aux(ax, i - 1, j);
      const Aux Axp = load_aux(ax, i + 1, j);
      const Aux Aym = load_aux(ax, i, j - 1);
      const Aux Ayp = load_aux(ax, i, j + 1);

      // faces x : reconstruction des etats de part et d'autre de chaque face
      const auto Lxm = reconstruct<Model>(u, i - 1, j, 0, +1, lim);
      const auto Rxm = reconstruct<Model>(u, i, j, 0, -1, lim);
      const auto Lxp = reconstruct<Model>(u, i, j, 0, +1, lim);
      const auto Rxp = reconstruct<Model>(u, i + 1, j, 0, -1, lim);
      const auto Fxm = nflux(model, Lxm, Axm, Rxm, Ac, 0);
      const auto Fxp = nflux(model, Lxp, Ac, Rxp, Axp, 0);

      // faces y
      const auto Lym = reconstruct<Model>(u, i, j - 1, 1, +1, lim);
      const auto Rym = reconstruct<Model>(u, i, j, 1, -1, lim);
      const auto Lyp = reconstruct<Model>(u, i, j, 1, +1, lim);
      const auto Ryp = reconstruct<Model>(u, i, j + 1, 1, -1, lim);
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
    });
  }
}

}  // namespace adc
