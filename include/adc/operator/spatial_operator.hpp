#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/operator/reconstruction.hpp>

#include <algorithm>

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

template <class Model>
ADC_HD inline typename Model::State rusanov_flux(const Model& m,
                                          const typename Model::State& UL,
                                          const Aux& AL,
                                          const typename Model::State& UR,
                                          const Aux& AR, int dir) {
  const auto FL = m.flux(UL, AL, dir);
  const auto FR = m.flux(UR, AR, dir);
  const Real sL = m.max_wave_speed(UL, AL, dir);
  const Real sR = m.max_wave_speed(UR, AR, dir);
  const Real alpha = sL > sR ? sL : sR;  // max device-safe (pas de std::max)
  typename Model::State F;
  for (int c = 0; c < Model::n_vars; ++c)
    F[c] = Real(0.5) * (FL[c] + FR[c]) - Real(0.5) * alpha * (UR[c] - UL[c]);
  return F;
}

// Valeur de cellule (i,j) extrapolee vers sa face +dir (sgn=+1) ou -dir (sgn=-1).
// Pour NoSlope (n_ghost==1) : pas de pente, aucune lecture de voisin.
template <class Model, class Limiter>
ADC_HD inline typename Model::State reconstruct(const ConstArray4& u, int i,
                                                int j, int dir, Real sgn,
                                                const Limiter& lim) {
  typename Model::State s = load_state<Model>(u, i, j);
  if constexpr (Limiter::n_ghost > 1) {
    for (int c = 0; c < Model::n_vars; ++c) {
      const Real am = (dir == 0) ? u(i, j, c) - u(i - 1, j, c)
                                 : u(i, j, c) - u(i, j - 1, c);
      const Real ap = (dir == 0) ? u(i + 1, j, c) - u(i, j, c)
                                 : u(i, j + 1, c) - u(i, j, c);
      s[c] += sgn * Real(0.5) * lim(am, ap);
    }
  }
  return s;
}

template <class Limiter = NoSlope, class Model>
void assemble_rhs(const Model& model, const MultiFab& U, const MultiFab& aux,
                  const Geometry& geom, MultiFab& R) {
  const Real dx = geom.dx(), dy = geom.dy();
  const Limiter lim{};
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
      const auto Fxm = rusanov_flux(model, Lxm, Axm, Rxm, Ac, 0);
      const auto Fxp = rusanov_flux(model, Lxp, Ac, Rxp, Axp, 0);

      // faces y
      const auto Lym = reconstruct<Model>(u, i, j - 1, 1, +1, lim);
      const auto Rym = reconstruct<Model>(u, i, j, 1, -1, lim);
      const auto Lyp = reconstruct<Model>(u, i, j, 1, +1, lim);
      const auto Ryp = reconstruct<Model>(u, i, j + 1, 1, -1, lim);
      const auto Fym = rusanov_flux(model, Lym, Aym, Rym, Ac, 1);
      const auto Fyp = rusanov_flux(model, Lyp, Ac, Ryp, Ayp, 1);

      const auto S = model.source(load_state<Model>(u, i, j), Ac);
      for (int c = 0; c < Model::n_vars; ++c)
        r(i, j, c) = S[c] - (Fxp[c] - Fxm[c]) / dx - (Fyp[c] - Fym[c]) / dy;
    });
  }
}

}  // namespace adc
