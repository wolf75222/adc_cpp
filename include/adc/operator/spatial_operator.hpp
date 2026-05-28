#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>

#include <algorithm>

// Operateur spatial : assemble le residu R(U, aux) = -div F(U, aux) + S(U, aux)
// sur les cellules valides d'un niveau. C'est la fleche "PDE -> systeme d'ODE"
// de la methode des lignes : l'integrateur ne connait que R, pas la
// discretisation.
//
// Premier ordre, flux de Rusanov (local Lax-Friedrichs) :
//   Fhat = 1/2 (F(UL) + F(UR)) - 1/2 alpha (UR - UL),  alpha = max vitesse d'onde
//
// La physique entre uniquement par le PhysicalModel (flux, max_wave_speed,
// source), tous ponctuels. aux (phi, grad phi) alimente flux ET source, ce qui
// fait servir le meme operateur pour diocotron (aux dans le flux) et
// Euler-Poisson (aux dans la source).
//
// Convention aux : composantes [0]=phi, [1]=d phi/dx, [2]=d phi/dy.

namespace adc {

template <class Model>
inline typename Model::State load_state(const ConstArray4& a, int i, int j) {
  typename Model::State u;
  for (int c = 0; c < Model::n_vars; ++c) u[c] = a(i, j, c);
  return u;
}

inline Aux load_aux(const ConstArray4& a, int i, int j) {
  return Aux{a(i, j, 0), a(i, j, 1), a(i, j, 2)};
}

template <class Model>
inline typename Model::State rusanov_flux(const Model& m,
                                          const typename Model::State& UL,
                                          const Aux& AL,
                                          const typename Model::State& UR,
                                          const Aux& AR, int dir) {
  const auto FL = m.flux(UL, AL, dir);
  const auto FR = m.flux(UR, AR, dir);
  const Real alpha = std::max(m.max_wave_speed(UL, AL, dir),
                              m.max_wave_speed(UR, AR, dir));
  typename Model::State F;
  for (int c = 0; c < Model::n_vars; ++c)
    F[c] = Real(0.5) * (FL[c] + FR[c]) - Real(0.5) * alpha * (UR[c] - UL[c]);
  return F;
}

// U et aux doivent avoir leurs ghosts remplis ; R porte le residu sur les
// cellules valides (memes BoxArray / DistributionMapping que U).
template <class Model>
void assemble_rhs(const Model& model, const MultiFab& U, const MultiFab& aux,
                  const Geometry& geom, MultiFab& R) {
  const Real dx = geom.dx(), dy = geom.dy();
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    Array4 r = R.fab(li).array();
    const Box2D v = R.box(li);
    for_each_cell(v, [=](int i, int j) {
      const auto Uc = load_state<Model>(u, i, j);
      const Aux Ac = load_aux(ax, i, j);
      const auto Uxm = load_state<Model>(u, i - 1, j);
      const auto Uxp = load_state<Model>(u, i + 1, j);
      const auto Uym = load_state<Model>(u, i, j - 1);
      const auto Uyp = load_state<Model>(u, i, j + 1);
      const Aux Axm = load_aux(ax, i - 1, j);
      const Aux Axp = load_aux(ax, i + 1, j);
      const Aux Aym = load_aux(ax, i, j - 1);
      const Aux Ayp = load_aux(ax, i, j + 1);

      const auto Fxm = rusanov_flux(model, Uxm, Axm, Uc, Ac, 0);
      const auto Fxp = rusanov_flux(model, Uc, Ac, Uxp, Axp, 0);
      const auto Fym = rusanov_flux(model, Uym, Aym, Uc, Ac, 1);
      const auto Fyp = rusanov_flux(model, Uc, Ac, Uyp, Ayp, 1);
      const auto S = model.source(Uc, Ac);

      for (int c = 0; c < Model::n_vars; ++c)
        r(i, j, c) =
            S[c] - (Fxp[c] - Fxm[c]) / dx - (Fyp[c] - Fym[c]) / dy;
    });
  }
}

}  // namespace adc
