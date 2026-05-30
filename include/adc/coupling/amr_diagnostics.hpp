#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence
#include <adc/mesh/multifab.hpp>

#include <algorithm>
#include <cmath>

// Diagnostics extraits des coupleurs (responsabilite c : masse, vitesse de derive).
// Free functions a portee de namespace (meme raison que detail:: dans coupler.hpp :
// seam GPU, un lambda etendu ne peut pas vivre dans une methode privee). Ici ce sont
// des boucles hote (device_fence puis acces hote), deplacees TEL QUEL : meme ordre
// d'iteration (j externe, i interne), memes constantes (1e-12), meme device_fence.
//
// mass()/max_drift_speed() des coupleurs deviennent des one-liners delegant ici.

namespace adc {

// masse de la composante 0 sur le niveau grossier (box unique). dV = dx * dy,
// passes tels que geom_.dx() / geom_.dy() pour rester bit-identique.
inline Real amr_mass(const MultiFab& coarse, const Box2D& dom, Real dx, Real dy) {
  device_fence();
  const ConstArray4 u = coarse.fab(0).const_array();
  const int nx = dom.nx(), ny = dom.ny();
  const Real dV = dx * dy;
  Real M = 0;
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) M += u(i, j, 0) * dV;
  return M;
}

// vitesse de derive max |grad phi| / B0 sur le niveau grossier (box unique).
// aux0 = aux_[0], composantes 1 et 2 = (d phi/dx, d phi/dy).
inline Real amr_max_drift_speed(const MultiFab& aux0, const Box2D& dom, Real B0) {
  device_fence();
  const ConstArray4 a = aux0.fab(0).const_array();
  const int nx = dom.nx(), ny = dom.ny();
  Real v = 0;
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i)
      v = std::max(v, std::hypot(a(i, j, 1), a(i, j, 2)) / B0);
  return std::max(v, Real(1e-12));
}

}  // namespace adc
