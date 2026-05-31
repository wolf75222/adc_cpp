#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence
#include <adc/mesh/multifab.hpp>

#include <algorithm>
#include <cmath>

// Diagnostics extraits des coupleurs (responsabilite c : masse, vitesse de derive).
// Free functions a portee de namespace (meme raison que detail:: dans coupler.hpp :
// seam GPU, un lambda etendu ne peut pas vivre dans une methode privee).
//
// amr_mass passe par le seam reducteur (for_each_cell_reduce_sum) : vraie reduction
// device sous Kokkos, boucle hote lexicographique (j externe, i interne) en serie /
// OpenMP, donc bit-identique a l'ancienne somme sur ces backends (cf. for_each.hpp).
// amr_max_drift_speed reste une boucle hote : son noyau utilise std::hypot, dont
// l'appelabilite device sous Kokkos/nvcc n'est pas verifiee ici, et le remplacer par
// sqrt(gx^2+gy^2) changerait le dernier bit. A router par le seam APRES confirmation
// d'une compilation GPU sur ROMEO (sinon regression bit-identique ou de build).

namespace adc {

// masse de la composante 0 sur le niveau grossier (box unique). dV = dx * dy passes
// tels que geom_.dx() / geom_.dy(). dV multiplie DANS le noyau (somme de u*dV), pas en
// facteur de la somme : somme(u)*dV differerait au dernier bit de somme(u*dV).
inline Real amr_mass(const MultiFab& coarse, const Box2D& dom, Real dx, Real dy) {
  const ConstArray4 u = coarse.fab(0).const_array();
  const int nx = dom.nx(), ny = dom.ny();
  const Real dV = dx * dy;
  return for_each_cell_reduce_sum(Box2D{{0, 0}, {nx - 1, ny - 1}},
                                  [u, dV] ADC_HD(int i, int j) { return u(i, j, 0) * dV; });
}

// vitesse de derive max |grad phi| / B0 sur le niveau grossier (box unique). aux0 =
// aux_[0], composantes 1 et 2 = (d phi/dx, d phi/dy). Boucle hote (std::hypot non
// confirme device : voir l'en-tete).
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
