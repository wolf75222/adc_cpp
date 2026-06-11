/// @file
/// @brief Diagnostics extraits des coupleurs AMR : masse et vitesse de derive max (responsabilite c).
///
/// Free functions a portee de namespace (meme raison que detail:: dans coupler.hpp : seam GPU, un
/// lambda etendu ne peut pas vivre dans une methode privee). amr_mass_mb passe par le seam reducteur
/// (for_each_cell_reduce_sum : vraie reduction Kokkos, Kokkos::Sum reassociee par tuile --
/// deterministe/idempotent). amr_max_drift_speed_mb reste une boucle hote
/// (std::hypot non confirme device sous nvcc ; le router casserait le dernier bit). Les variantes
/// mono-box (...) se ramenent aux variantes _mb (un seul fab couvrant le domaine, bit a bit). AUCUNE
/// reduction MPI ici : le coupleur decide d'all_reduce selon sa politique d'ownership.

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
// Kokkos (Kokkos::Sum), reassociee par tuile -- deterministe/idempotent mais pas
// bit-identique a une somme lexicographique ecrite a la main (cf. for_each.hpp).
// amr_max_drift_speed reste une boucle hote : son noyau utilise std::hypot, dont
// l'appelabilite device sous Kokkos/nvcc n'est pas verifiee ici, et le remplacer par
// sqrt(gx^2+gy^2) changerait le dernier bit. A router par le seam APRES confirmation
// d'une compilation GPU sur ROMEO (sinon regression bit-identique ou de build).

namespace adc {

// --- forme MULTI-BOX (canonique) : somme/max sur les cellules valides de TOUS les fabs
// locaux, SANS reduction MPI (le coupleur decide d'all_reduce ou non selon sa politique
// d'ownership). C'est l'implementation unique ; les variantes mono-box ci-dessous s'y
// ramenent (un seul fab dont la box vaut le domaine -> bit a bit identique). Cela retire
// la duplication entre AmrCoupler (mono-box) et AmrCouplerMP (multi-box / distribue).

// somme locale de u(.,.,0) * dV sur les cellules valides. dV multiplie DANS le noyau.
/// Masse LOCALE : somme de u(.,.,0) * dx * dy sur les cellules valides de TOUS les fabs locaux, SANS
/// reduction MPI (l'appelant decide d'all_reduce). Forme canonique multi-box.
inline Real amr_mass_mb(const MultiFab& coarse, Real dx, Real dy) {
  const Real dV = dx * dy;
  Real M = 0;
  for (int li = 0; li < coarse.local_size(); ++li) {
    const ConstArray4 u = coarse.fab(li).const_array();
    M += for_each_cell_reduce_sum(
        coarse.box(li), [u, dV] ADC_HD(int i, int j) { return u(i, j, 0) * dV; });
  }
  return M;
}

// max local de |grad phi| / B0 (aux comp 1,2 = grad phi). Boucle hote (std::hypot non
// confirme device : voir l'en-tete). SANS plancher (applique par l'appelant).
/// Vitesse de derive max LOCALE : max de |grad phi| / B0 (aux comp 1, 2 = grad phi) sur les cellules
/// valides, SANS plancher (applique par l'appelant) ni reduction MPI. Boucle hote (std::hypot).
inline Real amr_max_drift_speed_mb(const MultiFab& aux0, Real B0) {
  device_fence();
  Real v = 0;
  for (int li = 0; li < aux0.local_size(); ++li) {
    const ConstArray4 a = aux0.fab(li).const_array();
    const Box2D b = aux0.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        v = std::max(v, std::hypot(a(i, j, 1), a(i, j, 2)) / B0);
  }
  return v;
}

// masse de la composante 0 sur le niveau grossier (box unique) : cas degenere de
// amr_mass_mb (un fab couvrant le domaine), bit a bit identique. dom conserve pour l'API.
/// Masse mono-box : cas degenere de amr_mass_mb (bit a bit). @p dom est ignore (conserve pour l'API).
inline Real amr_mass(const MultiFab& coarse, const Box2D& dom, Real dx, Real dy) {
  (void)dom;
  return amr_mass_mb(coarse, dx, dy);
}

// vitesse de derive max sur le grossier (box unique) + plancher 1e-12 (garde-fou CFL).
/// Vitesse de derive max mono-box + plancher 1e-12 (garde-fou CFL). @p dom ignore (conserve pour l'API).
inline Real amr_max_drift_speed(const MultiFab& aux0, const Box2D& dom, Real B0) {
  (void)dom;
  return std::max(amr_max_drift_speed_mb(aux0, B0), Real(1e-12));
}

}  // namespace adc
