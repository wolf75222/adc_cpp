/// @file
/// @brief Arithmetique de MultiFab (saxpy, lincomb, norm_inf, dot) sur les cellules VALIDES.
///
/// Briques des etages des integrateurs et des solveurs de Krylov. Suppose des layouts IDENTIQUES
/// (meme BoxArray, meme DistributionMapping). Operations point a point -> l'ALIASING est sans danger
/// (x ou y == z autorise). norm_inf / dot passent par le seam reducteur (vraie reduction Kokkos).
/// dot fait un all_reduce COLLECTIF : il DOIT etre appele sur CHAQUE rang (y compris un rang sans
/// box) sous MPI, sinon interblocage. NOTE FP : dot/sum sont reassocies par tuile (Kokkos::Sum,
/// deterministe/idempotent mais non bit-identique a une somme lexicographique, pour tous les espaces
/// Kokkos) ; norm_inf est exact partout. Les kernels sont des FONCTEURS NOMMES device-clean (nvcc cross-TU).

#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/parallel/comm.hpp>  // all_reduce_sum : produit scalaire COLLECTIF (Krylov sous MPI)

#include <algorithm>

// Combinaisons lineaires de MultiFab sur les cellules valides, pour les etages
// des integrateurs. Suppose des layouts identiques (meme BoxArray, meme
// DistributionMapping). Operations point a point, donc l'aliasing (x ou y == z)
// est sans danger.

namespace adc {

namespace detail {
// FONCTEURS NOMMES (et non lambdas ADC_HD) pour les kernels d'arithmetique MultiFab. Meme recette que
// le chemin block (#64) : ces operations sont premiere-instanciees depuis le V-cycle MG, lui-meme tire
// depuis une TU externe (harness/loader natif) ; une lambda etendue a cette place fait buter nvcc sur
// l'emission du kernel device (kernel-stub nul -> segfault Cuda a -O Release sans -g, #93). Corps
// strictement identique aux anciennes lambdas -> bit-identique sur CPU et device.
struct SaxpyKernel {
  Array4 Y;
  ConstArray4 X;
  Real a;
  int c;
  ADC_HD void operator()(int i, int j) const { Y(i, j, c) += a * X(i, j, c); }
};

struct LincombKernel {
  Array4 Z;
  ConstArray4 X, Y;
  Real a, b;
  int c;
  ADC_HD void operator()(int i, int j) const {
    Z(i, j, c) = a * X(i, j, c) + b * Y(i, j, c);
  }
};

// Reducteur |f(i,j,comp)| -> max, passe DIRECTEMENT a reduce_max_cell (aucune lambda etendue
// d'enveloppe, a la difference de for_each_cell_reduce_max). C'est le chemin device-clean documente
// dans for_each.hpp. Signature reducteur (i, j, Real& acc) ; meme Kokkos::Max / meme boucle hote
// sequentielle -> bit-identique a l'ancien norm_inf (max et fabs sans arrondi).
struct NormInfKernel {
  ConstArray4 a;
  int comp;
  ADC_HD void operator()(int i, int j, Real& acc) const {
    const Real v = a(i, j, comp);
    const Real av = v < 0 ? -v : v;
    if (av > acc) acc = av;
  }
};

// Reducteur x(i,j,comp) * y(i,j,comp) -> somme, passe DIRECTEMENT a reduce_sum_cell (aucune lambda
// etendue d'enveloppe). Foncteur NOMME device-clean (meme recette que NormInfKernel) pour le produit
// scalaire du solveur de Krylov, tire d'une TU externe. Signature reducteur (i, j, Real& acc).
struct DotKernel {
  ConstArray4 x, y;
  int comp;
  ADC_HD void operator()(int i, int j, Real& acc) const {
    acc += x(i, j, comp) * y(i, j, comp);
  }
};
}  // namespace detail

// y <- y + a x
/// y <- y + a x sur TOUTES les composantes des cellules valides. Layouts identiques exiges.
inline void saxpy(MultiFab& y, Real a, const MultiFab& x) {
  const int nc = y.ncomp();
  for (int li = 0; li < y.local_size(); ++li) {
    Array4 Y = y.fab(li).array();
    const ConstArray4 X = x.fab(li).const_array();
    const Box2D b = y.fab(li).box();
    for (int c = 0; c < nc; ++c)
      for_each_cell(b, detail::SaxpyKernel{Y, X, a, c});
  }
}

// norme infinie sur les cellules valides d'une composante. Chaque fab local est
// reduit par for_each_cell_reduce_max sur |f(i,j,comp)| (vraie reduction Kokkos,
// Kokkos::Max), agrege par max hote sur les fabs.
//
// Plus de device_fence() en tete : sous Kokkos parallel_reduce est bloquant et
// absorbe la barriere. EXACT partout : max et fabs sont sans arrondi et le max
// est associatif/commutatif en IEEE754, donc bit-identique a l'ancien norm_inf
// quel que soit le backend (l'ordre de reduction ne change aucun bit).
/// Norme infinie max |f(.,.,comp)| sur les cellules valides (LOCALE, sans all_reduce MPI). EXACTE
/// sur tous les backends (max sans arrondi, associatif/commutatif) -> bit-identique partout.
inline Real norm_inf(const MultiFab& mf, int comp = 0) {
  Real m = 0;
  for (int li = 0; li < mf.local_size(); ++li) {
    const ConstArray4 a = mf.fab(li).const_array();
    m = std::max(m, reduce_max_cell(mf.box(li), detail::NormInfKernel{a, comp}));
  }
  return m;  // all-reduce max MPI plus tard (iso-comportement, non ajoute ici)
}

// z <- a x + b y
/// z <- a x + b y sur TOUTES les composantes des cellules valides. Layouts identiques ; aliasing sur.
inline void lincomb(MultiFab& z, Real a, const MultiFab& x, Real b,
                    const MultiFab& y) {
  const int nc = z.ncomp();
  for (int li = 0; li < z.local_size(); ++li) {
    Array4 Z = z.fab(li).array();
    const ConstArray4 X = x.fab(li).const_array();
    const ConstArray4 Y = y.fab(li).const_array();
    const Box2D bb = z.fab(li).box();
    for (int c = 0; c < nc; ++c)
      for_each_cell(bb, detail::LincombKernel{Z, X, Y, a, b, c});
  }
}

// Produit scalaire sum_cells x . y sur les cellules VALIDES de la composante comp, reduit sur tous
// les rangs (all-reduce). Brique des solveurs de Krylov (BiCGStab : rho, alpha, omega, betas). Chaque
// fab local est reduit par reduce_sum_cell (vraie reduction Kokkos, Kokkos::Sum), les fabs locaux
// agreges par somme hote, puis all_reduce_sum agrege les rangs.
//
// COLLECTIF, OBLIGATOIRE SOUS MPI : all_reduce_sum est appele sur CHAQUE rang, y compris un rang
// SANS box (local_size()==0, qui contribue alors 0 a la somme locale). Sans cet appel sur tous les
// rangs, MPI_Allreduce interbloque (collective desynchronisee) ; le solveur de Krylov ne doit donc
// JAMAIS court-circuiter dot() sur un rang vide. En serie all_reduce_sum est l'identite.
//
// NOTE FP (comme sum()) : Kokkos::Sum reassocie la somme par tuile, donc dot n'est pas bit-identique
// a une somme lexicographique (deterministe/idempotent toutefois, tous espaces Kokkos). Sous MPI, l'all-reduce
// rend la MEME valeur a tous les rangs (MPI_SUM sur un meme jeu de contributions locales), donc le
// critere d'arret du Krylov se declenche a la MEME iteration partout (pas de desynchronisation).
/// Produit scalaire Sum_cells x.y sur la composante comp, reduit sur TOUS les rangs (all_reduce).
/// COLLECTIF, OBLIGATOIRE SOUS MPI : doit etre appele sur chaque rang (y compris vide), sinon
/// interblocage. NOTE FP : non bit-identique entre backends sous Kokkos ; l'all-reduce rend la meme
/// valeur a tous les rangs (pas de desynchronisation du critere d'arret Krylov).
inline Real dot(const MultiFab& x, const MultiFab& y, int comp = 0) {
  Real s = 0;
  for (int li = 0; li < x.local_size(); ++li) {
    const ConstArray4 X = x.fab(li).const_array();
    const ConstArray4 Y = y.fab(li).const_array();
    s += reduce_sum_cell(x.box(li), detail::DotKernel{X, Y, comp});
  }
  return static_cast<Real>(all_reduce_sum(static_cast<double>(s)));
}

}  // namespace adc
