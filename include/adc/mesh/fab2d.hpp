/// @file
/// @brief Fab2D : donnees mono-grille sur une Box2D (equivalent maison du FArrayBox d'AMReX) ;
///        Array4 / ConstArray4 : handles legers POD device-copyables sur ce buffer.
///
/// Buffer contigu couvrant la box VALIDE etendue de n_ghost couches, avec n_comp composantes.
/// Layout composante-LENTE (comme AMReX Array4) : pour une composante donnee, le plan (i, j) est
/// contigu avec i en index rapide -> chaque variable est une tranche SoA contigue (bonne
/// vectorisation par variable). Index : c*(nx_tot*ny_tot) + (j-jg0)*nx_tot + (i-ig0). Array4 /
/// ConstArray4 sont des handles legers (pointeur brut + strides) trivialement copiables et
/// capturables PAR VALEUR dans un fonctor (semantique d'une vue Kokkos) : on capture le handle, pas
/// le Fab ; operator() est ADC_HD (device-callable). Le stockage vit en memoire UNIFIEE (cf.
/// allocator.hpp).

#pragma once

#include <adc/core/allocator.hpp>
#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>

#include <cassert>
#include <cstdint>
#include <vector>

// Fab2D : donnees mono-grille sur une Box2D, l'equivalent maison du FArrayBox
// d'AMReX. Buffer contigu couvrant la box valide etendue de n_ghost couches,
// avec n_comp composantes.
//
// Layout composante-lente (comme AMReX Array4) : pour une composante donnee, le
// plan (i, j) est contigu avec i en index rapide. Chaque variable est donc une
// tranche SoA contigue, bon pour la vectorisation sur une variable.
//   idx(i, j, c) = c * (nx_tot * ny_tot) + (j - jg0) * nx_tot + (i - ig0)
//
// Array4 / ConstArray4 sont des handles legers (pointeur brut + strides) :
// trivialement copiables, capturables par valeur dans un fonctor. C'est la
// piece qui calque la semantique d'une vue Kokkos : on capture le handle, pas
// le Fab, et le corps de kernel ne voit qu'un pointeur et des strides.

namespace adc {

/// Handle d'ECRITURE POD (pointeur brut + strides) sur le buffer d'un Fab2D, indexe par (i, j, c)
/// EN INDICES GLOBAUX (ig0/jg0 = coin bas de la box etendue). Trivialement copiable, capturable par
/// valeur dans un kernel device. INVARIANT : ne possede RIEN ; valide tant que le Fab source l'est.
struct Array4 {
  Real* p{nullptr};
  int nx_tot{0};
  std::int64_t comp_stride{0};
  int ig0{0}, jg0{0};  // indices globaux du coin bas de la box etendue

  /// Reference a la cellule (i, j) de la composante c (indices globaux). ADC_HD. Aucun controle
  /// de bornes (chemin chaud / device) : l'appelant garantit (i, j, c) dans la box etendue.
  ADC_HD Real& operator()(int i, int j, int c = 0) const {
    return p[c * comp_stride + static_cast<std::int64_t>(j - jg0) * nx_tot + (i - ig0)];
  }
};

/// Handle de LECTURE seule (pendant const d'Array4) : meme layout et meme contrat (POD
/// device-copyable, indices globaux, sans controle de bornes). ADC_HD.
struct ConstArray4 {
  const Real* p{nullptr};
  int nx_tot{0};
  std::int64_t comp_stride{0};
  int ig0{0}, jg0{0};

  /// Valeur de la cellule (i, j) de la composante c (indices globaux). ADC_HD, aucun controle de bornes.
  ADC_HD Real operator()(int i, int j, int c = 0) const {
    return p[c * comp_stride + static_cast<std::int64_t>(j - jg0) * nx_tot + (i - ig0)];
  }
};

/// Donnees mono-grille sur une Box2D : box VALIDE + ng couches de ghosts, ncomp composantes, layout
/// composante-lente. POSSEDE son buffer (memoire unifiee). On expose des handles Array4 /
/// ConstArray4 pour les kernels (capture par valeur), jamais le Fab lui-meme.
class Fab2D {
 public:
  Fab2D() = default;

  /// Alloue la box valide etendue de ng ghosts, ncomp composantes, initialisee a 0.
  Fab2D(const Box2D& valid, int ncomp, int ng)
      : valid_(valid),
        ng_(ng),
        ncomp_(ncomp),
        gbox_(valid.grow(ng)),
        nx_tot_(gbox_.nx()),
        ny_tot_(gbox_.ny()),
        data_(static_cast<std::int64_t>(nx_tot_) * ny_tot_ * ncomp, Real{0}) {}

  /// Box VALIDE (hors ghosts).
  const Box2D& box() const { return valid_; }
  /// Box etendue (valide + ng ghosts) = empreinte memoire reelle.
  const Box2D& grown_box() const { return gbox_; }
  /// Nombre de composantes.
  int ncomp() const { return ncomp_; }
  /// Nombre de couches de ghosts.
  int n_ghost() const { return ng_; }
  /// Taille du buffer (nx_tot * ny_tot * ncomp).
  std::int64_t size() const { return static_cast<std::int64_t>(data_.size()); }

  /// Acces HOTE (i, j, c) en ecriture (assert de bornes en debug). Ne pas appeler dans un kernel
  /// device : passer par array() (handle POD).
  Real& operator()(int i, int j, int c = 0) { return data_[idx(i, j, c)]; }
  /// Acces HOTE (i, j, c) en lecture (assert de bornes en debug).
  Real operator()(int i, int j, int c = 0) const { return data_[idx(i, j, c)]; }

  /// Handle d'ECRITURE (POD device-copyable) sur ce Fab. Valide tant que le Fab vit.
  Array4 array() {
    return Array4{data_.data(), nx_tot_,
                  static_cast<std::int64_t>(nx_tot_) * ny_tot_, gbox_.lo[0], gbox_.lo[1]};
  }
  /// Handle de LECTURE (POD device-copyable) sur ce Fab. Valide tant que le Fab vit.
  ConstArray4 const_array() const {
    return ConstArray4{data_.data(), nx_tot_,
                       static_cast<std::int64_t>(nx_tot_) * ny_tot_, gbox_.lo[0],
                       gbox_.lo[1]};
  }

  /// Pointeur brut sur le buffer (passe directement a MPI en memoire unifiee, p.ex.).
  Real* data() { return data_.data(); }
  const Real* data() const { return data_.data(); }
  /// Remplit tout le buffer (valides + ghosts) avec la valeur v.
  void set_val(Real v) { std::fill(data_.begin(), data_.end(), v); }

 private:
  // index lineaire (i, j, c) dans le layout composante-lente ; assert de bornes en debug.
  std::int64_t idx(int i, int j, int c) const {
    assert(gbox_.contains(i, j) && c >= 0 && c < ncomp_);
    return c * static_cast<std::int64_t>(nx_tot_) * ny_tot_ +
           static_cast<std::int64_t>(j - gbox_.lo[1]) * nx_tot_ + (i - gbox_.lo[0]);
  }

  Box2D valid_{};
  int ng_{0};
  int ncomp_{1};
  Box2D gbox_{};
  int nx_tot_{0}, ny_tot_{0};
  // stockage : hote (std::allocator) ou memoire unifiee CUDA (cf. allocator.hpp).
  std::vector<Real, fab_allocator<Real>> data_{};
};

}  // namespace adc
