#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>

#include <cassert>
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

struct Array4 {
  Real* p{nullptr};
  int nx_tot{0};
  long comp_stride{0};
  int ig0{0}, jg0{0};  // indices globaux du coin bas de la box etendue

  Real& operator()(int i, int j, int c = 0) const {
    return p[c * comp_stride + static_cast<long>(j - jg0) * nx_tot + (i - ig0)];
  }
};

struct ConstArray4 {
  const Real* p{nullptr};
  int nx_tot{0};
  long comp_stride{0};
  int ig0{0}, jg0{0};

  Real operator()(int i, int j, int c = 0) const {
    return p[c * comp_stride + static_cast<long>(j - jg0) * nx_tot + (i - ig0)];
  }
};

class Fab2D {
 public:
  Fab2D() = default;

  Fab2D(const Box2D& valid, int ncomp, int ng)
      : valid_(valid),
        ng_(ng),
        ncomp_(ncomp),
        gbox_(valid.grow(ng)),
        nx_tot_(gbox_.nx()),
        ny_tot_(gbox_.ny()),
        data_(static_cast<long>(nx_tot_) * ny_tot_ * ncomp, Real{0}) {}

  const Box2D& box() const { return valid_; }
  const Box2D& grown_box() const { return gbox_; }
  int ncomp() const { return ncomp_; }
  int n_ghost() const { return ng_; }
  long size() const { return static_cast<long>(data_.size()); }

  Real& operator()(int i, int j, int c = 0) { return data_[idx(i, j, c)]; }
  Real operator()(int i, int j, int c = 0) const { return data_[idx(i, j, c)]; }

  Array4 array() {
    return Array4{data_.data(), nx_tot_,
                  static_cast<long>(nx_tot_) * ny_tot_, gbox_.lo[0], gbox_.lo[1]};
  }
  ConstArray4 const_array() const {
    return ConstArray4{data_.data(), nx_tot_,
                       static_cast<long>(nx_tot_) * ny_tot_, gbox_.lo[0],
                       gbox_.lo[1]};
  }

  Real* data() { return data_.data(); }
  const Real* data() const { return data_.data(); }
  void set_val(Real v) { std::fill(data_.begin(), data_.end(), v); }

 private:
  long idx(int i, int j, int c) const {
    assert(gbox_.contains(i, j) && c >= 0 && c < ncomp_);
    return c * static_cast<long>(nx_tot_) * ny_tot_ +
           static_cast<long>(j - gbox_.lo[1]) * nx_tot_ + (i - gbox_.lo[0]);
  }

  Box2D valid_{};
  int ng_{0};
  int ncomp_{1};
  Box2D gbox_{};
  int nx_tot_{0}, ny_tot_{0};
  std::vector<Real> data_{};
};

}  // namespace adc
