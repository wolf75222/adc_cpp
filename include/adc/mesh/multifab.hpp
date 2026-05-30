#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence
#include <adc/parallel/comm.hpp>

#include <utility>
#include <vector>

// MultiFab : champ distribue, l'equivalent du MultiFab d'AMReX. Porte le
// decoupage (BoxArray), la repartition (DistributionMapping), le nombre de
// composantes et de ghosts, et n'alloue que les Fab2D possedes par ce rang.
//
// C'est ici que vit le parallelisme de donnees ; la couche physique ne le voit
// jamais. L'iteration se fait sur les fabs locaux :
//   for (int li = 0; li < mf.local_size(); ++li) {
//     auto a = mf.fab(li).array();
//     for_each_cell(mf.box(li), [=](int i, int j) { ... });
//   }

namespace adc {

class MultiFab {
 public:
  MultiFab() = default;

  MultiFab(BoxArray ba, DistributionMapping dm, int ncomp, int ngrow)
      : ba_(std::move(ba)),
        dm_(std::move(dm)),
        ncomp_(ncomp),
        ngrow_(ngrow),
        local_index_(ba_.size(), -1) {
    const int me = my_rank();
    for (int i = 0; i < ba_.size(); ++i) {
      if (dm_[i] == me) {
        local_index_[i] = static_cast<int>(fabs_.size());
        global_of_local_.push_back(i);
        fabs_.emplace_back(ba_[i], ncomp_, ngrow_);
      }
    }
  }

  const BoxArray& box_array() const { return ba_; }
  const DistributionMapping& dmap() const { return dm_; }
  int ncomp() const { return ncomp_; }
  int n_grow() const { return ngrow_; }

  int local_size() const { return static_cast<int>(fabs_.size()); }
  Fab2D& fab(int li) { return fabs_[li]; }
  const Fab2D& fab(int li) const { return fabs_[li]; }
  const Box2D& box(int li) const { return fabs_[li].box(); }
  int global_index(int li) const { return global_of_local_[li]; }
  // Indice local d'une box globale, ou -1 si elle n'est pas sur ce rang.
  int local_index_of(int global) const { return local_index_[global]; }

  void set_val(Real v) {
    device_fence();  // GPU : un kernel a pu ecrire ces fabs ; barriere avant le
                     // remplissage hote (sinon course ecriture hote/kernel).
    for (auto& f : fabs_) f.set_val(v);
  }

 private:
  BoxArray ba_{};
  DistributionMapping dm_{};
  int ncomp_{1};
  int ngrow_{0};
  std::vector<Fab2D> fabs_{};       // fabs possedes localement
  std::vector<int> local_index_{};  // box globale -> indice local (-1 sinon)
  std::vector<int> global_of_local_{};  // indice local -> box globale
};

// Somme des cellules valides de la composante comp, reduite sur tous les rangs
// (all-reduce). Chaque fab local est reduit par for_each_cell_reduce_sum (vraie
// reduction device sous Kokkos, boucle hote en serie/OpenMP), puis on agrege les
// fabs locaux par somme hote (peu de fabs, ordre stable par rang) avant
// MPI_Allreduce ; en serie all_reduce_sum est l'identite.
//
// Plus de device_fence() en tete : sous Kokkos parallel_reduce est bloquant cote
// hote et absorbe la barriere. NOTE FP : sous Kokkos l'ordre de sommation par
// tuile differe de la boucle hote ; sum n'est donc PLUS bit-identique a l'ancien
// sum sur ce backend (ecart au dernier bit). En serie et OpenMP il reste exact.
inline Real sum(const MultiFab& mf, int comp = 0) {
  Real s = 0;
  for (int li = 0; li < mf.local_size(); ++li) {
    const ConstArray4 a = mf.fab(li).const_array();
    s += for_each_cell_reduce_sum(
        mf.box(li), [a, comp] ADC_HD(int i, int j) { return a(i, j, comp); });
  }
  return static_cast<Real>(all_reduce_sum(static_cast<double>(s)));
}

}  // namespace adc
