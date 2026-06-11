/// @file
/// @brief MultiFab : champ DISTRIBUE sur un niveau (equivalent du MultiFab d'AMReX).
///
/// Porte le decoupage (BoxArray), la repartition (DistributionMapping), le nombre de composantes et
/// de ghosts, et n'alloue que les Fab2D POSSEDES par ce rang. C'est ici que vit le parallelisme de
/// donnees ; la couche physique ne le voit jamais. L'iteration se fait sur les fabs LOCAUX :
/// for (int li = 0; li < mf.local_size(); ++li) { auto a = mf.fab(li).array(); for_each_cell(...); }.
/// sync_host()/sync_device() encodent l'intention d'acces (residence des donnees, cf. for_each.hpp) ;
/// sous memoire unifiee sync_host = device_fence() cible, sync_device = no-op. sum() reduit sur tous
/// les rangs (all_reduce) : Kokkos::Sum reassocie par tuile (deterministe/idempotent, non bit-identique
/// a une somme lexicographique).

#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence, sync_host, sync_device
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

/// Champ distribue sur un niveau : decoupage (BoxArray) + repartition (DistributionMapping) +
/// ncomp composantes + ngrow ghosts. N'alloue que les fabs possedes par CE rang ; l'iteration se
/// fait sur local_size() (indices LOCAUX). global_index/local_index_of font le pont local <-> global.
class MultiFab {
 public:
  MultiFab() = default;

  /// Construit le champ : alloue un Fab2D (ncomp composantes, ngrow ghosts) pour CHAQUE box que ce
  /// rang possede selon dm. Les boxes des autres rangs ne sont pas allouees ici.
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

  /// Decoupage GLOBAL du niveau (toutes les boxes, tous rangs).
  const BoxArray& box_array() const { return ba_; }
  /// Repartition GLOBALE (rang proprietaire par box).
  const DistributionMapping& dmap() const { return dm_; }
  /// Nombre de composantes.
  int ncomp() const { return ncomp_; }
  /// Nombre de couches de ghosts.
  int n_grow() const { return ngrow_; }

  /// Nombre de fabs POSSEDES par ce rang (borne des indices locaux).
  int local_size() const { return static_cast<int>(fabs_.size()); }
  /// Fab local d'indice li (0 <= li < local_size()), en ecriture.
  Fab2D& fab(int li) { return fabs_[li]; }
  /// Fab local d'indice li, en lecture.
  const Fab2D& fab(int li) const { return fabs_[li]; }
  /// Box VALIDE du fab local li.
  const Box2D& box(int li) const { return fabs_[li].box(); }
  /// Indice GLOBAL (dans box_array) du fab local li.
  int global_index(int li) const { return global_of_local_[li]; }
  // Indice local d'une box globale, ou -1 si elle n'est pas sur ce rang.
  /// Indice LOCAL de la box globale @p global, ou -1 si elle n'est pas possedee par ce rang.
  int local_index_of(int global) const { return local_index_[global]; }

  // Residence explicite des donnees de ce MultiFab (cf. for_each.hpp). Encodent
  // l'INTENTION d'acces : sync_host() avant un acces HOTE (operator(), boucle,
  // set_val), sync_device() avant un kernel DEVICE. Sous memoire unifiee
  // (Kokkos::SharedSpace) sync_host() est un device_fence() cible et
  // sync_device() un no-op ; le comportement reste donc bit-identique a l'ancien
  // code. Sur un futur chemin non unifie (buffers separes) ces methodes
  // porteraient la migration deep_copy et le suivi de residence par fab.
  /// Rend la residence HOTE valide (avant un acces hote : operator(), boucle, set_val). Sous memoire
  /// unifiee = un device_fence() cible.
  void sync_host() { adc::sync_host(); }
  /// Marque une residence DEVICE (avant un kernel). No-op sous memoire unifiee.
  void sync_device() { adc::sync_device(); }

  /// Remplit toutes les cellules (valides + ghosts) de tous les fabs locaux avec v. Synchronise la
  /// residence hote au prealable (un kernel a pu ecrire ces fabs).
  void set_val(Real v) {
    sync_host();  // un kernel a pu ecrire ces fabs ; on rend la residence hote
                  // valide avant le remplissage hote (sinon course ecriture
                  // hote/kernel). Sous memoire unifiee = un device_fence().
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
// reduction Kokkos, Kokkos::Sum), puis on agrege les fabs locaux par somme hote
// (peu de fabs, ordre stable par rang) avant MPI_Allreduce ; sans MPI (un seul
// rang) all_reduce_sum est l'identite.
//
// Plus de device_fence() en tete : sous Kokkos parallel_reduce est bloquant cote
// hote et absorbe la barriere. NOTE FP : Kokkos::Sum reassocie la somme par tuile,
// donc sum n'est PAS bit-identique a une somme lexicographique ecrite a la main
// (deterministe/idempotent toutefois : memes donnees, meme espace Kokkos -> memes bits).
/// Somme des cellules VALIDES de la composante comp, reduite sur TOUS les rangs (all_reduce).
/// COLLECTIF sous MPI. NOTE FP : Kokkos::Sum reassocie par tuile (deterministe/idempotent, non
/// bit-identique a une somme lexicographique).
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
