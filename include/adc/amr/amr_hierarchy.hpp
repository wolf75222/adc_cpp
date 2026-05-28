#pragma once

#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/parallel/comm.hpp>

#include <cassert>
#include <utility>
#include <vector>

// AmrHierarchy : la pile de niveaux raffines. Niveau 0 = le plus grossier.
// Ratio de raffinement entier fixe (2 par defaut). Chaque niveau porte son
// domaine en espace d'indices, son BoxArray et son champ MultiFab.
//
// La construction des niveaux fins (a partir du tagging et du clustering
// Berger-Rigoutsos) est l'etape suivante (regrid). Ici on fournit le conteneur
// et l'ajout explicite d'un niveau, suffisant pour un raffinement statique.

namespace adc {

class AmrHierarchy {
 public:
  AmrHierarchy(const Box2D& coarse_domain, int max_grid_size, int ncomp,
               int ngrow, int ref_ratio = 2)
      : ref_ratio_(ref_ratio), ncomp_(ncomp), ngrow_(ngrow) {
    BoxArray ba = BoxArray::from_domain(coarse_domain, max_grid_size);
    domain_.push_back(coarse_domain);
    ba_.push_back(ba);
    data_.emplace_back(ba, DistributionMapping(ba.size(), n_ranks()), ncomp,
                       ngrow);
  }

  // Ajoute un niveau fin defini par son BoxArray (en espace d'indices fin).
  void add_level(const BoxArray& fine_ba) {
    const int lev = num_levels();
    domain_.push_back(domain_[lev - 1].refine(ref_ratio_));
    ba_.push_back(fine_ba);
    data_.emplace_back(fine_ba, DistributionMapping(fine_ba.size(), n_ranks()),
                       ncomp_, ngrow_);
  }

  // Installe (ajoute ou remplace) un niveau fin a l'indice lev. Remplacer un
  // niveau invalide les niveaux plus fins, qui sont supprimes. Utilise par le
  // regrid.
  void install_level(int lev, const BoxArray& fine_ba, MultiFab data) {
    assert(lev >= 1 && lev <= num_levels());
    const Box2D dom = domain_[lev - 1].refine(ref_ratio_);
    if (lev == num_levels()) {
      domain_.push_back(dom);
      ba_.push_back(fine_ba);
      data_.push_back(std::move(data));
    } else {
      domain_[lev] = dom;
      ba_[lev] = fine_ba;
      data_[lev] = std::move(data);
      domain_.resize(lev + 1);
      ba_.resize(lev + 1);
      data_.resize(lev + 1);
    }
  }

  // Supprime tous les niveaux strictement plus fins que lev.
  void clear_above(int lev) {
    if (lev + 1 < num_levels()) {
      domain_.resize(lev + 1);
      ba_.resize(lev + 1);
      data_.resize(lev + 1);
    }
  }

  int num_levels() const { return static_cast<int>(data_.size()); }
  int ref_ratio() const { return ref_ratio_; }
  int ncomp() const { return ncomp_; }
  int n_grow() const { return ngrow_; }

  const Box2D& domain(int lev) const { return domain_[lev]; }
  const BoxArray& boxes(int lev) const { return ba_[lev]; }
  MultiFab& data(int lev) { return data_[lev]; }
  const MultiFab& data(int lev) const { return data_[lev]; }

 private:
  int ref_ratio_;
  int ncomp_;
  int ngrow_;
  std::vector<Box2D> domain_{};
  std::vector<BoxArray> ba_{};
  std::vector<MultiFab> data_{};
};

}  // namespace adc
