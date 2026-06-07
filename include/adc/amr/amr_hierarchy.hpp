/// @file
/// @brief AmrHierarchy : la pile de niveaux de raffinement (conteneur de la hierarchie AMR).
///
/// Couche : `include/adc/amr` (primitives geometriques AMR).
/// Role : porte, par niveau, le domaine en espace d'indices, le BoxArray et le champ MultiFab.
/// Niveau 0 = le plus grossier ; ratio de raffinement entier fixe (2 par defaut).
/// Contrat : conteneur pur ; le PEUPLEMENT des niveaux fins (tagging + clustering Berger-Rigoutsos)
/// est l'affaire de regrid.hpp. Ici on fournit l'ajout/remplacement explicite d'un niveau, suffisant
/// pour un raffinement statique.
///
/// Invariants :
/// - domain(lev) == domain(lev-1).refine(ref_ratio) : domaines emboites par le ratio fixe ;
/// - les trois vecteurs (domain_, ba_, data_) ont toujours la meme taille = num_levels() ;
/// - remplacer ou nettoyer un niveau invalide et supprime tous les niveaux plus fins.

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

/// Pile de niveaux raffinis (domaine + BoxArray + MultiFab par niveau), niveau 0 le plus grossier.
///
/// Usage : construite avec le seul niveau 0 (grossier), puis etoffee par add_level / install_level
/// (statique) ou par regrid_level (dynamique).
/// Contrat : ref_ratio entier fixe ; domain(lev) decoule de domain(lev-1) par refine(ref_ratio).
/// Invariants : domain_, ba_ et data_ restent de longueur num_levels() ; install/clear tronquent les
/// niveaux plus fins pour garder la pile coherente.
class AmrHierarchy {
 public:
  /// Construit la hierarchie avec son seul niveau 0 (grossier).
  /// @param coarse_domain domaine du niveau 0 en espace d'indices (coins lo/hi INCLUSIFS).
  /// @param max_grid_size taille max de box pour le decoupage en BoxArray.
  /// @param ncomp nombre de composantes du champ par cellule.
  /// @param ngrow nombre de couches de ghosts du MultiFab.
  /// @param ref_ratio ratio de raffinement entier entre niveaux consecutifs (2 par defaut).
  AmrHierarchy(const Box2D& coarse_domain, int max_grid_size, int ncomp,
               int ngrow, int ref_ratio = 2)
      : ref_ratio_(ref_ratio), ncomp_(ncomp), ngrow_(ngrow) {
    BoxArray ba = BoxArray::from_domain(coarse_domain, max_grid_size);
    domain_.push_back(coarse_domain);
    ba_.push_back(ba);
    data_.emplace_back(ba, DistributionMapping(ba.size(), n_ranks()), ncomp,
                       ngrow);
  }

  /// Ajoute un niveau fin defini par son BoxArray (en espace d'indices fin).
  /// @param fine_ba boxes du nouveau niveau, exprimees dans l'espace d'indices raffine.
  /// Le domaine du niveau est deduit par refine(ref_ratio) du domaine du niveau precedent.
  void add_level(const BoxArray& fine_ba) {
    const int lev = num_levels();
    domain_.push_back(domain_[lev - 1].refine(ref_ratio_));
    ba_.push_back(fine_ba);
    data_.emplace_back(fine_ba, DistributionMapping(fine_ba.size(), n_ranks()),
                       ncomp_, ngrow_);
  }

  /// Installe (ajoute ou remplace) un niveau fin a l'indice lev. Utilise par le regrid.
  /// @param lev indice du niveau a installer ; doit verifier 1 <= lev <= num_levels().
  /// @param fine_ba boxes du niveau dans l'espace d'indices raffine.
  /// @param data MultiFab deja construit pour ce niveau (transfere par move).
  /// Remplacer un niveau existant INVALIDE et supprime tous les niveaux plus fins.
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

  /// Supprime tous les niveaux strictement plus fins que lev (no-op si lev est deja le plus fin).
  void clear_above(int lev) {
    if (lev + 1 < num_levels()) {
      domain_.resize(lev + 1);
      ba_.resize(lev + 1);
      data_.resize(lev + 1);
    }
  }

  /// Nombre de niveaux presents (>= 1 : le niveau 0 existe toujours).
  int num_levels() const { return static_cast<int>(data_.size()); }
  /// Ratio de raffinement entier entre niveaux consecutifs.
  int ref_ratio() const { return ref_ratio_; }
  /// Nombre de composantes du champ par cellule.
  int ncomp() const { return ncomp_; }
  /// Nombre de couches de ghosts des MultiFab.
  int n_grow() const { return ngrow_; }

  /// Domaine du niveau lev en espace d'indices (coins lo/hi INCLUSIFS).
  const Box2D& domain(int lev) const { return domain_[lev]; }
  /// BoxArray (decoupage en boxes) du niveau lev.
  const BoxArray& boxes(int lev) const { return ba_[lev]; }
  /// Champ du niveau lev (acces mutable).
  MultiFab& data(int lev) { return data_[lev]; }
  /// Champ du niveau lev (acces const).
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
