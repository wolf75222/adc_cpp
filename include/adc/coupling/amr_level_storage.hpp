#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/multifab.hpp>

#include <utility>
#include <vector>

// Hierarchie AMR extraite des coupleurs (responsabilite a : stockage des niveaux
// + aux). Le coupleur n'ORDONNE plus que les operations ; ce stack DETIENT la pile
// de niveaux std::vector<Level> et la pile d'aux MultiFab parallele, et porte le
// cablage L_[k].aux = &aux_[k].
//
// Generique sur Level (AmrLevelMF mono-box ou AmrLevelMP multi-box) : les deux
// portent un membre U (MultiFab) et un membre aux (const MultiFab*), seuls champs
// touches ici. La repartition (DistributionMapping) est portee par les MultiFab,
// jamais supposee mono-rang : le stack ne fige pas la repartition.
//
// Invariant d'adresses : aux_ est dimensionne UNE seule fois au ctor puis jamais
// redimensionne (les pointeurs L_[k].aux pointent dans aux_). reattach_aux(k)
// remplace l'element aux_[k] en place (pas de resize) et recable L_[k].aux.

namespace adc {

template <class Level>
class AmrLevelStack {
 public:
  AmrLevelStack(const Box2D& dom, std::vector<Level> levels)
      : dom_(dom), L_(std::move(levels)) {
    nlev_ = static_cast<int>(L_.size());
    aux_.resize(nlev_);  // addresses stables : aux_ n'est plus redimensionne
    for (int k = 0; k < nlev_; ++k) {
      aux_[k] = MultiFab(L_[k].U.box_array(), L_[k].U.dmap(), 3, 1);
      L_[k].aux = &aux_[k];
    }
  }

  std::vector<Level>& levels() { return L_; }
  const std::vector<Level>& levels() const { return L_; }
  MultiFab& coarse() { return L_[0].U; }
  const MultiFab& coarse() const { return L_[0].U; }
  const Box2D& domain() const { return dom_; }
  int nlev() const { return nlev_; }

  std::vector<Level>& L() { return L_; }
  std::vector<MultiFab>& aux() { return aux_; }
  MultiFab& aux(int k) { return aux_[k]; }
  const MultiFab& aux(int k) const { return aux_[k]; }

  // realloc en place de aux_[k] sur la box courante de L_[k].U + recablage du
  // pointeur. Bit-identique au bloc inline d'origine (meme MultiFab(...,3,1)).
  void reattach_aux(int k) {
    aux_[k] = MultiFab(L_[k].U.box_array(), L_[k].U.dmap(), 3, 1);
    L_[k].aux = &aux_[k];
  }

 private:
  Box2D dom_;
  std::vector<Level> L_;
  std::vector<MultiFab> aux_;
  int nlev_ = 0;
};

}  // namespace adc
