/// @file
/// @brief BoxHash : hash spatial pour la recherche rapide des boxes intersectant une requete.
///
/// Une grille de bins uniforme (technique classique de hachage spatial) associe a chaque bin la liste des boxes
/// qui le touchent : trouver les boxes dont la region peut intersecter une box requete devient
/// ~O(1) par requete (la recherche des paires de halos passe de O(N) a ~O(n), n << N). Construit
/// UNE fois par maillage, reutilisable tant qu'il ne change pas. INVARIANT de non-omission : si une
/// box intersecte la requete, elles partagent une cellule donc un bin -> query() renvoie un
/// SUR-ENSEMBLE trie sans doublon, l'appelant teste l'intersection exacte.

#pragma once

#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

// Hash spatial (technique classique de hachage spatial) : une grille de bins uniforme
// associe a chaque bin la liste des boxes qui le touchent. Trouver les boxes
// dont la region valide peut intersecter une box requete devient ~O(1) par
// requete au lieu de balayer tout le BoxArray (la recherche des paires de
// halos passe de O(N) a ~O(n), n << N). Construit une fois par maillage,
// reutilisable tant qu'il ne change pas (amorti sur les pas de temps).
//
// query() renvoie un SUR-ENSEMBLE trie, sans doublon : l'appelant teste
// l'intersection exacte. Garantie de non-omission : si une box intersecte la
// requete, elles partagent au moins une cellule, donc un bin.

namespace adc {

/// Index spatial des boxes d'un BoxArray par grille de bins. Reference la BoxArray par INDICE
/// (les indices retournes par query() sont des indices globaux dans la BoxArray d'origine) ;
/// valide tant que cette BoxArray ne change pas.
class BoxHash {
 public:
  /// Construit l'index : bin = cote d'un bin (cellules) ; bin <= 0 force bin = 1. Chaque box est
  /// inseree dans tous les bins qu'elle recouvre. Cout proportionnel a la surface totale en bins.
  BoxHash(const BoxArray& ba, int bin) : bin_(bin > 0 ? bin : 1) {
    for (int i = 0; i < ba.size(); ++i) {
      const Box2D& b = ba[i];
      for (int by = fdiv(b.lo[1]); by <= fdiv(b.hi[1]); ++by)
        for (int bx = fdiv(b.lo[0]); bx <= fdiv(b.hi[0]); ++bx)
          bins_[key(bx, by)].push_back(i);
    }
  }

  // indices (tries, sans doublon) des boxes susceptibles d'intersecter q.
  /// Indices (TRIES, sans doublon) des boxes susceptibles d'intersecter q : SUR-ENSEMBLE garanti
  /// (aucune box intersectante n'est omise). L'appelant teste l'intersection exacte. Vide si q vide.
  std::vector<int> query(const Box2D& q) const {
    std::vector<int> out;
    if (q.empty()) return out;
    for (int by = fdiv(q.lo[1]); by <= fdiv(q.hi[1]); ++by)
      for (int bx = fdiv(q.lo[0]); bx <= fdiv(q.hi[0]); ++bx) {
        auto it = bins_.find(key(bx, by));
        if (it != bins_.end())
          out.insert(out.end(), it->second.begin(), it->second.end());
      }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
  }

 private:
  // division entiere par bin_ arrondie vers le bas (gere les coords negatives).
  int fdiv(int x) const {
    return x >= 0 ? x / bin_ : -((-x + bin_ - 1) / bin_);
  }
  static std::int64_t key(int bx, int by) {
    return (static_cast<std::int64_t>(bx) << 32) |
           (static_cast<std::int64_t>(static_cast<std::uint32_t>(by)) & INT64_C(0xffffffff));
  }

  int bin_;
  std::unordered_map<std::int64_t, std::vector<int>> bins_;
};

// Taille de bin raisonnable : la plus grande extension de box du BoxArray, de
// sorte que les boxes voisines tombent dans des bins adjacents.
/// Taille de bin recommandee pour un BoxArray : la plus grande extension de box (au moins 1), de
/// sorte que les boxes voisines tombent dans des bins adjacents (compromis memoire / selectivite).
inline int suggest_bin(const BoxArray& ba) {
  int m = 1;
  for (int i = 0; i < ba.size(); ++i)
    m = std::max({m, ba[i].nx(), ba[i].ny()});
  return m;
}

}  // namespace adc
