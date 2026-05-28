#pragma once

#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>

#include <algorithm>
#include <unordered_map>
#include <vector>

// Hash spatial (cf. bibliographie sect. 3.3) : une grille de bins uniforme
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

class BoxHash {
 public:
  BoxHash(const BoxArray& ba, int bin) : bin_(bin > 0 ? bin : 1) {
    for (int i = 0; i < ba.size(); ++i) {
      const Box2D& b = ba[i];
      for (int by = fdiv(b.lo[1]); by <= fdiv(b.hi[1]); ++by)
        for (int bx = fdiv(b.lo[0]); bx <= fdiv(b.hi[0]); ++bx)
          bins_[key(bx, by)].push_back(i);
    }
  }

  // indices (tries, sans doublon) des boxes susceptibles d'intersecter q.
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
  static long key(int bx, int by) {
    return (static_cast<long>(bx) << 32) |
           (static_cast<long>(static_cast<unsigned>(by)) & 0xffffffffL);
  }

  int bin_;
  std::unordered_map<long, std::vector<int>> bins_;
};

// Taille de bin raisonnable : la plus grande extension de box du BoxArray, de
// sorte que les boxes voisines tombent dans des bins adjacents.
inline int suggest_bin(const BoxArray& ba) {
  int m = 1;
  for (int i = 0; i < ba.size(); ++i)
    m = std::max({m, ba[i].nx(), ba[i].ny()});
  return m;
}

}  // namespace adc
