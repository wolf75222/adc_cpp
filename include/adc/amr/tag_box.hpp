/// @file
/// @brief TagBox : grille dense de marqueurs (0/1) sur une region, entree du clustering Berger-Rigoutsos.
///
/// Couche : `include/adc/amr` (primitives geometriques AMR).
/// Role : structure de marquage des cellules a raffiner, produite par le tagging et consommee par
/// berger_rigoutsos. Pure arithmetique entiere sur indices, aucune physique, aucun parallelisme.
/// Contrat : indexee dans l'espace d'indices de sa box (coins lo/hi INCLUSIFS, convention Box2D) ;
/// stockage dense i-rapide (j-lent).
///
/// Invariants :
/// - une TagBox couvre EXACTEMENT sa box ; l'indexation lineaire suppose i dans [lo[0], hi[0]] et
///   j dans [lo[1], hi[1]] ;
/// - les marqueurs sont 0 (non tague) ou 1 (tague) ;
/// - pour MPI plus tard, les tags repartis seront rassembles sur cette grille avant clustering (le
///   clustering est bon marche face au reste).

#pragma once

#include <adc/mesh/box2d.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

// TagBox : grille dense de marqueurs (0/1) sur une region, entree du clustering
// Berger-Rigoutsos. Stockage i-rapide. Pour MPI plus tard, les tags repartis
// seront rassembles sur cette grille avant clustering (le clustering est bon
// marche face au reste).

namespace adc {

/// Grille dense de marqueurs 0/1 sur une box, entree du clustering Berger-Rigoutsos.
///
/// Usage : remplie par le tagging (tag_cells), eventuellement dilatee (grow_tags) puis fusionnee
/// (tag_union), enfin clusterisee (berger_rigoutsos).
/// Contrat : les acces (i, j) sont dans l'espace d'indices de `box` (coins lo/hi INCLUSIFS).
/// Invariants : `t` a exactement box.num_cells() entrees (0 si la box est vide), rangees i-rapide.
struct TagBox {
  Box2D box{};
  std::vector<char> t{};

  TagBox() = default;
  /// Construit une TagBox couvrant b, tous marqueurs a 0 (buffer dimensionne sur b.num_cells()).
  explicit TagBox(const Box2D& b)
      : box(b),
        t(static_cast<std::size_t>(std::max<std::int64_t>(0, b.num_cells())), 0) {}

  /// Acces en ecriture au marqueur (i, j) ; (i, j) DOIT etre dans box (aucun controle de bornes).
  char& operator()(int i, int j) { return t[idx(i, j)]; }
  /// Acces en lecture au marqueur (i, j) ; (i, j) DOIT etre dans box (aucun controle de bornes).
  char operator()(int i, int j) const { return t[idx(i, j)]; }
  /// true si (i, j) est dans box ET tague ; sur (controle de bornes inclus, contrairement a operator()).
  bool tagged(int i, int j) const {
    return box.contains(i, j) && t[idx(i, j)] != 0;
  }

  /// Nombre de cellules taguees (somme des marqueurs).
  std::int64_t count() const {
    std::int64_t c = 0;
    for (char x : t) c += x;
    return c;
  }

 private:
  // Indice lineaire i-rapide du marqueur (i, j) dans `t` ; suppose (i, j) dans box.
  std::size_t idx(int i, int j) const {
    return static_cast<std::size_t>(j - box.lo[1]) * box.nx() + (i - box.lo[0]);
  }
};

/// Union (OU logique cellule a cellule) de plusieurs TagBox partageant EXACTEMENT la meme box.
///
/// Brique du regrid d'union des tags multi-blocs (regrid conservatif = hierarchie commune,
/// cellules co-localisees, union des tags ; docs/AMR_REGRID_UNION_TAGS_DESIGN.md etape R3).
/// @param parts TagBox a fusionner ; toutes DOIVENT couvrir le meme domaine parent.
/// @return TagBox sur la box commune, marquee la ou au moins un membre l'etait ; liste vide -> TagBox vide.
/// @throws std::runtime_error si une box discorde (l'indexation lineaire melangerait deux geometries).
// Sans dependance physique (quelques lignes) : stockage i-rapide -> simple |= sur le buffer.
inline TagBox tag_union(const std::vector<TagBox>& parts) {
  if (parts.empty()) return TagBox{};
  TagBox out(parts[0].box);
  for (const TagBox& tb : parts) {
    if (tb.box.lo[0] != out.box.lo[0] || tb.box.lo[1] != out.box.lo[1] ||
        tb.box.hi[0] != out.box.hi[0] || tb.box.hi[1] != out.box.hi[1])
      throw std::runtime_error(
          "tag_union : toutes les TagBox doivent partager EXACTEMENT la meme box (meme domaine "
          "parent) pour l'union cellule a cellule");
    const std::size_t n = std::min(out.t.size(), tb.t.size());
    for (std::size_t k = 0; k < n; ++k) out.t[k] |= tb.t[k];
  }
  return out;
}

}  // namespace adc
