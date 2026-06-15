/// @file
/// @brief Box2D : l'espace d'indices entier d'une grille cartesienne 2D, cellule au centre.
///
/// Brique de base de la pile AMR, inspiree du Box d'AMReX. Coins lo / hi INCLUSIFS (convention
/// AMReX) ; box VIDE si hi < lo par direction. Pure arithmetique entiere : aucune donnee, aucun
/// parallelisme, entierement testable. Les indices peuvent etre NEGATIFS (couches de ghosts), d'ou
/// la division PLANCHER de coarsen (coherente de part et d'autre de zero). length/nx/ny sont ADC_HD
/// (appeles depuis Geometry::dx()/dy() dans un kernel device). 2D concret pour coller aux cibles
/// physiques ; le passage Dim-template est une generalisation laissee pour plus tard.

#pragma once

#include <adc/core/types.hpp>  // ADC_HD : nx/ny/length appeles depuis Geometry::dx() dans un kernel device

#include <algorithm>
#include <cstdint>

// Box2D : l'espace d'indices entier d'une grille cartesienne 2D, cellule au
// centre. C'est la brique de base de la pile AMR, inspiree du Box d'AMReX.
// Coins lo / hi inclusifs (convention AMReX). Pure arithmetique
// entiere : aucune donnee, aucun parallelisme, entierement testable.
//
// Les indices peuvent etre negatifs (couches de ghosts) ; coarsen utilise donc
// une division plancher pour rester coherent de part et d'autre de zero.
//
// 2D concret pour coller aux cibles physiques (transport scalaire a derive,
// fluide compressible auto-gravitant). Le passage Dim-template est une
// generalisation mecanique laissee pour plus tard.

namespace adc {

/// Espace d'indices entier 2D, cellule au centre. Coins lo/hi INCLUSIFS ; box vide si hi < lo.
/// POD pur (aucune donnee de champ) : trivialement copiable, capturable par valeur dans un kernel.
/// INVARIANT : les indices peuvent etre negatifs (ghosts) ; refine/coarsen sont des bijections par
/// blocs (refine puis coarsen redonne la box, mais coarsen puis refine l'arrondit au bloc).
struct Box2D {
  int lo[2]{0, 0};
  int hi[2]{-1, -1};  // vide par defaut (hi < lo)

  /// Box [0, nx-1] x [0, ny-1] couvrant nx*ny cellules a partir de l'origine d'indices.
  static Box2D from_extents(int nx, int ny) {
    return Box2D{{0, 0}, {nx - 1, ny - 1}};
  }

  // ADC_HD : Geometry::dx()/dy() (eux-memes ADC_HD) lisent domain.nx()/ny() ; un kernel device qui
  // appelle geom.x_cell(i) descend jusqu'ici. Sans ADC_HD c'est un __host__ depuis __device__ -> nvcc
  // rend du GARBAGE (souvent 0) sans erreur. Arithmetique entiere pure, device-safe, hote inchange.
  /// Nombre de cellules dans la direction d (= hi[d] - lo[d] + 1) ; negatif si la box est vide. ADC_HD.
  ADC_HD int length(int d) const { return hi[d] - lo[d] + 1; }
  /// Largeur (direction 0). ADC_HD (appele depuis Geometry::dx() en kernel device).
  ADC_HD int nx() const { return length(0); }
  /// Hauteur (direction 1). ADC_HD (appele depuis Geometry::dy() en kernel device).
  ADC_HD int ny() const { return length(1); }
  /// Nombre total de cellules (nx*ny, plancher 0 par direction) : 0 si la box est vide.
  std::int64_t num_cells() const {
    return static_cast<std::int64_t>(std::max(0, nx())) * std::max(0, ny());
  }
  /// true si la box ne contient aucune cellule (hi < lo dans une direction).
  bool empty() const { return hi[0] < lo[0] || hi[1] < lo[1]; }

  /// true si la cellule (i, j) est dans la box (bornes lo/hi incluses).
  bool contains(int i, int j) const {
    return i >= lo[0] && i <= hi[0] && j >= lo[1] && j <= hi[1];
  }
  /// true si la box b (non vide) est entierement incluse dans *this.
  bool contains(const Box2D& b) const {
    return !b.empty() && b.lo[0] >= lo[0] && b.hi[0] <= hi[0] &&
           b.lo[1] >= lo[1] && b.hi[1] <= hi[1];
  }

  /// Dilate la box de n cellules dans TOUTES les directions (couche de ghosts uniforme).
  Box2D grow(int n) const {
    return {{lo[0] - n, lo[1] - n}, {hi[0] + n, hi[1] + n}};
  }
  /// Dilate de n cellules dans la SEULE direction d (n peut etre negatif pour retrecir).
  Box2D grow(int d, int n) const {
    Box2D b = *this;
    b.lo[d] -= n;
    b.hi[d] += n;
    return b;
  }
  /// Translate la box de s cellules dans la direction d (lo et hi decales du meme s).
  Box2D shift(int d, int s) const {
    Box2D b = *this;
    b.lo[d] += s;
    b.hi[d] += s;
    return b;
  }

  // Raffinement cellule-par-cellule : [lo, hi] -> [lo*r, hi*r + r-1].
  /// Raffine d'un ratio r : chaque cellule devient un bloc r x r ([lo, hi] -> [lo*r, hi*r + r-1]).
  Box2D refine(int r) const {
    return {{lo[0] * r, lo[1] * r}, {hi[0] * r + r - 1, hi[1] * r + r - 1}};
  }
  // Inverse : division plancher (gere les indices negatifs des ghosts).
  /// Grossit d'un ratio r par division PLANCHER de chaque coin (gere les indices negatifs des ghosts).
  Box2D coarsen(int r) const {
    return {{div_floor(lo[0], r), div_floor(lo[1], r)},
            {div_floor(hi[0], r), div_floor(hi[1], r)}};
  }

  /// Intersection des deux boxes (eventuellement vide : hi < lo si elles ne se recouvrent pas).
  Box2D intersect(const Box2D& o) const {
    return {{std::max(lo[0], o.lo[0]), std::max(lo[1], o.lo[1])},
            {std::min(hi[0], o.hi[0]), std::min(hi[1], o.hi[1])}};
  }

  bool operator==(const Box2D&) const = default;

 private:
  // division entiere arrondie vers le bas (gere a < 0), pour coarsen.
  static int div_floor(int a, int r) {
    int q = a / r, rem = a % r;
    return (rem != 0 && ((rem < 0) != (r < 0))) ? q - 1 : q;
  }
};

}  // namespace adc
