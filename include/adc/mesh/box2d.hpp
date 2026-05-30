#pragma once

#include <algorithm>

// Box2D : l'espace d'indices entier d'une grille cartesienne 2D, cellule au
// centre. C'est la brique de base de la pile AMR, inspiree du Box d'AMReX.
// Coins lo / hi inclusifs (convention AMReX). Pure arithmetique
// entiere : aucune donnee, aucun parallelisme, entierement testable.
//
// Les indices peuvent etre negatifs (couches de ghosts) ; coarsen utilise donc
// une division plancher pour rester coherent de part et d'autre de zero.
//
// 2D concret pour coller aux cibles physiques (diocotron, Euler-Poisson). Le
// passage Dim-template est une generalisation mecanique laissee pour plus tard.

namespace adc {

struct Box2D {
  int lo[2]{0, 0};
  int hi[2]{-1, -1};  // vide par defaut (hi < lo)

  static Box2D from_extents(int nx, int ny) {
    return Box2D{{0, 0}, {nx - 1, ny - 1}};
  }

  int length(int d) const { return hi[d] - lo[d] + 1; }
  int nx() const { return length(0); }
  int ny() const { return length(1); }
  long num_cells() const {
    return static_cast<long>(std::max(0, nx())) * std::max(0, ny());
  }
  bool empty() const { return hi[0] < lo[0] || hi[1] < lo[1]; }

  bool contains(int i, int j) const {
    return i >= lo[0] && i <= hi[0] && j >= lo[1] && j <= hi[1];
  }
  bool contains(const Box2D& b) const {
    return !b.empty() && b.lo[0] >= lo[0] && b.hi[0] <= hi[0] &&
           b.lo[1] >= lo[1] && b.hi[1] <= hi[1];
  }

  Box2D grow(int n) const {
    return {{lo[0] - n, lo[1] - n}, {hi[0] + n, hi[1] + n}};
  }
  Box2D grow(int d, int n) const {
    Box2D b = *this;
    b.lo[d] -= n;
    b.hi[d] += n;
    return b;
  }
  Box2D shift(int d, int s) const {
    Box2D b = *this;
    b.lo[d] += s;
    b.hi[d] += s;
    return b;
  }

  // Raffinement cellule-par-cellule : [lo, hi] -> [lo*r, hi*r + r-1].
  Box2D refine(int r) const {
    return {{lo[0] * r, lo[1] * r}, {hi[0] * r + r - 1, hi[1] * r + r - 1}};
  }
  // Inverse : division plancher (gere les indices negatifs des ghosts).
  Box2D coarsen(int r) const {
    return {{div_floor(lo[0], r), div_floor(lo[1], r)},
            {div_floor(hi[0], r), div_floor(hi[1], r)}};
  }

  Box2D intersect(const Box2D& o) const {
    return {{std::max(lo[0], o.lo[0]), std::max(lo[1], o.lo[1])},
            {std::min(hi[0], o.hi[0]), std::min(hi[1], o.hi[1])}};
  }

  bool operator==(const Box2D&) const = default;

 private:
  static int div_floor(int a, int r) {
    int q = a / r, rem = a % r;
    return (rem != 0 && ((rem < 0) != (r < 0))) ? q - 1 : q;
  }
};

}  // namespace adc
