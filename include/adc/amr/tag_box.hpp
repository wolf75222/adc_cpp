#pragma once

#include <adc/mesh/box2d.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

// TagBox : grille dense de marqueurs (0/1) sur une region, entree du clustering
// Berger-Rigoutsos. Stockage i-rapide. Pour MPI plus tard, les tags repartis
// seront rassembles sur cette grille avant clustering (le clustering est bon
// marche face au reste).

namespace adc {

struct TagBox {
  Box2D box{};
  std::vector<char> t{};

  TagBox() = default;
  explicit TagBox(const Box2D& b)
      : box(b),
        t(static_cast<std::size_t>(std::max<long>(0, b.num_cells())), 0) {}

  char& operator()(int i, int j) { return t[idx(i, j)]; }
  char operator()(int i, int j) const { return t[idx(i, j)]; }
  bool tagged(int i, int j) const {
    return box.contains(i, j) && t[idx(i, j)] != 0;
  }

  long count() const {
    long c = 0;
    for (char x : t) c += x;
    return c;
  }

 private:
  std::size_t idx(int i, int j) const {
    return static_cast<std::size_t>(j - box.lo[1]) * box.nx() + (i - box.lo[0]);
  }
};

}  // namespace adc
