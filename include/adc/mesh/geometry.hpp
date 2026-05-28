#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>

// Geometry : la correspondance entre l'espace d'indices (Box2D) et l'espace
// physique d'un niveau. Domaine physique fixe, pas de maille dx/dy decroissant
// avec le raffinement. Centres de cellule definis pour tout indice, y compris
// les ghosts (indices negatifs).

namespace adc {

struct Geometry {
  Box2D domain{};
  Real xlo = 0, xhi = 1, ylo = 0, yhi = 1;

  Real dx() const { return (xhi - xlo) / domain.nx(); }
  Real dy() const { return (yhi - ylo) / domain.ny(); }
  Real x_cell(int i) const { return xlo + (i + Real(0.5)) * dx(); }
  Real y_cell(int j) const { return ylo + (j + Real(0.5)) * dy(); }

  // Meme extent physique, domaine d'indices raffine.
  Geometry refine(int r) const { return Geometry{domain.refine(r), xlo, xhi, ylo, yhi}; }
};

}  // namespace adc
