#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>

// Operateur de Poisson 5 points et lisseur Gauss-Seidel red-black, briques de
// la multigrille geometrique maison.
//
// Convention : on resout le laplacien lap(phi) = f. Pour le diocotron,
// f = model.elliptic_rhs (densite de charge signee). Le solveur ne connait que
// phi, f, la geometrie et les CL.
//
// Le balayage red-black est parallelisable : avec le stencil 5 points, une
// cellule rouge (i+j pair) ne depend que de cellules noires, et inversement.
// On remplit les ghosts avant chaque couleur (porte les mises a jour des
// voisins entre boxes / rangs).

namespace adc {

// lap = laplacien(phi) sur les cellules valides. phi : ghosts deja remplis.
inline void apply_laplacian(const MultiFab& phi, const Geometry& geom,
                            MultiFab& lap) {
  const Real idx2 = Real(1) / (geom.dx() * geom.dx());
  const Real idy2 = Real(1) / (geom.dy() * geom.dy());
  for (int li = 0; li < phi.local_size(); ++li) {
    const ConstArray4 p = phi.fab(li).const_array();
    Array4 L = lap.fab(li).array();
    const Box2D v = lap.box(li);
    for_each_cell(v, [=](int i, int j) {
      L(i, j) = (p(i + 1, j) - 2 * p(i, j) + p(i - 1, j)) * idx2 +
                (p(i, j + 1) - 2 * p(i, j) + p(i, j - 1)) * idy2;
    });
  }
}

// res = f - laplacien(phi). Remplit d'abord les ghosts de phi.
inline void poisson_residual(MultiFab& phi, const MultiFab& f,
                             const Geometry& geom, const BCRec& bc,
                             MultiFab& res) {
  fill_ghosts(phi, geom.domain, bc);
  const Real idx2 = Real(1) / (geom.dx() * geom.dx());
  const Real idy2 = Real(1) / (geom.dy() * geom.dy());
  for (int li = 0; li < phi.local_size(); ++li) {
    const ConstArray4 p = phi.fab(li).const_array();
    const ConstArray4 ff = f.fab(li).const_array();
    Array4 r = res.fab(li).array();
    const Box2D v = res.box(li);
    for_each_cell(v, [=](int i, int j) {
      const Real lap = (p(i + 1, j) - 2 * p(i, j) + p(i - 1, j)) * idx2 +
                       (p(i, j + 1) - 2 * p(i, j) + p(i, j - 1)) * idy2;
      r(i, j) = ff(i, j) - lap;
    });
  }
}

namespace detail {
inline void gs_color(MultiFab& phi, const MultiFab& f, const Geometry& geom,
                     int color) {
  const Real idx2 = Real(1) / (geom.dx() * geom.dx());
  const Real idy2 = Real(1) / (geom.dy() * geom.dy());
  const Real diag = 2 * idx2 + 2 * idy2;
  for (int li = 0; li < phi.local_size(); ++li) {
    Array4 p = phi.fab(li).array();
    const ConstArray4 ff = f.fab(li).const_array();
    const Box2D v = phi.box(li);
    for_each_cell(v, [=](int i, int j) {
      if (((i + j) & 1) == color) {
        const Real off = (p(i + 1, j) + p(i - 1, j)) * idx2 +
                         (p(i, j + 1) + p(i, j - 1)) * idy2;
        p(i, j) = (off - ff(i, j)) / diag;
      }
    });
  }
}
}  // namespace detail

// Un balayage Gauss-Seidel red-black complet (rouge puis noir).
inline void gs_rb_sweep(MultiFab& phi, const MultiFab& f, const Geometry& geom,
                        const BCRec& bc) {
  fill_ghosts(phi, geom.domain, bc);
  detail::gs_color(phi, f, geom, 0);  // rouge : (i+j) pair
  fill_ghosts(phi, geom.domain, bc);
  detail::gs_color(phi, f, geom, 1);  // noir : (i+j) impair
}

inline void gs_smooth(MultiFab& phi, const MultiFab& f, const Geometry& geom,
                      const BCRec& bc, int nsweeps) {
  for (int s = 0; s < nsweeps; ++s) gs_rb_sweep(phi, f, geom, bc);
}

}  // namespace adc
