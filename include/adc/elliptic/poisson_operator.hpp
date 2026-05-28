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
// f = model.elliptic_rhs (densite de charge signee).
//
// Masque optionnel (embedded boundary) : un MultiFab mask (1 = actif, 0 =
// conducteur) fige phi=0 dans les cellules conductrices. Le lisseur et le residu
// sautent ces cellules ; les cellules actives voisines lisent phi=0 chez le
// conducteur, ce qui impose la condition de Dirichlet sur la frontiere
// (en escalier). Le point fixe du lisseur fin definit donc la solution masquee
// correcte, independamment de la qualite de la correction grossiere.
//
// Le balayage red-black est parallelisable : avec le stencil 5 points, une
// cellule rouge ne depend que de cellules noires.

namespace adc {

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

// res = f - laplacien(phi) sur les cellules actives, 0 sur les conductrices.
inline void poisson_residual(MultiFab& phi, const MultiFab& f,
                             const Geometry& geom, const BCRec& bc,
                             MultiFab& res, const MultiFab* mask = nullptr) {
  fill_ghosts(phi, geom.domain, bc);
  const Real idx2 = Real(1) / (geom.dx() * geom.dx());
  const Real idy2 = Real(1) / (geom.dy() * geom.dy());
  for (int li = 0; li < phi.local_size(); ++li) {
    const ConstArray4 p = phi.fab(li).const_array();
    const ConstArray4 ff = f.fab(li).const_array();
    Array4 r = res.fab(li).array();
    const Box2D v = res.box(li);
    const bool hm = mask != nullptr;
    const ConstArray4 mk = hm ? mask->fab(li).const_array() : ConstArray4{};
    for_each_cell(v, [=](int i, int j) {
      if (hm && mk(i, j) == Real(0)) {
        r(i, j) = 0;
        return;
      }
      const Real lap = (p(i + 1, j) - 2 * p(i, j) + p(i - 1, j)) * idx2 +
                       (p(i, j + 1) - 2 * p(i, j) + p(i, j - 1)) * idy2;
      r(i, j) = ff(i, j) - lap;
    });
  }
}

namespace detail {
inline void gs_color(MultiFab& phi, const MultiFab& f, const Geometry& geom,
                     int color, const MultiFab* mask) {
  const Real idx2 = Real(1) / (geom.dx() * geom.dx());
  const Real idy2 = Real(1) / (geom.dy() * geom.dy());
  const Real diag = 2 * idx2 + 2 * idy2;
  for (int li = 0; li < phi.local_size(); ++li) {
    Array4 p = phi.fab(li).array();
    const ConstArray4 ff = f.fab(li).const_array();
    const Box2D v = phi.box(li);
    const bool hm = mask != nullptr;
    const ConstArray4 mk = hm ? mask->fab(li).const_array() : ConstArray4{};
    for_each_cell(v, [=](int i, int j) {
      if (((i + j) & 1) != color) return;
      if (hm && mk(i, j) == Real(0)) return;  // conducteur : fige
      const Real off = (p(i + 1, j) + p(i - 1, j)) * idx2 +
                       (p(i, j + 1) + p(i, j - 1)) * idy2;
      p(i, j) = (off - ff(i, j)) / diag;
    });
  }
}
}  // namespace detail

inline void gs_rb_sweep(MultiFab& phi, const MultiFab& f, const Geometry& geom,
                        const BCRec& bc, const MultiFab* mask = nullptr) {
  fill_ghosts(phi, geom.domain, bc);
  detail::gs_color(phi, f, geom, 0, mask);  // rouge
  fill_ghosts(phi, geom.domain, bc);
  detail::gs_color(phi, f, geom, 1, mask);  // noir
}

inline void gs_smooth(MultiFab& phi, const MultiFab& f, const Geometry& geom,
                      const BCRec& bc, int nsweeps,
                      const MultiFab* mask = nullptr) {
  for (int s = 0; s < nsweeps; ++s) gs_rb_sweep(phi, f, geom, bc, mask);
}

// Force phi=0 dans les cellules conductrices (mask==0).
inline void zero_conductor(MultiFab& phi, const MultiFab& mask) {
  for (int li = 0; li < phi.local_size(); ++li) {
    Array4 p = phi.fab(li).array();
    const ConstArray4 mk = mask.fab(li).const_array();
    const Box2D v = phi.box(li);
    for_each_cell(v, [=](int i, int j) {
      if (mk(i, j) == Real(0)) p(i, j) = 0;
    });
  }
}

}  // namespace adc
