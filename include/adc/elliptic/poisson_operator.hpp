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
// Convention : on resout le laplacien lap(phi) = f. Le second membre vient du
// modele, f = model.elliptic_rhs (p.ex. une densite de charge signee).
//
// Masque optionnel (embedded boundary) : un MultiFab mask (1 = actif, 0 =
// conducteur) fige phi=0 dans les cellules conductrices. Le lisseur et le residu
// sautent ces cellules ; les cellules actives voisines lisent phi=0 chez le
// conducteur, ce qui impose la condition de Dirichlet sur la frontiere.
//
// Coefficients CUT-CELL optionnels (coef) : par defaut le stencil est uniforme
// (diag = 2/dx^2 + 2/dy^2), ce qui place la CL Dirichlet sur le contour en ESCALIER
// des centres de cellules (erreur O(1) au bord embedded). Si un champ coef a 5
// composantes est fourni (w_xm, w_xp, w_ym, w_yp, w_diag, par cellule active), le
// stencil les utilise : ce sont les poids de Shortley-Weller calcules a partir des
// DISTANCES reelles au bord (cellules coupees), imposant phi=0 sur le CERCLE et non
// l'escalier (ordre 2 jusqu'au bord). Loin du bord (theta=1 partout) les coef valent
// 1/dx^2, 1/dy^2, 2/dx^2+2/dy^2 -> identiques au stencil uniforme. coef==nullptr
// redonne EXACTEMENT le chemin historique (bit-identique).
//
// Le balayage red-black est parallelisable : avec le stencil 5 points, une
// cellule rouge ne depend que de cellules noires.

namespace adc {

inline void apply_laplacian(const MultiFab& phi, const Geometry& geom,
                            MultiFab& lap, const MultiFab* coef = nullptr) {
  const Real idx2 = Real(1) / (geom.dx() * geom.dx());
  const Real idy2 = Real(1) / (geom.dy() * geom.dy());
  for (int li = 0; li < phi.local_size(); ++li) {
    const ConstArray4 p = phi.fab(li).const_array();
    Array4 L = lap.fab(li).array();
    const Box2D v = lap.box(li);
    const bool hc = coef != nullptr;
    const ConstArray4 cf = hc ? coef->fab(li).const_array() : ConstArray4{};
    for_each_cell(v, [=] ADC_HD(int i, int j) {
      if (hc)
        L(i, j) = cf(i, j, 1) * p(i + 1, j) + cf(i, j, 0) * p(i - 1, j) +
                  cf(i, j, 3) * p(i, j + 1) + cf(i, j, 2) * p(i, j - 1) -
                  cf(i, j, 4) * p(i, j);
      else
        L(i, j) = (p(i + 1, j) - 2 * p(i, j) + p(i - 1, j)) * idx2 +
                  (p(i, j + 1) - 2 * p(i, j) + p(i, j - 1)) * idy2;
    });
  }
}

// res = f - laplacien(phi) sur les cellules actives, 0 sur les conductrices.
inline void poisson_residual(MultiFab& phi, const MultiFab& f,
                             const Geometry& geom, const BCRec& bc,
                             MultiFab& res, const MultiFab* mask = nullptr,
                             const MultiFab* coef = nullptr) {
  device_fence();  // GPU : phi a pu etre ecrit par un kernel (lisseur) ; on
                   // attend avant la lecture hote de fill_ghosts.
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
    const bool hc = coef != nullptr;
    const ConstArray4 cf = hc ? coef->fab(li).const_array() : ConstArray4{};
    for_each_cell(v, [=] ADC_HD(int i, int j) {
      if (hm && mk(i, j) == Real(0)) {
        r(i, j) = 0;
        return;
      }
      const Real lap =
          hc ? (cf(i, j, 1) * p(i + 1, j) + cf(i, j, 0) * p(i - 1, j) +
                cf(i, j, 3) * p(i, j + 1) + cf(i, j, 2) * p(i, j - 1) -
                cf(i, j, 4) * p(i, j))
             : ((p(i + 1, j) - 2 * p(i, j) + p(i - 1, j)) * idx2 +
                (p(i, j + 1) - 2 * p(i, j) + p(i, j - 1)) * idy2);
      r(i, j) = ff(i, j) - lap;
    });
  }
}

namespace detail {
inline void gs_color(MultiFab& phi, const MultiFab& f, const Geometry& geom,
                     int color, const MultiFab* mask, const MultiFab* coef) {
  const Real idx2 = Real(1) / (geom.dx() * geom.dx());
  const Real idy2 = Real(1) / (geom.dy() * geom.dy());
  const Real diag0 = 2 * idx2 + 2 * idy2;
  for (int li = 0; li < phi.local_size(); ++li) {
    Array4 p = phi.fab(li).array();
    const ConstArray4 ff = f.fab(li).const_array();
    const Box2D v = phi.box(li);
    const bool hm = mask != nullptr;
    const ConstArray4 mk = hm ? mask->fab(li).const_array() : ConstArray4{};
    const bool hc = coef != nullptr;
    const ConstArray4 cf = hc ? coef->fab(li).const_array() : ConstArray4{};
    for_each_cell(v, [=] ADC_HD(int i, int j) {
      if (((i + j) & 1) != color) return;
      if (hm && mk(i, j) == Real(0)) return;  // conducteur : fige phi=0
      Real off, diag;
      if (hc) {  // stencil cut-cell (Shortley-Weller) ; voisin conducteur = phi=0 sur le cercle
        off = cf(i, j, 1) * p(i + 1, j) + cf(i, j, 0) * p(i - 1, j) +
              cf(i, j, 3) * p(i, j + 1) + cf(i, j, 2) * p(i, j - 1);
        diag = cf(i, j, 4);
      } else {
        off = (p(i + 1, j) + p(i - 1, j)) * idx2 +
              (p(i, j + 1) + p(i, j - 1)) * idy2;
        diag = diag0;
      }
      p(i, j) = (off - ff(i, j)) / diag;
    });
  }
}
}  // namespace detail

inline void gs_rb_sweep(MultiFab& phi, const MultiFab& f, const Geometry& geom,
                        const BCRec& bc, const MultiFab* mask = nullptr,
                        const MultiFab* coef = nullptr) {
  device_fence();  // attend le kernel precedent avant la lecture hote des halos
  fill_ghosts(phi, geom.domain, bc);
  detail::gs_color(phi, f, geom, 0, mask, coef);  // rouge (kernel GPU)
  device_fence();  // le balayage noir lit les valeurs rouges via fill_ghosts hote
  fill_ghosts(phi, geom.domain, bc);
  detail::gs_color(phi, f, geom, 1, mask, coef);  // noir
}

inline void gs_smooth(MultiFab& phi, const MultiFab& f, const Geometry& geom,
                      const BCRec& bc, int nsweeps, const MultiFab* mask = nullptr,
                      const MultiFab* coef = nullptr) {
  for (int s = 0; s < nsweeps; ++s) gs_rb_sweep(phi, f, geom, bc, mask, coef);
}

// Force phi=0 dans les cellules conductrices (mask==0).
inline void zero_conductor(MultiFab& phi, const MultiFab& mask) {
  for (int li = 0; li < phi.local_size(); ++li) {
    Array4 p = phi.fab(li).array();
    const ConstArray4 mk = mask.fab(li).const_array();
    const Box2D v = phi.box(li);
    for_each_cell(v, [=] ADC_HD(int i, int j) {
      if (mk(i, j) == Real(0)) p(i, j) = 0;
    });
  }
}

}  // namespace adc
