#pragma once

#include <adc/core/types.hpp>
#include <adc/elliptic/poisson_operator.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/mesh/refinement.hpp>
#include <adc/parallel/comm.hpp>

#include <functional>
#include <utility>
#include <vector>

// Multigrille geometrique maison pour lap(phi) = f.
//
// V-cycle classique : pre-lissage, restriction du residu (average_down) sur une
// grille deux fois plus grossiere, resolution recursive de l'equation de
// correction lap(e) = residu a CL homogenes, prolongation de la correction
// (interpolate) ajoutee a phi, post-lissage. Au niveau le plus grossier, on
// lisse longuement (bottom solve).
//
// La hierarchie de grilles est obtenue en grossissant le domaine par 2 jusqu'a
// une taille minimale. Restriction et prolongation reutilisent les operateurs
// de transfert AMR (average_down / interpolate).

namespace adc {

inline BCRec homogeneous(const BCRec& b) {
  BCRec h = b;
  h.xlo_val = h.xhi_val = h.ylo_val = h.yhi_val = 0;
  return h;
}

class GeometricMG {
 public:
  // active(x, y) : predicat optionnel "cellule active" (interieur du conducteur).
  // Vide => tout actif (pas de paroi embedded).
  GeometricMG(const Geometry& geom, const BoxArray& ba, const BCRec& bc,
              std::function<bool(Real, Real)> active = {}, int min_coarse = 2,
              int nu1 = 2, int nu2 = 2, int nbottom = 50)
      : bc_(bc), active_(std::move(active)), nu1_(nu1), nu2_(nu2),
        nbottom_(nbottom) {
    add_level(geom, ba);
    while (true) {
      const Geometry g = lev_.back().geom;
      if (g.domain.nx() % 2 || g.domain.ny() % 2) break;
      if (g.domain.nx() / 2 < min_coarse || g.domain.ny() / 2 < min_coarse)
        break;
      Geometry gc{g.domain.coarsen(2), g.xlo, g.xhi, g.ylo, g.yhi};
      add_level(gc, coarsen(lev_.back().ba, 2));
    }
    if (active_) {
      // chaque niveau evalue son propre masque depuis le cercle physique
      for (auto& L : lev_) {
        L.mask = MultiFab(L.ba, L.dm, 1, 0);
        for (int li = 0; li < L.mask.local_size(); ++li) {
          Array4 m = L.mask.fab(li).array();
          const Geometry& g = L.geom;
          for_each_cell(L.mask.box(li), [=, this](int i, int j) {
            m(i, j) = active_(g.x_cell(i), g.y_cell(j)) ? Real(1) : Real(0);
          });
        }
      }
    }
  }

  MultiFab& phi() { return lev_[0].phi; }
  MultiFab& rhs() { return lev_[0].rhs; }
  const Geometry& geom() const { return lev_[0].geom; }
  int num_levels() const { return static_cast<int>(lev_.size()); }

  void vcycle() { vcycle_rec(0, bc_); }

  // V-cycles jusqu'a residu relatif < rel_tol (ou max_cycles). Renvoie le
  // nombre de cycles effectues. phi est conserve entre appels (warm start).
  int solve(Real rel_tol, int max_cycles) {
    const Real r0 = current_residual();
    if (r0 <= Real(0)) return 0;
    for (int c = 1; c <= max_cycles; ++c) {
      vcycle();
      if (current_residual() <= rel_tol * r0) return c;
    }
    return max_cycles;
  }

  // Residu courant (norme infinie) au niveau le plus fin.
  Real current_residual() {
    poisson_residual(lev_[0].phi, lev_[0].rhs, lev_[0].geom, bc_, lev_[0].res,
                     mask_ptr(0));
    return norm_inf(lev_[0].res);
  }

 private:
  struct MGLevel {
    Geometry geom;
    BoxArray ba;
    DistributionMapping dm;
    MultiFab phi, rhs, res, mask;
  };

  const MultiFab* mask_ptr(int l) { return active_ ? &lev_[l].mask : nullptr; }

  void add_level(const Geometry& g, const BoxArray& ba) {
    DistributionMapping dm(ba.size(), n_ranks());
    lev_.push_back(MGLevel{g, ba, dm, MultiFab(ba, dm, 1, 1),
                           MultiFab(ba, dm, 1, 0), MultiFab(ba, dm, 1, 0),
                           MultiFab{}});
  }

  void vcycle_rec(int l, const BCRec& bc) {
    MGLevel& L = lev_[l];
    const MultiFab* mk = mask_ptr(l);
    gs_smooth(L.phi, L.rhs, L.geom, bc, nu1_, mk);

    if (l + 1 == static_cast<int>(lev_.size())) {
      gs_smooth(L.phi, L.rhs, L.geom, bc, nbottom_, mk);  // bottom solve
      if (mk) zero_conductor(L.phi, L.mask);
      return;
    }

    poisson_residual(L.phi, L.rhs, L.geom, bc, L.res, mk);
    MGLevel& C = lev_[l + 1];
    average_down(L.res, C.rhs, 2);  // restriction du residu
    C.phi.set_val(0.0);
    vcycle_rec(l + 1, homogeneous(bc));

    MultiFab corr(L.ba, L.dm, 1, 0);
    interpolate(C.phi, corr, 2);  // prolongation de la correction
    saxpy(L.phi, Real(1), corr);
    if (mk) zero_conductor(L.phi, L.mask);  // refige le conducteur
    gs_smooth(L.phi, L.rhs, L.geom, bc, nu2_, mk);
  }

  BCRec bc_;
  std::function<bool(Real, Real)> active_;
  int nu1_, nu2_, nbottom_;
  std::vector<MGLevel> lev_;
};

}  // namespace adc
