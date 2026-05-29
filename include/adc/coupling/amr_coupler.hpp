#pragma once

#include <adc/core/types.hpp>
#include <adc/elliptic/elliptic_solver.hpp>  // concept EllipticSolver
#include <adc/elliptic/geometric_mg.hpp>
#include <adc/integrator/amr_reflux_mf.hpp>  // AmrLevelMF, amr_step_multilevel_mf, mf_average_down
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/mesh/refinement.hpp>  // coarsen_index

#include <algorithm>
#include <cmath>
#include <vector>

// Stepper couple AMR E x B, PORTE sur la pile MultiFab + le seam. La hierarchie est
// une std::vector<AmrLevelMF> (mono-box par niveau), l'integration passe par
// amr_step_multilevel_mf (sous-cyclage Berger-Oliger + reflux, via compute_face_fluxes
// -> MUSCL/HLL/HLLC/GPU dispo) et l'elliptique par un EllipticSolver (defaut MG).
//
//   sync_down (fin -> grossier) -> Poisson grossier -> aux = grad phi (grossier)
//   + injection vers les fins -> amr_step_multilevel_mf (N niveaux conservatif).
//
// Le critere de raffinement / regrid reste a l'appelant (specifique au probleme) ; il
// manipule levels() pour reconstruire les box fines, le stepper resynchronise aux.
// Generique sur le modele (Diocotron ; tout modele a alpha, n_i0, B0, flux E x B).

namespace adc {

template <class Model, class Elliptic = GeometricMG>
class AmrCoupler {
  static_assert(EllipticSolver<Elliptic>, "Elliptic doit modeler EllipticSolver");

 public:
  AmrCoupler(const Model& model, const Geometry& geom, const BoxArray& ba_coarse,
             const BCRec& bc, std::vector<AmrLevelMF> levels)
      : model_(model),
        geom_(geom),
        dom_(geom.domain),
        mg_(geom, ba_coarse, bc),
        L_(std::move(levels)) {
    nlev_ = static_cast<int>(L_.size());
    aux_.resize(nlev_);  // addresses stables : aux_ n'est plus redimensionne
    for (int k = 0; k < nlev_; ++k) {
      aux_[k] = MultiFab(L_[k].U.box_array(), L_[k].U.dmap(), 3, 1);
      L_[k].aux = &aux_[k];
    }
  }

  std::vector<AmrLevelMF>& levels() { return L_; }
  const std::vector<AmrLevelMF>& levels() const { return L_; }
  MultiFab& coarse() { return L_[0].U; }
  const MultiFab& coarse() const { return L_[0].U; }
  const Box2D& domain() const { return dom_; }

  // moyenne fin -> grossier sur toute la hierarchie (avant le solve Poisson).
  void sync_down() {
    for (int k = nlev_ - 1; k >= 1; --k)
      mf_average_down(L_[k].U, L_[k - 1].U, L_[k - 1].rCI0, L_[k - 1].rCI1,
                      L_[k - 1].rCJ0, L_[k - 1].rCJ1);
  }

  // Poisson grossier + aux = grad phi + injection vers les fins. Reconstruit aux_[k]
  // si un regrid a change une box (les pointeurs L_[k].aux restent valides).
  void compute_aux() {
    for (int k = 0; k < nlev_; ++k)
      if (!(aux_[k].box(0) == L_[k].U.box(0))) {
        aux_[k] = MultiFab(L_[k].U.box_array(), L_[k].U.dmap(), 3, 1);
        L_[k].aux = &aux_[k];
      }
    const int nx = dom_.nx(), ny = dom_.ny();
    const Real dx = geom_.dx(), dy = geom_.dy();
    {
      Array4 f = mg_.rhs().fab(0).array();
      const ConstArray4 u0 = L_[0].U.fab(0).const_array();
      device_fence();
      for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
          f(i, j) = model_.alpha * (u0(i, j, 0) - model_.n_i0);
    }
    mg_.solve();
    const ConstArray4 p = mg_.phi().fab(0).const_array();
    Array4 a0 = aux_[0].fab(0).array();
    device_fence();
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        a0(i, j, 0) = p(i, j);
        a0(i, j, 1) = (p(i + 1, j) - p(i - 1, j)) / (2 * dx);
        a0(i, j, 2) = (p(i, j + 1) - p(i, j - 1)) / (2 * dy);
      }
    fill_boundary(aux_[0], dom_, Periodicity{true, true});
    for (int k = 1; k < nlev_; ++k) inject_aux(aux_[k - 1], aux_[k]);
  }

  void update() {  // resout les champs sans avancer en temps
    sync_down();
    compute_aux();
  }

  void step(Real dt) {
    update();
    amr_step_multilevel_mf<NoSlope, RusanovFlux>(model_, L_, dom_, dt);
  }

  Real max_drift_speed() const {
    device_fence();
    const ConstArray4 a = aux_[0].fab(0).const_array();
    const int nx = dom_.nx(), ny = dom_.ny();
    Real v = 0;
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i)
        v = std::max(v, std::hypot(a(i, j, 1), a(i, j, 2)) / model_.B0);
    return std::max(v, Real(1e-12));
  }

  Real mass() const {
    device_fence();
    const ConstArray4 u = L_[0].U.fab(0).const_array();
    const int nx = dom_.nx(), ny = dom_.ny();
    const Real dV = geom_.dx() * geom_.dy();
    Real M = 0;
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) M += u(i, j, 0) * dV;
    return M;
  }

 private:
  // injection piecewise-constante de aux (3 comp) parent -> enfant (valides + ghosts).
  static void inject_aux(const MultiFab& parent, MultiFab& child) {
    device_fence();
    const ConstArray4 pp = parent.fab(0).const_array();
    Array4 c = child.fab(0).array();
    const Box2D g = child.fab(0).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i)
        for (int k = 0; k < 3; ++k)
          c(i, j, k) = pp(coarsen_index(i, 2), coarsen_index(j, 2), k);
  }

  Model model_;
  Geometry geom_;
  Box2D dom_;
  Elliptic mg_;
  std::vector<AmrLevelMF> L_;
  std::vector<MultiFab> aux_;
  int nlev_ = 0;
};

}  // namespace adc
