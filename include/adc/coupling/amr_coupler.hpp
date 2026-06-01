#pragma once

#include <adc/core/types.hpp>
#include <adc/coupling/amr_diagnostics.hpp>  // amr_mass, amr_max_drift_speed
#include <adc/coupling/amr_level_storage.hpp>  // AmrLevelStack
#include <adc/coupling/coupler.hpp>  // detail::coupler_eval_rhs (f = model.elliptic_rhs(U))
#include <adc/elliptic/elliptic_solver.hpp>  // concept EllipticSolver
#include <adc/elliptic/geometric_mg.hpp>
#include <adc/integrator/amr_reflux_mf.hpp>  // AmrLevelMP, advance_amr, mf_average_down_mb
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/mesh/refinement.hpp>  // coarsen_index

#include <utility>
#include <vector>

// Stepper couple AMR E x B, PORTE sur la pile MultiFab + le seam. La hierarchie est
// une std::vector<AmrLevelMP> (un MultiFab par niveau) detenue par un AmrLevelStack ;
// l'integration passe par advance_amr, le moteur AMR unifie multi-patch (sous-cyclage
// Berger-Oliger + reflux coverage-aware, via compute_face_fluxes -> MUSCL/HLL/HLLC/GPU
// dispo) et l'elliptique par un EllipticSolver (defaut MG). Le mono-box est le cas
// degenere (une box par niveau) de ce moteur, bit-identique a l'ancien chemin mono-box
// (garde 1 de test_amr_multilevel_multipatch).
//
//   sync_down (fin -> grossier) -> Poisson grossier -> aux = grad phi (grossier)
//   + injection vers les fins -> advance_amr (N niveaux conservatif).
//
// La classe coupleur n'ORDONNE plus que les operations : la hierarchie (stockage des
// niveaux + aux) est sortie dans AmrLevelStack, les diagnostics (masse, derive) dans
// amr_diagnostics.hpp. Le critere de raffinement / regrid reste a l'appelant
// (specifique au probleme) ; il manipule levels() pour reconstruire les box fines (la
// region raffinee est portee par le box_array du niveau fin, plus d'indices rC*), le
// stepper resynchronise aux. Generique sur le modele (Diocotron ; tout modele a alpha,
// n_i0, B0, flux E x B).

namespace adc {

namespace detail {
// Primitive de couplage (pas de la policy) : injection piecewise-constante de aux
// (3 comp) parent -> enfant (valides + ghosts), mono-box. Free function a portee de
// namespace (seam GPU, comme coupler_eval_rhs). Corps deplace TEL QUEL.
inline void coupler_inject_aux(const MultiFab& parent, MultiFab& child) {
  device_fence();
  const ConstArray4 pp = parent.fab(0).const_array();
  Array4 c = child.fab(0).array();
  const Box2D g = child.fab(0).grown_box();
  for (int j = g.lo[1]; j <= g.hi[1]; ++j)
    for (int i = g.lo[0]; i <= g.hi[0]; ++i)
      for (int k = 0; k < 3; ++k)
        c(i, j, k) = pp(coarsen_index(i, 2), coarsen_index(j, 2), k);
}
}  // namespace detail

template <class Model, class Elliptic = GeometricMG>
class AmrCoupler {
  static_assert(EllipticSolver<Elliptic>, "Elliptic doit modeler EllipticSolver");

 public:
  AmrCoupler(const Model& model, const Geometry& geom, const BoxArray& ba_coarse,
             const BCRec& bc, std::vector<AmrLevelMP> levels)
      : model_(model),
        geom_(geom),
        mg_(geom, ba_coarse, bc),
        stack_(geom.domain, std::move(levels)) {}

  std::vector<AmrLevelMP>& levels() { return stack_.levels(); }
  const std::vector<AmrLevelMP>& levels() const { return stack_.levels(); }
  MultiFab& coarse() { return stack_.coarse(); }
  const MultiFab& coarse() const { return stack_.coarse(); }
  const Box2D& domain() const { return stack_.domain(); }

  // moyenne fin -> grossier sur toute la hierarchie (avant le solve Poisson). Region
  // deduite du box_array du niveau fin (mf_average_down_mb) : bit-identique au mono-box.
  void sync_down() {
    auto& L = stack_.L();
    for (int k = stack_.nlev() - 1; k >= 1; --k) mf_average_down_mb(L[k].U, L[k - 1].U);
  }

  // Poisson grossier + aux = grad phi + injection vers les fins. Reconstruit aux_[k]
  // si un regrid a change une box (les pointeurs L_[k].aux restent valides).
  void compute_aux() {
    auto& L = stack_.L();
    for (int k = 0; k < stack_.nlev(); ++k)
      if (!(stack_.aux(k).box(0) == L[k].U.box(0))) stack_.reattach_aux(k);
    const Box2D& dom = stack_.domain();
    const int nx = dom.nx(), ny = dom.ny();
    const Real dx = geom_.dx(), dy = geom_.dy();
    // second membre via le modele (pas de formule recopiee) : f = elliptic_rhs(U)
    detail::coupler_eval_rhs(L[0].U, mg_.rhs(), model_);
    mg_.solve();
    const ConstArray4 p = mg_.phi().fab(0).const_array();
    Array4 a0 = stack_.aux(0).fab(0).array();
    device_fence();
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        a0(i, j, 0) = p(i, j);
        a0(i, j, 1) = (p(i + 1, j) - p(i - 1, j)) / (2 * dx);
        a0(i, j, 2) = (p(i, j + 1) - p(i, j - 1)) / (2 * dy);
      }
    fill_boundary(stack_.aux(0), dom, Periodicity{true, true});
    for (int k = 1; k < stack_.nlev(); ++k)
      detail::coupler_inject_aux(stack_.aux(k - 1), stack_.aux(k));
  }

  void update() {  // resout les champs sans avancer en temps
    sync_down();
    compute_aux();
  }

  // Discretisation spatiale (limiteur + flux) selectionnable, comme pour Coupler::step.
  // Defaut FirstOrder = NoSlope + Rusanov : strictement identique a l'ancien step().
  template <class Disc = FirstOrder>
  void step(Real dt) {
    update();
    advance_amr<typename Disc::Limiter, typename Disc::NumericalFlux>(
        model_, stack_.L(), stack_.domain(), dt);
  }

  Real max_drift_speed() const {
    return amr_max_drift_speed(stack_.aux(0), stack_.domain(), model_.B0);
  }

  Real mass() const {
    return amr_mass(stack_.coarse(), stack_.domain(), geom_.dx(), geom_.dy());
  }

 private:
  Model model_;
  Geometry geom_;
  Elliptic mg_;
  AmrLevelStack<AmrLevelMP> stack_;
};

}  // namespace adc
