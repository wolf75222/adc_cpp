#pragma once

#include <adc/core/types.hpp>
#include <adc/coupling/amr_diagnostics.hpp>     // amr_mass, amr_max_drift_speed
#include <adc/coupling/amr_level_storage.hpp>    // AmrLevelStack
#include <adc/coupling/amr_regrid_coupler.hpp>   // amr_regrid_finest (Berger-Rigoutsos)
#include <adc/coupling/coupler.hpp>  // detail::coupler_eval_rhs (f = model.elliptic_rhs(U))
#include <adc/elliptic/elliptic_solver.hpp>
#include <adc/elliptic/geometric_mg.hpp>
#include <adc/integrator/amr_reflux_mf.hpp>  // AmrLevelMP, amr_step_multilevel_multipatch, mf_*_mb
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/mesh/refinement.hpp>  // coarsen_index

#include <utility>
#include <vector>

// Coupleur AMR E x B MULTI-PATCH : meme role qu'AmrCoupler (Poisson grossier -> aux =
// grad phi -> injection -> pas AMR conservatif) mais la hierarchie est multi-box a chaque
// niveau (std::vector<AmrLevelMP>, detenue par un AmrLevelStack) et l'integration passe par
// amr_step_multilevel_multipatch (reflux coverage-aware route vers la box parente). De plus
// regrid() reconstruit le niveau fin a la volee par Berger-Rigoutsos. Niveau 0 = box unique
// (le domaine, pour le Poisson).
//
// La classe n'ORDONNE plus que les operations : la hierarchie est sortie dans AmrLevelStack,
// le regrid Berger-Rigoutsos dans amr_regrid_coupler.hpp, les diagnostics dans
// amr_diagnostics.hpp. Se reduit BIT A BIT a AmrCoupler quand chaque niveau n'a qu'une box
// (garde de validation).

namespace adc {

namespace detail {
// Primitive de couplage (pas de la policy) : injection piecewise-constante de aux
// parent multi-box -> enfant multi-box (valides + ghosts). DISTRIBUE, meme schema que
// mf_fill_fine_ghosts_mb. Deux cas de parent :
//  - REPLIQUE (niveau 0, replicated_parent=true) : parent entierement local sur chaque rang,
//    lecture directe via mf_find_box. C'est le grossier replique (dmap par-rang) : parallel_copy
//    violerait l'hypothese de metadonnees repliquees. Chemin bit-identique a l'historique.
//  - REPARTI (intermediaire, replicated_parent=false) : le parent peut etre sur un autre rang,
//    on amene ses regions valides sur une grille enfant-coarsen LOCALE par parallel_copy, puis
//    on injecte. Une cellule enfant hors couverture parente (box_array GLOBAL) est laissee
//    intacte, comme le chemin replique (qui la sautait via mf_find_box < 0).
// En serie les deux chemins sont identiques (parent local partout, parallel_copy = copie memoire).
inline void coupler_inject_aux_mb(const MultiFab& parent, MultiFab& child,
                                  bool replicated_parent = true) {
  const int nc = child.ncomp();
  if (replicated_parent) {
    device_fence();
    for (int lc = 0; lc < child.local_size(); ++lc) {
      Array4 c = child.fab(lc).array();
      const Box2D g = child.fab(lc).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
          const int ci = coarsen_index(i, 2), cj = coarsen_index(j, 2);
          const int pb = mf_find_box(parent, ci, cj);
          if (pb < 0) continue;
          const ConstArray4 pp = parent.fab(pb).const_array();
          for (int k = 0; k < nc; ++k) c(i, j, k) = pp(ci, cj, k);
        }
    }
    return;
  }
  const BoxArray& pba = parent.box_array();  // GLOBAL : couverture independante du rang
  auto covered = [&](int ci, int cj) {
    for (int b = 0; b < pba.size(); ++b)
      if (pba[b].contains(ci, cj)) return true;
    return false;
  };
  const BoxArray ccoarse = coarsen_grown(child.box_array(), child.n_grow(), 2);
  MultiFab Pc(ccoarse, child.dmap(), parent.ncomp(), 0);
  parallel_copy(Pc, parent);  // regions parentes (depuis n'importe quel rang) -> grille locale
  device_fence();
  for (int lc = 0; lc < child.local_size(); ++lc) {
    Array4 c = child.fab(lc).array();
    const ConstArray4 pp = Pc.fab(lc).const_array();
    const Box2D g = child.fab(lc).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
        const int ci = coarsen_index(i, 2), cj = coarsen_index(j, 2);
        if (!covered(ci, cj)) continue;  // hors couverture -> on laisse la valeur enfant
        for (int k = 0; k < nc; ++k) c(i, j, k) = pp(ci, cj, k);
      }
  }
}
}  // namespace detail

template <class Model, class Elliptic = GeometricMG>
class AmrCouplerMP {
  static_assert(EllipticSolver<Elliptic>, "Elliptic doit modeler EllipticSolver");

 public:
  AmrCouplerMP(const Model& model, const Geometry& geom, const BoxArray& ba_coarse,
               const BCRec& bc, std::vector<AmrLevelMP> levels)
      : model_(model), geom_(geom), mg_(geom, ba_coarse, bc),
        stack_(geom.domain, std::move(levels)) {}

  std::vector<AmrLevelMP>& levels() { return stack_.levels(); }
  MultiFab& coarse() { return stack_.coarse(); }
  const MultiFab& coarse() const { return stack_.coarse(); }
  const Box2D& domain() const { return stack_.domain(); }
  int nlev() const { return stack_.nlev(); }

  void sync_down() {  // moyenne fin -> grossier sur toute la hierarchie (multi-box)
    auto& L = stack_.L();
    for (int k = stack_.nlev() - 1; k >= 1; --k) mf_average_down_mb(L[k].U, L[k - 1].U);
  }

  void compute_aux() {  // Poisson grossier + grad phi + injection vers les fins
    auto& L = stack_.L();
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
    // parent aux(k-1) replique seulement au niveau 0 (grossier mono-box) ; au-dela, reparti.
    for (int k = 1; k < stack_.nlev(); ++k)
      detail::coupler_inject_aux_mb(stack_.aux(k - 1), stack_.aux(k), /*replicated_parent=*/k == 1);
  }

  void update() { sync_down(); compute_aux(); }

  void step(Real dt) {
    update();
    amr_step_multilevel_multipatch<NoSlope, RusanovFlux>(model_, stack_.L(), stack_.domain(), dt);
  }

  // Regrid du niveau FIN par Berger-Rigoutsos (delegue a amr_regrid_finest) :
  // reconstruit les patchs (report des donnees fines, sinon interp parent) + l'aux.
  // margin = nesting. Le coupleur ne fait qu'ordonner l'appel.
  template <class Crit>
  void regrid(Crit crit, int grow = 2, int margin = 2) {
    amr_regrid_finest(stack_.L(), stack_.aux(), stack_.domain(), crit, grow, margin);
  }

  Real mass() const {
    return amr_mass(stack_.coarse(), stack_.domain(), geom_.dx(), geom_.dy());
  }

  Real max_drift_speed() const {
    return amr_max_drift_speed(stack_.aux(0), stack_.domain(), model_.B0);
  }

 private:
  Model model_;
  Geometry geom_;
  Elliptic mg_;
  AmrLevelStack<AmrLevelMP> stack_;
};

}  // namespace adc
