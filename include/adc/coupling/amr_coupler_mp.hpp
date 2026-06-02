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
#include <adc/parallel/comm.hpp>    // all_reduce_sum / all_reduce_max (masse/derive distribuees)

#include <algorithm>   // std::max
#include <cmath>       // std::hypot
#include <functional>  // std::function (predicat de paroi conductrice passe au MG)
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
  // active : predicat optionnel "cellule active" (interieur du conducteur), pour la paroi
  // conductrice circulaire de l'instabilite colonne (passe tel quel au multigrille). Vide
  // par defaut -> pas de paroi (comportement historique inchange). Seul le grossier porte la
  // paroi : les patchs fins raffinent le bord d'anneau, strictement a l'interieur du mur.
  // replicated_coarse : POLITIQUE D'OWNERSHIP du niveau 0 (grossier). Les DEUX modes sont
  // stables et leur equivalence est prouvee bit a bit (test_mpi_decoarse, maxdiff=0) :
  //   true  (DEFAUT performant) : grossier mono-box REPLIQUE sur tous les rangs. Meilleur solve
  //          MG grossier (pas de degenerescence du multigrille), zero communication pour le
  //          Poisson grossier, reference robuste -> le bon defaut pour les cas petits/moyens.
  //   false (mode scalable EXPLICITE) : grossier multi-box REPARTI round-robin. Leve le verrou
  //          memoire O(NX*NY*nrangs) du niveau 0, necessaire a tres grande echelle. Mais le MG
  //          geometrique degenere pour un grossier finement decoupe (>2x2 boxes ne pavent pas la
  //          grille la plus grossiere) : a reserver aux cas ou la memoire niveau-0 est le verrou.
  // Critere : mettre false UNIQUEMENT quand la scalabilite memoire l'exige ; sinon garder true.
  // La suppression du chemin replique est REPORTEE tant que le reparti n'est pas strictement
  // superieur. Le mg_ recoit le meme drapeau (sinon, sous MPI replique, le grossier tomberait sur
  // le seul rang 0 et compute_aux lirait un phi absent ailleurs). En serie, les deux coincident.
  AmrCouplerMP(const Model& model, const Geometry& geom, const BoxArray& ba_coarse,
               const BCRec& bc, std::vector<AmrLevelMP> levels,
               std::function<bool(Real, Real)> active = {},
               bool replicated_coarse = true)
      : model_(model), geom_(geom),
        mg_(geom, ba_coarse, bc, std::move(active), replicated_coarse),
        stack_(geom.domain, std::move(levels)), replicated_coarse_(replicated_coarse) {}

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
    const Real dx = geom_.dx(), dy = geom_.dy();
    // second membre via le modele (pas de formule recopiee) : f = elliptic_rhs(U)
    detail::coupler_eval_rhs(L[0].U, mg_.rhs(), model_);
    mg_.solve();  // laisse phi avec ses ghosts remplis (dernier gs_rb_sweep -> fill_ghosts)
    device_fence();
    // aux = (phi, grad phi) par fab grossier LOCAL : couvre le mono-box replique (1 fab) comme
    // le multi-box reparti (de-replication). Les derivees au bord de box lisent les ghosts de
    // phi remplis par le solve (echange inter-box distribue via fill_boundary). mg_.phi() et
    // aux(0) partagent le meme layout (meme BoxArray + DistributionMapping) -> fab(li) <-> box(li).
    for (int li = 0; li < mg_.phi().local_size(); ++li) {
      const ConstArray4 p = mg_.phi().fab(li).const_array();
      Array4 a = stack_.aux(0).fab(li).array();
      const Box2D b = stack_.aux(0).box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
          a(i, j, 0) = p(i, j);
          a(i, j, 1) = (p(i + 1, j) - p(i - 1, j)) / (2 * dx);
          a(i, j, 2) = (p(i, j + 1) - p(i, j - 1)) / (2 * dy);
        }
    }
    fill_boundary(stack_.aux(0), dom, Periodicity{true, true});
    // parent aux(k-1) replique seulement si le niveau 0 l'est : sinon il est REPARTI (multi-box)
    // et l'injection passe par parallel_copy. Au-dela du niveau 1, le parent est toujours reparti.
    for (int k = 1; k < stack_.nlev(); ++k)
      detail::coupler_inject_aux_mb(stack_.aux(k - 1), stack_.aux(k),
                                    /*replicated_parent=*/(k == 1) && replicated_coarse_);
  }

  void update() { sync_down(); compute_aux(); }

  // Discretisation spatiale selectionnable (defaut FirstOrder = NoSlope + Rusanov,
  // strictement identique a l'ancien step()).
  template <class Disc = FirstOrder>
  void step(Real dt) {
    update();
    advance_amr<typename Disc::Limiter, typename Disc::NumericalFlux>(
        model_, stack_.L(), stack_.domain(), dt, Periodicity{true, true}, replicated_coarse_);
  }

  // Regrid du niveau FIN par Berger-Rigoutsos (delegue a amr_regrid_finest) :
  // reconstruit les patchs (report des donnees fines, sinon interp parent) + l'aux.
  // margin = nesting. Le coupleur ne fait qu'ordonner l'appel.
  template <class Crit>
  void regrid(Crit crit, int grow = 2, int margin = 2) {
    amr_regrid_finest(stack_.L(), stack_.aux(), stack_.domain(), crit, grow, margin);
  }

  // masse grossiere via le diagnostic partage amr_mass_mb (mono-box replique comme
  // multi-box reparti). Grossier replique : la somme locale EST deja la masse totale
  // (chaque rang detient tout) -> pas d'all_reduce. Reparti : part locale -> all_reduce_sum.
  Real mass() const {
    const Real M = amr_mass_mb(stack_.coarse(), geom_.dx(), geom_.dy());
    return replicated_coarse_ ? M : all_reduce_sum(M);
  }

  // vitesse de derive max via amr_max_drift_speed_mb + plancher. all_reduce_max correct
  // dans les DEUX cas : sous replication le max local est deja global (idempotent) ;
  // reparti, on prend le max des parts.
  Real max_drift_speed() const {
    const Real v = amr_max_drift_speed_mb(stack_.aux(0), model_.B0);
    return all_reduce_max(std::max(v, Real(1e-12)));
  }

  /// @brief Max wave speed over the coarse level via `model.max_wave_speed`.
  ///
  /// Model-generic CFL speed (any `PhysicalModel`), unlike `max_drift_speed` which is
  /// specific to the E x B drift (`model.B0`). For a pure E x B transport it equals the drift speed.
  ///
  /// @returns the max over coarse cells and both directions, reduced across ranks.
  /// @note `update()` must have run so that `aux(0)` holds the current `grad phi`.
  Real max_wave_speed() {
    Real w = Real(1e-12);
    MultiFab& U = stack_.coarse();
    MultiFab& A = stack_.aux(0);
    for (int li = 0; li < U.local_size(); ++li) {
      const ConstArray4 u = U.fab(li).const_array();
      const ConstArray4 a = A.fab(li).const_array();
      const Box2D b = U.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
          const auto us = load_state<Model>(u, i, j);
          const Aux ax = load_aux(a, i, j);
          w = std::max(w, std::max(model_.max_wave_speed(us, ax, 0),
                                   model_.max_wave_speed(us, ax, 1)));
        }
    }
    return all_reduce_max(w);
  }

 private:
  Model model_;
  Geometry geom_;
  Elliptic mg_;
  AmrLevelStack<AmrLevelMP> stack_;
  bool replicated_coarse_;  // niveau 0 replique (true) ou multi-box reparti (false, de-replication)
};

}  // namespace adc
