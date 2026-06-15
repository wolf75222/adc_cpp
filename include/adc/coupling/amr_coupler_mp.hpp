#pragma once

#include <adc/core/types.hpp>
#include <adc/coupling/amr_diagnostics.hpp>     // amr_mass, amr_max_drift_speed
#include <adc/coupling/amr_level_storage.hpp>    // AmrLevelStack
#include <adc/coupling/amr_regrid_coupler.hpp>   // amr_regrid_finest (Berger-Rigoutsos)
#include <adc/coupling/coupler.hpp>  // detail::coupler_eval_rhs (f = model.elliptic_rhs(U))
#include <adc/numerics/elliptic/composite_fac_poisson.hpp>  // solveur Poisson COMPOSITE FAC 2 niveaux (opt-in)
#include <adc/numerics/elliptic/elliptic_solver.hpp>
#include <adc/numerics/elliptic/geometric_mg.hpp>
#include <adc/numerics/time/amr_reflux_mf.hpp>  // AmrLevelMP, amr_step_multilevel_multipatch, mf_*_mb
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/mesh/refinement.hpp>  // coarsen_index
#include <adc/parallel/comm.hpp>    // all_reduce_sum / all_reduce_max (masse/derive distribuees)

#include <algorithm>   // std::max
#include <cmath>       // std::hypot
#include <cstddef>     // std::size_t
#include <functional>  // std::function (predicat de paroi conductrice passe au MG)
#include <stdexcept>   // std::runtime_error (garde de taille densite)
#include <utility>     // std::pair, std::move
#include <vector>

/// @file
/// @brief AmrCouplerMP : coupleur AMR E x B MULTI-PATCH (Poisson grossier -> aux = grad phi ->
///        injection fine -> pas AMR conservatif), hierarchie multi-box par niveau.
///
/// Meme role qu'AmrCoupler mais chaque niveau est multi-box (std::vector<AmrLevelMP> detenu par un
/// AmrLevelStack) et l'integration passe par amr_step_multilevel_multipatch (reflux coverage-aware).
/// regrid() reconstruit le niveau fin a la volee par Berger-Rigoutsos. Niveau 0 = box unique pour le
/// Poisson. La classe ne fait qu'ORDONNER les operations (hierarchie sortie dans AmrLevelStack,
/// regrid dans amr_regrid_coupler.hpp, diagnostics dans amr_diagnostics.hpp). INVARIANT : se reduit
/// BIT A BIT a AmrCoupler quand chaque niveau n'a qu'une box (garde de validation). Politique
/// d'ownership du niveau 0 via replicated_coarse (replique vs reparti, equivalence prouvee bit a bit).
/// Les detail:: sont les primitives DISTRIBUEES (injection aux, ecriture/lecture densite, layout).

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
// Ecrit une densite initiale (composante 0, n*n row-major en indices GLOBAUX) sur le niveau
// grossier, MULTI-BOX et DISTRIBUTION-AWARE : chaque rang ne touche que ses fabs LOCAUX et lit
// rho a l'indice GLOBAL (i,j) de la cellule. Pour Euler (ncomp 4) pose aussi qty de mouvement
// nulle + energie thermique r/(gamma-1) ; ncomp 1 ne touche que la densite. Mono-box replique :
// un seul fab couvrant le domaine, indices globaux == locaux -> bit-identique a l'ecriture directe
// historique. Multi-box reparti : chaque box locale lit sa fenetre de rho.
inline void coupler_write_coarse(MultiFab& U, const std::vector<double>& rho, int n, int ncomp,
                                 double gamma) {
  if (static_cast<int>(rho.size()) != n * n)
    throw std::runtime_error("AMR coupler : densite initiale de taille != n*n");
  const Real gm1 = Real(gamma) - Real(1);
  device_fence();
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 u = U.fab(li).array();
    const Box2D v = U.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        const Real r = rho[static_cast<std::size_t>(j) * n + i];
        u(i, j, 0) = r;
        if (ncomp >= 3) { u(i, j, 1) = 0; u(i, j, 2) = 0; }
        if (ncomp == 4) u(i, j, 3) = r / gm1;
      }
  }
}

// Ecrit l'ETAT CONSERVATIF INITIAL COMPLET (toutes les composantes) sur le niveau grossier depuis un
// champ plat @p state composante-majeur (c*n*n + j*n + i), de taille ncomp*n*n. Pendant exact de
// coupler_write_coarse pour le seed multi-composantes : meme parcours de boites (mono-box replique
// ET multi-box reparti, indices GLOBAUX (i,j)), seul l'ecriture par cellule differe -- ici on copie
// les ncomp composantes positionnellement (pas de densite/qty-mvt/energie cables ; l'appelant fournit
// deja le conservatif, p.ex. [rho, rho*u, rho*v]). gamma omis (pas d'energie deduite). Index calcule
// en std::size_t (pas d'overflow int a grand n, contrairement a la validation int de
// coupler_write_coarse). Sert au seed de derive (set_conservative_state, Probleme 2 hoffart).
inline void coupler_write_coarse_state(MultiFab& U, const std::vector<double>& state, int n,
                                       int ncomp) {
  const std::size_t nn = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  if (state.size() != nn * static_cast<std::size_t>(ncomp))
    throw std::runtime_error("AMR coupler : etat initial de taille != ncomp*n*n (etat conservatif "
                             "complet ; ncomp == n_vars du modele)");
  device_fence();
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 u = U.fab(li).array();
    const Box2D v = U.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        for (int c = 0; c < ncomp; ++c)
          u(i, j, c) = state[static_cast<std::size_t>(c) * nn +
                             static_cast<std::size_t>(j) * static_cast<std::size_t>(n) +
                             static_cast<std::size_t>(i)];
  }
}

// Lit la densite grossiere (composante 0) en un champ n*n row-major GLOBAL, MULTI-BOX et
// DISTRIBUTION-AWARE. Chaque rang ecrit ses cellules locales dans un buffer n*n initialise a 0
// puis, si reparti, all_reduce_sum_inplace recompose le champ complet sur TOUS les rangs (les
// boites sont disjointes -> la somme cross-rang reconstruit exactement le champ). Mono-box
// replique : un seul fab couvre tout, le buffer est deja complet, all_reduce serait l'identite
// -> on l'evite (bit-identique a la lecture directe historique fab(0)).
inline std::vector<double> coupler_read_coarse(const MultiFab& U, int n, bool replicated) {
  device_fence();
  std::vector<double> out(static_cast<std::size_t>(n) * n, 0.0);
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const Box2D v = U.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        out[static_cast<std::size_t>(j) * n + i] = u(i, j, 0);
  }
  if (!replicated) all_reduce_sum_inplace(out.data(), static_cast<int>(out.size()));
  return out;
}

// Lit le potentiel phi du niveau grossier (composante 0 de aux(0), ecrite par compute_aux apres le
// solve Poisson) en un champ n*n row-major GLOBAL, MULTI-BOX et DISTRIBUTION-AWARE. aux(0) partage
// EXACTEMENT le layout du grossier U (meme BoxArray + DistributionMapping, cf. amr_level_storage :
// aux_[0] est construit sur U.box_array()/U.dmap()), donc la recomposition est identique a
// coupler_read_coarse : buffer local n*n, all_reduce_sum si reparti (boites disjointes -> somme
// exacte), evitee en mono-box replique (champ deja complet). PRECONDITION : update()/compute_aux a
// tourne au moins une fois (sinon aux(0) vaut 0). Pendant strict de coupler_read_coarse pour phi.
inline std::vector<double> coupler_read_coarse_phi(const MultiFab& aux0, int n, bool replicated) {
  device_fence();
  std::vector<double> out(static_cast<std::size_t>(n) * n, 0.0);
  for (int li = 0; li < aux0.local_size(); ++li) {
    const ConstArray4 a = aux0.fab(li).const_array();
    const Box2D v = aux0.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        out[static_cast<std::size_t>(j) * n + i] = a(i, j, 0);
  }
  if (!replicated) all_reduce_sum_inplace(out.data(), static_cast<int>(out.size()));
  return out;
}

// Injecte le grossier dans les cellules valides d'un patch fin (constant par morceaux, ratio 2),
// MULTI-BOX et DISTRIBUTION-AWARE. Rend la hierarchie coherente avant le premier sync_down (le
// patch seed est a 0). Mono-box replique : grossier entierement local, lecture directe via
// mf_find_box (toujours trouve) ; aucune collective -> bit-identique a l'historique fab(0).
// Multi-box reparti : on amene les regions grossieres necessaires sur une grille enfant-coarsen
// LOCALE par parallel_copy (meme schema que coupler_inject_aux_mb), puis on injecte.
inline void coupler_inject_coarse_to_fine_mb(const MultiFab& Uc, MultiFab& Uf, bool replicated) {
  const int nc = Uf.ncomp();
  if (replicated) {
    device_fence();
    for (int li = 0; li < Uf.local_size(); ++li) {
      Array4 f = Uf.fab(li).array();
      const Box2D v = Uf.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
          const int ci = coarsen_index(i, 2), cj = coarsen_index(j, 2);
          const int pb = mf_find_box(Uc, ci, cj);
          if (pb < 0) continue;
          const ConstArray4 c = Uc.fab(pb).const_array();
          for (int k = 0; k < nc; ++k) f(i, j, k) = c(ci, cj, k);
        }
    }
    return;
  }
  const BoxArray ccoarse = coarsen(Uf.box_array(), 2);  // empreinte grossiere (cellules valides)
  MultiFab Pc(ccoarse, Uf.dmap(), Uc.ncomp(), 0);
  parallel_copy(Pc, Uc);  // regions grossieres (depuis n'importe quel rang) -> grille locale
  device_fence();
  for (int li = 0; li < Uf.local_size(); ++li) {
    Array4 f = Uf.fab(li).array();
    const ConstArray4 c = Pc.fab(li).const_array();
    const Box2D v = Uf.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        const int ci = coarsen_index(i, 2), cj = coarsen_index(j, 2);
        for (int k = 0; k < nc; ++k) f(i, j, k) = c(ci, cj, k);
      }
  }
}

// Construit le niveau grossier (BoxArray + DistributionMapping) du chemin AmrSystem selon la
// politique d'ownership, en un SEUL point pour les deux chemins de build (natif + compile) :
//  - replique (distribute=false, DEFAUT) : mono-box couvrant le domaine, dmap = my_rank() partout
//    (la box vit sur chaque rang). En serie my_rank()=0 -> identique au round-robin, bit a bit.
//    C'est le layout qu'attend GeometricMG(replicated=true) et l'historique.
//  - reparti (distribute=true) : multi-box BoxArray::from_domain(dom, max_grid) reparti round-robin
//    DistributionMapping(ba.size(), n_ranks()). Chaque rang ne porte que ses tuiles -> le Poisson
//    grossier et le transport grossier se distribuent (strong-scaling). max_grid<=0 => n/2 (2x2).
inline std::pair<BoxArray, DistributionMapping> coupler_make_coarse_layout(int n, bool distribute,
                                                                           int max_grid) {
  const Box2D dom = Box2D::from_extents(n, n);
  if (!distribute) {
    BoxArray ba(std::vector<Box2D>{dom});
    return {ba, DistributionMapping(std::vector<int>{my_rank()})};
  }
  const int mg = (max_grid > 0) ? max_grid : (n / 2 > 0 ? n / 2 : n);
  BoxArray ba = BoxArray::from_domain(dom, mg);
  return {ba, DistributionMapping(ba.size(), n_ranks())};
}

}  // namespace detail

/// Coupleur AMR E x B multi-patch. @tparam Model : PhysicalModel (flux, source, elliptic_rhs,
/// max_wave_speed). @tparam Elliptic : backend elliptique (concept EllipticSolver, defaut GeometricMG).
/// ORCHESTRE seulement : la hierarchie vit dans un AmrLevelStack<AmrLevelMP>, le solve Poisson dans
/// mg_, le regrid dans amr_regrid_finest. Se reduit bit a bit a AmrCoupler en mono-box par niveau.
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
        stack_(geom.domain, std::move(levels), aux_comps<Model>()),
        replicated_coarse_(replicated_coarse) {}

  std::vector<AmrLevelMP>& levels() { return stack_.levels(); }
  MultiFab& coarse() { return stack_.coarse(); }
  const MultiFab& coarse() const { return stack_.coarse(); }
  // aux du niveau grossier : (phi, dphi/dx, dphi/dy), composante 0 = phi (cf. compute_aux). Meme
  // layout que coarse(). Lu par le hook potential d'AmrSystem (coupler_read_coarse_phi).
  MultiFab& aux0() { return stack_.aux(0); }
  const MultiFab& aux0() const { return stack_.aux(0); }
  const Box2D& domain() const { return stack_.domain(); }
  int nlev() const { return stack_.nlev(); }

  // ----------------------------------------------------------------------------------------------
  // CHECKPOINT / RESTART AMR mono-rang (ADC-65). Le coupleur mono-bloc porte l'ETAT CONSERVATIF
  // COMPLET par niveau (toutes composantes) + le phi (warm-start du multigrille), et sait IMPOSER
  // une hierarchie fine SAUVEE (au lieu du clustering Berger-Rigoutsos sur tags). MONO-RANG : les
  // accesseurs bouclent sur local_size() + device_fence(), SANS gather MPI (la facade rejette np>1
  // ET le multi-blocs en amont ; la reprise multi-rangs/multi-blocs est une suite documentee).
  // ----------------------------------------------------------------------------------------------

  // Lit l'etat conservatif COMPLET (toutes composantes) du niveau @p k en un champ plat
  // composante-majeur c*nf*nf + j*nf + i, nf = n << k (n = cote du grossier). Les cellules HORS
  // patchs (niveau fin non couvert) restent a 0 : un niveau fin n'est defini que dans ses patchs
  // (au restart on ne reecrit QUE les cellules des patchs, cf. set_level_state).
  std::vector<double> level_state(int k) {
    std::vector<AmrLevelMP>& L = stack_.L();
    if (k < 0 || k >= static_cast<int>(L.size()))
      throw std::runtime_error("AmrCouplerMP::level_state : niveau hors bornes");
    MultiFab& U = L[k].U;
    const int nc = U.ncomp();
    const std::size_t nf = static_cast<std::size_t>(stack_.domain().nx()) << k;
    std::vector<double> out(static_cast<std::size_t>(nc) * nf * nf, 0.0);
    device_fence();
    for (int li = 0; li < U.local_size(); ++li) {
      const ConstArray4 u = U.fab(li).const_array();
      const Box2D v = U.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          for (int c = 0; c < nc; ++c)
            out[static_cast<std::size_t>(c) * nf * nf + static_cast<std::size_t>(j) * nf +
                static_cast<std::size_t>(i)] = u(i, j, c);
    }
    return out;
  }

  // Restaure l'etat conservatif complet du niveau @p k depuis @p s (meme layout que level_state).
  // Ecrit UNIQUEMENT les cellules VALIDES des fabs locaux (les patchs) : les ghosts sont refaits au
  // prochain update()/advance (exactement comme apres un regrid), et une cellule fine hors patch
  // n'existe pas. NO RE-PROLONGATION : l'etat est restaure TEL QUEL (pas d'injection coarse->fine).
  void set_level_state(int k, const std::vector<double>& s) {
    std::vector<AmrLevelMP>& L = stack_.L();
    if (k < 0 || k >= static_cast<int>(L.size()))
      throw std::runtime_error("AmrCouplerMP::set_level_state : niveau hors bornes");
    MultiFab& U = L[k].U;
    const int nc = U.ncomp();
    const std::size_t nf = static_cast<std::size_t>(stack_.domain().nx()) << k;
    if (s.size() != static_cast<std::size_t>(nc) * nf * nf)
      throw std::runtime_error("AmrCouplerMP::set_level_state : taille de l'etat != ncomp*nf*nf");
    device_fence();
    for (int li = 0; li < U.local_size(); ++li) {
      Array4 u = U.fab(li).array();
      const Box2D v = U.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          for (int c = 0; c < nc; ++c)
            u(i, j, c) = s[static_cast<std::size_t>(c) * nf * nf + static_cast<std::size_t>(j) * nf +
                           static_cast<std::size_t>(i)];
    }
  }

  // Lit le potentiel phi du niveau @p k, champ plat nf*nf row-major (nf = n << k), zeros hors patchs.
  // Niveau 0 : le WARM-START du multigrille -- mg_.phi() (cellules VALIDES), l'etat reellement
  // reutilise par le PROCHAIN solve (GeometricMG::solve conserve phi entre appels). Niveau >= 1 :
  // aux(k) composante 0 (informatif ; recompute a l'update). C'est mg_.phi() niveau 0 qui rend la
  // reprise BIT-IDENTIQUE (le 1er solve post-restart repart du meme guess que le run continu).
  std::vector<double> level_potential(int k) {
    if (k < 0 || k >= stack_.nlev())
      throw std::runtime_error("AmrCouplerMP::level_potential : niveau hors bornes");
    const std::size_t nf = static_cast<std::size_t>(stack_.domain().nx()) << k;
    std::vector<double> out(nf * nf, 0.0);
    device_fence();
    const MultiFab& P = (k == 0) ? mg_.phi() : stack_.aux(k);
    for (int li = 0; li < P.local_size(); ++li) {
      const ConstArray4 p = P.fab(li).const_array();
      const Box2D v = P.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          out[static_cast<std::size_t>(j) * nf + static_cast<std::size_t>(i)] = p(i, j, 0);
    }
    return out;
  }

  // Restaure le potentiel du niveau @p k. Niveau 0 : warm-start mg_.phi() (cellules valides) -> la
  // reprise du multigrille est BIT-IDENTIQUE (le 1er solve post-restart repart du meme guess). Niveau
  // >= 1 : aux(k) comp 0 (recompute a l'update ; restauration idempotente, sans effet sur la dynamique).
  void set_level_potential(int k, const std::vector<double>& p) {
    if (k < 0 || k >= stack_.nlev())
      throw std::runtime_error("AmrCouplerMP::set_level_potential : niveau hors bornes");
    const std::size_t nf = static_cast<std::size_t>(stack_.domain().nx()) << k;
    if (p.size() != nf * nf)
      throw std::runtime_error("AmrCouplerMP::set_level_potential : taille phi != nf*nf");
    device_fence();
    MultiFab& P = (k == 0) ? mg_.phi() : stack_.aux(k);
    for (int li = 0; li < P.local_size(); ++li) {
      Array4 q = P.fab(li).array();
      const Box2D v = P.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          q(i, j, 0) = p[static_cast<std::size_t>(j) * nf + static_cast<std::size_t>(i)];
    }
  }

  // Impose la hierarchie du niveau fin (restart) : reconstruit le niveau 1 sur le BoxArray @p
  // fine_boxes SAUVE (au lieu du clustering Berger-Rigoutsos sur tags), via la MEME mecanique que le
  // regrid (regrid_field_on_layout : interp parent + report fin), puis reattache l'aux du niveau 1.
  // Le contenu valide reconstruit est ECRASE ensuite par set_level_state (restauration tel quel) : on
  // ne s'appuie ici que sur le LAYOUT impose. MONO-RANG, hierarchie mono-bloc a 2 niveaux (on n'impose
  // donc QUE le niveau 1). Rejet clair si la hierarchie n'a pas de niveau fin ou si aucune boite sauvee.
  void set_hierarchy(const std::vector<Box2D>& fine_boxes) {
    std::vector<AmrLevelMP>& L = stack_.L();
    if (L.size() < 2)
      throw std::runtime_error("AmrCouplerMP::set_hierarchy : hierarchie mono-niveau (aucun patch "
                               "fin a imposer)");
    if (fine_boxes.empty())
      throw std::runtime_error("AmrCouplerMP::set_hierarchy : aucune boite fine sauvee (restart d'une "
                               "hierarchie a patchs fins requis)");
    const int ngf = L[1].U.n_grow();  // herite la largeur de ghost du fin courant (parite schema)
    BoxArray fb(fine_boxes);
    DistributionMapping dmap(static_cast<int>(fb.size()), n_ranks());  // mono-rang -> tout sur rang 0
    L[1].U = regrid_field_on_layout(fb, dmap, L[0].U, L[1].U, /*pk=*/0, ngf, replicated_coarse_);
    stack_.reattach_aux(1);  // realloc aux[1] sur le nouveau layout + recable L[1].aux
  }

  void sync_down() {  // moyenne fin -> grossier sur toute la hierarchie (multi-box)
    auto& L = stack_.L();
    for (int k = stack_.nlev() - 1; k >= 1; --k) mf_average_down_mb(L[k].U, L[k - 1].U);
  }

  /// OPT-IN : remplace le Poisson AMR Option A (solve grossier + injection grad constant par morceaux)
  /// par un solve elliptique COMPOSITE FAC (le patch fin RAFFINE l'elliptique). Cf. CompositeFacPoisson.
  /// Perimetre Phase 2 : 2 niveaux, UN patch fin mono-box interieur, grossier replique (mono-rang).
  /// Hors de ce cadre, compute_aux retombe sur Option A (bit-identique).
  void set_composite_poisson(bool v) { composite_poisson_ = v; }
  bool composite_poisson() const { return composite_poisson_; }

  void compute_aux() {  // Poisson grossier + grad phi + injection vers les fins
    auto& L = stack_.L();
    const Box2D& dom = stack_.domain();
    const Real dx = geom_.dx(), dy = geom_.dy();
    // CHEMIN COMPOSITE (opt-in) : le patch fin raffine VRAIMENT l'elliptique. Cadre supporte = 2 niveaux,
    // UN patch fin mono-box, grossier replique (Phase 2). Sinon Option A ci-dessous (bit-identique).
    if (composite_poisson_ && replicated_coarse_ && stack_.nlev() == 2 &&
        L[1].U.box_array().size() == 1) {
      compute_aux_composite();
      return;
    }
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

  /// Met la hierarchie a jour avant un pas : sync_down (fin -> grossier) puis compute_aux (Poisson
  /// grossier + grad phi + injection vers les fins).
  void update() { sync_down(); compute_aux(); }

  // Discretisation spatiale selectionnable (defaut FirstOrder = NoSlope + Rusanov,
  // strictement identique a l'ancien step()). recon_prim selectionne la reconstruction
  // primitive (meme parametre qu'assemble_rhs / System) ; false (defaut) -> conservative.
  // imex : traite la source raide en IMPLICITE (backward_euler) plutot qu'en Euler avant ;
  // false (defaut) -> traitement explicite historique, bit-identique. La source restant
  // cellule-locale (hors registres de reflux), le split implicite preserve la conservation.
  /// Avance la hierarchie d'un pas dt : update() puis advance_amr (sous-cyclage Berger-Oliger +
  /// reflux + average_down conservatifs). @tparam Disc : discretisation spatiale (limiteur + flux,
  /// defaut FirstOrder bit-identique a l'historique). recon_prim : reconstruction primitive ; imex :
  /// source raide en implicite (backward_euler). Defauts (false) -> chemin explicite historique.
  /// @p nopts : OPTIONS du Newton de la source implicite IMEX (budget d'iterations, tolerances,
  /// fd_eps, damping, fail_policy), threadees jusqu'a backward_euler_source par advance_amr ->
  /// subcycle_level_mp -> mf_apply_source_treatment. DEFAUT {} = constantes historiques (2 iters,
  /// 1e-7, ...) -> chemin (2a) BIT-IDENTIQUE a l'ancien appel. Sans effet si imex==false. Le masque
  /// IMEX partiel n'est PAS porte par ce chemin mono-bloc (backward-Euler plein), seules les OPTIONS
  /// le sont (AmrSystem mono-bloc cable les options Newton mais pas le masque ni les diagnostics).
  /// @p tmethod : methode temporelle (kEuler par defaut = Euler avant historique bit-identique ;
  /// kSsprk3 = SSPRK3 ordre 3 + reflux par etage). kSsprk3 exige imex == false (rejet sinon).
  template <class Disc = FirstOrder>
  void step(Real dt, bool recon_prim = false, bool imex = false, const NewtonOptions& nopts = {},
            AmrTimeMethod tmethod = AmrTimeMethod::kEuler) {
    update();
    advance_amr<typename Disc::Limiter, typename Disc::NumericalFlux>(
        model_, stack_.L(), stack_.domain(), dt, Periodicity{true, true}, replicated_coarse_,
        recon_prim, imex, nopts, tmethod);
  }

  /// AVANCE DE TRANSPORT SEULE (hyperbolique), SANS update() ni source. Pendant de step() prive de
  /// son solve de champ et avec imex==false : c'est l'avance HYPERBOLIQUE PURE (-div F) du chemin
  /// amr-schur, ou le champ est resolu separement (update()) et la source est jouee par l'etage
  /// condense global (AmrCondensedSchurSourceStepper), exactement comme le chemin uniforme entrelace
  /// solve_fields / transport (s.advance) / etage source (run_source_stage). Le modele doit etre
  /// SOURCE-FREE (brique source NoSource) pour que la source ne soit pas comptee deux fois (une fois
  /// ici en Euler avant, une fois par l'etage Schur) : c'est le contrat du chemin amr-schur, miroir du
  /// time=Strang(Explicit, CondensedSchur) uniforme ou le bloc est ajoute avec son seul transport.
  template <class Disc = FirstOrder>
  void advance_transport(Real dt, bool recon_prim = false) {
    advance_amr<typename Disc::Limiter, typename Disc::NumericalFlux>(
        model_, stack_.L(), stack_.domain(), dt, Periodicity{true, true}, replicated_coarse_,
        recon_prim, /*imex=*/false);
  }

  // Regrid du niveau FIN par Berger-Rigoutsos (delegue a amr_regrid_finest) :
  // reconstruit les patchs (report des donnees fines, sinon interp parent) + l'aux.
  // margin = nesting. Le coupleur ne fait qu'ordonner l'appel.
  template <class Crit>
  void regrid(Crit crit, int grow = 2, int margin = 2) {
    amr_regrid_finest(stack_.L(), stack_.aux(), stack_.domain(), crit, grow, margin,
                      aux_comps<Model>(), replicated_coarse_);
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

  /// @brief Vitesse d'onde max sur le niveau grossier via `model.max_wave_speed`.
  ///
  /// Vitesse CFL generique au modele (tout `PhysicalModel`), contrairement a `max_drift_speed`
  /// qui est specifique a la derive E x B (`model.B0`). Pour un transport E x B pur, elle egale
  /// la vitesse de derive.
  ///
  /// @return le max sur les cellules grossieres et les deux directions, reduit sur les rangs.
  /// @note `update()` doit avoir tourne pour que `aux(0)` porte le `grad phi` courant.
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
          const Aux ax = load_aux<aux_comps<Model>()>(a, i, j);
          w = std::max(w, std::max(model_.max_wave_speed(us, ax, 0),
                                   model_.max_wave_speed(us, ax, 1)));
        }
    }
    return all_reduce_max(w);
  }

 private:
  /// Pas Poisson COMPOSITE FAC (chemin opt-in). Solve l'elliptique sur grossier + patch fin couple par
  /// FAC, puis pose aux PAR NIVEAU depuis le phi DE CHAQUE NIVEAU : aux fin = (phi_f, grad fin) ou grad
  /// fin = diff centree sur phi_f (resolu a la finesse), PAS l'injection grad grossier constant d'Option A.
  void compute_aux_composite() {
    auto& L = stack_.L();
    const Box2D& dom = stack_.domain();
    const Box2D fine_box = L[1].U.box_array()[0];
    if (!fac_built_ || !same_box(fac_fine_box_, fine_box)) {
      fac_ = std::make_shared<CompositeFacPoisson>(geom_, mg_.box_array(), mg_.bc(), fine_box, 2);
      fac_fine_box_ = fine_box;
      fac_built_ = true;
    }
    // f = elliptic_rhs(U) PAR NIVEAU : le fin a son PROPRE second membre raffine (pas une injection).
    detail::coupler_eval_rhs(L[0].U, fac_->rhs_coarse(), model_);
    detail::coupler_eval_rhs(L[1].U, fac_->rhs_fine(), model_);
    fac_->solve();
    device_fence();
    // aux niveau 0 (grossier) : phi + grad depuis phi_coarse (memes stencils centres que le chemin Option A).
    fill_ghosts(fac_->phi_coarse(), dom, mg_.bc());
    detail::coupler_grad_phi(fac_->phi_coarse(), stack_.aux(0), Real(1) / (Real(2) * geom_.dx()),
                             Real(1) / (Real(2) * geom_.dy()));
    fill_boundary(stack_.aux(0), dom, Periodicity{true, true});
    // aux niveau 1 (fin) : phi + grad depuis phi_fine -> grad FIN (diff centree fine, lit les ghosts C-F
    // bilineaires) = le gain de fidelite vs le grad grossier constant injecte par Option A.
    detail::coupler_grad_phi(fac_->phi_fine(), stack_.aux(1), Real(1) / (Real(2) * L[1].dx),
                             Real(1) / (Real(2) * L[1].dy));
  }

  static bool same_box(const Box2D& a, const Box2D& b) {
    return a.lo[0] == b.lo[0] && a.lo[1] == b.lo[1] && a.hi[0] == b.hi[0] && a.hi[1] == b.hi[1];
  }

  Model model_;
  Geometry geom_;
  Elliptic mg_;
  AmrLevelStack<AmrLevelMP> stack_;
  bool replicated_coarse_;  // niveau 0 replique (true) ou multi-box reparti (false, de-replication)
  // Chemin Poisson COMPOSITE FAC (opt-in, set_composite_poisson). fac_ construit paresseusement sur le
  // patch fin courant (reconstruit si le patch change apres regrid). Defaut OFF -> Option A bit-identique.
  bool composite_poisson_ = false;
  bool fac_built_ = false;
  std::shared_ptr<CompositeFacPoisson> fac_;
  Box2D fac_fine_box_{};
};

}  // namespace adc
