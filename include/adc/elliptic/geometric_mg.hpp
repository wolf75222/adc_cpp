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
  // replicated : si true, chaque niveau (mono-box couvrant le domaine) est REPLIQUE sur
  // tous les rangs (dmap = my_rank() partout) au lieu du round-robin par defaut. Chaque rang
  // resout alors le MEME Poisson grossier redondamment, SANS communication (V-cycle par-fab,
  // fill_boundary sur une box couvrant le domaine est local, et current_residual reduit par
  // norm_inf = all_reduce_MAX, idempotent sous replication). C'est ce qu'attend le coupleur
  // AMR (niveau 0 replique). En serie my_rank()=0 -> identique au round-robin, bit a bit.
  GeometricMG(const Geometry& geom, const BoxArray& ba, const BCRec& bc,
              std::function<bool(Real, Real)> active = {}, bool replicated = false,
              int min_coarse = 2, int nu1 = 2, int nu2 = 2, int nbottom = 50)
      : bc_(bc), active_(std::move(active)), nu1_(nu1), nu2_(nu2),
        nbottom_(nbottom), replicated_(replicated) {
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
          const Box2D b = L.mask.box(li);
          // initialisation hote (predicat std::function non device-callable) ;
          // ecrit la memoire unifiee avant tout kernel.
          for (int j = b.lo[1]; j <= b.hi[1]; ++j)
            for (int i = b.lo[0]; i <= b.hi[0]; ++i)
              m(i, j) = active_(g.x_cell(i), g.y_cell(j)) ? Real(1) : Real(0);
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

  // Interface du concept EllipticSolver : solve() sans argument (tolerance par
  // defaut) et residual() (alias de current_residual). Permet aux coupleurs de
  // dependre du concept, pas de GeometricMG en dur.
  void solve() { solve(Real(1e-8), 50); }
  Real residual() { return current_residual(); }

  // Solve DURCI pour le bord embedded a haute resolution. Sur grille fine, le V-cycle
  // geometrique diverge parfois pres de la paroi conductrice : le coarsening est
  // NON-Galerkin et le masque du cercle est re-evalue par niveau, donc la correction
  // grossiere devient incoherente avec le bord fin et le lissage nu1=nu2=2 ne la domine
  // plus (rayon spectral du cycle > 1). Le potentiel diverge alors a chaque appel (le
  // warm start propage la divergence d'un pas a l'autre), d'ou un nan du champ a haute
  // resolution (voir docs/HERO_RUN_AMR.md). La divergence est ERRATIQUE en resolution
  // (elle depend de l'alignement du cercle sur la hierarchie de grilles).
  //
  // Strategie, BIT-IDENTIQUE quand le solveur converge (ou stagne) deja :
  //   1. cycle standard au lissage courant : EXACTEMENT le corps de solve(rel_tol,
  //      max_cycles), donc identique aux runs deja stables ;
  //   2. SEULEMENT si le residu final EXCEDE le residu initial (vraie divergence,
  //      ratio > 1 ; pas une simple stagnation ratio < 1, qu'on garde telle quelle pour
  //      rester bit-identique) : on durcit le lissage de facon STICKY (nu double, conserve
  //      pour les pas suivants, le warm start repartant alors au lissage durci) et on
  //      REPART A FROID (phi=0, le warm start portait l'etat diverge), jusqu'a convergence
  //      ou saturation de nu. Plus de lissage rend le V-cycle contractant (le GS domine la
  //      correction grossiere incoherente) : cf. balayage, nu=2 diverge a nc=640, nu>=4
  //      converge. Tout run aujourd'hui stable n'a PAS diverge (divergence -> nan -> non
  //      enregistre), donc la phase 2 ne se declenche jamais pour eux : bit-identique.
  int solve_robust(Real rel_tol, int max_cycles) {
    const Real r0 = current_residual();
    if (r0 <= Real(0)) return 0;
    int total = 0;
    for (int c = 1; c <= max_cycles; ++c) {  // phase 1 : identique a solve()
      vcycle();
      ++total;
      if (current_residual() <= rel_tol * r0) return total;
    }
    if (current_residual() <= r0) return total;  // stagnation (pas divergence) : on garde tel quel
    while (nu1_ < 64 || nu2_ < 64) {             // phase 2 : durcissement sticky + restart a froid
      if (nu1_ < 64) nu1_ *= 2;
      if (nu2_ < 64) nu2_ *= 2;
      lev_[0].phi.set_val(Real(0));
      for (int c = 1; c <= max_cycles; ++c) {
        vcycle();
        ++total;
        if (current_residual() <= rel_tol * r0) return total;
      }
    }
    return total;  // meilleur effort au lissage maximal (residu deja sous r0 : pas de divergence)
  }

  // Residu courant (norme infinie) au niveau le plus fin. all_reduce_max OBLIGATOIRE pour
  // un grossier MULTI-BOX REPARTI : sans lui, norm_inf rend le max LOCAL (different par rang),
  // donc le critere d'arret du V-cycle se declenche a des iterations differentes selon le rang
  // -> nombre de V-cycles (et d'appels fill_boundary) different -> desynchronisation des flux
  // MPI (MPI_ERR_TRUNCATE). Idempotent sous replication (max local = global sur chaque rang) et
  // identite en serie -> bit-identique au comportement historique.
  Real current_residual() {
    poisson_residual(lev_[0].phi, lev_[0].rhs, lev_[0].geom, bc_, lev_[0].res,
                     mask_ptr(0));
    return all_reduce_max(norm_inf(lev_[0].res));
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
    DistributionMapping dm = replicated_
        ? DistributionMapping(std::vector<int>(ba.size(), my_rank()))
        : DistributionMapping(ba.size(), n_ranks());
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
  bool replicated_ = false;
  std::vector<MGLevel> lev_;
};

}  // namespace adc
