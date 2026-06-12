#pragma once

#include <adc/core/types.hpp>
#include <adc/numerics/elliptic/cut_fraction.hpp>
#include <adc/numerics/elliptic/poisson_operator.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/mesh/refinement.hpp>
#include <adc/parallel/comm.hpp>

#include <cstdio>   // ADC_TRACE_SOLVE_FIELDS : trace de diagnostic device (#93), inerte par defaut
#include <cstdlib>  // getenv
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

// Trace de DIAGNOSTIC du V-cycle MG (jalon #93). Active uniquement si ADC_TRACE_SOLVE_FIELDS est pose ;
// stderr + flush immediat pour localiser le dernier marqueur avant un crash device. INERTE par defaut.
namespace detail {
inline void mg_trace_mark(const char* w) {
  static const bool on = std::getenv("ADC_TRACE_SOLVE_FIELDS") != nullptr;
  if (on) { std::fprintf(stderr, "[mg] %s\n", w); std::fflush(stderr); }
}

// Copie de la composante 0 d'un champ fin (eps/eps_y/kappa discretise) vers le niveau fin du MG.
// FONCTEUR NOMME (et non lambda ADC_HD) : meme recette device-clean que le reste (#93). Corps
// identique -> bit-identique. Inerte sur le chemin a eps constant, exerce des qu'un champ est cable.
struct CopyComp0Kernel {
  Array4 d;
  ConstArray4 s;
  ADC_HD void operator()(int i, int j) const { d(i, j) = s(i, j, 0); }
};
}  // namespace detail

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
  //
  // cut_cell + levelset : bord embedded d'ORDRE 2 (Shortley-Weller) au lieu de
  // l'escalier. levelset(x, y) est une fonction de niveau (< 0 a l'interieur, signe
  // du bord) ; pour le cercle conducteur, levelset = hypot(x - cx, y - cy) - Rwall.
  // Chaque cellule active recoit 5 coefficients calcules a partir des distances au
  // bord (fraction de coupe theta par direction). active est alors deduit du signe de
  // levelset s'il n'est pas fourni. cut_cell=false => stencil escalier historique (bit-identique).
  GeometricMG(const Geometry& geom, const BoxArray& ba, const BCRec& bc,
              std::function<bool(Real, Real)> active = {}, bool replicated = false,
              int min_coarse = 2, int nu1 = 2, int nu2 = 2, int nbottom = 50,
              bool cut_cell = false, std::function<Real(Real, Real)> levelset = {})
      : bc_(bc), active_(std::move(active)), nu1_(nu1), nu2_(nu2),
        nbottom_(nbottom), replicated_(replicated), cut_cell_(cut_cell),
        levelset_(std::move(levelset)) {
    if (cut_cell_ && levelset_ && !active_)
      active_ = [ls = levelset_](Real x, Real y) { return ls(x, y) < Real(0); };
    add_level(geom, ba);
    while (true) {
      const Geometry g = lev_.back().geom;
      if (g.domain.nx() % 2 || g.domain.ny() % 2) break;
      if (g.domain.nx() / 2 < min_coarse || g.domain.ny() / 2 < min_coarse)
        break;
      // Stop si une boite du niveau courant ne se coarsen pas PROPREMENT : sur un domaine
      // MULTI-BOX (max_grid_size < n), les boites retrecissent par 2 a chaque niveau et
      // finissent a 1 cellule ; coarsen(ba, 2) ferait alors retomber PLUSIEURS boites fines
      // distinctes sur la MEME cellule grossiere -> BoxArray grossier DEGENERE (boites en
      // doublon couvrant la meme cellule). average_down lit un bloc r x r par cellule grossiere
      // (F(r*I+a, r*J+b)) : pour un fab fin de 1 cellule (0 fantome) trois des quatre lectures
      // tombent HORS des bornes du buffer (indices negatifs), donc en memoire non initialisee.
      // En serie le tas est stable (lecture deterministe), mais sous le chemin MPI le tas est
      // remue et la lecture devient ERRATIQUE (ecart ponctuel jusqu'au blow-up). On garde donc
      // le niveau courant comme grille la plus grossiere. refine(coarsen(b)) == b caracterise
      // exactement les boites alignees ET de taille paire (coarsening exact, sans doublon ni
      // debordement) ; mono-box et multi-box non degenere ne franchissent jamais ce break ->
      // hierarchie (et resultat) STRICTEMENT inchanges sur ces cas.
      const BoxArray& cur = lev_.back().ba;
      bool coarsenable = true;
      for (int i = 0; i < cur.size(); ++i)
        if (!(cur[i].coarsen(2).refine(2) == cur[i])) { coarsenable = false; break; }
      if (!coarsenable) break;
      Geometry gc{g.domain.coarsen(2), g.xlo, g.xhi, g.ylo, g.yhi};
      add_level(gc, coarsen(lev_.back().ba, 2));
    }
    // Tampons de V-cycle (corr/cfine) alloues UNE fois pour chaque niveau NON-bottom. cfine adopte le
    // layout exact que average_down/interpolate auraient alloue en interne : coarsen(L.ba, 2) sur la dmap
    // FINE (L.dm), 0 ghost. Il est REUTILISE pour la restriction (average_down(L.res, C.rhs)) ET la
    // prolongation (interpolate(C.phi, L.corr)) du meme niveau (usages disjoints dans le temps -> un seul
    // tampon suffit). Le bottom n'en a pas besoin (retour anticipe de vcycle_rec) et sa coarsen serait
    // degeneree (raison meme de l'arret du coarsening) -> non alloue.
    for (int l = 0; l + 1 < static_cast<int>(lev_.size()); ++l) {
      lev_[l].corr  = MultiFab(lev_[l].ba, lev_[l].dm, 1, 0);
      lev_[l].cfine = MultiFab(coarsen(lev_[l].ba, 2), lev_[l].dm, 1, 0);
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
    if (cut_cell_ && levelset_) {
      // coefficients Shortley-Weller par cellule active, calcules par niveau a partir
      // des fractions de coupe du level-set (crossing lineaire). w_diag grossit pres du
      // bord (cellule coupee) mais le systeme RESTE diagonalement dominant (le GS converge) :
      // on ne clampe theta qu'a 1e-3 pour eviter la division par 0, sans degrader l'ordre 2
      // (un clamp plus large, ex. 0.05, deplace les pires cellules coupees et casse l'ordre).
      for (auto& L : lev_) {
        L.coef = MultiFab(L.ba, L.dm, 5, 0);
        const Geometry& g = L.geom;
        const Real dx = g.dx(), dy = g.dy();
        for (int li = 0; li < L.coef.local_size(); ++li) {
          Array4 c = L.coef.fab(li).array();
          const ConstArray4 m = L.mask.fab(li).const_array();
          const Box2D b = L.coef.box(li);
          // Primitive PARTAGEE de croisement de face (cut_fraction.hpp) : MEME geometrie d'ouverture
          // que le futur transport EB. detail::cut_fraction reprend a l'identique l'ancienne lambda
          // 'cut' (cut_distance, memes branches et meme clamp 1e-3) et detail::shortley_weller la
          // formule des 5 poids -> coef BIT-IDENTIQUE a l'assemblage inline d'avant le refactor.
          const auto& ls = levelset_;
          for (int j = b.lo[1]; j <= b.hi[1]; ++j)
            for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
              if (m(i, j) == Real(0)) {  // conducteur : coef inutilise (cellule sautee)
                for (int k = 0; k < 5; ++k) c(i, j, k) = 0;
                continue;
              }
              const detail::CutFraction cf =
                  detail::cut_fraction(ls, g.x_cell(i), g.y_cell(j), dx, dy);
              const detail::ShortleyWellerWeights w = detail::shortley_weller(cf);
              c(i, j, 0) = w.w_xm;    // w_xm sur p(i-1)
              c(i, j, 1) = w.w_xp;    // w_xp sur p(i+1)
              c(i, j, 2) = w.w_ym;    // w_ym sur p(i,j-1)
              c(i, j, 3) = w.w_yp;    // w_yp sur p(i,j+1)
              c(i, j, 4) = w.w_diag;  // w_diag
            }
        }
      }
    }
  }

  MultiFab& phi() { return lev_[0].phi; }
  MultiFab& rhs() { return lev_[0].rhs; }
  const Geometry& geom() const { return lev_[0].geom; }
  int num_levels() const { return static_cast<int>(lev_.size()); }

  // Active la permittivite VARIABLE eps(x) : l'operateur passe de lap(phi)=f a
  // div(eps grad phi)=f. eps est un champ AU CENTRE des cellules, evalue par la
  // fonction analytique fournie sur CHAQUE niveau de la hierarchie (comme le masque
  // et les coefficients cut-cell), puis ses ghosts sont remplis. Evaluer eps niveau
  // par niveau (plutot que restreindre depuis le niveau fin) donne la permittivite
  // EXACTE a chaque resolution grossiere, ce qui preserve l'ordre 2. Appeler une fois
  // apres construction, avant solve. NE PAS appeler => eps uniforme (chemin historique).
  void set_epsilon(std::function<Real(Real, Real)> eps_fn) {
    const BCRec ebc = eps_bc();
    for (auto& L : lev_) {
      L.eps = MultiFab(L.ba, L.dm, 1, 1);  // 1 ghost : voisins de bord de box lus
      const Geometry& g = L.geom;
      for (int li = 0; li < L.eps.local_size(); ++li) {
        Array4 e = L.eps.fab(li).array();
        const Box2D b = L.eps.box(li);
        // initialisation hote (fonction std::function non device-callable)
        for (int j = b.lo[1]; j <= b.hi[1]; ++j)
          for (int i = b.lo[0]; i <= b.hi[0]; ++i)
            e(i, j) = eps_fn(g.x_cell(i), g.y_cell(j));
      }
      fill_ghosts(L.eps, g.domain, ebc);
    }
    has_eps_ = true;
  }

  // Surcharge prenant un champ eps DEJA discretise (MultiFab a 1 composante, defini
  // sur la grille du niveau le plus fin). Il est copie sur le niveau fin puis
  // RESTREINT (average_down, moyenne 2x2) vers les niveaux grossiers, et ses ghosts
  // sont remplis a chaque niveau. A utiliser quand eps vient d'un champ par cellule
  // (pas d'une formule analytique) : c'est le point d'entree pour le cablage System.
  void set_epsilon(const MultiFab& eps_fine) {
    const BCRec ebc = eps_bc();
    for (auto& L : lev_) L.eps = MultiFab(L.ba, L.dm, 1, 1);
    // niveau fin : copie de la composante 0
    for (int li = 0; li < lev_[0].eps.local_size(); ++li) {
      Array4 e = lev_[0].eps.fab(li).array();
      const ConstArray4 s = eps_fine.fab(li).const_array();
      const Box2D b = lev_[0].eps.box(li);
      for_each_cell(b, detail::CopyComp0Kernel{e, s});
    }
    fill_ghosts(lev_[0].eps, lev_[0].geom.domain, ebc);
    // niveaux grossiers : moyenne conservative du milieu, puis ghosts
    for (int l = 1; l < num_levels(); ++l) {
      average_down(lev_[l - 1].eps, lev_[l].eps, 2);
      fill_ghosts(lev_[l].eps, lev_[l].geom.domain, ebc);
    }
    has_eps_ = true;
  }

  // Active la permittivite ANISOTROPE : l'operateur passe de div(eps grad phi) (eps
  // scalaire) a div(diag(eps_x, eps_y) grad phi). Les faces NORMALES A X utilisent eps_x,
  // les faces NORMALES A Y utilisent eps_y. eps_x est cable comme le eps isotrope (set
  // le champ eps interne, faces x) et eps_y un SECOND champ (faces y). Memes conventions
  // que set_epsilon : champ AU CENTRE, evalue PAR NIVEAU (permittivite exacte au grossier,
  // ordre 2 preserve) puis ghosts remplis. Cas d'usage : milieu/maillage anisotrope.
  // Donner eps_x_fn == eps_y_fn redonne l'operateur isotrope eps=eps_x. Composable avec
  // set_reaction (kappa). Appeler une fois apres construction, avant solve.
  void set_epsilon_anisotropic(std::function<Real(Real, Real)> eps_x_fn,
                               std::function<Real(Real, Real)> eps_y_fn) {
    set_epsilon(std::move(eps_x_fn));  // faces x : reutilise le cablage eps isotrope
    const BCRec ebc = eps_bc();
    for (auto& L : lev_) {
      L.eps_y = MultiFab(L.ba, L.dm, 1, 1);  // 1 ghost : voisins de bord de box lus
      const Geometry& g = L.geom;
      for (int li = 0; li < L.eps_y.local_size(); ++li) {
        Array4 e = L.eps_y.fab(li).array();
        const Box2D b = L.eps_y.box(li);
        for (int j = b.lo[1]; j <= b.hi[1]; ++j)
          for (int i = b.lo[0]; i <= b.hi[0]; ++i)
            e(i, j) = eps_y_fn(g.x_cell(i), g.y_cell(j));
      }
      fill_ghosts(L.eps_y, g.domain, ebc);
    }
    has_eps_y_ = true;
  }

  // Surcharge prenant deux champs DEJA discretises (grille du niveau le plus fin), copies
  // sur le niveau fin puis RESTREINTS (average_down) vers les grossiers et ghosts remplis,
  // exactement comme set_epsilon(const MultiFab&). Point d'entree pour un cablage par champ
  // (ex. depuis System). eps_x porte les faces x, eps_y les faces y.
  void set_epsilon_anisotropic(const MultiFab& eps_x_fine, const MultiFab& eps_y_fine) {
    set_epsilon(eps_x_fine);  // faces x : reutilise le cablage eps isotrope (+ restriction)
    const BCRec ebc = eps_bc();
    for (auto& L : lev_) L.eps_y = MultiFab(L.ba, L.dm, 1, 1);
    for (int li = 0; li < lev_[0].eps_y.local_size(); ++li) {
      Array4 e = lev_[0].eps_y.fab(li).array();
      const ConstArray4 s = eps_y_fine.fab(li).const_array();
      const Box2D b = lev_[0].eps_y.box(li);
      for_each_cell(b, detail::CopyComp0Kernel{e, s});
    }
    fill_ghosts(lev_[0].eps_y, lev_[0].geom.domain, ebc);
    for (int l = 1; l < num_levels(); ++l) {
      average_down(lev_[l - 1].eps_y, lev_[l].eps_y, 2);
      fill_ghosts(lev_[l].eps_y, lev_[l].geom.domain, ebc);
    }
    has_eps_y_ = true;
  }

  // Active le terme de REACTION kappa(x) : l'operateur passe de div(eps grad phi) = f a
  // div(eps grad phi) - kappa phi = f (Poisson ECRANTE / Helmholtz ; kappa = 1/lambda_D^2 pour
  // l'ecrantage de Debye). kappa >= 0 rend l'operateur plus diagonalement dominant (la multigrille
  // converge au moins aussi bien). C'est un coefficient PHYSIQUE (unite 1/longueur^2), DIAGONAL :
  // lu en (i,j) seulement (aucun voisin), donc 0 ghost ; restreint par moyenne sur les niveaux
  // grossiers (meme valeur physique echantillonnee). NE PAS appeler => kappa = 0 (Poisson, chemin
  // historique strictement inchange). Composable avec set_epsilon (eps(x) et kappa(x) ensemble).
  void set_reaction(std::function<Real(Real, Real)> kappa_fn) {
    for (auto& L : lev_) {
      L.kappa = MultiFab(L.ba, L.dm, 1, 0);
      const Geometry& g = L.geom;
      for (int li = 0; li < L.kappa.local_size(); ++li) {
        Array4 k = L.kappa.fab(li).array();
        const Box2D b = L.kappa.box(li);
        for (int j = b.lo[1]; j <= b.hi[1]; ++j)
          for (int i = b.lo[0]; i <= b.hi[0]; ++i)
            k(i, j) = kappa_fn(g.x_cell(i), g.y_cell(j));
      }
    }
    has_kappa_ = true;
  }

  // Surcharge : champ kappa DEJA discretise (MultiFab 1 composante, grille fine), copie sur le
  // niveau fin puis RESTREINT (average_down) vers les grossiers. Point d'entree pour le cablage
  // System (un champ kappa par cellule).
  void set_reaction(const MultiFab& kappa_fine) {
    for (auto& L : lev_) L.kappa = MultiFab(L.ba, L.dm, 1, 0);
    for (int li = 0; li < lev_[0].kappa.local_size(); ++li) {
      Array4 k = lev_[0].kappa.fab(li).array();
      const ConstArray4 s = kappa_fine.fab(li).const_array();
      const Box2D b = lev_[0].kappa.box(li);
      for_each_cell(b, detail::CopyComp0Kernel{k, s});
    }
    for (int l = 1; l < num_levels(); ++l) average_down(lev_[l - 1].kappa, lev_[l].kappa, 2);
    has_kappa_ = true;
  }

  // Active les COEFFICIENTS HORS-DIAGONAUX du tenseur PLEIN A = [[eps_x, Axy], [Ayx, eps_y]] :
  // l'operateur passe de div(diag(eps_x, eps_y) grad phi) a div(A grad phi), en ajoutant les flux
  // CROISES d_x(Axy d_y phi) + d_y(Ayx d_x phi) (cf. poisson_operator.hpp). A peut etre NON
  // symetrique (Axy != Ayx). Memes conventions que set_epsilon : champs AU CENTRE, evalues PAR
  // NIVEAU (coefficient exact au grossier) puis ghosts remplis (la moyenne de face lit le voisin a
  // i+-1 / j+-1). Composable avec set_epsilon[_anisotropic] et set_reaction. Appeler une fois apres
  // construction, avant solve. NE PAS appeler => bloc DIAGONAL (chemin actuel bit-identique).
  // AVERTISSEMENT : pour A fortement non symetrique le V-cycle GS-5-points (lisseur du bloc
  // DIAGONAL, termes croises EXPLICITES) peut ne PAS converger ; un Krylov serait alors requis.
  void set_cross_terms(std::function<Real(Real, Real)> a_xy_fn,
                       std::function<Real(Real, Real)> a_yx_fn) {
    const BCRec ebc = eps_bc();
    for (auto& L : lev_) {
      L.a_xy = MultiFab(L.ba, L.dm, 1, 1);  // 1 ghost : la moyenne de face lit le voisin de bord
      L.a_yx = MultiFab(L.ba, L.dm, 1, 1);
      const Geometry& g = L.geom;
      for (int li = 0; li < L.a_xy.local_size(); ++li) {
        Array4 fxy = L.a_xy.fab(li).array();
        Array4 fyx = L.a_yx.fab(li).array();
        const Box2D b = L.a_xy.box(li);
        // initialisation hote (std::function non device-callable) ; memoire unifiee avant kernel
        for (int j = b.lo[1]; j <= b.hi[1]; ++j)
          for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
            const Real x = g.x_cell(i), y = g.y_cell(j);
            fxy(i, j) = a_xy_fn(x, y);
            fyx(i, j) = a_yx_fn(x, y);
          }
      }
      fill_ghosts(L.a_xy, g.domain, ebc);
      fill_ghosts(L.a_yx, g.domain, ebc);
    }
    has_cross_ = true;
  }

  // Surcharge prenant deux champs DEJA discretises (grille du niveau le plus fin), copies sur le
  // niveau fin puis RESTREINTS (average_down) vers les grossiers et ghosts remplis, exactement comme
  // set_epsilon_anisotropic(const MultiFab&, const MultiFab&). Point d'entree pour des termes croises
  // PAR CELLULE (ex. A = I + c rho B^{-1} de la condensation de Schur, ou rho varie en espace, donc
  // a_xy/a_yx ne sont pas des formules analytiques mais des champs). Les coefficients croises ne
  // servent QUE le residu / la matvec PLEINE (le lisseur GS reste 5 points, bloc diagonal) ; leur
  // restriction au grossier ne sert donc qu'a un eventuel residu MG sur l'operateur plein (le
  // preconditionneur Krylov, lui, est cable SANS termes croises -> partie symetrique). NE PAS appeler
  // => bloc DIAGONAL (chemin actuel bit-identique).
  void set_cross_terms(const MultiFab& a_xy_fine, const MultiFab& a_yx_fine) {
    const BCRec ebc = eps_bc();
    for (auto& L : lev_) {
      L.a_xy = MultiFab(L.ba, L.dm, 1, 1);
      L.a_yx = MultiFab(L.ba, L.dm, 1, 1);
    }
    for (int li = 0; li < lev_[0].a_xy.local_size(); ++li) {
      Array4 fxy = lev_[0].a_xy.fab(li).array();
      Array4 fyx = lev_[0].a_yx.fab(li).array();
      const ConstArray4 sxy = a_xy_fine.fab(li).const_array();
      const ConstArray4 syx = a_yx_fine.fab(li).const_array();
      const Box2D b = lev_[0].a_xy.box(li);
      for_each_cell(b, detail::CopyComp0Kernel{fxy, sxy});
      for_each_cell(b, detail::CopyComp0Kernel{fyx, syx});
    }
    fill_ghosts(lev_[0].a_xy, lev_[0].geom.domain, ebc);
    fill_ghosts(lev_[0].a_yx, lev_[0].geom.domain, ebc);
    for (int l = 1; l < num_levels(); ++l) {
      average_down(lev_[l - 1].a_xy, lev_[l].a_xy, 2);
      average_down(lev_[l - 1].a_yx, lev_[l].a_yx, 2);
      fill_ghosts(lev_[l].a_xy, lev_[l].geom.domain, ebc);
      fill_ghosts(lev_[l].a_yx, lev_[l].geom.domain, ebc);
    }
    has_cross_ = true;
  }

  void vcycle() { vcycle_rec(0, bc_); }

  // V-cycles jusqu'a residu sous le plancher mixte (ou max_cycles). Renvoie le nombre
  // de cycles effectues. phi est conserve entre appels (warm start).
  //
  // Critere d'arret MIXTE relatif/absolu (convention hypre/AMReX) :
  //   residu <= max(rel_tol * r0, abs_tol)
  // abs_tol est un plancher ABSOLU sur la norme du residu (MEMES unites que current_residual(),
  // donc rapporte a l'echelle du probleme par l'appelant qui la connait : aucune constante magique
  // n'est cuite ici). Defaut 0 -> max(rel_tol*r0, 0) = rel_tol*r0, soit le critere relatif
  // historique a l'identique. Le plancher evite de sur-resoudre un etat DEJA converge (r0 minuscule,
  // typique d'un solve HORS PAS sur etat inchange) : early-exit sans cycler si r0 est sous abs_tol.
  int solve(Real rel_tol, int max_cycles, Real abs_tol = Real(0)) {
    detail::mg_trace_mark("solve: avant current_residual initial");
    const Real r0 = current_residual();
    detail::mg_trace_mark("solve: apres current_residual initial");
    if (r0 <= abs_tol) return 0;  // deja sous le plancher (ou nul) ; abs_tol=0 -> ancien test r0<=0
    const Real stop = (rel_tol * r0 > abs_tol) ? rel_tol * r0 : abs_tol;  // max(rel_tol*r0, abs_tol)
    for (int c = 1; c <= max_cycles; ++c) {
      detail::mg_trace_mark("solve: avant vcycle");
      vcycle();
      detail::mg_trace_mark("solve: apres vcycle");
      if (current_residual() <= stop) return c;
    }
    return max_cycles;
  }

  // Interface du concept EllipticSolver : solve() sans argument (tolerance par
  // defaut) et residual() (alias de current_residual). Permet aux coupleurs de
  // dependre du concept, pas de GeometricMG en dur. Propage abs_tol_ (plancher
  // absolu, defaut 0 -> critere relatif historique a l'identique) au critere mixte.
  void solve() { solve(Real(1e-8), 50, abs_tol_); }
  Real residual() { return current_residual(); }

  // Plancher ABSOLU sur le residu utilise par le solve() sans argument (chemin du concept
  // EllipticSolver, emprunte par les coupleurs / le runtime). Memes unites que residual().
  // Defaut 0 : le critere reste purement relatif (comportement historique bit-identique).
  // Le poser > 0 (a une valeur rapportee a l'echelle du probleme, ex. eps * ||rhs||) fait sortir
  // sans cycler les solves HORS PAS sur un etat deja converge (residu initial sous le plancher).
  void set_abs_tol(Real abs_tol) { abs_tol_ = abs_tol; }
  Real abs_tol() const { return abs_tol_; }

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
    for (int c = 1; c <= max_cycles; ++c) {  // phase 1 : EXACTEMENT le corps de solve()
      vcycle();
      ++total;
      if (current_residual() <= rel_tol * r0) return total;  // -> bit-identique aux runs enregistres
    }
    if (current_residual() <= r0) return total;  // stagnation (pas divergence) : on garde tel quel
    // phase 2 : divergence du V-cycle au bord embedded. Durcissement du lissage LOCAL au solve
    // (nu1_/nu2_ sauves puis RESTAURES avant chaque retour) : pas de ratchet permanent sur le hot
    // path, le surcout n'est paye QUE par le solve qui diverge ; les solves suivants repartent au
    // lissage nominal (reproductibilite preservee, cout independant de l'historique). Restart a
    // froid (phi=0, le warm start portait l'etat diverge). Plus de lissage rend le cycle contractant.
    const int nu1_save = nu1_, nu2_save = nu2_;
    while (nu1_ < 64 || nu2_ < 64) {
      if (nu1_ < 64) nu1_ *= 2;
      if (nu2_ < 64) nu2_ *= 2;
      lev_[0].phi.set_val(Real(0));
      for (int c = 1; c <= max_cycles; ++c) {
        vcycle();
        ++total;
        if (current_residual() <= rel_tol * r0) { nu1_ = nu1_save; nu2_ = nu2_save; return total; }
      }
    }
    nu1_ = nu1_save; nu2_ = nu2_save;
    return total;  // meilleur effort au lissage maximal (residu deja sous r0 : pas de divergence)
  }

  // Residu courant (norme infinie) au niveau le plus fin. all_reduce_max OBLIGATOIRE pour
  // un grossier MULTI-BOX REPARTI : sans lui, norm_inf rend le max LOCAL (different par rang),
  // donc le critere d'arret du V-cycle se declenche a des iterations differentes selon le rang
  // -> nombre de V-cycles (et d'appels fill_boundary) different -> desynchronisation des flux
  // MPI (MPI_ERR_TRUNCATE). Idempotent sous replication (max local = global sur chaque rang) et
  // identite en serie -> bit-identique au comportement historique.
  Real current_residual() {
    detail::mg_trace_mark("current_residual: avant poisson_residual");
    poisson_residual(lev_[0].phi, lev_[0].rhs, lev_[0].geom, bc_, lev_[0].res,
                     mask_ptr(0), coef_ptr(0), eps_ptr(0), kappa_ptr(0), eps_y_ptr(0),
                     a_xy_ptr(0), a_yx_ptr(0));
    detail::mg_trace_mark("current_residual: apres poisson_residual, avant norm_inf");
    const Real r = all_reduce_max(norm_inf(lev_[0].res));
    detail::mg_trace_mark("current_residual: apres norm_inf");
    return r;
  }

  // ACCES aux pointeurs de coefficient de l'operateur du NIVEAU FIN (level 0) et a la CL. Exposent
  // EXACTEMENT ce que current_residual() passe a poisson_residual : un appelant externe (le solveur
  // de Krylov, qui utilise apply_laplacian comme matvec et a besoin d'une matvec COHERENTE avec le
  // residu MG) reutilise ainsi le meme operateur, sans dupliquer le cablage des champs eps/kappa/Axy.
  // nullptr quand le terme correspondant est inactif (cf. les *_ptr internes). Additif : aucun chemin
  // existant ne les appelle, le comportement par defaut est inchange.
  const MultiFab* op_mask() { return mask_ptr(0); }
  const MultiFab* op_coef() { return coef_ptr(0); }
  const MultiFab* op_eps() { return eps_ptr(0); }
  const MultiFab* op_kappa() { return kappa_ptr(0); }
  const MultiFab* op_eps_y() { return eps_y_ptr(0); }
  const MultiFab* op_a_xy() { return a_xy_ptr(0); }
  const MultiFab* op_a_yx() { return a_yx_ptr(0); }
  const BCRec& bc() const { return bc_; }
  const BoxArray& box_array() const { return lev_[0].ba; }
  const DistributionMapping& dmap() const { return lev_[0].dm; }

 private:
  struct MGLevel {
    Geometry geom;
    BoxArray ba;
    DistributionMapping dm;
    MultiFab phi, rhs, res, mask, coef, eps, kappa, eps_y, a_xy, a_yx;
    // Tampons de V-cycle REUTILISES, alloues une fois par le constructeur pour les niveaux NON-bottom :
    // corr = correction prolongee (layout du niveau) ; cfine = grille "fin coarsen" partagee par la
    // restriction (average_down) et la prolongation (interpolate) du niveau. Le bottom les laisse vides
    // (vcycle_rec y retourne avant de les toucher, et sa coarsen serait degeneree).
    MultiFab corr, cfine;
  };

  const MultiFab* mask_ptr(int l) { return active_ ? &lev_[l].mask : nullptr; }
  const MultiFab* coef_ptr(int l) { return cut_cell_ ? &lev_[l].coef : nullptr; }
  const MultiFab* eps_ptr(int l) { return has_eps_ ? &lev_[l].eps : nullptr; }
  const MultiFab* kappa_ptr(int l) { return has_kappa_ ? &lev_[l].kappa : nullptr; }
  // eps_y absent => nullptr => operateur isotrope (eps_y = eps_x) inchange.
  const MultiFab* eps_y_ptr(int l) { return has_eps_y_ ? &lev_[l].eps_y : nullptr; }
  // termes croises absents => nullptr => bloc DIAGONAL (chemin actuel inchange).
  const MultiFab* a_xy_ptr(int l) { return has_cross_ ? &lev_[l].a_xy : nullptr; }
  const MultiFab* a_yx_ptr(int l) { return has_cross_ ? &lev_[l].a_yx : nullptr; }

  // CL utilisee pour remplir les ghosts du champ eps : on garde le periodique mais
  // on remplace tout bord physique (Dirichlet ou outflow de phi) par une
  // extrapolation gradient-nul (eps_ghost = eps interieur), ce qui donne une
  // permittivite de face = eps au bord (face sur le contour du domaine).
  BCRec eps_bc() const {
    auto fo = [](BCType t) { return t == BCType::Periodic ? t : BCType::Foextrap; };
    BCRec b;
    b.xlo = fo(bc_.xlo); b.xhi = fo(bc_.xhi);
    b.ylo = fo(bc_.ylo); b.yhi = fo(bc_.yhi);
    return b;
  }

  void add_level(const Geometry& g, const BoxArray& ba) {
    DistributionMapping dm = replicated_
        ? DistributionMapping(std::vector<int>(ba.size(), my_rank()))
        : DistributionMapping(ba.size(), n_ranks());
    lev_.push_back(MGLevel{g, ba, dm, MultiFab(ba, dm, 1, 1),
                           MultiFab(ba, dm, 1, 0), MultiFab(ba, dm, 1, 0),
                           MultiFab{}, MultiFab{}, MultiFab{}, MultiFab{}, MultiFab{},
                           MultiFab{}, MultiFab{}, MultiFab{}, MultiFab{}});
  }

  void vcycle_rec(int l, const BCRec& bc) {
    MGLevel& L = lev_[l];
    const MultiFab* mk = mask_ptr(l);
    const MultiFab* ck = coef_ptr(l);
    const MultiFab* ep = eps_ptr(l);
    const MultiFab* kp = kappa_ptr(l);
    const MultiFab* ey = eps_y_ptr(l);  // nullptr => isotrope (eps_y = eps_x)
    const MultiFab* axy = a_xy_ptr(l);  // nullptr => bloc diagonal (pas de flux croise)
    const MultiFab* ayx = a_yx_ptr(l);
    // NB : gs_smooth reste 5 POINTS (bloc diagonal). Les termes croises sont EXPLICITES : seul le
    // residu (poisson_residual) les porte. Le lisseur GS ne touche que la diagonale -> sa diag reste
    // dominante (kappa>=0, eps>0) ; le couplage croise est relegue au residu, comme la convention de
    // l'entete. Pour A symetrique-defini-positif le V-cycle reste contractant ; pour A non symetrique
    // fort, il peut diverger (cf. set_cross_terms, observation reportee).
    if (l == 0) detail::mg_trace_mark("vcycle_rec(0): avant gs_smooth(nu1) [premier kernel GS]");
    gs_smooth(L.phi, L.rhs, L.geom, bc, nu1_, mk, ck, ep, kp, ey);
    if (l == 0) detail::mg_trace_mark("vcycle_rec(0): apres gs_smooth(nu1)");

    if (l + 1 == static_cast<int>(lev_.size())) {
      gs_smooth(L.phi, L.rhs, L.geom, bc, nbottom_, mk, ck, ep, kp, ey);  // bottom solve
      if (mk) zero_conductor(L.phi, L.mask);
      return;
    }

    poisson_residual(L.phi, L.rhs, L.geom, bc, L.res, mk, ck, ep, kp, ey, axy, ayx);
    if (l == 0) detail::mg_trace_mark("vcycle_rec(0): apres poisson_residual");
    MGLevel& C = lev_[l + 1];
    average_down(L.res, C.rhs, 2, L.cfine);  // restriction du residu (tampon cfine reutilise)
    if (l == 0) detail::mg_trace_mark("vcycle_rec(0): apres average_down");
    C.phi.set_val(0.0);
    vcycle_rec(l + 1, homogeneous(bc));
    if (l == 0) detail::mg_trace_mark("vcycle_rec(0): apres recursion grossiere");

    interpolate(C.phi, L.corr, 2, L.cfine);  // prolongation de la correction (tampons corr/cfine reutilises)
    if (l == 0) detail::mg_trace_mark("vcycle_rec(0): apres interpolate");
    saxpy(L.phi, Real(1), L.corr);
    if (l == 0) detail::mg_trace_mark("vcycle_rec(0): apres saxpy");
    if (mk) zero_conductor(L.phi, L.mask);  // refige le conducteur
    gs_smooth(L.phi, L.rhs, L.geom, bc, nu2_, mk, ck, ep, kp, ey);
    if (l == 0) detail::mg_trace_mark("vcycle_rec(0): apres gs_smooth(nu2)");
  }

  BCRec bc_;
  std::function<bool(Real, Real)> active_;
  int nu1_, nu2_, nbottom_;
  bool replicated_ = false;
  bool cut_cell_ = false;
  bool has_eps_ = false;
  bool has_eps_y_ = false;
  bool has_kappa_ = false;
  bool has_cross_ = false;  // coefficients hors-diagonaux Axy/Ayx (tenseur PLEIN) actifs
  Real abs_tol_ = Real(0);  // plancher absolu du solve() sans argument (0 = critere relatif seul)
  std::function<Real(Real, Real)> levelset_;
  std::vector<MGLevel> lev_;
};

}  // namespace adc
