#pragma once
// Helpers MF de base : avance divergence, source, average_down, ghosts coarse-fine (mono-box).

#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/refinement.hpp>  // coarsen_index
#include <adc/numerics/spatial_operator.hpp>  // compute_face_fluxes, xface_box, yface_box
#include <adc/numerics/time/implicit_stepper.hpp>  // backward_euler_source (pas implicite IMEX)

#include <vector>

namespace adc {

// Methode temporelle d'un pas AMR (sous-cyclage Berger-Oliger). kEuler (DEFAUT) = avance Euler
// avant a chaque sous-pas (chemin historique, strictement bit-identique) ; kSsprk3 = SSPRK3
// (Shu-Osher, 3 etages, ordre 3) avec reflux par etage (flux effectif convexe). L'enum est passe
// PAR VALEUR (POD) le long du chemin advance_amr -> subcycle_level_mp ; l'ABI plate du loader .so
// le transporte sous forme d'entier (AmrBuildParams::time_method), 0 == kEuler.
enum class AmrTimeMethod : int { kEuler = 0, kSsprk3 = 1 };

// Foncteur NOMME device-clean (meme recette que mf_arith.hpp : premiere instanciation possible
// depuis une TU loader externe, ou une lambda etendue fait buter nvcc) du RHS methode-des-lignes
// a UN niveau AMR : R = -div(Fx,Fy) + S(U, aux), evalue a UN MEME etat. C'est la divergence de
// mf_advance_faces (signe oppose, sans dt) FUSIONNEE avec la source de mf_apply_source. Sert
// UNIQUEMENT aux etages SSPRK3 (mf_eval_rhs), ou L(U) = -div F + S doit etre pris au meme etat
// d'etage (vrai SSPRK methode-des-lignes), contrairement au splitting transport-puis-source du
// chemin Euler. Sans source (modele a S == 0) R se reduit a -div F.
template <class Model>
struct AmrSspRhsKernel {
  Model m;
  ConstArray4 u, ax, fx, fy;
  Array4 R;
  Real dx, dy;
  ADC_HD void operator()(int i, int j) const {
    const auto S = m.source(load_state<Model>(u, i, j), load_aux<aux_comps<Model>()>(ax, i, j));
    for (int c = 0; c < Model::n_vars; ++c)
      R(i, j, c) = -((fx(i + 1, j, c) - fx(i, j, c)) / dx + (fy(i, j + 1, c) - fy(i, j, c)) / dy) +
                   S[c];
  }
};

// R <- -div(Fx,Fy) + S(U, aux) sur les cellules valides (RHS methode-des-lignes a UN niveau, evalue
// a l'etat U). Pendant "combine" de mf_advance_faces + mf_apply_source pour les etages SSPRK3 (le
// flux d'etage Fx/Fy est suppose deja calcule par compute_face_fluxes a l'etat U).
template <class Model>
inline void mf_eval_rhs(const Model& m, const MultiFab& U, const MultiFab& aux, const MultiFab& Fx,
                        const MultiFab& Fy, Real dx, Real dy, MultiFab& R) {
  for (int li = 0; li < U.local_size(); ++li)
    for_each_cell(U.box(li),
                  AmrSspRhsKernel<Model>{m, U.fab(li).const_array(), aux.fab(li).const_array(),
                                         Fx.fab(li).const_array(), Fy.fab(li).const_array(),
                                         R.fab(li).array(), dx, dy});
}

/// Foncteur NOMME device-clean : U <- U - dt div(Fx,Fy) sur une cellule valide.
struct AmrAdvanceFacesKernel {
  Array4 u;
  ConstArray4 fx, fy;
  Real dx, dy, dt;
  int nc;
  ADC_HD void operator()(int i, int j) const {
    for (int c = 0; c < nc; ++c)
      u(i, j, c) -= dt * ((fx(i + 1, j, c) - fx(i, j, c)) / dx +
                          (fy(i, j + 1, c) - fy(i, j, c)) / dy);
  }
};

// U <- U - dt div(Fx,Fy) sur les cellules valides (GPU via for_each_cell).
inline void mf_advance_faces(MultiFab& U, const MultiFab& Fx, const MultiFab& Fy,
                             Real dx, Real dy, Real dt) {
  const int nc = U.ncomp();
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 u = U.fab(li).array();
    const ConstArray4 fx = Fx.fab(li).const_array(), fy = Fy.fab(li).const_array();
    for_each_cell(U.box(li), AmrAdvanceFacesKernel{u, fx, fy, dx, dy, dt, nc});
  }
}

// U <- U + dt S(U, aux) sur les cellules valides : terme source applique en Euler
// avant a chaque sous-pas AMR (cellule-local, pas de reflux). Sans cela le chemin AMR
// (compute_face_fluxes -> divergence) ignorerait model.source. Pour un modele a source
// nulle (transport scalaire pur) ceci ajoute dt*0 : bit-identique. La DIFFUSION, elle, est portee
// par compute_face_fluxes comme FLUX de face Fickien (-nu grad u), donc vue par le
// reflux et conservative aux interfaces coarse-fine : ce n'est PAS une source locale.
/// Foncteur NOMME device-clean (template Model, cf. AmrSspRhsKernel) : U <- U + dt S(U, aux)
/// sur une cellule valide.
template <class Model>
struct AmrApplySourceKernel {
  Model m;
  Array4 u;
  ConstArray4 uc, ax;
  Real dt;
  ADC_HD void operator()(int i, int j) const {
    const auto S = m.source(load_state<Model>(uc, i, j),
                            load_aux<aux_comps<Model>()>(ax, i, j));
    for (int c = 0; c < Model::n_vars; ++c) u(i, j, c) += dt * S[c];
  }
};

template <class Model>
inline void mf_apply_source(const Model& m, MultiFab& U, const MultiFab& aux, Real dt) {
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 u = U.fab(li).array();
    const ConstArray4 uc = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    for_each_cell(U.box(li), AmrApplySourceKernel<Model>{m, u, uc, ax, dt});
  }
}

// Traitement temporel de la SOURCE a un sous-pas AMR, apres l'avance de transport
// (mf_advance_faces, deja sans source car compute_face_fluxes ne porte que model.flux) :
//   - EXPLICITE (imex == false, DEFAUT) : Euler avant, U += dt S(U, aux) -- l'appel
//     historique mf_apply_source, donc bit-identique au chemin existant.
//   - IMEX (imex == true) : source IMPLICITE raide, W = U + dt S(W, aux) resolu EN PLACE par
//     backward_euler_source (Newton local, jacobienne par differences finies, foncteur device
//     NOMME BackwardEulerSourceKernel). C'est le pendant AMR de l'avance IMEX du System
//     (block_builder.hpp::AdvanceImex) : meme demi-pas explicite (le transport est porte par le
//     reflux conservatif) + meme pas implicite sur la source. La source restant CELLULE-LOCALE
//     (aucun flux de face), elle n'entre PAS dans les registres de reflux : le split implicite ne
//     touche donc pas la conservation aux interfaces grossier-fin. Le CHOIX est un drapeau runtime
//     (pas de lambda injectee dans le chemin device) : il selectionne deux fonctions HOTE, chacune
//     lancant son propre kernel a foncteur nomme.
//
// OPTIONS NEWTON (@p nopts) : pilotent le Newton local de la source implicite (budget d'iterations,
// tolerances, fd_eps, damping, fail_policy). DEFAUT {} = constantes historiques (2 iters, 1e-7, ...)
// -> chemin (2a) bit-identique a l'ancien appel backward_euler_source(m, aux, U, dt). Le mono-bloc AMR
// (AmrCouplerMP::step) les thread depuis AmrSystem (vague 3 -> options mono-bloc cablees). Le masque
// IMEX partiel n'est PAS porte par ce chemin (mono-bloc coupleur = backward-Euler plein) : on passe
// donc le masque par defaut (inactif). Pas de rapport diagnostics ici (report == nullptr implicite).
template <class Model>
inline void mf_apply_source_treatment(const Model& m, MultiFab& U, const MultiFab& aux, Real dt,
                                      bool imex, const NewtonOptions& nopts = {}) {
  if (imex)
    // Forme a OPTIONS (Newton pilote par nopts), masque inactif, sans rapport. Defaut nopts={} =>
    // identique a la forme historique a iters figes (2), donc bit-identique tant que nopts est defaut.
    backward_euler_source(m, aux, U, dt, nopts, ImplicitMask<Model::n_vars>{});
  else
    mf_apply_source(m, U, aux, dt);        // Euler avant historique (bit-identique)
}

/// Foncteur NOMME device-clean : moyenne 2x2 fin -> grossier sur une cellule grossiere.
struct AmrAverageDownKernel {
  ConstArray4 f;
  Array4 c;
  int nc;
  ADC_HD void operator()(int I, int J) const {
    for (int k = 0; k < nc; ++k)
      c(I, J, k) = Real(0.25) * (f(2 * I, 2 * J, k) + f(2 * I + 1, 2 * J, k) +
                                 f(2 * I, 2 * J + 1, k) + f(2 * I + 1, 2 * J + 1, k));
  }
};

// moyenne fin -> grossier (ratio 2) sur la region couverte (coords grossieres).
inline void mf_average_down(const MultiFab& Uf, MultiFab& Uc, int CI0, int CI1,
                            int CJ0, int CJ1) {
  const int nc = Uc.ncomp();
  const ConstArray4 f = Uf.fab(0).const_array();
  Array4 c = Uc.fab(0).array();
  for_each_cell(Box2D{{CI0, CJ0}, {CI1, CJ1}}, AmrAverageDownKernel{f, c, nc});
}

// Helper coarse-fine de premier niveau (revue, point ghosts) : remplit UNE cellule ghost
// fine (i,j) par interpolation espace (constant par morceaux : cellule grossiere couvrante)
// + temps (lineaire entre l'etat parent ancien/nouveau). frac = position temporelle du
// sous-pas dans le pas parent. Centralise l'arithmetique partagee par mf_fill_fine_ghosts_t
// (mono-box), mf_fill_fine_ghosts_multi (multi-box) et mf_fill_fine_ghosts_mb (multi-niveau) :
// une seule formule (1-frac)*co + frac*cn, bit-identique aux trois corps precedents.
inline void fill_cf_ghost_cell(Array4 f, const ConstArray4& co, const ConstArray4& cn,
                               int i, int j, int nc, Real frac) {
  const int ci = coarsen_index(i, 2), cj = coarsen_index(j, 2);
  for (int k = 0; k < nc; ++k)
    f(i, j, k) = (1 - frac) * co(ci, cj, k) + frac * cn(ci, cj, k);
}

// ghosts du fin = interp espace (constant par morceaux) + temps (lineaire) depuis le
// grossier ancien/nouveau. frac = position temporelle du sous-pas dans le pas grossier.
inline void mf_fill_fine_ghosts_t(MultiFab& Uf, const MultiFab& Uc_old,
                                   const MultiFab& Uc_new, Real frac) {
  device_fence();  // lecture/ecriture hote sur memoire unifiee
  const int nc = Uf.ncomp();
  Array4 f = Uf.fab(0).array();
  const ConstArray4 co = Uc_old.fab(0).const_array();
  const ConstArray4 cn = Uc_new.fab(0).const_array();
  const Box2D v = Uf.box(0), g = Uf.fab(0).grown_box();
  for (int j = g.lo[1]; j <= g.hi[1]; ++j)
    for (int i = g.lo[0]; i <= g.hi[0]; ++i)
      if (!v.contains(i, j)) fill_cf_ghost_cell(f, co, cn, i, j, nc, frac);
}

}  // namespace adc
