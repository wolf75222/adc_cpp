#pragma once

#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/refinement.hpp>  // coarsen_index
#include <adc/numerics/spatial_operator.hpp>  // compute_face_fluxes, xface_box, yface_box
#include <adc/parallel/comm.hpp>  // all_reduce_sum_inplace (reflux multi-patch distribue)

#include <vector>

// Moteur AMR conservatif (Berger-Oliger r=2 + reflux) sur MultiFab + le seam, GENERIQUE
// (Limiter, NumericalFlux, N composantes ; flux par compute_face_fluxes -> MUSCL / HLL /
// HLLC / compressible auto-gravitant / GPU). Deux piles dans ce header :
//   - PRODUCTION : advance_amr (+ LevelHierarchy, AmrLevelMP), moteur N-niveaux
//     MULTI-PATCH et DISTRIBUE (MPI). C'est l'entree unifiee (cf. bas du fichier).
//   - ORACLE (detail::) : amr_step_*_mf + AmrLevelMF, le moteur MONO-BOX d'origine,
//     pendant MultiFab de amr_reflux.hpp::amr_step_2level (Fab2D, 1 comp, Rusanov).
//     Plus utilise en production, conserve pour valider le multi-patch bit-identique
//     au mono-box (chaine Fab2D -> MF -> MP).

namespace adc {

// U <- U - dt div(Fx,Fy) sur les cellules valides (GPU via for_each_cell).
inline void mf_advance_faces(MultiFab& U, const MultiFab& Fx, const MultiFab& Fy,
                             Real dx, Real dy, Real dt) {
  const int nc = U.ncomp();
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 u = U.fab(li).array();
    const ConstArray4 fx = Fx.fab(li).const_array(), fy = Fy.fab(li).const_array();
    for_each_cell(U.box(li), [=] ADC_HD(int i, int j) {
      for (int c = 0; c < nc; ++c)
        u(i, j, c) -= dt * ((fx(i + 1, j, c) - fx(i, j, c)) / dx +
                            (fy(i, j + 1, c) - fy(i, j, c)) / dy);
    });
  }
}

// U <- U + dt S(U, aux) sur les cellules valides : terme source applique en Euler
// avant a chaque sous-pas AMR (cellule-local, pas de reflux). Sans cela le chemin AMR
// (compute_face_fluxes -> divergence) ignorerait model.source. Pour un modele a source
// nulle (transport scalaire pur) ceci ajoute dt*0 : bit-identique. La DIFFUSION, elle, est portee
// par compute_face_fluxes comme FLUX de face Fickien (-nu grad u), donc vue par le
// reflux et conservative aux interfaces coarse-fine : ce n'est PAS une source locale.
template <class Model>
inline void mf_apply_source(const Model& m, MultiFab& U, const MultiFab& aux, Real dt) {
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 u = U.fab(li).array();
    const ConstArray4 uc = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    for_each_cell(U.box(li), [=] ADC_HD(int i, int j) {
      const auto S = m.source(load_state<Model>(uc, i, j),
                              load_aux<aux_comps<Model>()>(ax, i, j));
      for (int c = 0; c < Model::n_vars; ++c) u(i, j, c) += dt * S[c];
    });
  }
}

// moyenne fin -> grossier (ratio 2) sur la region couverte (coords grossieres).
inline void mf_average_down(const MultiFab& Uf, MultiFab& Uc, int CI0, int CI1,
                            int CJ0, int CJ1) {
  const int nc = Uc.ncomp();
  const ConstArray4 f = Uf.fab(0).const_array();
  Array4 c = Uc.fab(0).array();
  for_each_cell(Box2D{{CI0, CJ0}, {CI1, CJ1}}, [=] ADC_HD(int I, int J) {
    for (int k = 0; k < nc; ++k)
      c(I, J, k) = Real(0.25) * (f(2 * I, 2 * J, k) + f(2 * I + 1, 2 * J, k) +
                                 f(2 * I, 2 * J + 1, k) + f(2 * I + 1, 2 * J + 1, k));
  });
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

// === PILE MF : ORACLE DE TEST, HORS PRODUCTION ===================================
// amr_step_2level_mf / amr_step_multilevel_mf + AmrLevelMF : le moteur AMR MONO-BOX
// d'origine. Plus appele en production (AmrCoupler passe par advance_amr, le moteur
// unifie multi-patch). Conserve en detail:: comme ORACLE de validation : la chaine
// Fab2D (amr_reflux.hpp::amr_step_2level) -> MF (ici) -> MP (detail::subcycle_level_mp)
// prouve le moteur MP bit-identique au mono-box, jusqu'a N niveaux (ce que Fab2D seul,
// 2-niveaux/1-comp/Rusanov, ne couvre pas). Gardes : test_amr_reflux_mf (vs Fab2D),
// test_amr_multilevel_mf, test_amr_multilevel_multipatch garde 1 (MP vs MF, max|dUc|=0).
// Les helpers partages restent a portee adc:: (mf_advance_faces, mf_average_down,
// mf_fill_fine_ghosts_t), utilises aussi par les tests d'init.
namespace detail {

// Un pas 2-niveaux conservatif. Uc : grossier (domaine periodique, ghosts pour le
// Limiter). Uf : fin (box raffinee). auxc/auxf : (phi, grad phi) prescrits, ghosts
// remplis. dt = pas grossier ; le fin fait r=2 sous-pas de dt/2 puis reflux.
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void amr_step_2level_mf(const Model& m, MultiFab& Uc, const Box2D& dom, Real dxc,
                        Real dyc, MultiFab& Uf, int CI0, int CI1, int CJ0, int CJ1,
                        const MultiFab& auxc, const MultiFab& auxf, Real dt) {
  const int r = 2, nc = Uc.ncomp();
  const Real dxf = dxc / 2, dyf = dyc / 2, dtf = dt / r;
  const int nJ = CJ1 - CJ0 + 1, nI = CI1 - CI0 + 1;
  MultiFab Uc_old = Uc;  // etat grossier au temps t (interp temporelle des ghosts fins)

  // --- flux grossiers aux 4 faces de la region fine (avant maj) ---
  fill_boundary(Uc, dom, Periodicity{true, true});
  MultiFab fxc(BoxArray(std::vector<Box2D>{xface_box(Uc.box(0))}), Uc.dmap(), nc, 0);
  MultiFab fyc(BoxArray(std::vector<Box2D>{yface_box(Uc.box(0))}), Uc.dmap(), nc, 0);
  compute_face_fluxes<Limiter, NumericalFlux>(m, Uc, auxc, fxc, fyc, dxc, dyc);
  std::vector<Real> cL(nJ * nc), cR(nJ * nc), cB(nI * nc), cT(nI * nc);
  {
    device_fence();
    const ConstArray4 FX = fxc.fab(0).const_array(), FY = fyc.fab(0).const_array();
    for (int J = CJ0; J <= CJ1; ++J)
      for (int k = 0; k < nc; ++k) {
        cL[(J - CJ0) * nc + k] = FX(CI0, J, k);
        cR[(J - CJ0) * nc + k] = FX(CI1 + 1, J, k);
      }
    for (int I = CI0; I <= CI1; ++I)
      for (int k = 0; k < nc; ++k) {
        cB[(I - CI0) * nc + k] = FY(I, CJ0, k);
        cT[(I - CI0) * nc + k] = FY(I, CJ1 + 1, k);
      }
  }
  mf_advance_faces(Uc, fxc, fyc, dxc, dyc, dt);  // Uc -> etat t+dt
  mf_apply_source(m, Uc, auxc, dt);  // source S(U,aux) au sous-pas

  // --- sous-cyclage fin : r sous-pas, accumulation des flux fins (x dtf) ---
  std::vector<Real> fL(nJ * nc, 0), fR(nJ * nc, 0), fB(nI * nc, 0), fT(nI * nc, 0);
  MultiFab fxf(BoxArray(std::vector<Box2D>{xface_box(Uf.box(0))}), Uf.dmap(), nc, 0);
  MultiFab fyf(BoxArray(std::vector<Box2D>{yface_box(Uf.box(0))}), Uf.dmap(), nc, 0);
  for (int s = 0; s < r; ++s) {
    mf_fill_fine_ghosts_t(Uf, Uc_old, Uc, Real(s) / r);
    compute_face_fluxes<Limiter, NumericalFlux>(m, Uf, auxf, fxf, fyf, dxf, dyf);
    device_fence();
    const ConstArray4 FX = fxf.fab(0).const_array(), FY = fyf.fab(0).const_array();
    for (int J = CJ0; J <= CJ1; ++J)
      for (int k = 0; k < nc; ++k) {
        fL[(J - CJ0) * nc + k] +=
            Real(0.5) * (FX(2 * CI0, 2 * J, k) + FX(2 * CI0, 2 * J + 1, k)) * dtf;
        fR[(J - CJ0) * nc + k] += Real(0.5) *
            (FX(2 * CI1 + 2, 2 * J, k) + FX(2 * CI1 + 2, 2 * J + 1, k)) * dtf;
      }
    for (int I = CI0; I <= CI1; ++I)
      for (int k = 0; k < nc; ++k) {
        fB[(I - CI0) * nc + k] +=
            Real(0.5) * (FY(2 * I, 2 * CJ0, k) + FY(2 * I + 1, 2 * CJ0, k)) * dtf;
        fT[(I - CI0) * nc + k] += Real(0.5) *
            (FY(2 * I, 2 * CJ1 + 2, k) + FY(2 * I + 1, 2 * CJ1 + 2, k)) * dtf;
      }
    mf_advance_faces(Uf, fxf, fyf, dxf, dyf, dtf);
    mf_apply_source(m, Uf, auxf, dtf);  // source S(U,aux) au sous-pas
  }

  mf_average_down(Uf, Uc, CI0, CI1, CJ0, CJ1);  // sync des cellules couvertes

  // --- reflux : flux grossier (x dt) remplace par somme des flux fins (x dtf) ---
  device_fence();
  Array4 c = Uc.fab(0).array();
  for (int J = CJ0; J <= CJ1; ++J)
    for (int k = 0; k < nc; ++k) {
      c(CI0 - 1, J, k) -= (fL[(J - CJ0) * nc + k] - cL[(J - CJ0) * nc + k] * dt) / dxc;
      c(CI1 + 1, J, k) += (fR[(J - CJ0) * nc + k] - cR[(J - CJ0) * nc + k] * dt) / dxc;
    }
  for (int I = CI0; I <= CI1; ++I)
    for (int k = 0; k < nc; ++k) {
      c(I, CJ0 - 1, k) -= (fB[(I - CI0) * nc + k] - cB[(I - CI0) * nc + k] * dt) / dyc;
      c(I, CJ1 + 1, k) += (fT[(I - CI0) * nc + k] - cT[(I - CI0) * nc + k] * dt) / dyc;
    }
}

// --- recursion N-niveaux (pendant MultiFab de amr_multilevel.hpp) ---

// Un niveau de la hierarchie MultiFab. aux detenu ailleurs (pointeur). rC* = region
// (coords de CE niveau) raffinee par l'enfant ; valable si has_fine.
struct AmrLevelMF {
  MultiFab U;
  const MultiFab* aux;
  Real dx, dy;
  int rCI0, rCI1, rCJ0, rCJ1;
  bool has_fine;
};

// Avance recursivement le niveau lev de dt (sous-cyclage Berger-Oliger r=2 + reflux).
// pOld/pNew = etats parent bornant le pas ; preg* = registre du parent (flux fins de
// CE niveau), nul si lev==0. Generique (Limiter, NumericalFlux, N comp), seam GPU.
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void subcycle_level_mf(const Model& m, std::vector<AmrLevelMF>& L, int lev, Real dt,
                       const Box2D& dom, const MultiFab* pOld, const MultiFab* pNew,
                       Real frac, std::vector<Real>* pregL, std::vector<Real>* pregR,
                       std::vector<Real>* pregB, std::vector<Real>* pregT) {
  const int r = 2;
  AmrLevelMF& lv = L[lev];
  const int nc = lv.U.ncomp();

  if (lev == 0)
    fill_boundary(lv.U, dom, Periodicity{true, true});
  else
    mf_fill_fine_ghosts_t(lv.U, *pOld, *pNew, frac);

  MultiFab fx(BoxArray(std::vector<Box2D>{xface_box(lv.U.box(0))}), lv.U.dmap(), nc, 0);
  MultiFab fy(BoxArray(std::vector<Box2D>{yface_box(lv.U.box(0))}), lv.U.dmap(), nc, 0);
  compute_face_fluxes<Limiter, NumericalFlux>(m, lv.U, *lv.aux, fx, fy, lv.dx, lv.dy);

  if (lev > 0) {  // contribution au registre du parent (flux fins x dt)
    device_fence();
    const ConstArray4 FX = fx.fab(0).const_array(), FY = fy.fab(0).const_array();
    const AmrLevelMF& par = L[lev - 1];
    const int pI0 = par.rCI0, pI1 = par.rCI1, pJ0 = par.rCJ0, pJ1 = par.rCJ1;
    for (int J = pJ0; J <= pJ1; ++J)
      for (int k = 0; k < nc; ++k) {
        (*pregL)[(J - pJ0) * nc + k] +=
            Real(0.5) * (FX(2 * pI0, 2 * J, k) + FX(2 * pI0, 2 * J + 1, k)) * dt;
        (*pregR)[(J - pJ0) * nc + k] += Real(0.5) *
            (FX(2 * pI1 + 2, 2 * J, k) + FX(2 * pI1 + 2, 2 * J + 1, k)) * dt;
      }
    for (int I = pI0; I <= pI1; ++I)
      for (int k = 0; k < nc; ++k) {
        (*pregB)[(I - pI0) * nc + k] +=
            Real(0.5) * (FY(2 * I, 2 * pJ0, k) + FY(2 * I + 1, 2 * pJ0, k)) * dt;
        (*pregT)[(I - pI0) * nc + k] += Real(0.5) *
            (FY(2 * I, 2 * pJ1 + 2, k) + FY(2 * I + 1, 2 * pJ1 + 2, k)) * dt;
      }
  }

  if (!lv.has_fine) {
    mf_advance_faces(lv.U, fx, fy, lv.dx, lv.dy, dt);  // feuille
    mf_apply_source(m, lv.U, *lv.aux, dt);  // source S(U,aux) au sous-pas
    return;
  }

  // niveau avec enfant : registre local + flux grossier sauve (sans dt)
  const int cI0 = lv.rCI0, cI1 = lv.rCI1, cJ0 = lv.rCJ0, cJ1 = lv.rCJ1;
  const int nI = cI1 - cI0 + 1, nJ = cJ1 - cJ0 + 1;
  std::vector<Real> cL(nJ * nc), cR(nJ * nc), cB(nI * nc), cT(nI * nc);
  {
    device_fence();
    const ConstArray4 FX = fx.fab(0).const_array(), FY = fy.fab(0).const_array();
    for (int J = cJ0; J <= cJ1; ++J)
      for (int k = 0; k < nc; ++k) {
        cL[(J - cJ0) * nc + k] = FX(cI0, J, k);
        cR[(J - cJ0) * nc + k] = FX(cI1 + 1, J, k);
      }
    for (int I = cI0; I <= cI1; ++I)
      for (int k = 0; k < nc; ++k) {
        cB[(I - cI0) * nc + k] = FY(I, cJ0, k);
        cT[(I - cI0) * nc + k] = FY(I, cJ1 + 1, k);
      }
  }
  std::vector<Real> fL(nJ * nc, 0), fR(nJ * nc, 0), fB(nI * nc, 0), fT(nI * nc, 0);

  MultiFab U_old = lv.U;  // etat t (interp temporelle de l'enfant)
  mf_advance_faces(lv.U, fx, fy, lv.dx, lv.dy, dt);  // lv.U -> t+dt
  mf_apply_source(m, lv.U, *lv.aux, dt);  // source S(U,aux) au sous-pas

  for (int s = 0; s < r; ++s)
    subcycle_level_mf<Limiter, NumericalFlux>(m, L, lev + 1, dt / r, dom, &U_old,
                                              &lv.U, Real(s) / r, &fL, &fR, &fB, &fT);

  mf_average_down(L[lev + 1].U, lv.U, cI0, cI1, cJ0, cJ1);

  device_fence();
  Array4 c = lv.U.fab(0).array();
  for (int J = cJ0; J <= cJ1; ++J)
    for (int k = 0; k < nc; ++k) {
      c(cI0 - 1, J, k) -= (fL[(J - cJ0) * nc + k] - cL[(J - cJ0) * nc + k] * dt) / lv.dx;
      c(cI1 + 1, J, k) += (fR[(J - cJ0) * nc + k] - cR[(J - cJ0) * nc + k] * dt) / lv.dx;
    }
  for (int I = cI0; I <= cI1; ++I)
    for (int k = 0; k < nc; ++k) {
      c(I, cJ0 - 1, k) -= (fB[(I - cI0) * nc + k] - cB[(I - cI0) * nc + k] * dt) / lv.dy;
      c(I, cJ1 + 1, k) += (fT[(I - cI0) * nc + k] - cT[(I - cI0) * nc + k] * dt) / lv.dy;
    }
}

// Driver : un pas dt de la hierarchie complete (niveau 0 = grossier).
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void amr_step_multilevel_mf(const Model& m, std::vector<AmrLevelMF>& L,
                            const Box2D& dom, Real dt) {
  subcycle_level_mf<Limiter, NumericalFlux>(m, L, 0, dt, dom, nullptr, nullptr,
                                            Real(0), nullptr, nullptr, nullptr, nullptr);
}

}  // namespace detail (oracle MF mono-box : Fab2D -> MF -> MP)

// --- MULTI-PATCH (plusieurs boxes fines par niveau) ---
// Le niveau fin est un MultiFab a N boxes. Le reflux est COVERAGE-AWARE : il ne
// corrige une cellule grossiere adjacente a une box fine que si elle n'est PAS
// couverte par une autre box fine (interface fin-grossier reelle ; les interfaces
// fin-fin sont gerees par fill_boundary). C'est la logique FluxRegister d'AMReX.

// PatchRange (revue, point 5 : role promu en type). Empreinte GROSSIERE [I0..I1]x[J0..J1]
// d'un patch fin sous ratio 2 : I0 = lo/2, I1 = (hi-1)/2 (patch aligne, lo pair / hi impair).
// Centralise le calcul d'empreinte repete inline dans average_down, la couverture et l'init
// des registres de reflux. NB : ce n'est PAS Box2D::coarsen (floor des deux bornes) mais la
// borne haute (hi-1)/2 historique ; on conserve l'arithmetique exacte (bit-identique).
struct PatchRange {
  int I0, I1, J0, J1;
  explicit PatchRange(const Box2D& fine)
      : I0(fine.lo[0] / 2), I1((fine.hi[0] - 1) / 2),
        J0(fine.lo[1] / 2), J1((fine.hi[1] - 1) / 2) {}
  Box2D box() const { return Box2D{{I0, J0}, {I1, J1}}; }  // empreinte grossiere (cellules)
};

// ghosts fins multi-box depuis le grossier (interp espace+temps), PUIS fill_boundary
// (fin-fin) ecrasera les ghosts couverts par une box voisine. coarse mono-box.
inline void mf_fill_fine_ghosts_multi(MultiFab& Uf, const MultiFab& Uc_old,
                                      const MultiFab& Uc_new, Real frac) {
  device_fence();
  const int nc = Uf.ncomp();
  const ConstArray4 co = Uc_old.fab(0).const_array();
  const ConstArray4 cn = Uc_new.fab(0).const_array();
  for (int li = 0; li < Uf.local_size(); ++li) {
    Array4 f = Uf.fab(li).array();
    const Box2D v = Uf.box(li), g = Uf.fab(li).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i)
        if (!v.contains(i, j)) fill_cf_ghost_cell(f, co, cn, i, j, nc, frac);
  }
}

// moyenne fin -> grossier sur l'empreinte de CHAQUE box fine (multi-box).
inline void mf_average_down_multi(const MultiFab& Uf, MultiFab& Uc) {
  const int nc = Uc.ncomp();
  Array4 c = Uc.fab(0).array();
  for (int li = 0; li < Uf.local_size(); ++li) {
    const ConstArray4 f = Uf.fab(li).const_array();
    const PatchRange pr(Uf.box(li));
    for_each_cell(pr.box(), [=] ADC_HD(int I, int J) {
      for (int k = 0; k < nc; ++k)
        c(I, J, k) = Real(0.25) * (f(2 * I, 2 * J, k) + f(2 * I + 1, 2 * J, k) +
                                   f(2 * I, 2 * J + 1, k) + f(2 * I + 1, 2 * J + 1, k));
    });
  }
}

// Remplissage periodique PUREMENT LOCAL des ghosts d'un grossier mono-box (auto-repli).
// Equivalent a fill_boundary periodique pour une seule box, mais SANS le plan MPI : sert
// au grossier REPLIQUE (copie par-rang), dont le DistributionMapping par-rang violerait
// l'hypothese de metadonnees repliquees de fill_boundary. Lit des cellules valides (indices
// repliees dans [0,N)) et n'ecrit que des ghosts : pas de course lecture/ecriture.
inline void fill_periodic_local(MultiFab& mf, const Box2D& dom) {
  device_fence();
  const int nc = mf.ncomp(), NX = dom.nx(), NY = dom.ny();
  auto wrap = [](int x, int n) { return (x % n + n) % n; };
  for (int li = 0; li < mf.local_size(); ++li) {
    Array4 a = mf.fab(li).array();
    const Box2D g = mf.fab(li).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i)
        if (i < 0 || i >= NX || j < 0 || j >= NY)
          for (int k = 0; k < nc; ++k) a(i, j, k) = a(wrap(i, NX), wrap(j, NY), k);
  }
}

// FluxRegister (revue, point 2 : role promu en type). Registre grossier a indexation GLOBALE
// sur une REGION (box, avec origine), pour remonter average_down (ecrasement des cellules
// couvertes, set) et reflux (addition aux cellules bordantes, add borne) a travers les rangs.
// Chaque rang remplit ses contributions LOCALES (0 ailleurs), gather() les somme par
// all_reduce_sum_inplace, puis chaque rang lit le total via at(). En serie all_reduce est
// l'identite -> bit a bit identique. Region a origine (0,0) = grille grossiere pleine ; region
// = boite englobante = chemin average_down multi-box. Memes formules d'index qu'avant.
struct FluxRegister {
  int I0, J0, NX, NY, nc;
  std::vector<Real> buf;
  FluxRegister(const Box2D& region, int ncomp)
      : I0(region.lo[0]), J0(region.lo[1]),
        NX(region.hi[0] - region.lo[0] + 1), NY(region.hi[1] - region.lo[1] + 1), nc(ncomp),
        buf(static_cast<std::size_t>(NX) * NY * ncomp, Real(0)) {}
  std::size_t idx(int I, int J, int k) const {
    return (static_cast<std::size_t>(J - J0) * NX + (I - I0)) * nc + k;
  }
  bool in(int I, int J) const { return I >= I0 && I < I0 + NX && J >= J0 && J < J0 + NY; }
  void set(int I, int J, int k, Real v) { buf[idx(I, J, k)] = v; }  // ecrasement (average_down)
  void add(int I, int J, int k, Real v) {                          // addition bordante (reflux)
    if (in(I, J)) buf[idx(I, J, k)] += v;
  }
  Real at(int I, int J, int k) const { return buf[idx(I, J, k)]; }
  void gather() { all_reduce_sum_inplace(buf.data(), static_cast<int>(buf.size())); }
};

// CoverageMask (revue, point 2 : part "couverture" de CoarseFineInterface). Masque grossier
// sur une REGION disant quelles cellules sont OMBRAGEES par un patch fin. Bati sur le
// box_array GLOBAL (connu de tous les rangs) -> MPI-safe. mark(box) marque l'empreinte
// grossiere d'un patch (intersectee a la region) ; covered(I,J) est borne (faux hors region).
// C'est ce qui empeche le double-reflux d'un joint fin-fin. Memes cellules qu'avant.
struct CoverageMask {
  int I0, J0, NX, NY;
  std::vector<char> cov;
  explicit CoverageMask(const Box2D& region)
      : I0(region.lo[0]), J0(region.lo[1]),
        NX(region.hi[0] - region.lo[0] + 1), NY(region.hi[1] - region.lo[1] + 1),
        cov(static_cast<std::size_t>(NX) * NY, 0) {}
  void mark(const Box2D& b) {  // marque les cellules de b intersectees a la region
    const int i0 = std::max(b.lo[0], I0), i1 = std::min(b.hi[0], I0 + NX - 1);
    const int j0 = std::max(b.lo[1], J0), j1 = std::min(b.hi[1], J0 + NY - 1);
    for (int J = j0; J <= j1; ++J)
      for (int I = i0; I <= i1; ++I) cov[(static_cast<std::size_t>(J - J0) * NX) + (I - I0)] = 1;
  }
  bool covered(int I, int J) const {
    if (I < I0 || I >= I0 + NX || J < J0 || J >= J0 + NY) return false;
    return cov[(static_cast<std::size_t>(J - J0) * NX) + (I - I0)] != 0;
  }
};

// SubcyclingSchedule (revue, point 5 : role promu en type). Cadence Berger-Oliger d'un
// niveau : ratio de raffinement temporel r, sous-pas dt/r, et position temporelle frac(s)
// = s/r du sous-pas s dans le pas parent. Centralise le `const int r = 2`, `dt / r` et
// `Real(s) / r` epars dans les boucles de sous-cyclage. Arithmetique strictement preservee :
// dt_sub(dt) == dt / r et frac(s) == Real(s) / r aux memes types, donc bit-identique.
struct SubcyclingSchedule {
  int r;
  explicit SubcyclingSchedule(int ratio = 2) : r(ratio) {}
  int count() const { return r; }                       // nombre de sous-pas
  Real dt_sub(Real dt) const { return dt / r; }         // pas fin = pas parent / r
  Real frac(int s) const { return Real(s) / r; }        // position temporelle du sous-pas s
};

// CoarseFineInterface (revue, point 2). L'interface grossier-fin d'un niveau : couverture
// (quelles cellules grossieres sont ombragees par un patch fin, via CoverageMask) + ROUTAGE
// bordant du reflux (quelle cellule grossiere borde quelle face de patch fin, et la
// correction conservative qu'on y verse). Centralise les deux logiques inline auparavant
// dupliquees dans amr_step_2level_multipatch et subcycle_level_mp. Construit le masque sur le
// box_array() GLOBAL des patchs fins (MPI-safe). route_reflux est un template sur le type de
// registre (Reg / RegMP, meme disposition de champs) : fonction nommee (pas de lambda
// generique), donc sure sous nvcc. Arithmetique bit-identique aux corps precedents.
struct CoarseFineInterface {
  CoverageMask cmask;
  // region = empreinte grossiere du niveau (origine (0,0), dims NX x NY) ; fine_ba = patchs
  // fins GLOBAUX (toutes les boxes, connues de tous les rangs). On marque l'empreinte
  // grossiere PatchRange de chaque patch.
  CoarseFineInterface(const Box2D& coarse_region, const BoxArray& fine_ba)
      : cmask(coarse_region) {
    for (int g = 0; g < fine_ba.size(); ++g) cmask.mark(PatchRange(fine_ba[g]).box());
  }
  bool covered(int I, int J) const { return cmask.covered(I, J); }

  // Verse la correction de reflux d'UN patch fin (registre g, coords parentes) dans le
  // registre grossier ref : sur chaque cellule grossiere BORDANTE non couverte par un autre
  // patch, (flux fin time-integre - flux grossier x dt) / dx|dy. Memes formules, meme ordre
  // (gauche/droite en x, bas/haut en y) que les corps inline d'origine.
  template <class Reg>
  void route_reflux(const Reg& g, Real dx, Real dy, Real dt, FluxRegister& ref, int nc) const {
    for (int J = g.J0; J <= g.J1; ++J)
      for (int k = 0; k < nc; ++k) {
        if (!covered(g.I0 - 1, J))
          ref.add(g.I0 - 1, J, k, -(g.fL[(J - g.J0) * nc + k] - g.cL[(J - g.J0) * nc + k] * dt) / dx);
        if (!covered(g.I1 + 1, J))
          ref.add(g.I1 + 1, J, k, +(g.fR[(J - g.J0) * nc + k] - g.cR[(J - g.J0) * nc + k] * dt) / dx);
      }
    for (int I = g.I0; I <= g.I1; ++I)
      for (int k = 0; k < nc; ++k) {
        if (!covered(I, g.J0 - 1))
          ref.add(I, g.J0 - 1, k, -(g.fB[(I - g.I0) * nc + k] - g.cB[(I - g.I0) * nc + k] * dt) / dy);
        if (!covered(I, g.J1 + 1))
          ref.add(I, g.J1 + 1, k, +(g.fT[(I - g.I0) * nc + k] - g.cT[(I - g.I0) * nc + k] * dt) / dy);
      }
  }
};

// Pas 2-niveaux conservatif, niveau fin MULTI-BOX. Uc : grossier mono-box (periodique).
// Uf : MultiFab a N boxes fines (ratio 2, strictement interieures, alignees grossier).
//
// Distribue (MPI) avec REPLICATION GROSSIERE. Le grossier mono-box est replique : chaque
// rang en detient une copie identique (DistributionMapping par-rang, ou init deterministe).
// L'avance grossiere (fill_boundary self-periodique, flux, advance) tourne identiquement
// sur chaque copie ; les patchs fins sont, eux, repartis. average_down (ecrasement des
// cellules couvertes) et reflux (addition aux cellules bordantes) remontent par deux
// buffers grossiers indexes global + all_reduce_sum_inplace, puis chaque rang applique a sa
// copie -> toutes restent identiques. En serie c'est bit a bit identique au chemin direct
// (voir le bloc final). Validation : test_mpi_amr_multipatch (np=1/2/4 bit-identiques). Le
// grossier est petit (niveau de base), la replication est donc assumee ; le chemin N-niveaux
// recursif (subcycle_level_mp) reste a generaliser de la meme facon (ROADMAP).
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void amr_step_2level_multipatch(const Model& m, MultiFab& Uc, const Box2D& dom, Real dxc,
                                Real dyc, MultiFab& Uf, const MultiFab& auxc,
                                const MultiFab& auxf, Real dt) {
  const SubcyclingSchedule sched(2);
  const int nc = Uc.ncomp();
  const Real dxf = dxc / 2, dyf = dyc / 2, dtf = sched.dt_sub(dt);
  const int NX = dom.nx(), NY = dom.ny();

  // interface grossier-fin : couverture (cellules grossieres ombragees par un patch fin) +
  // routage bordant du reflux. Couverture batie sur le BoxArray GLOBAL (toutes les boxes,
  // connues de tous les rangs) -> correct sous MPI.
  const CoarseFineInterface cfi(Box2D{{0, 0}, {NX - 1, NY - 1}}, Uf.box_array());
  auto covered = [&](int I, int J) { return cfi.covered(I, J); };

  MultiFab Uc_old = Uc;
  fill_periodic_local(Uc, dom);  // grossier replique -> fill periodique local (pas de plan MPI)
  MultiFab fxc(BoxArray(std::vector<Box2D>{xface_box(Uc.box(0))}), Uc.dmap(), nc, 0);
  MultiFab fyc(BoxArray(std::vector<Box2D>{yface_box(Uc.box(0))}), Uc.dmap(), nc, 0);
  compute_face_fluxes<Limiter, NumericalFlux>(m, Uc, auxc, fxc, fyc, dxc, dyc);

  // registre par box fine : flux grossier (sans dt) sauve aux 4 faces.
  struct Reg { int I0, I1, J0, J1; std::vector<Real> cL, cR, cB, cT, fL, fR, fB, fT; };
  std::vector<Reg> regs(Uf.local_size());
  {
    device_fence();
    const ConstArray4 FX = fxc.fab(0).const_array(), FY = fyc.fab(0).const_array();
    for (int li = 0; li < Uf.local_size(); ++li) {
      const PatchRange pr(Uf.box(li));
      Reg& g = regs[li];
      g.I0 = pr.I0; g.I1 = pr.I1;
      g.J0 = pr.J0; g.J1 = pr.J1;
      const int nJ = g.J1 - g.J0 + 1, nI = g.I1 - g.I0 + 1;
      g.cL.assign(nJ * nc, 0); g.cR.assign(nJ * nc, 0);
      g.cB.assign(nI * nc, 0); g.cT.assign(nI * nc, 0);
      g.fL.assign(nJ * nc, 0); g.fR.assign(nJ * nc, 0);
      g.fB.assign(nI * nc, 0); g.fT.assign(nI * nc, 0);
      for (int J = g.J0; J <= g.J1; ++J)
        for (int k = 0; k < nc; ++k) {
          g.cL[(J - g.J0) * nc + k] = FX(g.I0, J, k);
          g.cR[(J - g.J0) * nc + k] = FX(g.I1 + 1, J, k);
        }
      for (int I = g.I0; I <= g.I1; ++I)
        for (int k = 0; k < nc; ++k) {
          g.cB[(I - g.I0) * nc + k] = FY(I, g.J0, k);
          g.cT[(I - g.I0) * nc + k] = FY(I, g.J1 + 1, k);
        }
    }
  }
  mf_advance_faces(Uc, fxc, fyc, dxc, dyc, dt);
  mf_apply_source(m, Uc, auxc, dt);  // source S(U,aux) au sous-pas

  // flux fins multi-box : une face-box par box fine GLOBALE, meme dmap que Uf. Bati sur le
  // box_array() global (et non les boxes locales) pour que BoxArray et DistributionMapping
  // aient la meme taille sous MPI : fxf.fab(li) correspond alors a Uf.fab(li) (meme dmap,
  // meme ordre global). En serie c'est identique (local == global).
  std::vector<Box2D> fxb, fyb;
  for (int g = 0; g < Uf.box_array().size(); ++g) {
    fxb.push_back(xface_box(Uf.box_array()[g]));
    fyb.push_back(yface_box(Uf.box_array()[g]));
  }
  MultiFab fxf(BoxArray(std::move(fxb)), Uf.dmap(), nc, 0);
  MultiFab fyf(BoxArray(std::move(fyb)), Uf.dmap(), nc, 0);
  const Box2D fdom = Box2D::from_extents(2 * NX, 2 * NY);

  for (int s = 0; s < sched.count(); ++s) {
    mf_fill_fine_ghosts_multi(Uf, Uc_old, Uc, sched.frac(s));
    fill_boundary(Uf, fdom, Periodicity{false, false});  // halos fin-fin
    compute_face_fluxes<Limiter, NumericalFlux>(m, Uf, auxf, fxf, fyf, dxf, dyf);
    device_fence();
    for (int li = 0; li < Uf.local_size(); ++li) {
      Reg& g = regs[li];
      const ConstArray4 FX = fxf.fab(li).const_array(), FY = fyf.fab(li).const_array();
      for (int J = g.J0; J <= g.J1; ++J)
        for (int k = 0; k < nc; ++k) {
          g.fL[(J - g.J0) * nc + k] +=
              Real(0.5) * (FX(2 * g.I0, 2 * J, k) + FX(2 * g.I0, 2 * J + 1, k)) * dtf;
          g.fR[(J - g.J0) * nc + k] += Real(0.5) *
              (FX(2 * g.I1 + 2, 2 * J, k) + FX(2 * g.I1 + 2, 2 * J + 1, k)) * dtf;
        }
      for (int I = g.I0; I <= g.I1; ++I)
        for (int k = 0; k < nc; ++k) {
          g.fB[(I - g.I0) * nc + k] +=
              Real(0.5) * (FY(2 * I, 2 * g.J0, k) + FY(2 * I + 1, 2 * g.J0, k)) * dtf;
          g.fT[(I - g.I0) * nc + k] += Real(0.5) *
              (FY(2 * I, 2 * g.J1 + 2, k) + FY(2 * I + 1, 2 * g.J1 + 2, k)) * dtf;
        }
    }
    mf_advance_faces(Uf, fxf, fyf, dxf, dyf, dtf);
    mf_apply_source(m, Uf, auxf, dtf);  // source S(U,aux) au sous-pas
  }

  // average_down + reflux DISTRIBUES, le grossier etant REPLIQUE (chaque rang detient une
  // copie identique apres l'avance grossiere deterministe). Chaque rang verse, pour ses
  // patchs fins LOCAUX, dans deux buffers grossiers indexes global :
  //   avg : la moyenne descendante sur les cellules COUVERTES (semantique ecrasement ;
  //         une seule contribution par cellule, les patchs etant disjoints) ;
  //   ref : la correction de reflux sur les cellules BORDANTES non couvertes (addition).
  // all_reduce_sum -> chaque rang a le total, puis applique a SA copie : couvert = avg,
  // bordant += ref. Toutes les copies restent identiques. En serie (np=1) l'all-reduce est
  // l'identite et c'est bit a bit identique au direct (0 + moyenne = moyenne exactement ;
  // avance + correction). Cout : deux buffers NX*NY*nc par rang (replication grossiere).
  device_fence();
  // registre restreint a l'INTERFACE coarse-fine (boite englobante des empreintes fines,
  // crue de 1 pour les cellules bordantes du reflux, clampee au domaine) : le all_reduce du
  // gather passe de O(NX*NY) a O(interface). Bit-identique : les cellules hors interface
  // etaient nulles (non couvertes, sans face), sautees a l'application.
  const Box2D fpc = coarsen(Uf.box_array(), 2).bounding_box();
  const Box2D rbox{{std::max(fpc.lo[0] - 1, 0), std::max(fpc.lo[1] - 1, 0)},
                   {std::min(fpc.hi[0] + 1, NX - 1), std::min(fpc.hi[1] + 1, NY - 1)}};
  FluxRegister avg(rbox, nc);  // moyenne descendante (ecrasement des cellules couvertes)
  FluxRegister ref(rbox, nc);  // reflux (addition aux cellules bordantes)
  for (int li = 0; li < Uf.local_size(); ++li) {
    const ConstArray4 f = Uf.fab(li).const_array();
    Reg& g = regs[li];
    for (int J = g.J0; J <= g.J1; ++J)
      for (int I = g.I0; I <= g.I1; ++I)
        for (int k = 0; k < nc; ++k)
          avg.set(I, J, k, Real(0.25) * (f(2 * I, 2 * J, k) + f(2 * I + 1, 2 * J, k) +
                                         f(2 * I, 2 * J + 1, k) + f(2 * I + 1, 2 * J + 1, k)));
    cfi.route_reflux(g, dxc, dyc, dt, ref, nc);  // reflux bordant coverage-aware
  }
  avg.gather();
  ref.gather();
  if (Uc.local_size() > 0) {  // chaque rang detenant une copie du grossier l'applique
    Array4 c = Uc.fab(0).array();
    const Box2D cb = Uc.box(0);
    for (int J = cb.lo[1]; J <= cb.hi[1]; ++J)
      for (int I = cb.lo[0]; I <= cb.hi[0]; ++I) {
        if (!ref.in(I, J)) continue;  // hors interface : ni moyenne ni reflux (etait 0)
        for (int k = 0; k < nc; ++k) {
          if (covered(I, J)) c(I, J, k) = avg.at(I, J, k);  // moyenne descendante
          c(I, J, k) += ref.at(I, J, k);                    // reflux (0 si pas de face)
        }
      }
  }
}

// --- MULTI-PATCH N-NIVEAUX (multi-box a CHAQUE niveau) ---
// Generalise subcycle_level_mf : chaque niveau est un MultiFab multi-box. Le reflux
// (registre FluxRegister) est coverage-aware ET route la correction vers la box PARENTE
// contenant la cellule grossiere adjacente. Se reduit BIT A BIT au chemin mono-box quand
// chaque niveau n'a qu'une box (garde de validation).
//
// Etat distribue (MPI) : DISTRIBUE et teste bit a bit identique np=1/2/4
// (test_mpi_amr_multipatch3, 3 niveaux avec un niveau intermediaire multi-box reparti dont le
// PARENT d'un patch fin tombe sur un autre rang). Le niveau 0 (grossier) est REPLIQUE comme au
// 2-niveaux ; les niveaux >0 sont repartis et jouent simultanement le role d'enfant et de
// parent. Les cinq points supposant le parent local (via mf_find_box) sont resolus :
//   1. mf_fill_fine_ghosts_mb : parent REPLIQUE (lev==1) lu localement ; parent REPARTI
//      (lev>=2) amene par parallel_copy (parent -> fine-coarsen) puis interpole ;
//   2. echantillonnage du registre grossier : parent REPLIQUE lu localement, parent REPARTI
//      amene par parallel_copy vers une grille FACE enfant-coarsen ;
//   3. mf_average_down_mb : moyenne versee dans un buffer grossier indexe GLOBAL +
//      all_reduce_sum, appliquee aux boxes parentes locales (replique : tous ; reparti : le
//      proprietaire) ;
//   4. reflux : meme buffer global + all_reduce, application gardee par appartenance locale
//      de la box parente (pas de double comptage car le parent reparti a un seul proprietaire);
//   5. couverture : deja batie sur le box_array() global (MPI-safe).
// En serie all_reduce est l'identite et parallel_copy se reduit a des copies memoire : le
// chemin distribue execute les memes operations flottantes que le mono-rang -> bit-identique.
// Note : AmrCouplerMP reste limite au mono-rang au-dela de 2 niveaux distribues, car son
// injection d'aux parent->enfant (inject_aux_mb) suppose encore le parent local via
// mf_find_box ; l'integrateur amr_step_multilevel_multipatch, lui, est distribue.

// box LOCALE (valide) contenant la cellule (I,J), ou -1.
inline int mf_find_box(const MultiFab& mf, int I, int J) {
  for (int li = 0; li < mf.local_size(); ++li)
    if (mf.box(li).contains(I, J)) return li;
  return -1;
}

// BoxArray des boites enfant GROSSIES de ngrow puis coarsenees (ratio 2). Chaque box
// couvre toutes les cellules grossieres dont l'enfant a besoin, ghosts compris : c'est la
// grille fine-coarsen du FillPatch (cf. refinement.hpp::interpolate).
inline BoxArray coarsen_grown(const BoxArray& ba, int ngrow, int r) {
  std::vector<Box2D> b;
  b.reserve(ba.size());
  for (int i = 0; i < ba.size(); ++i) b.push_back(ba[i].grow(ngrow).coarsen(r));
  return BoxArray{std::move(b)};
}

// ghosts fins multi-box depuis un parent MULTI-BOX (interp espace constant + temps lin),
// DISTRIBUE. Deux cas de parent :
//  - REPLIQUE (niveau 0, replicated_parent=true) : le parent est entierement local sur chaque
//    rang, on lit directement via mf_find_box (toujours trouve) ; aucune collective. C'est le
//    chemin du grossier replique, comme le 2-niveaux (parallel_copy violerait l'hypothese de
//    metadonnees repliquees du parent, dmap par-rang).
//  - REPARTI (intermediaire) : le parent peut etre sur un autre rang ; on amene ses regions
//    valides sur une grille enfant-coarsen LOCALE par parallel_copy (routage MPI gere la), puis
//    on interpole. Plus aucun echec silencieux remote.
// En serie les deux chemins sont identiques (parent local partout, parallel_copy = copie memoire).
inline void mf_fill_fine_ghosts_mb(MultiFab& Uf, const MultiFab& Po, const MultiFab& Pn,
                                   Real frac, bool replicated_parent = true) {
  const int nc = Uf.ncomp(), ng = Uf.n_grow();
  if (replicated_parent) {
    device_fence();
    for (int li = 0; li < Uf.local_size(); ++li) {
      Array4 f = Uf.fab(li).array();
      const Box2D v = Uf.box(li), g = Uf.fab(li).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i)
          if (!v.contains(i, j)) {
            const int ci = coarsen_index(i, 2), cj = coarsen_index(j, 2);
            const int pb = mf_find_box(Po, ci, cj);
            if (pb < 0) continue;  // hors couverture parente -> laisse au fill_boundary
            const ConstArray4 po = Po.fab(pb).const_array(), pn = Pn.fab(pb).const_array();
            fill_cf_ghost_cell(f, po, pn, i, j, nc, frac);
          }
    }
    return;
  }
  const BoxArray pcoarse_ba = coarsen_grown(Uf.box_array(), ng, 2);
  MultiFab Pco(pcoarse_ba, Uf.dmap(), nc, 0), Pcn(pcoarse_ba, Uf.dmap(), nc, 0);
  parallel_copy(Pco, Po);  // regions parentes (depuis n'importe quel rang) -> grille locale
  parallel_copy(Pcn, Pn);
  device_fence();
  for (int li = 0; li < Uf.local_size(); ++li) {
    Array4 f = Uf.fab(li).array();
    const ConstArray4 po = Pco.fab(li).const_array(), pn = Pcn.fab(li).const_array();
    const Box2D v = Uf.box(li), g = Uf.fab(li).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i)
        if (!v.contains(i, j)) fill_cf_ghost_cell(f, po, pn, i, j, nc, frac);
  }
}

// moyenne fin multi-box -> parent multi-box (chaque cellule routee vers sa box parente),
// DISTRIBUE. La box parente d'une cellule grossiere peut etre sur un autre rang, et le parent
// peut etre soit REPLIQUE (niveau 0, chaque rang en a une copie), soit REPARTI (intermediaire,
// un seul proprietaire). Les deux sont couverts par un buffer grossier indexe GLOBAL :
// chaque rang verse la moyenne 2x2 de SES patchs fins locaux (0 ailleurs ; patchs disjoints
// donc une seule contribution par cellule couverte), all_reduce_sum -> chaque rang a le total,
// puis applique a SES boxes parentes locales (ecrasement). Replique : tous appliquent la meme
// valeur a leur copie. Reparti : seul le proprietaire applique. En serie all_reduce est
// l'identite (0 + moyenne = moyenne) -> bit a bit identique au routage direct.
inline void mf_average_down_mb(const MultiFab& Uf, MultiFab& Uc) {
  const int nc = std::min(Uf.ncomp(), Uc.ncomp());
  // boite englobante grossiere (indices GLOBAUX) couvrant toutes les empreintes enfant.
  const BoxArray cba = coarsen(Uf.box_array(), 2);
  Box2D bb{{0, 0}, {-1, -1}};
  for (int g = 0; g < cba.size(); ++g)
    bb = (g == 0) ? cba[g] : Box2D{{std::min(bb.lo[0], cba[g].lo[0]), std::min(bb.lo[1], cba[g].lo[1])},
                                   {std::max(bb.hi[0], cba[g].hi[0]), std::max(bb.hi[1], cba[g].hi[1])}};
  if (bb.empty()) { all_reduce_sum_inplace(nullptr, 0); return; }  // collective appariee a vide
  FluxRegister avg(bb, nc);  // moyenne descendante multi-box (region = boite englobante)
  // couverture GLOBALE (toutes les empreintes enfant) : seules ces cellules sont ecrasees ;
  // la boite englobante peut contenir des trous entre patchs disjoints qu'il ne faut PAS ecraser.
  CoverageMask cmask(bb);
  for (int g = 0; g < cba.size(); ++g) cmask.mark(cba[g]);
  auto covered = [&](int I, int J) { return cmask.covered(I, J); };
  device_fence();
  for (int lf = 0; lf < Uf.local_size(); ++lf) {
    const ConstArray4 f = Uf.fab(lf).const_array();
    const PatchRange pr(Uf.box(lf));
    for (int J = pr.J0; J <= pr.J1; ++J)
      for (int I = pr.I0; I <= pr.I1; ++I)
        for (int k = 0; k < nc; ++k)
          avg.set(I, J, k, Real(0.25) * (f(2 * I, 2 * J, k) + f(2 * I + 1, 2 * J, k) +
                                         f(2 * I, 2 * J + 1, k) + f(2 * I + 1, 2 * J + 1, k)));
  }
  avg.gather();
  for (int pb = 0; pb < Uc.local_size(); ++pb) {
    Array4 c = Uc.fab(pb).array();
    const Box2D inter = Uc.box(pb).intersect(bb);
    for (int J = inter.lo[1]; J <= inter.hi[1]; ++J)
      for (int I = inter.lo[0]; I <= inter.hi[0]; ++I)
        if (covered(I, J))
          for (int k = 0; k < nc; ++k) c(I, J, k) = avg.at(I, J, k);
  }
}

// un niveau de la hierarchie multi-patch (U + aux multi-box, meme BoxArray).
struct AmrLevelMP {
  MultiFab U;
  const MultiFab* aux;
  Real dx, dy;
};

// registre par patch enfant (coords PARENTES I0..J1). c* = flux grossier (sans dt) ;
// f* = flux fin time-integre accumule par l'enfant durant le sous-cyclage.
struct RegMP {
  int I0, I1, J0, J1;
  std::vector<Real> cL, cR, cB, cT, fL, fR, fB, fT;
};

namespace detail {  // moteur N-niveaux multi-patch INTERNE ; la facade publique est advance_amr

template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void subcycle_level_mp(const Model& m, std::vector<AmrLevelMP>& L, int lev, Real dt,
                       const Box2D& base_dom, Periodicity base_per, const MultiFab* pOld,
                       const MultiFab* pNew, Real frac, std::vector<RegMP>* parentRegs,
                       bool coarse_replicated = true, bool recon_prim = false) {
  const SubcyclingSchedule sched(2);
  const int nc = L[lev].U.ncomp();
  AmrLevelMP& lv = L[lev];
  const int np = lv.U.local_size();

  if (lev == 0) {
    fill_boundary(lv.U, base_dom, base_per);
  } else {
    // parent (niveau lev-1) REPLIQUE seulement s'il s'agit du niveau 0 (lev == 1) ; sinon
    // reparti -> FillPatch par parallel_copy.
    mf_fill_fine_ghosts_mb(lv.U, *pOld, *pNew, frac,
                           /*replicated_parent=*/(lev == 1) && coarse_replicated);
    const Box2D fdom = Box2D::from_extents(base_dom.nx() << lev, base_dom.ny() << lev);
    fill_boundary(lv.U, fdom, Periodicity{false, false});  // halos fin-fin
  }

  // face-box par box GLOBALE + meme dmap (cf. amr_step_2level_multipatch) : BoxArray et
  // DistributionMapping de meme taille sous MPI, fx.fab(li) <-> lv.U.fab(li). Identique en
  // serie (local == global).
  std::vector<Box2D> fxb, fyb;
  for (int g = 0; g < lv.U.box_array().size(); ++g) {
    fxb.push_back(xface_box(lv.U.box_array()[g]));
    fyb.push_back(yface_box(lv.U.box_array()[g]));
  }
  MultiFab fx(BoxArray(std::move(fxb)), lv.U.dmap(), nc, 0);
  MultiFab fy(BoxArray(std::move(fyb)), lv.U.dmap(), nc, 0);
  compute_face_fluxes<Limiter, NumericalFlux>(m, lv.U, *lv.aux, fx, fy, lv.dx, lv.dy, recon_prim);
  device_fence();

  if (parentRegs) {  // role FIN : flux fins de CE niveau dans le registre du parent
    for (int li = 0; li < np; ++li) {
      RegMP& g = (*parentRegs)[li];
      const ConstArray4 FX = fx.fab(li).const_array(), FY = fy.fab(li).const_array();
      for (int J = g.J0; J <= g.J1; ++J)
        for (int k = 0; k < nc; ++k) {
          g.fL[(J - g.J0) * nc + k] +=
              Real(0.5) * (FX(2 * g.I0, 2 * J, k) + FX(2 * g.I0, 2 * J + 1, k)) * dt;
          g.fR[(J - g.J0) * nc + k] += Real(0.5) *
              (FX(2 * g.I1 + 2, 2 * J, k) + FX(2 * g.I1 + 2, 2 * J + 1, k)) * dt;
        }
      for (int I = g.I0; I <= g.I1; ++I)
        for (int k = 0; k < nc; ++k) {
          g.fB[(I - g.I0) * nc + k] +=
              Real(0.5) * (FY(2 * I, 2 * g.J0, k) + FY(2 * I + 1, 2 * g.J0, k)) * dt;
          g.fT[(I - g.I0) * nc + k] += Real(0.5) *
              (FY(2 * I, 2 * g.J1 + 2, k) + FY(2 * I + 1, 2 * g.J1 + 2, k)) * dt;
        }
    }
  }

  if (lev + 1 >= static_cast<int>(L.size())) {  // feuille
    mf_advance_faces(lv.U, fx, fy, lv.dx, lv.dy, dt);
    mf_apply_source(m, lv.U, *lv.aux, dt);  // source S(U,aux) au sous-pas
    return;
  }

  // role GROSSIER pour lev+1 : interface grossier-fin (couverture GLOBALE MPI-safe + routage
  // bordant du reflux) + registres + flux grossier sauve.
  const int NX = base_dom.nx() << lev, NY = base_dom.ny() << lev;
  const CoarseFineInterface cfi(Box2D{{0, 0}, {NX - 1, NY - 1}}, L[lev + 1].U.box_array());
  auto covered = [&](int I, int J) { return cfi.covered(I, J); };

  // Point 2 distribue : le flux grossier fx/fy vit sur la dmap du PARENT (lv.U), donc fx.fab
  // est sur le rang proprietaire de la box parente, pas forcement celui de l'enfant. Deux cas :
  //  - parent REPLIQUE (lev == 0) : fx/fy entierement locaux, on echantillonne directement via
  //    mf_find_box (toujours trouve) ; aucune collective (parallel_copy violerait la
  //    replication du parent) ;
  //  - parent REPARTI (lev >= 1) : on amene les flux grossiers necessaires sur une grille FACE
  //    enfant-coarsen (dmap de l'enfant) par parallel_copy, chaque enfant lit alors localement.
  // Le niveau 0 n'est REPLIQUE que si coarse_replicated : a la de-replication (grossier multi-box
  // reparti), il devient REPARTI comme les niveaux fins, et mf_find_box(lv.U, I, J) renverrait -1
  // pour une cellule grossiere bordante possedee par un rang DISTANT (-> fab(-1), segfault). On
  // route alors vers le chemin parallel_copy (empreinte grossiere par enfant), MPI-correct.
  const bool replicated_parent = (lev == 0) && coarse_replicated;
  const BoxArray cba = coarsen(L[lev + 1].U.box_array(), 2);  // empreinte grossiere par enfant
  MultiFab cfx, cfy;
  if (!replicated_parent) {
    std::vector<Box2D> cfxb, cfyb;
    for (int g = 0; g < cba.size(); ++g) {
      cfxb.push_back(xface_box(cba[g]));
      cfyb.push_back(yface_box(cba[g]));
    }
    cfx = MultiFab(BoxArray(std::move(cfxb)), L[lev + 1].U.dmap(), nc, 0);
    cfy = MultiFab(BoxArray(std::move(cfyb)), L[lev + 1].U.dmap(), nc, 0);
    parallel_copy(cfx, fx);  // flux grossier aux faces -> grille enfant-coarsen locale
    parallel_copy(cfy, fy);
  }
  device_fence();

  std::vector<RegMP> regs(L[lev + 1].U.local_size());
  for (int lc = 0; lc < L[lev + 1].U.local_size(); ++lc) {
    const PatchRange pr(L[lev + 1].U.box(lc));
    RegMP& g = regs[lc];
    g.I0 = pr.I0; g.I1 = pr.I1;
    g.J0 = pr.J0; g.J1 = pr.J1;
    const int nJ = g.J1 - g.J0 + 1, nI = g.I1 - g.I0 + 1;
    g.cL.assign(nJ * nc, 0); g.cR.assign(nJ * nc, 0);
    g.cB.assign(nI * nc, 0); g.cT.assign(nI * nc, 0);
    g.fL.assign(nJ * nc, 0); g.fR.assign(nJ * nc, 0);
    g.fB.assign(nI * nc, 0); g.fT.assign(nI * nc, 0);
    if (replicated_parent) {
      for (int J = g.J0; J <= g.J1; ++J) {
        const int bL = mf_find_box(lv.U, g.I0, J), bR = mf_find_box(lv.U, g.I1, J);
        const ConstArray4 FXL = fx.fab(bL).const_array(), FXR = fx.fab(bR).const_array();
        for (int k = 0; k < nc; ++k) {
          g.cL[(J - g.J0) * nc + k] = FXL(g.I0, J, k);
          g.cR[(J - g.J0) * nc + k] = FXR(g.I1 + 1, J, k);
        }
      }
      for (int I = g.I0; I <= g.I1; ++I) {
        const int bB = mf_find_box(lv.U, I, g.J0), bT = mf_find_box(lv.U, I, g.J1);
        const ConstArray4 FYB = fy.fab(bB).const_array(), FYT = fy.fab(bT).const_array();
        for (int k = 0; k < nc; ++k) {
          g.cB[(I - g.I0) * nc + k] = FYB(I, g.J0, k);
          g.cT[(I - g.I0) * nc + k] = FYT(I, g.J1 + 1, k);
        }
      }
    } else {
      const ConstArray4 FX = cfx.fab(lc).const_array(), FY = cfy.fab(lc).const_array();
      for (int J = g.J0; J <= g.J1; ++J)
        for (int k = 0; k < nc; ++k) {
          g.cL[(J - g.J0) * nc + k] = FX(g.I0, J, k);
          g.cR[(J - g.J0) * nc + k] = FX(g.I1 + 1, J, k);
        }
      for (int I = g.I0; I <= g.I1; ++I)
        for (int k = 0; k < nc; ++k) {
          g.cB[(I - g.I0) * nc + k] = FY(I, g.J0, k);
          g.cT[(I - g.I0) * nc + k] = FY(I, g.J1 + 1, k);
        }
    }
  }

  MultiFab U_old = lv.U;  // etat t (interp temporelle pour les enfants)
  mf_advance_faces(lv.U, fx, fy, lv.dx, lv.dy, dt);
  mf_apply_source(m, lv.U, *lv.aux, dt);  // source S(U,aux) au sous-pas
  for (int s = 0; s < sched.count(); ++s)
    subcycle_level_mp<Limiter, NumericalFlux>(m, L, lev + 1, sched.dt_sub(dt), base_dom, base_per,
                                              &U_old, &lv.U, sched.frac(s), &regs,
                                              coarse_replicated, recon_prim);
  mf_average_down_mb(L[lev + 1].U, lv.U);  // point 3 distribue (parallel_copy)

  // Point 4 distribue : reflux coverage-aware. La cellule grossiere bordante peut appartenir
  // a une box parente REMOTE. On verse, pour chaque enfant LOCAL, la correction (cL/fL deja
  // locaux apres parallel_copy) dans un buffer grossier indexe GLOBAL, all_reduce -> chaque
  // rang a le registre complet, puis chaque rang applique a SES boxes parentes locales (le
  // parent etant reparti, chaque cellule n'a qu'un proprietaire : pas de double comptage).
  // En serie all_reduce est l'identite et l'application directe -> bit a bit identique.
  device_fence();
  // registre restreint a l'INTERFACE : boite englobante des empreintes fines (coarsen du
  // niveau lev+1), crue de 1, clampee au niveau lev. all_reduce O(interface), bit-identique.
  const Box2D fpcn = coarsen(L[lev + 1].U.box_array(), 2).bounding_box();
  const Box2D rbox{{std::max(fpcn.lo[0] - 1, 0), std::max(fpcn.lo[1] - 1, 0)},
                   {std::min(fpcn.hi[0] + 1, NX - 1), std::min(fpcn.hi[1] + 1, NY - 1)}};
  FluxRegister ref(rbox, nc);  // reflux N-niveaux (interface)
  for (int lc = 0; lc < static_cast<int>(regs.size()); ++lc)
    cfi.route_reflux(regs[lc], lv.dx, lv.dy, dt, ref, nc);  // reflux bordant coverage-aware
  ref.gather();
  for (int pb = 0; pb < lv.U.local_size(); ++pb) {  // application aux boxes parentes locales
    Array4 c = lv.U.fab(pb).array();
    const Box2D pbx = lv.U.box(pb);
    for (int J = pbx.lo[1]; J <= pbx.hi[1]; ++J)
      for (int I = pbx.lo[0]; I <= pbx.hi[0]; ++I) {
        if (!ref.in(I, J)) continue;  // hors interface : reflux nul
        for (int k = 0; k < nc; ++k) c(I, J, k) += ref.at(I, J, k);
      }
  }
}

// Driver : un pas dt de la hierarchie multi-patch N-niveaux (niveau 0 = grossier).
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void amr_step_multilevel_multipatch(const Model& m, std::vector<AmrLevelMP>& L,
                                    const Box2D& dom, Real dt,
                                    Periodicity per = Periodicity{true, true},
                                    bool coarse_replicated = true, bool recon_prim = false) {
  subcycle_level_mp<Limiter, NumericalFlux>(m, L, 0, dt, dom, per, nullptr, nullptr,
                                            Real(0), nullptr, coarse_replicated, recon_prim);
}

}  // namespace detail (moteur N-niveaux multi-patch)

// --- Moteur AMR unifie (revue, point 5) ---
// La hierarchie AMR comme OBJET nomme que le moteur fait avancer, plutot qu'une famille de
// fonctions amr_step_* dont le cas (2/N niveaux, mono/multi-box) est encode dans le NOM.
// Entree unifiee : advance_amr(m, LevelHierarchy&, dt), facade fidele du moteur N-niveaux
// multi-patch (verifie 2 ET 3 niveaux, maxdiff = 0, par test_advance_amr). Les ROLES de la
// revue, leurs supports actuels (types nommes, ou code restant a promouvoir) :
//   OwnershipPolicy     = DistributionMapping (qui possede quel patch)        -> alias ci-dessous
//   AmrLevel            = AmrLevelMP (box + donnees + aux + dx d'un niveau)
//   PatchRange          = TYPE NOMME : empreinte grossiere [I0..I1]x[J0..J1] d'un patch fin
//                         (ratio 2), partagee par average_down, couverture et registres
//   CoarseFineGhost     = TYPE/HELPER NOMME : fill_cf_ghost_cell (interp espace+temps par
//                         cellule ghost), partage par les trois mf_fill_fine_ghosts_*
//   CoarseFineInterface = TYPE NOMME : couverture (CoverageMask) + routage bordant du reflux
//                         (route_reflux), partage par subcycle_level_mp et amr_step_2level_multipatch
//   FluxRegister        = TYPE NOMME : buffers avg/ref a index global + all_reduce_sum_inplace
//   SubcyclingSchedule  = TYPE NOMME : cadence Berger-Oliger (ratio r, dt/r, frac s/r) par niveau
//   RegridPolicy        = amr_regrid_finest (Berger-Rigoutsos), cote coupleur
using OwnershipPolicy = DistributionMapping;

struct LevelHierarchy {
  std::vector<AmrLevelMP> levels;    // niveau 0 = grossier, niveaux >0 = patchs fins
  Box2D base_dom;                    // empreinte du niveau de base
  Periodicity base_per{true, true};  // CL du domaine de base
  bool coarse_replicated = true;     // niveau 0 replique (true) ou multi-box reparti (false)
  bool recon_prim = false;           // reconstruction primitive (cf. compute_face_fluxes)
};

// Entree unifiee de production : avance la hierarchie d'un pas dt. Forme "pieces" (le coupleur
// possede son propre stack et passe les vecteurs directement) et forme LevelHierarchy. La porte
// TRANSMET coarse_replicated au moteur ; sans cela un grossier de-replique repasserait en
// replique (mf_find_box au lieu de parallel_copy). coarse_replicated=true (defaut) -> identique
// au comportement historique. recon_prim selectionne la reconstruction primitive (variables
// (rho, u, p)) au lieu de conservative : meme parametre qu'assemble_rhs, fige a l'ajout du bloc ;
// false (defaut) -> conservative, strictement bit-identique a l'historique.
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void advance_amr(const Model& m, std::vector<AmrLevelMP>& levels, const Box2D& base_dom, Real dt,
                 Periodicity base_per = Periodicity{true, true}, bool coarse_replicated = true,
                 bool recon_prim = false) {
  detail::amr_step_multilevel_multipatch<Limiter, NumericalFlux>(m, levels, base_dom, dt, base_per,
                                                                 coarse_replicated, recon_prim);
}

template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void advance_amr(const Model& m, LevelHierarchy& h, Real dt) {
  advance_amr<Limiter, NumericalFlux>(m, h.levels, h.base_dom, dt, h.base_per, h.coarse_replicated,
                                      h.recon_prim);
}

}  // namespace adc
