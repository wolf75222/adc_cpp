#pragma once
// Moteur de sous-cyclage multi-patch N-niveaux : amr_step_2level_multipatch,
// subcycle_level_mp, amr_step_multilevel_multipatch, AmrLevelMP, RegMP.

#include <adc/mesh/mf_arith.hpp>  // saxpy, lincomb (etages SSPRK3, foncteurs nommes device-clean)
#include <adc/mesh/refinement.hpp>  // coarsen, parallel_copy
#include <adc/numerics/time/amr_flux_helpers.hpp>
#include <adc/numerics/time/amr_patch_range.hpp>

namespace adc {

// --- MULTI-PATCH (plusieurs boxes fines par niveau) ---
// Le niveau fin est un MultiFab a N boxes. Le reflux est COVERAGE-AWARE : il ne
// corrige une cellule grossiere adjacente a une box fine que si elle n'est PAS
// couverte par une autre box fine (interface fin-grossier reelle ; les interfaces
// fin-fin sont gerees par fill_boundary). C'est la logique FluxRegister d'AMReX.

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

// Remplit les ghosts d'un niveau AMR : niveau 0 = CL du domaine de base (fill_boundary) ; niveau
// > 0 = ghosts grossier-fin temps-interpoles depuis le parent a la position frac (mf_fill_fine_ghosts_mb)
// PUIS halos fin-fin (fill_boundary). Factorise de la tete de subcycle_level_mp, REUTILISE par
// l'avance SSPRK3 qui doit reremplir les ghosts AVANT chaque evaluation de flux d'etage. Le parent
// n'est REPLIQUE que pour lev == 1 (niveau 0 replique), sinon reparti (parallel_copy interne).
inline void ssprk3_refill_level_ghosts(MultiFab& U, int lev, const Box2D& base_dom,
                                       Periodicity base_per, const MultiFab* pOld,
                                       const MultiFab* pNew, Real frac, bool coarse_replicated) {
  if (lev == 0) {
    fill_boundary(U, base_dom, base_per);
  } else {
    mf_fill_fine_ghosts_mb(U, *pOld, *pNew, frac, (lev == 1) && coarse_replicated);
    const Box2D fdom = Box2D::from_extents(base_dom.nx() << lev, base_dom.ny() << lev);
    fill_boundary(U, fdom, Periodicity{false, false});
  }
}

// SSPRK3 (Shu-Osher, 3 etages, ordre 3) a UN niveau AMR. (1) Avance lv.U de t a t+dt :
//   U1 = U0 + dt L(U0) ; U2 = 3/4 U0 + 1/4 (U1 + dt L(U1)) ; U_new = 1/3 U0 + 2/3 (U2 + dt L(U2))
// avec L(U) = -div F(U) + S(U) (source EXPLICITE par etage, evaluee au meme etat que le flux : vrai
// SSPRK methode-des-lignes, cf. mf_eval_rhs -- l'IMEX n'est PAS supporte, rejet en amont). (2) Remplit
// (fx, fy) du FLUX EFFECTIF du pas    Feff = 1/6 F(U0) + 1/6 F(U1) + 2/3 F(U2)    qui est EXACTEMENT
// le flux de transport vu par l'etat final (U_new = U0 - dt div Feff + dt Seff). C'est ce flux que le
// reflux conservatif doit enregistrer (cote grossier g.c* et cote fin g.f*), d'ou son ecriture dans
// (fx, fy) la ou le chemin Euler y laisse l'unique flux F(U0). En ENTREE (fx, fy) contiennent deja
// F(U0) (etage 0, calcule par l'appelant avant l'appel). Entre etages, les ghosts sont remis a jour
// par ssprk3_refill_level_ghosts au MEME frac : le bord grossier-fin est GELE sur le sous-pas (les
// niveaux ne croisent pas leurs etages, cf. en-tete subcycle_level_mp / sous-cyclage). saxpy/lincomb
// et le foncteur RHS sont des kernels device-clean (foncteurs nommes), pas de lambda etendue.
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void ssprk3_advance_level(const Model& m, AmrLevelMP& lv, Real dt, MultiFab& fx, MultiFab& fy,
                          bool recon_prim, int lev, const Box2D& base_dom, Periodicity base_per,
                          const MultiFab* pOld, const MultiFab* pNew, Real frac,
                          bool coarse_replicated) {
  const int nc = lv.U.ncomp();
  MultiFab U0 = lv.U;  // etat de depart t (combinaisons convexes de Shu-Osher)
  MultiFab R(lv.U.box_array(), lv.U.dmap(), nc, 0);
  MultiFab Fxs(fx.box_array(), fx.dmap(), nc, 0), Fys(fy.box_array(), fy.dmap(), nc, 0);  // flux d'etage

  // --- etage 0 : F(U0) deja dans (fx, fy), R0 = -div F0 + S(U0) ---
  mf_eval_rhs(m, lv.U, *lv.aux, fx, fy, lv.dx, lv.dy, R);
  saxpy(lv.U, dt, R);                          // lv.U = U1 = U0 + dt R0
  lincomb(fx, Real(1) / 6, fx, Real(0), fx);   // Feff <- 1/6 F0 (aliasing point a point, sans danger)
  lincomb(fy, Real(1) / 6, fy, Real(0), fy);

  // --- etage 1 : F(U1) ---
  ssprk3_refill_level_ghosts(lv.U, lev, base_dom, base_per, pOld, pNew, frac, coarse_replicated);
  compute_face_fluxes<Limiter, NumericalFlux>(m, lv.U, *lv.aux, Fxs, Fys, lv.dx, lv.dy, recon_prim);
  device_fence();
  mf_eval_rhs(m, lv.U, *lv.aux, Fxs, Fys, lv.dx, lv.dy, R);  // R1 = -div F1 + S(U1)
  saxpy(lv.U, dt, R);                                         // lv.U = U1 + dt R1
  lincomb(lv.U, Real(3) / 4, U0, Real(1) / 4, lv.U);          // lv.U = U2
  saxpy(fx, Real(1) / 6, Fxs);                                // Feff += 1/6 F1
  saxpy(fy, Real(1) / 6, Fys);

  // --- etage 2 : F(U2) ---
  ssprk3_refill_level_ghosts(lv.U, lev, base_dom, base_per, pOld, pNew, frac, coarse_replicated);
  compute_face_fluxes<Limiter, NumericalFlux>(m, lv.U, *lv.aux, Fxs, Fys, lv.dx, lv.dy, recon_prim);
  device_fence();
  mf_eval_rhs(m, lv.U, *lv.aux, Fxs, Fys, lv.dx, lv.dy, R);  // R2 = -div F2 + S(U2)
  saxpy(lv.U, dt, R);                                         // lv.U = U2 + dt R2
  lincomb(lv.U, Real(1) / 3, U0, Real(2) / 3, lv.U);          // lv.U = U_new (t + dt)
  saxpy(fx, Real(2) / 3, Fxs);                                // Feff += 2/3 F2
  saxpy(fy, Real(2) / 3, Fys);
  device_fence();  // (fx, fy) = Feff et lv.U = U_new coherents pour les lectures hote (parentRegs/reflux)
}

template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void subcycle_level_mp(const Model& m, std::vector<AmrLevelMP>& L, int lev, Real dt,
                       const Box2D& base_dom, Periodicity base_per, const MultiFab* pOld,
                       const MultiFab* pNew, Real frac, std::vector<RegMP>* parentRegs,
                       bool coarse_replicated = true, bool recon_prim = false,
                       bool imex = false, const NewtonOptions& nopts = {},
                       AmrTimeMethod tmethod = AmrTimeMethod::kEuler) {
  // SSPRK3 + IMEX : combinaison NON VALIDEE (la source raide implicite par etage SSP n'a pas ete
  // verifiee), rejetee EXPLICITEMENT plutot que jouee en silence. La facade ne peut pas la produire
  // (time.kind est un selecteur unique : "ssprk3" XOR "imex"), garde de defense en profondeur ici.
  if (tmethod == AmrTimeMethod::kSsprk3 && imex)
    throw std::runtime_error(
        "subcycle_level_mp : SSPRK3 + IMEX non supporte (combinaison non validee) ; utiliser "
        "time='ssprk3' (source explicite par etage) ou time='imex' (Euler avant + source implicite)");
  const SubcyclingSchedule sched(2);
  const int nc = L[lev].U.ncomp();
  AmrLevelMP& lv = L[lev];
  const int np = lv.U.local_size();
  const bool ssprk3 = (tmethod == AmrTimeMethod::kSsprk3);

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

  // SSPRK3 : on AVANCE D'ABORD lv.U de t a t+dt (3 etages) ET on remplace (fx, fy) -- qui contiennent
  // l'unique flux F(U0) du chemin Euler -- par le FLUX EFFECTIF Feff = 1/6 F0 + 1/6 F1 + 2/3 F2. Tout
  // le reste de la fonction (registre du parent, registres des enfants, flux grossier sauve, reflux)
  // lit (fx, fy) et l'etat avance EXACTEMENT comme en Euler : enregistrer Feff (au lieu de F0) rend le
  // reflux conservatif pour le pas SSP complet (le cote grossier g.c* = Feff grossier, le cote fin
  // g.f* = somme des Feff fins sous-cycles, et la correction -(g.f - g.c*dt)/dx remplace bien le flux
  // grossier effectif par le flux fin effectif). L'etat de depart est sauve AVANT l'avance pour
  // l'interpolation temporelle des enfants (role grossier). En Euler (ssprk3 == false) ce bloc est
  // saute et l'avance reste celle d'origine, en place plus bas -> strictement bit-identique.
  const bool is_leaf = (lev + 1 >= static_cast<int>(L.size()));
  MultiFab ssp_U_old;  // etat t (capture pre-avance) ; rempli seulement pour SSPRK3 + role grossier
  if (ssprk3) {
    if (!is_leaf) ssp_U_old = lv.U;  // les enfants interpolent entre cet etat (t) et lv.U avance (t+dt)
    ssprk3_advance_level<Limiter, NumericalFlux>(m, lv, dt, fx, fy, recon_prim, lev, base_dom,
                                                 base_per, pOld, pNew, frac, coarse_replicated);
  }

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

  if (is_leaf) {  // feuille
    if (!ssprk3) {  // Euler avant (chemin historique) ; SSPRK3 a deja avance lv.U ci-dessus
      mf_advance_faces(lv.U, fx, fy, lv.dx, lv.dy, dt);
      mf_apply_source_treatment(m, lv.U, *lv.aux, dt, imex, nopts);  // source explicite ou IMEX (options Newton)
    }
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

  // etat t pour l'interpolation temporelle des enfants. SSPRK3 : lv.U est DEJA avance (l'avance a eu
  // lieu plus haut, dans ssprk3_advance_level), l'etat t est la copie pre-avance ssp_U_old ; Euler :
  // lv.U est encore l'etat t ici (l'avance est juste en dessous), donc U_old = lv.U (copie historique).
  MultiFab U_old = ssprk3 ? ssp_U_old : lv.U;
  if (!ssprk3) {  // Euler avant (chemin historique) ; SSPRK3 a deja avance lv.U
    mf_advance_faces(lv.U, fx, fy, lv.dx, lv.dy, dt);
    mf_apply_source_treatment(m, lv.U, *lv.aux, dt, imex, nopts);  // source explicite ou IMEX (options Newton)
  }
  for (int s = 0; s < sched.count(); ++s)  // chaque sous-pas fin = un pas SSP complet (tmethod propage)
    subcycle_level_mp<Limiter, NumericalFlux>(m, L, lev + 1, sched.dt_sub(dt), base_dom, base_per,
                                              &U_old, &lv.U, sched.frac(s), &regs,
                                              coarse_replicated, recon_prim, imex, nopts, tmethod);
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
                                    bool coarse_replicated = true, bool recon_prim = false,
                                    bool imex = false, const NewtonOptions& nopts = {},
                                    AmrTimeMethod tmethod = AmrTimeMethod::kEuler) {
  subcycle_level_mp<Limiter, NumericalFlux>(m, L, 0, dt, dom, per, nullptr, nullptr, Real(0), nullptr,
                                            coarse_replicated, recon_prim, imex, nopts, tmethod);
}

}  // namespace detail (moteur N-niveaux multi-patch)

}  // namespace adc
