#pragma once

#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/refinement.hpp>  // coarsen_index
#include <adc/operator/spatial_operator.hpp>  // compute_face_fluxes, xface_box, yface_box
#include <adc/parallel/comm.hpp>  // all_reduce_sum_inplace (reflux multi-patch distribue)

#include <vector>

// Pas AMR 2-niveaux conservatif (Berger-Oliger r=2 + reflux), version PORTEE sur
// MultiFab + le seam, et GENERIQUE (Limiter, NumericalFlux, N composantes). C'est le
// pendant MultiFab de integrator/amr_reflux.hpp::amr_step_2level (Fab2D, 1 comp,
// Rusanov 1er ordre) : meme algorithme, mais les flux passent par compute_face_fluxes
// (donc MUSCL / HLL / HLLC / Euler-Poisson / GPU dispo) et les boucles bulk par
// for_each_cell. Region fine = cellules grossieres [CI0..CI1]x[CJ0..CJ1] (strictement
// interieures), raffinees ratio 2. Mono-box par niveau (le multi-patch viendra).

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
      if (!v.contains(i, j)) {
        const int ci = coarsen_index(i, 2), cj = coarsen_index(j, 2);
        for (int k = 0; k < nc; ++k)
          f(i, j, k) = (1 - frac) * co(ci, cj, k) + frac * cn(ci, cj, k);
      }
}

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
  compute_face_fluxes<Limiter, NumericalFlux>(m, Uc, auxc, fxc, fyc);
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

  // --- sous-cyclage fin : r sous-pas, accumulation des flux fins (x dtf) ---
  std::vector<Real> fL(nJ * nc, 0), fR(nJ * nc, 0), fB(nI * nc, 0), fT(nI * nc, 0);
  MultiFab fxf(BoxArray(std::vector<Box2D>{xface_box(Uf.box(0))}), Uf.dmap(), nc, 0);
  MultiFab fyf(BoxArray(std::vector<Box2D>{yface_box(Uf.box(0))}), Uf.dmap(), nc, 0);
  for (int s = 0; s < r; ++s) {
    mf_fill_fine_ghosts_t(Uf, Uc_old, Uc, Real(s) / r);
    compute_face_fluxes<Limiter, NumericalFlux>(m, Uf, auxf, fxf, fyf);
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
  compute_face_fluxes<Limiter, NumericalFlux>(m, lv.U, *lv.aux, fx, fy);

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

// --- MULTI-PATCH (plusieurs boxes fines par niveau) ---
// Le niveau fin est un MultiFab a N boxes. Le reflux est COVERAGE-AWARE : il ne
// corrige une cellule grossiere adjacente a une box fine que si elle n'est PAS
// couverte par une autre box fine (interface fin-grossier reelle ; les interfaces
// fin-fin sont gerees par fill_boundary). C'est la logique FluxRegister d'AMReX.

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
        if (!v.contains(i, j)) {
          const int ci = coarsen_index(i, 2), cj = coarsen_index(j, 2);
          for (int k = 0; k < nc; ++k)
            f(i, j, k) = (1 - frac) * co(ci, cj, k) + frac * cn(ci, cj, k);
        }
  }
}

// moyenne fin -> grossier sur l'empreinte de CHAQUE box fine (multi-box).
inline void mf_average_down_multi(const MultiFab& Uf, MultiFab& Uc) {
  const int nc = Uc.ncomp();
  Array4 c = Uc.fab(0).array();
  for (int li = 0; li < Uf.local_size(); ++li) {
    const ConstArray4 f = Uf.fab(li).const_array();
    const Box2D fb = Uf.box(li);
    const int I0 = fb.lo[0] / 2, I1 = (fb.hi[0] - 1) / 2;
    const int J0 = fb.lo[1] / 2, J1 = (fb.hi[1] - 1) / 2;
    for_each_cell(Box2D{{I0, J0}, {I1, J1}}, [=] ADC_HD(int I, int J) {
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
  const int r = 2, nc = Uc.ncomp();
  const Real dxf = dxc / 2, dyf = dyc / 2, dtf = dt / r;
  const int NX = dom.nx(), NY = dom.ny();

  // masque de couverture : cellules grossieres couvertes par une box fine. Bati sur le
  // BoxArray GLOBAL (toutes les boxes, connues de tous les rangs) -> correct sous MPI.
  std::vector<char> cov(static_cast<std::size_t>(NX) * NY, 0);
  for (int g = 0; g < Uf.box_array().size(); ++g) {
    const Box2D fb = Uf.box_array()[g];
    for (int J = fb.lo[1] / 2; J <= (fb.hi[1] - 1) / 2; ++J)
      for (int I = fb.lo[0] / 2; I <= (fb.hi[0] - 1) / 2; ++I)
        cov[static_cast<std::size_t>(J) * NX + I] = 1;
  }
  auto covered = [&](int I, int J) {
    return (I >= 0 && I < NX && J >= 0 && J < NY) ? cov[static_cast<std::size_t>(J) * NX + I] : char(0);
  };

  MultiFab Uc_old = Uc;
  fill_periodic_local(Uc, dom);  // grossier replique -> fill periodique local (pas de plan MPI)
  MultiFab fxc(BoxArray(std::vector<Box2D>{xface_box(Uc.box(0))}), Uc.dmap(), nc, 0);
  MultiFab fyc(BoxArray(std::vector<Box2D>{yface_box(Uc.box(0))}), Uc.dmap(), nc, 0);
  compute_face_fluxes<Limiter, NumericalFlux>(m, Uc, auxc, fxc, fyc);

  // registre par box fine : flux grossier (sans dt) sauve aux 4 faces.
  struct Reg { int I0, I1, J0, J1; std::vector<Real> cL, cR, cB, cT, fL, fR, fB, fT; };
  std::vector<Reg> regs(Uf.local_size());
  {
    device_fence();
    const ConstArray4 FX = fxc.fab(0).const_array(), FY = fyc.fab(0).const_array();
    for (int li = 0; li < Uf.local_size(); ++li) {
      const Box2D fb = Uf.box(li);
      Reg& g = regs[li];
      g.I0 = fb.lo[0] / 2; g.I1 = (fb.hi[0] - 1) / 2;
      g.J0 = fb.lo[1] / 2; g.J1 = (fb.hi[1] - 1) / 2;
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

  for (int s = 0; s < r; ++s) {
    mf_fill_fine_ghosts_multi(Uf, Uc_old, Uc, Real(s) / r);
    fill_boundary(Uf, fdom, Periodicity{false, false});  // halos fin-fin
    compute_face_fluxes<Limiter, NumericalFlux>(m, Uf, auxf, fxf, fyf);
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
  const std::size_t N = static_cast<std::size_t>(NX) * NY * nc;
  std::vector<Real> avg(N, Real(0)), ref(N, Real(0));
  auto idx = [&](int I, int J, int k) { return (static_cast<std::size_t>(J) * NX + I) * nc + k; };
  auto radd = [&](int I, int J, int k, Real v) {
    if (I >= 0 && I < NX && J >= 0 && J < NY) ref[idx(I, J, k)] += v;
  };
  for (int li = 0; li < Uf.local_size(); ++li) {
    const ConstArray4 f = Uf.fab(li).const_array();
    Reg& g = regs[li];
    for (int J = g.J0; J <= g.J1; ++J)
      for (int I = g.I0; I <= g.I1; ++I)
        for (int k = 0; k < nc; ++k)
          avg[idx(I, J, k)] = Real(0.25) * (f(2 * I, 2 * J, k) + f(2 * I + 1, 2 * J, k) +
                                            f(2 * I, 2 * J + 1, k) + f(2 * I + 1, 2 * J + 1, k));
    for (int J = g.J0; J <= g.J1; ++J)
      for (int k = 0; k < nc; ++k) {
        if (!covered(g.I0 - 1, J))
          radd(g.I0 - 1, J, k, -(g.fL[(J - g.J0) * nc + k] - g.cL[(J - g.J0) * nc + k] * dt) / dxc);
        if (!covered(g.I1 + 1, J))
          radd(g.I1 + 1, J, k, +(g.fR[(J - g.J0) * nc + k] - g.cR[(J - g.J0) * nc + k] * dt) / dxc);
      }
    for (int I = g.I0; I <= g.I1; ++I)
      for (int k = 0; k < nc; ++k) {
        if (!covered(I, g.J0 - 1))
          radd(I, g.J0 - 1, k, -(g.fB[(I - g.I0) * nc + k] - g.cB[(I - g.I0) * nc + k] * dt) / dyc);
        if (!covered(I, g.J1 + 1))
          radd(I, g.J1 + 1, k, +(g.fT[(I - g.I0) * nc + k] - g.cT[(I - g.I0) * nc + k] * dt) / dyc);
      }
  }
  all_reduce_sum_inplace(avg.data(), static_cast<int>(N));
  all_reduce_sum_inplace(ref.data(), static_cast<int>(N));
  if (Uc.local_size() > 0) {  // chaque rang detenant une copie du grossier l'applique
    Array4 c = Uc.fab(0).array();
    const Box2D cb = Uc.box(0);
    for (int J = cb.lo[1]; J <= cb.hi[1]; ++J)
      for (int I = cb.lo[0]; I <= cb.hi[0]; ++I)
        for (int k = 0; k < nc; ++k) {
          if (covered(I, J)) c(I, J, k) = avg[idx(I, J, k)];  // moyenne descendante
          c(I, J, k) += ref[idx(I, J, k)];                    // reflux (0 si pas de face)
        }
  }
}

// --- MULTI-PATCH N-NIVEAUX (multi-box a CHAQUE niveau) ---
// Generalise subcycle_level_mf : chaque niveau est un MultiFab multi-box. Le reflux
// (registre FluxRegister) est coverage-aware ET route la correction vers la box PARENTE
// contenant la cellule grossiere adjacente. Se reduit BIT A BIT au chemin mono-box quand
// chaque niveau n'a qu'une box (garde de validation).

// box LOCALE (valide) contenant la cellule (I,J), ou -1.
inline int mf_find_box(const MultiFab& mf, int I, int J) {
  for (int li = 0; li < mf.local_size(); ++li)
    if (mf.box(li).contains(I, J)) return li;
  return -1;
}

// ghosts fins multi-box depuis un parent MULTI-BOX (interp espace constant + temps lin).
inline void mf_fill_fine_ghosts_mb(MultiFab& Uf, const MultiFab& Po, const MultiFab& Pn,
                                   Real frac) {
  device_fence();
  const int nc = Uf.ncomp();
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
          for (int k = 0; k < nc; ++k)
            f(i, j, k) = (1 - frac) * po(ci, cj, k) + frac * pn(ci, cj, k);
        }
  }
}

// moyenne fin multi-box -> parent multi-box (chaque cellule routee vers sa box parente).
inline void mf_average_down_mb(const MultiFab& Uf, MultiFab& Uc) {
  device_fence();
  const int nc = Uc.ncomp();
  for (int lf = 0; lf < Uf.local_size(); ++lf) {
    const ConstArray4 f = Uf.fab(lf).const_array();
    const Box2D fb = Uf.box(lf);
    const int I0 = fb.lo[0] / 2, I1 = (fb.hi[0] - 1) / 2;
    const int J0 = fb.lo[1] / 2, J1 = (fb.hi[1] - 1) / 2;
    for (int J = J0; J <= J1; ++J)
      for (int I = I0; I <= I1; ++I) {
        const int pb = mf_find_box(Uc, I, J);
        if (pb < 0) continue;
        Array4 c = Uc.fab(pb).array();
        for (int k = 0; k < nc; ++k)
          c(I, J, k) = Real(0.25) * (f(2 * I, 2 * J, k) + f(2 * I + 1, 2 * J, k) +
                                     f(2 * I, 2 * J + 1, k) + f(2 * I + 1, 2 * J + 1, k));
      }
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

template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void subcycle_level_mp(const Model& m, std::vector<AmrLevelMP>& L, int lev, Real dt,
                       const Box2D& base_dom, Periodicity base_per, const MultiFab* pOld,
                       const MultiFab* pNew, Real frac, std::vector<RegMP>* parentRegs) {
  const int r = 2, nc = L[lev].U.ncomp();
  AmrLevelMP& lv = L[lev];
  const int np = lv.U.local_size();

  if (lev == 0) {
    fill_boundary(lv.U, base_dom, base_per);
  } else {
    mf_fill_fine_ghosts_mb(lv.U, *pOld, *pNew, frac);
    const Box2D fdom = Box2D::from_extents(base_dom.nx() << lev, base_dom.ny() << lev);
    fill_boundary(lv.U, fdom, Periodicity{false, false});  // halos fin-fin
  }

  std::vector<Box2D> fxb, fyb;
  for (int li = 0; li < np; ++li) {
    fxb.push_back(xface_box(lv.U.box(li)));
    fyb.push_back(yface_box(lv.U.box(li)));
  }
  MultiFab fx(BoxArray(std::move(fxb)), lv.U.dmap(), nc, 0);
  MultiFab fy(BoxArray(std::move(fyb)), lv.U.dmap(), nc, 0);
  compute_face_fluxes<Limiter, NumericalFlux>(m, lv.U, *lv.aux, fx, fy);
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
    return;
  }

  // role GROSSIER pour lev+1 : couverture + registres + flux grossier sauve.
  const int NX = base_dom.nx() << lev, NY = base_dom.ny() << lev;
  std::vector<char> cov(static_cast<std::size_t>(NX) * NY, 0);  // couverture GLOBALE (MPI-safe)
  for (int g = 0; g < L[lev + 1].U.box_array().size(); ++g) {
    const Box2D cb = L[lev + 1].U.box_array()[g];
    for (int J = cb.lo[1] / 2; J <= (cb.hi[1] - 1) / 2; ++J)
      for (int I = cb.lo[0] / 2; I <= (cb.hi[0] - 1) / 2; ++I)
        cov[static_cast<std::size_t>(J) * NX + I] = 1;
  }
  auto covered = [&](int I, int J) {
    return (I >= 0 && I < NX && J >= 0 && J < NY) ? cov[static_cast<std::size_t>(J) * NX + I]
                                                  : char(0);
  };

  std::vector<RegMP> regs(L[lev + 1].U.local_size());
  for (int lc = 0; lc < L[lev + 1].U.local_size(); ++lc) {
    const Box2D cb = L[lev + 1].U.box(lc);
    RegMP& g = regs[lc];
    g.I0 = cb.lo[0] / 2; g.I1 = (cb.hi[0] - 1) / 2;
    g.J0 = cb.lo[1] / 2; g.J1 = (cb.hi[1] - 1) / 2;
    const int nJ = g.J1 - g.J0 + 1, nI = g.I1 - g.I0 + 1;
    g.cL.assign(nJ * nc, 0); g.cR.assign(nJ * nc, 0);
    g.cB.assign(nI * nc, 0); g.cT.assign(nI * nc, 0);
    g.fL.assign(nJ * nc, 0); g.fR.assign(nJ * nc, 0);
    g.fB.assign(nI * nc, 0); g.fT.assign(nI * nc, 0);
    for (int J = g.J0; J <= g.J1; ++J) {
      const int pbL = mf_find_box(lv.U, g.I0, J), pbR = mf_find_box(lv.U, g.I1, J);
      const ConstArray4 FXL = fx.fab(pbL).const_array(), FXR = fx.fab(pbR).const_array();
      for (int k = 0; k < nc; ++k) {
        g.cL[(J - g.J0) * nc + k] = FXL(g.I0, J, k);
        g.cR[(J - g.J0) * nc + k] = FXR(g.I1 + 1, J, k);
      }
    }
    for (int I = g.I0; I <= g.I1; ++I) {
      const int pbB = mf_find_box(lv.U, I, g.J0), pbT = mf_find_box(lv.U, I, g.J1);
      const ConstArray4 FYB = fy.fab(pbB).const_array(), FYT = fy.fab(pbT).const_array();
      for (int k = 0; k < nc; ++k) {
        g.cB[(I - g.I0) * nc + k] = FYB(I, g.J0, k);
        g.cT[(I - g.I0) * nc + k] = FYT(I, g.J1 + 1, k);
      }
    }
  }

  MultiFab U_old = lv.U;  // etat t (interp temporelle pour les enfants)
  mf_advance_faces(lv.U, fx, fy, lv.dx, lv.dy, dt);
  for (int s = 0; s < r; ++s)
    subcycle_level_mp<Limiter, NumericalFlux>(m, L, lev + 1, dt / r, base_dom, base_per,
                                              &U_old, &lv.U, Real(s) / r, &regs);
  mf_average_down_mb(L[lev + 1].U, lv.U);

  device_fence();  // reflux coverage-aware route vers la box parente.
  for (int lc = 0; lc < static_cast<int>(regs.size()); ++lc) {
    RegMP& g = regs[lc];
    for (int J = g.J0; J <= g.J1; ++J)
      for (int k = 0; k < nc; ++k) {
        if (!covered(g.I0 - 1, J)) {
          const int pb = mf_find_box(lv.U, g.I0 - 1, J);
          if (pb >= 0)
            lv.U.fab(pb).array()(g.I0 - 1, J, k) -=
                (g.fL[(J - g.J0) * nc + k] - g.cL[(J - g.J0) * nc + k] * dt) / lv.dx;
        }
        if (!covered(g.I1 + 1, J)) {
          const int pb = mf_find_box(lv.U, g.I1 + 1, J);
          if (pb >= 0)
            lv.U.fab(pb).array()(g.I1 + 1, J, k) +=
                (g.fR[(J - g.J0) * nc + k] - g.cR[(J - g.J0) * nc + k] * dt) / lv.dx;
        }
      }
    for (int I = g.I0; I <= g.I1; ++I)
      for (int k = 0; k < nc; ++k) {
        if (!covered(I, g.J0 - 1)) {
          const int pb = mf_find_box(lv.U, I, g.J0 - 1);
          if (pb >= 0)
            lv.U.fab(pb).array()(I, g.J0 - 1, k) -=
                (g.fB[(I - g.I0) * nc + k] - g.cB[(I - g.I0) * nc + k] * dt) / lv.dy;
        }
        if (!covered(I, g.J1 + 1)) {
          const int pb = mf_find_box(lv.U, I, g.J1 + 1);
          if (pb >= 0)
            lv.U.fab(pb).array()(I, g.J1 + 1, k) +=
                (g.fT[(I - g.I0) * nc + k] - g.cT[(I - g.I0) * nc + k] * dt) / lv.dy;
        }
      }
  }
}

// Driver : un pas dt de la hierarchie multi-patch N-niveaux (niveau 0 = grossier).
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void amr_step_multilevel_multipatch(const Model& m, std::vector<AmrLevelMP>& L,
                                    const Box2D& dom, Real dt,
                                    Periodicity per = Periodicity{true, true}) {
  subcycle_level_mp<Limiter, NumericalFlux>(m, L, 0, dt, dom, per, nullptr, nullptr,
                                            Real(0), nullptr);
}

}  // namespace adc
