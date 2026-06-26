#pragma once
#include <pops/numerics/time/amr/reflux/amr_flux_helpers.hpp>
#include <pops/amr/hierarchy/refinement_ratio.hpp>

/// @file
/// @brief Single-box MultiFab AMR stack: struct detail::AmrLevelMF and the recursive
///        engine (amr_step_2level_mf, subcycle_level_mf, amr_step_multilevel_mf). TEST
///        ORACLE, out of production.
///
/// Layer: `include/pops/numerics/time`.
/// Role: original SINGLE-BOX AMR engine, kept in detail:: as a validation oracle.
///       Intermediate link of the Fab2D (amr_reflux.hpp) -> MF (here) -> MP
///       (amr_subcycling.hpp) parity chain, which proves the multi-patch engine
///       bit-identical to the single-box one up to N levels (which the Fab2D
///       2-level/1-comp/Rusanov-only test does not cover).
/// Contract: one conservative step with Berger-Oliger subcycling (ratio 2, dt/2 per
///           substep) with reflux at each coarse-fine interface; periodic BCs at level 0.
///
/// Invariants:
/// - no longer called in production (AmrCoupler goes through advance_amr);
/// - parity guards: test_amr_reflux_mf (vs Fab2D), test_amr_multilevel_mf,
///   test_amr_multilevel_multipatch guard 1 (MP vs MF, max|dUc| == 0);
/// - aux held elsewhere (pointer); rC* = region (coords of THIS level) refined by the
///   child, valid only if has_fine.

namespace pops {

static_assert(kAmrRefRatio == 2, "ratio-2-structural kernels below assume kAmrRefRatio == 2");

namespace detail {  // single-box MF oracle: Fab2D -> MF -> MP

// One level of the MultiFab hierarchy. aux held elsewhere (pointer). rC* = region
// (coords of THIS level) refined by the child; valid if has_fine.
struct AmrLevelMF {
  MultiFab U;
  const MultiFab* aux;
  Real dx, dy;
  int rCI0, rCI1, rCJ0, rCJ1;
  bool has_fine;
};

// One conservative 2-level step. Uc: coarse (periodic domain, ghosts for the
// Limiter). Uf: fine (refined box). auxc/auxf: (phi, grad phi) prescribed, ghosts
// filled. dt = coarse step; the fine one does r=2 substeps of dt/2 then reflux.
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void amr_step_2level_mf(const Model& m, MultiFab& Uc, const Box2D& dom, Real dxc, Real dyc,
                        MultiFab& Uf, int CI0, int CI1, int CJ0, int CJ1, const MultiFab& auxc,
                        const MultiFab& auxf, Real dt) {
  const int r = kAmrRefRatio, nc = Uc.ncomp();
  const Real dxf = dxc / kAmrRefRatio, dyf = dyc / kAmrRefRatio, dtf = dt / r;
  const int nJ = CJ1 - CJ0 + 1, nI = CI1 - CI0 + 1;
  MultiFab Uc_old = Uc;  // coarse state at time t (temporal interp of fine ghosts)

  // --- coarse fluxes at the 4 faces of the fine region (before update) ---
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
  mf_advance_faces(Uc, fxc, fyc, dxc, dyc, dt);  // Uc -> state t+dt
  mf_apply_source(m, Uc, auxc, dt);              // source S(U,aux) at the substep

  // --- fine subcycling: r substeps, accumulation of fine fluxes (x dtf) ---
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
        fR[(J - CJ0) * nc + k] +=
            Real(0.5) * (FX(2 * CI1 + 2, 2 * J, k) + FX(2 * CI1 + 2, 2 * J + 1, k)) * dtf;
      }
    for (int I = CI0; I <= CI1; ++I)
      for (int k = 0; k < nc; ++k) {
        fB[(I - CI0) * nc + k] +=
            Real(0.5) * (FY(2 * I, 2 * CJ0, k) + FY(2 * I + 1, 2 * CJ0, k)) * dtf;
        fT[(I - CI0) * nc + k] +=
            Real(0.5) * (FY(2 * I, 2 * CJ1 + 2, k) + FY(2 * I + 1, 2 * CJ1 + 2, k)) * dtf;
      }
    mf_advance_faces(Uf, fxf, fyf, dxf, dyf, dtf);
    mf_apply_source(m, Uf, auxf, dtf);  // source S(U,aux) at the substep
  }

  mf_average_down(Uf, Uc, CI0, CI1, CJ0, CJ1);  // sync of covered cells

  // --- reflux: coarse flux (x dt) replaced by sum of fine fluxes (x dtf) ---
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

// --- N-level recursion (MultiFab counterpart of amr_multilevel.hpp) ---

// Recursively advances level lev by dt (Berger-Oliger subcycling r=2 + reflux).
// pOld/pNew = parent states bounding the step; preg* = parent register (fine fluxes of
// THIS level), null if lev==0. Generic (Limiter, NumericalFlux, N comp), GPU seam.
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void subcycle_level_mf(const Model& m, std::vector<AmrLevelMF>& L, int lev, Real dt,
                       const Box2D& dom, const MultiFab* pOld, const MultiFab* pNew, Real frac,
                       std::vector<Real>* pregL, std::vector<Real>* pregR, std::vector<Real>* pregB,
                       std::vector<Real>* pregT) {
  const int r = kAmrRefRatio;
  AmrLevelMF& lv = L[lev];
  const int nc = lv.U.ncomp();

  if (lev == 0)
    fill_boundary(lv.U, dom, Periodicity{true, true});
  else
    mf_fill_fine_ghosts_t(lv.U, *pOld, *pNew, frac);

  MultiFab fx(BoxArray(std::vector<Box2D>{xface_box(lv.U.box(0))}), lv.U.dmap(), nc, 0);
  MultiFab fy(BoxArray(std::vector<Box2D>{yface_box(lv.U.box(0))}), lv.U.dmap(), nc, 0);
  compute_face_fluxes<Limiter, NumericalFlux>(m, lv.U, *lv.aux, fx, fy, lv.dx, lv.dy);

  if (lev > 0) {  // contribution to the parent register (fine fluxes x dt)
    device_fence();
    const ConstArray4 FX = fx.fab(0).const_array(), FY = fy.fab(0).const_array();
    const AmrLevelMF& par = L[lev - 1];
    const int pI0 = par.rCI0, pI1 = par.rCI1, pJ0 = par.rCJ0, pJ1 = par.rCJ1;
    for (int J = pJ0; J <= pJ1; ++J)
      for (int k = 0; k < nc; ++k) {
        (*pregL)[(J - pJ0) * nc + k] +=
            Real(0.5) * (FX(2 * pI0, 2 * J, k) + FX(2 * pI0, 2 * J + 1, k)) * dt;
        (*pregR)[(J - pJ0) * nc + k] +=
            Real(0.5) * (FX(2 * pI1 + 2, 2 * J, k) + FX(2 * pI1 + 2, 2 * J + 1, k)) * dt;
      }
    for (int I = pI0; I <= pI1; ++I)
      for (int k = 0; k < nc; ++k) {
        (*pregB)[(I - pI0) * nc + k] +=
            Real(0.5) * (FY(2 * I, 2 * pJ0, k) + FY(2 * I + 1, 2 * pJ0, k)) * dt;
        (*pregT)[(I - pI0) * nc + k] +=
            Real(0.5) * (FY(2 * I, 2 * pJ1 + 2, k) + FY(2 * I + 1, 2 * pJ1 + 2, k)) * dt;
      }
  }

  if (!lv.has_fine) {
    mf_advance_faces(lv.U, fx, fy, lv.dx, lv.dy, dt);  // leaf
    mf_apply_source(m, lv.U, *lv.aux, dt);             // source S(U,aux) at the substep
    return;
  }

  // level with a child: local register + coarse flux saved (without dt)
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

  MultiFab U_old = lv.U;                             // state t (temporal interp of the child)
  mf_advance_faces(lv.U, fx, fy, lv.dx, lv.dy, dt);  // lv.U -> t+dt
  mf_apply_source(m, lv.U, *lv.aux, dt);             // source S(U,aux) at the substep

  for (int s = 0; s < r; ++s)
    subcycle_level_mf<Limiter, NumericalFlux>(m, L, lev + 1, dt / r, dom, &U_old, &lv.U,
                                              Real(s) / r, &fL, &fR, &fB, &fT);

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

// Driver: one dt step of the full hierarchy (level 0 = coarse).
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void amr_step_multilevel_mf(const Model& m, std::vector<AmrLevelMF>& L, const Box2D& dom, Real dt) {
  subcycle_level_mf<Limiter, NumericalFlux>(m, L, 0, dt, dom, nullptr, nullptr, Real(0), nullptr,
                                            nullptr, nullptr, nullptr);
}

}  // namespace detail

}  // namespace pops
