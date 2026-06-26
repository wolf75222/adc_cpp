#pragma once

#include <pops/numerics/time/reference/amr_reflux.hpp>
#include <pops/amr/hierarchy/refinement_ratio.hpp>
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/storage/fab2d.hpp>

#include <vector>

// Multi-level AMR (N nested levels, ratio 2): recursive Berger-Oliger
// subcycling with reflux at each coarse-fine interface. Generalizes
// amr_step_2level: each level that owns a child plays the role of the
// "coarse" of the 2-level step with respect to that child.
//
// Assumption (single-box per level): each level l>=1 is a single
// rectangular box, ratio-2 refinement of the subdomain [rCI0..rCI1]x[rCJ0..rCJ1]
// of the parent level l-1, strictly interior (>=1 coarse cell of margin)
// so that the fine ghosts, coarsened, fall back into valid cells
// of the parent. Level 0 covers the periodic domain.
//
// Invariants preserved from amr_step_2level, applied recursively:
//   - coarse flux saved on the 4 faces of the child (x dt of the current level);
//   - fine fluxes accumulated in the parent register (x dt of the child), at
//     each substep;
//   - average_down child -> parent over the covered region;
//   - reflux: the adjacent coarse cell receives (sum of fine fluxes - coarse
//     flux x dt) / dx.
// Result: conservation to roundoff over the whole hierarchy, stability via the
// space-time FillPatch (fine ghosts interpolated between old/new parent).

/// @file
/// @brief REFERENCE N-level AMR on single-box Fab2D: struct AmrLevel and the Berger-Oliger
///        recursion (subcycle_level, amr_step_multilevel). DEPRECATED.
///
/// Layer: `include/pops/numerics/time`.
/// Role: documented ground truth (cf. docs/ARCHITECTURE.md) generalizing amr_step_2level:
///        each level that owns a child plays the "coarse" of the 2-level step with respect to
///        that child. Conservation to roundoff via reflux at each coarse-fine interface,
///        stability via space-time FillPatch of the fine ghosts.
///
/// Invariants:
/// - DEPRECATED: no include in the core, the tests or the bindings; the production
///   engine is amr_reflux_mf.hpp (multi-patch MultiFab stack) of which single-box is the
///   degenerate case. To remove after migration of the references to amr_reflux_mf.hpp;
/// - single-box assumption: each level l>=1 is ONE ratio-2 refined box of a subdomain
///   of the parent, strictly interior (>=1 coarse cell of margin);
/// - aux held elsewhere (pointer); rC* = region refined by the child, valid if has_fine.

namespace pops {

static_assert(kAmrRefRatio == 2, "ratio-2-structural kernels below assume kAmrRefRatio == 2");

// One level of the hierarchy. aux is held elsewhere (recomputed per step);
// only a pointer is kept. rC* describes the region (in coords of THIS level)
// refined by the child l+1; valid only if has_fine.
struct AmrLevel {
  Fab2D U;                     // state, 1 component, 1 ghost
  const Fab2D* aux;            // [phi, gx, gy], ghosts filled
  double dx, dy;               // space step of this level
  int rCI0, rCI1, rCJ0, rCJ1;  // region refined by the child (local coords)
  bool has_fine;               // does a level l+1 exist?
};

// Applies U -= dt div(F) from already-computed fluxes (ghosts untouched).
inline void apply_flux_div_1c(Fab2D& U, const Fab2D& fx, const Fab2D& fy, double dx, double dy,
                              double dt) {
  Array4 uu = U.array();
  const ConstArray4 FX = fx.const_array();
  const ConstArray4 FY = fy.const_array();
  const Box2D v = U.box();
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i)
      uu(i, j) -= dt * ((FX(i + 1, j) - FX(i, j)) / dx + (FY(i, j + 1) - FY(i, j)) / dy);
}

// Recursively advances level lev by dt. pOld/pNew = parent states
// (old/new) bounding the parent step; frac = temporal position of the
// current substep within that step (s/r). The quadruplet preg = parent register
// (accumulator of the fine fluxes of THIS level); null if lev == 0.
template <class Model>
void subcycle_level(const Model& m, std::vector<AmrLevel>& L, int lev, double dt, const Box2D& dom,
                    const Fab2D* pOld, const Fab2D* pNew, double frac, std::vector<double>* pregL,
                    std::vector<double>* pregR, std::vector<double>* pregB,
                    std::vector<double>* pregT) {
  const int r = 2;
  AmrLevel& lv = L[lev];

  // --- 1. ghosts: periodic (level 0) or space-time injection ---
  if (lev == 0)
    fill_periodic_fab(lv.U, dom);
  else
    fill_fine_ghosts_t(lv.U, *pOld, *pNew, frac);

  // --- 2. fluxes of this level (start-of-substep state) ---
  Fab2D fx(xface_box(lv.U.box()), 1, 0), fy(yface_box(lv.U.box()), 1, 0);
  compute_fluxes_1c(m, lv.U, *lv.aux, fx, fy);
  const ConstArray4 FX = fx.const_array();
  const ConstArray4 FY = fy.const_array();

  // --- 3. contribution to the parent register (fine fluxes x dt of this level) ---
  if (lev > 0) {
    const AmrLevel& par = L[lev - 1];
    const int pI0 = par.rCI0, pI1 = par.rCI1, pJ0 = par.rCJ0, pJ1 = par.rCJ1;
    for (int J = pJ0; J <= pJ1; ++J) {
      (*pregL)[J - pJ0] += 0.5 * (FX(2 * pI0, 2 * J) + FX(2 * pI0, 2 * J + 1)) * dt;
      (*pregR)[J - pJ0] += 0.5 * (FX(2 * pI1 + 2, 2 * J) + FX(2 * pI1 + 2, 2 * J + 1)) * dt;
    }
    for (int I = pI0; I <= pI1; ++I) {
      (*pregB)[I - pI0] += 0.5 * (FY(2 * I, 2 * pJ0) + FY(2 * I + 1, 2 * pJ0)) * dt;
      (*pregT)[I - pI0] += 0.5 * (FY(2 * I, 2 * pJ1 + 2) + FY(2 * I + 1, 2 * pJ1 + 2)) * dt;
    }
  }

  if (!lv.has_fine) {
    apply_flux_div_1c(lv.U, fx, fy, lv.dx, lv.dy, dt);  // leaf: plain advance
    return;
  }

  // --- 4. level with child: local register + coarse flux saved ---
  const int cI0 = lv.rCI0, cI1 = lv.rCI1, cJ0 = lv.rCJ0, cJ1 = lv.rCJ1;
  const int nI = cI1 - cI0 + 1, nJ = cJ1 - cJ0 + 1;
  std::vector<double> cL(nJ), cR(nJ), cB(nI), cT(nI);  // coarse flux (without dt)
  for (int J = cJ0; J <= cJ1; ++J) {
    cL[J - cJ0] = FX(cI0, J);
    cR[J - cJ0] = FX(cI1 + 1, J);
  }
  for (int I = cI0; I <= cI1; ++I) {
    cB[I - cI0] = FY(I, cJ0);
    cT[I - cI0] = FY(I, cJ1 + 1);
  }
  std::vector<double> fL(nJ, 0), fR(nJ, 0), fB(nI, 0), fT(nI, 0);  // fine fluxes

  const Fab2D U_old = lv.U;                           // state t (for the child's temporal interp)
  apply_flux_div_1c(lv.U, fx, fy, lv.dx, lv.dy, dt);  // lv.U becomes the state t+dt

  // --- 5. subcycling of the child: r substeps of dt/r ---
  for (int s = 0; s < r; ++s)
    subcycle_level(m, L, lev + 1, dt / r, dom, &U_old, &lv.U, double(s) / r, &fL, &fR, &fB, &fT);

  average_down_fab(L[lev + 1].U, lv.U, cI0, cI1, cJ0, cJ1);  // sync covered

  // --- 6. reflux: coarse flux (x dt) replaced by sum of fine fluxes ---
  Array4 c = lv.U.array();
  for (int J = cJ0; J <= cJ1; ++J) {
    c(cI0 - 1, J) -= (fL[J - cJ0] - cL[J - cJ0] * dt) / lv.dx;
    c(cI1 + 1, J) += (fR[J - cJ0] - cR[J - cJ0] * dt) / lv.dx;
  }
  for (int I = cI0; I <= cI1; ++I) {
    c(I, cJ0 - 1) -= (fB[I - cI0] - cB[I - cI0] * dt) / lv.dy;
    c(I, cJ1 + 1) += (fT[I - cI0] - cT[I - cI0] * dt) / lv.dy;
  }
}

// Driver: one dt step of the complete hierarchy (level 0 = coarse).
template <class Model>
void amr_step_multilevel(const Model& m, std::vector<AmrLevel>& L, const Box2D& dom, double dt) {
  subcycle_level(m, L, 0, dt, dom, nullptr, nullptr, 0.0, nullptr, nullptr, nullptr, nullptr);
}

}  // namespace pops
