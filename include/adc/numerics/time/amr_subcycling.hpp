#pragma once
#include <adc/mesh/storage/mf_arith.hpp>  // saxpy, lincomb (SSPRK3 stages, named device-clean functors)
#include <adc/amr/hierarchy/refinement_ratio.hpp>
#include <adc/mesh/layout/refinement.hpp>  // coarsen, parallel_copy
#include <adc/numerics/time/amr_flux_helpers.hpp>
#include <adc/numerics/time/amr_patch_range.hpp>

#include <cassert>  // assert (replicated-parent invariant: mf_find_box always finds it)

/// @file
/// @brief AMR multi-patch subcycling engine (several fine boxes per level): 2-level step
///        (amr_step_2level_multipatch), N-level recursion (detail::subcycle_level_mp,
///        detail::amr_step_multilevel_multipatch), SSPRK3 per-stage advance, multi-box helpers
///        (mf_fill_fine_ghosts_mb, mf_average_down_mb, mf_find_box, coarsen_grown) and types
///        AmrLevelMP / RegMP. This is the engine behind advance_amr.
///
/// Layer: `include/adc/numerics/time`.
/// Role: COVERAGE-AWARE reflux in the style of AMReX FluxRegister -- a coarse cell adjacent to a
///        fine patch is corrected ONLY if it is not covered by another patch (fine-fine
///        interfaces are handled by fill_boundary).
///
/// Invariants:
/// - distributed (MPI) with COARSE REPLICATION: the single-box coarse level is replicated on each
///   rank (local periodic fill), the fine patches distributed; average_down and reflux gather up
///   through GLOBAL-indexed coarse buffers + all_reduce_sum_inplace, then each rank applies to
///   its copy -> all stay identical. In serial this is bit-for-bit identical to the direct path;
/// - validation: test_mpi_amr_multipatch (np=1/2/4 bit-identical);
/// - SSPRK3 refills the ghosts BEFORE each stage flux evaluation (ssprk3_refill_level_ghosts),
///   and requires imex == false;
/// - saxpy/lincomb and the helper kernels are device-clean (named functors).

namespace adc {

static_assert(kAmrRefRatio == 2, "ratio-2-structural kernels below assume kAmrRefRatio == 2");

// --- MULTI-PATCH (several fine boxes per level) ---
// The fine level is a MultiFab with N boxes. Reflux is COVERAGE-AWARE: it corrects a coarse
// cell adjacent to a fine box only if it is NOT covered by another fine box (real fine-coarse
// interface; fine-fine interfaces are handled by fill_boundary). This is AMReX FluxRegister
// logic.

// Conservative 2-level step, MULTI-BOX fine level. Uc: single-box coarse (periodic).
// Uf: MultiFab with N fine boxes (ratio 2, strictly interior, coarse-aligned).
//
// Distributed (MPI) with COARSE REPLICATION. The single-box coarse level is replicated: each
// rank holds an identical copy (per-rank DistributionMapping, or deterministic init). The coarse
// advance (self-periodic fill_boundary, flux, advance) runs identically on each copy; the fine
// patches are distributed. average_down (overwrite of covered cells) and reflux (addition to
// bordering cells) gather up through two global-indexed coarse buffers + all_reduce_sum_inplace,
// then each rank applies to its copy -> all stay identical. In serial this is bit-for-bit
// identical to the direct path (see the final block). Validation: test_mpi_amr_multipatch
// (np=1/2/4 bit-identical). The coarse level is small (base level), so the replication is
// accepted; the recursive N-level path (subcycle_level_mp) still has to be generalized the same
// way (ROADMAP).
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void amr_step_2level_multipatch(const Model& m, MultiFab& Uc, const Box2D& dom, Real dxc, Real dyc,
                                MultiFab& Uf, const MultiFab& auxc, const MultiFab& auxf, Real dt) {
  const SubcyclingSchedule sched(2);
  const int nc = Uc.ncomp();
  const Real dxf = dxc / kAmrRefRatio, dyf = dyc / kAmrRefRatio, dtf = sched.dt_sub(dt);
  const int NX = dom.nx(), NY = dom.ny();

  // coarse-fine interface: coverage (coarse cells shadowed by a fine patch) + bordering reflux
  // routing. Coverage built on the GLOBAL BoxArray (all boxes, known to all ranks) -> correct
  // under MPI.
  const CoarseFineInterface cfi(Box2D{{0, 0}, {NX - 1, NY - 1}}, Uf.box_array());
  auto covered = [&](int I, int J) { return cfi.covered(I, J); };

  MultiFab Uc_old = Uc;
  fill_periodic_local(Uc, dom);  // replicated coarse -> local periodic fill (no MPI plan)
  MultiFab fxc(BoxArray(std::vector<Box2D>{xface_box(Uc.box(0))}), Uc.dmap(), nc, 0);
  MultiFab fyc(BoxArray(std::vector<Box2D>{yface_box(Uc.box(0))}), Uc.dmap(), nc, 0);
  compute_face_fluxes<Limiter, NumericalFlux>(m, Uc, auxc, fxc, fyc, dxc, dyc);

  // per fine-box register: coarse flux (without dt) saved at the 4 faces.
  struct Reg {
    int I0, I1, J0, J1;
    std::vector<Real> cL, cR, cB, cT, fL, fR, fB, fT;
  };
  std::vector<Reg> regs(Uf.local_size());
  {
    device_fence();
    const ConstArray4 FX = fxc.fab(0).const_array(), FY = fyc.fab(0).const_array();
    for (int li = 0; li < Uf.local_size(); ++li) {
      const PatchRange pr(Uf.box(li));
      Reg& g = regs[li];
      g.I0 = pr.I0;
      g.I1 = pr.I1;
      g.J0 = pr.J0;
      g.J1 = pr.J1;
      const int nJ = g.J1 - g.J0 + 1, nI = g.I1 - g.I0 + 1;
      g.cL.assign(nJ * nc, 0);
      g.cR.assign(nJ * nc, 0);
      g.cB.assign(nI * nc, 0);
      g.cT.assign(nI * nc, 0);
      g.fL.assign(nJ * nc, 0);
      g.fR.assign(nJ * nc, 0);
      g.fB.assign(nI * nc, 0);
      g.fT.assign(nI * nc, 0);
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
  mf_apply_source(m, Uc, auxc, dt);  // source S(U,aux) at the substep

  // multi-box fine fluxes: one face-box per GLOBAL fine box, same dmap as Uf. Built on the global
  // box_array() (not the local boxes) so that BoxArray and DistributionMapping have the same size
  // under MPI: fxf.fab(li) then corresponds to Uf.fab(li) (same dmap, same global order). In
  // serial it is identical (local == global).
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
    fill_boundary(Uf, fdom, Periodicity{false, false});  // fine-fine halos
    compute_face_fluxes<Limiter, NumericalFlux>(m, Uf, auxf, fxf, fyf, dxf, dyf);
    device_fence();
    for (int li = 0; li < Uf.local_size(); ++li) {
      Reg& g = regs[li];
      const ConstArray4 FX = fxf.fab(li).const_array(), FY = fyf.fab(li).const_array();
      for (int J = g.J0; J <= g.J1; ++J)
        for (int k = 0; k < nc; ++k) {
          g.fL[(J - g.J0) * nc + k] +=
              Real(0.5) * (FX(2 * g.I0, 2 * J, k) + FX(2 * g.I0, 2 * J + 1, k)) * dtf;
          g.fR[(J - g.J0) * nc + k] +=
              Real(0.5) * (FX(2 * g.I1 + 2, 2 * J, k) + FX(2 * g.I1 + 2, 2 * J + 1, k)) * dtf;
        }
      for (int I = g.I0; I <= g.I1; ++I)
        for (int k = 0; k < nc; ++k) {
          g.fB[(I - g.I0) * nc + k] +=
              Real(0.5) * (FY(2 * I, 2 * g.J0, k) + FY(2 * I + 1, 2 * g.J0, k)) * dtf;
          g.fT[(I - g.I0) * nc + k] +=
              Real(0.5) * (FY(2 * I, 2 * g.J1 + 2, k) + FY(2 * I + 1, 2 * g.J1 + 2, k)) * dtf;
        }
    }
    mf_advance_faces(Uf, fxf, fyf, dxf, dyf, dtf);
    mf_apply_source(m, Uf, auxf, dtf);  // source S(U,aux) at the substep
  }

  // DISTRIBUTED average_down + reflux, the coarse level being REPLICATED (each rank holds an
  // identical copy after the deterministic coarse advance). Each rank deposits, for its LOCAL
  // fine patches, into two global-indexed coarse buffers:
  //   avg: the average-down over the COVERED cells (overwrite semantics; a single contribution
  //        per cell since the patches are disjoint);
  //   ref: the reflux correction on the uncovered BORDERING cells (addition).
  // all_reduce_sum -> each rank has the total, then applies to ITS copy: covered = avg,
  // bordering += ref. All copies stay identical. In serial (np=1) the all-reduce is the identity
  // and it is bit-for-bit identical to the direct path (0 + average = average exactly; advance +
  // correction). Cost: two NX*NY*nc buffers per rank (coarse replication).
  device_fence();
  // register restricted to the coarse-fine INTERFACE (bounding box of the fine footprints, grown
  // by 1 for the bordering reflux cells, clamped to the domain): the gather all_reduce goes from
  // O(NX*NY) to O(interface). Bit-identical: cells outside the interface were zero (uncovered,
  // without a face), skipped at application.
  const Box2D fpc = coarsen(Uf.box_array(), kAmrRefRatio).bounding_box();
  const Box2D rbox{{std::max(fpc.lo[0] - 1, 0), std::max(fpc.lo[1] - 1, 0)},
                   {std::min(fpc.hi[0] + 1, NX - 1), std::min(fpc.hi[1] + 1, NY - 1)}};
  FluxRegister avg(rbox, nc);  // average-down (overwrite of covered cells)
  FluxRegister ref(rbox, nc);  // reflux (addition to bordering cells)
  for (int li = 0; li < Uf.local_size(); ++li) {
    const ConstArray4 f = Uf.fab(li).const_array();
    Reg& g = regs[li];
    for (int J = g.J0; J <= g.J1; ++J)
      for (int I = g.I0; I <= g.I1; ++I)
        for (int k = 0; k < nc; ++k)
          avg.set(I, J, k,
                  Real(0.25) * (f(2 * I, 2 * J, k) + f(2 * I + 1, 2 * J, k) +
                                f(2 * I, 2 * J + 1, k) + f(2 * I + 1, 2 * J + 1, k)));
    cfi.route_reflux(g, dxc, dyc, dt, ref, nc);  // coverage-aware bordering reflux
  }
  avg.gather();
  ref.gather();
  if (Uc.local_size() > 0) {  // each rank holding a copy of the coarse level applies it
    Array4 c = Uc.fab(0).array();
    const Box2D cb = Uc.box(0);
    for (int J = cb.lo[1]; J <= cb.hi[1]; ++J)
      for (int I = cb.lo[0]; I <= cb.hi[0]; ++I) {
        if (!ref.in(I, J))
          continue;  // outside interface: neither average nor reflux (was 0)
        for (int k = 0; k < nc; ++k) {
          if (covered(I, J))
            c(I, J, k) = avg.at(I, J, k);  // average-down
          c(I, J, k) += ref.at(I, J, k);   // reflux (0 if no face)
        }
      }
  }
}

// --- N-LEVEL MULTI-PATCH (multi-box at EACH level) ---
// Generalizes subcycle_level_mf: each level is a multi-box MultiFab. Reflux (FluxRegister) is
// coverage-aware AND routes the correction to the PARENT box containing the adjacent coarse cell.
// Reduces BIT-FOR-BIT to the single-box path when each level has only one box (validation guard).
//
// Distributed state (MPI): DISTRIBUTED and tested bit-for-bit identical np=1/2/4
// (test_mpi_amr_multipatch3, 3 levels with a distributed multi-box intermediate level whose fine
// patch PARENT falls on another rank). Level 0 (coarse) is REPLICATED as in the 2-level case;
// levels >0 are distributed and play the role of both child and parent simultaneously. The five
// points assuming a local parent (via mf_find_box) are resolved:
//   1. mf_fill_fine_ghosts_mb: REPLICATED parent (lev==1) read locally; DISTRIBUTED parent
//      (lev>=2) brought in by parallel_copy (parent -> fine-coarsen) then interpolated;
//   2. coarse register sampling: REPLICATED parent read locally, DISTRIBUTED parent brought in by
//      parallel_copy onto a child-coarsen FACE grid;
//   3. mf_average_down_mb: average deposited in a GLOBAL-indexed coarse buffer + all_reduce_sum,
//      applied to the local parent boxes (replicated: all; distributed: the owner);
//   4. reflux: same global buffer + all_reduce, application guarded by local ownership of the
//      parent box (no double counting since the distributed parent has a single owner);
//   5. coverage: already built on the global box_array() (MPI-safe).
// In serial all_reduce is the identity and parallel_copy reduces to memory copies: the
// distributed path runs the same floating-point operations as the single-rank one -> bit-
// identical.
// Note: AmrCouplerMP remains limited to single-rank beyond 2 distributed levels, because its
// parent->child aux injection (inject_aux_mb) still assumes the parent is local via mf_find_box;
// the integrator amr_step_multilevel_multipatch, on the other hand, is distributed.

// LOCAL (valid) box containing cell (I,J), or -1.
inline int mf_find_box(const MultiFab& mf, int I, int J) {
  for (int li = 0; li < mf.local_size(); ++li)
    if (mf.box(li).contains(I, J))
      return li;
  return -1;
}

// BoxArray of the child boxes grown by ngrow then coarsened (ratio 2). Each box covers all the
// coarse cells the child needs, ghosts included: this is the FillPatch fine-coarsen grid (cf.
// refinement.hpp::interpolate).
inline BoxArray coarsen_grown(const BoxArray& ba, int ngrow, int r) {
  std::vector<Box2D> b;
  b.reserve(ba.size());
  for (int i = 0; i < ba.size(); ++i)
    b.push_back(ba[i].grow(ngrow).coarsen(r));
  return BoxArray{std::move(b)};
}

// multi-box fine ghosts from a MULTI-BOX parent (constant-space + linear-time interp),
// DISTRIBUTED. Two parent cases:
//  - REPLICATED (level 0, replicated_parent=true): the parent is fully local on each rank, read
//    directly via mf_find_box (always found); no collective. This is the replicated-coarse path,
//    like the 2-level case (parallel_copy would violate the replicated-metadata assumption of the
//    parent, per-rank dmap).
//  - DISTRIBUTED (intermediate): the parent may be on another rank; its valid regions are brought
//    onto a LOCAL child-coarsen grid by parallel_copy (MPI routing handled there), then
//    interpolated. No more silent remote failures.
// In serial both paths are identical (parent local everywhere, parallel_copy = memory copy).
inline void mf_fill_fine_ghosts_mb(MultiFab& Uf, const MultiFab& Po, const MultiFab& Pn, Real frac,
                                   bool replicated_parent = true, Real pos_floor = Real(0),
                                   int pos_comp = 0) {
  const int nc = Uf.ncomp(), ng = Uf.n_grow();
  if (replicated_parent) {
    device_fence();
    for (int li = 0; li < Uf.local_size(); ++li) {
      Array4 f = Uf.fab(li).array();
      const Box2D v = Uf.box(li), g = Uf.fab(li).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i)
          if (!v.contains(i, j)) {
            const int ci = coarsen_index(i, kAmrRefRatio), cj = coarsen_index(j, kAmrRefRatio);
            const int pb = mf_find_box(Po, ci, cj);
            if (pb < 0)
              continue;  // outside parent coverage -> leave to fill_boundary
            const ConstArray4 po = Po.fab(pb).const_array(), pn = Pn.fab(pb).const_array();
            fill_cf_ghost_cell(f, po, pn, i, j, nc, frac, pos_floor, pos_comp);
          }
    }
    return;
  }
  const BoxArray pcoarse_ba = coarsen_grown(Uf.box_array(), ng, kAmrRefRatio);
  MultiFab Pco(pcoarse_ba, Uf.dmap(), nc, 0), Pcn(pcoarse_ba, Uf.dmap(), nc, 0);
  parallel_copy(Pco, Po);  // parent regions (from any rank) -> local grid
  parallel_copy(Pcn, Pn);
  device_fence();
  for (int li = 0; li < Uf.local_size(); ++li) {
    Array4 f = Uf.fab(li).array();
    const ConstArray4 po = Pco.fab(li).const_array(), pn = Pcn.fab(li).const_array();
    const Box2D v = Uf.box(li), g = Uf.fab(li).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i)
        if (!v.contains(i, j))
          fill_cf_ghost_cell(f, po, pn, i, j, nc, frac, pos_floor, pos_comp);
  }
}

// multi-box fine average -> multi-box parent (each cell routed to its parent box), DISTRIBUTED.
// The parent box of a coarse cell may be on another rank, and the parent may be either REPLICATED
// (level 0, each rank has a copy) or DISTRIBUTED (intermediate, a single owner). Both are covered
// by a GLOBAL-indexed coarse buffer: each rank deposits the 2x2 average of ITS local fine patches
// (0 elsewhere; disjoint patches so a single contribution per covered cell), all_reduce_sum ->
// each rank has the total, then applies to ITS local parent boxes (overwrite). Replicated: all
// apply the same value to their copy. Distributed: only the owner applies. In serial all_reduce
// is the identity (0 + average = average) -> bit-for-bit identical to the direct routing.
inline void mf_average_down_mb(const MultiFab& Uf, MultiFab& Uc) {
  const int nc = std::min(Uf.ncomp(), Uc.ncomp());
  // coarse bounding box (GLOBAL indices) covering all the child footprints.
  const BoxArray cba = coarsen(Uf.box_array(), kAmrRefRatio);
  Box2D bb{{0, 0}, {-1, -1}};
  for (int g = 0; g < cba.size(); ++g)
    bb = (g == 0) ? cba[g]
                  : Box2D{{std::min(bb.lo[0], cba[g].lo[0]), std::min(bb.lo[1], cba[g].lo[1])},
                          {std::max(bb.hi[0], cba[g].hi[0]), std::max(bb.hi[1], cba[g].hi[1])}};
  if (bb.empty()) {
    all_reduce_sum_inplace(nullptr, 0);
    return;
  }  // empty matched collective
  FluxRegister avg(bb, nc);  // multi-box average-down (region = bounding box)
  // GLOBAL coverage (all child footprints): only these cells are overwritten; the bounding box
  // may contain holes between disjoint patches that must NOT be overwritten.
  CoverageMask cmask(bb);
  for (int g = 0; g < cba.size(); ++g)
    cmask.mark(cba[g]);
  auto covered = [&](int I, int J) { return cmask.covered(I, J); };
  device_fence();
  for (int lf = 0; lf < Uf.local_size(); ++lf) {
    const ConstArray4 f = Uf.fab(lf).const_array();
    const PatchRange pr(Uf.box(lf));
    for (int J = pr.J0; J <= pr.J1; ++J)
      for (int I = pr.I0; I <= pr.I1; ++I)
        for (int k = 0; k < nc; ++k)
          avg.set(I, J, k,
                  Real(0.25) * (f(2 * I, 2 * J, k) + f(2 * I + 1, 2 * J, k) +
                                f(2 * I, 2 * J + 1, k) + f(2 * I + 1, 2 * J + 1, k)));
  }
  avg.gather();
  for (int pb = 0; pb < Uc.local_size(); ++pb) {
    Array4 c = Uc.fab(pb).array();
    const Box2D inter = Uc.box(pb).intersect(bb);
    for (int J = inter.lo[1]; J <= inter.hi[1]; ++J)
      for (int I = inter.lo[0]; I <= inter.hi[0]; ++I)
        if (covered(I, J))
          for (int k = 0; k < nc; ++k)
            c(I, J, k) = avg.at(I, J, k);
  }
}

// one level of the multi-patch hierarchy (U + multi-box aux, same BoxArray).
struct AmrLevelMP {
  MultiFab U;
  const MultiFab* aux;
  Real dx, dy;
};

// per child-patch register (PARENT coords I0..J1). c* = coarse flux (without dt);
// f* = time-integrated fine flux accumulated by the child during subcycling.
struct RegMP {
  int I0, I1, J0, J1;
  std::vector<Real> cL, cR, cB, cT, fL, fR, fB, fT;
};

namespace detail {  // INTERNAL N-level multi-patch engine; the public facade is advance_amr

// Fills the ghosts of an AMR level: level 0 = base-domain BC (fill_boundary); level > 0 =
// time-interpolated coarse-fine ghosts from the parent at position frac (mf_fill_fine_ghosts_mb)
// THEN fine-fine halos (fill_boundary). Factored out of the head of subcycle_level_mp, REUSED by
// the SSPRK3 advance which must refill the ghosts BEFORE each stage flux evaluation. The parent
// is REPLICATED only for lev == 1 (replicated level 0), otherwise distributed (internal
// parallel_copy).
inline void ssprk3_refill_level_ghosts(MultiFab& U, int lev, const Box2D& base_dom,
                                       Periodicity base_per, const MultiFab* pOld,
                                       const MultiFab* pNew, Real frac, bool coarse_replicated,
                                       Real pos_floor = Real(0), int pos_comp = 0) {
  if (lev == 0) {
    fill_boundary(U, base_dom, base_per);
  } else {
    mf_fill_fine_ghosts_mb(U, *pOld, *pNew, frac, (lev == 1) && coarse_replicated, pos_floor,
                           pos_comp);
    const Box2D fdom = Box2D::from_extents(base_dom.nx() << lev, base_dom.ny() << lev);
    fill_boundary(U, fdom, Periodicity{false, false});
  }
}

// SSPRK3 (Shu-Osher, 3 stages, order 3) on ONE AMR level. (1) Advance lv.U from t to t+dt:
//   U1 = U0 + dt L(U0); U2 = 3/4 U0 + 1/4 (U1 + dt L(U1)); U_new = 1/3 U0 + 2/3 (U2 + dt L(U2))
// with L(U) = -div F(U) + S(U) (EXPLICIT source per stage, evaluated at the same state as the
// flux: true SSPRK method-of-lines, cf. mf_eval_rhs -- IMEX is NOT supported, rejected upstream).
// (2) Fills (fx, fy) with the EFFECTIVE FLUX of the step    Feff = 1/6 F(U0) + 1/6 F(U1) + 2/3 F(U2)
// which is EXACTLY the transport flux seen by the final state (U_new = U0 - dt div Feff + dt Seff).
// This is the flux the conservative reflux must record (coarse side g.c* and fine side g.f*), hence
// its write into (fx, fy) where the Euler path leaves the single flux F(U0). On INPUT (fx, fy)
// already contain F(U0) (stage 0, computed by the caller before the call). Between stages, the
// ghosts are refreshed by ssprk3_refill_level_ghosts at the SAME frac: the coarse-fine boundary is
// FROZEN over the substep (the levels do not cross their stages, cf. subcycle_level_mp header /
// subcycling). saxpy/lincomb and the RHS functor are device-clean kernels (named functors), no
// extended lambda.
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void ssprk3_advance_level(const Model& m, AmrLevelMP& lv, Real dt, MultiFab& fx, MultiFab& fy,
                          bool recon_prim, int lev, const Box2D& base_dom, Periodicity base_per,
                          const MultiFab* pOld, const MultiFab* pNew, Real frac,
                          bool coarse_replicated, Real pos_floor = Real(0)) {
  const int nc = lv.U.ncomp();
  // Density-role component for the C/F ghost floor (ADC-259), resolved ONCE on the host. pos_floor<=0
  // -> 0 without model introspection (positivity_comp short-circuit) -> bit-identical historical path.
  const int pos_comp = detail::positivity_comp<Model>(pos_floor);
  MultiFab U0 = lv.U;  // starting state t (Shu-Osher convex combinations)
  MultiFab R(lv.U.box_array(), lv.U.dmap(), nc, 0);
  MultiFab Fxs(fx.box_array(), fx.dmap(), nc, 0),
      Fys(fy.box_array(), fy.dmap(), nc, 0);  // stage flux

  // --- stage 0: F(U0) already in (fx, fy), R0 = -div F0 + S(U0) ---
  mf_eval_rhs(m, lv.U, *lv.aux, fx, fy, lv.dx, lv.dy, R);
  saxpy(lv.U, dt, R);                         // lv.U = U1 = U0 + dt R0
  lincomb(fx, Real(1) / 6, fx, Real(0), fx);  // Feff <- 1/6 F0 (pointwise aliasing, safe)
  lincomb(fy, Real(1) / 6, fy, Real(0), fy);

  // --- stage 1: F(U1) ---
  ssprk3_refill_level_ghosts(lv.U, lev, base_dom, base_per, pOld, pNew, frac, coarse_replicated,
                             pos_floor, pos_comp);
  compute_face_fluxes<Limiter, NumericalFlux>(m, lv.U, *lv.aux, Fxs, Fys, lv.dx, lv.dy, recon_prim,
                                              pos_floor);
  device_fence();
  mf_eval_rhs(m, lv.U, *lv.aux, Fxs, Fys, lv.dx, lv.dy, R);  // R1 = -div F1 + S(U1)
  saxpy(lv.U, dt, R);                                        // lv.U = U1 + dt R1
  lincomb(lv.U, Real(3) / 4, U0, Real(1) / 4, lv.U);         // lv.U = U2
  saxpy(fx, Real(1) / 6, Fxs);                               // Feff += 1/6 F1
  saxpy(fy, Real(1) / 6, Fys);

  // --- stage 2: F(U2) ---
  ssprk3_refill_level_ghosts(lv.U, lev, base_dom, base_per, pOld, pNew, frac, coarse_replicated,
                             pos_floor, pos_comp);
  compute_face_fluxes<Limiter, NumericalFlux>(m, lv.U, *lv.aux, Fxs, Fys, lv.dx, lv.dy, recon_prim,
                                              pos_floor);
  device_fence();
  mf_eval_rhs(m, lv.U, *lv.aux, Fxs, Fys, lv.dx, lv.dy, R);  // R2 = -div F2 + S(U2)
  saxpy(lv.U, dt, R);                                        // lv.U = U2 + dt R2
  lincomb(lv.U, Real(1) / 3, U0, Real(2) / 3, lv.U);         // lv.U = U_new (t + dt)
  saxpy(fx, Real(2) / 3, Fxs);                               // Feff += 2/3 F2
  saxpy(fy, Real(2) / 3, Fys);
  device_fence();  // (fx, fy) = Feff and lv.U = U_new consistent for host reads (parentRegs/reflux)
}

template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void subcycle_level_mp(const Model& m, std::vector<AmrLevelMP>& L, int lev, Real dt,
                       const Box2D& base_dom, Periodicity base_per, const MultiFab* pOld,
                       const MultiFab* pNew, Real frac, std::vector<RegMP>* parentRegs,
                       bool coarse_replicated = true, bool recon_prim = false, bool imex = false,
                       const NewtonOptions& nopts = {},
                       AmrTimeMethod tmethod = AmrTimeMethod::kEuler, Real pos_floor = Real(0)) {
  // SSPRK3 + IMEX: combination NOT VALIDATED (the per-stage implicit stiff source under SSP has
  // not been verified), rejected EXPLICITLY rather than run silently. The facade cannot produce it
  // (time.kind is a single selector: "ssprk3" XOR "imex"), defense-in-depth guard here.
  if (tmethod == AmrTimeMethod::kSsprk3 && imex)
    throw std::runtime_error(
        "subcycle_level_mp: SSPRK3 + IMEX unsupported (combination not validated); use "
        "time='ssprk3' (explicit source per stage) or time='imex' (forward Euler + implicit "
        "source)");
  const SubcyclingSchedule sched(2);
  const int nc = L[lev].U.ncomp();
  // Density-role component for the C/F ghost floor (ADC-259), resolved ONCE on the host. pos_floor<=0
  // -> 0 without model introspection (positivity_comp short-circuit) -> bit-identical historical path.
  const int pos_comp = detail::positivity_comp<Model>(pos_floor);
  AmrLevelMP& lv = L[lev];
  const int np = lv.U.local_size();
  const bool ssprk3 = (tmethod == AmrTimeMethod::kSsprk3);

  if (lev == 0) {
    fill_boundary(lv.U, base_dom, base_per);
  } else {
    // parent (level lev-1) REPLICATED only if it is level 0 (lev == 1); otherwise distributed ->
    // FillPatch by parallel_copy.
    mf_fill_fine_ghosts_mb(lv.U, *pOld, *pNew, frac,
                           /*replicated_parent=*/(lev == 1) && coarse_replicated, pos_floor,
                           pos_comp);
    const Box2D fdom = Box2D::from_extents(base_dom.nx() << lev, base_dom.ny() << lev);
    fill_boundary(lv.U, fdom, Periodicity{false, false});  // fine-fine halos
  }

  // face-box per GLOBAL box + same dmap (cf. amr_step_2level_multipatch): BoxArray and
  // DistributionMapping of the same size under MPI, fx.fab(li) <-> lv.U.fab(li). Identical in
  // serial (local == global).
  std::vector<Box2D> fxb, fyb;
  for (int g = 0; g < lv.U.box_array().size(); ++g) {
    fxb.push_back(xface_box(lv.U.box_array()[g]));
    fyb.push_back(yface_box(lv.U.box_array()[g]));
  }
  MultiFab fx(BoxArray(std::move(fxb)), lv.U.dmap(), nc, 0);
  MultiFab fy(BoxArray(std::move(fyb)), lv.U.dmap(), nc, 0);
  compute_face_fluxes<Limiter, NumericalFlux>(m, lv.U, *lv.aux, fx, fy, lv.dx, lv.dy, recon_prim,
                                              pos_floor);
  device_fence();

  // SSPRK3: we FIRST advance lv.U from t to t+dt (3 stages) AND replace (fx, fy) -- which contain
  // the single flux F(U0) of the Euler path -- with the EFFECTIVE FLUX Feff = 1/6 F0 + 1/6 F1 +
  // 2/3 F2. All the rest of the function (parent register, child registers, saved coarse flux,
  // reflux) reads (fx, fy) and the advanced state EXACTLY as in Euler: recording Feff (instead of
  // F0) makes the reflux conservative for the full SSP step (the coarse side g.c* = coarse Feff,
  // the fine side g.f* = sum of the subcycled fine Feff, and the correction -(g.f - g.c*dt)/dx
  // correctly replaces the effective coarse flux with the effective fine flux). The starting state
  // is saved BEFORE the advance for the temporal interpolation of the children (coarse role). In
  // Euler (ssprk3 == false) this block is skipped and the advance stays the original one, in place
  // below -> strictly bit-identical.
  const bool is_leaf = (lev + 1 >= static_cast<int>(L.size()));
  MultiFab ssp_U_old;  // state t (pre-advance capture); filled only for SSPRK3 + coarse role
  if (ssprk3) {
    if (!is_leaf)
      ssp_U_old = lv.U;  // the children interpolate between this state (t) and advanced lv.U (t+dt)
    ssprk3_advance_level<Limiter, NumericalFlux>(m, lv, dt, fx, fy, recon_prim, lev, base_dom,
                                                 base_per, pOld, pNew, frac, coarse_replicated,
                                                 pos_floor);
  }

  if (parentRegs) {  // FINE role: fine fluxes of THIS level into the parent register
    for (int li = 0; li < np; ++li) {
      RegMP& g = (*parentRegs)[li];
      const ConstArray4 FX = fx.fab(li).const_array(), FY = fy.fab(li).const_array();
      for (int J = g.J0; J <= g.J1; ++J)
        for (int k = 0; k < nc; ++k) {
          g.fL[(J - g.J0) * nc + k] +=
              Real(0.5) * (FX(2 * g.I0, 2 * J, k) + FX(2 * g.I0, 2 * J + 1, k)) * dt;
          g.fR[(J - g.J0) * nc + k] +=
              Real(0.5) * (FX(2 * g.I1 + 2, 2 * J, k) + FX(2 * g.I1 + 2, 2 * J + 1, k)) * dt;
        }
      for (int I = g.I0; I <= g.I1; ++I)
        for (int k = 0; k < nc; ++k) {
          g.fB[(I - g.I0) * nc + k] +=
              Real(0.5) * (FY(2 * I, 2 * g.J0, k) + FY(2 * I + 1, 2 * g.J0, k)) * dt;
          g.fT[(I - g.I0) * nc + k] +=
              Real(0.5) * (FY(2 * I, 2 * g.J1 + 2, k) + FY(2 * I + 1, 2 * g.J1 + 2, k)) * dt;
        }
    }
  }

  if (is_leaf) {    // leaf
    if (!ssprk3) {  // forward Euler (legacy path); SSPRK3 already advanced lv.U above
      mf_advance_faces(lv.U, fx, fy, lv.dx, lv.dy, dt);
      mf_apply_source_treatment(m, lv.U, *lv.aux, dt, imex,
                                nopts);  // explicit or IMEX source (Newton options)
    }
    return;
  }

  // COARSE role for lev+1: coarse-fine interface (GLOBAL MPI-safe coverage + bordering reflux
  // routing) + registers + saved coarse flux.
  const int NX = base_dom.nx() << lev, NY = base_dom.ny() << lev;
  const CoarseFineInterface cfi(Box2D{{0, 0}, {NX - 1, NY - 1}}, L[lev + 1].U.box_array());
  auto covered = [&](int I, int J) { return cfi.covered(I, J); };

  // Distributed point 2: the coarse flux fx/fy lives on the PARENT dmap (lv.U), so fx.fab is on
  // the rank owning the parent box, not necessarily the child's. Two cases:
  //  - REPLICATED parent (lev == 0): fx/fy fully local, sampled directly via mf_find_box (always
  //    found); no collective (parallel_copy would violate parent replication);
  //  - DISTRIBUTED parent (lev >= 1): the needed coarse fluxes are brought onto a child-coarsen
  //    FACE grid (child dmap) by parallel_copy, each child then reads locally.
  // Level 0 is REPLICATED only if coarse_replicated: at de-replication (distributed multi-box
  // coarse level), it becomes DISTRIBUTED like the fine levels, and mf_find_box(lv.U, I, J) would
  // return -1 for a bordering coarse cell owned by a REMOTE rank (-> fab(-1), segfault). We then
  // route to the parallel_copy path (per-child coarse footprint), MPI-correct.
  const bool replicated_parent = (lev == 0) && coarse_replicated;
  const BoxArray cba =
      coarsen(L[lev + 1].U.box_array(), kAmrRefRatio);  // per-child coarse footprint
  MultiFab cfx, cfy;
  if (!replicated_parent) {
    std::vector<Box2D> cfxb, cfyb;
    for (int g = 0; g < cba.size(); ++g) {
      cfxb.push_back(xface_box(cba[g]));
      cfyb.push_back(yface_box(cba[g]));
    }
    cfx = MultiFab(BoxArray(std::move(cfxb)), L[lev + 1].U.dmap(), nc, 0);
    cfy = MultiFab(BoxArray(std::move(cfyb)), L[lev + 1].U.dmap(), nc, 0);
    parallel_copy(cfx, fx);  // coarse face fluxes -> local child-coarsen grid
    parallel_copy(cfy, fy);
  }
  device_fence();

  std::vector<RegMP> regs(L[lev + 1].U.local_size());
  for (int lc = 0; lc < L[lev + 1].U.local_size(); ++lc) {
    const PatchRange pr(L[lev + 1].U.box(lc));
    RegMP& g = regs[lc];
    g.I0 = pr.I0;
    g.I1 = pr.I1;
    g.J0 = pr.J0;
    g.J1 = pr.J1;
    const int nJ = g.J1 - g.J0 + 1, nI = g.I1 - g.I0 + 1;
    g.cL.assign(nJ * nc, 0);
    g.cR.assign(nJ * nc, 0);
    g.cB.assign(nI * nc, 0);
    g.cT.assign(nI * nc, 0);
    g.fL.assign(nJ * nc, 0);
    g.fR.assign(nJ * nc, 0);
    g.fB.assign(nI * nc, 0);
    g.fT.assign(nI * nc, 0);
    if (replicated_parent) {
      for (int J = g.J0; J <= g.J1; ++J) {
        const int bL = mf_find_box(lv.U, g.I0, J), bR = mf_find_box(lv.U, g.I1, J);
        // replicated-parent invariant: parent fully local (cf. above), mf_find_box always finds
        // the box; a -1 would index fab(-1) (segfault). The distributed case goes through the else.
        assert(bL >= 0 && bR >= 0 &&
               "subcycle_level_mp: replicated-parent invariant violated (coarse box x not found)");
        const ConstArray4 FXL = fx.fab(bL).const_array(), FXR = fx.fab(bR).const_array();
        for (int k = 0; k < nc; ++k) {
          g.cL[(J - g.J0) * nc + k] = FXL(g.I0, J, k);
          g.cR[(J - g.J0) * nc + k] = FXR(g.I1 + 1, J, k);
        }
      }
      for (int I = g.I0; I <= g.I1; ++I) {
        const int bB = mf_find_box(lv.U, I, g.J0), bT = mf_find_box(lv.U, I, g.J1);
        // same replicated-parent invariant as above (y faces): coarse box always found.
        assert(bB >= 0 && bT >= 0 &&
               "subcycle_level_mp: replicated-parent invariant violated (coarse box y not found)");
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

  // state t for the temporal interpolation of the children. SSPRK3: lv.U is ALREADY advanced (the
  // advance happened above, in ssprk3_advance_level), the state t is the pre-advance copy
  // ssp_U_old; Euler: lv.U is still the state t here (the advance is just below), so U_old = lv.U
  // (legacy copy).
  MultiFab U_old = ssprk3 ? ssp_U_old : lv.U;
  if (!ssprk3) {  // forward Euler (legacy path); SSPRK3 already advanced lv.U
    mf_advance_faces(lv.U, fx, fy, lv.dx, lv.dy, dt);
    mf_apply_source_treatment(m, lv.U, *lv.aux, dt, imex,
                              nopts);  // explicit or IMEX source (Newton options)
  }
  for (int s = 0; s < sched.count();
       ++s)  // each fine substep = one full SSP step (tmethod propagated)
    subcycle_level_mp<Limiter, NumericalFlux>(
        m, L, lev + 1, sched.dt_sub(dt), base_dom, base_per, &U_old, &lv.U, sched.frac(s), &regs,
        coarse_replicated, recon_prim, imex, nopts, tmethod, pos_floor);
  mf_average_down_mb(L[lev + 1].U, lv.U);  // distributed point 3 (parallel_copy)

  // Distributed point 4: coverage-aware reflux. The bordering coarse cell may belong to a REMOTE
  // parent box. For each LOCAL child, we deposit the correction (cL/fL already local after
  // parallel_copy) into a GLOBAL-indexed coarse buffer, all_reduce -> each rank has the full
  // register, then each rank applies to ITS local parent boxes (the parent being distributed,
  // each cell has only one owner: no double counting). In serial all_reduce is the identity and
  // application is direct -> bit-for-bit identical.
  device_fence();
  // register restricted to the INTERFACE: bounding box of the fine footprints (coarsen of level
  // lev+1), grown by 1, clamped to level lev. all_reduce O(interface), bit-identical.
  const Box2D fpcn = coarsen(L[lev + 1].U.box_array(), kAmrRefRatio).bounding_box();
  const Box2D rbox{{std::max(fpcn.lo[0] - 1, 0), std::max(fpcn.lo[1] - 1, 0)},
                   {std::min(fpcn.hi[0] + 1, NX - 1), std::min(fpcn.hi[1] + 1, NY - 1)}};
  FluxRegister ref(rbox, nc);  // N-level reflux (interface)
  for (int lc = 0; lc < static_cast<int>(regs.size()); ++lc)
    cfi.route_reflux(regs[lc], lv.dx, lv.dy, dt, ref, nc);  // coverage-aware bordering reflux
  ref.gather();
  for (int pb = 0; pb < lv.U.local_size(); ++pb) {  // application to the local parent boxes
    Array4 c = lv.U.fab(pb).array();
    const Box2D pbx = lv.U.box(pb);
    for (int J = pbx.lo[1]; J <= pbx.hi[1]; ++J)
      for (int I = pbx.lo[0]; I <= pbx.hi[0]; ++I) {
        if (!ref.in(I, J))
          continue;  // outside interface: zero reflux
        for (int k = 0; k < nc; ++k)
          c(I, J, k) += ref.at(I, J, k);
      }
  }
}

// Driver: one dt step of the N-level multi-patch hierarchy (level 0 = coarse).
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void amr_step_multilevel_multipatch(const Model& m, std::vector<AmrLevelMP>& L, const Box2D& dom,
                                    Real dt, Periodicity per = Periodicity{true, true},
                                    bool coarse_replicated = true, bool recon_prim = false,
                                    bool imex = false, const NewtonOptions& nopts = {},
                                    AmrTimeMethod tmethod = AmrTimeMethod::kEuler,
                                    Real pos_floor = Real(0)) {
  subcycle_level_mp<Limiter, NumericalFlux>(m, L, 0, dt, dom, per, nullptr, nullptr, Real(0),
                                            nullptr, coarse_replicated, recon_prim, imex, nopts,
                                            tmethod, pos_floor);
}

}  // namespace detail

}  // namespace adc
