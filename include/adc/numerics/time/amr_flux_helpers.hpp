#pragma once

#include <adc/mesh/index/box2d.hpp>
#include <adc/amr/hierarchy/refinement_ratio.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/boundary/fill_boundary.hpp>
#include <adc/mesh/execution/for_each.hpp>
#include <adc/mesh/storage/multifab.hpp>
#include <adc/mesh/layout/refinement.hpp>                 // coarsen_index
#include <adc/numerics/spatial_operator.hpp>       // compute_face_fluxes, xface_box, yface_box
#include <adc/numerics/time/implicit_stepper.hpp>  // backward_euler_source (IMEX implicit step)

#include <vector>

/// @file
/// @brief Basic MultiFab building blocks of an AMR step: AmrTimeMethod enum, device-clean functors and
///        advance helpers (flux divergence, explicit/IMEX source, 2x2 average_down,
///        space-time coarse-fine ghosts mono-box), shared by the whole subcycling path.
///
/// Layer: `include/adc/numerics/time`.
/// Role: provide the kernels reused by amr_level / amr_patch_range / amr_subcycling.
///        mf_advance_faces (U -= dt div F), mf_apply_source (U += dt S, forward Euler),
///        mf_apply_source_treatment (explicit OR IMEX backward-Euler per runtime flag),
///        mf_eval_rhs (R = -div F + S at the same state, for SSPRK3 stages), mf_average_down,
///        fill_cf_ghost_cell / mf_fill_fine_ghosts_t.
///
/// Invariants:
/// - kernels = NAMED functors (AmrSspRhsKernel, AmrAdvanceFacesKernel, ...) and not lambdas:
///   a first instantiation from an external loader TU or an extended lambda would
///   make nvcc choke;
/// - the source is CELL-LOCAL (no face flux): it does not enter the reflux, so
///   the IMEX split does not touch conservation at coarse-fine interfaces;
/// - mf_apply_source_treatment with nopts={} (default) reproduces the legacy 2-iter
///   Newton call -> bit-identical;
/// - the device paths read/write unified memory: device_fence() before host read.

namespace adc {

static_assert(kAmrRefRatio == 2, "ratio-2-structural kernels below assume kAmrRefRatio == 2");

// Time method of an AMR step (Berger-Oliger subcycling). kEuler (DEFAULT) = forward Euler
// advance at each substep (legacy path, strictly bit-identical); kSsprk3 = SSPRK3
// (Shu-Osher, 3 stages, order 3) with per-stage reflux (convex effective flux). The enum is passed
// BY VALUE (POD) along the advance_amr -> subcycle_level_mp path; the flat ABI of the .so loader
// carries it as an integer (AmrBuildParams::time_method), 0 == kEuler.
enum class AmrTimeMethod : int { kEuler = 0, kSsprk3 = 1 };

// Device-clean NAMED functor (same recipe as mf_arith.hpp: a first instantiation possible
// from an external loader TU, or an extended lambda makes nvcc choke) of the method-of-lines RHS
// at ONE AMR level: R = -div(Fx,Fy) + S(U, aux), evaluated at ONE SAME state. It is the divergence of
// mf_advance_faces (opposite sign, without dt) FUSED with the source of mf_apply_source. Used
// ONLY by the SSPRK3 stages (mf_eval_rhs), where L(U) = -div F + S must be taken at the same
// stage state (true method-of-lines SSPRK), unlike the transport-then-source splitting of the
// Euler path. Without a source (model with S == 0) R reduces to -div F.
template <class Model>
struct AmrSspRhsKernel {
  Model m;
  ConstArray4 u, ax, fx, fy;
  Array4 R;
  Real dx, dy;
  ADC_HD void operator()(int i, int j) const {
    const auto S = m.source(load_state<Model>(u, i, j), load_aux<aux_comps<Model>()>(ax, i, j));
    for (int c = 0; c < Model::n_vars; ++c)
      R(i, j, c) =
          -((fx(i + 1, j, c) - fx(i, j, c)) / dx + (fy(i, j + 1, c) - fy(i, j, c)) / dy) + S[c];
  }
};

// R <- -div(Fx,Fy) + S(U, aux) on the valid cells (method-of-lines RHS at ONE level, evaluated
// at the state U). Fused "combine" of mf_advance_faces + mf_apply_source for the SSPRK3 stages (the
// stage flux Fx/Fy is assumed already computed by compute_face_fluxes at the state U).
template <class Model>
inline void mf_eval_rhs(const Model& m, const MultiFab& U, const MultiFab& aux, const MultiFab& Fx,
                        const MultiFab& Fy, Real dx, Real dy, MultiFab& R) {
  for (int li = 0; li < U.local_size(); ++li)
    for_each_cell(U.box(li),
                  AmrSspRhsKernel<Model>{m, U.fab(li).const_array(), aux.fab(li).const_array(),
                                         Fx.fab(li).const_array(), Fy.fab(li).const_array(),
                                         R.fab(li).array(), dx, dy});
}

/// Device-clean NAMED functor: U <- U - dt div(Fx,Fy) on a valid cell.
struct AmrAdvanceFacesKernel {
  Array4 u;
  ConstArray4 fx, fy;
  Real dx, dy, dt;
  int nc;
  ADC_HD void operator()(int i, int j) const {
    for (int c = 0; c < nc; ++c)
      u(i, j, c) -=
          dt * ((fx(i + 1, j, c) - fx(i, j, c)) / dx + (fy(i, j + 1, c) - fy(i, j, c)) / dy);
  }
};

// U <- U - dt div(Fx,Fy) on the valid cells (GPU via for_each_cell).
inline void mf_advance_faces(MultiFab& U, const MultiFab& Fx, const MultiFab& Fy, Real dx, Real dy,
                             Real dt) {
  const int nc = U.ncomp();
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 u = U.fab(li).array();
    const ConstArray4 fx = Fx.fab(li).const_array(), fy = Fy.fab(li).const_array();
    for_each_cell(U.box(li), AmrAdvanceFacesKernel{u, fx, fy, dx, dy, dt, nc});
  }
}

// U <- U + dt S(U, aux) on the valid cells: source term applied with forward Euler
// at each AMR substep (cell-local, no reflux). Without it the AMR path
// (compute_face_fluxes -> divergence) would ignore model.source. For a model with a null
// source (pure scalar transport) this adds dt*0: bit-identical. DIFFUSION, in contrast, is carried
// by compute_face_fluxes as a Fickian face FLUX (-nu grad u), thus seen by the
// reflux and conservative at coarse-fine interfaces: it is NOT a local source.
/// Device-clean NAMED functor (template Model, see AmrSspRhsKernel): U <- U + dt S(U, aux)
/// on a valid cell.
template <class Model>
struct AmrApplySourceKernel {
  Model m;
  Array4 u;
  ConstArray4 uc, ax;
  Real dt;
  ADC_HD void operator()(int i, int j) const {
    const auto S = m.source(load_state<Model>(uc, i, j), load_aux<aux_comps<Model>()>(ax, i, j));
    for (int c = 0; c < Model::n_vars; ++c)
      u(i, j, c) += dt * S[c];
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

// Temporal treatment of the SOURCE at an AMR substep, after the transport advance
// (mf_advance_faces, already without source since compute_face_fluxes only carries model.flux):
//   - EXPLICIT (imex == false, DEFAULT): forward Euler, U += dt S(U, aux) -- the legacy
//     mf_apply_source call, thus bit-identical to the existing path.
//   - IMEX (imex == true): stiff IMPLICIT source, W = U + dt S(W, aux) solved IN PLACE by
//     backward_euler_source (local Newton, finite-difference Jacobian, NAMED device functor
//     BackwardEulerSourceKernel). It is the AMR counterpart of the System IMEX advance
//     (block_builder.hpp::AdvanceImex): same explicit half-step (transport is carried by the
//     conservative reflux) + same implicit step on the source. The source remaining CELL-LOCAL
//     (no face flux), it does NOT enter the reflux registers: the implicit split thus does
//     not touch conservation at coarse-fine interfaces. The CHOICE is a runtime flag
//     (no lambda injected into the device path): it selects two HOST functions, each
//     launching its own named-functor kernel.
//
// NEWTON OPTIONS (@p nopts): drive the local Newton of the implicit source (iteration budget,
// tolerances, fd_eps, damping, fail_policy). DEFAULT {} = legacy constants (2 iters, 1e-7, ...)
// -> path (2a) bit-identical to the old call backward_euler_source(m, aux, U, dt). The AMR mono-block
// (AmrCouplerMP::step) threads them from AmrSystem (wave 3 -> mono-block options wired). The partial
// IMEX mask is NOT carried by this path (mono-block coupler = full backward-Euler): so the
// default mask (inactive) is passed. No diagnostics report here (report == nullptr implicit).
template <class Model>
inline void mf_apply_source_treatment(const Model& m, MultiFab& U, const MultiFab& aux, Real dt,
                                      bool imex, const NewtonOptions& nopts = {}) {
  if (imex)
    // OPTIONS form (Newton driven by nopts), inactive mask, no report. Default nopts={} =>
    // identical to the legacy form with fixed iters (2), thus bit-identical as long as nopts is default.
    backward_euler_source(m, aux, U, dt, nopts, ImplicitMask<Model::n_vars>{});
  else
    mf_apply_source(m, U, aux, dt);  // legacy forward Euler (bit-identical)
}

/// Device-clean NAMED functor: 2x2 average fine -> coarse on a coarse cell.
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

// average fine -> coarse (ratio 2) on the covered region (coarse coords).
inline void mf_average_down(const MultiFab& Uf, MultiFab& Uc, int CI0, int CI1, int CJ0, int CJ1) {
  const int nc = Uc.ncomp();
  const ConstArray4 f = Uf.fab(0).const_array();
  Array4 c = Uc.fab(0).array();
  for_each_cell(Box2D{{CI0, CJ0}, {CI1, CJ1}}, AmrAverageDownKernel{f, c, nc});
}

// First-level coarse-fine helper (review, point ghosts): fills ONE fine ghost
// cell (i,j) by spatial interpolation (piecewise constant: covering coarse cell)
// + time (linear between the old/new parent state). frac = temporal position of the
// substep within the parent step. Centralizes the arithmetic shared by mf_fill_fine_ghosts_t
// (mono-box), mf_fill_fine_ghosts_multi (multi-box) and mf_fill_fine_ghosts_mb (multi-level):
// a single formula (1-frac)*co + frac*cn, bit-identical to the three previous bodies.
inline void fill_cf_ghost_cell(Array4 f, const ConstArray4& co, const ConstArray4& cn, int i, int j,
                               int nc, Real frac, Real pos_floor = Real(0), int pos_comp = 0) {
  const int ci = coarsen_index(i, kAmrRefRatio), cj = coarsen_index(j, kAmrRefRatio);
  for (int k = 0; k < nc; ++k)
    f(i, j, k) = (1 - frac) * co(ci, cj, k) + frac * cn(ci, cj, k);
  // Zhang-Shu positivity floor on the C/F fine GHOST MEAN (ADC-259): clamp the Density role only
  // (pos_comp, resolved on the host by the caller via positivity_comp<Model>) to >= pos_floor. The
  // refined-patch C/F interface is the highest-risk site: reconstruct_pp's order-1 fallback brings a
  // sub-floor face back to its SOURCE-CELL mean, and at a fine cell bordering the interface that
  // source is a ghost; without this clamp the fallback target itself could be sub-floor (the coarse
  // mean is not floored), defeating the guarantee. Momenta/energy stay interpolated -> the ghost
  // velocity m/rho only DROPS at quasi-vacuum (bounded, mirror of the single-block mean fallback).
  // pos_floor <= 0 short-circuits (bit-identical). Ghost cells are never averaged-down nor summed in
  // mass, so the clamp is conservation-safe (cf. ADC-259 design: average-down immunity + the reflux
  // coarse-side register reads a separate fab, so the two-sided telescoping is preserved exactly).
  if (pos_floor > Real(0) && f(i, j, pos_comp) < pos_floor)
    f(i, j, pos_comp) = pos_floor;
}

// fine ghosts = spatial interp (piecewise constant) + time (linear) from the
// old/new coarse. frac = temporal position of the substep within the coarse step.
inline void mf_fill_fine_ghosts_t(MultiFab& Uf, const MultiFab& Uc_old, const MultiFab& Uc_new,
                                  Real frac, Real pos_floor = Real(0), int pos_comp = 0) {
  device_fence();  // host read/write on unified memory
  const int nc = Uf.ncomp();
  Array4 f = Uf.fab(0).array();
  const ConstArray4 co = Uc_old.fab(0).const_array();
  const ConstArray4 cn = Uc_new.fab(0).const_array();
  const Box2D v = Uf.box(0), g = Uf.fab(0).grown_box();
  for (int j = g.lo[1]; j <= g.hi[1]; ++j)
    for (int i = g.lo[0]; i <= g.hi[0]; ++i)
      if (!v.contains(i, j))
        fill_cf_ghost_cell(f, co, cn, i, j, nc, frac, pos_floor, pos_comp);
}

}  // namespace adc
