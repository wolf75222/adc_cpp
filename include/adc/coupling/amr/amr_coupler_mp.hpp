#pragma once

#include <adc/core/types.hpp>
#include <adc/amr/refinement_ratio.hpp>
#include <adc/coupling/amr/amr_diagnostics.hpp>     // amr_mass, amr_max_drift_speed
#include <adc/coupling/amr/amr_level_storage.hpp>   // AmrLevelStack
#include <adc/coupling/amr/amr_regrid_coupler.hpp>  // amr_regrid_finest (Berger-Rigoutsos)
#include <adc/coupling/single/coupler.hpp>  // detail::coupler_eval_rhs (f = model.elliptic_rhs(U))
#include <adc/numerics/elliptic/mg/composite_fac_poisson.hpp>  // COMPOSITE FAC 2-level Poisson solver (opt-in)
#include <adc/numerics/elliptic/interface/elliptic_solver.hpp>
#include <adc/numerics/elliptic/mg/geometric_mg.hpp>
#include <adc/numerics/time/amr_reflux_mf.hpp>  // AmrLevelMP, amr_step_multilevel_multipatch, mf_*_mb
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/mesh/refinement.hpp>  // coarsen_index
#include <adc/parallel/comm.hpp>    // all_reduce_sum / all_reduce_max (distributed mass/drift)

#include <algorithm>   // std::max
#include <cmath>       // std::hypot
#include <cstddef>     // std::size_t
#include <functional>  // std::function (conducting-wall predicate passed to the MG)
#include <map>  // named_aux_: model-named aux fields (comp -> coarse field), re-applied by compute_aux
#include <stdexcept>  // std::runtime_error (density size guard)
#include <utility>    // std::pair, std::move
#include <vector>

/// @file
/// @brief AmrCouplerMP: MULTI-PATCH E x B AMR coupler (coarse Poisson -> aux = grad phi ->
///        fine injection -> conservative AMR step), multi-box per-level hierarchy.
///
/// Same role as AmrCoupler but each level is multi-box (std::vector<AmrLevelMP> held by an
/// AmrLevelStack) and integration goes through amr_step_multilevel_multipatch (coverage-aware reflux).
/// regrid() rebuilds the fine level on the fly via Berger-Rigoutsos. Level 0 = single box for the
/// Poisson. The class only ORDERS the operations (hierarchy stored in AmrLevelStack,
/// regrid in amr_regrid_coupler.hpp, diagnostics in amr_diagnostics.hpp). INVARIANT: reduces
/// BIT FOR BIT to AmrCoupler when each level has a single box (validation guard). Level-0
/// ownership policy via replicated_coarse (replicated vs distributed, equivalence proven bit for bit).
/// The detail:: are the DISTRIBUTED primitives (aux injection, density write/read, layout).

namespace adc {

namespace detail {
// Coupling primitive (not policy): piecewise-constant injection of aux from
// multi-box parent -> multi-box child (valid + ghosts). DISTRIBUTED, same scheme as
// mf_fill_fine_ghosts_mb. Two parent cases:
//  - REPLICATED (level 0, replicated_parent=true): parent fully local on each rank,
//    direct read via mf_find_box. This is the replicated coarse (per-rank dmap): parallel_copy
//    would violate the replicated-metadata assumption. Bit-identical path to the historical one.
//  - DISTRIBUTED (intermediate, replicated_parent=false): the parent may be on another rank,
//    we bring its valid regions onto a LOCAL child-coarsen grid via parallel_copy, then
//    inject. A child cell outside the parent coverage (GLOBAL box_array) is left
//    intact, like the replicated path (which skipped it via mf_find_box < 0).
// In serial both paths are identical (parent local everywhere, parallel_copy = memory copy).
inline void coupler_inject_aux_mb(const MultiFab& parent, MultiFab& child,
                                  bool replicated_parent = true) {
  const int nc = child.ncomp();
  if (replicated_parent) {
    device_fence();
    for (int lc = 0; lc < child.local_size(); ++lc) {
      Array4 c = child.fab(lc).array();
      const Box2D g = child.fab(lc).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
          const int ci = coarsen_index(i, kAmrRefRatio), cj = coarsen_index(j, kAmrRefRatio);
          const int pb = mf_find_box(parent, ci, cj);
          if (pb < 0)
            continue;
          const ConstArray4 pp = parent.fab(pb).const_array();
          for (int k = 0; k < nc; ++k)
            c(i, j, k) = pp(ci, cj, k);
        }
    }
    return;
  }
  const BoxArray& pba = parent.box_array();  // GLOBAL: rank-independent coverage
  auto covered = [&](int ci, int cj) {
    for (int b = 0; b < pba.size(); ++b)
      if (pba[b].contains(ci, cj))
        return true;
    return false;
  };
  const BoxArray ccoarse = coarsen_grown(child.box_array(), child.n_grow(), kAmrRefRatio);
  MultiFab Pc(ccoarse, child.dmap(), parent.ncomp(), 0);
  parallel_copy(Pc, parent);  // parent regions (from any rank) -> local grid
  device_fence();
  for (int lc = 0; lc < child.local_size(); ++lc) {
    Array4 c = child.fab(lc).array();
    const ConstArray4 pp = Pc.fab(lc).const_array();
    const Box2D g = child.fab(lc).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
        const int ci = coarsen_index(i, kAmrRefRatio), cj = coarsen_index(j, kAmrRefRatio);
        if (!covered(ci, cj))
          continue;  // outside coverage -> keep the child value
        for (int k = 0; k < nc; ++k)
          c(i, j, k) = pp(ci, cj, k);
      }
  }
}
// Writes an initial density (component 0, n*n row-major in GLOBAL indices) on the coarse
// level, MULTI-BOX and DISTRIBUTION-AWARE: each rank touches only its LOCAL fabs and reads
// rho at the cell GLOBAL index (i,j). For Euler (ncomp 4) it also sets zero momentum
// + thermal energy r/(gamma-1); ncomp 1 touches only density. Replicated mono-box:
// a single fab covering the domain, global == local indices -> bit-identical to the historical
// direct write. Distributed multi-box: each local box reads its window of rho.
inline void coupler_write_coarse(MultiFab& U, const std::vector<double>& rho, int n, int ncomp,
                                 double gamma) {
  if (static_cast<int>(rho.size()) != n * n)
    throw std::runtime_error("AMR coupler: initial density of size != n*n");
  const Real gm1 = Real(gamma) - Real(1);
  device_fence();
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 u = U.fab(li).array();
    const Box2D v = U.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        const Real r = rho[static_cast<std::size_t>(j) * n + i];
        u(i, j, 0) = r;
        if (ncomp >= 3) {
          u(i, j, 1) = 0;
          u(i, j, 2) = 0;
        }
        if (ncomp == 4)
          u(i, j, 3) = r / gm1;
      }
  }
}

// Writes the FULL INITIAL CONSERVATIVE STATE (all components) on the coarse level from a
// flat component-major field @p state (c*n*n + j*n + i), of size ncomp*n*n. Counterpart of
// coupler_write_coarse for the multi-component seed: same box traversal (replicated mono-box
// AND distributed multi-box, GLOBAL indices (i,j)), only the per-cell write differs -- here we copy
// the ncomp components positionally (no density/momentum/energy wiring; the caller already provides
// the conservative, e.g. [rho, rho*u, rho*v]). gamma omitted (no energy derived). Index computed
// in std::size_t (no int overflow at large n, unlike the int validation of
// coupler_write_coarse). Used for the drift seed (set_conservative_state).
inline void coupler_write_coarse_state(MultiFab& U, const std::vector<double>& state, int n,
                                       int ncomp) {
  const std::size_t nn = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  if (state.size() != nn * static_cast<std::size_t>(ncomp))
    throw std::runtime_error(
        "AMR coupler: initial state of size != ncomp*n*n (full conservative "
        "state; ncomp == model n_vars)");
  device_fence();
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 u = U.fab(li).array();
    const Box2D v = U.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        for (int c = 0; c < ncomp; ++c)
          u(i, j, c) = state[static_cast<std::size_t>(c) * nn +
                             static_cast<std::size_t>(j) * static_cast<std::size_t>(n) +
                             static_cast<std::size_t>(i)];
  }
}

// Reads the coarse density (component 0) into a GLOBAL n*n row-major field, MULTI-BOX and
// DISTRIBUTION-AWARE. Each rank writes its local cells into an n*n buffer initialized to 0
// then, if distributed, all_reduce_sum_inplace recomposes the full field on ALL ranks (the
// boxes are disjoint -> the cross-rank sum reconstructs the field exactly). Replicated mono-box:
// a single fab covers everything, the buffer is already complete, all_reduce would be the identity
// -> we avoid it (bit-identical to the historical direct read fab(0)).
inline std::vector<double> coupler_read_coarse(const MultiFab& U, int n, bool replicated) {
  device_fence();
  std::vector<double> out(static_cast<std::size_t>(n) * n, 0.0);
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const Box2D v = U.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        out[static_cast<std::size_t>(j) * n + i] = u(i, j, 0);
  }
  if (!replicated)
    all_reduce_sum_inplace(out.data(), static_cast<int>(out.size()));
  return out;
}

// Reads the coarse-level potential phi (component 0 of aux(0), written by compute_aux after the
// Poisson solve) into a GLOBAL n*n row-major field, MULTI-BOX and DISTRIBUTION-AWARE. aux(0) shares
// EXACTLY the layout of the coarse U (same BoxArray + DistributionMapping, cf. amr_level_storage:
// aux_[0] is built on U.box_array()/U.dmap()), so the recomposition is identical to
// coupler_read_coarse: local n*n buffer, all_reduce_sum if distributed (disjoint boxes -> exact
// sum), avoided in replicated mono-box (field already complete). PRECONDITION: update()/compute_aux
// has run at least once (otherwise aux(0) is 0). Strict counterpart of coupler_read_coarse for phi.
inline std::vector<double> coupler_read_coarse_phi(const MultiFab& aux0, int n, bool replicated) {
  device_fence();
  std::vector<double> out(static_cast<std::size_t>(n) * n, 0.0);
  for (int li = 0; li < aux0.local_size(); ++li) {
    const ConstArray4 a = aux0.fab(li).const_array();
    const Box2D v = aux0.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        out[static_cast<std::size_t>(j) * n + i] = a(i, j, 0);
  }
  if (!replicated)
    all_reduce_sum_inplace(out.data(), static_cast<int>(out.size()));
  return out;
}

// Injects the coarse into the valid cells of a fine patch (piecewise constant, ratio 2),
// MULTI-BOX and DISTRIBUTION-AWARE. Makes the hierarchy consistent before the first sync_down (the
// seed patch is at 0). Replicated mono-box: coarse fully local, direct read via
// mf_find_box (always found); no collective -> bit-identical to the historical fab(0).
// Distributed multi-box: we bring the needed coarse regions onto a LOCAL child-coarsen grid
// via parallel_copy (same scheme as coupler_inject_aux_mb), then inject.
inline void coupler_inject_coarse_to_fine_mb(const MultiFab& Uc, MultiFab& Uf, bool replicated) {
  const int nc = Uf.ncomp();
  if (replicated) {
    device_fence();
    for (int li = 0; li < Uf.local_size(); ++li) {
      Array4 f = Uf.fab(li).array();
      const Box2D v = Uf.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
          const int ci = coarsen_index(i, kAmrRefRatio), cj = coarsen_index(j, kAmrRefRatio);
          const int pb = mf_find_box(Uc, ci, cj);
          if (pb < 0)
            continue;
          const ConstArray4 c = Uc.fab(pb).const_array();
          for (int k = 0; k < nc; ++k)
            f(i, j, k) = c(ci, cj, k);
        }
    }
    return;
  }
  const BoxArray ccoarse = coarsen(Uf.box_array(), kAmrRefRatio);  // coarse footprint (valid cells)
  MultiFab Pc(ccoarse, Uf.dmap(), Uc.ncomp(), 0);
  parallel_copy(Pc, Uc);  // coarse regions (from any rank) -> local grid
  device_fence();
  for (int li = 0; li < Uf.local_size(); ++li) {
    Array4 f = Uf.fab(li).array();
    const ConstArray4 c = Pc.fab(li).const_array();
    const Box2D v = Uf.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        const int ci = coarsen_index(i, kAmrRefRatio), cj = coarsen_index(j, kAmrRefRatio);
        for (int k = 0; k < nc; ++k)
          f(i, j, k) = c(ci, cj, k);
      }
  }
}

// Builds the coarse level (BoxArray + DistributionMapping) of the AmrSystem path according to the
// ownership policy, in a SINGLE point for both build paths (native + compiled):
//  - replicated (distribute=false, DEFAULT): mono-box covering the domain, dmap = my_rank() everywhere
//    (the box lives on each rank). In serial my_rank()=0 -> identical to round-robin, bit for bit.
//    This is the layout GeometricMG(replicated=true) and the historical one expect.
//  - distributed (distribute=true): multi-box BoxArray::from_domain(dom, max_grid) spread round-robin
//    DistributionMapping(ba.size(), n_ranks()). Each rank carries only its tiles -> the coarse
//    Poisson and coarse transport distribute (strong-scaling). max_grid<=0 => n/2 (2x2).
inline std::pair<BoxArray, DistributionMapping> coupler_make_coarse_layout(int n, bool distribute,
                                                                           int max_grid) {
  const Box2D dom = Box2D::from_extents(n, n);
  if (!distribute) {
    BoxArray ba(std::vector<Box2D>{dom});
    return {ba, DistributionMapping(std::vector<int>{my_rank()})};
  }
  const int mg = (max_grid > 0) ? max_grid : (n / 2 > 0 ? n / 2 : n);
  BoxArray ba = BoxArray::from_domain(dom, mg);
  return {ba, DistributionMapping(ba.size(), n_ranks())};
}

}  // namespace detail

/// Multi-patch E x B AMR coupler. @tparam Model: PhysicalModel (flux, source, elliptic_rhs,
/// max_wave_speed). @tparam Elliptic: elliptic backend (EllipticSolver concept, default GeometricMG).
/// ORCHESTRATES only: the hierarchy lives in an AmrLevelStack<AmrLevelMP>, the Poisson solve in
/// mg_, the regrid in amr_regrid_finest. Reduces bit for bit to AmrCoupler in mono-box per level.
template <class Model, class Elliptic = GeometricMG>
class AmrCouplerMP {
  static_assert(EllipticSolver<Elliptic>, "Elliptic must model EllipticSolver");

 public:
  // active: optional "active cell" predicate (interior of the conductor), for the circular
  // conducting wall of the column instability (passed as-is to the multigrid). Empty
  // by default -> no wall (historical behavior unchanged). Only the coarse carries the
  // wall: the fine patches refine the ring edge, strictly inside the wall.
  // replicated_coarse: level-0 (coarse) OWNERSHIP POLICY. BOTH modes are
  // stable and their equivalence is proven bit for bit (test_mpi_decoarse, maxdiff=0):
  //   true  (performant DEFAULT): coarse mono-box REPLICATED on all ranks. Best coarse
  //          MG solve (no multigrid degeneration), zero communication for the
  //          coarse Poisson, robust reference -> the right default for small/medium cases.
  //   false (EXPLICIT scalable mode): coarse multi-box DISTRIBUTED round-robin. Lifts the
  //          O(NX*NY*nranks) memory lock of level 0, required at very large scale. But the
  //          geometric MG degenerates for a finely-split coarse (>2x2 boxes do not tile the
  //          coarsest grid): reserve for cases where the level-0 memory is the lock.
  // Criterion: set false ONLY when memory scalability requires it; otherwise keep true.
  // Removing the replicated path is DEFERRED as long as the distributed one is not strictly
  // superior. mg_ receives the same flag (otherwise, under replicated MPI, the coarse would fall on
  // the single rank 0 and compute_aux would read a phi absent elsewhere). In serial, both coincide.
  AmrCouplerMP(const Model& model, const Geometry& geom, const BoxArray& ba_coarse, const BCRec& bc,
               std::vector<AmrLevelMP> levels, std::function<bool(Real, Real)> active = {},
               bool replicated_coarse = true)
      : model_(model),
        geom_(geom),
        mg_(geom, ba_coarse, bc, std::move(active), replicated_coarse),
        stack_(geom.domain, std::move(levels), aux_comps<Model>()),
        replicated_coarse_(replicated_coarse) {}

  std::vector<AmrLevelMP>& levels() { return stack_.levels(); }
  MultiFab& coarse() { return stack_.coarse(); }
  const MultiFab& coarse() const { return stack_.coarse(); }
  // coarse-level aux: (phi, dphi/dx, dphi/dy), component 0 = phi (cf. compute_aux). Same
  // layout as coarse(). Read by the AmrSystem potential hook (coupler_read_coarse_phi).
  MultiFab& aux0() { return stack_.aux(0); }
  const MultiFab& aux0() const { return stack_.aux(0); }

  /// Registers a model-NAMED aux field (ADC-291) at shared-channel component @p comp (>= kAuxNamedBase),
  /// as a coarse base-level field @p field (n*n row-major, global cell index j*nx+i). STATIC user field
  /// re-applied by compute_aux every update (so it persists across regrid) and injected coarse->fine.
  /// Single-block AMR counterpart of System::set_aux_field_component. The facade validates comp/size and
  /// resolves the name. No-op default (no named field -> empty map -> bit-identical).
  void set_named_aux(int comp, std::vector<Real> field) {
    named_aux_[comp] = std::move(field);
    apply_named_aux();  // stack_ exists at ctor: reflect onto the coarse aux right away
  }
  /// Registers a per-field aux HALO policy (ADC-369) for the named component @p comp. compute_aux
  /// applies it onto the COARSE aux AFTER the shared fill, overriding only that component's
  /// physical-face ghosts (periodic faces stay periodic). Single-block AMR counterpart of
  /// System::set_aux_field_halo_component. No-op default.
  void set_named_aux_bc(int comp, AuxHaloPolicy policy) { named_aux_bc_[comp] = policy; }
  const Box2D& domain() const { return stack_.domain(); }
  int nlev() const { return stack_.nlev(); }

  // ----------------------------------------------------------------------------------------------
  // SINGLE-RANK AMR CHECKPOINT / RESTART (ADC-65). The mono-block coupler carries the FULL
  // CONSERVATIVE STATE per level (all components) + the phi (multigrid warm-start), and can IMPOSE
  // a SAVED fine hierarchy (instead of Berger-Rigoutsos clustering on tags). SINGLE-RANK: the
  // accessors loop over local_size() + device_fence(), WITHOUT MPI gather (the facade rejects np>1
  // AND multi-block upstream; multi-rank/multi-block restart is a documented follow-up).
  // ----------------------------------------------------------------------------------------------

  // Reads the FULL conservative state (all components) of level @p k into a flat
  // component-major field c*nf*nf + j*nf + i, nf = n << k (n = coarse side). The cells OUTSIDE
  // patches (uncovered fine level) stay at 0: a fine level is only defined within its patches
  // (at restart we rewrite ONLY the patch cells, cf. set_level_state).
  std::vector<double> level_state(int k) {
    std::vector<AmrLevelMP>& L = stack_.L();
    if (k < 0 || k >= static_cast<int>(L.size()))
      throw std::runtime_error("AmrCouplerMP::level_state: level out of bounds");
    MultiFab& U = L[k].U;
    const int nc = U.ncomp();
    const std::size_t nf = static_cast<std::size_t>(stack_.domain().nx()) << k;
    std::vector<double> out(static_cast<std::size_t>(nc) * nf * nf, 0.0);
    device_fence();
    for (int li = 0; li < U.local_size(); ++li) {
      const ConstArray4 u = U.fab(li).const_array();
      const Box2D v = U.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          for (int c = 0; c < nc; ++c)
            out[static_cast<std::size_t>(c) * nf * nf + static_cast<std::size_t>(j) * nf +
                static_cast<std::size_t>(i)] = u(i, j, c);
    }
    return out;
  }

  // Restores the full conservative state of level @p k from @p s (same layout as level_state).
  // Writes ONLY the VALID cells of the local fabs (the patches): the ghosts are redone at the
  // next update()/advance (exactly like after a regrid), and a fine cell outside a patch
  // does not exist. NO RE-PROLONGATION: the state is restored AS-IS (no coarse->fine injection).
  void set_level_state(int k, const std::vector<double>& s) {
    std::vector<AmrLevelMP>& L = stack_.L();
    if (k < 0 || k >= static_cast<int>(L.size()))
      throw std::runtime_error("AmrCouplerMP::set_level_state: level out of bounds");
    MultiFab& U = L[k].U;
    const int nc = U.ncomp();
    const std::size_t nf = static_cast<std::size_t>(stack_.domain().nx()) << k;
    if (s.size() != static_cast<std::size_t>(nc) * nf * nf)
      throw std::runtime_error("AmrCouplerMP::set_level_state: state size != ncomp*nf*nf");
    device_fence();
    for (int li = 0; li < U.local_size(); ++li) {
      Array4 u = U.fab(li).array();
      const Box2D v = U.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          for (int c = 0; c < nc; ++c)
            u(i, j, c) = s[static_cast<std::size_t>(c) * nf * nf +
                           static_cast<std::size_t>(j) * nf + static_cast<std::size_t>(i)];
    }
  }

  // Reads the potential phi of level @p k, flat nf*nf row-major field (nf = n << k), zeros outside patches.
  // Level 0: the multigrid WARM-START -- mg_.phi() (VALID cells), the state actually
  // reused by the NEXT solve (GeometricMG::solve keeps phi between calls). Level >= 1:
  // aux(k) component 0 (informational; recomputed at update). It is mg_.phi() level 0 that makes the
  // restart BIT-IDENTICAL (the 1st post-restart solve starts from the same guess as the continuous run).
  std::vector<double> level_potential(int k) {
    if (k < 0 || k >= stack_.nlev())
      throw std::runtime_error("AmrCouplerMP::level_potential: level out of bounds");
    const std::size_t nf = static_cast<std::size_t>(stack_.domain().nx()) << k;
    std::vector<double> out(nf * nf, 0.0);
    device_fence();
    const MultiFab& P = (k == 0) ? mg_.phi() : stack_.aux(k);
    for (int li = 0; li < P.local_size(); ++li) {
      const ConstArray4 p = P.fab(li).const_array();
      const Box2D v = P.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          out[static_cast<std::size_t>(j) * nf + static_cast<std::size_t>(i)] = p(i, j, 0);
    }
    return out;
  }

  // Restores the potential of level @p k. Level 0: warm-start mg_.phi() (valid cells) -> the
  // multigrid restart is BIT-IDENTICAL (the 1st post-restart solve starts from the same guess). Level
  // >= 1: aux(k) comp 0 (recomputed at update; idempotent restore, no effect on the dynamics).
  void set_level_potential(int k, const std::vector<double>& p) {
    if (k < 0 || k >= stack_.nlev())
      throw std::runtime_error("AmrCouplerMP::set_level_potential: level out of bounds");
    const std::size_t nf = static_cast<std::size_t>(stack_.domain().nx()) << k;
    if (p.size() != nf * nf)
      throw std::runtime_error("AmrCouplerMP::set_level_potential: phi size != nf*nf");
    device_fence();
    MultiFab& P = (k == 0) ? mg_.phi() : stack_.aux(k);
    for (int li = 0; li < P.local_size(); ++li) {
      Array4 q = P.fab(li).array();
      const Box2D v = P.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          q(i, j, 0) = p[static_cast<std::size_t>(j) * nf + static_cast<std::size_t>(i)];
    }
  }

  // Imposes the fine-level hierarchy (restart): rebuilds level 1 on the SAVED @p
  // fine_boxes BoxArray (instead of Berger-Rigoutsos clustering on tags), via the SAME mechanism as
  // regrid (regrid_field_on_layout: parent interp + fine carry-over), then reattaches the level-1 aux.
  // The rebuilt valid content is OVERWRITTEN afterwards by set_level_state (restore as-is): here we
  // rely only on the IMPOSED LAYOUT. SINGLE-RANK, 2-level mono-block hierarchy (so we impose
  // ONLY level 1). Clear rejection if the hierarchy has no fine level or if no box was saved.
  void set_hierarchy(const std::vector<Box2D>& fine_boxes) {
    std::vector<AmrLevelMP>& L = stack_.L();
    if (L.size() < 2)
      throw std::runtime_error(
          "AmrCouplerMP::set_hierarchy: mono-level hierarchy (no fine patch "
          "to impose)");
    if (fine_boxes.empty())
      throw std::runtime_error(
          "AmrCouplerMP::set_hierarchy: no saved fine box (restart of a "
          "fine-patch hierarchy required)");
    const int ngf = L[1].U.n_grow();  // inherit the ghost width of the current fine (scheme parity)
    BoxArray fb(fine_boxes);
    DistributionMapping dmap(static_cast<int>(fb.size()),
                             n_ranks());  // single-rank -> all on rank 0
    L[1].U = regrid_field_on_layout(fb, dmap, L[0].U, L[1].U, /*pk=*/0, ngf, replicated_coarse_);
    stack_.reattach_aux(1);  // realloc aux[1] on the new layout + rewire L[1].aux
  }

  void sync_down() {  // average fine -> coarse over the whole hierarchy (multi-box)
    auto& L = stack_.L();
    for (int k = stack_.nlev() - 1; k >= 1; --k)
      mf_average_down_mb(L[k].U, L[k - 1].U);
  }

  /// OPT-IN: replaces the Option A AMR Poisson (coarse solve + piecewise-constant gradient injection)
  /// with a COMPOSITE FAC elliptic solve (the fine patch REFINES the elliptic). Cf. CompositeFacPoisson.
  /// Phase 2 scope: 2 levels, ONE interior mono-box fine patch, replicated coarse (single-rank).
  /// Outside this scope, compute_aux falls back to Option A (bit-identical).
  void set_composite_poisson(bool v) { composite_poisson_ = v; }
  bool composite_poisson() const { return composite_poisson_; }

  void compute_aux() {  // coarse Poisson + grad phi + injection to the fine levels
    auto& L = stack_.L();
    const Box2D& dom = stack_.domain();
    const Real dx = geom_.dx(), dy = geom_.dy();
    // COMPOSITE path (opt-in): the fine patch TRULY refines the elliptic. Supported scope = 2 levels,
    // ONE mono-box fine patch, replicated coarse (Phase 2). Otherwise Option A below (bit-identical).
    if (composite_poisson_ && replicated_coarse_ && stack_.nlev() == 2 &&
        L[1].U.box_array().size() == 1) {
      compute_aux_composite();
      return;
    }
    // right-hand side via the model (no copied formula): f = elliptic_rhs(U)
    detail::coupler_eval_rhs(L[0].U, mg_.rhs(), model_);
    mg_.solve();  // leaves phi with its ghosts filled (last gs_rb_sweep -> fill_ghosts)
    device_fence();
    // aux = (phi, grad phi) per LOCAL coarse fab: covers the replicated mono-box (1 fab) as well as
    // the distributed multi-box (de-replication). The box-edge derivatives read the ghosts of
    // phi filled by the solve (distributed inter-box exchange via fill_boundary). mg_.phi() and
    // aux(0) share the same layout (same BoxArray + DistributionMapping) -> fab(li) <-> box(li).
    for (int li = 0; li < mg_.phi().local_size(); ++li) {
      const ConstArray4 p = mg_.phi().fab(li).const_array();
      Array4 a = stack_.aux(0).fab(li).array();
      const Box2D b = stack_.aux(0).box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
          a(i, j, 0) = p(i, j);
          a(i, j, 1) = (p(i + 1, j) - p(i - 1, j)) / (2 * dx);
          a(i, j, 2) = (p(i, j + 1) - p(i, j - 1)) / (2 * dy);
        }
    }
    // model-NAMED aux (ADC-291): re-apply the static named fields onto the coarse valid cells BEFORE
    // fill_boundary (ghosts) and the injection (lines below), so they reach every level and survive a
    // regrid. The loop above wrote only comps 0..2 (phi/grad), so this never clobbers them. No-op
    // without a named field (bit-identical).
    apply_named_aux();
    fill_boundary(stack_.aux(0), dom, Periodicity{true, true});
    apply_named_aux_bc();  // ADC-369: per-field halo override on the coarse physical ghosts (after the
                           // shared fill); no-op on a periodic domain / without a policy.
    // parent aux(k-1) replicated only if level 0 is: otherwise it is DISTRIBUTED (multi-box)
    // and the injection goes through parallel_copy. Beyond level 1, the parent is always distributed.
    for (int k = 1; k < stack_.nlev(); ++k)
      detail::coupler_inject_aux_mb(stack_.aux(k - 1), stack_.aux(k),
                                    /*replicated_parent=*/(k == 1) && replicated_coarse_);
  }

  /// Updates the hierarchy before a step: sync_down (fine -> coarse) then compute_aux (coarse
  /// Poisson + grad phi + injection to the fine levels).
  void update() {
    sync_down();
    compute_aux();
  }

  // Selectable spatial discretization (default FirstOrder = NoSlope + Rusanov,
  // strictly identical to the old step()). recon_prim selects the primitive
  // reconstruction (same parameter as assemble_rhs / System); false (default) -> conservative.
  // imex: treats the stiff source IMPLICITLY (backward_euler) rather than forward Euler;
  // false (default) -> historical explicit treatment, bit-identical. The source being
  // cell-local (outside reflux registers), the implicit split preserves conservation.
  /// Advances the hierarchy by one step dt: update() then advance_amr (Berger-Oliger subcycling +
  /// reflux + conservative average_down). @tparam Disc: spatial discretization (limiter + flux,
  /// default FirstOrder bit-identical to the historical one). recon_prim: primitive reconstruction; imex:
  /// stiff source implicit (backward_euler). Defaults (false) -> historical explicit path.
  /// @p nopts: OPTIONS of the IMEX implicit-source Newton (iteration budget, tolerances,
  /// fd_eps, damping, fail_policy), threaded down to backward_euler_source by advance_amr ->
  /// subcycle_level_mp -> mf_apply_source_treatment. DEFAULT {} = historical constants (2 iters,
  /// 1e-7, ...) -> path (2a) BIT-IDENTICAL to the old call. No effect if imex==false. The
  /// partial IMEX mask is NOT carried by this mono-block path (full backward-Euler), only the OPTIONS
  /// are (the mono-block AmrSystem wires the Newton options but not the mask or the diagnostics).
  /// @p tmethod: time method (kEuler by default = historical forward Euler bit-identical;
  /// kSsprk3 = order-3 SSPRK3 + per-stage reflux). kSsprk3 requires imex == false (rejected otherwise).
  template <class Disc = FirstOrder>
  void step(Real dt, bool recon_prim = false, bool imex = false, const NewtonOptions& nopts = {},
            AmrTimeMethod tmethod = AmrTimeMethod::kEuler, Real pos_floor = Real(0)) {
    update();
    advance_amr<typename Disc::Limiter, typename Disc::NumericalFlux>(
        model_, stack_.L(), stack_.domain(), dt, Periodicity{true, true}, replicated_coarse_,
        recon_prim, imex, nopts, tmethod, pos_floor);
  }

  /// TRANSPORT-ONLY ADVANCE (hyperbolic), WITHOUT update() or source. Counterpart of step() stripped
  /// of its field solve and with imex==false: this is the PURE HYPERBOLIC advance (-div F) of the
  /// amr-schur path, where the field is solved separately (update()) and the source is played by the
  /// global condensed stage (AmrCondensedSchurSourceStepper), exactly like the uniform path interleaving
  /// solve_fields / transport (s.advance) / source stage (run_source_stage). The model must be
  /// SOURCE-FREE (NoSource source brick) so that the source is not counted twice (once
  /// here in forward Euler, once by the Schur stage): this is the contract of the amr-schur path, mirror of
  /// the uniform time=Strang(Explicit, CondensedSchur) where the block is added with its transport only.
  template <class Disc = FirstOrder>
  void advance_transport(Real dt, bool recon_prim = false, Real pos_floor = Real(0)) {
    advance_amr<typename Disc::Limiter, typename Disc::NumericalFlux>(
        model_, stack_.L(), stack_.domain(), dt, Periodicity{true, true}, replicated_coarse_,
        recon_prim, /*imex=*/false, NewtonOptions{}, AmrTimeMethod::kEuler, pos_floor);
  }

  // Regrid of the FINE level by Berger-Rigoutsos (delegated to amr_regrid_finest):
  // rebuilds the patches (carry over fine data, otherwise parent interp) + the aux.
  // margin = nesting. The coupler only orders the call.
  template <class Crit>
  void regrid(Crit crit, int grow = 2, int margin = 2) {
    amr_regrid_finest(stack_.L(), stack_.aux(), stack_.domain(), crit, grow, margin,
                      aux_comps<Model>(), replicated_coarse_);
  }

  // coarse mass via the shared diagnostic amr_mass_mb (replicated mono-box as well as
  // distributed multi-box). Replicated coarse: the local sum IS already the total mass
  // (each rank holds everything) -> no all_reduce. Distributed: local part -> all_reduce_sum.
  Real mass() const {
    const Real M = amr_mass_mb(stack_.coarse(), geom_.dx(), geom_.dy());
    return replicated_coarse_ ? M : all_reduce_sum(M);
  }

  // max drift speed via amr_max_drift_speed_mb + floor. all_reduce_max correct
  // in BOTH cases: under replication the local max is already global (idempotent);
  // distributed, we take the max of the parts.
  Real max_drift_speed() const {
    const Real v = amr_max_drift_speed_mb(stack_.aux(0), model_.B0);
    return all_reduce_max(std::max(v, Real(1e-12)));
  }

  /// @brief Max wave speed on the coarse level via `model.max_wave_speed`.
  ///
  /// Model-generic CFL speed (any `PhysicalModel`), unlike `max_drift_speed`
  /// which is specific to the E x B drift (`model.B0`). For a pure E x B transport, it equals
  /// the drift speed.
  ///
  /// @return the max over the coarse cells and the two directions, reduced over the ranks.
  /// @note `update()` must have run so that `aux(0)` carries the current `grad phi`.
  Real max_wave_speed() {
    Real w = Real(1e-12);
    MultiFab& U = stack_.coarse();
    MultiFab& A = stack_.aux(0);
    for (int li = 0; li < U.local_size(); ++li) {
      const ConstArray4 u = U.fab(li).const_array();
      const ConstArray4 a = A.fab(li).const_array();
      const Box2D b = U.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
          const auto us = load_state<Model>(u, i, j);
          const Aux ax = load_aux<aux_comps<Model>()>(a, i, j);
          w = std::max(
              w, std::max(model_.max_wave_speed(us, ax, 0), model_.max_wave_speed(us, ax, 1)));
        }
    }
    return all_reduce_max(w);
  }

 private:
  /// COMPOSITE FAC Poisson step (opt-in path). Solves the elliptic on coarse + fine patch coupled by
  /// FAC, then sets aux PER LEVEL from the phi OF EACH LEVEL: fine aux = (phi_f, fine grad) where fine
  /// grad = centered diff on phi_f (solved at fine resolution), NOT the constant coarse-grad injection of Option A.
  void compute_aux_composite() {
    // ADC-291 NOTE: unlike compute_aux (Option A), this opt-in composite-FAC path does NOT re-apply
    // named aux onto the fine level (it derives each level's aux from the FAC phi, with no coarse->fine
    // aux injection). The coarse named comp survives (the grad writes touch only comps 0..2), but a
    // fine-level model reading extra_field(k) would read 0. set_composite_poisson is C++-only and not
    // facade-reachable, so named aux cannot hit this path today; carrying named aux to the composite
    // fine level is a documented follow-up (cf. adc.capabilities()['aux']['followups']).
    auto& L = stack_.L();
    const Box2D& dom = stack_.domain();
    const Box2D fine_box = L[1].U.box_array()[0];
    if (!fac_built_ || !same_box(fac_fine_box_, fine_box)) {
      fac_ = std::make_shared<CompositeFacPoisson>(geom_, mg_.box_array(), mg_.bc(), fine_box, 2);
      fac_fine_box_ = fine_box;
      fac_built_ = true;
    }
    // f = elliptic_rhs(U) PER LEVEL: the fine has its OWN refined right-hand side (not an injection).
    detail::coupler_eval_rhs(L[0].U, fac_->rhs_coarse(), model_);
    detail::coupler_eval_rhs(L[1].U, fac_->rhs_fine(), model_);
    fac_->solve();
    device_fence();
    // level-0 aux (coarse): phi + grad from phi_coarse (same centered stencils as the Option A path).
    fill_ghosts(fac_->phi_coarse(), dom, mg_.bc());
    detail::coupler_grad_phi(fac_->phi_coarse(), stack_.aux(0), Real(1) / (Real(2) * geom_.dx()),
                             Real(1) / (Real(2) * geom_.dy()));
    fill_boundary(stack_.aux(0), dom, Periodicity{true, true});
    // level-1 aux (fine): phi + grad from phi_fine -> FINE grad (fine centered diff, reads the C-F
    // bilinear ghosts) = the fidelity gain vs the constant coarse grad injected by Option A.
    detail::coupler_grad_phi(fac_->phi_fine(), stack_.aux(1), Real(1) / (Real(2) * L[1].dx),
                             Real(1) / (Real(2) * L[1].dy));
  }

  static bool same_box(const Box2D& a, const Box2D& b) {
    return a.lo[0] == b.lo[0] && a.lo[1] == b.lo[1] && a.hi[0] == b.hi[0] && a.hi[1] == b.hi[1];
  }

  Model model_;
  Geometry geom_;
  Elliptic mg_;
  AmrLevelStack<AmrLevelMP> stack_;
  bool
      replicated_coarse_;  // level 0 replicated (true) or distributed multi-box (false, de-replication)
  // COMPOSITE FAC Poisson path (opt-in, set_composite_poisson). fac_ built lazily on the
  // current fine patch (rebuilt if the patch changes after regrid). Default OFF -> Option A bit-identical.
  bool composite_poisson_ = false;
  bool fac_built_ = false;
  std::shared_ptr<CompositeFacPoisson> fac_;
  Box2D fac_fine_box_{};
  // Model-NAMED aux fields (ADC-291): component (>= kAuxNamedBase) -> coarse base-level field
  // (n*n row-major). STATIC user fields re-applied by compute_aux each update (so they persist across
  // regrid). Empty by default -> bit-identical. cf. set_named_aux / apply_named_aux.
  std::map<int, std::vector<Real>> named_aux_;
  // Per-field aux HALO policy (ADC-369): component -> uniform boundary policy, applied to the coarse aux
  // after the shared fill (apply_named_aux_bc). Empty by default -> bit-identical.
  std::map<int, AuxHaloPolicy> named_aux_bc_;

  // Re-applies the model-NAMED aux fields onto the COARSE shared aux valid cells. Mirror of
  // SystemFieldSolver::apply_named_aux_one and AmrRuntime::apply_named_aux: per local fab (MPI-safe),
  // valid cells only, global flat index j*nx+i. compute_aux runs the coarse->fine injection right
  // after, carrying the named comps to the fine levels. No-op without a named field.
  void apply_named_aux() {
    if (named_aux_.empty())
      return;
    const int row = stack_.domain().nx();
    for (const auto& [comp, field] : named_aux_) {
      if (field.empty() || comp >= stack_.aux(0).ncomp())
        continue;
      for (int li = 0; li < stack_.aux(0).local_size(); ++li) {
        Array4 a = stack_.aux(0).fab(li).array();
        const Box2D v = stack_.aux(0).box(li);
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i)
            a(i, j, comp) = field[static_cast<std::size_t>(j) * row + i];
      }
    }
  }

  // Per-field aux HALO override (ADC-369) on the COARSE aux, AFTER the shared fill. Overrides only each
  // declared component's physical-face ghosts; aux_halo_override(mg_.bc(), policy) keeps periodic faces
  // periodic (so on a periodic domain this is a no-op). Mirror of SystemFieldSolver::apply_named_aux_bc.
  void apply_named_aux_bc() {
    if (named_aux_bc_.empty())
      return;
    for (const auto& [comp, policy] : named_aux_bc_) {
      if (comp >= stack_.aux(0).ncomp())
        continue;
      fill_physical_bc(stack_.aux(0), stack_.domain(), aux_halo_override(mg_.bc(), policy), comp);
    }
  }
};

}  // namespace adc
