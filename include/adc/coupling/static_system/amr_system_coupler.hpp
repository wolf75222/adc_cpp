#pragma once

#include <adc/core/coupled_system.hpp>
#include <adc/amr/refinement_ratio.hpp>
#include <adc/core/types.hpp>
#include <adc/coupling/amr/amr_coupler_mp.hpp>  // detail::coupler_inject_aux_mb
#include <adc/coupling/base/aux_fill.hpp>  // detail::derive_aux_bc + detail::fill_bz_box (shared)
#include <adc/coupling/source/coupled_source.hpp>  // CoupledSourceFor
#include <adc/coupling/base/elliptic_rhs.hpp>
#include <adc/numerics/elliptic/elliptic_problem.hpp>  // field_postprocess, FieldPostProcess
#include <adc/numerics/elliptic/elliptic_solver.hpp>
#include <adc/numerics/elliptic/geometric_mg.hpp>
#include <adc/numerics/time/amr_reflux_mf.hpp>     // AmrLevelMP, advance_amr, mf_average_down_mb
#include <adc/numerics/time/implicit_stepper.hpp>  // backward_euler_source
#include <adc/numerics/time/scheduler.hpp>         // block_substeps_v, block_time_treatment_v
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/parallel/comm.hpp>  // all_reduce_sum

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

/// @file
/// @brief AmrSystemCoupler: multi-species SYSTEM coupler on AMR (milestone 2.3).
///
/// Carries a CoupledSystem on an AMR hierarchy: each block has ITS OWN level hierarchy, all species
/// SHARE the same AMR grid, the same aux field (phi, grad phi [, B_z, ...]) and the same coarse
/// Poisson. Orchestration: sync_down (per block) -> coarse Poisson f = Sum_s q_s n_s -> coarse aux +
/// injection to the fine levels -> each block advances via advance_amr<Disc_block> (with its scheme
/// and its substeps; implicit/IMEX blocks delegated to a callback). STRONG INVARIANT: all blocks live
/// on EXACTLY the same grid per level (the aux is shared); same_layout_or_throw checks this at the
/// ctor. PoissonCadence chooses the re-solve frequency of phi (OncePerStep frozen vs PerSubstep).
/// Single-block = bit-identical path to history (loops over the other blocks are empty).

namespace adc {

static_assert(kAmrRefRatio == 2, "refine(1 << k) assumes a power-of-two (ratio-2) cascade");

/// Re-solve frequency of the Poisson on AMR: OncePerStep (phi solved once per macro-step, frozen
/// during the advance; cheapest); PerSubstep (phi re-solved before each species substep, more
/// faithful for a field-driven transport, more expensive).
enum class PoissonCadence { OncePerStep, PerSubstep };

// EXPLICIT layout of a shared AMR hierarchy (point 2 of the multi-block capstone, first MINIMAL
// step). Single source of truth on the GRID that all blocks share: per level the BoxArray (the boxes
// AND their order), the DistributionMapping (rank per box), dx/dy, and the number of levels
// (= ba.size()). Today this information is implicit, scattered across each AmrLevelMP (U.box_array() /
// U.dmap() / dx,dy). This type only EXTRACTS it for the same_layout_or_throw guard: it does NOT
// replace EquationBlock / AmrLevelMP and introduces NO block abstraction (the wide AmrBlock of the
// design is a LATER step, and only if needed). The layout of a stack of levels is read via from_levels.
/// Single source of truth on the GRID shared by all blocks: per level the BoxArray (boxes AND order),
/// the DistributionMapping (rank per box) and dx/dy. EXTRACTED only (replaces neither EquationBlock nor
/// AmrLevelMP); used by the same_layout_or_throw guard.
struct AmrHierarchyLayout {
  std::vector<BoxArray> ba;             // [level]: boxes of the level (set AND order)
  std::vector<DistributionMapping> dm;  // [level], parallel to ba: MPI rank per box
  std::vector<Real> dx, dy;             // [level]: grid spacing (= dx_coarse / 2^k)

  /// Number of levels (= ba.size()).
  int nlev() const { return static_cast<int>(ba.size()); }

  // Reads the layout carried by the level stack of ONE block (each AmrLevelMP carries
  // U.box_array() / U.dmap() / dx,dy). No copy of field data: only the grid.
  /// Extracts the layout (BoxArray + DistributionMapping + dx/dy per level) from the level stack of ONE
  /// block. No copy of field data, only the grid.
  static AmrHierarchyLayout from_levels(const std::vector<AmrLevelMP>& levels) {
    AmrHierarchyLayout L;
    const int n = static_cast<int>(levels.size());
    L.ba.reserve(n);
    L.dm.reserve(n);
    L.dx.reserve(n);
    L.dy.reserve(n);
    for (const auto& lv : levels) {
      L.ba.push_back(lv.U.box_array());
      L.dm.push_back(lv.U.dmap());
      L.dx.push_back(lv.dx);
      L.dy.push_back(lv.dy);
    }
    return L;
  }
};

namespace detail {
template <class>
inline constexpr bool amr_always_false_v = false;

// EXACT comparison of the grids of two levels (point 1): same BoxArray (boxes AND order), same
// DistributionMapping (rank per box), same dx/dy (bit-for-bit). Returns true if everything matches.
// dx/dy are the level spacings, identical by construction if the boxes are; we compare them anyway to
// catch a mis-wired geometry.
inline bool same_level_layout(const BoxArray& a_ba, const DistributionMapping& a_dm, Real a_dx,
                              Real a_dy, const BoxArray& b_ba, const DistributionMapping& b_dm,
                              Real b_dx, Real b_dy) {
  return a_ba.boxes() == b_ba.boxes() && a_dm.ranks() == b_dm.ranks() && a_dx == b_dx &&
         a_dy == b_dy;
}

// LAYOUT CONSISTENCY guard between blocks (point 1 of the capstone). The aux is SHARED per level: all
// blocks MUST live on EXACTLY the same grid at each level, otherwise the rewiring
// levels[k].aux = &aux_[k] and the advance read an inconsistent grid (silent out-of-bound access).
// The old check only compared the NUMBER of boxes (.size()); here we compare EXACTLY: number of
// levels, then per level BoxArray (boxes AND order), DistributionMapping and dx/dy. Throws a clear
// error at the FIRST discrepancy (block and level located). A single block matches itself trivially ->
// single-block path strictly bit-identical (the loop over the other blocks is empty).
inline void same_layout_or_throw(const std::vector<std::vector<AmrLevelMP>>& block_levels) {
  if (block_levels.empty())
    return;
  const auto& ref = block_levels[0];
  const int nlev = static_cast<int>(ref.size());
  for (std::size_t b = 1; b < block_levels.size(); ++b) {
    const auto& cur = block_levels[b];
    if (static_cast<int>(cur.size()) != nlev)
      throw std::runtime_error(
          "AmrSystemCoupler: all blocks must have the same number of levels "
          "(shared AMR layout)");
    for (int k = 0; k < nlev; ++k) {
      if (!same_level_layout(cur[k].U.box_array(), cur[k].U.dmap(), cur[k].dx, cur[k].dy,
                             ref[k].U.box_array(), ref[k].U.dmap(), ref[k].dx, ref[k].dy))
        throw std::runtime_error(
            "AmrSystemCoupler: inconsistent AMR layout between blocks (the shared aux requires the "
            "SAME BoxArray [boxes and order], the SAME DistributionMapping and the SAME dx/dy per "
            "level)");
    }
  }
}
}  // namespace detail

/// Multi-species system coupler on AMR. @tparam System: CoupledSystem (blocks/species).
/// @tparam RhsAssembler: assembler of the Poisson RHS (f = Sum_s q_s n_s, e.g. ChargeDensityRhs).
/// @tparam Elliptic: elliptic backend (EllipticSolver concept, default GeometricMG). PRECONDITION:
/// all blocks share EXACTLY the same AMR layout per level (checked at the ctor).
template <CoupledSystemLike System, class RhsAssembler, class Elliptic = GeometricMG>
class AmrSystemCoupler {
  static_assert(EllipticSolver<Elliptic>, "the elliptic backend must model EllipticSolver");

 public:
  // block_levels[b] = hierarchy of block b (level 0 = coarse on ba_coarse, levels
  // > 0 = fine patches). The AmrLevelMP carry U + dx/dy per level; their aux pointer
  // is (re)wired here to the SHARED aux. The ctor also re-points block.state to the
  // coarse level of its hierarchy, so that the system RHS (ChargeDensityRhs) reads
  // the coarse densities correctly.
  // bz: out-of-plane magnetic field B_z(x, y) provided by the user (constant or field),
  // shared by ALL blocks. Set on the B_z component (index kAuxBaseComps) of the SHARED aux
  // channel of EACH level, from the cell centers OF THAT LEVEL (each level has its own
  // geometry / dx). AMR analog of the bz_ of SystemAssembler (non-AMR path). A block that
  // reads B_z (n_aux=4) sees it at all levels, a base block (3) ignores the component. Without
  // a block with an extra field (width 3) or if bz is empty: no-op -> bit-identical to history.
  AmrSystemCoupler(System system, const Geometry& geom, const BoxArray& ba_coarse,
                   const BCRec& bcPhi, RhsAssembler rhs_assembler,
                   std::vector<std::vector<AmrLevelMP>> block_levels,
                   Periodicity base_per = Periodicity{true, true}, bool replicated_coarse = true,
                   PoissonCadence cadence = PoissonCadence::OncePerStep,
                   std::function<bool(Real, Real)> active = {},
                   std::function<Real(Real, Real)> bz = {})
      : system_(std::move(system)),
        rhs_assembler_(std::move(rhs_assembler)),
        geom_(geom),
        dom_(geom.domain),
        base_per_(base_per),
        bcPhi_(bcPhi),
        aux_bc_(detail::derive_aux_bc(bcPhi)),
        replicated_coarse_(replicated_coarse),
        cadence_(cadence),
        mg_(geom, ba_coarse, bcPhi, std::move(active), replicated_coarse),
        block_levels_(std::move(block_levels)),
        bz_(std::move(bz)) {
    // Construction checks (Codex review): without them, a malformed hierarchy
    // causes a silent out-of-bound access in the wiring / the advance.
    if (block_levels_.size() != System::n_blocks)
      throw std::runtime_error(
          "AmrSystemCoupler: block_levels must have one level vector per block "
          "(size != n_blocks)");
    nlev_ = block_levels_.empty() ? 0 : static_cast<int>(block_levels_[0].size());
    if (nlev_ == 0)
      throw std::runtime_error("AmrSystemCoupler: at least one level (coarse) required");
    // EXACT layout consistency between blocks (the aux is shared per level): same number of
    // levels, and per level same BoxArray (boxes AND order), same DistributionMapping, same
    // dx/dy. Replaces the old check that only compared the NUMBER of boxes (.size()).
    // Single-block: the check is trivial (a single block) -> bit-identical to history.
    detail::same_layout_or_throw(block_levels_);
    // SHARED aux: one MultiFab (phi, grad phi [, B_z, ...]) per level, on the common grid.
    // Sized once -> stable addresses for the blocks' aux pointers. Width =
    // max of aux_comps<Model> over the blocks (at least 3): a block reading B_z (n_aux > 3) has
    // the room at EACH level, a base block ignores the extra components. Without a block with an
    // extra field -> width 3 -> allocation strictly bit-identical to history.
    aux_ncomp_ = system_aux_comps(system_);
    aux_.resize(nlev_);
    for (int k = 0; k < nlev_; ++k)
      aux_[k] =
          MultiFab(block_levels_[0][k].U.box_array(), block_levels_[0][k].U.dmap(), aux_ncomp_, 1);
    for (auto& levels : block_levels_)
      for (int k = 0; k < nlev_; ++k)
        levels[k].aux = &aux_[k];

    // re-point each block to ITS coarse level (block.U() = coarse of the block).
    std::size_t b = 0;
    system_.for_each_block([&](auto& block) {
      block.state = &block_levels_[b][0].U;
      ++b;
    });

    fill_bz();  // populates B_z per level (no-op if no block requests it or if bz is empty)
  }

  // Setter (parity with the ctor: alternative to set B_z after construction). Immediately
  // re-populates the aux channel of each level. Effective no-op if the aux width <= base.
  void set_bz(std::function<Real(Real, Real)> bz) {
    bz_ = std::move(bz);
    fill_bz();
  }

  System& system() { return system_; }
  const System& system() const { return system_; }
  MultiFab& phi() { return mg_.phi(); }
  int nlev() const { return nlev_; }
  const MultiFab& aux(int k) const { return aux_[k]; }
  // WRITE access to the shared aux channel of level k (parity with SystemAssembler::aux()):
  // allows populating an extra component (B_z, ...) that field_postprocess does not touch
  // (it only writes phi/grad, comp 0..2). The width is aux_ncomp_ (max aux_comps of the blocks).
  MultiFab& aux(int k) { return aux_[k]; }
  int aux_ncomp() const { return aux_ncomp_; }
  std::vector<AmrLevelMP>& levels(std::size_t b) { return block_levels_[b]; }
  MultiFab& coarse(std::size_t b) { return block_levels_[b][0].U; }
  const MultiFab& coarse(std::size_t b) const { return block_levels_[b][0].U; }
  // number of Poisson solves of the last step(): diagnostic of the cadence.
  int solve_count() const { return solve_count_; }

  // sync_down (per block) + coarse system Poisson + coarse aux + fine injection.
  /// Solves the fields: average_down per block, coarse system Poisson (RHS = Sum_s q_s n_s),
  /// coarse aux (phi, grad phi) then injection to the fine levels + re-sets B_z per level. Increments
  /// solve_count().
  void solve_fields() {
    ++solve_count_;
    for (auto& levels : block_levels_)
      for (int k = nlev_ - 1; k >= 1; --k)
        mf_average_down_mb(levels[k].U, levels[k - 1].U);

    rhs_assembler_(system_, mg_.rhs());  // f = Sum_s q_s n_s on the coarse level
    mg_.solve();

    // coarse aux = (phi, grad phi) via the SAME clean path as the single-level
    // SystemCoupler (Codex review 9.4): fill the ghosts of phi according to bcPhi_, then
    // field_postprocess, then fill the ghosts of aux according to aux_bc_ (derived from bcPhi_).
    // Handles the non-periodic case (Foextrap) instead of a hard-coded periodic fill_boundary.
    fill_ghosts(mg_.phi(), dom_, bcPhi_);
    const Real cx = Real(1) / (2 * geom_.dx()), cy = Real(1) / (2 * geom_.dy());
    field_postprocess(mg_.phi(), aux_[0], cx, cy,
                      FieldPostProcess{FieldPostProcess::GradSign::Plus, true});
    fill_ghosts(aux_[0], dom_, aux_bc_);
    for (int k = 1; k < nlev_; ++k)
      detail::coupler_inject_aux_mb(aux_[k - 1], aux_[k],
                                    /*replicated_parent=*/(k == 1) && replicated_coarse_);

    // B_z PER LEVEL (not just propagated): coupler_inject_aux_mb copies ALL the components
    // of the parent (including B_z) to the fine levels, which would overwrite the fine B_z with a
    // coarse B_z injected (constant per coarse cell). So we re-set B_z from bz_ at the FINE centers
    // after the injection, so that a spatially varying B_z is sampled at the level resolution.
    // Static and cheap; no-op if the aux width <= base or bz empty (constant B_z: this re-fill is
    // idempotent, the injection would have sufficed).
    fill_bz();
  }

  // Advances the system by one step. Explicit blocks: advance_amr with their Disc and their
  // species substeps. Implicit / IMEX blocks: delegated to the callback (coupler, block,
  // levels, dt), Newton / IMEX branch point (default AmrImplicitSourceStepper).
  /// Advances the system by one macro-step dt. Explicit blocks via advance_amr (their Disc + substeps);
  /// implicit/IMEX blocks delegated to @p implicit_advance (coupler, block, levels, dt). The per-block
  /// cadence (stride) holds a slow block then catches it up (hold-then-catch-up).
  template <class ImplicitAdvance>
  void step(Real dt, ImplicitAdvance&& implicit_advance) {
    solve_count_ = 0;
    solve_fields();
    std::size_t b = 0;
    system_.for_each_block([&](auto& block) {
      using Block = std::decay_t<decltype(block)>;
      using Disc = typename Block::Spatial;
      constexpr TimeTreatment treatment = block_time_treatment_v<Block>;
      constexpr int n = block_substeps_v<Block>;
      constexpr int stride = block_stride_v<Block>;
      const std::size_t bi = b++;
      // HOLD-THEN-CATCH-UP cadence (add_block doc, sec.8.2 C): the block is HELD at the
      // macro-steps 0..stride-2 and catches up at macro-step stride-1 (when
      // (macro_step_+1) % stride == 0). Avoids a slow block advancing ahead
      // at the first macro-step (macro_step_=0, old condition 0%stride==0 true),
      // which put the slow block IN THE FUTURE relative to the fast blocks.
      // stride=1: (macro_step_+1)%1==0 always true -> every step, bit-identical.
      if ((macro_step_ + 1) % stride != 0)
        return;
      const Real bdt = dt * static_cast<Real>(stride);
      auto& levels = block_levels_[bi];
      if constexpr (treatment == TimeTreatment::Explicit) {
        const Real h = bdt / static_cast<Real>(n);
        for (int s = 0; s < n; ++s) {
          // PerSubstep: re-solves phi before each subsequent substep (the charge has
          // moved); the first reuses the head solve. OncePerStep: phi frozen.
          if (cadence_ == PoissonCadence::PerSubstep && s > 0)
            solve_fields();
          advance_amr<typename Disc::Limiter, typename Disc::NumericalFlux>(
              block.model, levels, dom_, h, base_per_, replicated_coarse_);
        }
      } else if constexpr (treatment == TimeTreatment::Implicit ||
                           treatment == TimeTreatment::IMEX) {
        // IMEX = true forward-backward (Codex review 9.1): explicit transport by the
        // AMR engine on a SOURCE-FREE model (-div F only), then implicit source by
        // the callback. Pure implicit: everything to the callback (no transport).
        if constexpr (treatment == TimeTreatment::IMEX)
          advance_amr<typename Disc::Limiter, typename Disc::NumericalFlux>(
              SourceFreeModel<typename Block::Model>{block.model}, levels, dom_, bdt, base_per_,
              replicated_coarse_);
        implicit_advance(*this, block, levels, bdt);
      }
    });
    ++macro_step_;
  }

  // Overload for a fully explicit system.
  /// Overload for a FULLY explicit system (static_assert if an implicit/IMEX block goes through it).
  void step(Real dt) {
    step(dt, [](auto&, auto& block, auto&, Real) {
      using Block = std::decay_t<decltype(block)>;
      static_assert(detail::amr_always_false_v<Block>,
                    "AmrSystemCoupler::step(dt) cannot advance an "
                    "implicit/IMEX block without a callback");
    });
  }

  // Inter-species coupling source on AMR (parity with SystemCoupler, Codex review
  // 9.5): forward-Euler splitting applied PER LEVEL. We refresh phi (aux per
  // level) then, at each level k, we temporarily re-point each block to its
  // level k and let the source read all the blocks + aux[k]. NoCoupledSource => no-op.
  template <class CoupledSource>
  void coupled_source_step(CoupledSource&& src, Real dt) {
    static_assert(CoupledSourceFor<std::decay_t<CoupledSource>, System>,
                  "coupled_source_step expects a CoupledSource: apply(system, aux, dt)");
    solve_fields();
    for (int k = 0; k < nlev_; ++k) {
      std::size_t b = 0;
      MultiFab* saved[System::n_blocks == 0 ? 1 : System::n_blocks];
      system_.for_each_block([&](auto& block) {
        saved[b] = block.state;
        block.state = &block_levels_[b][k].U;
        ++b;
      });
      src.apply(system_, aux_[k], dt);
      b = 0;
      system_.for_each_block([&](auto& block) { block.state = saved[b++]; });
    }
    // COVERAGE INVARIANT: the source was applied independently on EACH
    // level, so a coarse cell COVERED by a fine patch now carries its
    // own coarse source, unrelated to the source seen by its fine children. A
    // covered coarse cell must, by definition, be the 2x2 average of its children
    // (it does not represent matter on its own, it is a coarse view of the fine).
    // We restore this consistency by a fine -> coarse cascade identical to that of
    // solve_fields and of the transport-IMEX path (subcycle_level_mp). Without it, the
    // amr_mass diagnostic (which sums only the coarse level) counts a ghost coarse source
    // under the patch. Single-level hierarchy: no covered cell, the loop
    // does not execute -> strictly bit-identical to history.
    for (auto& levels : block_levels_)
      for (int k = nlev_ - 1; k >= 1; --k)
        mf_average_down_mb(levels[k].U, levels[k - 1].U);
  }

  // mass of component 0 of the coarse level of block b (sum u*dV over local fabs;
  // replicated coarse -> local sum = total, otherwise all_reduce).
  Real mass(std::size_t b) const {
    const MultiFab& U = block_levels_[b][0].U;
    const Real dV = geom_.dx() * geom_.dy();
    Real M = 0;
    for (int li = 0; li < U.local_size(); ++li) {
      const ConstArray4 u = U.fab(li).const_array();
      M += for_each_cell_reduce_sum(U.box(li),
                                    [u, dV] ADC_HD(int i, int j) { return u(i, j, 0) * dV; });
    }
    return replicated_coarse_ ? M : all_reduce_sum(M);
  }

 private:
  System system_;
  RhsAssembler rhs_assembler_;
  // Width of the SHARED aux channel: maximum of aux_comps<Model> over all the blocks (at least
  // kAuxBaseComps). The shared channel per level must be at least as wide as the most demanding
  // block so that load_aux<aux_comps<Model>> never reads out of bound in the AMR paths
  // (compute_face_fluxes, mf_apply_source, ...); a less demanding block simply ignores the extra
  // components. Exact analog of SystemAssembler::system_aux_comps (non-AMR path). Without a
  // block with an extra field, the width stays 3 -> allocation strictly bit-identical to history.
  static int system_aux_comps(const System& sys) {
    int w = kAuxBaseComps;
    sys.for_each_block([&](const auto& b) {
      using Model = std::decay_t<decltype(b.model)>;
      const int c = aux_comps<Model>();
      if (c > w)
        w = c;
    });
    return w;
  }
  // Populates the aux B_z component (index kAuxBaseComps) of the shared channel of EACH level from
  // bz_(x, y). B_z is static (external to the elliptic): set once (at the ctor / set_bz),
  // preserved by solve_fields (field_postprocess only writes phi/grad, comp 0..2; we re-set
  // after the coarse->fine injection which would copy a coarse B_z) and by the advance (the
  // AMR engine does not touch the aux). Each level has ITS geometry: level k = geom_.refine(1 << k),
  // same physical extents but refined index domain, so x_cell/y_cell point to the physical
  // center of the FINE cell. We fill the GROWN box (valid + halos) directly from
  // bz_(x, y): bz_ being a pure function of the physical position, its evaluation at the ghost
  // centers gives the physically correct B_z there too (independent of the BC of the fine patch,
  // without periodicity ambiguity on a patch domain). No-op if the aux width <= kAuxBaseComps (no
  // block reads B_z) or if bz_ is empty: RUNTIME guard (the width is only known at construction) ->
  // base model strictly bit-identical to history.
  void fill_bz() {
    if (!bz_ || aux_ncomp_ <= kAuxBaseComps)
      return;
    for (int k = 0; k < nlev_; ++k) {
      const Geometry gk = geom_.refine(1 << k);  // geometry of level k (dx = dx_coarse / 2^k)
      MultiFab& A = aux_[k];
      for (int li = 0; li < A.local_size(); ++li) {
        Fab2D& f = A.fab(li);
        // grown box (valid + halos): B_z(x,y) correct everywhere, geometry of level k.
        detail::fill_bz_box(f, f.grown_box(), gk, bz_);
      }
    }
  }

  Geometry geom_;
  Box2D dom_;
  Periodicity base_per_;
  BCRec bcPhi_, aux_bc_;
  bool replicated_coarse_;
  PoissonCadence cadence_;
  mutable int solve_count_ = 0;
  int macro_step_ = 0;  // macro-step counter (per-block stride cadence)
  Elliptic mg_;
  std::vector<std::vector<AmrLevelMP>> block_levels_;  // [block][level]
  std::vector<MultiFab> aux_;                          // [level], shared
  int aux_ncomp_ =
      kAuxBaseComps;  // width of the shared aux channel (max aux_comps over the blocks)
  int nlev_ = 0;
  std::function<Real(Real, Real)> bz_;  // external B_z(x, y) (empty if not provided)
};

// Default implicit on AMR: backward-Euler (Newton) on the model source, applied
// to EACH level of the block hierarchy. AMR pendant of ImplicitSourceStepper; same
// stability (unconditional for a linear relaxation). No solver on the user side.
/// Default implicit callback for AmrSystemCoupler::step: backward-Euler (Newton) on the model source,
/// applied to EACH level of the hierarchy, followed by a fine -> coarse cascade
/// (coverage consistency, cf. coupled_source_step). @p iters: Newton iterations per stage.
struct AmrImplicitSourceStepper {
  int iters = 2;

  template <class Coupler, class Block, class Levels>
  void operator()(Coupler& coupler, Block& block, Levels& levels, Real dt) const {
    const int nlev = static_cast<int>(levels.size());
    for (int k = 0; k < nlev; ++k)
      backward_euler_source(block.model, coupler.aux(k), levels[k].U, dt, iters);
    // COVERAGE INVARIANT (cf. coupled_source_step): the implicit source was
    // solved independently level by level, so the COVERED coarse cells
    // carry a ghost coarse source instead of the 2x2 average of their fine
    // children. We restore consistency by the same fine -> coarse cascade as the
    // transport-IMEX path, so that a covered coarse cell stays the coarse view of the fine
    // (otherwise amr_mass, sum of only the coarse level, double-counts the patch source).
    // Single-level: no covered cell, empty loop -> bit-identical to history.
    for (int k = nlev - 1; k >= 1; --k)
      mf_average_down_mb(levels[k].U, levels[k - 1].U);
  }
};

// "Advancing" alias (tutor feedback sec.8.2 B, sec.9.6): AmrSystemCoupler assembles (system
// Poisson + aux per level) AND advances (step, reflux, subcycling). Splitting into two classes is
// cosmetic and deferred (the unified class is validated bit-identical).
template <CoupledSystemLike System, class RhsAssembler, class Elliptic = GeometricMG>
using AmrSystemDriver = AmrSystemCoupler<System, RhsAssembler, Elliptic>;

}  // namespace adc
