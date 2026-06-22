#pragma once

#include <adc/core/state.hpp>  // kAuxBaseComps (B_z channel read by the condensed source stage)
#include <adc/core/types.hpp>  // Real
#include <adc/coupling/source/coupled_source_program.hpp>  // CoupledFreqKernel (per-cell coupled frequency)
#include <adc/mesh/for_each.hpp>  // reduce_max_cell (max mu over the cells, device-clean functor)
#include <adc/parallel/comm.hpp>  // all_reduce_min/max (global bounds: identical dt on all ranks)
#include <adc/runtime/detail/grid_context.hpp>  // GeometryMode (disk transport dispatch)

#include <stdexcept>  // std::runtime_error (disk mode requested without disk advance on a block)

#include <algorithm>  // std::min, std::max (CFL: min grid physical step, min dt over the blocks)
#include <cmath>      // std::isfinite, std::ceil (step_cfl / step_adaptive)
#include <limits>     // std::numeric_limits (per-block CFL: dt = min over the blocks)
#include <string>     // last_dt_bound (name of the active bound of the last step_cfl)
#include <vector>

/// @file
/// @brief SystemStepper: the TIME ADVANCE responsibility extracted from the god-class System::Impl
///        (audit Lot B, continuation of SystemFieldSolver #176). Extracted VERBATIM from python/system.cpp:
///        no change to numerics, to the CFL formula, to the stride/substeps cadence, to the
///        semantics of the macro-step counter, to the fences, nor to the order (solve_fields; advance;
///        source stage; couplings). STRICTLY bit-identical -- the code is moved as-is, only
///        access to the SHARED members of Impl (sp, fields_, aux, couplings, t, macro_step_, geom,
///        pgeom_, polar_) goes through the back-pointer owner_->.
///
/// CONTRACT / INVARIANTS
/// - ORCHESTRATES the time advance: step(dt), advance(dt, nsteps), step_cfl(cfl), step_adaptive(cfl),
///   plus the cadence helpers (stride_due), the Schur-condensed source stage (run_source_stage) and the
///   inter-species couplings (apply_couplings) that the steps invoke AFTER transport.
/// - READS (without owning) via owner_->: the block list (sp) and each advance closure (s.advance),
///   the elliptic solver (fields_, for solve_fields() at the head of the step and fields_.ell_phi() read by
///   the source stage), the SHARED aux and its B_z component (kAuxBaseComps), the coupling list, the
///   time t and the macro_step_ counter (which it advances), the geometry (Cartesian geom / polar pgeom_)
///   and the polar_ flag for the CFL physical step h.
/// - CFL PHYSICAL STEP h: Cartesian = min(dx, dy); POLAR = min(dr, r_min * dtheta) (the azimuthal step
///   r*dtheta is minimal at the inner radius r_min of the ring -> most constraining edge).
/// - MULTIRATE CADENCE INVARIANT (hold-then-catch-up): a block of cadence M is HELD as long as
///   (macro_step + 1) % M != 0, then advances by an effective step M*dt at the macro-step where
///   (macro_step + 1) % M == 0 (END of window). macro_step_ is incremented ONCE per macro-step, AFTER the
///   advance of the blocks and the couplings. DO NOT reorder solve_fields; advance; run_source_stage;
///   apply_couplings; t += dt; ++macro_step_.
/// - PER-BLOCK CFL FORMULA (substeps-aware, post-#121): dt <= cfl * h * substeps_b / (stride_b * w_b);
///   the global dt is the min over the evolving blocks. PRESERVED as is.
///
/// Since System::Impl stays PRIVATE to python/system.cpp, this helper is a TEMPLATE parameterized on the
/// real Impl type (same technique as system_field_solver / native_loader): python/system.cpp instantiates
/// it with System::Impl after defining Impl. owner_ is an Impl* (the helper lifetime is subordinate to
/// that of Impl). System::step / advance / step_cfl / step_adaptive become simple delegations to stepper_.

namespace adc {
namespace stepper {

/// Time SPLITTING policy of the macro-step (hyperbolic transport H + source stage S).
///  - Lie: H(dt); S(dt) once (Godunov, 1st order). THIS IS THE DEFAULT, bit-identical to
///              history: a single solve_fields at the head of the step, advance then run_source_stage
///              interleaved in the same block loop (cf. step()).
///  - Strang: H(dt/2); S(dt); H(dt/2) (symmetric, 2nd order as soon as H and S are). Requires
///              RE-SOLVING solve_fields BETWEEN the stages (cf. step()): see the comment of the
///              Strang branch and docs/HOFFART_STEP_SEQUENCE.md (the SINGLE head solve_fields does not
///              suffice for the 2nd half-advance, which would otherwise read a stale phi).
enum class SplitScheme { Lie, Strang };

/// SystemStepper<Impl>: see the contract above. All methods are MEMBERS because they
/// share the step orchestration; accesses to the SHARED state of Impl go through owner_-> verbatim.
/// Templated on Impl to stay free of any dependency on the (private) definition of System::Impl.
template <class Impl>
class SystemStepper {
 public:
  /// @param owner back-pointer to System::Impl (lifetime subordinate to that of Impl).
  explicit SystemStepper(Impl* owner) : owner_(owner) {}

  /// Chooses the time splitting policy (default Lie = bit-identical). See SplitScheme.
  void set_scheme(SplitScheme scheme) { scheme_ = scheme; }
  SplitScheme scheme() const { return scheme_; }

  /// True if a block of cadence @p stride CATCHES UP at this macro-step (END of window).
  /// STRIDE SEMANTICS = HOLD-THEN-CATCH-UP (catch-up at the END of the window). A block of cadence M is
  /// HELD (not advanced) on the macro-steps where (macro_step + 1) % M != 0, then advances by an effective
  /// step M*dt at the macro-step where (macro_step + 1) % M == 0, i.e. at the END of its window of M
  /// macro-steps. At macro-step k, the system time is (k+1)*dt and the block that CATCHES UP has then
  /// advanced by the same cumulative (k+1)*dt: it is temporally CONSISTENT with the fast blocks, never
  /// "in the future". (The old semantics advanced at the START of the window, macro_step % M == 0: at k=0
  /// the block already advanced M*dt while the system advanced only dt -> anticipated block, wrong
  /// Poisson/source coupling.)
  static bool stride_due(int macro_step, int stride) { return (macro_step + 1) % stride == 0; }

  /// Inter-species COUPLING sources: applied by SPLITTING (one explicit additive step of dt)
  /// AFTER the transport of each block. Each coupling is a for_each_cell (DEVICE kernel) reading /
  /// updating several blocks at the same point; they order after the transport on the same
  /// execution space, hence no prior device_fence (no more host access).
  void apply_couplings(Real dt) {
    if (owner_->couplings.empty())
      return;
    for (auto& c : owner_->couplings)
      c(dt);
  }

  /// Step bound from PER-CELL COUPLED FREQUENCIES (CoupledSource.frequency with an Expr,
  /// refinement of the CONSTANT frequency). For each registered program: reduces the MAX of mu(U)
  /// over the LOCAL fabs of the FIRST input block (CoupledFreqKernel, named device-clean functor;
  /// same MPI-safe convention as apply_couplings), GLOBAL all_reduce_max, then dt <= cfl / max(mu).
  /// Updates @p dt (and @p reason if non-null) if the bound is tighter. max(mu) <= 0 = no
  /// bound this step. Reason "coupled_source:<label>" -- SAME prefix as the constant frequency, for a
  /// uniform diagnostic. Per-cell counterpart of the constant loop of step_cfl / step_adaptive;
  /// no per-cell source registered -> empty loop, bit-identical trajectory.
  ///
  /// MPI: all_reduce_max is called by ALL ranks, the SAME number of times (coupled_freq_exprs_ is
  /// identical on all ranks) -> symmetric collective, identical dt everywhere (no deadlock). A
  /// rank with no local box reduces m=0 (neutral for MAX). WARNING: the Array4 are rebuilt at
  /// EACH step (the fabs may be reallocated), like apply_couplings.
  void apply_coupled_freq_expr_bounds(double cfl, double& dt, std::string* reason) const {
    Impl* P = owner_;
    for (const auto& ce : P->coupled_freq_exprs_) {
      Real m = 0;
      if (ce.n_in > 0) {
        auto& Uref = P->sp[static_cast<std::size_t>(ce.ins[0].sidx)].U;
        for (int li = 0; li < Uref.local_size(); ++li) {
          CoupledFreqKernel kern;
          kern.n_in = ce.n_in;
          kern.n_const = static_cast<int>(ce.kconsts.size());
          for (int c = 0; c < ce.n_in; ++c) {
            kern.in[c] = P->sp[static_cast<std::size_t>(ce.ins[static_cast<std::size_t>(c)].sidx)]
                             .U.fab(li)
                             .array();
            kern.in_comp[c] = ce.ins[static_cast<std::size_t>(c)].comp;
          }
          for (int c = 0; c < kern.n_const; ++c)
            kern.consts[c] = ce.kconsts[static_cast<std::size_t>(c)];
          kern.prog = ce.prog;
          m = std::max(m, reduce_max_cell(Uref.box(li), kern));
        }
      } else {
        // Program WITHOUT an input field (constant frequency expressed in bytecode): evaluated once
        // on the constants alone (no box to traverse); identical on all ranks.
        Real reg[kCsMaxReg];
        const int nc = static_cast<int>(ce.kconsts.size());
        for (int c = 0; c < nc; ++c)
          reg[c] = ce.kconsts[static_cast<std::size_t>(c)];
        const Real mu0 = ce.prog.eval(reg);
        if (mu0 > Real(0))
          m = mu0;
      }
      const double mu = all_reduce_max(static_cast<double>(m));  // ALL ranks (collective symmetry)
      if (mu > 0.0) {
        const double dt_cs = cfl / mu;
        if (dt_cs < dt) {
          dt = dt_cs;
          if (reason)
            *reason = "coupled_source:" + ce.label;
        }
      }
    }
  }

  /// MIN physical step of the grid, shared by step_cfl / step_adaptive: Cartesian = min(dx, dy);
  /// POLAR = min(dr, r_min * dtheta) -- the azimuthal physical step r*dtheta is minimal at the inner
  /// radius r_min of the ring (the most constraining edge for the CFL). Reads rank-local geometry
  /// only (no collective).
  Real cfl_grid_h() const {
    Impl* P = owner_;
    return P->polar_ ? std::min(P->pgeom_.dr(), P->pgeom_.r_min * P->pgeom_.dtheta())
                     : std::min(P->geom.dx(), P->geom.dy());
  }

  /// GLOBAL step bounds (System::add_dt_bound): multi-block coupling, Schur/Poisson, AMR/scheduler.
  /// One HOST evaluation per step and per bound; <= 0 or non-finite = does not constrain this step
  /// (neutralized to +inf BEFORE the global min). ALL_REDUCE_MIN mandatory: the callback is
  /// evaluated PER RANK (it may read a rank-local state); without the global min each rank would
  /// choose a different dt -> desynchronized step collectives (Krylov / fill_boundary) -> MPI
  /// deadlock. In serial all_reduce_min is the identity (bit-identical). @p reason, if non-null, is
  /// set to "global:<label>" for the winning bound (step_cfl tracks it; step_adaptive passes nullptr).
  /// MPI: dt_bounds_ is identical on all ranks and `if(!g.fn)` is rank-uniform, so the collective is
  /// symmetric (same count/order on every rank) -- factoring keeps the deadlock-safety unchanged.
  void apply_global_dt_bounds(double& dt, std::string* reason) const {
    Impl* P = owner_;
    for (const auto& g : P->dt_bounds_) {
      if (!g.fn)
        continue;
      double v = g.fn();
      if (!(v > 0.0) || !std::isfinite(v))
        v = std::numeric_limits<double>::infinity();
      v = all_reduce_min(v);
      if (v < dt) {
        dt = v;
        if (reason)
          *reason = "global:" + g.label;
      }
    }
  }

  /// PROJECTION PONCTUELLE post-pas (ADC-177) : U <- project(U, aux) par bloc, appliquee UNE fois a
  /// la FIN de chaque macro-pas ENTIER (apres transport + etage source + couplages ; jamais par etage
  /// RK), sur les cellules VALIDES seulement. Les GHOSTS ne sont pas projetes : chaque consommateur
  /// de ghosts (residu de transport) refait fill_ghosts en tete d'evaluation (cf. BlockRhsEval), donc
  /// l'etat fantome est reconstruit du valide projete au pas suivant -- aucun fill_boundary ici.
  /// Appliquee a TOUS les blocs evolutifs munis d'une projection, y compris les blocs TENUS par leur
  /// cadence stride : leur etat peut avoir change via les couplages, et une projection etant
  /// IDEMPOTENTE par contrat (cf. HasPointwiseProjection), la re-application sur un etat deja projete
  /// est neutre. Bloc sans projection (s.project vide) : jamais interroge -- cout nul, les 4 pas
  /// (step / step_strang / step_cfl / step_adaptive) restent bit-identiques a l'historique.
  void apply_projections() {
    for (auto& s : owner_->sp) {
      if (!s.evolve)
        continue;  // bloc gele : fond fixe jamais modifie, rien a projeter
      if (s.project)
        s.project(s.U);
    }
  }

  /// Schur-CONDENSED SOURCE STAGE (OPT-IN, cf. set_source_stage). No-op if the block has no source
  /// stage (s.schur == nullptr): the default path stays BIT-IDENTICAL. Otherwise, AFTER the hyperbolic
  /// transport of the block (already played by s.advance), we play the AUTONOMOUS source stage
  /// (CondensedSchurSourceStepper, #126) on the post-transport state:
  ///   - state = s.U (rho frozen in the source, mom/E updated);
  ///   - phi    = the system Poisson potential (ell_phi(), warm start phi^n from solve_fields
  ///              at the head of step) -- the stage solves its OWN condensed operator and WRITES phi^{n+1}
  ///              into it, it does NOT call solve_fields again (no duplication);
  ///   - B_z    = aux channel at index kAuxBaseComps (populated + ghosts filled by solve_fields).
  /// theta/dt of the theta-scheme; dt = eff_dt (stride factor already included by the caller, like s.advance).
  void run_source_stage(typename Impl::Species& s, Real eff_dt) {
    // GEOMETRY DISPATCH (Path A step 2c): a block carries AT MOST ONE condensed source stage (set_source_stage
    // builds the Cartesian OR the polar one depending on the System geometry). The POLAR one
    // (PolarCondensedSchurSourceStepper, #212) has the SAME step(state, phi, bz, c_bz, theta, dt) signature as
    // the Cartesian one (#126): only the pointer changes. The Cartesian path stays BIT-IDENTICAL (schur_polar
    // == nullptr in Cartesian -> we take the original schur branch, unchanged).
    if (s.schur_polar) {
      s.schur_polar->step(s.U, owner_->fields_.ell_phi(), owner_->aux, s.schur_bz_comp,
                          static_cast<Real>(s.schur_theta), eff_dt);
      return;
    }
    if (s.schur) {
      s.schur->step(s.U, owner_->fields_.ell_phi(), owner_->aux, s.schur_bz_comp,
                    static_cast<Real>(s.schur_theta), eff_dt);
      return;
    }
    // GENERIC SOURCE STAGE (fallback): plays ONLY if NO condensed Schur stage (production path
    // UNTOUCHED, bit-identical). Advances the block source stage IN PLACE on eff_dt. nullptr
    // (default) -> no-op, as before. Used by generic splitting (adc.Strang) and order tests.
    if (s.source_step)
      s.source_step(s.U, eff_dt);
  }

  /// TRANSPORT ADVANCE of block @p s over @p dt in @p n substeps, DISPATCHED by the System geometry
  /// mode (worksite T5-PR3). This is the SOLE wiring point of the disk in the step (the 4 steps -- step,
  /// step_strang, step_cfl, step_adaptive -- go through here):
  ///   - None (default): s.advance (assemble_rhs, full Cartesian). BIT-IDENTICAL.
  ///   - Staircase, fixed disk: s.advance_masked (assemble_rhs_masked, 0/1 mask).
  ///   - CutCell, fixed disk: s.advance_eb (assemble_rhs_eb, cut-cell EB).
  /// An embedded-boundary mode requested WITHOUT a fixed domain (eb_set_ == false) FALLS BACK to
  /// s.advance: the mode alone (without set_disc_domain) must not change the transport. A mode with a
  /// fixed domain but on a block that DID NOT build the embedded-boundary advance (e.g. polar block /
  /// loaded from an earlier .so) raises an EXPLICIT error rather than SILENTLY playing the full path
  /// (the T2 footgun: believing the boundary active while the transport ignores it). The
  /// embedded-boundary advances MIMIC s.advance (same RK / IMEX scheme, same limiter / flux); only the
  /// transport residual is dispatched.
  void advance_transport_n(typename Impl::Species& s, Real dt, int n) {
    const GeometryMode mode = owner_->geometry_mode_;
    if (mode == GeometryMode::None || !owner_->eb_set_) {
      s.advance(s.U, dt,
                n);  // default path (or mode without a fixed embedded boundary): BIT-IDENTICAL
      return;
    }
    if (mode == GeometryMode::Staircase) {
      if (!s.advance_masked)
        throw std::runtime_error(
            "SystemStepper: embedded-boundary mode 'staircase' requested but block '" + s.name +
            "' exposes no masked transport advance (level-set transport not wired for this block)");
      s.advance_masked(s.U, dt, n);
      return;
    }
    // CutCell
    if (!s.advance_eb)
      throw std::runtime_error(
          "SystemStepper: embedded-boundary mode 'cutcell' requested but block '" + s.name +
          "' exposes no cut-cell EB transport advance (level-set transport not wired for this "
          "block)");
    s.advance_eb(s.U, dt, n);
  }

  /// TRANSPORT ADVANCE of block @p s over @p eff_dt in s.substeps substeps, dispatched by the mode (cf.
  /// advance_transport_n). Reuses s.substeps as the former s.advance of the step / step_cfl steps.
  void advance_transport(typename Impl::Species& s, Real eff_dt) {
    advance_transport_n(s, eff_dt, s.substeps);
  }

  /// HALF transport advance (Strang): dispatch of advance_transport over @p half_dt = eff_dt/2 --
  /// the disk mode ALSO honors the Strang path (H(dt/2) S(dt) H(dt/2)).
  void advance_transport_half(typename Impl::Species& s, Real eff_dt) {
    advance_transport_n(s, Real(0.5) * eff_dt, s.substeps);
  }

  /// FULL-STEP advance of every DUE block over @p dt: effective step eff_dt = stride * dt (cadence
  /// catch-up), then transport (dispatched by the geometry mode, cf. advance_transport) and the OPT-IN
  /// Schur source stage (no-op otherwise, cf. run_source_stage). Frozen blocks (!evolve) and HELD blocks
  /// (outside their stride window, cf. stride_due) are skipped. This is the verbatim per-block loop
  /// shared by step (Lie) and step_cfl. The two OTHER macro-steps cannot reuse it: step_strang splits
  /// transport into two HALF advances around a re-solved source stage (3 interleaved sub-loops), and
  /// step_adaptive subcycles each block n_b times (advance_transport_n). It introduces NO NEW collective:
  /// the underlying transport (halo exchanges) and Schur source stage (fill_ghosts + all_reduce in the
  /// condensed solve) DO communicate, but this helper iterates the same blocks in the same order, so each
  /// rank issues the SAME sequence of collectives as the inlined loops did -- no MPI desync, bit-identical.
  void advance_due_blocks(double dt) {
    Impl* P = owner_;
    for (auto& s : P->sp) {
      if (!s.evolve)
        continue;  // frozen block: not advanced
      if (!stride_due(P->macro_step_, s.stride))
        continue;                                     // hold: not at the stride window end
      const Real eff_dt = Real(dt) * Real(s.stride);  // catch-up: effective step s.stride * dt
      advance_transport(s,
                        eff_dt);  // transport DISPATCHED by the geometry mode (None: assemble_rhs)
      run_source_stage(s, eff_dt);  // OPT-IN: Schur-condensed source stage (no-op otherwise)
    }
  }

  /// One macro-step of length @p dt. ORDER INVARIANT per scheme (cf. SplitScheme):
  ///  - Lie (default, bit-identical): solve_fields; for each DUE block (stride cadence honored)
  ///    advance(dt) then run_source_stage(dt) interleaved; couplings; t += dt; ++macro_step.
  ///  - Strang: H(dt/2); S(dt); H(dt/2), with a solve_fields RE-SOLVED between each stage
  ///    (cf. step_strang() and docs/HOFFART_STEP_SEQUENCE.md).
  void step(double dt) {
    if (scheme_ == SplitScheme::Strang) {
      step_strang(dt);
      return;
    }
    Impl* P = owner_;
    // COUPLING / POISSON: solve_fields assembles f = Sum_s elliptic_rhs_s(U_s) on the CURRENT state of
    // each block. A HELD block (cadence M, outside the window end) contributes with its STALE state (its
    // last advance, thus frozen until its next catch-up): stale density / charge in the Poisson sum as
    // long as it has not caught up. Assumed stride choice (loose coupling of the slow block).
    P->solve_fields();
    advance_due_blocks(
        dt);  // DUE blocks: catch-up transport + opt-in source stage (shared with step_cfl)
    apply_couplings(Real(dt));  // inter-species coupled sources (splitting), after transport
    apply_projections();  // projection ponctuelle POST-PAS ENTIER (ADC-177) ; no-op sans projection
    P->t += dt;
    P->macro_step_++;
  }

  /// One STRANG macro-step (symmetric, 2nd order): H(dt/2); S(dt); H(dt/2). Reuses s.advance for
  /// the HALF transport advances and run_source_stage for the FULL source stage (no new stepper).
  ///
  /// phi CONSISTENCY (critical point, cf. docs/HOFFART_STEP_SEQUENCE.md): the potential phi / the aux
  /// fields (grad phi) that the transport READS are populated by solve_fields from the CURRENT density.
  /// The SINGLE solve_fields at the head of the step (sufficient for Lie splitting, where a single
  /// transport advance follows) DOES NOT SUFFICE here: between the 1st half-advance and the 2nd, the
  /// density has changed (1st half-advance + source stage), so the 2nd half-advance would read a STALE
  /// phi. We thus RE-SOLVE solve_fields BEFORE each stage that consumes phi:
  ///   1. solve_fields()  -> phi consistent with rho^n            (for H(dt/2))
  ///   2. H(dt/2)         (s.advance over eff_dt/2)
  ///   3. solve_fields()  -> phi consistent with rho after H(dt/2) (for S(dt): warm start / aux field)
  ///   4. S(dt)           (run_source_stage over eff_dt; the Schur stage WRITES phi^{n+1})
  ///   5. solve_fields()  -> phi consistent with rho after S(dt)   (for the 2nd H(dt/2))
  ///   6. H(dt/2)         (s.advance over eff_dt/2)
  /// Without step 5, the 2nd half-advance would read either the Schur phi (overwritten at the next step
  /// anyway) or a phi of stale rho: the Strang symmetry (thus the 2nd order) would be broken. Steps
  /// 1/3/5 are SYSTEM solve_fields (sum over all blocks), outside the block loop.
  ///
  /// stride CADENCE: evaluated ONCE per macro-step (stride_due at the START), so that the two
  /// half-advances and the source stage of one macro-step concern the SAME set of DUE blocks. The
  /// effective step eff_dt = dt * stride (catch-up) is identical to Lie; only the transport is split
  /// into two halves eff_dt/2.
  void step_strang(double dt) {
    Impl* P = owner_;
    // (1) phi consistent with rho^n, for the 1st transport half-advance.
    P->solve_fields();
    // (2) H(dt/2): 1st transport half-advance of each DUE block. s.substeps substeps (unchanged).
    // Dispatched by the geometry mode (None: assemble_rhs; Staircase/CutCell: disk operator).
    for (auto& s : P->sp) {
      if (!s.evolve)
        continue;
      if (!stride_due(P->macro_step_, s.stride))
        continue;
      const Real eff_dt = Real(dt) * Real(s.stride);
      advance_transport_half(s, eff_dt);
    }
    // (3) phi RE-SOLVED on the post-H(dt/2) density, for the source stage.
    P->solve_fields();
    // (4) S(dt): FULL source stage of each DUE block (no-op if no Schur stage, like Lie).
    for (auto& s : P->sp) {
      if (!s.evolve)
        continue;
      if (!stride_due(P->macro_step_, s.stride))
        continue;
      const Real eff_dt = Real(dt) * Real(s.stride);
      run_source_stage(s, eff_dt);
    }
    // (5) phi RE-SOLVED on the post-source density: WITHOUT this solve the 2nd half-advance would read a
    //     stale phi (cf. docs/HOFFART_STEP_SEQUENCE.md, the single head solve_fields is insufficient).
    P->solve_fields();
    // (6) H(dt/2): 2nd transport half-advance, closing the symmetric Strang step. SAME dispatch.
    for (auto& s : P->sp) {
      if (!s.evolve)
        continue;
      if (!stride_due(P->macro_step_, s.stride))
        continue;
      const Real eff_dt = Real(dt) * Real(s.stride);
      advance_transport_half(s, eff_dt);
    }
    apply_couplings(
        Real(dt));  // inter-species coupled sources (splitting), after the symmetric step
    // Projection ponctuelle POST-PAS ENTIER (ADC-177) : UNE application apres le pas Strang complet
    // H(dt/2) S(dt) H(dt/2), jamais entre les etages (semantique post-pas, pas post-etage).
    apply_projections();
    P->t += dt;
    P->macro_step_++;
  }

  /// Advances by @p nsteps macro-steps of length @p dt (loop over step).
  void advance(double dt, int nsteps) {
    for (int s = 0; s < nsteps; ++s)
      step(dt);
  }

  /// One macro-step at CFL dt: dt = min over the evolving blocks of the block step BOUNDS, then advances
  /// like step. @return the dt used. SUBSTEPS-AWARE (post-#121): bit-identical to the old
  /// formula only for substeps=1 (cf. backward-compatibility note).
  ///
  /// STEP POLICY (audit 2026-06, step_cfl worksite): the historical TRANSPORT bound
  /// dt <= cfl*h*substeps_b/(stride_b*w_b) stays the base, but the step now AGGREGATES, per block:
  ///   - the SOURCE FREQUENCY bound (s.source_frequency, HasSourceFrequency trait):
  ///     effective substep stride*dt/substeps <= cfl/mu -> dt <= cfl*substeps/(stride*mu), WITHOUT h
  ///     (a local source bounds in 1/time, not in length/time);
  ///   - the direct ADMISSIBLE STEP (s.stability_dt, HasStabilityDt trait):
  ///     stride*dt/substeps <= dt_adm -> dt <= dt_adm*substeps/stride, WITHOUT cfl (the model already
  ///     declares an admissible step);
  ///   - the CFL speed itself can be the declared STABILITY speed (HasStabilitySpeed
  ///     trait): s.max_speed is then wired onto stability_speed (cf. make_max_speed).
  /// Then the GLOBAL bounds (P->dt_bounds_: multi-block coupling, Schur/Poisson, AMR/scheduler,
  /// set by System::add_dt_bound): dt <= fn() each, one HOST evaluation per step (no per-cell
  /// callback). A block / a system WITHOUT optional bounds keeps a step STRICTLY identical to
  /// history (empty functions are not queried). The ACTIVE bound of the last step is consultable via
  /// last_dt_bound() ("transport:<block>", "source_frequency:<block>",
  /// "stability_dt:<block>", "global:<label>", "degenerate").
  /// 2D CONVENTION NOTE (audit ADC-182): the per-cell speed is w = max(wx, wy) -- NOT the
  /// sum. In unsplit 2D the effective Courant number thus reaches 2*cfl when wx ~ wy:
  /// cfl = 0.4 (case default) stays < 1 (safe), this is also the convention of the sweep
  /// references (HLL step); cfl >= 0.5 in unsplit 2D is MARGINAL -- to avoid without study.
  double step_cfl(double cfl) {
    Impl* P = owner_;
    P->solve_fields();
    // MIN physical step of the grid (Cartesian min(dx,dy) / polar min(dr, r_min*dtheta), cf.
    // cfl_grid_h). The rest of the CFL formula (per block, substeps/stride) is unchanged.
    const Real h = cfl_grid_h();
    // PER-BLOCK CFL, STRIDE AND SUBSTEPS FACTOR INCLUDED. A block of cadence M advances by an effective
    // step M*dt in substeps_b substeps, so each substep is worth stride_b * dt / substeps_b: the stable
    // condition per substep is stride_b * dt / substeps_b <= cfl * h / w_b, that is
    //   dt <= cfl * h * substeps_b / (stride_b * w_b).
    // The GLOBAL dt is the min over the evolving blocks (the most constraining). Without this, the step
    // computed on w_max alone then multiplied by M would violate the CFL by a factor M on the stride block.
    //
    // BACKWARD COMPATIBILITY (post-#121). The formula is SUBSTEPS-AWARE: with substeps_b > 1, the dt
    // returned is substeps_b times larger than the old formula dt = cfl*h/(stride*w).
    // bit-identical only for substeps=1 (at any stride); step_cfl is now substeps-aware
    // (dt = cfl*h*substeps/(stride*w)), so a step_cfl run with substeps>1 advances a larger dt
    // than before #121 (CFL-maximal step, each substep at the stability limit).
    // To reproduce a run calibrated with the old formula, use step(dt) with the explicit historical
    // dt, NOT step_cfl.
    double dt = std::numeric_limits<double>::infinity();
    std::string reason = "degenerate";
    for (auto& s : P->sp) {
      if (!s.evolve)
        continue;  // frozen block: does not constrain the step
      const Real w = std::max(s.max_speed(s.U), kCflSpeedFloor);
      double dt_b = cfl * static_cast<double>(h) * static_cast<double>(s.substeps) /
                    (static_cast<double>(s.stride) * static_cast<double>(w));
      const char* why = "transport";
      // SOURCE FREQUENCY bound (optional; mu <= 0 = does not constrain).
      if (s.source_frequency) {
        const Real mu = s.source_frequency(s.U);
        if (mu > Real(0)) {
          const double dt_src = cfl * static_cast<double>(s.substeps) /
                                (static_cast<double>(s.stride) * static_cast<double>(mu));
          if (dt_src < dt_b) {
            dt_b = dt_src;
            why = "source_frequency";
          }
        }
      }
      // Direct ADMISSIBLE STEP (optional; <= 0 = does not constrain; cfl NOT applied).
      if (s.stability_dt) {
        const Real db = s.stability_dt(s.U);
        if (db > Real(0)) {
          const double dt_adm = static_cast<double>(db) * static_cast<double>(s.substeps) /
                                static_cast<double>(s.stride);
          if (dt_adm < dt_b) {
            dt_b = dt_adm;
            why = "stability_dt";
          }
        }
      }
      if (dt_b < dt) {
        dt = dt_b;
        reason = std::string(why) + ":" + s.name;
      }
    }
    // DECLARED frequencies of the coupled sources (CoupledSource.frequency): the couplings
    // apply ONCE per MACRO-step (apply_couplings(dt)), so the bound applies to the
    // macro-dt directly: dt <= cfl / mu (NO substeps/stride factor -- those apply
    // only to the block subcycled transport, not to the coupling splitting).
    for (const auto& cs : P->coupled_freqs_) {
      if (!(cs.mu > 0.0))
        continue;
      const double dt_cs = cfl / cs.mu;
      if (dt_cs < dt) {
        dt = dt_cs;
        reason = "coupled_source:" + cs.label;
      }
    }
    // PER-CELL frequencies (CoupledSource.frequency with an Expr): mu(U) reduced (MAX) per cell at
    // this step, global all_reduce_max, dt <= cfl / max(mu). Same reason "coupled_source:<label>" as the
    // constant. No per-cell source -> no-op (bit-identical).
    apply_coupled_freq_expr_bounds(cfl, dt, &reason);
    // GLOBAL bounds (System::add_dt_bound): all_reduce_min over the registered bounds, tracking the
    // winning reason (see apply_global_dt_bounds for the MPI deadlock-safety rationale).
    apply_global_dt_bounds(dt, &reason);
    if (!std::isfinite(dt)) {
      dt = cfl * static_cast<double>(h) /
           static_cast<double>(kCflSpeedFloor);  // all frozen: degenerate step
      reason = "degenerate";
    }
    last_dt_reason_ = std::move(reason);
    advance_due_blocks(
        dt);  // DUE blocks: catch-up transport + opt-in source stage (shared with step Lie)
    apply_couplings(Real(dt));
    apply_projections();  // projection ponctuelle POST-PAS ENTIER (ADC-177) ; no-op sans projection
    P->t += dt;
    P->macro_step_++;
    return dt;
  }

  /// One MULTIRATE macro-step: the macro-step = stable step of the SLOWEST block; each faster block
  /// is subcycled n_b = ceil(stride_b * w_b / w_min) times. aux frozen over the macro-step (coupling
  /// once-per-step). @return the macro-step.
  ///
  /// OPTIONAL BOUNDS (audit 2026-06): like step_cfl, the macro-step is then REDUCED by the
  /// block bounds (source_frequency / stability_dt, applied to the effective substep
  /// stride_b*macro_dt/n_b -- n_b does not depend on dt, so the clamp is exact) and by the GLOBAL
  /// bounds (P->dt_bounds_). Without optional bounds, macro_dt is STRICTLY historical.
  double step_adaptive(double cfl) {
    Impl* P = owner_;
    P->solve_fields();
    // Multirate: macro-step = stable step of the SLOWEST block; each faster block is
    // subcycled n_b. aux frozen over the macro-step (coupling once-per-step). STRIDE SEMANTICS =
    // hold-then-catch-up: a block of cadence M is HELD as long as (macro_step + 1) % M != 0, then
    // advances by an effective step M*macro_dt at the window end (cf. stride_due).
    Real wmin = Real(1e30);
    std::vector<Real> wb;
    wb.reserve(P->sp.size());
    for (auto& s : P->sp) {
      const Real w = s.evolve ? s.max_speed(s.U) : Real(0);  // frozen block: out of cadence
      wb.push_back(w);
      if (s.evolve)
        wmin = std::min(wmin, w);
    }
    if (wmin >= Real(1e30))
      wmin = kCflSpeedFloor;      // no evolving block (all frozen)
    const Real h = cfl_grid_h();  // Cartesian min(dx,dy) / polar min(dr, r_min*dtheta)
    double macro_dt = cfl * static_cast<double>(h) / static_cast<double>(wmin);
    // OPTIONAL block bounds: each block subcycles n_b times its effective step
    // stride_b*macro_dt; the substep stride_b*macro_dt/n_b must satisfy the source / admissible
    // step bounds of the block. n_b (formula identical to the advance loop below) does not depend
    // on macro_dt: the clamp is done BEFORE the advance, n_b stays consistent.
    for (std::size_t b = 0; b < P->sp.size(); ++b) {
      auto& s = P->sp[b];
      if (!s.evolve)
        continue;
      if (!s.source_frequency && !s.stability_dt)
        continue;
      int n = static_cast<int>(
          std::ceil(static_cast<double>(s.stride) * static_cast<double>(wb[b] / wmin)));
      if (n < 1)
        n = 1;
      if (s.source_frequency) {
        const Real mu = s.source_frequency(s.U);
        if (mu > Real(0))
          macro_dt =
              std::min(macro_dt, cfl * static_cast<double>(n) /
                                     (static_cast<double>(s.stride) * static_cast<double>(mu)));
      }
      if (s.stability_dt) {
        const Real db = s.stability_dt(s.U);
        if (db > Real(0))
          macro_dt = std::min(macro_dt, static_cast<double>(db) * static_cast<double>(n) /
                                            static_cast<double>(s.stride));
      }
    }
    // Declared frequencies of the coupled sources (cf. step_cfl): bound on the MACRO-step.
    for (const auto& cs : P->coupled_freqs_) {
      if (cs.mu > 0.0)
        macro_dt = std::min(macro_dt, cfl / cs.mu);
    }
    // PER-CELL frequencies (Expr): MAX of mu(U) per cell, all_reduce_max, bound on the macro-step
    // (cf. step_cfl). step_adaptive does not track the active reason -> reason = nullptr.
    apply_coupled_freq_expr_bounds(cfl, macro_dt, nullptr);
    // GLOBAL bounds (System::add_dt_bound), like step_cfl; step_adaptive does not track the active
    // reason -> nullptr (same all_reduce_min, identical dt on all ranks).
    apply_global_dt_bounds(macro_dt, nullptr);
    for (std::size_t b = 0; b < P->sp.size(); ++b) {
      auto& s = P->sp[b];
      if (!s.evolve)
        continue;  // frozen block: not advanced
      if (!stride_due(P->macro_step_, s.stride))
        continue;  // hold: not at the stride window end
      // Stable subcycling of the EFFECTIVE step M*macro_dt: each substep must satisfy
      // M*macro_dt / n <= cfl*h / w_b, i.e. n >= ceil(M * w_b / w_min). The stride factor M is thus
      // carried by the number of substeps (without it, n on w_b/w_min alone would violate the CFL by a factor M).
      int n = static_cast<int>(
          std::ceil(static_cast<double>(s.stride) * static_cast<double>(wb[b] / wmin)));
      if (n < 1)
        n = 1;
      const Real eff_dt = Real(macro_dt) * Real(s.stride);  // catch-up: effective step M*macro_dt
      advance_transport_n(s, eff_dt,
                          n);  // transport DISPATCHED by the geometry mode (n adaptive substeps)
      run_source_stage(s, eff_dt);  // OPT-IN: Schur-condensed source stage (no-op otherwise)
    }
    apply_couplings(Real(macro_dt));
    apply_projections();  // projection ponctuelle POST-MACRO-PAS (ADC-177), pas par sous-cycle
    P->t += macro_dt;
    P->macro_step_++;
    return macro_dt;
  }

  /// Name of the ACTIVE bound (the one that fixed dt) of the last step_cfl: "transport:<block>",
  /// "source_frequency:<block>", "stability_dt:<block>", "global:<label>", "degenerate", or "" if
  /// no step_cfl has run yet. Diagnostic (System::last_dt_bound).
  const std::string& last_dt_reason() const { return last_dt_reason_; }

 private:
  Impl* owner_;
  SplitScheme scheme_ = SplitScheme::Lie;  // default Lie (Godunov): bit-identical to history
  std::string last_dt_reason_;             // active bound of the last step_cfl (diagnostic)
};

}  // namespace stepper
}  // namespace adc
