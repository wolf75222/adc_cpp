#pragma once

#include <adc/core/cold.hpp>  // ADC_COLD_FN: COLD block-builder no-optimize attribute (ADC-337)
#include <adc/core/types.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/for_each.hpp>  // for_each_cell (projection ponctuelle post-pas, ADC-177)
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/numerical_flux.hpp>
#include <adc/numerics/reconstruction.hpp>
#include <adc/numerics/spatial_operator.hpp>
#include <adc/numerics/spatial_operator_eb.hpp>  // assemble_rhs_eb (cut-cell EB) + detail::DiscLevelSet (T5-PR2)
#include <adc/numerics/time/implicit_stepper.hpp>
#include <adc/numerics/time/time_steppers.hpp>
#include <adc/runtime/dispatch_tags.hpp>  // UNIQUE registry of tags (validate_limiter/riemann, limiter_n_ghost)
#include <adc/runtime/grid_context.hpp>  // GridContext + BlockClosures (shared lightweight header)
#include <adc/numerics/embedded_boundary.hpp>  // detail::DiscDomain (built-in level-set domain instance)

#include <cmath>  // std::sqrt (ARS(2,2,2) coefficients: gamma = 1 - 1/sqrt(2), host)
#include <functional>
#include <memory>  // std::shared_ptr (shared scratch of the HLL wave speed cache, opt-in)
#include <stdexcept>
#include <string>
#include <type_traits>  // std::is_same_v (cache engages only for the HLL flux)
#include <utility>
#include <vector>

/// @file
/// @brief Builds the closures of a block (time advance + residual + Poisson contribution) from a
///        COMPILED model (CompositeModel) and a grid context.
///
/// This code used to live in System::Impl; it is extracted into a header so that the SAME template
/// path (assemble_rhs<Limiter, Flux>, inlinable and device-ready) is instantiable from an EXTERNAL
/// TRANSLATION UNIT. It is the brick that lets a DSL-generated model be compiled AOT (ahead-of-time)
/// and then plugged into the System via the PRODUCTION path (HLLC/Roe flux, order 2, GPU), no longer
/// only via the virtual host path of the dynamic block.
///
/// The System remains the sole owner of the mesh and the aux; GridContext only carries immutable
/// copies of them (domain, BC, geometry) and a non-owning POINTER to the aux (stable address,
/// lifetime longer than the block).

namespace adc {

// GridContext and BlockClosures: defined in adc/runtime/grid_context.hpp (lightweight header, also
// included by system.hpp to expose grid_context() / install_block() without pulling in the numerics).

namespace detail {
/// Residual functor -div F + S (fill_ghosts then assemble_rhs), passed TO THE TimeStepper as RhsEval.
/// NAMED FUNCTOR (not a lambda): this is what take_step receives and what triggers the instantiation
/// of assemble_rhs<Limiter, Flux> (and its device AssembleRhsKernel). First-instantiated from an
/// EXTERNAL TU (add_compiled_model), a lambda here makes nvcc choke on emitting the nested device
/// kernel (Heisenbug: OK Serial + compute-sanitizer, segfault at Cuda run time). A class has a stable
/// instantiation context -> robust device codegen. Body identical to the former lambda -> residual
/// bit-identical to add_block on CPU (and, intended, on device).
template <class Limiter, class Flux, class Model>
struct BlockRhsEval {
  Model model;
  const GridContext* ctx;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  /// Per-cell wave speed scratch (HLL cache, opt-in). nullptr (default) -> per-face path strictly
  /// unchanged. Non-null ONLY for the HLL flux (cf. build_block): the cached branch is instantiated
  /// only for Flux == HLLFlux, so model.wave_speeds is always present there.
  std::shared_ptr<MultiFab> ws_cache;
  void operator()(MultiFab& U, MultiFab& R) const {
    fill_ghosts(U, ctx->dom, ctx->bc);
    if constexpr (std::is_same_v<Flux, HLLFlux>) {
      if (ws_cache) {
        // Re-allocate the scratch at the current layout (4 components, 1 ghost): covers an AMR regrid
        // or a first call (shared_ptr to an empty MultiFab). Otherwise reuse the existing allocation.
        if (ws_cache->local_size() != U.local_size() || ws_cache->ncomp() != 4)
          *ws_cache = MultiFab(U.box_array(), U.dmap(), 4, 1);
        assemble_rhs_hll_cached<Limiter>(model, U, *ctx->aux, ctx->geom, R, *ws_cache, recon_prim,
                                         pos_floor);
        return;
      }
    }
    assemble_rhs<Limiter, Flux>(model, U, *ctx->aux, ctx->geom, R, recon_prim, pos_floor);
  }
};

/// EXPLICIT advance: n substeps of the @c Stepper stepper (SSPRK2 by default, SSPRK3 optional) on the
/// transport+source residual. The RK scheme is a template parameter (NAMED FUNCTOR from the core:
/// SSPRK2Step / SSPRK3Step) -> same device-clean contract as SSPRK2Step. SSPRK2 reproduces the
/// historical advance exactly (bit-identical).
template <class Limiter, class Flux, class Model, class Stepper = SSPRK2Step>
struct AdvanceExplicit {
  Model m;
  GridContext ctx;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  std::shared_ptr<MultiFab> ws_cache;  ///< HLL wave speed cache (opt-in); nullptr -> per-face path
  void operator()(MultiFab& U, Real dt, int n) const {
    const Real h = dt / static_cast<Real>(n);
    const BlockRhsEval<Limiter, Flux, Model> rhs{m, &ctx, recon_prim, pos_floor, ws_cache};
    run_explicit_substeps<Stepper>(rhs, U, h, n);
  }
};

/// IMEX advance: per substep, EXPLICIT half-step (source-free transport) + stiff IMPLICIT source.
/// @c mask: implicit mask CARRIED BY THE BLOCK (overrides the model default is_implicit), resolved
/// once when the block is added against its variable names/roles. Inactive mask (default) ->
/// backward_euler falls back on model_is_implicit -> advance bit-identical to the historical one.
template <class Limiter, class Flux, class Model>
struct AdvanceImex {
  Model m;
  GridContext ctx;
  bool recon_prim;
  ImplicitMask<Model::n_vars> mask{};
  NewtonOptions nopts{};            // block Newton options (defaults = historical: 2 iters, 1e-7)
  NewtonReport* nreport = nullptr;  // OPT-IN diagnostics (stable address, owned by System::Impl)
  Real pos_floor = Real(0);         ///< Zhang-Shu positivity limiter (<= 0: inactive)
  void operator()(MultiFab& U, Real dt, int n) const {
    const Real h = dt / static_cast<Real>(n);
    const BlockRhsEval<Limiter, Flux, SourceFreeModel<Model>> rhs{SourceFreeModel<Model>{m}, &ctx,
                                                                  recon_prim, pos_floor};
    if (nreport)
      nreport->reset();  // report AGGREGATED over the n substeps of THIS advance
    for (int s = 0; s < n; ++s) {
      ForwardEuler{}.take_step(rhs, U, h);  // explicit half-step: source-free transport
      backward_euler_source(m, *ctx.aux, U, h, nopts, mask,
                            nreport);  // implicit source (stiff relaxation)
    }
  }
};

/// IMEX-RK ARS(2,2,2) advance (Ascher, Ruuth, Spiteri 1997; "Implicit-explicit Runge-Kutta methods
/// for time-dependent partial differential equations", Appl. Numer. Math. 25): explicit transport
/// (L = -div F) coupled to the stiff implicit source (per-cell LOCAL backward-Euler), ORDER 2. It is a
/// family DISTINCT from and PARALLEL to AdvanceImex (which remains the default order-1 backward-Euler,
/// UNTOUCHED and bit-identical). Coefficients: gamma = 1 - 1/sqrt(2), delta = 1 - 1/(2 gamma).
///
/// Tableaux (stiffly accurate, implicit part SDIRK; c = [0, gamma, 1] for both):
///   explicit: A_E = [[0, 0, 0], [gamma, 0, 0], [delta, 1-delta, 0]],  b_E = [delta, 1-delta, 0]
///   implicit: A_I = [[0, 0, 0], [0, gamma, 0], [0, 1-gamma, gamma]],  b_I = [0, 1-gamma, gamma]
///
/// b_E == last row of A_E and b_I == last row of A_I -> STIFFLY ACCURATE scheme -> the final solution
/// IS the last stage (U^{n+1} = U^(3)), no final recombination. With L = the SourceFreeModel transport
/// and S = the source of the full model, the per-stage recurrence is:
///   U^(1) = U^n                                        (first row of A_I is zero: no solve, S^(1) unused)
///   L1    = L(U^n)
///   U^(2) = U^n + dt*gamma*L1 + dt*gamma*S(U^(2))       (implicit solve: backward_euler_source at step
///                                                        dt*gamma on the base U^n + dt*gamma*L1)
///   L2    = L(U^(2))
///   U^(3) = U^n + dt*delta*L1 + dt*(1-delta)*L2 + dt*(1-gamma)*S^(2) + dt*gamma*S(U^(3))
///   U^{n+1} = U^(3)
/// The term dt*gamma*S^(2) is NOT re-evaluated: by construction of the stage-2 solve,
/// dt*gamma*S^(2) = U^(2) - base2 (the solve increment), so dt*(1-gamma)*S^(2) = ((1-gamma)/gamma) *
/// (U^(2) - base2). NO extra source kernel: we REUSE BlockRhsEval<SourceFreeModel> (transport, SAME
/// mechanism as the explicit half-step of AdvanceImex), backward_euler_source (local implicit solve)
/// and saxpy/lincomb (stages). Device-clean (no new kernel).
///
/// FULLY IMPLICIT SOURCE: the partial IMEX mask is NOT wired here (the consistency relation
/// dt*gamma*S^(2) = U^(2) - base2 assumes a homogeneous stage solve; a per-component forward-backward
/// treatment would mix the explicit/implicit tableaux). System::add_block therefore rejects
/// implicit_vars/implicit_roles with time='imexrk_ars222'. The Newton options (nopts), on the other
/// hand, are carried through: they parametrize BOTH implicit stage solves.
///
/// The stage MultiFabs are allocated ONCE per advance (outside the substep loop): Un (U^n), L1/L2
/// (transport residuals), base2 (stage-2 base, re-read at stage 3) without ghosts (read on valid
/// cells); work (stage state) with the ghosts of U (it is passed to the transport residual).
template <class Limiter, class Flux, class Model>
struct AdvanceImexRkArs222 {
  Model m;
  GridContext ctx;
  bool recon_prim;
  NewtonOptions nopts{};            // Newton options of the stage solves (defaults = historical)
  NewtonReport* nreport = nullptr;  // OPT-IN diagnostics (stable address, owned by System::Impl)
  Real pos_floor = Real(0);         ///< Zhang-Shu positivity limiter (<= 0: inactive)
  void operator()(MultiFab& U, Real dt, int n) const {
    const Real h = dt / static_cast<Real>(n);
    const Real gamma = Real(1) - Real(1) / std::sqrt(Real(2));
    const Real delta = Real(1) - Real(1) / (Real(2) * gamma);
    const Real cS2 = (Real(1) - gamma) / gamma;  // factor of (U^(2) - base2) at stage 3
    const ImplicitMask<Model::n_vars> mask{};    // FULLY implicit source (inactive mask)
    // Source-free transport residual (L = -div F): SAME mechanism as the explicit half-step of AdvanceImex.
    const BlockRhsEval<Limiter, Flux, SourceFreeModel<Model>> rhs{SourceFreeModel<Model>{m}, &ctx,
                                                                  recon_prim, pos_floor};
    const int nc = U.ncomp();
    MultiFab Un(U.box_array(), U.dmap(), nc, 0);             // U^n
    MultiFab L1(U.box_array(), U.dmap(), nc, 0);             // L(U^n)
    MultiFab L2(U.box_array(), U.dmap(), nc, 0);             // L(U^(2))
    MultiFab base2(U.box_array(), U.dmap(), nc, 0);          // U^n + dt*gamma*L1
    MultiFab work(U.box_array(), U.dmap(), nc, U.n_grow());  // stage state (passed to transport)
    if (nreport)
      nreport->reset();  // report AGGREGATED over the substeps AND the 2 stage solves
    for (int s = 0; s < n; ++s) {
      // Stage 1: U^(1) = U^n; L1 = L(U^n).
      lincomb(Un, Real(1), U, Real(0), U);  // Un = U^n (valid cells)
      rhs(U, L1);                           // L1 = L(U^n)  (fill_ghosts(U) + assemble_rhs)
      // Stage 2: U^(2) = base2 + dt*gamma*S(U^(2)),  base2 = U^n + dt*gamma*L1.
      lincomb(base2, Real(1), Un, h * gamma, L1);     // base2 = U^n + dt*gamma*L1
      lincomb(work, Real(1), base2, Real(0), base2);  // work = base2
      backward_euler_source(m, *ctx.aux, work, h * gamma, nopts, mask, nreport);  // work = U^(2)
      rhs(work, L2);                                                              // L2 = L(U^(2))
      // Stage 3: U <- base3 = U^n + dt*delta*L1 + dt*(1-delta)*L2 + ((1-gamma)/gamma)*(U^(2) - base2).
      lincomb(U, Real(1), Un, h * delta, L1);  // U = U^n + dt*delta*L1
      saxpy(U, h * (Real(1) - delta), L2);     // + dt*(1-delta)*L2
      saxpy(U, cS2, work);                     // + ((1-gamma)/gamma)*U^(2)
      saxpy(U, -cS2, base2);                   // - ((1-gamma)/gamma)*base2  -> U = base3
      backward_euler_source(m, *ctx.aux, U, h * gamma, nopts, mask,
                            nreport);  // U = U^(3) = U^{n+1}
    }
  }
};

/// Frozen residual (fill_ghosts + assemble_rhs) installed as the block's rhs_into.
/// Functor of the dt_hotspot diagnostic (ADC-182): dominant cell of the block CFL.
/// HOST (the internal reductions are device); named, like MaxSpeed.
template <class Model>
struct HotspotFn {
  Model m;
  GridContext ctx;
  void operator()(const MultiFab& U, Real& w, int& i, int& j) const {
    max_wave_speed_hotspot_mf(m, U, *ctx.aux, ctx.dom.nx(), w, i, j);
  }
};

/// Kernel device de la PROJECTION PONCTUELLE post-pas (ADC-177) :
/// U(i, j) <- m.project(U(i, j), aux(i, j)). FONCTEUR NOMME (meme contrat device que BlockRhsEval).
/// Lecture et ecriture sur le MEME fab : acces strictement ponctuel (aucun voisin lu), donc aucune
/// dependance inter-cellule -- parallelisable sans tampon.
template <class Model>
struct ProjectCellKernel {
  Model m;
  Array4 u;        // ecriture (etat du bloc)
  ConstArray4 uc;  // lecture (meme fab, vue const)
  ConstArray4 a;   // aux du System (phi, grad phi, champs extra)
  ADC_HD void operator()(int i, int j) const {
    const typename Model::State p =
        m.project(load_state<Model>(uc, i, j), load_aux<aux_comps<Model>()>(a, i, j));
    for (int c = 0; c < Model::n_vars; ++c)
      u(i, j, c) = p[c];
  }
};

/// Foncteur HOTE de la projection ponctuelle : for_each_cell du kernel sur les cellules VALIDES de
/// chaque fab local. Les GHOSTS ne sont pas projetes : tout consommateur de ghosts (residu de
/// transport) refait fill_ghosts en tete d'evaluation (cf. BlockRhsEval), donc l'etat fantome est
/// reconstruit du valide projete au pas suivant -- aucun fill_boundary necessaire ici.
template <class Model>
struct PointwiseProject {
  Model m;
  GridContext ctx;
  void operator()(MultiFab& U) const {
    for (int li = 0; li < U.local_size(); ++li)
      for_each_cell(U.box(li),
                    ProjectCellKernel<Model>{m, U.fab(li).array(), U.fab(li).const_array(),
                                             ctx.aux->fab(li).const_array()});
  }
};

template <class Limiter, class Flux, class Model>
struct RhsInto {
  Model m;
  GridContext ctx;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  std::shared_ptr<MultiFab> ws_cache;  ///< HLL wave speed cache (opt-in); nullptr -> per-face path
  void operator()(MultiFab& U, MultiFab& R) const {
    // Delegates to BlockRhsEval (fill_ghosts + assemble_rhs OR cached path): single source of the residual.
    BlockRhsEval<Limiter, Flux, Model>{m, &ctx, recon_prim, pos_floor, ws_cache}(U, R);
  }
};

// ============================================================================
// DISC ROUTING (T5-PR3 work): DISC residual evaluators + the advances that carry them.
// ============================================================================
// The transport residual of a block goes through BlockRhsEval (assemble_rhs, full cartesian). The two
// evaluators below SUBSTITUTE the disc operator for assemble_rhs, reading the System geometry BY
// POINTER (stable address of an Impl member) at step time -- so the add_block / set_disc_domain order
// is indifferent. NAMED functors (same device contract as BlockRhsEval).

/// MASKED transport residual (Staircase mode): fill_ghosts then assemble_rhs_masked on the
/// cell-centered 0/1 mask of the System (read via @c mask, pointer to Impl::domain_mask_, stable
/// address). The mask has the SAME layout as U (same ba/dm, 1 ghost). Inactive cell -> residual 0;
/// face toward an inactive cell -> zero normal flux (FV wall). The flux / reconstruction are REUSED
/// verbatim.
template <class Limiter, class Flux, class Model>
struct BlockRhsEvalMasked {
  Model model;
  const GridContext* ctx;
  const MultiFab* mask;  // Impl::domain_mask_ (NOT owned; stable address)
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  void operator()(MultiFab& U, MultiFab& R) const {
    fill_ghosts(U, ctx->dom, ctx->bc);
    assemble_rhs_masked<Limiter, Flux>(model, U, *ctx->aux, *mask, ctx->geom, R, recon_prim,
                                       pos_floor);
  }
};

/// CUT-CELL / EB transport residual (CutCell mode): fill_ghosts then assemble_rhs_eb on the level
/// set of the System embedded boundary (read via @c eb_domain, pointer to Impl::eb_domain_, stable
/// address). The device-callable level set is built HERE on the HOST (detail::disc_level_set(*eb_domain)
/// -> DiscLevelSet, a NAMED FUNCTOR capturing three doubles BY VALUE) and passed BY VALUE to
/// assemble_rhs_eb: the device kernel therefore receives NO std::function, it stays device-clean
/// (cf. spatial_operator_eb.hpp). The descriptor is a DiscDomain (the built-in instance of the
/// level-set contract, numerics/embedded_boundary.hpp).
template <class Limiter, class Flux, class Model>
struct BlockRhsEvalEb {
  Model model;
  const GridContext* ctx;
  const DiscDomain* eb_domain;  // Impl::eb_domain_ (NOT owned; stable address)
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  void operator()(MultiFab& U, MultiFab& R) const {
    fill_ghosts(U, ctx->dom, ctx->bc);
    assemble_rhs_eb<Limiter, Flux>(model, U, *ctx->aux, disc_level_set(*eb_domain), ctx->geom, R,
                                   recon_prim, kEbKappaMin, pos_floor);
  }
};

/// MASKED EXPLICIT advance: n substeps of the @c Stepper stepper on the MASKED transport residual.
/// Mimics AdvanceExplicit exactly (same RK math, same limiter / flux): only the residual changes.
template <class Limiter, class Flux, class Model, class Stepper = SSPRK2Step>
struct AdvanceExplicitMasked {
  Model m;
  GridContext ctx;
  const MultiFab* mask;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  void operator()(MultiFab& U, Real dt, int n) const {
    const Real h = dt / static_cast<Real>(n);
    const BlockRhsEvalMasked<Limiter, Flux, Model> rhs{m, &ctx, mask, recon_prim, pos_floor};
    run_explicit_substeps<Stepper>(rhs, U, h, n);
  }
};

/// CUT-CELL / EB EXPLICIT advance: n substeps of the @c Stepper stepper on the EB transport residual.
/// Mimics AdvanceExplicit exactly: only the residual (assemble_rhs_eb) changes.
template <class Limiter, class Flux, class Model, class Stepper = SSPRK2Step>
struct AdvanceExplicitEb {
  Model m;
  GridContext ctx;
  const DiscDomain* eb_domain;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  void operator()(MultiFab& U, Real dt, int n) const {
    const Real h = dt / static_cast<Real>(n);
    const BlockRhsEvalEb<Limiter, Flux, Model> rhs{m, &ctx, eb_domain, recon_prim, pos_floor};
    run_explicit_substeps<Stepper>(rhs, U, h, n);
  }
};

/// MASKED IMEX advance: MASKED EXPLICIT half-step (source-free transport) + stiff IMPLICIT source.
/// Mimics AdvanceImex: the transport (forward-Euler) reads the MASKED source-free residual; the
/// implicit source (backward_euler_source) is UNCHANGED (per-cell local, off the disc boundary). An
/// inactive cell has a zero transport residual then undergoes the local source -- like T2 / EB, only
/// the transport BOUNDARY is closed, the source stays cell-local.
template <class Limiter, class Flux, class Model>
struct AdvanceImexMasked {
  Model m;
  GridContext ctx;
  const MultiFab* mask;
  bool recon_prim;
  ImplicitMask<Model::n_vars> mask_impl{};
  NewtonOptions nopts{};
  NewtonReport* nreport = nullptr;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive)
  void operator()(MultiFab& U, Real dt, int n) const {
    const Real h = dt / static_cast<Real>(n);
    const BlockRhsEvalMasked<Limiter, Flux, SourceFreeModel<Model>> rhs{
        SourceFreeModel<Model>{m}, &ctx, mask, recon_prim, pos_floor};
    if (nreport)
      nreport->reset();
    for (int s = 0; s < n; ++s) {
      ForwardEuler{}.take_step(rhs, U, h);
      backward_euler_source(m, *ctx.aux, U, h, nopts, mask_impl, nreport);
    }
  }
};

/// CUT-CELL / EB IMEX advance: EB EXPLICIT half-step (source-free transport) + stiff IMPLICIT source.
/// Mimics AdvanceImex: transport via assemble_rhs_eb, implicit source unchanged (cell-local).
template <class Limiter, class Flux, class Model>
struct AdvanceImexEb {
  Model m;
  GridContext ctx;
  const DiscDomain* eb_domain;
  bool recon_prim;
  ImplicitMask<Model::n_vars> mask_impl{};
  NewtonOptions nopts{};
  NewtonReport* nreport = nullptr;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive)
  void operator()(MultiFab& U, Real dt, int n) const {
    const Real h = dt / static_cast<Real>(n);
    const BlockRhsEvalEb<Limiter, Flux, SourceFreeModel<Model>> rhs{
        SourceFreeModel<Model>{m}, &ctx, eb_domain, recon_prim, pos_floor};
    if (nreport)
      nreport->reset();
    for (int s = 0; s < n; ++s) {
      ForwardEuler{}.take_step(rhs, U, h);
      backward_euler_source(m, *ctx.aux, U, h, nopts, mask_impl, nreport);
    }
  }
};
}  // namespace detail

/// Builds the device-clean POD implicit mask of an N-variable model from a list of component indices
/// (empty -> INACTIVE mask -> model default, bit-identical). Any index outside [0, N) is ignored here
/// (the validation / clear message lives on the System::add_block side, which resolves names/roles into
/// indices and throws on an absent name/role).
template <int N>
ADC_COLD_FN ImplicitMask<N> make_implicit_mask(const std::vector<int>& implicit_components) {
  ImplicitMask<N> mask;
  if (implicit_components.empty())
    return mask;  // inactive: model default
  mask.active = true;
  for (int c : implicit_components)
    if (c >= 0 && c < N)
      mask.flag[c] = true;
  return mask;
}

/// Closures (advance + residual) for a frozen spatial scheme (Limiter x Flux). The RK math comes from
/// the core TimeStepper: in explicit, SSPRK2 (default), SSPRK3 or ForwardEuler ("euler", order 1,
/// fidelity to first-order references -- validation, never default) according to @p method;
/// ForwardEuler + backward_euler_source in IMEX. The closures are NAMED FUNCTORS (cf. namespace detail)
/// and not lambdas: the add_compiled_model path (first instantiation from an external TU) then emits
/// cleanly under nvcc. @p method affects ONLY the explicit advance (IMEX keeps its ForwardEuler
/// half-step + implicit source); "ssprk2" reproduces the historical advance (bit-identical). In IMEX
/// (@p imex), @p method "imexrk_ars222" selects the IMEX-RK ARS(2,2,2) family (order 2, advance
/// PARALLEL to AdvanceImex, full cartesian only); any other value keeps the historical backward-Euler
/// IMEX (order 1, bit-identical).
/// @p implicit_components: indices of the conserved variables to handle IMPLICITLY in the IMEX source
/// (mask CARRIED BY THE BLOCK, overrides the model default). EMPTY (default) -> inactive mask -> model
/// default is_implicit -> bit-identical. No effect outside IMEX (the explicit has no implicit step).
/// The optional EMBEDDED-BOUNDARY transport advances (advance_masked / advance_eb) are built when @p ctx
/// carries the System level-set domain (ctx.domain_mask / ctx.eb_domain, T5-PR3 work); otherwise they
/// stay empty and the stepper falls back on advance (bit-identical). STABLE addresses of Impl members,
/// read BY POINTER at step time -> the add_block / set_disc_domain order is indifferent. The
/// embedded-boundary advances MIMIC advance (same RK / IMEX, same limiter / flux); only the transport
/// residual is dispatched (assemble_rhs_masked / _eb).
template <class Limiter, class Flux, class Model>
ADC_COLD_FN BlockClosures build_block(const Model& m, const GridContext& ctx, bool imex,
                                      bool recon_prim, const std::string& method = "ssprk2",
                                      const std::vector<int>& implicit_components = {},
                                      const NewtonOptions& newton_opts = {},
                                      NewtonReport* newton_report = nullptr,
                                      Real pos_floor = Real(0), bool wave_speed_cache = false) {
  const MultiFab* domain_mask = ctx.domain_mask;
  const detail::DiscDomain* eb_domain = ctx.eb_domain;
  BlockClosures bc;
  const ImplicitMask<Model::n_vars> impl_mask =
      make_implicit_mask<Model::n_vars>(implicit_components);
  // SHARED scratch of the HLL wave speed cache (opt-in): a single MultiFab for the explicit advance and
  // rhs_into (never called concurrently). nullptr when the option is OFF -> BlockRhsEval keeps the
  // per-face path (bit-identical). Allocated at the real layout on the first call (cf. BlockRhsEval).
  std::shared_ptr<MultiFab> ws_cache =
      wave_speed_cache ? std::make_shared<MultiFab>() : std::shared_ptr<MultiFab>{};
  if (imex) {
    if (method == "imexrk_ars222") {
      // IMEX-RK FAMILY, ARS(2,2,2) scheme (order 2): advance PARALLEL to AdvanceImex, FULLY implicit
      // source (impl_mask ignored: the facade already rejects a partial mask with this scheme). FULL
      // CARTESIAN ONLY: we do NOT build an embedded-boundary advance (advance_masked / advance_eb stay
      // empty) -> an embedded-boundary geometry mode on this block throws an EXPLICIT error at step time
      // (SystemStepper::advance_transport_n), never a silent cartesian.
      bc.advance = detail::AdvanceImexRkArs222<Limiter, Flux, Model>{
          m, ctx, recon_prim, newton_opts, newton_report, pos_floor};
    } else {
      // Historical IMEX (local backward-Euler, order 1): UNTOUCHED, bit-identical.
      bc.advance = detail::AdvanceImex<Limiter, Flux, Model>{
          m, ctx, recon_prim, impl_mask, newton_opts, newton_report, pos_floor};
      if (domain_mask)
        bc.advance_masked = detail::AdvanceImexMasked<Limiter, Flux, Model>{
            m, ctx, domain_mask, recon_prim, impl_mask, newton_opts, newton_report, pos_floor};
      if (eb_domain)
        bc.advance_eb = detail::AdvanceImexEb<Limiter, Flux, Model>{
            m, ctx, eb_domain, recon_prim, impl_mask, newton_opts, newton_report, pos_floor};
    }
  } else if (method == "euler") {
    bc.advance = detail::AdvanceExplicit<Limiter, Flux, Model, ForwardEuler>{m, ctx, recon_prim,
                                                                             pos_floor, ws_cache};
    if (domain_mask)
      bc.advance_masked = detail::AdvanceExplicitMasked<Limiter, Flux, Model, ForwardEuler>{
          m, ctx, domain_mask, recon_prim, pos_floor};
    if (eb_domain)
      bc.advance_eb = detail::AdvanceExplicitEb<Limiter, Flux, Model, ForwardEuler>{
          m, ctx, eb_domain, recon_prim, pos_floor};
  } else if (method == "ssprk3") {
    bc.advance = detail::AdvanceExplicit<Limiter, Flux, Model, SSPRK3Step>{m, ctx, recon_prim,
                                                                           pos_floor, ws_cache};
    if (domain_mask)
      bc.advance_masked = detail::AdvanceExplicitMasked<Limiter, Flux, Model, SSPRK3Step>{
          m, ctx, domain_mask, recon_prim, pos_floor};
    if (eb_domain)
      bc.advance_eb = detail::AdvanceExplicitEb<Limiter, Flux, Model, SSPRK3Step>{
          m, ctx, eb_domain, recon_prim, pos_floor};
  } else if (method == "ssprk2") {
    bc.advance = detail::AdvanceExplicit<Limiter, Flux, Model, SSPRK2Step>{m, ctx, recon_prim,
                                                                           pos_floor, ws_cache};
    if (domain_mask)
      bc.advance_masked = detail::AdvanceExplicitMasked<Limiter, Flux, Model, SSPRK2Step>{
          m, ctx, domain_mask, recon_prim, pos_floor};
    if (eb_domain)
      bc.advance_eb = detail::AdvanceExplicitEb<Limiter, Flux, Model, SSPRK2Step>{
          m, ctx, eb_domain, recon_prim, pos_floor};
  } else {
    throw std::runtime_error("System: unknown explicit time method '" + method +
                             "' (euler|ssprk2|ssprk3)");
  }
  bc.rhs_into = detail::RhsInto<Limiter, Flux, Model>{m, ctx, recon_prim, pos_floor, ws_cache};
  bc.hotspot =
      detail::HotspotFn<Model>{m, ctx};  // dt_hotspot diagnostic (ADC-182), off the hot path
  // PROJECTION PONCTUELLE post-pas (ADC-177) : fabriquee SEULEMENT si le modele declare le trait
  // (HasPointwiseProjection, cf. core/physical_model.hpp) ; vide sinon -> le stepper ne l'interroge
  // jamais (chemin historique bit-identique). Partagee par add_block ET add_compiled_model (les deux
  // passent par make_block) : un .so 'production' la transporte donc nativement.
  if constexpr (HasPointwiseProjection<Model>)
    bc.project = detail::PointwiseProject<Model>{m, ctx};
  return bc;
}

/// Dispatch of the spatial scheme (limiter x Riemann flux) -> compiled closures. HLLC / Roe guarded
/// by requires: they demand a 4-variable transport exposing pressure (otherwise an explicit error).
/// "weno5" = WENO5-Z reconstruction (order 5, 5-point stencil, 3 ghosts); spatial_operator routes to
/// weno5z when Limiter::n_ghost >= 3 (the caller must allocate 3 ghosts, cf. block_n_ghost).
/// @p method chooses the EXPLICIT advance (ssprk2 by default, ssprk3 | euler optional); no effect in IMEX.
/// @p implicit_components: IMEX implicit mask carried by the block (indices; empty = model default,
/// bit-identical). cf. build_block.
// Per-flux limiter ladders, split out of make_block (ADC-335) so each flux's build_block leaves can be
// instantiated in their OWN translation unit (python/system_compressible_<flux>.cpp). Each body is the
// VERBATIM content of make_block's old `if (riem == "<flux>")` branch (same capability if-constexpr,
// same limiter ladder, same throws) -> bit-identical. make_block (below) is now a thin riem dispatcher
// that calls these; it stays the entry point for the non-subdivided callers (exb/isothermal seams, the
// .so/AOT loader path). The flux string is implied by which helper is called -> validation moves to the
// make_block dispatcher (kept) and, for the per-flux seam path, to the caller (System).
template <class Model>
ADC_COLD_FN BlockClosures make_block_rusanov(const Model& m, const std::string& lim,
                                             const GridContext& ctx, bool imex, bool recon_prim,
                                             const std::string& method,
                                             const std::vector<int>& implicit_components,
                                             const NewtonOptions& newton_opts,
                                             NewtonReport* newton_report, Real pos_floor) {
  if (lim == "none")
    return build_block<NoSlope, RusanovFlux>(m, ctx, imex, recon_prim, method, implicit_components,
                                             newton_opts, newton_report, pos_floor);
  if (lim == "minmod")
    return build_block<Minmod, RusanovFlux>(m, ctx, imex, recon_prim, method, implicit_components,
                                            newton_opts, newton_report, pos_floor);
  if (lim == "vanleer")
    return build_block<VanLeer, RusanovFlux>(m, ctx, imex, recon_prim, method, implicit_components,
                                             newton_opts, newton_report, pos_floor);
  if (lim == "weno5")
    return build_block<Weno5, RusanovFlux>(m, ctx, imex, recon_prim, method, implicit_components,
                                           newton_opts, newton_report, pos_floor);
  throw_registry_dispatch_mismatch("System", "limiteur", lim);
}

template <class Model>
ADC_COLD_FN BlockClosures make_block_hll(const Model& m, const std::string& lim,
                                         const GridContext& ctx, bool imex, bool recon_prim,
                                         const std::string& method,
                                         const std::vector<int>& implicit_components,
                                         const NewtonOptions& newton_opts,
                                         NewtonReport* newton_report, Real pos_floor,
                                         bool wave_speed_cache) {
  // HLL (Harten-Lax-van Leer, 2 waves): less diffusive than Rusanov (dissipation ~ signed |sR-sL|
  // instead of symmetric 2*max|v|), but does NOT require pressure (unlike HLLC/Roe) -- only SIGNED
  // wave speeds model.wave_speeds. Available as soon as a model exposes its signed eigenvalues (the
  // DSL emits wave_speeds as soon as a primitive 'p' is declared, even cold isothermal p=0 -> c=0 ->
  // HLL degenerates to upwind, still less diffusive than Rusanov at the contact). Does NOT REQUIRE
  // n_vars==4 nor a pressure: usable by a 3-var isothermal model (rho, m_x, m_y) exposing signed
  // wave speeds but no pressure, where hllc/roe are rejected. Gated on the presence of wave_speeds
  // (otherwise a CLEAR error, not a compilation failure for a scalar model without a signed wave,
  // e.g. ExB transport).
  if constexpr (requires(const Model mm, typename Model::State s, Aux a, Real r) {
                  mm.wave_speeds(s, a, 0, r, r);
                }) {
    // wave_speed_cache (opt-in) forwarded ONLY here: the wave speed cache only engages for the HLL
    // flux (BlockRhsEval guarded by Flux == HLLFlux). rusanov/hllc/roe ignore it.
    if (lim == "none")
      return build_block<NoSlope, HLLFlux>(m, ctx, imex, recon_prim, method, implicit_components,
                                           newton_opts, newton_report, pos_floor, wave_speed_cache);
    if (lim == "minmod")
      return build_block<Minmod, HLLFlux>(m, ctx, imex, recon_prim, method, implicit_components,
                                          newton_opts, newton_report, pos_floor, wave_speed_cache);
    if (lim == "vanleer")
      return build_block<VanLeer, HLLFlux>(m, ctx, imex, recon_prim, method, implicit_components,
                                           newton_opts, newton_report, pos_floor, wave_speed_cache);
    if (lim == "weno5")
      return build_block<Weno5, HLLFlux>(m, ctx, imex, recon_prim, method, implicit_components,
                                         newton_opts, newton_report, pos_floor, wave_speed_cache);
    throw_registry_dispatch_mismatch("System", "limiteur", lim);
  } else {
    throw std::runtime_error(
        "System: flux 'hll' requires signed wave speeds "
        "(model.wave_speeds: declare a primitive 'p' / eigenvalues); "
        "this transport -> 'rusanov'");
  }
}

template <class Model>
ADC_COLD_FN BlockClosures make_block_hllc(const Model& m, const std::string& lim,
                                          const GridContext& ctx, bool imex, bool recon_prim,
                                          const std::string& method,
                                          const std::vector<int>& implicit_components,
                                          const NewtonOptions& newton_opts,
                                          NewtonReport* newton_report, Real pos_floor) {
  // HLLC PATHS: (a) HasHLLCStructure capability (the model provides contact_speed +
  // hllc_star_state -> GENERIC contact-resolving algorithm, no assumed layout), OR
  // (b) the CANONICAL Euler 2D path (n_vars == 4 + pressure, bit-identical historical
  // implementation). Without either, explicit rejection with the capability remedy.
  if constexpr (HasHLLCStructure<Model> ||
                (Model::n_vars == 4 &&
                 requires(const Model mm, typename Model::State s) { mm.pressure(s); })) {
    if (lim == "none")
      return build_block<NoSlope, HLLCFlux>(m, ctx, imex, recon_prim, method, implicit_components,
                                            newton_opts, newton_report, pos_floor);
    if (lim == "minmod")
      return build_block<Minmod, HLLCFlux>(m, ctx, imex, recon_prim, method, implicit_components,
                                           newton_opts, newton_report, pos_floor);
    if (lim == "vanleer")
      return build_block<VanLeer, HLLCFlux>(m, ctx, imex, recon_prim, method, implicit_components,
                                            newton_opts, newton_report, pos_floor);
    if (lim == "weno5")
      return build_block<Weno5, HLLCFlux>(m, ctx, imex, recon_prim, method, implicit_components,
                                          newton_opts, newton_report, pos_floor);
    throw_registry_dispatch_mismatch("System", "limiteur", lim);
  } else {
    throw std::runtime_error(
        "System: flux 'hllc' requires a compressible Euler 2D transport "
        "(4 variables + pressure) OR the model's HLLC capability "
        "(pressure + wave_speeds + contact_speed + hllc_star_state, cf. "
        "HasHLLCStructure); this transport -> 'hll'/'rusanov'");
  }
}

template <class Model>
ADC_COLD_FN BlockClosures make_block_roe(const Model& m, const std::string& lim,
                                         const GridContext& ctx, bool imex, bool recon_prim,
                                         const std::string& method,
                                         const std::vector<int>& implicit_components,
                                         const NewtonOptions& newton_opts,
                                         NewtonReport* newton_report, Real pos_floor) {
  // ROE PATHS: (a) HasRoeDissipation capability (the model provides its full Roe dissipation
  // d = |A_roe| dU -> GENERIC Roe-like solver), OR (b) the CANONICAL ideal-gas Euler 2D path
  // (bit-identical historical). Without either, explicit rejection.
  if constexpr (HasRoeDissipation<Model> ||
                (Model::n_vars == 4 &&
                 requires(const Model mm, typename Model::State s) { mm.pressure(s); })) {
    if (lim == "none")
      return build_block<NoSlope, RoeFlux>(m, ctx, imex, recon_prim, method, implicit_components,
                                           newton_opts, newton_report, pos_floor);
    if (lim == "minmod")
      return build_block<Minmod, RoeFlux>(m, ctx, imex, recon_prim, method, implicit_components,
                                          newton_opts, newton_report, pos_floor);
    if (lim == "vanleer")
      return build_block<VanLeer, RoeFlux>(m, ctx, imex, recon_prim, method, implicit_components,
                                           newton_opts, newton_report, pos_floor);
    if (lim == "weno5")
      return build_block<Weno5, RoeFlux>(m, ctx, imex, recon_prim, method, implicit_components,
                                         newton_opts, newton_report, pos_floor);
    throw_registry_dispatch_mismatch("System", "limiteur", lim);
  } else {
    throw std::runtime_error(
        "System: flux 'roe' requires a compressible Euler 2D transport "
        "(4 variables + pressure) OR the model's Roe capability "
        "(roe_dissipation, cf. HasRoeDissipation); this transport -> "
        "'hll'/'rusanov'");
  }
}

template <class Model>
ADC_COLD_FN BlockClosures make_block(const Model& m, const std::string& lim,
                                     const std::string& riem, const GridContext& ctx, bool imex,
                                     bool recon_prim, const std::string& method = "ssprk2",
                                     const std::vector<int>& implicit_components = {},
                                     const NewtonOptions& newton_opts = {},
                                     NewtonReport* newton_report = nullptr,
                                     Real pos_floor = Real(0), bool wave_speed_cache = false) {
  // CENTRALIZED VALIDATION (registry dispatch_tags.hpp) BEFORE the dispatch: same tag acceptances /
  // rejections as before, identical messages (validate_* keeps the historical wording). The flux
  // dispatch now forwards to the per-flux helpers above (each holds the unchanged capability
  // `if constexpr` guard + limiter ladder); the final throw stays a registry/dispatch-inconsistency
  // guard (unreachable after validate_riemann).
  validate_riemann(riem, /*polar=*/false, "System");
  validate_limiter(lim, "System");
  if (riem == "rusanov")
    return make_block_rusanov(m, lim, ctx, imex, recon_prim, method, implicit_components,
                              newton_opts, newton_report, pos_floor);
  if (riem == "hll")
    return make_block_hll(m, lim, ctx, imex, recon_prim, method, implicit_components, newton_opts,
                          newton_report, pos_floor, wave_speed_cache);
  if (riem == "hllc")
    return make_block_hllc(m, lim, ctx, imex, recon_prim, method, implicit_components, newton_opts,
                           newton_report, pos_floor);
  if (riem == "roe")
    return make_block_roe(m, lim, ctx, imex, recon_prim, method, implicit_components, newton_opts,
                          newton_report, pos_floor);
  throw_registry_dispatch_mismatch("System", "flux", riem);
}

/// Number of ghosts required by the spatial scheme @p lim (single source: Limiter::n_ghost). Used for
/// the allocation of a block state MultiFab, so that the wide WENO5 stencil (5 points, 3 ghosts) does
/// not read out of bounds -- cf. AmrSystem allocates with Limiter::n_ghost (PR #22). Default 2 (MUSCL)
/// for an unknown limiter: that is the historical allocation, hence bit-identical.
inline int block_n_ghost(const std::string& lim) {
  // SINGLE source: limiter_n_ghost(lim) (registry dispatch_tags.hpp). The default 2 (MUSCL) for an
  // unknown limiter is carried by the registry -> same historical allocation, bit-identical. The
  // static_asserts below (this TU sees BOTH the registry AND the types) guarantee that the kLimiters
  // table never drifts from the real::n_ghost constants.
  static_assert(limiter_n_ghost_ct("none") == NoSlope::n_ghost, "kLimiters[none].n_ghost drifted");
  static_assert(limiter_n_ghost_ct("minmod") == Minmod::n_ghost,
                "kLimiters[minmod].n_ghost drifted");
  static_assert(limiter_n_ghost_ct("vanleer") == VanLeer::n_ghost,
                "kLimiters[vanleer].n_ghost drifted");
  static_assert(limiter_n_ghost_ct("weno5") == Weno5::n_ghost, "kLimiters[weno5].n_ghost drifted");
  return limiter_n_ghost(lim);
}

namespace detail {
/// Block max wave speed functor (max_wave_speed_mf, reduction over the seam). NAMED FUNCTOR:
/// max_wave_speed_mf instantiates MaxWaveSpeedKernel (already a device functor); wrapping it in a
/// named class rather than a lambda preserves the cross-TU instantiation context under nvcc.
template <class Model>
struct MaxSpeed {
  Model m;
  GridContext ctx;
  Real operator()(const MultiFab& U) const { return max_wave_speed_mf(m, U, *ctx.aux); }
};

/// Block max STABILITY speed functor (HasStabilitySpeed trait): replaces MaxSpeed in the CFL when the
/// model declares stability_speed (the Riemann solvers keep max_wave_speed).
template <class Model>
struct MaxStabilitySpeed {
  Model m;
  GridContext ctx;
  Real operator()(const MultiFab& U) const { return max_stability_speed_mf(m, U, *ctx.aux); }
};

/// Block max source frequency functor (HasSourceFrequency trait, bound dt <= cfl/mu without h).
template <class Model>
struct MaxSourceFreq {
  Model m;
  GridContext ctx;
  Real operator()(const MultiFab& U) const { return max_source_frequency_mf(m, U, *ctx.aux); }
};

/// Block min admissible step functor (HasStabilityDt trait; 0 = no cell constrains it).
template <class Model>
struct MinStabilityDt {
  Model m;
  GridContext ctx;
  Real operator()(const MultiFab& U) const { return min_stability_dt_mf(m, U, *ctx.aux); }
};

/// Poisson contribution functor: rhs += elliptic_rhs(U) (pure HOST loop, no device kernel).
template <class Model>
struct PoissonRhs {
  Model m;
  void operator()(const MultiFab& U, MultiFab& rhs) const {
    for (int li = 0; li < rhs.local_size(); ++li) {
      Array4 r = rhs.fab(li).array();
      const ConstArray4 u = U.fab(li).const_array();
      const Box2D b = rhs.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          r(i, j) += m.elliptic_rhs(load_state<Model>(u, i, j));
    }
  }
};
}  // namespace detail

/// Closure of the speed used by the block CFL step. If the model declares the OPTIONAL stability_speed
/// trait (HasStabilitySpeed), THAT is what drives the CFL (stability lambda*); otherwise STRICT
/// fallback on max_wave_speed (historical behavior, bit-identical). The Riemann solvers always read
/// max_wave_speed: this choice only changes the step policy.
template <class Model>
std::function<Real(const MultiFab&)> make_max_speed(const Model& m, const GridContext& ctx) {
  if constexpr (HasStabilitySpeed<Model>)
    return detail::MaxStabilitySpeed<Model>{m, ctx};
  else
    return detail::MaxSpeed<Model>{m, ctx};
}

/// Closure of the block max source frequency (bound dt <= cfl * substeps / (stride * mu)). EMPTY (null
/// std::function) if the model does not declare the trait -> the stepper ignores it (historical
/// behavior).
template <class Model>
std::function<Real(const MultiFab&)> make_source_frequency(const Model& m, const GridContext& ctx) {
  if constexpr (HasSourceFrequency<Model>)
    return detail::MaxSourceFreq<Model>{m, ctx};
  else
    return {};
}

/// Closure of the block min admissible step (bound dt <= stability_dt * substeps / stride, WITHOUT
/// cfl). EMPTY if the model does not declare the trait -> ignored by the stepper (historical).
template <class Model>
std::function<Real(const MultiFab&)> make_stability_dt(const Model& m, const GridContext& ctx) {
  if constexpr (HasStabilityDt<Model>)
    return detail::MinStabilityDt<Model>{m, ctx};
  else
    return {};
}

/// Block contribution to the Poisson right-hand side: rhs += elliptic_rhs(U) (host loop).
template <class Model>
std::function<void(const MultiFab&, MultiFab&)> make_poisson_rhs(const Model& m) {
  return detail::PoissonRhs<Model>{m};
}

/// PER-CELL (one cell) cons <-> prim conversions of the MODEL, type-erased over arrays of
/// Model::n_vars doubles. First = primitive -> conservative (M.to_conservative, init from the
/// primitives), second = conservative -> primitive (M.to_primitive, diagnostic). Captures the model by
/// value (frozen when the block is added). For a model WITHOUT a conversion (pure scalar, no
/// hyperbolic brick) both are the IDENTITY -- exact for a scalar transport (prim == cons).
/// Model::Prim shares the Model::n_vars width of State (HyperbolicPhysicalModel contract), so the flat
/// arrays align component by component. Shared by add_block (native) and add_compiled_model (compiled):
/// the SAME conversion serves both paths.
template <class Model>
std::pair<std::function<void(const double*, double*)>, std::function<void(const double*, double*)>>
make_cell_convert(const Model& m) {
  constexpr int NV = Model::n_vars;
  if constexpr (HasPrimitiveVars<Model>) {
    auto p2c = [m](const double* in, double* out) {
      typename Model::Prim p{};
      for (int c = 0; c < NV; ++c)
        p[c] = static_cast<Real>(in[c]);
      const typename Model::State u = m.to_conservative(p);
      for (int c = 0; c < NV; ++c)
        out[c] = static_cast<double>(u[c]);
    };
    auto c2p = [m](const double* in, double* out) {
      typename Model::State u{};
      for (int c = 0; c < NV; ++c)
        u[c] = static_cast<Real>(in[c]);
      const typename Model::Prim p = m.to_primitive(u);
      for (int c = 0; c < NV; ++c)
        out[c] = static_cast<double>(p[c]);
    };
    return {std::function<void(const double*, double*)>(p2c),
            std::function<void(const double*, double*)>(c2p)};
  } else {
    auto id = [](const double* in, double* out) {
      for (int c = 0; c < NV; ++c)
        out[c] = in[c];
    };
    return {std::function<void(const double*, double*)>(id),
            std::function<void(const double*, double*)>(id)};
  }
}

}  // namespace adc
