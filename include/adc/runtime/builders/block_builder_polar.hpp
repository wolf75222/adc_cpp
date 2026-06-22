#pragma once

#include <adc/core/foundation/types.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/geometry.hpp>  // PolarGeometry
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>  // BCRec, fill_ghosts, fill_boundary
#include <adc/numerics/numerical_flux.hpp>
#include <adc/numerics/reconstruction.hpp>
#include <adc/numerics/spatial_operator_polar.hpp>  // assemble_rhs_polar (REUSED verbatim)
#include <adc/numerics/time/time_steppers.hpp>      // SSPRK2Step / SSPRK3Step (core RK math)
#include <adc/parallel/comm.hpp>   // all_reduce_max (MPI-safe collective reduction)
#include <adc/physics/bricks/bricks.hpp>  // ExBVelocityPolar, CompositeModel, source/elliptic bricks
#include <adc/runtime/detail/dispatch_tags.hpp>  // UNIQUE registry of tags (validate_limiter/riemann)
#include <adc/runtime/detail/grid_context.hpp>   // BlockClosures (light header)
#include <adc/runtime/builders/model_factory.hpp>  // detail::dispatch_source / dispatch_elliptic (REUSED)
#include <adc/runtime/detail/model_registry.hpp>  // transport_tags_csv: polar-wired transport list (ADC-331)
#include <adc/runtime/model_spec.hpp>

#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/// @file
/// @brief POLAR counterpart of block_builder.hpp: builds a block's closures (time advance +
///        residual + Poisson contribution + wave speed) on an ANNULAR grid (PolarGeometry), by
///        REUSING assemble_rhs_polar (include/adc/numerics/spatial_operator_polar.hpp). This is the
///        Polar Phase 2b path: polar transport THROUGH System.step.
///
/// STRICT SEPARATION from the cartesian path: block_builder.hpp (cartesian assemble_rhs, Geometry) stays
/// UNTOUCHED -> a cartesian System is bit-identical. The polar path is PURELY ADDITIVE, opt-in
/// (enabled when cfg.geometry == "polar", see system.cpp). We reuse the SAME source / elliptic
/// bricks (dispatch_source / dispatch_elliptic from model_factory.hpp: they carry no geometry) and the
/// SAME ExB transport, but carried by the ExBVelocityPolar brick (physical components in the local
/// basis (e_r, e_theta)).
///
/// NAMED FUNCTORS (and not extended lambdas), like block_builder.hpp (#64/#97/#133): robust device
/// emission if the Model-template kernel is first-instantiated cross-TU. assemble_rhs_polar and its
/// kernels are already device-clean (all validated on a CUDA device, e.g. GH200); these closures
/// merely chain them.

namespace adc {

/// POLAR mesh + transport BC + aux shared by a block's closures (counterpart of
/// GridContext). @c aux is NOT owned: it points to the System's aux (stable address). bc carries the
/// radial BC (Foextrap = wall / outflow); theta is PERIODIC (handled by fill_ghosts via bc.ylo/yhi).
struct PolarGridContext {
  Box2D dom;                ///< index domain (without ghost): nx() = nr, ny() = ntheta
  BCRec bc;                 ///< BC: r (xlo/xhi) physical, theta (ylo/yhi) periodic
  PolarGeometry geom;       ///< ring (r_min, r_max, dr, dtheta)
  MultiFab* aux = nullptr;  ///< System's aux (phi, grad_r, grad_theta); NOT owned
};

namespace detail {

/// Builds the POLAR transport brick and calls v(transport). Two wired transports:
///   - "exb": ExBVelocityPolar, scalar ExB advection in the local basis (e_r, e_theta);
///   - "isothermal": IsothermalFluxPolar (Path A step 1), isothermal fluid 3 var (rho, rho v_r,
///                    rho v_theta) in polar metric. The PHYSICAL flux is that of IsothermalFlux
///                    cartesian (reused verbatim); the 1/r metric (divergence (1/r) d_r(r F_r) +
///                    (1/r) d_theta(F_theta)) AND the GEOMETRIC curvature term (centrifugal
///                    -rho v_theta^2/r + cross curvature) are carried by assemble_rhs_polar /
///                    IsothermalFluxPolar::polar_geom_source. Electrostatic coupling = existing
///                    SCALAR polar Poisson + LOCAL source (PotentialForce), explicit regime.
/// "compressible" transport (Euler 4 var with energy) in polar stays out of scope: its energy flux
/// and curvature term do not yet have a polar brick -> EXPLICIT error.
template <class Visitor>
void dispatch_transport_polar(const ModelSpec& m, Visitor&& v) {
  if (m.transport == "exb")
    return v(ExBVelocityPolar{Real(m.B0)});
  if (m.transport == "isothermal")
    return v(IsothermalFluxPolar{IsothermalFlux{Real(m.cs2), Real(m.vacuum_floor)}});
  // Wired polar transports = the registry rows with polar_ok (model_registry.hpp); the list is
  // single-sourced via transport_tags_csv(/*polar=*/true). 'compressible' (Euler with energy) has no
  // polar brick yet -> not polar_ok -> rejected here with the same explicit "unsupported" message.
  throw std::runtime_error("polar transport '" + m.transport +
                           "' unsupported (wired in polar: " + transport_tags_csv(/*polar=*/true) +
                           "; 'compressible' (Euler with energy) in polar is a later phase)");
}

/// Assembles the POLAR CompositeModel designated by @p m and calls visitor(model). REUSES
/// dispatch_source / dispatch_elliptic from model_factory.hpp (source / elliptic right-hand-side
/// bricks IDENTICAL to the cartesian ones: they carry no geometry). Only the transport brick changes
/// (ExBVelocityPolar or IsothermalFluxPolar). dispatch_source<TR::n_vars> filters automatically:
/// scalar ExB transport (1 var) -> only source 'none'; isothermal fluid transport (3 var) ->
/// 'none' | 'potential' (-rho grad phi) | 'gravity' | 'magnetic'/'lorentz' (q v x B_z, B_z read from
/// the aux, EXPLICIT regime) | 'potential_magnetic'/'potential_lorentz' (electrostatic + Lorentz sum =
/// full magnetized-plasma force in polar geometry) also valid. The Lorentz force is ALGEBRAIC and INVARIANT
/// under orientation of the local orthonormal frame: the SAME MagneticLorentzForce brick serves both
/// geometries (cartesian and polar), like PotentialForce / GravityForce. The 1/r metric and the
/// curvature stay carried by the transport.
template <class Visitor>
void dispatch_model_polar(const ModelSpec& m, Visitor&& visitor) {
  dispatch_transport_polar(m, [&](auto tr) {
    using TR = decltype(tr);
    // AUTOMATIC resolution by roles (audit sec.5), IDENTICAL to the cartesian one (bind_variable_roles
    // from model_factory.hpp). ExBVelocityPolar (density=0) / IsothermalFluxPolar (rho=0, m_x=1, m_y=2,
    // inherits from IsothermalFlux) declare canonical roles -> resolved indices == defaults ->
    // bit-identical. Resolved at construction (host); never on device.
    const VariableSet cons = TR::conservative_vars();
    dispatch_source<TR::n_vars>(m, [&](auto src) {
      dispatch_elliptic(m, [&](auto ell) {
        bind_variable_roles(src, cons);
        bind_variable_roles(ell, cons);
        visitor(CompositeModel<TR, decltype(src), decltype(ell)>{tr, src, ell});
      });
    });
  });
}

/// Fills the ghosts of a MultiFab on the polar grid (theta periodic + r physical). fill_ghosts
/// already routes periodic vs physical by BCRec (xlo/xhi physical, ylo/yhi periodic): we call it
/// VERBATIM. This is the analogue of the cartesian fill_ghosts(U, dom, bc) of BlockRhsEval.
inline void fill_ghosts_polar(MultiFab& U, const Box2D& dom, const BCRec& bc) {
  fill_ghosts(U, dom, bc);
}

/// Polar residual functor R = -div_polar F + S (fill_ghosts then assemble_rhs_polar). NAMED FUNCTOR
/// (counterpart of cartesian detail::BlockRhsEval): this is what take_step receives, triggering the
/// instantiation of assemble_rhs_polar<Limiter, Flux> and its device kernels. @c wall_radial: solid
/// radial wall (no-penetration) -> mass conserved to machine precision (see assemble_rhs_polar).
template <class Limiter, class Flux, class Model>
struct PolarBlockRhsEval {
  Model model;
  const PolarGridContext* ctx;
  bool recon_prim;
  bool wall_radial;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  void operator()(MultiFab& U, MultiFab& R) const {
    fill_ghosts_polar(U, ctx->dom, ctx->bc);
    assemble_rhs_polar<Limiter, Flux>(model, U, *ctx->aux, ctx->geom, R, recon_prim, wall_radial,
                                      pos_floor);
  }
};

/// EXPLICIT polar advance: n substeps of the @c Stepper stepper (SSPRK2 by default, SSPRK3 optional)
/// on the polar transport residual. Counterpart of cartesian detail::AdvanceExplicit.
template <class Limiter, class Flux, class Model, class Stepper = SSPRK2Step>
struct PolarAdvanceExplicit {
  Model m;
  PolarGridContext ctx;
  bool recon_prim;
  bool wall_radial;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  void operator()(MultiFab& U, Real dt, int n) const {
    const Real h = dt / static_cast<Real>(n);
    const PolarBlockRhsEval<Limiter, Flux, Model> rhs{m, &ctx, recon_prim, wall_radial, pos_floor};
    run_explicit_substeps<Stepper>(rhs, U, h, n);
  }
};

/// Frozen polar residual (fill_ghosts + assemble_rhs_polar) installed as the block's rhs_into (eval_rhs).
template <class Limiter, class Flux, class Model>
struct PolarRhsInto {
  Model m;
  PolarGridContext ctx;
  bool recon_prim;
  bool wall_radial;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  void operator()(MultiFab& U, MultiFab& R) const {
    fill_ghosts_polar(U, ctx.dom, ctx.bc);
    assemble_rhs_polar<Limiter, Flux>(m, U, *ctx.aux, ctx.geom, R, recon_prim, wall_radial,
                                      pos_floor);
  }
};

/// Max wave-speed functor of the POLAR block: reduction over the valid cells of
/// max_wave_speed(model, U, aux) in both directions (r, theta). Pure HOST loop (no device kernel)
/// -- the polar speed comes from the aux (grad_r, grad_theta) already host-resident after
/// solve_fields; that is enough for the CFL step. Counterpart of cartesian detail::MaxSpeed.
template <class Model>
struct PolarMaxSpeed {
  Model m;
  const MultiFab* aux;
  Real operator()(const MultiFab& U) const {
    Real wmax = Real(0);
    for (int li = 0; li < U.local_size(); ++li) {
      const ConstArray4 u = U.fab(li).const_array();
      const ConstArray4 a = aux->fab(li).const_array();
      const Box2D v = U.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
          const typename Model::State us = load_state<Model>(u, i, j);
          const Aux ac = load_aux<aux_comps<Model>()>(a, i, j);
          for (int dir = 0; dir < 2; ++dir) {
            const Real w = m.max_wave_speed(us, ac, dir);
            if (w > wmax)
              wmax = w;
          }
        }
    }
    return static_cast<Real>(all_reduce_max(static_cast<double>(wmax)));
  }
};

/// POLAR Poisson contribution functor: rhs += elliptic_rhs(U) (pure HOST loop). IDENTICAL to the
/// cartesian detail::PoissonRhs: the elliptic brick (charge q n) carries no geometry; the polar
/// metric (volume r dr dtheta) is carried by the PolarPoissonSolver solver, not by the per-cell
/// pointwise RHS (the solver expects f as-is, like the cartesian FFT solver).
template <class Model>
struct PolarPoissonRhs {
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

/// Derives the POLAR aux in the local basis (e_r, e_theta) from the potential @p phi resolved by
/// PolarPoissonSolver: aux[0] = phi; aux[1] = grad_r = d phi/dr; aux[2] = grad_theta = (1/r) d phi/d
/// theta. This is the layout expected by ExBVelocityPolar (v_r = -grad_theta/B, v_theta = grad_r/B).
///
/// KEY INVARIANT (cause of the fixed bug): @p phi is allocated by the direct solver WITHOUT ghost (mono-box).
/// We thus NEVER read a radial index out of domain: the radial derivative is CENTERED in the interior and
/// one-sided second order at both walls (i = lo: forward; i = hi: backward), without touching phi(lo-1) /
/// phi(hi+1). In theta (PERIODIC) we wrap the index (j-1 -> jhi, j+1 -> jlo) instead of reading the
/// nonexistent azimuthal ghost. Pure HOST loop (phi host-resident after solve()). Does NOT fill the ghosts of
/// the aux: the caller does it AFTER (fill_ghosts: theta periodic, r physical) for the transport.
/// PRECONDITION nr >= 3 (the one-sided second-order stencil reads p(i+2)/p(i-2) at the walls): IMPOSED
/// upstream by check_geometry (python/system.cpp) and adc.PolarMesh (nr >= 3), not merely assumed.
inline void derive_aux_polar(const MultiFab& phi, MultiFab& aux, const PolarGeometry& g) {
  const Real dr = g.dr(), dth = g.dtheta();
  for (int li = 0; li < aux.local_size(); ++li) {
    const ConstArray4 p = phi.fab(li).const_array();
    Array4 a = aux.fab(li).array();
    const Box2D v = aux.box(li);
    const int ilo = v.lo[0], ihi = v.hi[0], jlo = v.lo[1], jhi = v.hi[1];
    for (int j = jlo; j <= jhi; ++j) {
      const int jm = (j == jlo) ? jhi : j - 1;  // theta periodic: index wrap (no ghost)
      const int jp = (j == jhi) ? jlo : j + 1;
      for (int i = ilo; i <= ihi; ++i) {
        const Real ri = g.r_cell(i);
        a(i, j, 0) = p(i, j);
        Real
            gr;  // grad_r = d phi/dr: centered in the interior, one-sided second order at the walls (phi without ghost in r)
        if (i == ilo)
          gr = (Real(-3) * p(i, j) + Real(4) * p(i + 1, j) - p(i + 2, j)) / (Real(2) * dr);
        else if (i == ihi)
          gr = (Real(3) * p(i, j) - Real(4) * p(i - 1, j) + p(i - 2, j)) / (Real(2) * dr);
        else
          gr = (p(i + 1, j) - p(i - 1, j)) / (Real(2) * dr);
        a(i, j, 1) = gr;
        a(i, j, 2) = (p(i, jp) - p(i, jm)) /
                     (Real(2) * dth * ri);  // grad_theta = (1/r) d phi/d theta (already /r)
      }
    }
  }
}

/// Closures (advance + residual) of a POLAR block for a frozen spatial scheme (Limiter x Flux). The RK
/// math comes from the core TimeStepper (SSPRK2 / SSPRK3). NAMED FUNCTORS (see namespace detail).
/// Counterpart of cartesian build_block, but without IMEX (Phase 2b scalar ExB transport: no stiff
/// source). @p wall_radial: solid radial wall (no-penetration) -> mass conservation to machine precision.
template <class Limiter, class Flux, class Model>
BlockClosures build_block_polar(const Model& m, const PolarGridContext& ctx, bool recon_prim,
                                const std::string& method, bool wall_radial,
                                Real pos_floor = Real(0)) {
  BlockClosures bc;
  if (method == "ssprk3") {
    bc.advance = detail::PolarAdvanceExplicit<Limiter, Flux, Model, SSPRK3Step>{
        m, ctx, recon_prim, wall_radial, pos_floor};
  } else if (method == "ssprk2") {
    bc.advance = detail::PolarAdvanceExplicit<Limiter, Flux, Model, SSPRK2Step>{
        m, ctx, recon_prim, wall_radial, pos_floor};
  } else {
    throw std::runtime_error("System (polar): unknown explicit time method '" + method +
                             "' (ssprk2|ssprk3)");
  }
  bc.rhs_into =
      detail::PolarRhsInto<Limiter, Flux, Model>{m, ctx, recon_prim, wall_radial, pos_floor};
  return bc;
}

/// Dispatch of the spatial scheme (frozen limiter, Riemann flux) -> compiled polar closures.
/// Two fluxes wired in polar, SAME template injection point as the cartesian one (build_block_polar
/// carries the Flux parameter down to assemble_rhs_polar<Limiter, Flux>):
///   - "rusanov": RusanovFlux, requires only max_wave_speed (valid for scalar ExB AND the
///                 isothermal fluid) -- DEFAULT, strictly bit-identical to history;
///   - "hll": HLLFlux (signed waves), GATE identical to the cartesian one (make_block) on the
///                 presence of model.wave_speeds. The polar isothermal fluid (IsothermalFluxPolar:
///                 inherits IsothermalFlux::wave_speeds) is eligible -> HLL less diffusive than Rusanov
///                 on the ring. The scalar ExB (ExBVelocityPolar, no wave_speeds) -> CLEAR rejection.
/// HLLC/Roe stay NOT wired in polar (assume n_vars==4 Euler with energy, without a polar energy-flux
/// brick) -> explicit rejection. "weno5" routes assemble_rhs_polar onto the WENO5-Z reconstruction
/// (3 ghosts) like the cartesian one. @p wall_radial: solid radial wall (mass conservation to machine
/// precision; see build_block_polar).
template <class Model>
BlockClosures make_block_polar(const Model& m, const std::string& lim, const std::string& riem,
                               const PolarGridContext& ctx, bool recon_prim,
                               const std::string& method, bool wall_radial,
                               Real pos_floor = Real(0)) {
  // CENTRALIZED VALIDATION (registry dispatch_tags.hpp) BEFORE the dispatch: in polar, rusanov AND
  // hll are wired (hll since the rest of the audit); HLLC/Roe and unknown tags raise the polar
  // message of the registry. The CAPABILITY GUARD (hll requires model.wave_speeds) stays an
  // `if constexpr` PER MODEL below, with its dedicated "requires ..." message.
  validate_riemann(riem, /*polar=*/true, "System (polar)");
  validate_limiter(lim, "System (polar)");
  if (riem == "rusanov") {
    if (lim == "none")
      return build_block_polar<NoSlope, RusanovFlux>(m, ctx, recon_prim, method, wall_radial,
                                                     pos_floor);
    if (lim == "minmod")
      return build_block_polar<Minmod, RusanovFlux>(m, ctx, recon_prim, method, wall_radial,
                                                    pos_floor);
    if (lim == "vanleer")
      return build_block_polar<VanLeer, RusanovFlux>(m, ctx, recon_prim, method, wall_radial,
                                                     pos_floor);
    if (lim == "weno5")
      return build_block_polar<Weno5, RusanovFlux>(m, ctx, recon_prim, method, wall_radial,
                                                   pos_floor);
    throw_registry_dispatch_mismatch("System (polar)", "limiter", lim);
  }
  if (riem == "hll") {
    // GATE IDENTICAL TO THE CARTESIAN ONE (block_builder.hpp make_block, 'hll' branch): HLL is available
    // as soon as a model exposes its SIGNED wave speeds model.wave_speeds (the polar isothermal fluid
    // inherits them from IsothermalFlux). No pressure required (unlike HLLC/Roe). A SCALAR transport
    // without signed wave (ExBVelocityPolar) does NOT satisfy the requires -> CLEAR error
    // (not a compilation failure for a scalar model). assemble_rhs_polar<Limiter, HLLFlux>
    // is already device-clean (named functors, flux REUSED verbatim from the cartesian one).
    if constexpr (requires(const Model mm, typename Model::State s, Aux a, Real r) {
                    mm.wave_speeds(s, a, 0, r, r);
                  }) {
      if (lim == "none")
        return build_block_polar<NoSlope, HLLFlux>(m, ctx, recon_prim, method, wall_radial,
                                                   pos_floor);
      if (lim == "minmod")
        return build_block_polar<Minmod, HLLFlux>(m, ctx, recon_prim, method, wall_radial,
                                                  pos_floor);
      if (lim == "vanleer")
        return build_block_polar<VanLeer, HLLFlux>(m, ctx, recon_prim, method, wall_radial,
                                                   pos_floor);
      if (lim == "weno5")
        return build_block_polar<Weno5, HLLFlux>(m, ctx, recon_prim, method, wall_radial,
                                                 pos_floor);
      throw_registry_dispatch_mismatch("System (polar)", "limiter", lim);
    } else {
      throw std::runtime_error(
          "System (polar): flux 'hll' requires signed wave speeds (model.wave_speeds); "
          "the scalar ExB transport does not provide them -> 'rusanov'. The polar isothermal fluid "
          "(transport='isothermal') declares them and accepts 'hll'.");
    }
  }
  throw_registry_dispatch_mismatch("System (polar)", "Riemann flux", riem);
}

/// Max wave-speed closure of the POLAR block (for the CFL step). @p aux points to the System's aux
/// (stable address): the polar ExB speed comes from grad_r / grad_theta of the aux.
template <class Model>
std::function<Real(const MultiFab&)> make_max_speed_polar(const Model& m, const MultiFab* aux) {
  return detail::PolarMaxSpeed<Model>{m, aux};
}

namespace detail {
/// Optional STEP BOUND closures of the POLAR block (StabilityPolicy, audit wave 3):
/// same device reductions as the cartesian ones (POINTWISE kernels with no geometry assumption -- the
/// geometry enters only through the physical step h of the stepper, min(dr, r_min*dtheta)). NAMED
/// functors (same cross-TU device contract as PolarMaxSpeed).
template <class Model>
struct PolarStabilitySpeed {
  Model m;
  const MultiFab* aux;
  Real operator()(const MultiFab& U) const { return max_stability_speed_mf(m, U, *aux); }
};
template <class Model>
struct PolarSourceFreq {
  Model m;
  const MultiFab* aux;
  Real operator()(const MultiFab& U) const { return max_source_frequency_mf(m, U, *aux); }
};
template <class Model>
struct PolarStabilityDt {
  Model m;
  const MultiFab* aux;
  Real operator()(const MultiFab& U) const { return min_stability_dt_mf(m, U, *aux); }
};
}  // namespace detail

/// CFL speed of the POLAR block: lambda* (HasStabilitySpeed trait) if the model declares it,
/// otherwise max_wave_speed (historical PolarMaxSpeed, bit-identical) -- SAME policy as
/// cartesian make_max_speed.
template <class Model>
std::function<Real(const MultiFab&)> make_cfl_speed_polar(const Model& m, const MultiFab* aux) {
  if constexpr (HasStabilitySpeed<Model>)
    return detail::PolarStabilitySpeed<Model>{m, aux};
  else
    return detail::PolarMaxSpeed<Model>{m, aux};
}

/// Max source frequency of the POLAR block (HasSourceFrequency trait); EMPTY without the trait (the
/// stepper does not query it, historical step policy).
template <class Model>
std::function<Real(const MultiFab&)> make_source_frequency_polar(const Model& m,
                                                                 const MultiFab* aux) {
  if constexpr (HasSourceFrequency<Model>)
    return detail::PolarSourceFreq<Model>{m, aux};
  else
    return {};
}

/// Min admissible step of the POLAR block (HasStabilityDt trait); EMPTY without the trait.
template <class Model>
std::function<Real(const MultiFab&)> make_stability_dt_polar(const Model& m, const MultiFab* aux) {
  if constexpr (HasStabilityDt<Model>)
    return detail::PolarStabilityDt<Model>{m, aux};
  else
    return {};
}

/// Block contribution to the POLAR Poisson right-hand side: rhs += elliptic_rhs(U) (host loop).
template <class Model>
std::function<void(const MultiFab&, MultiFab&)> make_poisson_rhs_polar(const Model& m) {
  return detail::PolarPoissonRhs<Model>{m};
}

}  // namespace adc
