#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/embedded_boundary.hpp>  // detail::DiscDomain (built-in level-set domain instance)

#include <functional>

/// @file
/// @brief Block grid context plus closures, shared between System (which installs them) and
///        block_builder.hpp (which builds them from a compiled model). LIGHT header (mesh plus
///        std::function, no numerics) so it can be included in the System public API without
///        pulling in assemble_rhs / flux / steppers.

namespace adc {

/// TRANSPORT GEOMETRY MODE of the macro-step (T5-PR3 effort, disc wiring in System::step).
///  - None: full Cartesian domain (default). Transport uses assemble_rhs (historical
///                path). BIT-IDENTICAL to history as long as no disc is set.
///  - Staircase: disc approximated by a cell-centered 0/1 MASK (active/inactive face gate,
///                staircase boundary). Transport uses assemble_rhs_masked (T2 effort).
///  - CutCell: disc as cut-cell / embedded-boundary (continuous alpha_f apertures plus volume
///                fraction kappa). Transport uses assemble_rhs_eb (T5-PR1/PR2 efforts).
/// The mode is held by the System (set_disc_domain mode= / set_geometry_mode) and read by the stepper
/// to DISPATCH each block transport advance. None stays the untouched production path.
enum class GeometryMode { None, Staircase, CutCell };

/// Mesh + transport BC + aux shared by a block closures. @c aux is NOT owned:
/// it points to the System aux (lifetime longer than the block, stable address).
///
/// EMBEDDED BOUNDARY / LEVEL-SET DOMAIN (T5-PR3 effort): @c domain_mask and @c eb_domain point (NOT
/// owned) to the 0/1 mask and the level-set domain descriptor of the System (members with STABLE
/// address). They are used ONLY to build the optional embedded-boundary transport advances
/// (build_block); read BY POINTER at the step, the order add_block / set_disc_domain does not matter.
/// nullptr -> no embedded-boundary advance (stepper on advance, bit-identical). The mask is
/// materialized / the descriptor is set by set_disc_domain (the disc is one instance of the contract,
/// cf. numerics/embedded_boundary.hpp).
struct GridContext {
  Box2D dom;                ///< domain (without ghost)
  BCRec bc;                 ///< transport BC
  Geometry geom;            ///< geometry (dx, dy, bounds)
  MultiFab* aux = nullptr;  ///< System aux (phi, grad phi); NOT owned
  const MultiFab* domain_mask = nullptr;  ///< 0/1 domain mask (Impl::domain_mask_); NOT owned
  const detail::DiscDomain* eb_domain =
      nullptr;  ///< level-set domain descriptor (Impl::eb_domain_); NOT owned
};

/// Compiled block closures, frozen at add time.
///
/// advance is the transport advance of the DEFAULT path (assemble_rhs, full Cartesian). The two
/// optional DISC advances (T5-PR3 effort) mimic advance EXACTLY (same RK / IMEX scheme,
/// same limiter / flux) but dispatch the transport residual to the disc operator:
///   - advance_masked: assemble_rhs_masked (0/1 mask, Staircase mode);
///   - advance_eb: assemble_rhs_eb (cut-cell EB, CutCell mode).
/// They read the System mask / level set BY POINTER at step time (not at
/// construction), so the order add_block / set_disc_domain does not matter. Empty (default) as long as
/// the block does not support disc routing: the stepper then falls back to advance (bit-identical).
struct BlockClosures {
  std::function<void(MultiFab&, Real, int)> advance;  ///< (U, dt, n): n substeps of dt/n
  std::function<void(MultiFab&, Real, int)> advance_masked;  ///< same, residual via assemble_rhs_masked
  std::function<void(MultiFab&, Real, int)> advance_eb;      ///< same, residual via assemble_rhs_eb
  std::function<void(MultiFab&, MultiFab&)> rhs_into;  ///< R <- -div F + S (Poisson frozen)
  /// dt_hotspot diagnostic (ADC-182): (U, w, i, j) -> GLOBAL cell dominating the transport
  /// CFL and its speed. OPTIONAL (empty = block without diagnostic, e.g. historical
  /// unrewired paths); never called by step/step_cfl (off the hot path).
  std::function<void(const MultiFab&, Real&, int&, int&)> hotspot;
  /// PROJECTION PONCTUELLE post-pas (ADC-177) : U <- project(U, aux) sur les cellules VALIDES du
  /// bloc, appliquee par le stepper a la FIN de chaque macro-pas ENTIER (jamais par etage RK).
  /// OPTIONNELLE (vide = bloc sans projection : jamais interrogee, cout nul, bit-identique).
  std::function<void(MultiFab&)> project;
};

}  // namespace adc
