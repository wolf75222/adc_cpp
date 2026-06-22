#pragma once

/// @file
/// @brief Umbrella for composable GENERIC physics bricks (compat). Re-exports the bricks by
///        category plus the CompositeModel that assembles them. The core knows NO named scenario:
///        a scenario is a COMPOSITION of bricks, chosen from the application (adc_cases).
///
/// Split by category (to match the target tree physics/{hyperbolic,source,elliptic,...}):
///   - physics/hyperbolic.hpp: ExBVelocity, CompressibleFlux (= Euler), IsothermalFlux;
///   - physics/source.hpp:     NoSource, PotentialForce, GravityForce;
///   - physics/elliptic.hpp:   ChargeDensity, BackgroundDensity, GravityCoupling;
///   - physics/composite.hpp:  CompositeModel<Hyperbolic, Source, Elliptic>.
/// Including this file gives EVERYTHING (as before); including a precise category is now possible.
///
/// PRODUCTION vs VALIDATION: this umbrella re-exports ONLY production/generic bricks. The
/// validation/reference bricks (AdvectionDiffusion, LangmuirMode, TwoFluidLinear) are NOT part of
/// this surface; they live under adc/validation/physics/ in namespace adc::validation (ADC-329).
/// The old adc/physics/{advection_diffusion,langmuir,two_fluid_isothermal}.hpp paths remain as
/// deprecated compat forwarders and are intentionally not aggregated here.

#include <adc/physics/composition/composite.hpp>
#include <adc/physics/bricks/elliptic.hpp>
#include <adc/physics/bricks/hyperbolic.hpp>
#include <adc/physics/bricks/source.hpp>
