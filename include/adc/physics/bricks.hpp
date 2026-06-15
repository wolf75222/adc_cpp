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

#include <adc/physics/composite.hpp>
#include <adc/physics/elliptic.hpp>
#include <adc/physics/hyperbolic.hpp>
#include <adc/physics/source.hpp>
