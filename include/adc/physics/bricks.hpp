#pragma once

/// @file
/// @brief Umbrella des briques physiques GENERIQUES composables (compat). Reexporte les briques par
///        categorie + le CompositeModel qui les assemble. Le coeur ne connait AUCUN scenario nomme :
///        un scenario est une COMPOSITION de briques, choisie depuis l'application (adc_cases).
///
/// Decoupage par categorie (pour coller a l'arbre cible physics/{hyperbolic,source,elliptic,...}) :
///   - physics/hyperbolic.hpp : ExBVelocity, CompressibleFlux (= Euler), IsothermalFlux ;
///   - physics/source.hpp     : NoSource, PotentialForce, GravityForce ;
///   - physics/elliptic.hpp   : ChargeDensity, BackgroundDensity, GravityCoupling ;
///   - physics/composite.hpp  : CompositeModel<Hyperbolic, Source, Elliptic>.
/// Inclure ce fichier donne TOUT (comme avant) ; inclure une categorie precise est desormais possible.

#include <adc/physics/composite.hpp>
#include <adc/physics/elliptic.hpp>
#include <adc/physics/hyperbolic.hpp>
#include <adc/physics/source.hpp>
