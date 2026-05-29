#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/multifab.hpp>

// Splitting d'operateur : decompose dU/dt = T(U) + S(U) en sous-pas resolus
// separement. Brique pour traiter une source raide (relaxation, collisions,
// ionisation) par un integrateur DIFFERENT de celui du transport, sans melanger
// les deux raideurs. Prerequis a l'ajout de sources chimiques/collisionnelles.
//
//   - lie_step (Godunov, 1er ordre)   : T(dt) puis S(dt)
//   - strang_step (2e ordre)          : S(dt/2), T(dt), S(dt/2)
//
// Strang est 2e ordre en temps DES QUE chaque sous-integrateur l'est (l'erreur de
// commutation [T,S] est en O(dt^3) par pas, O(dt^2) globale). L'integrateur est
// agnostique : T et S sont des callables (MultiFab&, Real)->void qui avancent leur
// sous-systeme EN PLACE. C'est l'equivalent maison de StrangSplitting /
// FractionalTime2OSplitting de MUFFIN.

namespace adc {

template <class TransportStep, class SourceStep>
void lie_step(MultiFab& U, Real dt, TransportStep T, SourceStep S) {
  T(U, dt);
  S(U, dt);
}

template <class TransportStep, class SourceStep>
void strang_step(MultiFab& U, Real dt, TransportStep T, SourceStep S) {
  S(U, Real(0.5) * dt);
  T(U, dt);
  S(U, Real(0.5) * dt);
}

}  // namespace adc
