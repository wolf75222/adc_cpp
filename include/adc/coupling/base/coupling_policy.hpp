/// @file
/// @brief Hyperbolic-elliptic temporal coupling policies (compile-time tag types).
///
/// How often the elliptic problem is solved within a time step. Tag types chosen by template at the
/// call site (Coupler::advance<Limiter, Policy>), NO runtime branch. PerStageCoupling:
/// phi (thus aux = grad phi) recomputed at EVERY RK stage -> most precise coupling, one elliptic
/// solve per stage. OncePerStepCoupling: phi solved only ONCE (start of step), aux frozen
/// during the stages -> cheaper, de facto splitting. (AMR sub-cycling and tile <-> FFT band
/// redistribution are policies of the same family, carried by AmrCoupler /
/// SpectralCoupler.)

#pragma once

namespace adc {

/// Tag: solves the elliptic problem at EVERY RK stage (aux follows the intermediate state, more precise).
struct PerStageCoupling {};
/// Tag: solves the elliptic problem ONCE per step (aux frozen during the stages, cheaper).
struct OncePerStepCoupling {};

}  // namespace adc
