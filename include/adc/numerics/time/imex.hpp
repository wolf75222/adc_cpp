#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/multifab.hpp>

// Integrateur IMEX (implicit-explicit) asymptotic-preserving (AP).
//
// Pour le vrai fluide compressible magnetise couple a un champ self-consistant
// (regime du papier Hoffart, arXiv:2510.11808), certains termes sont RAIDES :
// force de Lorentz, limite de Debye lambda_D -> 0, quasi-neutralite. Les traiter
// en explicite imposerait
// dt ~ raideur (Debye, gyrofrequence) -> impraticable. Un schema IMEX prend ces
// termes en IMPLICITE et le transport en EXPLICITE. La propriete AP : quand le
// petit parametre (eps = lambda_D^2, 1/omega_c, ...) -> 0, le schema reste stable
// a dt FIXE et capture la dynamique limite (equilibre / quasi-neutralite), sans
// resoudre l'echelle raide.
//
// imex_euler_step : IMEX d'Euler (forward-backward), 1er ordre.
//   U^{n+1} = U^n + dt T(U^n) + dt S(U^{n+1})
// Texpl(U, dt) : avance le transport EN PLACE (U <- U + dt T(U)), donc apres
//   l'appel U porte le membre connu U^n + dt T(U^n).
// Simpl(U, dt) : resout EN PLACE U <- W tel que W = U + dt S(W), ou U (en entree)
//   est le membre connu. Source lineaire (relaxation) : analytique ; non lineaire
//   (Lorentz complet) : Newton local. C'est l'analogue de APIMEXTwoFluidIsothermal
//   de MUFFIN, le pont entre notre modele reduit de derive et la physique complete.

namespace adc {

template <class TransportStep, class ImplicitSourceSolve>
void imex_euler_step(MultiFab& U, Real dt, TransportStep Texpl,
                     ImplicitSourceSolve Simpl) {
  Texpl(U, dt);  // explicite : U devient le membre connu U^n + dt T(U^n)
  Simpl(U, dt);  // implicite : resout U = connu + dt S(U) (source raide) en place
}

}  // namespace adc
