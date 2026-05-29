#pragma once

#include <adc/core/types.hpp>

#include <cmath>

// Limiteurs de pente pour la reconstruction MUSCL. Chaque limiteur expose son
// nombre de ghosts requis (1 = premier ordre sans pente, 2 = reconstruction
// lineaire limitee) et une fonction (diff arriere, diff avant) -> pente limitee.
//
// Choix de conception : le limiteur est un parametre de template (polymorphisme
// statique, kernel inlinable, device-safe). NoSlope court-circuite tout calcul
// de pente, donc le chemin premier ordre ne lit qu'un ghost.

namespace adc {

struct NoSlope {
  static constexpr int n_ghost = 1;
  ADC_HD Real operator()(Real, Real) const { return Real(0); }
};

// minmod : TVD, robuste, mais ecrete les extrema lisses (ordre 1 local aux pics).
struct Minmod {
  static constexpr int n_ghost = 2;
  ADC_HD Real operator()(Real a, Real b) const {
    if (a * b <= Real(0)) return Real(0);
    const Real fa = a < 0 ? -a : a, fb = b < 0 ? -b : b;  // |.| device-safe
    return (fa < fb) ? a : b;
  }
};

// van Leer : limiteur lisse, meilleur ordre aux extrema que minmod.
struct VanLeer {
  static constexpr int n_ghost = 2;
  ADC_HD Real operator()(Real a, Real b) const {
    const Real ab = a * b;
    if (ab <= Real(0)) return Real(0);
    return Real(2) * ab / (a + b);
  }
};

}  // namespace adc
