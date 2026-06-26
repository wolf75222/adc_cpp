// La discretisation spatiale (limiteur + flux) et les tags d'integration en temps
// sont des types nommes du coeur. Verification statique des bundles et des aliases :
// aucun modele requis (le coeur ne connait aucune physique).

#include <pops/numerics/time/integrators/time_integrator.hpp>
#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/numerics/fv/reconstruction.hpp>
#include <pops/numerics/fv/spatial_discretisation.hpp>

#include <cstdio>
#include <type_traits>

using namespace pops;

// Un bundle expose Limiter et NumericalFlux.
static_assert(std::is_same_v<FirstOrder::Limiter, NoSlope>);
static_assert(std::is_same_v<FirstOrder::NumericalFlux, RusanovFlux>);
static_assert(std::is_same_v<MusclVanLeer::Limiter, VanLeer>);
static_assert(std::is_same_v<MusclVanLeer::NumericalFlux, RusanovFlux>);
static_assert(std::is_same_v<MusclVanLeerHLLC::Limiter, VanLeer>);
static_assert(std::is_same_v<MusclVanLeerHLLC::NumericalFlux, HLLCFlux>);

// On peut composer librement reconstruction x flux.
using MinmodHLL = SpatialDiscretisation<Minmod, HLLFlux>;
static_assert(std::is_same_v<MinmodHLL::Limiter, Minmod>);
static_assert(std::is_same_v<MinmodHLL::NumericalFlux, HLLFlux>);

// Les tags d'integration en temps sont distincts.
static_assert(!std::is_same_v<SSPRK2, SSPRK3>);
static_assert(TimePolicyTraits<SSPRK2>::substeps == 1);
static_assert(TimePolicyTraits<ExplicitTime<SSPRK3, 4>>::substeps == 4);
static_assert(TimePolicyTraits<ImplicitTime<UserTimeIntegrator, 10>>::treatment ==
              TimeTreatment::Implicit);

int main() {
  std::printf("OK test_spatial_discretisation\n");
  return 0;
}
