/// @file
/// @brief ADC-329 compat check for the validation/reference bricks.
///
/// Proves that the validation bricks (AdvectionDiffusion, LangmuirMode, TwoFluidLinear) compile via
/// BOTH the new canonical include paths (adc/validation/physics/...) and the deprecated forwarders
/// (adc/physics/...), and that the deprecated adc:: aliases name the SAME types as the canonical
/// adc::validation:: ones. Pure compile-time / identity check: no numerics, no behavior assertion.

// New canonical location: namespace adc::validation.
#include <adc/validation/physics/advection_diffusion.hpp>
#include <adc/validation/physics/langmuir.hpp>
#include <adc/validation/physics/two_fluid_isothermal.hpp>

// Deprecated forwarders: old include paths must still compile and alias adc::<Type>.
#include <adc/physics/advection_diffusion.hpp>
#include <adc/physics/langmuir.hpp>
#include <adc/physics/two_fluid_isothermal.hpp>

#include <type_traits>

// The deprecated adc:: name and the canonical adc::validation:: name must be the same type.
static_assert(std::is_same_v<adc::AdvectionDiffusion, adc::validation::AdvectionDiffusion>,
              "adc::AdvectionDiffusion must alias adc::validation::AdvectionDiffusion (ADC-329)");
static_assert(std::is_same_v<adc::LangmuirMode, adc::validation::LangmuirMode>,
              "adc::LangmuirMode must alias adc::validation::LangmuirMode (ADC-329)");
static_assert(std::is_same_v<adc::TwoFluidLinear, adc::validation::TwoFluidLinear>,
              "adc::TwoFluidLinear must alias adc::validation::TwoFluidLinear (ADC-329)");

int main() {
  // Odr-use each brick as a complete type so header rot is caught at build time.
  adc::validation::AdvectionDiffusion ad{};
  (void)ad.diffusivity();

  adc::validation::LangmuirMode lm{};
  (void)lm.omega();

  adc::validation::TwoFluidLinear tf{};
  adc::Real w_fast = 0, w_slow = 0;
  tf.dispersion(w_fast, w_slow);
  (void)w_fast;
  (void)w_slow;

  // Odr-use the DEPRECATED adc:: spellings (reached via the old forwarder paths) as complete
  // objects, not merely type-compared, so the legacy include path is exercised end to end.
  adc::AdvectionDiffusion ad_legacy{};
  (void)ad_legacy.diffusivity();

  adc::LangmuirMode lm_legacy{};
  (void)lm_legacy.omega();

  adc::TwoFluidLinear tf_legacy{};
  adc::Real wf_legacy = 0, ws_legacy = 0;
  tf_legacy.dispersion(wf_legacy, ws_legacy);
  (void)wf_legacy;
  (void)ws_legacy;

  return 0;
}
