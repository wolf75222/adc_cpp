// ADC-291: the C++ canonical aux name<->component table (adc/core/aux_names.hpp) is the mirror of
// AUX_CANONICAL (python/adc/dsl.py), generated from the SAME single source as adc::Aux (the base
// contract phi/grad_x/grad_y + the ADC_AUX_FIELDS X-macro B_z/T_e). It lets a C++ caller resolve a
// canonical aux field by name WITHOUT the Python facade. This test pins the C++ side; the
// C++<->Python coherence is pinned by python/tests/test_capabilities.py.

#include <adc/core/aux_names.hpp>
#include <adc/core/state.hpp>

#include <cstdio>
#include <string_view>

using namespace adc;

// Compile-time coherence: the table is constexpr, so the canonical indices are pinned at build time
// (a drift from ADC_AUX_FIELDS / the base contract is a hard compile error, not a runtime surprise).
static_assert(aux_canonical_index("phi") == 0, "phi must be aux component 0");
static_assert(aux_canonical_index("B_z") == 3, "B_z must be aux component 3 (ADC_AUX_FIELDS)");
static_assert(aux_canonical_index("T_e") == 4, "T_e must be aux component 4 (ADC_AUX_FIELDS)");
static_assert(aux_canonical_index("kappa") == -1, "a model-named field is not canonical");
static_assert(kAuxMaxComps == kAuxNamedBase + kAuxMaxExtra, "kAuxMaxComps = base + max extras");

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  // canonical name -> component (mirror of AUX_CANONICAL on the DSL side)
  chk(aux_canonical_index("phi") == 0, "phi=0");
  chk(aux_canonical_index("grad_x") == 1, "grad_x=1");
  chk(aux_canonical_index("grad_y") == 2, "grad_y=2");
  chk(aux_canonical_index("B_z") == 3, "B_z=3");
  chk(aux_canonical_index("T_e") == 4, "T_e=4");
  // a model-NAMED field (resolved per block by the facade) is NOT a canonical aux field
  chk(aux_canonical_index("kappa") == -1, "named field not canonical");
  chk(aux_canonical_index("") == -1, "empty not canonical");

  // inverse coherence: component -> canonical name
  chk(aux_canonical_name(0) == "phi", "name(0)=phi");
  chk(aux_canonical_name(4) == "T_e", "name(4)=T_e");
  chk(aux_canonical_name(kAuxNamedBase) == std::string_view{}, "name(kAuxNamedBase) empty");

  // the canonical extras live STRICTLY below the named base (B_z/T_e never collide with extra[k])
  chk(aux_canonical_index("B_z") < kAuxNamedBase, "B_z below named base");
  chk(aux_canonical_index("T_e") < kAuxNamedBase, "T_e below named base");

  if (fails == 0)
    std::printf("OK test_aux_names\n");
  return fails == 0 ? 0 : 1;
}
