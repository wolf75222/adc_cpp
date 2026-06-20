// Locks the centralized native AMR refinement-ratio invariant
// (include/adc/amr/refinement_ratio.hpp): the single constant kAmrRefRatio and
// the require_supported_ref_ratio guard that replaces the literal `2` formerly
// scattered across the coarse/fine paths (ADC-295). Deliberately light: it pulls
// only the invariant header plus AmrHierarchy (whose ctor validates the ratio
// BEFORE any allocation, so the rejection path needs no mesh/device setup).
//
// It verifies (1) the invariant value, (2) the guard accepts the supported ratio
// and rejects any other with a clear message naming the home header, and (3) the
// AmrHierarchy ctor refuses a non-2 ratio at construction (the single entry point)
// rather than silently mis-coarsening.

#include <adc/amr/amr_hierarchy.hpp>
#include <adc/amr/refinement_ratio.hpp>

#include <cstdio>
#include <stdexcept>
#include <string>

using namespace adc;

namespace {

// Runs fn(), returns true if it threw, capturing the message in @p msg.
template <class Fn>
bool throws(Fn&& fn, std::string& msg) {
  try {
    fn();
    msg.clear();
    return false;
  } catch (const std::exception& e) {
    msg = e.what();
    return true;
  }
}

int failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    ++failures;
  }
}

}  // namespace

int main() {
  // (1) The invariant value (compile-time + run-time, so a drift is caught both ways).
  static_assert(kAmrRefRatio == 2,
                "the native AMR hierarchy only supports refinement ratio 2 today");
  check(kAmrRefRatio == 2, "kAmrRefRatio == 2");

  // (2) The guard: accepts the supported ratio, rejects any other with a clear message.
  std::string msg;
  check(!throws([] { require_supported_ref_ratio(kAmrRefRatio); }, msg),
        "require_supported_ref_ratio(kAmrRefRatio) does not throw");

  check(throws([] { require_supported_ref_ratio(3); }, msg),
        "require_supported_ref_ratio(3) throws");
  check(msg.find('3') != std::string::npos && msg.find("refinement_ratio.hpp") != std::string::npos,
        "rejection message names the ratio and the home header");

  check(throws([] { require_supported_ref_ratio(1); }, msg),
        "require_supported_ref_ratio(1) throws");
  check(throws([] { require_supported_ref_ratio(4); }, msg),
        "require_supported_ref_ratio(4) throws");

  // (3) The single entry point: AmrHierarchy refuses a non-2 ratio at construction.
  // The guard is the ctor's first statement, so this throws before any allocation.
  const Box2D dom{{0, 0}, {7, 7}};
  check(throws(
            [&] {
              AmrHierarchy h(dom, /*max_grid_size=*/8, /*ncomp=*/1, /*ngrow=*/0,
                             /*ref_ratio=*/3);
            },
            msg),
        "AmrHierarchy with ref_ratio 3 is rejected at construction");
  check(msg.find("refinement ratio") != std::string::npos,
        "AmrHierarchy rejection message explains the ratio constraint");

  if (failures == 0) {
    std::printf("test_ref_ratio: OK\n");
    return 0;
  }
  std::printf("test_ref_ratio: %d failure(s)\n", failures);
  return 1;
}
