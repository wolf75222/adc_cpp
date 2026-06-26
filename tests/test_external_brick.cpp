// BrickRegistry: the process-global external-C++ brick registry backing Spec 3 section 21-22
// (criterion 20, ADC-463). Exercises macro registration, lookup of a registered/unknown id, the
// id listing, and the manifest fields a registered brick carries -- no Program/codegen needed.

#include <pops/runtime/program/external_brick.hpp>

#include "test_harness.hpp"  // pops::test::Checker

#include <string>

using pops::runtime::program::BrickManifestEntry;
using pops::runtime::program::BrickRegistry;

// Register two bricks at static-init time via the macro. The third argument is the
// requirements/capabilities CSV the manifest surfaces (a host-only string).
POPS_REGISTER_BRICK("test_hllc", "riemann", "pressure,wave_speeds");
POPS_REGISTER_BRICK("test_precond", "preconditioner", "");

int main() {
  pops::test::Checker chk;

  const BrickRegistry& reg = BrickRegistry::instance();

  // --- lookup of a registered brick returns its manifest entry ---
  {
    const BrickManifestEntry* e = reg.lookup("test_hllc");
    chk(e != nullptr, "lookup_registered_hllc");
    if (e != nullptr) {
      chk(e->id == "test_hllc", "hllc_id");
      chk(e->category == "riemann", "hllc_category");
      chk(e->requirements == "pressure,wave_speeds", "hllc_requirements");
    }
    const BrickManifestEntry* p = reg.lookup("test_precond");
    chk(p != nullptr, "lookup_registered_precond");
    if (p != nullptr) {
      chk(p->category == "preconditioner", "precond_category");
      chk(p->requirements.empty(), "precond_no_requirements");
    }
  }

  // --- lookup of an unknown id returns null (never throws) ---
  {
    chk(reg.lookup("not_registered") == nullptr, "lookup_unknown_is_null");
  }

  // --- ids() lists every registered brick ---
  {
    const std::vector<std::string>& ids = reg.ids();
    bool has_hllc = false;
    bool has_precond = false;
    for (const std::string& id : ids) {
      if (id == "test_hllc")
        has_hllc = true;
      if (id == "test_precond")
        has_precond = true;
    }
    chk(has_hllc, "ids_lists_hllc");
    chk(has_precond, "ids_lists_precond");
    chk(ids.size() >= 2, "ids_at_least_two");
  }

  // --- register_brick is idempotent on id: re-registering replaces, never duplicates ---
  {
    const std::size_t before = reg.ids().size();
    BrickRegistry::instance().register_brick(
        {"test_hllc", "riemann", "pressure,wave_speeds,contact_speed", ""});
    chk(reg.ids().size() == before, "reregister_does_not_duplicate");
    const BrickManifestEntry* e = reg.lookup("test_hllc");
    chk(e != nullptr && e->requirements == "pressure,wave_speeds,contact_speed",
        "reregister_replaces_entry");
  }

  return chk.failed();
}
