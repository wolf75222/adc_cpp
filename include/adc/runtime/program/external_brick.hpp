#pragma once

// Process-global registry of EXTERNAL C++ bricks (Spec 3 section 21-22, criterion 20, ADC-463). A
// Spec 3 brick is native / generated / macro / external-C++; this header owns the last category. A
// user ships a brick in a standalone `.so` that, at static-init time, registers a manifest entry via
// `ADC_REGISTER_BRICK(id, category, requirements)`. The host then exports a C `adc_brick_manifest()`
// function (JSON over `BrickRegistry::instance().ids()` + each entry) that `adc.lib.load_cpp_library`
// dlopens and parses, so `adc.lib.riemann.User("id")` surfaces a real, requirement-carrying
// descriptor. This is a HOST registry (no ADC_HD, no device state): it catalogs the brick's identity
// and requirements; the brick's numerical kernel stays a separate concern wired by the codegen.

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace adc::runtime::program {

// The manifest of one external C++ brick: its identity plus the requirements/capabilities the
// selector and codegen need. All fields are host strings (no device data); `requirements` and
// `capabilities` are comma-separated lists (the manifest's wire form), empty when none.
struct BrickManifestEntry {
  std::string id;            // the brick id a user selects (e.g. "my_hllc")
  std::string category;      // the catalog slot ("riemann", "preconditioner", ...)
  std::string requirements;  // CSV of required model capabilities ("pressure,wave_speeds"), or ""
  std::string capabilities;  // CSV of capabilities the brick PROVIDES, or ""
};

// A process-global catalog of registered external bricks, keyed by id. Populated at static-init
// time by `ADC_REGISTER_BRICK` (in the user's `.so`) and read by the host's `adc_brick_manifest()`
// exporter. Construction order across translation units is unspecified, so the registry is a
// function-local static (the Meyers singleton) -- it is constructed on first use, before any
// `ADC_REGISTER_BRICK` static initializer can run against it.
class BrickRegistry {
 public:
  // The single process-global instance.
  static BrickRegistry& instance() {
    static BrickRegistry registry;
    return registry;
  }

  // Register (or replace) `entry` under `entry.id`. Idempotent on id: re-registering the same id
  // overwrites the manifest and never duplicates the id in `ids()` (last registration wins).
  void register_brick(const BrickManifestEntry& entry) {
    auto it = index_.find(entry.id);
    if (it == index_.end()) {
      index_.emplace(entry.id, entries_.size());
      entries_.push_back(entry);
      ids_.push_back(entry.id);
    } else {
      entries_[it->second] = entry;
    }
  }

  // The manifest of brick `id`, or nullptr if no such brick is registered (never throws).
  const BrickManifestEntry* lookup(const std::string& id) const {
    auto it = index_.find(id);
    return it == index_.end() ? nullptr : &entries_[it->second];
  }

  // Every registered brick id, in registration order.
  const std::vector<std::string>& ids() const { return ids_; }

  // Every registered manifest entry, in registration order.
  const std::vector<BrickManifestEntry>& entries() const { return entries_; }

  std::size_t size() const { return entries_.size(); }

  void clear() {
    entries_.clear();
    ids_.clear();
    index_.clear();
  }

 private:
  BrickRegistry() = default;

  std::vector<BrickManifestEntry> entries_;  // registration-order manifest list
  std::vector<std::string> ids_;             // registration-order id list (mirrors entries_)
  std::unordered_map<std::string, std::size_t> index_;  // id -> index into entries_
};

// Registers a manifest entry at static-init time. Use at namespace scope in a brick's `.so`:
//   ADC_REGISTER_BRICK("my_hllc", "riemann", "pressure,wave_speeds");
// The capabilities field is left empty by this 3-argument form; a brick that PROVIDES capabilities
// calls `BrickRegistry::instance().register_brick({...})` directly. The trailing static is a unique
// dummy whose initializer performs the registration (zero-cost at runtime, runs once before main).
#define ADC_REGISTER_BRICK(brick_id, brick_category, brick_requirements)            \
  static const bool ADC_REGISTER_BRICK_CAT_(adc_brick_registered_, __LINE__) = [] { \
    ::adc::runtime::program::BrickRegistry::instance().register_brick(              \
        {(brick_id), (brick_category), (brick_requirements), ""});                  \
    return true;                                                                    \
  }()

// Token-paste helpers so several ADC_REGISTER_BRICK uses in one TU get distinct static names.
#define ADC_REGISTER_BRICK_CAT_(a, b) ADC_REGISTER_BRICK_CAT2_(a, b)
#define ADC_REGISTER_BRICK_CAT2_(a, b) a##b

}  // namespace adc::runtime::program
