#pragma once

// A lightweight per-node / per-brick profiler for the compiled Program step (Spec 3 section 29-30,
// ADC-459). It accumulates named wall-clock scopes -- one per Program node, native brick, or step
// phase -- plus a few integer counters (kernels, cache hits/misses, scheduled nodes due/skipped),
// and renders the report `sim.profile_report()` returns.
//
// Cost when disabled: record()/count() are a single predictable branch (zero work). A ProfileScope
// is cheap but NOT free even when off -- it still constructs its name string and reads the clock
// twice -- so wrap a per-node / per-brick scope (the intended granularity), not the tightest inner
// loops. Single-threaded: a Profiler / ProfileScope is not thread-safe and must not be shared inside
// a parallel region. Host/serial today; on a device backend a Kokkos::fence() must precede the scope
// close so the timing reflects the kernel (the fence lives at the call site, not here -- this header
// has no Kokkos dependency). MPI: each rank profiles itself; the report is per-rank (any reduction
// belongs to the System integration).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace pops::runtime::program {

class Profiler {
 public:
  struct Entry {
    std::uint64_t count = 0;
    double total_s = 0.0;
    double min_s = 0.0;
    double max_s = 0.0;
    double mean_s() const { return count != 0 ? total_s / static_cast<double>(count) : 0.0; }
  };

  void enable() { enabled_ = true; }
  void disable() { enabled_ = false; }
  bool enabled() const { return enabled_; }

  // Drop all accumulated timings and counters (kept across enable/disable; cleared explicitly).
  void reset() {
    order_.clear();
    entries_.clear();
    counters_.clear();
    counter_order_.clear();
  }

  // Record one timed sample of `name`, in seconds. No-op when disabled.
  void record(const std::string& name, double seconds) {
    if (!enabled_) {
      return;
    }
    auto it = entries_.find(name);
    if (it == entries_.end()) {
      order_.push_back(name);
      entries_.emplace(name,
                       Entry{.count = 1, .total_s = seconds, .min_s = seconds, .max_s = seconds});
    } else {
      Entry& e = it->second;
      e.count += 1;
      e.total_s += seconds;
      e.min_s = std::min(e.min_s, seconds);
      e.max_s = std::max(e.max_s, seconds);
    }
  }

  // Bump a named integer counter (e.g. "kernels", "cache_hits", "nodes_skipped"). No-op when off.
  void count(const std::string& name, std::int64_t by = 1) {
    if (!enabled_) {
      return;
    }
    auto it = counters_.find(name);
    if (it == counters_.end()) {
      counter_order_.push_back(name);
      counters_.emplace(name, by);
    } else {
      it->second += by;
    }
  }

  // Track a named PEAK counter: set it to max(current, value) instead of accumulating. Used for the
  // scratch peak memory (the largest single scratch allocation seen, in bytes), where the running sum
  // is meaningless. First-seen creates the counter at @p value. No-op when disabled.
  void count_max(const std::string& name, std::int64_t value) {
    if (!enabled_) {
      return;
    }
    auto it = counters_.find(name);
    if (it == counters_.end()) {
      counter_order_.push_back(name);
      counters_.emplace(name, value);
    } else {
      it->second = std::max(it->second, value);
    }
  }

  const Entry* entry(const std::string& name) const {
    auto it = entries_.find(name);
    return it == entries_.end() ? nullptr : &it->second;
  }

  std::int64_t counter(const std::string& name) const {
    auto it = counters_.find(name);
    return it == counters_.end() ? 0 : it->second;
  }

  // Sum of every scope's total time (the "total" line of the report).
  double total_s() const {
    double t = 0.0;
    for (const auto& name : order_) {
      t += entries_.at(name).total_s;
    }
    return t;
  }

  std::size_t scope_count() const { return order_.size(); }

  // A human-readable report in first-seen order: one line per scope (count / total / mean / min /
  // max), then the counters. The exact text the Python `sim.profile_report()` returns.
  std::string report() const {
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(6);
    os << "Profiler report (total " << total_s() << " s, " << order_.size() << " scopes)\n";
    for (const auto& name : order_) {
      const Entry& e = entries_.at(name);
      os << "  " << name << "  count=" << e.count << "  total=" << e.total_s
         << "s  mean=" << e.mean_s() << "s  min=" << e.min_s << "s  max=" << e.max_s << "s\n";
    }
    if (!counter_order_.empty()) {
      os << "counters:";
      for (const auto& name : counter_order_) {
        os << "  " << name << "=" << counters_.at(name);
      }
      os << "\n";
    }
    return os.str();
  }

 private:
  bool enabled_ = false;
  std::vector<std::string> order_;  // scope names, first-seen order (stable report)
  std::map<std::string, Entry> entries_;
  std::vector<std::string> counter_order_;  // counter names, first-seen order
  std::map<std::string, std::int64_t> counters_;
};

// RAII scope: times its own lifetime into `prof` under `name`. One per Program node / brick call.
// Construct it at the top of the work; its destructor records the elapsed wall-clock seconds.
class ProfileScope {
 public:
  ProfileScope(Profiler& prof, std::string name)
      : prof_(prof), name_(std::move(name)), t0_(std::chrono::steady_clock::now()) {}

  ~ProfileScope() {
    const auto t1 = std::chrono::steady_clock::now();
    // A timing alloc failure must never abort the program nor escape this destructor: swallow any
    // exception from record() (the worst case is one lost sample).
    try {
      prof_.record(name_, std::chrono::duration<double>(t1 - t0_).count());
    } catch (...) {  // NOLINT(bugprone-empty-catch) -- a profiler never throws out of a scope
    }
  }

  ProfileScope(const ProfileScope&) = delete;
  ProfileScope& operator=(const ProfileScope&) = delete;
  ProfileScope(ProfileScope&&) = delete;
  ProfileScope& operator=(ProfileScope&&) = delete;

 private:
  Profiler& prof_;
  std::string name_;
  std::chrono::steady_clock::time_point t0_;
};

}  // namespace pops::runtime::program
