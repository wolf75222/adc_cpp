// Profiler: per-node / per-brick timing accumulation + counters + report (Spec 3 section 29-30,
// ADC-459). Exercises the data structure directly (no step needed): record/aggregate, RAII scope
// timing + nesting, the disabled no-op path, counters, and the report contents.

#include <adc/runtime/program/profiler.hpp>

#include "test_harness.hpp"  // adc::test::Checker (shared counter + assertion)

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>

using adc::runtime::program::Profiler;
using adc::runtime::program::ProfileScope;

namespace {

void busy_us(int micros) {
  // A real elapsed interval (steady_clock-measurable) without depending on sleep precision.
  const auto t0 = std::chrono::steady_clock::now();
  const auto want = std::chrono::microseconds(micros);
  while (std::chrono::steady_clock::now() - t0 < want) {
    // spin
  }
}

}  // namespace

int main() {
  adc::test::Checker chk;

  // --- record() aggregates count / total / min / max ---
  {
    Profiler p;
    p.enable();
    p.record("riemann", 0.20);
    p.record("riemann", 0.10);
    p.record("riemann", 0.30);
    const auto* e = p.entry("riemann");
    chk(e != nullptr, "entry_present");
    chk(e != nullptr && e->count == 3, "count");
    chk(e != nullptr && std::abs(e->total_s - 0.60) < 1e-12, "total");
    chk(e != nullptr && std::abs(e->min_s - 0.10) < 1e-12, "min");
    chk(e != nullptr && std::abs(e->max_s - 0.30) < 1e-12, "max");
    chk(e != nullptr && std::abs(e->mean_s() - 0.20) < 1e-12, "mean");
    chk(std::abs(p.total_s() - 0.60) < 1e-12, "total_s");
    chk(p.scope_count() == 1, "scope_count");
  }

  // --- disabled is a no-op (record / count ignored) ---
  {
    Profiler p;  // disabled by default
    chk(!p.enabled(), "disabled_by_default");
    p.record("x", 1.0);
    p.count("k");
    chk(p.entry("x") == nullptr, "disabled_record_noop");
    chk(p.counter("k") == 0, "disabled_count_noop");
    p.enable();
    p.record("x", 1.0);
    p.disable();
    p.record("x", 1.0);  // ignored
    chk(p.entry("x") != nullptr && p.entry("x")->count == 1, "reenable_then_disable");
  }

  // --- counters accumulate; unknown counter reads 0 ---
  {
    Profiler p;
    p.enable();
    p.count("kernels", 3);
    p.count("kernels");
    p.count("cache_hits", 5);
    chk(p.counter("kernels") == 4, "counter_sum");
    chk(p.counter("cache_hits") == 5, "counter_value");
    chk(p.counter("never") == 0, "counter_absent_zero");
  }

  // --- count_max tracks a PEAK, not a sum (scratch peak memory, ADC-459) ---
  {
    Profiler p;
    p.enable();
    p.count_max("scratch_peak_bytes", 100);  // first-seen creates at the value
    chk(p.counter("scratch_peak_bytes") == 100, "count_max_first");
    p.count_max("scratch_peak_bytes", 40);  // smaller: peak unchanged
    chk(p.counter("scratch_peak_bytes") == 100, "count_max_keeps_peak");
    p.count_max("scratch_peak_bytes", 250);  // larger: peak rises
    chk(p.counter("scratch_peak_bytes") == 250, "count_max_rises");
    // disabled count_max is a no-op
    p.disable();
    p.count_max("scratch_peak_bytes", 9999);
    p.enable();
    chk(p.counter("scratch_peak_bytes") == 250, "count_max_disabled_noop");
  }

  // --- ProfileScope times a real interval; nested scopes both record ---
  {
    Profiler p;
    p.enable();
    {
      ProfileScope outer(p, "step");
      busy_us(300);
      {
        ProfileScope inner(p, "riemann");
        busy_us(150);
      }
    }
    const auto* step = p.entry("step");
    const auto* riemann = p.entry("riemann");
    chk(step != nullptr && step->count == 1, "scope_outer_recorded");
    chk(riemann != nullptr && riemann->count == 1, "scope_inner_recorded");
    // the outer scope encloses the inner, so it is at least as long
    chk(step != nullptr && riemann != nullptr && step->total_s >= riemann->total_s,
        "scope_nesting_order");
    chk(step != nullptr && step->total_s > 0.0, "scope_positive_time");
  }

  // --- a disabled ProfileScope records nothing ---
  {
    Profiler p;  // off
    {
      ProfileScope s(p, "ignored");
      busy_us(100);
    }
    chk(p.entry("ignored") == nullptr, "disabled_scope_noop");
  }

  // --- report() is in first-seen order and contains scopes + counters ---
  {
    Profiler p;
    p.enable();
    p.record("fields", 0.5);
    p.record("transport", 0.25);
    p.count("nodes_skipped", 2);
    const std::string r = p.report();
    chk(r.find("fields") != std::string::npos, "report_has_fields");
    chk(r.find("transport") != std::string::npos, "report_has_transport");
    chk(r.find("nodes_skipped=2") != std::string::npos, "report_has_counter");
    chk(r.find("fields") < r.find("transport"), "report_first_seen_order");
  }

  // --- reset() clears everything ---
  {
    Profiler p;
    p.enable();
    p.record("a", 1.0);
    p.count("c");
    p.reset();
    chk(p.entry("a") == nullptr && p.counter("c") == 0 && p.scope_count() == 0, "reset");
  }

  if (chk.fails() == 0) {
    std::printf("OK test_profiler\n");
  }
  return chk.failed();
}
