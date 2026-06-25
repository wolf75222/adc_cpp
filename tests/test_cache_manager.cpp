// CacheManager: the per-node value cache backing the unified Program scheduler (Spec 3, ADC-458).
// Exercises the due/store/retrieve/accumulate logic + cold-start + a MultiFab store-retrieve
// bit-identity, with no Program/codegen needed.

#include <adc/runtime/program/cache_manager.hpp>

#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/layout/distribution_mapping.hpp>
#include <adc/mesh/storage/multifab.hpp>

#include "test_harness.hpp"  // adc::test::Checker

#include <cmath>
#include <cstdio>

using namespace adc;
using adc::runtime::program::CacheManager;

namespace {

MultiFab make_mf(double fill) {
  Box2D dom = Box2D::from_extents(8, 8);
  BoxArray ba = BoxArray::from_domain(dom, 4);
  DistributionMapping dm(ba.size(), n_ranks());
  MultiFab mf(ba, dm, /*ncomp=*/1, /*ngrow=*/1);
  mf.set_val(fill);
  return mf;
}

}  // namespace

int main() {
  adc::test::Checker chk;

  // --- is_due: cold start is always due; then every N ---
  {
    CacheManager c;
    chk(c.is_due(1, 0, 10), "cold_start_due");  // never stored -> due
    chk(!c.has(1), "cold_start_absent");
    c.store(1, make_mf(1.0), 0);
    chk(c.has(1), "present_after_store");
    chk(c.is_due(1, 0, 10), "due_at_step0");  // 0 % 10 == 0
    chk(!c.is_due(1, 1, 10), "not_due_at_step1");
    chk(!c.is_due(1, 9, 10), "not_due_at_step9");
    chk(c.is_due(1, 10, 10), "due_at_step10");
    chk(c.is_due(1, 20, 10), "due_at_step20");
    chk(c.is_due(1, 5, 1), "every1_always_due");  // every_n<=1 -> always
  }

  // --- store / retrieve bit-identity (via the valid-cell sum) ---
  {
    CacheManager c;
    MultiFab v = make_mf(2.0);
    const double want = sum(v);  // 2.0 * 64 valid cells
    c.store(7, v, 3);
    const MultiFab& got = c.retrieve(7);
    chk(std::fabs(sum(got) - want) < 1e-12, "retrieve_bit_identity");
    chk(c.last_update_step(7) == 3, "last_update_step");
    // a second store overwrites + refreshes the step
    c.store(7, make_mf(5.0), 11);
    chk(std::fabs(sum(c.retrieve(7)) - 5.0 * 64) < 1e-12, "store_overwrites");
    chk(c.last_update_step(7) == 11, "last_update_step_refreshed");
  }

  // --- accumulate_dt: sums skipped dt, store resets it ---
  {
    CacheManager c;
    c.store(2, make_mf(1.0), 0);
    chk(std::fabs(c.accumulated_dt(2)) < 1e-15, "accum_zero_after_store");
    c.accumulate_dt(2, 0.001);
    c.accumulate_dt(2, 0.002);
    c.accumulate_dt(2, 0.0005);
    chk(std::fabs(c.accumulated_dt(2) - 0.0035) < 1e-12, "accum_sum");  // real sum, not N*dt
    c.store(2, make_mf(1.0), 10);  // recompute resets the accumulator
    chk(std::fabs(c.accumulated_dt(2)) < 1e-15, "accum_reset_on_store");
  }

  // --- multiple independent nodes + clear ---
  {
    CacheManager c;
    c.store(1, make_mf(1.0), 0);
    c.store(2, make_mf(2.0), 0);
    chk(c.size() == 2, "two_slots");
    chk(std::fabs(sum(c.retrieve(1)) - 64.0) < 1e-12, "node1_independent");
    chk(std::fabs(sum(c.retrieve(2)) - 128.0) < 1e-12, "node2_independent");
    c.clear();
    chk(c.size() == 0 && !c.has(1), "clear");
  }

  if (chk.fails() == 0) {
    std::printf("OK test_cache_manager\n");
  }
  return chk.failed();
}
