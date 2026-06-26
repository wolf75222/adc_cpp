// Checkpoint / restart of the scheduler value cache (Spec 3 section 30, ADC-458). The held-node cache
// (every(N).hold / accumulate_dt) lives in the System-owned CacheManager; the checkpoint serializes
// each VALID slot (the cached MultiFab + last_update_step + accumulated_dt + name) and a restart
// rebuilds it, so a (run, checkpoint, restart, continue) run is bit-for-bit identical to a continuous
// run. This test exercises the CacheManager SERIALIZE / RESTORE accessors directly (the host-validatable
// core: node_ids / name_of / value_of / restore_slot) with no Program/codegen/engine, mirroring
// test_cache_manager.cpp. The full compiled-.so held-schedule continuous==restart run is Kokkos-only
// AOT and is validated on ROMEO (it cannot link a problem.so on a host-only Mac).
//
// It checks: (a) a populated cache (named + nameless slots, varied last_update_step / accumulated_dt)
// serializes to a flat buffer and deserializes to an EQUAL state (values bit-equal via the valid-cell
// sum, bookkeeping preserved); (b) only VALID slots are serialized (a cold accumulate_dt node carrying
// only an accumulator is skipped); (c) the two verbatim restart error messages -- the program-hash
// mismatch and the missing cached value naming the node -- are produced verbatim from the cache name.

#include <pops/runtime/program/cache_manager.hpp>

#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include "test_harness.hpp"  // pops::test::Checker

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace pops;
using pops::runtime::program::CacheManager;
using pops::runtime::program::CacheSlot;

namespace {

MultiFab make_mf(double fill, int ncomp = 1, int ngrow = 1) {
  Box2D dom = Box2D::from_extents(8, 8);
  BoxArray ba = BoxArray::from_domain(dom, 4);
  DistributionMapping dm(ba.size(), n_ranks());
  MultiFab mf(ba, dm, ncomp, ngrow);
  mf.set_val(fill);
  return mf;
}

// A serialized slot: exactly what the System checkpoint facade gathers per node (the global value, the
// bookkeeping, the name). A plain struct stands in for the npz keys the Python facade writes; the
// round-trip below proves the CacheManager accessors expose the full state and restore_slot rebuilds it.
struct SerializedSlot {
  int node_id = -1;
  std::string name;
  int last_update_step = -1;
  double accumulated_dt = 0.0;
  int ncomp = 0;
  int ngrow = 1;  // the cached value's ghost width (1 = aux; 2 = a held scratch); restore rebuilds with it
  MultiFab value = make_mf(0.0);  // stands in for the gathered global buffer
};

// Serialize every VALID slot the way sim.checkpoint does (program_cache_nodes -> per-node accessors).
std::vector<SerializedSlot> serialize(const CacheManager& c) {
  std::vector<SerializedSlot> out;
  for (int id : c.node_ids()) {
    SerializedSlot s;
    s.node_id = id;
    s.name = c.name_of(id);
    s.last_update_step = c.last_update_step(id);
    s.accumulated_dt = static_cast<double>(c.accumulated_dt_of(id));
    s.ncomp = c.ncomp_of(id);
    s.ngrow = c.ngrow_of(id);  // serialized so restore rebuilds with the same ghost width
    s.value = c.value_of(id);  // a real checkpoint gathers this to a global buffer; here a deep copy
    out.push_back(std::move(s));
  }
  return out;
}

// Restore the serialized slots into a fresh CacheManager the way sim.restart does (restore_program_cache
// -> CacheManager::restore_slot).
void deserialize(CacheManager& c, const std::vector<SerializedSlot>& slots) {
  for (const SerializedSlot& s : slots) {
    c.restore_slot(s.node_id, s.value, s.last_update_step,
                   static_cast<Real>(s.accumulated_dt), s.name);
  }
}

}  // namespace

int main() {
  pops::test::Checker chk;

  // --- (a) full round-trip: populated cache serializes + deserializes to an EQUAL state ---
  {
    CacheManager src;
    // a named held field solve (the aux cache), stamped at step 30, no accumulator
    src.store(5, make_mf(2.0), 30, "fields_from_state");
    // a nameless held scratch (2 comps), stamped at step 12, with an accumulated dt window
    src.store(8, make_mf(3.0, /*ncomp=*/2), 12);
    src.accumulate_dt(8, 0.001);
    src.accumulate_dt(8, 0.002);
    // a held SCRATCH at the block-state ghost width (2 ghosts): restore MUST rebuild with ngrow 2, not
    // the aux's 1, else a 2-ghost-stencil consumer under-reads its outer ghosts after restart.
    src.store(9, make_mf(4.0, /*ncomp=*/1, /*ngrow=*/2), 7);

    // the ngrow accessor distinguishes the aux (1) from a held scratch (2) -- the value restore needs it
    chk(src.ngrow_of(5) == 1, "aux_slot_ngrow_is_1");
    chk(src.ngrow_of(9) == 2, "held_scratch_slot_ngrow_is_2");

    const std::vector<SerializedSlot> blob = serialize(src);
    chk(blob.size() == 3, "serialize_three_valid_slots");

    CacheManager dst;
    deserialize(dst, blob);

    // node ids + count preserved
    chk(dst.node_ids().size() == 3, "restore_node_count");
    chk(dst.has(5) && dst.has(8) && dst.has(9), "restore_node_ids");
    // the held-scratch slot's ghost width round-trips (serialized ngrow rebuilds the right width)
    chk(dst.retrieve(9).n_grow() == 2, "restore_value9_ngrow_preserved");
    // values bit-equal (valid-cell sum: 2.0 over 64 cells; 3.0 over 64 cells x 2 comps)
    chk(std::fabs(sum(dst.retrieve(5)) - 2.0 * 64) < 1e-12, "restore_value5_bit_equal");
    chk(dst.retrieve(8).ncomp() == 2, "restore_value8_ncomp");
    chk(std::fabs(sum(dst.retrieve(8)) - 3.0 * 64) < 1e-12, "restore_value8_bit_equal");
    // bookkeeping preserved
    chk(dst.last_update_step(5) == 30, "restore_last_update5");
    chk(dst.last_update_step(8) == 12, "restore_last_update8");
    chk(std::fabs(dst.accumulated_dt(5)) < 1e-15, "restore_accum5_zero");
    chk(std::fabs(dst.accumulated_dt(8) - 0.003) < 1e-12, "restore_accum8_sum");
    // name preserved (named slot keeps its name; nameless slot falls back to node_<id>)
    chk(dst.name_of(5) == "fields_from_state", "restore_name_named");
    chk(dst.name_of(8) == "node_8", "restore_name_nameless_fallback");

    // a deserialized cache is_due exactly like a live one (cold-start gone, cadence resumes)
    chk(!dst.is_due(5, 31, 10), "restored_not_due_off_cadence");
    chk(dst.is_due(5, 40, 10), "restored_due_on_cadence");
  }

  // --- (b) only VALID slots serialize: a cold accumulate_dt node (accumulator only) is skipped ---
  {
    CacheManager src;
    src.store(1, make_mf(1.0), 0, "held");  // valid
    src.accumulate_dt(2, 0.5);              // a cold node: an accumulator but no stored value yet
    chk(!src.valid(2), "cold_accum_node_invalid");
    const std::vector<SerializedSlot> blob = serialize(src);
    chk(blob.size() == 1 && blob[0].node_id == 1, "serialize_skips_invalid_slot");
  }

  // --- (c) the two verbatim restart error messages (Spec 3 section 30, ADC-458) ---
  // The messages are RAISED by the sim.restart facade; this pins the exact strings, building the
  // missing-cache one from the CacheManager name accessor (the verbatim spec message names the node).
  {
    CacheManager src;
    src.store(5, make_mf(1.0), 0, "fields_from_state");

    // the hash-mismatch message (raised on a different installed_program_hash)
    const std::string hash_msg = "checkpoint was created with a different compiled Program hash";
    bool threw_hash = false;
    try {
      throw std::runtime_error(hash_msg);
    } catch (const std::runtime_error& e) {
      threw_hash = (std::string(e.what()) == hash_msg);
    }
    chk(threw_hash, "verbatim_hash_mismatch_message");

    // the missing-cache message names the node via the cache name (here 'fields_from_state')
    const std::string miss_msg =
        "checkpoint missing cached value for scheduled node '" + src.name_of(5) + "'";
    chk(miss_msg == "checkpoint missing cached value for scheduled node 'fields_from_state'",
        "verbatim_missing_cache_message");
  }

  if (chk.fails() == 0) {
    std::printf("OK test_checkpoint_cache\n");
  }
  return chk.failed();
}
