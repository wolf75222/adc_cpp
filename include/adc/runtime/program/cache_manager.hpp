#pragma once

// Per-node value cache for the unified Program scheduler (Spec 3 section 17-18, ADC-458). A Program
// node with a non-always schedule (e.g. every(N).hold) recomputes only when DUE and reuses the
// cached field in between; an accumulate_dt policy also tracks the summed dt of the skipped steps so
// the held result is applied with eff_dt = sum(dt_skipped), not N * dt_current. This header owns the
// typed cache (a MultiFab + its bookkeeping) and the due/store/retrieve/accumulate logic; the
// codegen wraps a scheduled node in `if (due) { recompute; store } else { retrieve }`, and the
// ProgramContext owns one CacheManager per installed Program. Host/serial today; checkpoint of the
// slots is a follow-up (it mirrors the System history serialization).

#include <cstddef>
#include <map>

#include <adc/core/foundation/types.hpp>  // Real
#include <adc/mesh/storage/multifab.hpp>  // MultiFab

namespace adc::runtime::program {

// One cached node value plus the bookkeeping the schedule needs.
struct CacheSlot {
  MultiFab value;                 // the cached field/state (a deep copy of the last recompute)
  int last_update_step = -1;      // macro step at the last recompute (-1 = never)
  Real accumulated_dt = Real(0);  // for accumulate_dt: summed dt of the steps skipped since
  bool valid = false;             // false until the first store (a cold-start node is always due)
};

class CacheManager {
 public:
  // Is node `node_id` due to recompute at `macro_step`? A node never stored (cold start) is always
  // due; otherwise it is due every `every_n` macro-steps (every_n <= 1 means every step).
  bool is_due(int node_id, int macro_step, int every_n) const {
    auto it = slots_.find(node_id);
    if (it == slots_.end() || !it->second.valid) {
      return true;
    }
    if (every_n <= 1) {
      return true;
    }
    return (macro_step % every_n) == 0;
  }

  // Store `value` as node `node_id`'s cached result computed at `macro_step`; resets accumulated_dt
  // (the held value is now fresh).
  void store(int node_id, const MultiFab& value, int macro_step) {
    CacheSlot& s = slots_[node_id];
    s.value = value;  // deep copy: the cache outlives the live scratch
    s.last_update_step = macro_step;
    s.accumulated_dt = Real(0);
    s.valid = true;
  }

  // The cached value of `node_id` (must be valid: guard with is_due()==false first).
  const MultiFab& retrieve(int node_id) const { return slots_.at(node_id).value; }

  // Add a skipped step's dt to node `node_id`'s accumulator (accumulate_dt policy).
  void accumulate_dt(int node_id, Real dt) { slots_.at(node_id).accumulated_dt += dt; }

  Real accumulated_dt(int node_id) const { return slots_.at(node_id).accumulated_dt; }

  bool has(int node_id) const {
    auto it = slots_.find(node_id);
    return it != slots_.end() && it->second.valid;
  }

  int last_update_step(int node_id) const {
    auto it = slots_.find(node_id);
    return it == slots_.end() ? -1 : it->second.last_update_step;
  }

  std::size_t size() const { return slots_.size(); }

  void clear() { slots_.clear(); }

 private:
  std::map<int, CacheSlot> slots_;  // node id (IR Value.id) -> its cache slot
};

}  // namespace adc::runtime::program
