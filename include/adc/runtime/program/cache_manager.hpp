#pragma once

// Per-node value cache for the unified Program scheduler (Spec 3 section 17-18, ADC-458). A Program
// node with a non-always schedule (e.g. every(N).hold) recomputes only when DUE and reuses the
// cached field in between; an accumulate_dt policy also tracks the summed dt of the skipped steps so
// the held result is applied with eff_dt = sum(dt_skipped), not N * dt_current. This header owns the
// typed cache (a MultiFab + its bookkeeping) and the due/store/retrieve/accumulate logic; the
// codegen wraps a scheduled node in `if (due) { recompute; store } else { retrieve }`, and the
// ProgramContext owns one CacheManager per installed Program. Host/serial today; checkpoint of the
// slots is a follow-up (ADC-458 section 30, it mirrors the System history serialization) -- this
// header exposes the in-memory state + accessors the serializer will read, it does not own the I/O.
//
// A slot caches EITHER the System aux (a held field solve restores phi/grad/E) OR a named scratch
// MultiFab (a held rhs / source / linear_combine restores its own scratch buffer). The store/retrieve
// API is the same in both cases (a deep MultiFab copy keyed by the IR node id); the codegen picks the
// aux or the scratch as the value to cache per node.

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

  // Add a skipped step's dt to node `node_id`'s accumulator (accumulate_dt policy). Creates the slot
  // if the node was never stored (a cold accumulate_dt node accumulates from its first skipped step,
  // before any recompute), so this never throws on an absent slot -- the sum is the actual skipped dt.
  void accumulate_dt(int node_id, Real dt) { slots_[node_id].accumulated_dt += dt; }

  // The accumulated skipped dt of `node_id` (0 if the node has no slot yet).
  Real accumulated_dt(int node_id) const {
    auto it = slots_.find(node_id);
    return it == slots_.end() ? Real(0) : it->second.accumulated_dt;
  }

  // The effective dt a due accumulate_dt step should apply: the actual dt of THIS step plus the dt
  // summed over the steps skipped since the last recompute. CRITICAL with a variable step_cfl: this
  // is the real sum of the skipped dt, never N * dt_current. Resets the accumulator (a fresh window
  // starts after this recompute); call once at the start of a due accumulate_dt step.
  Real effective_dt(int node_id, Real dt_now) {
    CacheSlot& s = slots_[node_id];
    const Real eff = dt_now + s.accumulated_dt;
    s.accumulated_dt = Real(0);
    return eff;
  }

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
