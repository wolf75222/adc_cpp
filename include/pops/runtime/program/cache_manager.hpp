#pragma once

// Per-node value cache for the unified Program scheduler (Spec 3 section 17-18, ADC-458). A Program
// node with a non-always schedule (e.g. every(N).hold) recomputes only when DUE and reuses the
// cached field in between; an accumulate_dt policy also tracks the summed dt of the skipped steps so
// the held result is applied with eff_dt = sum(dt_skipped), not N * dt_current. This header owns the
// typed cache (a MultiFab + its bookkeeping) and the due/store/retrieve/accumulate logic; the
// codegen wraps a scheduled node in `if (due) { recompute; store } else { retrieve }`. The System
// owns one CacheManager per installed Program (System::program_cache()) so the checkpoint can reach
// it; the ProgramContext forwards its cache_* seam ops to that single System-owned manager.
//
// CHECKPOINT (Spec 3 section 30, ADC-458): the held cache state (the cached MultiFab, last_update_step,
// accumulated_dt, valid) round-trips through the EXISTING checkpoint -- exactly the way the System
// history rings do (gather_global / write_state, MPI-safe, bit-identical). The System exposes the
// serialize/restore accessors over THIS class (node_ids / name_of / valid / value_of / restore_slot);
// the sim.checkpoint / sim.restart facade gathers and scatters the slots alongside the block state.
// A restart against a DIFFERENT compiled Program is rejected by the program-hash guard, and a held
// scheduled node whose cached value the checkpoint never recorded fails loud at restart.
//
// A slot caches EITHER the System aux (a held field solve restores phi/grad/E) OR a named scratch
// MultiFab (a held rhs / source / linear_combine restores its own scratch buffer). The store/retrieve
// API is the same in both cases (a deep MultiFab copy keyed by the IR node id); the codegen picks the
// aux or the scratch as the value to cache per node. A slot carries an optional human-readable NAME
// (the scheduled operator, e.g. "fields_from_state") so the restart can name a missing node; the
// codegen's nameless store defaults it to "node_<id>", and the named store is the documented seam a
// later codegen export uses to carry the operator name verbatim.

#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <pops/core/foundation/types.hpp>  // Real
#include <pops/mesh/storage/multifab.hpp>  // MultiFab

namespace pops::runtime::program {

// One cached node value plus the bookkeeping the schedule needs.
struct CacheSlot {
  MultiFab value;                 // the cached field/state (a deep copy of the last recompute)
  int last_update_step = -1;      // macro step at the last recompute (-1 = never)
  Real accumulated_dt = Real(0);  // for accumulate_dt: summed dt of the steps skipped since
  bool valid = false;             // false until the first store (a cold-start node is always due)
  std::string name;               // scheduled node name ("fields_from_state"); "" = "node_<id>"
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

  // Same as store(), but also tags the slot with a human-readable NAME (the scheduled operator, e.g.
  // "fields_from_state") so the checkpoint can name a missing node verbatim at restart. The nameless
  // store above leaves the name empty (the checkpoint then falls back to "node_<id>"); a later codegen
  // export of the operator name routes through here without changing the cached-value semantics.
  void store(int node_id, const MultiFab& value, int macro_step, const std::string& name) {
    store(node_id, value, macro_step);
    slots_[node_id].name = name;
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

  // --- Checkpoint serialize / restore (Spec 3 section 30, ADC-458) ---
  // The System reads these to gather each slot into the checkpoint and writes restore_slot to rebuild
  // the cache on restart, mirroring the history ring serialization. Only VALID slots are serialized
  // (a never-stored cold node carries no value); the codegen-keyed integer id is the stable key.

  // Node ids of every VALID slot, ascending (std::map order). A slot is valid once it has been stored;
  // a cold accumulate_dt node that only has an accumulator (no value yet) is skipped -- its eff_dt
  // window restarts from the first post-restart skipped step (the held value it would read is absent).
  std::vector<int> node_ids() const {
    std::vector<int> ids;
    for (const auto& [id, s] : slots_) {
      if (s.valid) {
        ids.push_back(id);
      }
    }
    return ids;
  }

  // The human-readable name of node `node_id` ("fields_from_state"), or "node_<id>" when the slot was
  // stored without one (today's nameless codegen). Used to name a missing node at restart.
  std::string name_of(int node_id) const {
    auto it = slots_.find(node_id);
    if (it == slots_.end() || it->second.name.empty()) {
      return "node_" + std::to_string(node_id);
    }
    return it->second.name;
  }

  bool valid(int node_id) const {
    auto it = slots_.find(node_id);
    return it != slots_.end() && it->second.valid;
  }

  Real accumulated_dt_of(int node_id) const { return accumulated_dt(node_id); }

  // The component count of node `node_id`'s cached value (for the checkpoint's per-slot ncomp key).
  int ncomp_of(int node_id) const { return slots_.at(node_id).value.ncomp(); }

  // The ghost-cell width of node `node_id`'s cached value (for the checkpoint's per-slot ngrow key).
  // The aux is 1-ghost, but a held SCRATCH slot is allocated at the block-state width (2 ghosts), and a
  // 2-ghost-stencil consumer reads the 2nd ghost layer -- so restore MUST rebuild with the SAME ngrow,
  // not a hard-coded 1 (else a held-scratch restart under-reads its outer ghosts).
  int ngrow_of(int node_id) const { return slots_.at(node_id).value.n_grow(); }

  // The cached MultiFab of node `node_id` (the checkpoint gathers it via the System's gather_global,
  // exactly like a history slot). @throws if the slot is absent.
  const MultiFab& value_of(int node_id) const { return slots_.at(node_id).value; }

  // RESTORE (restart) node `node_id` from a checkpoint: take ownership of the restored `value` (the
  // System scattered the global buffer into it), tag the bookkeeping, and mark it valid -- the inverse
  // of node_ids/value_of. `name` may be empty (defaults to "node_<id>" on read). Replaces any existing
  // slot for that id (a re-installed Program re-keys by the same ids).
  void restore_slot(int node_id, MultiFab value, int last_update_step, Real accumulated_dt,
                    const std::string& name) {
    CacheSlot& s = slots_[node_id];
    s.value = std::move(value);
    s.last_update_step = last_update_step;
    s.accumulated_dt = accumulated_dt;
    s.valid = true;
    s.name = name;
  }

 private:
  std::map<int, CacheSlot> slots_;  // node id (IR Value.id) -> its cache slot
};

}  // namespace pops::runtime::program
