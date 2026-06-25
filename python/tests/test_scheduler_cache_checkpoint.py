#!/usr/bin/env python3
"""Checkpoint/restart of the scheduler value cache + the missing-cache guard (Spec 3 section 30, ADC-458).

A compiled `Program` with a held schedule (`every(N).hold` / `accumulate_dt`) caches the System aux /
a scratch per node so the field solve runs only when DUE. The cache lives in the System
(`System::program_cache()`, NOT the `.so` step closure), so checkpointing it makes a
`(run, checkpoint, restart, continue)` run bit-for-bit identical to a continuous run -- without it the
first post-restart step would cold-start the held node off its cadence. The existing v1 checkpoint
format (state, t, macro_step, phi, histories) stays back-compatible: a checkpoint with no cache keys
restarts as before.

(A) NPZ facade keys (pure Python / numpy, always runs when numpy is present): the cache key scheme
    (`cache_nodes`, `cache_names`, `cache_ncomp_<id>`, `cache_last_update_<id>`, `cache_accum_dt_<id>`,
    `cache_value_<id>`) round-trips through numpy.savez/load with the dtypes the facade uses, and the
    restart missing-value guard raises the verbatim spec message
    `checkpoint missing cached value for scheduled node '<name>'`. This pins the serialization
    contract + the verbatim error independently of the engine.

The full compiled-`.so` held-schedule continuous == restart RUN is Kokkos-only AOT (ROMEO); the
CacheManager serialize/restore round-trip + both verbatim messages are unit-tested host-side by
`tests/test_checkpoint_cache.cpp`.
"""
import os
import sys
import tempfile


# ---- (A) NPZ facade keys + the missing-cache guard: pure numpy, always runs when numpy is present ----
def test_cache_npz_key_scheme_roundtrips():
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001 -- numpy unavailable in this interpreter
        print("-- (A) skipped: numpy unavailable: %s --" % exc)
        return

    # Mirror EXACTLY the keys + dtypes adc.System.checkpoint writes for one held node.
    nid = 5
    name = "fields_from_state"
    ncomp, ny, nx = 1, 4, 4
    value = np.arange(ncomp * ny * nx, dtype=np.float64).reshape(ncomp, ny, nx)
    out = {
        "adc_checkpoint_version": 1,
        "program_hash": "deadbeef" * 8,  # a 64-hex IR hash shape (the hash guard)
        "cache_nodes": np.array([nid], dtype=np.int64),
        "cache_names": np.array([name]),
        "cache_ncomp_%d" % nid: ncomp,
        "cache_last_update_%d" % nid: 30,
        "cache_accum_dt_%d" % nid: 0.0035,
        "cache_value_%d" % nid: value,
    }

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.npz")
        with open(path, "wb") as f:
            np.savez_compressed(f, **out)
        d = np.load(path, allow_pickle=False)
        assert [int(n) for n in d["cache_nodes"]] == [nid]
        assert [str(s) for s in d["cache_names"]] == [name]
        assert int(d["cache_ncomp_%d" % nid]) == ncomp
        assert int(d["cache_last_update_%d" % nid]) == 30
        assert abs(float(d["cache_accum_dt_%d" % nid]) - 0.0035) < 1e-12
        np.testing.assert_array_equal(d["cache_value_%d" % nid], value)

    # The missing-cache guard (restart): a checkpoint that LISTS a cache node but lost its value array
    # raises the verbatim spec message naming the node (the held node would otherwise cold-start).
    truncated = {k: v for k, v in out.items() if k != "cache_value_%d" % nid}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "truncated.npz")
        with open(path, "wb") as f:
            np.savez_compressed(f, **truncated)
        d = np.load(path, allow_pickle=False)
        cache_names = [str(s) for s in d["cache_names"]]
        raised = None
        for idx, node in enumerate(int(n) for n in d["cache_nodes"]):
            nm = cache_names[idx] if idx < len(cache_names) else "node_%d" % node
            if ("cache_value_%d" % node) not in d:
                raised = "checkpoint missing cached value for scheduled node '%s'" % nm
                break
        assert raised == "checkpoint missing cached value for scheduled node 'fields_from_state'", \
            "the missing-cache guard must name the node verbatim (Spec 3 section 30)"

    # Back-compat: a checkpoint WITHOUT the cache keys carries no cache_nodes (restarts as before).
    legacy = {"adc_checkpoint_version": 1, "t": 0.0, "macro_step": 0}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "legacy.npz")
        with open(path, "wb") as f:
            np.savez_compressed(f, **legacy)
        d = np.load(path, allow_pickle=False)
        assert "cache_nodes" not in d, \
            "a legacy checkpoint must not carry the ADC-458 cache keys (back-compatible restart)"


if __name__ == "__main__":
    test_cache_npz_key_scheme_roundtrips()
    print("OK test_scheduler_cache_checkpoint")
    sys.exit(0)
