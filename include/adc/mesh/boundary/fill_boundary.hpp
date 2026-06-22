/// @file
/// @brief fill_boundary: INTRA-level halo exchange (fills ghosts from neighbors).
///
/// Fills the ghosts of each Fab from the VALID regions of neighboring boxes in the same
/// MultiFab, with optional periodic wrapping. Single-rank: direct memory copies. Multi-rank
/// (ADC_HAS_MPI + n_ranks()>1): metadata is REPLICATED -> each rank enumerates DETERMINISTICALLY
/// the same job list, so buffers line up without negotiating sizes (MPI_Isend/Irecv,
/// tag 0). Two-phase API (classic compute/comm overlap): fill_boundary_begin posts the
/// exchanges, fill_boundary_end waits and unpacks; fill_boundary chains both (blocking). Ghosts
/// OUTSIDE the domain without periodicity are NOT touched here (those are the physical BCs,
/// physical_bc.hpp). The pack/unpack kernels are device-clean NAMED FUNCTORS (nvcc limitation).
///
/// The job schedule (BoxHash + local/global enumeration) is MEMOIZED per (layout, Periodicity,
/// domain) on the MultiFab (ADC-260, halo_schedule.hpp): it is a pure function of the invariant
/// layout, so only the copy/pack/MPI/unpack of the live data reruns. The plan is replayed in the
/// SAME deterministic order as the original inline enumeration -> bit-identical buffers.

#pragma once

#include <adc/mesh/index/box2d.hpp>
#include <adc/mesh/index/box_hash.hpp>
#include <adc/mesh/storage/fab2d.hpp>
#include <adc/mesh/execution/for_each.hpp>
#include <adc/mesh/boundary/halo_schedule.hpp>
#include <adc/mesh/storage/multifab.hpp>
#include <adc/parallel/comm.hpp>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace adc {

/// Per-direction periodicity: halo wrapping in x and/or y during the exchange (false = open edge,
/// left to the physical BCs).
struct Periodicity {
  bool x = false;
  bool y = false;
};

namespace detail {

// NAMED FUNCTORS (not ADC_HD lambdas) for the halo-exchange kernels. Same reasons as the rest of the
// elliptic/mesh path (#93, recipe #64): fill_boundary is first instantiated from the MG V-cycle pulled
// from an external TU; an extended lambda there stalls device kernel emission under nvcc (-O Release
// without -g). Strictly identical body -> bit-identical CPU and device.
struct CopyShiftedKernel {
  Array4 d;
  ConstArray4 s;
  int sx, sy, c;
  ADC_HD void operator()(int i, int j) const { d(i, j, c) = s(i - sx, j - sy, c); }
};

// dst(i, j, c) = src(i - sx, j - sy, c) for (i, j) in region.
inline void copy_shifted(Fab2D& dst, const Fab2D& src, const Box2D& region, int sx, int sy,
                         int ncomp) {
  Array4 d = dst.array();
  ConstArray4 s = src.const_array();
  for (int c = 0; c < ncomp; ++c)
    for_each_cell(region, CopyShiftedKernel{d, s, sx, sy, c});
}

// Pack of a send job: sb[b0 + c*rsz + off] = s(i - sx, jc - sy, c), off = (jc-lo1)*rnx + (i-lo0).
struct PackKernel {
  Real* sb;
  ConstArray4 s;
  std::int64_t b0, rsz;
  int lo0, lo1, rnx, sx, sy, ncl;
  ADC_HD void operator()(int i, int jc) const {
    const std::int64_t off = static_cast<std::int64_t>(jc - lo1) * rnx + (i - lo0);
    for (int c = 0; c < ncl; ++c)
      sb[b0 + static_cast<std::int64_t>(c) * rsz + off] = s(i - sx, jc - sy, c);
  }
};

// Unpack of a receive job: d(i, jc, c) = rb[b0 + c*rsz + off], off = (jc-lo1)*rnx + (i-lo0).
struct UnpackKernel {
  const Real* rb;
  Array4 d;
  std::int64_t b0, rsz;
  int lo0, lo1, rnx, ncl;
  ADC_HD void operator()(int i, int jc) const {
    const std::int64_t off = static_cast<std::int64_t>(jc - lo1) * rnx + (i - lo0);
    for (int c = 0; c < ncl; ++c)
      d(i, jc, c) = rb[b0 + static_cast<std::int64_t>(c) * rsz + off];
  }
};

// Enumerates the halo schedule for (mf layout, per, domain): the BoxHash build + the local (dst AND
// src local) and, under MPI with n_ranks()>1, the global (cross-rank send/recv) job lists. This is
// the per-call work that ADC-260 hoists out of fill_boundary_begin; it runs ONCE per distinct
// (layout, Periodicity, domain) and is then replayed from the cache. Jobs are produced in the SAME
// deterministic order as the legacy inline loops (local: li x shifts x sorted gB; global: gF x
// shifts x sorted gB), so the packed buffers stay bit-identical and the per-rank send/recv lists
// stay aligned. Bumps the build counter (cache-engagement test hook).
inline void build_halo_schedule(const MultiFab& mf, const Box2D& domain, Periodicity per,
                                HaloSchedule& sched) {
  ++halo_schedule_build_counter();
  const int ng = mf.n_grow();
  const int Lx = domain.nx();
  const int Ly = domain.ny();
  const BoxArray& ba = mf.box_array();

  std::vector<int> sxv = {0};
  if (per.x) {
    sxv.push_back(Lx);
    sxv.push_back(-Lx);
  }
  std::vector<int> syv = {0};
  if (per.y) {
    syv.push_back(Ly);
    syv.push_back(-Ly);
  }
  std::vector<std::pair<int, int>> shifts;
  for (int sx : sxv)
    for (int sy : syv)
      shifts.push_back({sx, sy});

  // spatial hash: restricts the neighbor-box search (see box_hash.hpp).
  const BoxHash hash(ba, suggest_bin(ba));

  // --- local jobs (local dst AND local src) ---
  for (int li = 0; li < mf.local_size(); ++li) {
    const int gF = mf.global_index(li);
    const Box2D gbox = mf.fab(li).box().grow(ng);
    for (auto [sx, sy] : shifts) {
      const Box2D Q = gbox.shift(0, -sx).shift(1, -sy);
      for (int gB : hash.query(Q)) {
        if (gB == gF && sx == 0 && sy == 0)
          continue;  // self, without shift
        const int srcLocal = mf.local_index_of(gB);
        if (srcLocal < 0)
          continue;  // non-local src -> MPI below
        const Box2D region = gbox.intersect(ba[gB].shift(0, sx).shift(1, sy));
        if (region.empty())
          continue;
        sched.local.push_back({gB, gF, sx, sy, region});
      }
    }
  }

#ifdef ADC_HAS_MPI
  const int np = n_ranks();
  if (np > 1) {
    const int me = my_rank();
    const DistributionMapping& dm = mf.dmap();
    sched.send.assign(np, {});
    sched.recv.assign(np, {});
    // deterministic global enumeration: (dst gF) x shifts x hash candidates (gB sorted). Identical
    // on all ranks -> aligned send/recv lists.
    for (int gF = 0; gF < ba.size(); ++gF) {
      const int od = dm[gF];
      const Box2D gbox = ba[gF].grow(ng);
      for (auto [sx, sy] : shifts) {
        const Box2D Q = gbox.shift(0, -sx).shift(1, -sy);
        for (int gB : hash.query(Q)) {
          if (gB == gF && sx == 0 && sy == 0)
            continue;
          const int os = dm[gB];
          if (od != me && os != me)
            continue;
          if (od == me && os == me)
            continue;
          const Box2D region = gbox.intersect(ba[gB].shift(0, sx).shift(1, sy));
          if (region.empty())
            continue;
          if (os == me)
            sched.send[od].push_back({gB, gF, sx, sy, region});
          else
            sched.recv[os].push_back({gB, gF, sx, sy, region});
        }
      }
    }
  }
#endif
}

// Returns the cached schedule for (mf layout, per, domain), building and memoizing it on first use.
inline std::shared_ptr<const HaloSchedule> get_halo_schedule(const MultiFab& mf,
                                                             const Box2D& domain, Periodicity per) {
  HaloScheduleCache& cache = mf.halo_cache();
  if (std::shared_ptr<const HaloSchedule> hit = cache.find(per.x, per.y, domain))
    return hit;
  std::shared_ptr<HaloSchedule> s = cache.add();
  s->per_x = per.x;
  s->per_y = per.y;
  s->domain = domain;
  build_halo_schedule(mf, domain, per, *s);
  return s;
}

}  // namespace detail

/// Opaque state of an in-flight halo exchange, returned by fill_boundary_begin and consumed by
/// fill_boundary_end. OWNS the send/receive buffers and the MPI_Request: they stay alive
/// (and at a stable address after move) until fill_boundary_end is called. Empty in single-rank.
/// Holds a shared handle to the cached schedule (ADC-260) so fill_boundary_end unpacks from the SAME
/// recv job list begin posted; the handle keeps the plan alive even if a later fill_boundary on the
/// same MultiFab appends another schedule.
struct HaloExchange {
  std::shared_ptr<const HaloSchedule> sched;  // replayed plan (null if mf has no ghost)
#ifdef ADC_HAS_MPI
  // Buffers in PINNED HOST memory (comm_allocator = Kokkos::SharedHostPinnedSpace under Kokkos,
  // std::allocator otherwise), NOT managed. The pack/unpack in for_each (device under Kokkos)
  // writes/reads directly into them since pinned host is device-accessible; BUT the pointer passed to
  // MPI is seen as HOST (cuPointerGetAttribute = HOST), so a CUDA-aware MPI (BTL smcuda) does NOT
  // attempt CUDA IPC on it. A managed/UVM pointer, on the other hand, triggered IPC, which DEADLOCKS
  // between two GPUs isolated by cgroup (srun --gpus-per-task=1: each rank sees only its GPU as device
  // 0, cuIpcOpenMemHandle of the peer's buffer impossible). See core/allocator.hpp (comm_allocator).
  std::vector<std::vector<Real, comm_allocator<Real>>> sbuf, rbuf;  // alive until end
  std::vector<MPI_Request> reqs;
  int nc = 0;
#endif
};

/// Phase 1 (non-blocking): does the LOCAL halo copies and posts the Isend/Irecv of the distant halos.
/// Returns the handle to pass to fill_boundary_end. Between begin and end the caller can advance the
/// interior. No-op if mf has no ghost. @p domain is used for periodic wrapping @p per.
inline HaloExchange fill_boundary_begin(MultiFab& mf, const Box2D& domain, Periodicity per = {}) {
  HaloExchange h;
  const int ng = mf.n_grow();
  if (ng == 0)
    return h;
  const int nc = mf.ncomp();
  // memoized schedule (BoxHash + enumeration) for this (layout, Periodicity, domain).
  const std::shared_ptr<const HaloSchedule> sched = detail::get_halo_schedule(mf, domain, per);
  h.sched = sched;

  // --- local copies (local dst AND local src), replayed from the cached plan ---
  for (const HaloJob& j : sched->local) {
    Fab2D& dst = mf.fab(mf.local_index_of(j.dst));
    const Fab2D& src = mf.fab(mf.local_index_of(j.src));
    detail::copy_shifted(dst, src, j.region, j.sx, j.sy, nc);
  }

#ifdef ADC_HAS_MPI
  if (n_ranks() <= 1)
    return h;
  const int np = n_ranks();
  h.nc = nc;

  auto buf_size = [&](const std::vector<HaloJob>& js) {
    std::int64_t n = 0;
    for (const auto& j : js)
      n += j.region.num_cells() * nc;
    return n;
  };
  h.sbuf.assign(np, {});
  h.rbuf.assign(np, {});
  // device PACK (for_each, parallel under Kokkos) into the pinned host buffers. Per-job layout:
  // c-major then (jj, ii), IDENTICAL to the old k++ order -> buffer bit-identical to the host path
  // (the CPU MPI ctests stay bit-identical at np=1/2/4). The peer rank enumerates in the same order,
  // so sbuf[A->B] and rbuf[B<-A] align without negotiating sizes.
  for (int r = 0; r < np; ++r) {
    const std::vector<HaloJob>& send_r = sched->send[r];
    if (send_r.empty())
      continue;
    h.sbuf[r].resize(buf_size(send_r));
    Real* sb = h.sbuf[r].data();
    std::int64_t base = 0;
    for (const auto& jb : send_r) {
      const ConstArray4 s = mf.fab(mf.local_index_of(jb.src)).const_array();
      const int lo0 = jb.region.lo[0], lo1 = jb.region.lo[1], rnx = jb.region.nx();
      const std::int64_t rsz = static_cast<std::int64_t>(rnx) * jb.region.ny();
      const int sx = jb.sx, sy = jb.sy, ncl = nc;
      const std::int64_t b0 = base;
      for_each_cell(jb.region, detail::PackKernel{sb, s, b0, rsz, lo0, lo1, rnx, sx, sy, ncl});
      base += rsz * nc;
    }
  }
  for (int r = 0; r < np; ++r)  // allocate the receive buffers
    if (!sched->recv[r].empty())
      h.rbuf[r].resize(buf_size(sched->recv[r]));
  device_fence();  // the pack kernels (and the local copies) must finish before MPI reads sbuf
  for (
      int r = 0; r < np;
      ++r) {  // non-blocking posting; MPI receives PINNED HOST pointers (seen HOST, no GPUDirect/CUDA IPC)
    if (!h.sbuf[r].empty()) {
      h.reqs.emplace_back();
      MPI_Isend(h.sbuf[r].data(), static_cast<int>(h.sbuf[r].size()), MPI_DOUBLE, r, 0,
                MPI_COMM_WORLD, &h.reqs.back());
    }
    if (!h.rbuf[r].empty()) {
      h.reqs.emplace_back();
      MPI_Irecv(h.rbuf[r].data(), static_cast<int>(h.rbuf[r].size()), MPI_DOUBLE, r, 0,
                MPI_COMM_WORLD, &h.reqs.back());
    }
  }
#endif
  return h;
}

/// Phase 2 (blocking): MPI_Waitall on the transfers posted by begin, then unpacks the received
/// buffers into the ghosts. @p h MUST come from the matching fill_boundary_begin on the same mf. No-op
/// in serial (no request).
inline void fill_boundary_end(MultiFab& mf, HaloExchange& h) {
#ifdef ADC_HAS_MPI
  if (h.reqs.empty())
    return;
  MPI_Waitall(static_cast<int>(h.reqs.size()), h.reqs.data(), MPI_STATUSES_IGNORE);
  // device UNPACK (for_each) from the received PINNED HOST buffers. Waitall guarantees the transfer is
  // complete; the kernel launched next reads the pinned host (device-accessible, coherent). Replayed
  // from the SAME cached recv list begin used (h.sched), so base offsets match the sender's layout.
  const HaloSchedule& sched = *h.sched;
  for (std::size_t r = 0; r < sched.recv.size(); ++r) {
    if (h.rbuf[r].empty())
      continue;
    const Real* rb = h.rbuf[r].data();
    std::int64_t base = 0;
    for (const auto& jb : sched.recv[r]) {
      Array4 d = mf.fab(mf.local_index_of(jb.dst)).array();
      const int lo0 = jb.region.lo[0], lo1 = jb.region.lo[1], rnx = jb.region.nx();
      const std::int64_t rsz = static_cast<std::int64_t>(rnx) * jb.region.ny();
      const int ncl = h.nc;
      const std::int64_t b0 = base;
      for_each_cell(jb.region, detail::UnpackKernel{rb, d, b0, rsz, lo0, lo1, rnx, ncl});
      base += rsz * ncl;
    }
  }
  // The unpack kernels above are ASYNC (device) and read h.rbuf; comm_allocator (pinned host) frees
  // IMMEDIATELY on destruction of h (no deferred free like the ManagedArena). So we drain the device
  // BEFORE the pinned buffers are freed -> no use-after-free.
  device_fence();
#else
  (void)mf;
  (void)h;
#endif
}

/// BLOCKING halo exchange: begin then end immediately (no overlap). Fills the intra-level +
/// periodic ghosts of @p mf; @p per sets the wrapping, @p domain the periodic fold.
inline void fill_boundary(MultiFab& mf, const Box2D& domain, Periodicity per = {}) {
  HaloExchange h = fill_boundary_begin(mf, domain, per);
  fill_boundary_end(mf, h);
}

}  // namespace adc
