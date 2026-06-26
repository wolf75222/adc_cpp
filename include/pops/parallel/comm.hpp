#pragma once

/// @file
/// @brief Parallel seam: minimal MPI abstraction (rank/size + collectives) with serial fallback.
///
/// Layer: `include/pops/parallel`.
/// Role: expose my_rank()/n_ranks() and a fixed set of global reductions (sum/min/max on
/// double and long, in-place sum on a double array, in-place OR on a marker array)
/// behind a single facade. Without POPS_HAS_MPI everything compiles to a serial identity; the rest
/// of the code never sees MPI_COMM_WORLD nor mpi.h directly.
/// Contract: every collective operates on MPI_COMM_WORLD; each rank must call them
/// in the same order (otherwise deadlock). The sum_inplace / or_inplace bricks feed
/// respectively the multi-patch AMR reflux and the gathering of regrid tags before
/// clustering; all_reduce_min guarantees a global dt identical on all ranks.
///
/// Invariants:
/// - even compiled with MPI, if MPI is not initialized my_rank()=0 / n_ranks()=1 (comm_active()
///   tests Initialized && !Finalized); call comm_init() at the start of main() for a distributed run;
/// - the all_reduce_* functions are COLLECTIVE: all ranks participate or none;
/// - in serial mode each function is the identity (no-op or returns the argument).

#ifdef POPS_HAS_MPI
#include <mpi.h>
#endif

namespace pops {

#ifdef POPS_HAS_MPI

inline bool comm_active() {
  int inited = 0, fin = 0;
  MPI_Initialized(&inited);
  MPI_Finalized(&fin);
  return inited && !fin;
}

inline void comm_init(int* argc = nullptr, char*** argv = nullptr) {
  int inited = 0;
  MPI_Initialized(&inited);
  if (!inited)
    MPI_Init(argc, argv);
}

inline void comm_finalize() {
  int fin = 0;
  MPI_Finalized(&fin);
  if (!fin)
    MPI_Finalize();
}

inline int my_rank() {
  if (!comm_active())
    return 0;
  int r = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}

inline int n_ranks() {
  if (!comm_active())
    return 1;
  int s = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &s);
  return s;
}

inline void barrier() {
  if (comm_active())
    MPI_Barrier(MPI_COMM_WORLD);
}

inline double all_reduce_sum(double x) {
  if (!comm_active())
    return x;
  double r = x;
  MPI_Allreduce(&x, &r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return r;
}

inline double all_reduce_max(double x) {
  if (!comm_active())
    return x;
  double r = x;
  MPI_Allreduce(&x, &r, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return r;
}

// Global min (counterpart of all_reduce_max). Brick for GLOBAL time step bounds
// (System::add_dt_bound): the host callback is evaluated PER RANK, the global min guarantees a dt
// IDENTICAL on all ranks (otherwise the collectives of the step -- Krylov, fill_boundary --
// would diverge -> deadlock). In serial: identity.
inline double all_reduce_min(double x) {
  if (!comm_active())
    return x;
  double r = x;
  MPI_Allreduce(&x, &r, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  return r;
}

// Element-by-element sum of an array, in place, on all ranks. Base brick of the
// distributed multi-patch AMR reflux: each rank fills the contributions of
// its local patches (0 elsewhere), all-reduce -> each rank has the complete register.
inline void all_reduce_sum_inplace(double* buf, int n) {
  if (!comm_active() || n <= 0)
    return;
  MPI_Allreduce(MPI_IN_PLACE, buf, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

inline long all_reduce_sum(long x) {
  if (!comm_active())
    return x;
  long r = x;
  MPI_Allreduce(&x, &r, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  return r;
}

// Element-by-element logical OR of a marker array (0/1), in place, on all ranks.
// Brick of the DISTRIBUTED-COARSE AMR regrid: each rank tags ONLY its local coarse
// boxes (tag_cells iterating over local_size()), so nobody sees the complete tag grid;
// the global OR gathers the tags on each rank before the Berger-Rigoutsos clustering, which then
// produces IDENTICAL fine patches everywhere (otherwise the fine BoxArray would differ per rank -> MPI
// desynchronized). See the note in tag_box.hpp ("the distributed tags will be gathered before clustering").
inline void all_reduce_or_inplace(char* buf, int n) {
  if (!comm_active() || n <= 0)
    return;
  MPI_Allreduce(MPI_IN_PLACE, buf, n, MPI_CHAR, MPI_BOR, MPI_COMM_WORLD);
}

#else  // ----- serial -----

inline bool comm_active() {
  return false;
}
inline void comm_init(int* = nullptr, char*** = nullptr) {}
inline void comm_finalize() {}
inline int my_rank() {
  return 0;
}
inline int n_ranks() {
  return 1;
}
inline void barrier() {}
inline double all_reduce_sum(double x) {
  return x;
}
inline double all_reduce_max(double x) {
  return x;
}
inline double all_reduce_min(double x) {
  return x;
}
inline long all_reduce_sum(long x) {
  return x;
}
inline void all_reduce_sum_inplace(double*, int) {}  // serial: identity
inline void all_reduce_or_inplace(char*, int) {}     // serial: identity

#endif

}  // namespace pops
