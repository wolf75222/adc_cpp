#pragma once

/// @file
/// @brief Low-level FFT brick: PoissonFFT (spectral periodic Poisson solver, slab-distributed)
///        plus 1D radix-2 FFT primitives (fft1d) and a direct DFT fallback (dft1d_direct).
///
/// Layer: `include/adc/numerics/elliptic/poisson`.
/// Role: solve EXACTLY the discrete periodic Laplacian lap_h phi = rho with phi of zero mean,
/// in ONE transform (no iteration, no tolerance). Works on slabs (flat per-rank vectors, NOT
/// MultiFab): this is the brick wrapped by PoissonFFTSolver/DistributedFFTSolver
/// (poisson_fft_solver.hpp). 2D FFT = local FFT-x -> parallel transpose (MPI_Alltoall) -> local FFT-y
/// -> division by the Laplacian eigenvalue -> inverse.
/// Contract: each rank owns Ny/np rows (full x); solve(rho_local, phi_local) with rho_local and
/// phi_local of size nyl_ x Nx_ (row-major). The ctor takes spectral (false = DISCRETE 5-point
/// stencil eigenvalue, bit-identical default; true = CONTINUOUS symbol -(kx^2+ky^2), signed
/// frequencies).
///
/// Invariants:
/// - the slab transpose requires Nx and Ny divisible by np (n_ranks);
/// - power-of-two lengths use the radix-2 FFT; any other size falls back to the correct
///   O(n^2) direct DFT (the radix-2 butterfly would overflow the buffer on an arbitrary n), so
///   single-rank accepts any n;
/// - mode kx=ky=0: zero eigenvalue -> phi_hat=0 (zero mean enforced);
/// - alltoall is the identity when np==1 (and without MPI).

#include <adc/parallel/comm.hpp>

#include <cmath>
#include <complex>
#include <numbers>  // std::numbers::pi (M_PI is not standard, absent under MSVC)
#include <utility>
#include <vector>

#ifdef ADC_HAS_MPI
#include <mpi.h>
#endif

namespace adc {

using cplx = std::complex<double>;

inline bool is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

// Direct O(n^2) DFT, CORRECTNESS fallback for lengths that are NOT powers of two
// (the radix-2 below assumes n = 2^k: on an arbitrary n its butterfly overflows
// the buffer, hence a corrupted, non-deterministic result). Same conventions as
// fft1d (inv=false: exp(-i...), inv=true: exp(+i...) with 1/n), so the spectral
// solver stays correct for an arbitrary Nx or Ny, at the cost of a quadratic
// runtime. On a power-of-two grid (the intended case) we keep the fast FFT.
inline void dft1d_direct(cplx* a, int n, bool inv) {
  std::vector<cplx> out(static_cast<std::size_t>(n));
  const double s = inv ? 1.0 : -1.0;
  for (int k = 0; k < n; ++k) {
    cplx acc(0.0, 0.0);
    for (int j = 0; j < n; ++j) {
      const double ang = s * 2.0 * std::numbers::pi * (static_cast<double>(k) * j / n);
      acc += a[j] * cplx(std::cos(ang), std::sin(ang));
    }
    out[static_cast<std::size_t>(k)] = inv ? acc / static_cast<double>(n) : acc;
  }
  for (int i = 0; i < n; ++i) a[i] = out[static_cast<std::size_t>(i)];
}

// In-place 1D radix-2 FFT (power-of-two length). inv=false: exp(-i...),
// inv=true: exp(+i...) with 1/n normalization. Falls back to the direct DFT when n
// is not a power of two (otherwise the radix-2 butterfly overflows the buffer).
inline void fft1d(cplx* a, int n, bool inv) {
  if (!is_pow2(n)) {
    dft1d_direct(a, n, inv);
    return;
  }
  for (int i = 1, j = 0; i < n; ++i) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }
  for (int len = 2; len <= n; len <<= 1) {
    const double ang = 2.0 * std::numbers::pi / len * (inv ? 1.0 : -1.0);
    const cplx wl(std::cos(ang), std::sin(ang));
    for (int i = 0; i < n; i += len) {
      cplx w(1.0, 0.0);
      for (int k = 0; k < len / 2; ++k) {
        const cplx u = a[i + k], v = a[i + k + len / 2] * w;
        a[i + k] = u + v;
        a[i + k + len / 2] = u - v;
        w *= wl;
      }
    }
  }
  if (inv)
    for (int i = 0; i < n; ++i) a[i] /= n;
}

class PoissonFFT {
 public:
  /// @p spectral: false (default) = DISCRETE 5-point stencil eigenvalue (consistent with the
  /// transport gradients, bit-identical to the historical behavior). true = CONTINUOUS symbol
  /// lambda(k) = -(kx^2 + ky^2) with signed frequencies k in [0..N/2-1, -N/2..-1] * (2pi/L) --
  /// exactly the convention of reference spectral solvers (e.g. poisson_fft.m of RIEMOM2D);
  /// the solution is then the SPECTRAL Poisson (exact on sinusoids), which differs from the
  /// discrete stencil by O(h^2).
  PoissonFFT(int Nx, int Ny, double Lx, double Ly, bool spectral = false)
      : Nx_(Nx),
        Ny_(Ny),
        np_(n_ranks()),
        rank_(my_rank()),
        nyl_(Ny / np_),
        nxl_(Nx / np_),
        dx_(Lx / Nx),
        dy_(Ly / Ny),
        spectral_(spectral) {}

  int ny_local() const { return nyl_; }  // rows (y) owned by this rank
  int nx() const { return Nx_; }
  int y_begin() const { return rank_ * nyl_; }  // first global row of the rank

  // rho_local and phi_local: nyl_ x Nx_ (row-major, global rows
  // [y_begin, y_begin+nyl_)). phi_local is (re)sized.
  void solve(const std::vector<double>& rho_local,
             std::vector<double>& phi_local) {
    std::vector<cplx> A(static_cast<std::size_t>(nyl_) * Nx_);
    for (std::size_t t = 0; t < A.size(); ++t) A[t] = cplx(rho_local[t], 0.0);
    for (int jl = 0; jl < nyl_; ++jl)
      fft1d(&A[static_cast<std::size_t>(jl) * Nx_], Nx_, false);  // FFT-x

    std::vector<cplx> B(static_cast<std::size_t>(nxl_) * Ny_);
    transpose_fwd(A, B);                                          // -> (nxl x Ny)
    for (int il = 0; il < nxl_; ++il)
      fft1d(&B[static_cast<std::size_t>(il) * Ny_], Ny_, false);  // FFT-y

    for (int il = 0; il < nxl_; ++il) {
      const int kx = rank_ * nxl_ + il;
      const int kxs = (kx < (Nx_ + 1) / 2) ? kx : kx - Nx_;  // signed frequency (Nyquist -> -N/2)
      const double wx = 2.0 * std::numbers::pi * kxs / (Nx_ * dx_);
      const double lx = spectral_ ? -(wx * wx)
                                  : (2.0 * std::cos(2.0 * std::numbers::pi * kx / Nx_) - 2.0) / (dx_ * dx_);
      for (int ky = 0; ky < Ny_; ++ky) {
        const int kys = (ky < (Ny_ + 1) / 2) ? ky : ky - Ny_;
        const double wy = 2.0 * std::numbers::pi * kys / (Ny_ * dy_);
        const double ly = spectral_ ? -(wy * wy)
                                    : (2.0 * std::cos(2.0 * std::numbers::pi * ky / Ny_) - 2.0) / (dy_ * dy_);
        const double lam = lx + ly;
        cplx& v = B[static_cast<std::size_t>(il) * Ny_ + ky];
        v = (std::abs(lam) < 1e-14) ? cplx(0.0, 0.0) : v / lam;
      }
    }

    for (int il = 0; il < nxl_; ++il)
      fft1d(&B[static_cast<std::size_t>(il) * Ny_], Ny_, true);  // IFFT-y
    transpose_bwd(B, A);                                         // -> (nyl x Nx)
    for (int jl = 0; jl < nyl_; ++jl)
      fft1d(&A[static_cast<std::size_t>(jl) * Nx_], Nx_, true);  // IFFT-x

    phi_local.resize(A.size());
    for (std::size_t t = 0; t < A.size(); ++t) phi_local[t] = A[t].real();
  }

 private:
  // transpose A(nyl x Nx) -> B(nxl x Ny): alltoall of (nyl x nxl) blocks.
  void transpose_fwd(const std::vector<cplx>& A, std::vector<cplx>& B) {
    const int blk = nyl_ * nxl_;
    std::vector<cplx> send(static_cast<std::size_t>(np_) * blk), recv(send.size());
    for (int s = 0; s < np_; ++s)
      for (int jl = 0; jl < nyl_; ++jl)
        for (int il = 0; il < nxl_; ++il)
          send[static_cast<std::size_t>(s) * blk + static_cast<std::size_t>(jl) * nxl_ + il] =
              A[static_cast<std::size_t>(jl) * Nx_ + static_cast<std::size_t>(s) * nxl_ + il];
    alltoall(send, recv, blk);
    for (int q = 0; q < np_; ++q)
      for (int jl = 0; jl < nyl_; ++jl)
        for (int il = 0; il < nxl_; ++il)
          B[static_cast<std::size_t>(il) * Ny_ + static_cast<std::size_t>(q) * nyl_ + jl] =
              recv[static_cast<std::size_t>(q) * blk + static_cast<std::size_t>(jl) * nxl_ + il];
  }

  // transpose B(nxl x Ny) -> A(nyl x Nx): alltoall of (nxl x nyl) blocks.
  void transpose_bwd(const std::vector<cplx>& B, std::vector<cplx>& A) {
    const int blk = nxl_ * nyl_;
    std::vector<cplx> send(static_cast<std::size_t>(np_) * blk), recv(send.size());
    for (int s = 0; s < np_; ++s)
      for (int il = 0; il < nxl_; ++il)
        for (int jl = 0; jl < nyl_; ++jl)
          send[static_cast<std::size_t>(s) * blk + static_cast<std::size_t>(il) * nyl_ + jl] =
              B[static_cast<std::size_t>(il) * Ny_ + static_cast<std::size_t>(s) * nyl_ + jl];
    alltoall(send, recv, blk);
    for (int q = 0; q < np_; ++q)
      for (int il = 0; il < nxl_; ++il)
        for (int jl = 0; jl < nyl_; ++jl)
          A[static_cast<std::size_t>(jl) * Nx_ + static_cast<std::size_t>(q) * nxl_ + il] =
              recv[static_cast<std::size_t>(q) * blk + static_cast<std::size_t>(il) * nyl_ + jl];
  }

  // all-to-all exchange of `blk` complex values per rank (identity if np==1).
  void alltoall(const std::vector<cplx>& send, std::vector<cplx>& recv, int blk) {
    if (np_ == 1) {
      recv = send;
      return;
    }
#ifdef ADC_HAS_MPI
    MPI_Alltoall(send.data(), 2 * blk, MPI_DOUBLE, recv.data(), 2 * blk,
                 MPI_DOUBLE, MPI_COMM_WORLD);
#else
    recv = send;
#endif
  }

  int Nx_, Ny_, np_, rank_, nyl_, nxl_;
  double dx_, dy_;
  bool spectral_ = false;
};

}  // namespace adc
