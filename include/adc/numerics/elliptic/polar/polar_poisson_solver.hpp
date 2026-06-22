#pragma once

#include <adc/core/foundation/types.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/geometry.hpp>  // PolarGeometry
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/elliptic/poisson/poisson_fft.hpp>  // fft1d (1D DFT/FFT reused verbatim)
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <complex>
#include <concepts>
#include <stdexcept>
#include <vector>

/// @file
/// @brief Direct POLAR Poisson solver on an annulus (PolarGeometry), Phase 2a.
///
/// Solves the polar Poisson equation on the ANNULAR mesh (r, theta) of the
/// annular polar grid (Phase 1, PolarGeometry):
///   (1/r) d_r(r d_r phi) + (1/r^2) d_theta^2 phi = f
/// with theta PERIODIC (the annulus covers [0, 2pi)) and a PHYSICAL BC at r_min / r_max
/// (Dirichlet or homogeneous Neumann). The annulus EXCLUDES r = 0 (r_min > 0): NO coordinate
/// singularity, unlike the full disk.
///
/// METHOD (FFT-in-theta + tridiagonal-in-r, robust, direct -- NO multigrid).
/// Scoping reported that the MG V-cycle can STAGNATE on the 1/r^2 operator in polar. The
/// classic robust method: since theta is periodic with constant coefficient, an FFT in theta
/// DECOUPLES the azimuthal modes m (the 1/r^2 d_theta^2 stencil diagonalizes under the DFT). For
/// each mode m we then solve a 1D radial ODE
///   (1/r) d_r(r d_r phi_m) - (lambda_theta(m)/r^2) phi_m = f_m
/// by a tridiagonal solve (Thomas) in r. This is EXACT per mode, robust (no iterative
/// convergence issue) and cheap (O(ntheta log ntheta) in theta + O(nr) per mode).
///
/// The 1D FFT (fft1d / dft1d_direct) is REUSED verbatim from poisson_fft.hpp (radix-2 on
/// powers of 2, fallback to direct O(n^2) DFT otherwise -> any ntheta accepted).
///
/// SPECTRAL AZIMUTHAL EIGENVALUE: the Fourier basis diagonalizes d_theta^2 EXACTLY. The mode
/// of DFT index m (m in [0, ntheta)) corresponds to the signed wavenumber k(m) = m if m <= ntheta/2,
/// otherwise m - ntheta (aliasing); d_theta^2 e^{i k theta} = -k^2 e^{i k theta}, so the eigenvalue
/// is -k(m)^2 (and NOT the 2-point stencil (2cos-2)/dtheta^2, which is only an O(dtheta^2) approximation
/// of -k^2). Using -k(m)^2 makes the theta direction SPECTRAL (exact for band-limited data,
/// i.e. azimuthal content carried by a small number of modes). The azimuthal term at cell
/// (i, m) is therefore (-k(m)^2 / r_i^2) phi_hat(i, m), DIAGONAL in m (1/r_i^2 local to row i).
///
/// RADIAL DISCRETIZATION (conservative finite volumes, order 2, like assemble_rhs_polar):
///   (1/r_i) [ r_{i+1/2}(phi_{i+1}-phi_i)/dr - r_{i-1/2}(phi_i - phi_{i-1})/dr ] / dr
/// that is, per mode m, the tridiagonal system (in phi_hat(., m)):
///   sub-diag   a_i = (1/r_i) r_{i-1/2} / dr^2
///   super-diag c_i = (1/r_i) r_{i+1/2} / dr^2
///   diag       b_i = -(a_i + c_i) - k(m)^2 / r_i^2
/// r_i = r_cell(i), r_{i+/-1/2} = r_face(i+1)/r_face(i) (all > 0 on the annulus).
///
/// BOUNDARY CONDITIONS IN r (Dirichlet or homogeneous Neumann, via BCRec.xlo / .xhi):
///   - Dirichlet (value v at the face): reflection ghost phi_{-1} = 2 v - phi_0. The coefficient
///     a_0 then folds into the diagonal (b_0 -= a_0) and 2 a_0 v moves to the right-hand side. Same at
///     r_max with c_{nr-1}, v = xhi_val.
///   - Homogeneous Neumann (Foextrap, zero gradient at the face): ghost phi_{-1} = phi_0 -> a_0 folds into
///     b_0 += a_0 (the flux at the wall is zero). Same c_{nr-1} at r_max.
/// theta stays PERIODIC (handled by the FFT, no azimuthal ghost).
///
/// MODE m = 0 + TWO Neumann boundaries: the pure radial operator has a kernel (the constant), so the
/// tridiagonal is SINGULAR. We fix the gauge by pinning phi_hat(0, 0) = 0 (phi of zero radial
/// mean for the constant mode), as the FFT solver pins the k = 0 mode. With at least
/// one Dirichlet boundary the operator is invertible for every m (no gauge to fix).
///
/// EllipticSolver CONTRACT (rhs()/phi()/solve()/residual()/geom()): MODEL adapted to polar. geom()
/// returns the PolarGeometry (not Geometry) -- this solver is NOT yet wired into System.step
/// (Phase 2b); it only composes as a standalone polar elliptic solver. ADDITIVE: no
/// Cartesian path (geometric_mg, PoissonFFTSolver) is touched.
///
/// SCOPE: single-rank, single box covering the annulus (like PoissonFFTSolver). The FFT-in-theta +
/// tridiag-in-r requires the full theta row AND the full radial column on one rank; the
/// distributed case would require a parallel transpose (out of scope for Phase 2a). HARD guard (active in
/// Release) if n_ranks() > 1 or ba.size() != 1, thrown on ALL ranks (no deadlock);
/// solve()/residual() are local_size()==0-safe.

namespace adc {

// Contract of POLAR elliptic solvers: same shape as EllipticSolver (elliptic_solver.hpp)
// but geom() returns a PolarGeometry (annulus (r, theta)) instead of a Cartesian Geometry. Polar
// counterpart of the Cartesian concept: a future polar coupler (Phase 2b) would depend on this CONCEPT, not
// on PolarPoissonSolver directly, exactly as Coupler depends on EllipticSolver. Non-intrusive
// sibling: no change to elliptic_solver.hpp nor to the Cartesian path.
template <class S>
concept PolarEllipticSolver = requires(S s) {
  { s.rhs() } -> std::same_as<MultiFab&>;
  { s.phi() } -> std::same_as<MultiFab&>;
  s.solve();
  { s.residual() } -> std::convertible_to<Real>;
  { s.geom() } -> std::convertible_to<const PolarGeometry&>;
};

class PolarPoissonSolver {
 public:
  /// @param geom annulus (r, theta), domain.nx() = nr radial cells, ny() = ntheta azimuthal.
  /// @param ba   BoxArray with ONE box covering the whole annulus (single-rank).
  /// @param bc   radial BC: xlo/xhi (Dirichlet -> xlo_val/xhi_val; Foextrap -> homogeneous Neumann).
  ///             theta (ylo/yhi) is treated as PERIODIC regardless of the value (FFT).
  PolarPoissonSolver(const PolarGeometry& geom, const BoxArray& ba, const BCRec& bc = BCRec{})
      : geom_(geom), bc_(bc), dm_(ba.size(), n_ranks()), phi_(ba, dm_, 1, 0), rhs_(ba, dm_, 1, 0) {
    if (n_ranks() != 1)
      throw std::runtime_error(
          "PolarPoissonSolver: unsupported under MPI (n_ranks>1); FFT-in-theta + tridiag-in-r "
          "requires the full theta row + radial column on one rank (parallel transpose = out of "
          "scope for Phase 2a)");
    if (ba.size() != 1)
      throw std::runtime_error(
          "PolarPoissonSolver: single box required (ba.size()==1) covering the whole annulus");
  }

  MultiFab& rhs() { return rhs_; }
  MultiFab& phi() { return phi_; }
  const PolarGeometry& geom() const { return geom_; }

  /// Solves (1/r) d_r(r d_r phi) + (1/r^2) d_theta^2 phi = rhs in place, by FFT-in-theta then a
  /// tridiagonal solve (Thomas) per azimuthal mode. phi() holds the solution after the call.
  void solve() {
    if (phi_.local_size() == 0)
      return;  // rank with no local box (MPI): nothing to do (cf. guard)
    // HOST/DEVICE COHERENCE: solve() is a HOST algorithm (std::vector / std::complex / fft1d /
    // Thomas) that reads the RHS via host pointers (f = rhs_.fab(0).const_array() below). The RHS
    // may have been filled by a device kernel (for_each_cell) possibly still in flight. We make the
    // host residency of the RHS valid BEFORE any host read, exactly like the other host readers
    // of the repo (cf. the sync_host()/device_fence() seam of for_each.hpp / MultiFab). Under Kokkos
    // Cuda = a targeted device_fence() (otherwise: stale data -> cudaErrorIllegalInstruction). Under
    // Kokkos Serial/OpenMP (host execution, unified memory) = a fence with no observable effect (no device
    // kernel in flight to drain).
    rhs_.sync_host();
    const int nr = geom_.domain.nx();
    const int nth = geom_.domain.ny();
    const Real dr = geom_.dr();   // theta is SPECTRAL (eigenvalue -k^2): dtheta does not appear
    const Box2D v = rhs_.box(0);  // valid cells: i in [v.lo[0]..], j in [v.lo[1]..]
    const ConstArray4 f = rhs_.fab(0).const_array();
    Array4 p = phi_.fab(0).array();

    // --- 1) FFT in theta, radial row by radial row: f(i, .) -> f_hat(i, m) ---
    // We store per radial row i a vector of ntheta complex values (the azimuthal spectrum of the row).
    std::vector<std::vector<cplx>> fhat(static_cast<std::size_t>(nr));
    for (int i = 0; i < nr; ++i) {
      std::vector<cplx>& row = fhat[static_cast<std::size_t>(i)];
      row.resize(static_cast<std::size_t>(nth));
      for (int j = 0; j < nth; ++j)
        row[static_cast<std::size_t>(j)] = cplx(f(v.lo[0] + i, v.lo[1] + j), 0.0);
      fft1d(row.data(), nth, /*inv=*/false);
    }

    // --- 2) Radial coefficients INDEPENDENT of the mode (pure geometry) ---
    // a_i (sub-diag), c_i (super-diag), and the radial part of the diagonal d_rad_i = -(a_i + c_i).
    std::vector<Real> a(static_cast<std::size_t>(nr)), c(static_cast<std::size_t>(nr)),
        d_rad(static_cast<std::size_t>(nr)), inv_r2(static_cast<std::size_t>(nr));
    for (int i = 0; i < nr; ++i) {
      const Real ri = geom_.r_cell(i);      // r at the center of cell i (> 0)
      const Real rm = geom_.r_face(i);      // r_{i-1/2} (low face)
      const Real rp = geom_.r_face(i + 1);  // r_{i+1/2} (high face)
      const Real inv_r_dr2 = Real(1) / (ri * dr * dr);
      a[static_cast<std::size_t>(i)] = rm * inv_r_dr2;
      c[static_cast<std::size_t>(i)] = rp * inv_r_dr2;
      d_rad[static_cast<std::size_t>(i)] =
          -(a[static_cast<std::size_t>(i)] + c[static_cast<std::size_t>(i)]);
      inv_r2[static_cast<std::size_t>(i)] = Real(1) / (ri * ri);
    }

    // Radial BC: Dirichlet (face) -> reflection ghost = 2 v - interior; Foextrap -> homogeneous Neumann
    // (ghost = interior). We fold the boundary coefficient into the diagonal and, for Dirichlet, we
    // inject 2 a/c * value into the right-hand side (mode m = 0 ONLY: the BC is real and constant
    // in theta, so it only contributes to the mean azimuthal mode).
    const bool dir_lo = bc_.xlo == BCType::Dirichlet;
    const bool dir_hi = bc_.xhi == BCType::Dirichlet;
    const Real vlo = bc_.xlo_val, vhi = bc_.xhi_val;
    const bool neumann_both = !dir_lo && !dir_hi;  // gauge to fix for mode m = 0

    // --- 3) One tridiagonal solve (Thomas) per mode m, on the complex vector phi_hat(., m) ---
    // The system is REAL (a, b, c real); we solve the real and imaginary parts with the SAME
    // matrix (Thomas on complex values: a/b/c real, complex right-hand side). phat[i][m] receives the
    // radial spectrum of mode m. We loop m on the outside (matrix fixed for this m), i on the inside.
    std::vector<std::vector<cplx>> phat(static_cast<std::size_t>(nr));
    for (int i = 0; i < nr; ++i)
      phat[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(nth));

    std::vector<cplx> rhs_m(static_cast<std::size_t>(nr)), sol_m(static_cast<std::size_t>(nr));
    std::vector<Real> bdiag(static_cast<std::size_t>(nr));
    for (int m = 0; m < nth; ++m) {
      // SIGNED wavenumber (DFT aliasing): k = m for m <= nth/2, otherwise m - nth. SPECTRAL
      // eigenvalue of d_theta^2 = -k^2 (exact, not the 2-point stencil). dtheta does not appear here.
      const int kw = (m <= nth / 2) ? m : m - nth;
      const Real lam_th = -static_cast<Real>(kw) * static_cast<Real>(kw);
      // Full diagonal of mode m: radial + azimuthal (lam_th/r_i^2 = -k^2/r_i^2).
      for (int i = 0; i < nr; ++i)
        bdiag[static_cast<std::size_t>(i)] =
            d_rad[static_cast<std::size_t>(i)] + lam_th * inv_r2[static_cast<std::size_t>(i)];
      for (int i = 0; i < nr; ++i)
        rhs_m[static_cast<std::size_t>(i)] =
            fhat[static_cast<std::size_t>(i)][static_cast<std::size_t>(m)];

      // Fold the radial BCs into the diagonal / right-hand side (of mode m).
      // r_min (i = 0): ghost phi_{-1}. Dirichlet -> phi_{-1} = 2 vlo - phi_0: b_0 -= a_0,
      // rhs_0 -= 2 a_0 vlo (mode 0). Neumann -> phi_{-1} = phi_0: b_0 += a_0.
      if (dir_lo) {
        bdiag[0] -= a[0];
        if (m == 0)
          rhs_m[0] -= cplx(Real(2) * a[0] * vlo * static_cast<Real>(nth), 0.0);
      } else {
        bdiag[0] += a[0];
      }
      // r_max (i = nr-1): ghost phi_{nr}. Dirichlet -> phi_{nr} = 2 vhi - phi_{nr-1}, Neumann -> = phi_{nr-1}.
      const std::size_t last = static_cast<std::size_t>(nr - 1);
      if (dir_hi) {
        bdiag[last] -= c[last];
        if (m == 0)
          rhs_m[last] -= cplx(Real(2) * c[last] * vhi * static_cast<Real>(nth), 0.0);
      } else {
        bdiag[last] += c[last];
      }

      // Mode m = 0 with TWO Neumann boundaries: singular radial operator (kernel = constant). We pin
      // phi_hat(0, 0) = 0 (gauge: zero radial mean of the constant mode) by forcing the first row to
      // the identity (diag 1, super-diag 0, rhs 0). Without this Thomas divides by a null pivot.
      const bool pin0 = neumann_both && m == 0;

      thomas_solve(a, bdiag, c, rhs_m, sol_m, nr, pin0);
      for (int i = 0; i < nr; ++i)
        phat[static_cast<std::size_t>(i)][static_cast<std::size_t>(m)] =
            sol_m[static_cast<std::size_t>(i)];
    }

    // --- 4) Inverse FFT in theta: phi_hat(i, m) -> phi(i, theta) (the real part, the imaginary part ~ round-off) ---
    for (int i = 0; i < nr; ++i) {
      std::vector<cplx>& row = phat[static_cast<std::size_t>(i)];
      fft1d(row.data(), nth, /*inv=*/true);
      for (int j = 0; j < nth; ++j)
        p(v.lo[0] + i, v.lo[1] + j) = row[static_cast<std::size_t>(j)].real();
    }
  }

  /// Discrete residual ||L_h phi - rhs|| (inf norm), reduced over the ranks. L_h is the EXACT operator that
  /// solve() inverts: FFT in theta (mode m, spectral eigenvalue -k(m)^2/r^2) + conservative order-2
  /// radial stencil, folded radial BC (Dirichlet: constant value = m=0 part; Neumann:
  /// homogeneous reflection). We evaluate it MODE BY MODE in Fourier space (where the theta
  /// diagonalization is exact), so the residual reflects ONLY the quality of the solve (and not the spectral
  /// gap vs finite differences of a physical theta stencil). For this DIRECT solver, ~round-off. The m=0
  /// double-Neumann mode is EXCLUDED (its row 0 is pinned = gauge: trivial identity equation).
  Real residual() {
    if (phi_.local_size() == 0)
      return static_cast<Real>(all_reduce_max(0.0));
    // Like solve(): HOST evaluation (FFT + stencil on host pointers). We make the host residency
    // of the RHS valid before reading (device kernel possibly in flight). phi_ is written host-side by
    // solve(), but we also fence for symmetry/robustness if residual() is called independently. Under
    // Kokkos Serial/OpenMP (host execution) = a fence with no observable effect; under Kokkos Cuda = targeted device_fence().
    rhs_.sync_host();
    phi_.sync_host();
    const int nr = geom_.domain.nx();
    const int nth = geom_.domain.ny();
    const Real dr = geom_.dr();
    const Box2D v = rhs_.box(0);
    const ConstArray4 p = phi_.fab(0).const_array();
    const ConstArray4 f = rhs_.fab(0).const_array();

    const bool dir_lo = bc_.xlo == BCType::Dirichlet;
    const bool dir_hi = bc_.xhi == BCType::Dirichlet;
    const bool neumann_both = !dir_lo && !dir_hi;

    // Radial coefficients (same as solve()).
    std::vector<Real> a(static_cast<std::size_t>(nr)), c(static_cast<std::size_t>(nr)),
        d_rad(static_cast<std::size_t>(nr)), inv_r2(static_cast<std::size_t>(nr));
    for (int i = 0; i < nr; ++i) {
      const Real ri = geom_.r_cell(i);
      const Real inv_r_dr2 = Real(1) / (ri * dr * dr);
      a[static_cast<std::size_t>(i)] = geom_.r_face(i) * inv_r_dr2;
      c[static_cast<std::size_t>(i)] = geom_.r_face(i + 1) * inv_r_dr2;
      d_rad[static_cast<std::size_t>(i)] =
          -(a[static_cast<std::size_t>(i)] + c[static_cast<std::size_t>(i)]);
      inv_r2[static_cast<std::size_t>(i)] = Real(1) / (ri * ri);
    }

    // FFT of phi and of f, radial row by radial row.
    std::vector<std::vector<cplx>> ph(static_cast<std::size_t>(nr)),
        fh(static_cast<std::size_t>(nr));
    for (int i = 0; i < nr; ++i) {
      ph[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(nth));
      fh[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(nth));
      for (int j = 0; j < nth; ++j) {
        ph[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
            cplx(p(v.lo[0] + i, v.lo[1] + j), 0.0);
        fh[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
            cplx(f(v.lo[0] + i, v.lo[1] + j), 0.0);
      }
      fft1d(ph[static_cast<std::size_t>(i)].data(), nth, false);
      fft1d(fh[static_cast<std::size_t>(i)].data(), nth, false);
    }

    Real rmax = 0;
    for (int m = 0; m < nth; ++m) {
      const int kw = (m <= nth / 2) ? m : m - nth;
      const Real lam_th = -static_cast<Real>(kw) * static_cast<Real>(kw);
      const bool pin0 =
          neumann_both && m == 0;  // gauge mode: row 0 pinned (excluded from the residual)
      for (int i = 0; i < nr; ++i) {
        if (pin0 && i == 0)
          continue;  // pinned identity row: not a physical equation
        const std::size_t I = static_cast<std::size_t>(i);
        Real bdiag = d_rad[I] + lam_th * inv_r2[I];
        cplx lhs(0.0, 0.0);
        cplx rm = fh[I][static_cast<std::size_t>(m)];
        // Low boundary (i==0): fold a_0 (Dirichlet: b-=a, rhs-=2 a vlo*nth at m=0; Neumann: b+=a).
        if (i == 0) {
          if (dir_lo) {
            bdiag -= a[0];
            if (m == 0)
              rm -= cplx(Real(2) * a[0] * bc_.xlo_val * static_cast<Real>(nth), 0.0);
          } else {
            bdiag += a[0];
          }
        } else {
          lhs += cplx(a[I], 0.0) * ph[static_cast<std::size_t>(i - 1)][static_cast<std::size_t>(m)];
        }
        // High boundary (i==nr-1): fold c.
        if (i == nr - 1) {
          if (dir_hi) {
            bdiag -= c[I];
            if (m == 0)
              rm -= cplx(Real(2) * c[I] * bc_.xhi_val * static_cast<Real>(nth), 0.0);
          } else {
            bdiag += c[I];
          }
        } else {
          lhs += cplx(c[I], 0.0) * ph[static_cast<std::size_t>(i + 1)][static_cast<std::size_t>(m)];
        }
        lhs += cplx(bdiag, 0.0) * ph[I][static_cast<std::size_t>(m)];
        // Residual of this mode (normalized by nth: the unnormalized DFT scales f_hat by nth; we
        // compare at the SAME scale, so the ratio is independent of the convention).
        const cplx r_m = (lhs - rm) / static_cast<double>(nth);
        rmax = std::max(rmax, static_cast<Real>(std::abs(r_m)));
      }
    }
    return static_cast<Real>(all_reduce_max(static_cast<double>(rmax)));
  }

 private:
  // Thomas algorithm (tridiagonal elimination) on matrices with REAL coefficients a/b/c and a
  // COMPLEX right-hand side. a[i] = sub-diag (couples i-1), b[i] = diag, c[i] = super-diag (couples i+1).
  // a[0] and c[n-1] are NOT used (boundaries). @p pin0: if true, pins x[0] = 0 (gauge for mode 0,
  // double Neumann) by replacing the first row with the identity (b_0 = 1, c_0 = 0, rhs_0 = 0). We
  // work on LOCAL copies of the diagonal / super-diagonal / rhs so as not to alter the caller's
  // arrays. No pivoting: the matrix is diagonally dominant (azimuthal term <= 0,
  // folded BCs) -> Thomas stable without permutation.
  void thomas_solve(const std::vector<Real>& a, const std::vector<Real>& b,
                    const std::vector<Real>& c, const std::vector<cplx>& rhs, std::vector<cplx>& x,
                    int n, bool pin0) const {
    const std::size_t N = static_cast<std::size_t>(n);
    bloc_.assign(N, 0.0);             // working diagonal
    cloc_.assign(N, 0.0);             // working super-diagonal
    rloc_.assign(N, cplx(0.0, 0.0));  // working right-hand side
    for (std::size_t i = 0; i < N; ++i) {
      bloc_[i] = b[i];
      cloc_[i] = c[i];
      rloc_[i] = rhs[i];
    }
    if (pin0) {  // row 0 -> identity: x[0] = 0, decoupled from the rest
      bloc_[0] = Real(1);
      cloc_[0] = Real(0);
      rloc_[0] = cplx(0.0, 0.0);
    }
    cgamma_.assign(N, cplx(0.0, 0.0));
    Real beta = bloc_[0];
    x[0] = rloc_[0] / cplx(beta, 0.0);
    for (std::size_t i = 1; i < N; ++i) {
      cgamma_[i] = cplx(cloc_[i - 1] / beta, 0.0);
      beta = bloc_[i] - a[i] * cloc_[i - 1] / beta;
      x[i] = (rloc_[i] - cplx(a[i], 0.0) * x[i - 1]) / cplx(beta, 0.0);
    }
    for (int i = n - 2; i >= 0; --i)
      x[static_cast<std::size_t>(i)] -=
          cgamma_[static_cast<std::size_t>(i + 1)] * x[static_cast<std::size_t>(i + 1)];
  }

  PolarGeometry geom_;
  BCRec bc_;
  DistributionMapping dm_;
  MultiFab phi_, rhs_;
  // Buffers reused by Thomas (avoid per-mode allocations). mutable: thomas_solve is const.
  mutable std::vector<cplx> cgamma_, rloc_;
  mutable std::vector<Real> bloc_, cloc_;
};

static_assert(PolarEllipticSolver<PolarPoissonSolver>,
              "PolarPoissonSolver must model PolarEllipticSolver");

}  // namespace adc
