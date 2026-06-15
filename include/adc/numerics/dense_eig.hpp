#pragma once
/// @file
/// @brief Spectrum extremes (real parts) of a small dense matrix: signed wave-speed bounds
/// supplied by a model (exact HLL via flux Jacobian).
///
/// GENERIC utility: the core knows nothing of the calling model -- it receives a dense block
/// Real[N][N] and returns min/max of the real parts of its spectrum. Intended consumer: the DSL
/// codegen wave_speeds_from_jacobian (the model supplies dF/dU, possibly by diagonal blocks;
/// each block passes through here). Any other "small eigenvalues on the stack" use is legitimate.
///
/// Algorithm (eigenvalues ONLY, never the vectors):
///   N == 1, 2: closed form (trace/determinant);
///   N >= 3: Hessenberg reduction (Householder, without accumulation) then QR iteration with
///               implicit Francis DOUBLE SHIFT and deflation (EISPACK/hqr formulation) --
///               complex conjugate pairs stay in real arithmetic (2x2 blocks),
///               exceptional shifts after 10 and 20 iterations on a single block.
///
/// ROBUSTNESS CONTRACT: if a block does not converge under the iteration cap, the result is the
/// Gershgorin FALLBACK over the WHOLE matrix (converged = false): an EXTERNAL bound always valid
/// for all real parts (sL <= all speeds <= sR, hence a stable HLL flux, just more diffusive). If
/// the fallback triggers, lmin/lmax are NOT the eigenvalues and max_im is 0 by CONVENTION (nothing
/// was computed, this is NOT a real-spectrum signal): any bit-match against an eig reference is then
/// void. converged (or the fallback output parameter) decides: never read lmin/lmax/max_im without
/// consulting it. The default cap (100) is sized so that the usual near-degenerate companion blocks
/// converge; the fallback remains the safety net for out-of-envelope cases.
///
/// HYPERBOLICITY: max_im returns the largest |Im(lambda)| encountered. A hyperbolic system has a
/// real spectrum (max_im ~ 0); a model that loses hyperbolicity does not silently receive a
/// plausible-but-wrong speed -- the caller decides (assert, warning, clamp).
///
/// PRECISION: simple, separated eigenvalues -> machine precision (tested at rtol 1e-10).
/// CLUSTERED eigenvalues of a non-symmetric matrix: conditioning ~ eps^(1/m) for a near-multiplicity
/// m (a limit of the problem, not of the algorithm) -- do not expect 1e-10 at a triple point. No
/// exactness claim is made under the iteration cap: the cap bounds the COST, the fallback bounds the
/// RESULT.
///
/// Device: ADC_HD, stack buffers (O(N^2)), zero allocation, zero std:: container, bounded loops,
/// no recursion -- only std::sqrt / std::fabs (resolved on device under nvcc/Kokkos, as on the flux
/// path).

#include <cmath>
#include <limits>

#include <adc/core/types.hpp>

namespace adc {

/// Result of real_eig_minmax: real-part extremes + diagnostic. The consumer (DSL codegen
/// wave_speeds_from_jacobian, ADC-87) receives the WHOLE structure: converged and max_im are part
/// of the safety contract, no overload silently drops them.
struct EigBounds {
  Real lmin;      ///< smallest real part (or Gershgorin lower bound if !converged)
  Real lmax;      ///< largest real part (or Gershgorin upper bound if !converged)
  Real max_im;    ///< largest |Im(lambda)| encountered (0 = real spectrum: hyperbolic). Has this
                  ///< meaning ONLY if converged: under fallback it is 0 by CONVENTION (the spectrum
                  ///< is not computed), certainly not a hyperbolicity signal -- read converged first.
  bool converged; ///< false -> Gershgorin fallback (valid external bound, NOT the spectrum)
};

namespace detail {

/// Gershgorin bound on the REAL PARTS: every lambda of the spectrum satisfies
/// lo <= Re(lambda) <= hi (disks centered at a_ii of radius the sum of the |off-diagonal| terms of
/// the row). Safe external bound for HLL, attained only if the matrix is diagonal.
template <int N>
ADC_HD inline void gershgorin_bounds(const Real (&A)[N][N], Real& lo, Real& hi) {
  for (int i = 0; i < N; ++i) {
    Real r = Real(0);
    for (int j = 0; j < N; ++j)
      if (j != i) r += std::fabs(A[i][j]);
    const Real l = A[i][i] - r, h = A[i][i] + r;
    if (i == 0 || l < lo) lo = l;
    if (i == 0 || h > hi) hi = h;
  }
}

/// Upper Hessenberg reduction by Householder reflections, IN PLACE, without accumulating the
/// transformations (eigenvalues only). Unconditionally stable.
template <int N>
ADC_HD inline void hessenberg_reduce(Real (&H)[N][N]) {
  Real v[N];  // Householder vector of the current step (components k..N-1)
  for (int k = 1; k <= N - 2; ++k) {
    Real scale = Real(0);
    for (int i = k; i < N; ++i) scale += std::fabs(H[i][k - 1]);
    if (scale == Real(0)) continue;  // column already zero below the subdiagonal
    Real h = Real(0);
    for (int i = k; i < N; ++i) {
      v[i] = H[i][k - 1] / scale;
      h += v[i] * v[i];
    }
    Real g = std::sqrt(h);
    if (v[k] > Real(0)) g = -g;
    h -= v[k] * g;       // h = v.v / 2 after updating v[k]
    v[k] -= g;
    if (h == Real(0)) continue;
    // P = I - v v^T / h; H <- P H P (column k-1 set explicitly, exact zeros below)
    for (int j = k; j < N; ++j) {  // P * H on rows k..N-1
      Real f = Real(0);
      for (int i = k; i < N; ++i) f += v[i] * H[i][j];
      f /= h;
      for (int i = k; i < N; ++i) H[i][j] -= f * v[i];
    }
    for (int i = 0; i < N; ++i) {  // H * P on columns k..N-1
      Real f = Real(0);
      for (int j = k; j < N; ++j) f += H[i][j] * v[j];
      f /= h;
      for (int j = k; j < N; ++j) H[i][j] -= f * v[j];
    }
    H[k][k - 1] = scale * g;
    for (int i = k + 1; i < N; ++i) H[i][k - 1] = Real(0);
  }
}

ADC_HD inline Real hqr_copysign(Real mag, Real sgn) {
  return sgn >= Real(0) ? std::fabs(mag) : -std::fabs(mag);
}

/// Accumulate an eigenvalue (re, im) into the current extremes. A named function rather than a
/// local lambda: device caution (nvcc and lambdas inside __host__ __device__ code).
ADC_HD inline void record_eig(Real re, Real im, Real& lmin, Real& lmax, Real& max_im,
                              bool& first) {
  if (first || re < lmin) lmin = re;
  if (first || re > lmax) lmax = re;
  const Real ai = std::fabs(im);
  if (first || ai > max_im) max_im = ai;
  first = false;
}

/// QR iteration with implicit Francis double shift on a Hessenberg matrix (EISPACK/hqr
/// formulation, eigenvalues only, blocks processed bottom-up with deflation). Accumulates min/max
/// of the real parts and max|Im| directly. @return true if the WHOLE spectrum is extracted under
/// the cap (@p max_iter_per_eig iterations per active block), false otherwise.
template <int N>
ADC_HD inline bool hqr_minmax(Real (&H)[N][N], Real& lmin, Real& lmax, Real& max_im,
                              int max_iter_per_eig) {
  constexpr Real kEps = std::numeric_limits<Real>::epsilon();  // follows the Real type
  Real anorm = Real(0);  // norm of the Hessenberg part (deflation criterion for the s == 0 cases)
  for (int i = 0; i < N; ++i)
    for (int j = (i > 0 ? i - 1 : 0); j < N; ++j) anorm += std::fabs(H[i][j]);
  if (anorm == Real(0)) {  // null matrix: spectrum {0}
    lmin = lmax = max_im = Real(0);
    return true;
  }

  bool first = true;

  int nn = N - 1;     // top index of the active block
  Real t = Real(0);   // cumulative shift (exceptional shifts)
  Real p = Real(0), q = Real(0), r = Real(0), x, y, z, w, s;
  while (nn >= 0) {
    int its = 0;
    int l;
    do {
      // deflation: smallest l such that H[l][l-1] is negligible
      for (l = nn; l >= 1; --l) {
        s = std::fabs(H[l - 1][l - 1]) + std::fabs(H[l][l]);
        if (s == Real(0)) s = anorm;
        if (std::fabs(H[l][l - 1]) <= kEps * s) {
          H[l][l - 1] = Real(0);
          break;
        }
      }
      x = H[nn][nn];
      if (l == nn) {  // real 1x1 eigenvalue
        record_eig(x + t, Real(0), lmin, lmax, max_im, first);
        --nn;
      } else {
        y = H[nn - 1][nn - 1];
        w = H[nn][nn - 1] * H[nn - 1][nn];
        if (l == nn - 1) {  // 2x2 block: real pair or complex conjugate pair
          p = Real(0.5) * (y - x);
          q = p * p + w;
          z = std::sqrt(std::fabs(q));
          x += t;
          if (q >= Real(0)) {  // two real values
            z = p + hqr_copysign(z, p);
            record_eig(x + z, Real(0), lmin, lmax, max_im, first);
            record_eig(z != Real(0) ? x - w / z : x + z, Real(0), lmin, lmax, max_im, first);
          } else {             // complex pair: Re = x + p, |Im| = z
            record_eig(x + p, z, lmin, lmax, max_im, first);
            record_eig(x + p, -z, lmin, lmax, max_im, first);
          }
          nn -= 2;
        } else {  // block > 2: one Francis double-shift iteration
          if (its == max_iter_per_eig) return false;  // cap reached -> caller fallback
          if (its == 10 || its == 20) {  // exceptional shift (slow cycles)
            t += x;
            for (int i = 0; i <= nn; ++i) H[i][i] -= x;
            s = std::fabs(H[nn][nn - 1]) + std::fabs(H[nn - 1][nn - 2]);
            y = x = Real(0.75) * s;
            w = Real(-0.4375) * s * s;
          }
          ++its;
          int m;
          for (m = nn - 2; m >= l; --m) {  // two consecutive small subdiagonals
            z = H[m][m];
            r = x - z;
            s = y - z;
            p = (r * s - w) / H[m + 1][m] + H[m][m + 1];
            q = H[m + 1][m + 1] - z - r - s;
            r = H[m + 2][m + 1];
            s = std::fabs(p) + std::fabs(q) + std::fabs(r);
            p /= s;
            q /= s;
            r /= s;
            if (m == l) break;
            const Real u = std::fabs(H[m][m - 1]) * (std::fabs(q) + std::fabs(r));
            const Real v = std::fabs(p) * (std::fabs(H[m - 1][m - 1]) + std::fabs(z)
                                           + std::fabs(H[m + 1][m + 1]));
            if (u <= kEps * v) break;
          }
          for (int i = m + 2; i <= nn; ++i) {
            H[i][i - 2] = Real(0);
            if (i > m + 2) H[i][i - 3] = Real(0);
          }
          for (int k = m; k <= nn - 1; ++k) {  // QR double-shift sweep on columns m..nn-1
            if (k != m) {
              p = H[k][k - 1];
              q = H[k + 1][k - 1];
              r = (k != nn - 1) ? H[k + 2][k - 1] : Real(0);
              x = std::fabs(p) + std::fabs(q) + std::fabs(r);
              if (x != Real(0)) {
                p /= x;
                q /= x;
                r /= x;
              }
            }
            s = hqr_copysign(std::sqrt(p * p + q * q + r * r), p);
            if (s == Real(0)) continue;
            if (k == m) {
              if (l != m) H[k][k - 1] = -H[k][k - 1];
            } else {
              H[k][k - 1] = -s * x;
            }
            p += s;
            x = p / s;
            y = q / s;
            z = r / s;
            q /= p;
            r /= p;
            for (int j = k; j <= nn; ++j) {  // transform rows k..k+2
              p = H[k][j] + q * H[k + 1][j];
              if (k != nn - 1) {
                p += r * H[k + 2][j];
                H[k + 2][j] -= p * z;
              }
              H[k + 1][j] -= p * y;
              H[k][j] -= p * x;
            }
            const int mmin = (nn < k + 3) ? nn : k + 3;
            for (int i = l; i <= mmin; ++i) {  // transform columns k..k+2
              p = x * H[i][k] + y * H[i][k + 1];
              if (k != nn - 1) {
                p += z * H[i][k + 2];
                H[i][k + 2] -= p * r;
              }
              H[i][k + 1] -= p * q;
              H[i][k] -= p;
            }
          }
        }
      }
    } while (l < nn - 1);
  }
  return true;
}

}  // namespace detail

/// Extremes of the REAL PARTS of the spectrum of a small dense block @p A, plus the largest |Im|
/// encountered and a convergence indicator (see the file header for the full contract: Gershgorin
/// fallback on non-convergence, max_im as a hyperbolicity-loss detector).
/// @p max_iter_per_eig: QR iteration cap per active block (default 100). The historical EISPACK
/// heuristic (30) does not suffice on near-degenerate companion blocks (near-double eigenvalues)
/// where deflation crawls: such a 5x5 block needs ~42 iterations, and below 30 it silently fell
/// back (wave speed over-estimated ~9x). 100 leaves more than double the margin; the overhead is
/// paid ONLY by pathological blocks (healthy cases converge in a few iterations). 0 forces the
/// fallback AS SOON AS an active block >= 3 exists (useful for testing the caller's contract); a
/// matrix that deflates entirely into 1x1 / 2x2 blocks (quasi-triangular) never iterates and
/// converges even at cap 0.
/// @p fallback: if non-null, receives true when the Gershgorin fallback triggered (spectrum NOT
/// computed), false otherwise. Default nullptr -> behavior unchanged for any existing caller;
/// mirror of !EigBounds::converged, for whoever wants only the flag (e.g. OR over several blocks).
template <int N>
ADC_HD inline EigBounds real_eig_minmax(const Real (&A)[N][N], int max_iter_per_eig = 100,
                                        bool* fallback = nullptr) {
  static_assert(N >= 1, "real_eig_minmax: N >= 1");
  static_assert(N <= 16, "real_eig_minmax: block limited to 16x16 (stack buffer O(N^2) per device "
                         "thread, ~2 KB; beyond that, a dense solver with allocation is more "
                         "appropriate than this path)");
  EigBounds b{Real(0), Real(0), Real(0), true};
  if constexpr (N == 1) {
    b.lmin = b.lmax = A[0][0];
  } else if constexpr (N == 2) {  // closed form: trace / determinant
    const Real tr2 = Real(0.5) * (A[0][0] + A[1][1]);
    const Real disc = Real(0.25) * (A[0][0] - A[1][1]) * (A[0][0] - A[1][1]) + A[0][1] * A[1][0];
    if (disc >= Real(0)) {
      const Real z = std::sqrt(disc);
      b.lmin = tr2 - z;
      b.lmax = tr2 + z;
    } else {
      b.lmin = b.lmax = tr2;
      b.max_im = std::sqrt(-disc);
    }
  } else {
    Real H[N][N];  // working copy (A is not modified)
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j) H[i][j] = A[i][j];
    detail::hessenberg_reduce(H);
    if (!detail::hqr_minmax(H, b.lmin, b.lmax, b.max_im, max_iter_per_eig)) {
      // non-convergence: external Gershgorin bound on the ORIGINAL matrix (the working copy is in
      // an intermediate state) -- safe bound, not the spectrum. max_im forced to 0 by CONVENTION
      // (nothing was computed), never to be interpreted as a real spectrum.
      detail::gershgorin_bounds(A, b.lmin, b.lmax);
      b.max_im = Real(0);
      b.converged = false;
    }
  }
  if (fallback) *fallback = !b.converged;
  return b;
}

}  // namespace adc
