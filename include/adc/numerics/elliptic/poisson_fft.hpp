#pragma once

#include <adc/parallel/comm.hpp>

#include <cmath>
#include <complex>
#include <numbers>  // std::numbers::pi (M_PI n est pas standard, absent sous MSVC)
#include <utility>
#include <vector>

#ifdef ADC_HAS_MPI
#include <mpi.h>
#endif

// Solveur de Poisson periodique SPECTRAL (FFT), distribue par bandes (slabs).
// Resout EXACTEMENT le Laplacien 5-points periodique  lap_h phi = rho  avec phi
// de moyenne nulle : aucune iteration, aucune tolerance. Pour un domaine
// periodique c'est le solveur propre par excellence (cf. DFTSolver des codes
// spectraux), pas un Poisson grossier ni replique.
//
// Decomposition : chaque rang possede Ny/np lignes (x complet). La 2D FFT se
// fait en FFT-x locale -> transpose parallele (MPI_Alltoall) -> FFT-y locale ->
// division par la valeur propre discrete du Laplacien -> inverse. La transposee
// par bandes impose Nx et Ny divisibles par np. Les puissances de 2 empruntent la
// FFT radix-2 rapide ; toute autre taille (ex. 48) retombe sur une DFT directe
// correcte mais quadratique (cf. dft1d_direct), donc mono-rang accepte tout n.
//
// Valeur propre du stencil 5-points sous la DFT :
//   lambda(kx,ky) = (2cos(2*pi*kx/Nx) - 2)/dx^2 + (2cos(2*pi*ky/Ny) - 2)/dy^2
// (mode kx=ky=0 : lambda=0 -> phi_hat=0, moyenne nulle). Variante spectral=true :
// symbole CONTINU -(kx^2+ky^2) (frequences signees), cf. doc du constructeur. Comme on emploie la
// valeur propre DISCRETE, la solution satisfait le Laplacien 5-points a
// l'arrondi : coherent avec les gradients par differences finies du transport.

namespace adc {

using cplx = std::complex<double>;

inline bool is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

// DFT directe O(n^2), repli de CORRECTION pour les longueurs qui ne sont PAS
// puissance de 2 (le radix-2 ci-dessous suppose n = 2^k : sur n quelconque sa
// butterfly deborde le buffer, d'ou un resultat corrompu et non deterministe).
// Memes conventions que fft1d (inv=false : exp(-i...), inv=true : exp(+i...) avec
// 1/n), donc le solveur spectral reste correct pour un Nx ou Ny arbitraire, au
// prix d'un cout quadratique. Sur grille puissance de 2 (le cas vise) on garde la
// FFT rapide.
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

// FFT 1D radix-2 en place (longueur puissance de 2). inv=false : exp(-i...),
// inv=true : exp(+i...) avec normalisation 1/n. Repli sur la DFT directe quand n
// n'est pas une puissance de 2 (sinon la butterfly radix-2 deborde le buffer).
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
  /// @p spectral : false (defaut) = valeur propre du stencil 5-points DISCRET (coherent avec les
  /// gradients du transport, bit-identique au comportement historique). true = symbole CONTINU
  /// lambda(k) = -(kx^2 + ky^2) avec frequences signees k in [0..N/2-1, -N/2..-1] * (2pi/L) --
  /// exactement la convention des solveurs spectraux de reference (ex. poisson_fft.m de RIEMOM2D) ;
  /// la solution est alors le Poisson SPECTRAL (exact sur les sinusoides), qui differe du stencil
  /// discret par O(h^2).
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

  int ny_local() const { return nyl_; }  // lignes (y) possedees par ce rang
  int nx() const { return Nx_; }
  int y_begin() const { return rank_ * nyl_; }  // 1ere ligne globale du rang

  // rho_local et phi_local : nyl_ x Nx_ (row-major, lignes globales
  // [y_begin, y_begin+nyl_)). phi_local est (re)dimensionne.
  void solve(const std::vector<double>& rho_local,
             std::vector<double>& phi_local) {
    std::vector<cplx> A(static_cast<std::size_t>(nyl_) * Nx_);
    for (std::size_t t = 0; t < A.size(); ++t) A[t] = cplx(rho_local[t], 0.0);
    for (int jl = 0; jl < nyl_; ++jl) fft1d(&A[jl * Nx_], Nx_, false);  // FFT-x

    std::vector<cplx> B(static_cast<std::size_t>(nxl_) * Ny_);
    transpose_fwd(A, B);                                          // -> (nxl x Ny)
    for (int il = 0; il < nxl_; ++il) fft1d(&B[il * Ny_], Ny_, false);  // FFT-y

    for (int il = 0; il < nxl_; ++il) {
      const int kx = rank_ * nxl_ + il;
      const int kxs = (kx < (Nx_ + 1) / 2) ? kx : kx - Nx_;  // frequence signee (Nyquist -> -N/2)
      const double wx = 2.0 * std::numbers::pi * kxs / (Nx_ * dx_);
      const double lx = spectral_ ? -(wx * wx)
                                  : (2.0 * std::cos(2.0 * std::numbers::pi * kx / Nx_) - 2.0) / (dx_ * dx_);
      for (int ky = 0; ky < Ny_; ++ky) {
        const int kys = (ky < (Ny_ + 1) / 2) ? ky : ky - Ny_;
        const double wy = 2.0 * std::numbers::pi * kys / (Ny_ * dy_);
        const double ly = spectral_ ? -(wy * wy)
                                    : (2.0 * std::cos(2.0 * std::numbers::pi * ky / Ny_) - 2.0) / (dy_ * dy_);
        const double lam = lx + ly;
        cplx& v = B[il * Ny_ + ky];
        v = (std::abs(lam) < 1e-14) ? cplx(0.0, 0.0) : v / lam;
      }
    }

    for (int il = 0; il < nxl_; ++il) fft1d(&B[il * Ny_], Ny_, true);  // IFFT-y
    transpose_bwd(B, A);                                         // -> (nyl x Nx)
    for (int jl = 0; jl < nyl_; ++jl) fft1d(&A[jl * Nx_], Nx_, true);  // IFFT-x

    phi_local.resize(A.size());
    for (std::size_t t = 0; t < A.size(); ++t) phi_local[t] = A[t].real();
  }

 private:
  // transpose A(nyl x Nx) -> B(nxl x Ny) : alltoall de blocs (nyl x nxl).
  void transpose_fwd(const std::vector<cplx>& A, std::vector<cplx>& B) {
    const int blk = nyl_ * nxl_;
    std::vector<cplx> send(static_cast<std::size_t>(np_) * blk), recv(send.size());
    for (int s = 0; s < np_; ++s)
      for (int jl = 0; jl < nyl_; ++jl)
        for (int il = 0; il < nxl_; ++il)
          send[s * blk + jl * nxl_ + il] = A[jl * Nx_ + s * nxl_ + il];
    alltoall(send, recv, blk);
    for (int q = 0; q < np_; ++q)
      for (int jl = 0; jl < nyl_; ++jl)
        for (int il = 0; il < nxl_; ++il)
          B[il * Ny_ + q * nyl_ + jl] = recv[q * blk + jl * nxl_ + il];
  }

  // transpose B(nxl x Ny) -> A(nyl x Nx) : alltoall de blocs (nxl x nyl).
  void transpose_bwd(const std::vector<cplx>& B, std::vector<cplx>& A) {
    const int blk = nxl_ * nyl_;
    std::vector<cplx> send(static_cast<std::size_t>(np_) * blk), recv(send.size());
    for (int s = 0; s < np_; ++s)
      for (int il = 0; il < nxl_; ++il)
        for (int jl = 0; jl < nyl_; ++jl)
          send[s * blk + il * nyl_ + jl] = B[il * Ny_ + s * nyl_ + jl];
    alltoall(send, recv, blk);
    for (int q = 0; q < np_; ++q)
      for (int il = 0; il < nxl_; ++il)
        for (int jl = 0; jl < nyl_; ++jl)
          A[jl * Nx_ + q * nxl_ + il] = recv[q * blk + il * nyl_ + jl];
  }

  // echange tous-vers-tous de `blk` complexes par rang (identite si np==1).
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
