# Spectral FFT Poisson


On a periodic domain, the Laplacian is diagonal in Fourier:
`phi_hat(k) = -rhs_hat(k) / (k_x^2 + k_y^2)`, mode `k=0` pinned to 0 (gauge). A direct
FFT + division + inverse FFT solves Poisson exactly (machine residual), without
iteration. Two variants exist, both models of the `EllipticSolver` concept:

- `PoissonFFTSolver` (`numerics/elliptic/poisson_fft_solver.hpp`), single-rank, single
  box. Its constructor raises a `std::runtime_error` as soon as `n_ranks() != 1` or
  `ba.size() != 1`. This safeguard is deliberate and active in Release (`NDEBUG` does not remove
  it): this direct solver would dereference `fab(0)` on a rank without a box (segfault). In
  serial, it is the exact and fastest solver for a periodic domain.
- `DistributedFFTSolver` (same header), FFT distributed by slabs: 1 slab
  per rank, parallel transpose by `MPI_Alltoall`. It is the MPI counterpart of the direct FFT,
  usable as `Coupler<Model, DistributedFFTSolver>`. Constraints: `Ny` divisible by
  `n_ranks()`, `Nx`/`Ny` powers of 2 (a fix handles `n` not a power of 2 on the
  single-rank side). In serial (`n_ranks() == 1`) a single slab covers the domain, identical to
  `PoissonFFTSolver`.

MG and FFT provably invert the same discrete Laplacian: the same canonical operator
`poisson_residual` applied to their two solutions gives residuals at round-off
(`~1e-14`) and solutions identical to `~1e-16`. The FFT pitfall: it requires the
periodic case, and the right-hand side must be zero-mean (otherwise `phi` drifts).
