#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>  // PolarGeometry
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/parallel/comm.hpp>

#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <vector>

/// @file
/// @brief Iterative POLAR elliptic operator with anisotropic TENSOR coefficient (cross terms).
///        Path A stage 2a (foundational brick toward the polar Schur).
///
/// CONTEXT. To build the FULL STIFF Euler-Poisson system in POLAR geometry (a strongly magnetized
/// regime, high omega_c), the condensed Schur stage (CondensedSchurSourceStepper, level 4 of
/// docs/SCHUR_CONDENSATION_DESIGN.md) is needed on the polar side. The lock is the elliptic OPERATOR:
/// the Schur condenses a FULL TENSOR operator A = I + c rho B^{-1} where B^{-1} is the Lorentz
/// rotation, which injects CROSS terms a_rt / a_tr (and a theta-dependent coefficient as soon as
/// rho or B_z varies in theta). The existing DIRECT PolarPoissonSolver (FFT-in-theta + tridiag-in-r,
/// polar_poisson_solver.hpp) diagonalizes theta via the FFT: it REQUIRES a coefficient CONSTANT in
/// theta and WITHOUT cross coupling. It is therefore structurally INCOMPATIBLE with the Schur operator.
/// This operator is the ITERATIVE (matrix-free) counterpart that accepts the full tensor.
///
/// EQUATION. On a ring (r, theta), r in [r_min, r_max] > 0 (NO r=0 singularity), theta PERIODIC,
/// we solve the operator in DIVERGENCE form with polar metric and tensor A in the ORTHONORMAL
/// physical frame (e_r, e_theta):
///   L(phi) = div(A grad phi),   grad phi = (d_r phi, (1/r) d_theta phi),
///   div F  = (1/r) d_r(r F_r) + (1/r) d_theta(F_theta),
///   A = [[a_rr, a_rt], [a_tr, a_tt]]  (possibly NON symmetric: a_rt != a_tr).
/// The flux is F = A grad phi:
///   F_r     = a_rr d_r phi + a_rt (1/r) d_theta phi
///   F_theta = a_tr d_r phi + a_tt (1/r) d_theta phi
/// ISOTROPIC case a_rr=a_tt=1, a_rt=a_tr=0:
///   L = (1/r) d_r(r d_r phi) + (1/r^2) d_theta^2 phi   (scalar polar Laplacian, cf. PolarPoissonSolver).
///
/// CONSERVATIVE FINITE-VOLUME DISCRETIZATION (order 2, like assemble_rhs_polar / PolarPoissonSolver).
/// Cell (i, j) of volume weight r_i dr dtheta. We integrate div(F) over the cell and divide by the
/// weight, which gives on each face the flux carried by its FV LENGTH:
///   - RADIAL faces i+-1/2 (radius r_face, face weight r_face dtheta):
///       DIAGONAL radial term: (1/r_i) [ r_{i+1/2} a_rr^{i+1/2} (phi_{i+1,j}-phi_{i,j})/dr
///                                      - r_{i-1/2} a_rr^{i-1/2} (phi_{i,j}-phi_{i-1,j})/dr ] / dr;
///       CROSS radial term a_rt (1/r) d_theta phi at face i+-1/2 (d_theta averaged over 4 corners,
///       1/r_{face} local to the face);
///   - AZIMUTHAL faces j+-1/2 (radius r_i, face weight dr):
///       DIAGONAL azimuthal term: a_tt^{j+-1/2} (phi_{i,j+1}-2 phi+phi_{i,j-1}) / (r_i^2 dtheta^2);
///       CROSS azimuthal term a_tr d_r phi at face j+-1/2 (d_r averaged over 4 corners).
/// FACE coefficient = ARITHMETIC mean of the two adjacent centers (order 2; for the radial/azimuthal
/// diagonal term of a smooth medium the arithmetic mean suffices; the cross term is not a normal flux
/// -> arithmetic mean as well, cf. cartesian poisson_operator.hpp). The coefficients a_rr/a_tt/
/// a_rt/a_tr are at the CENTER of the cells (1 component each, ghosts filled by the caller).
///
/// METRIC VALIDITY: the radial faces use r_face(i) > 0 (ring, r_min > 0); no division by r=0. The
/// stencil diagonal is the SUM of the face weights (for the Jacobi preconditioner); in the isotropic
/// case it gives back EXACTLY the scalar polar stencil.
///
/// BOUNDARY CONDITIONS. theta PERIODIC (periodic ghosts, fill_ghosts with ylo/yhi=Periodic).
/// r_min/r_max: PHYSICAL BC Dirichlet (value at the face, reflection ghost 2 v - interior) or
/// homogeneous Neumann (Foextrap, ghost = interior). This is exactly the cartesian fill_ghosts applied
/// in (r, theta) (index direction 0 = radial, 1 = azimuthal; cf. PolarGeometry convention). The
/// Dirichlet data is folded into the residual (AFFINE operator) for the true residual r0, and
/// LINEARIZED (offset c_bc subtracted) for the matvec in the loop (correction directions): same
/// mechanics as the cartesian TensorKrylovSolver (krylov_solver.hpp).
///
/// SINGULAR OPERATOR (pure radial Neumann + periodic theta, no reaction): the CONSTANT is in the
/// kernel of L_int; BiCGStab diverges without treatment. We FIX THE GAUGE by PROJECTION onto the
/// subspace of ZERO FV MEAN (weighted mean r dr dtheta removed from the initial residual, the
/// solution, and each preconditioned correction direction). This is the ITERATIVE counterpart of the
/// mode-0 pinning of the direct PolarPoissonSolver, without perturbing the stencil. Detected
/// automatically (both radial boundaries non-Dirichlet); at least ONE Dirichlet boundary => invertible
/// operator, NO pinning, bit-identical Dirichlet path. The solution is then defined modulo a constant.
///
/// SOLVER: MATRIX-FREE BiCGStab Krylov (handles the NON symmetric cross term), PRECONDITIONED by a
/// SIMPLE preconditioner (NO MG V-cycle: the scoping, polar_poisson_solver.hpp warning, reports that
/// the MG V-cycle STAGNATES on the polar 1/r^2 anisotropy; we stick to Krylov + simple precond as
/// requested). Two preconditioners (cf. enum PolarPrecond):
///   - Jacobi (diagonal): the simplest, but the iteration count GROWS like 1/h^2 (ill-conditioned
///     Laplacian). On a fine grid it PLATEAUS without converging (cf. test, block E). Sanity check / fallback.
///   - RadialLine (DEFAULT): inverts EXACTLY the RADIAL tridiagonal of the diagonal block per theta
///     line (Thomas), azimuthal diagonal lumped. Attacks the strong radial coupling + the 1/r^2
///     anisotropy; LOW iteration count with MODERATE growth (measured: isotropic ~ x2 per grid
///     doubling, tensor ~ x2.4). Stays "simple" (no MG hierarchy, no coarse grid).
/// HONEST LIMITATION: the CROSS term and the AZIMUTHAL coupling are NOT in the preconditioner (which
/// only inverts the radial part) -> the iteration count of the tensor case grows faster than the
/// isotropic one. This is acceptable for the foundational brick (SOLID convergence, no stagnation
/// observed); an azimuthal preconditioner (theta line) or a robust MG-line would be a LATER
/// refinement. BiCGStab has a FIXED memory footprint (r, rhat, p, v, s, t, + the preconditioned
/// phat/shat) and does not store a growing Krylov basis.
///
/// SCOPE: MULTI-RANK MPI AND MULTI-BOX (several boxes PER RANK) by AZIMUTHAL splitting (theta only)
/// for the RadialLine precond, or by FREE 2D TILING (r AND theta) for the Jacobi precond. The dot
/// products (dot/L2 norm) are collective (all_reduce via mf_arith::dot) and local_size()==0-safe;
/// they loop over ALL the local boxes (mf_arith). The matvec goes through fill_ghosts (MPI halo
/// exchange + physical BC) which also fills the DIAGONAL CORNERS p(i+-1, j+-1) between adjacent boxes
/// -- read by the CROSS TERMS a_rt/a_tr of the 9-POINT stencil (fill_boundary enumerates the 8 shift
/// directions, fill_physical_bc extends the radial BC into the theta halo: without this corner, the
/// cross term would be WRONG at the box boundary, cf. test_polar_schur_multibox). The gauge projection
/// project_mean accumulates sum/vol over ALL the local boxes then all_reduces (numerator + denominator).
/// The RadialLine precond (Thomas in r) requires that a full RADIAL LINE (column i=0..nr-1 at fixed
/// theta) fit in ONE box (otherwise the sequential sweep in r would cross a box boundary): we therefore
/// require that EACH box cover the FULL RADIAL RANGE of the domain (splitting in theta only, BUT ANY
/// number of theta boxes per rank). M^{-1} is then BLOCK-DIAGONAL per box (no communication along r);
/// the tridiagonals are per LOCAL BOX (line_*_[li]). A splitting that cuts r UNDER RadialLine is
/// REFUSED by an explicit guard (check_radial_columns, instead of a silent wrong result). The Jacobi
/// fallback (PolarPrecond::Jacobi, per cell) has NO layout constraint: it accepts a full 2D TILING
/// (cuts r AND theta), at the cost of an iteration count growing like 1/h^2. The theta splitting is
/// legitimate anyway: theta is the PERIODIC direction and carries the unstable azimuthal mode;
/// cutting r (few cells, strong radial coupling) would be counterproductive.
/// SINGLE-RANK / SINGLE BOX: BIT-IDENTICAL path (all_reduce = identity in serial; the local_size()
/// loop = the fab(0) loop when there is only one box covering the whole ring).
///
/// ADDITIVE: no existing path is touched. The DIRECT scalar PolarPoissonSolver stays UNTOUCHED
/// (separate path); the CARTESIAN Schur stays BIT-IDENTICAL. This header is OPT-IN (stage 2b = wiring
/// a polar SchurReconstructKernel + a coupler, out of scope here).

namespace adc {

namespace detail {

/// RADIAL FACE contribution to the diagonal + cross stencil, at (i, j). Returns the local L_int
/// WITHOUT the azimuthal term (computed separately). Free device-clean functor (ADC_HD). Face
/// coefficients = arithmetic mean; metric r_face/r_i; cross term a_rt (1/r_face) (d_theta phi)_face.
/// Cross terms absent (hrt=false) -> only the diagonal radial term contributes, bit-identical to the
/// scalar polar stencil with a_rr=1.
ADC_HD inline Real polar_radial_div(const ConstArray4& p, const ConstArray4& arr, bool hrt,
                                    const ConstArray4& art, int i, int j, Real ri, Real rfm,
                                    Real rfp, Real idr, Real idth) {
  // Face a_rr coefficients (arithmetic mean of the adjacent centers).
  const Real arr_p = Real(0.5) * (arr(i, j) + arr(i + 1, j));
  const Real arr_m = Real(0.5) * (arr(i, j) + arr(i - 1, j));
  // DIAGONAL radial term: (1/r_i) [ r_{i+1/2} a_rr_p (phi_{i+1}-phi_i)/dr - r_{i-1/2} a_rr_m (phi_i-phi_{i-1})/dr ] / dr.
  Real out = (rfp * arr_p * (p(i + 1, j) - p(i, j)) - rfm * arr_m * (p(i, j) - p(i - 1, j))) *
             (idr * idr / ri);
  if (hrt) {  // CROSS radial term a_rt (1/r_face) d_theta phi at face i+-1/2 (d_theta averaged over 4 corners).
    const Real art_p = Real(0.5) * (art(i, j) + art(i + 1, j));
    const Real art_m = Real(0.5) * (art(i, j) + art(i - 1, j));
    // (d_theta phi)_face averaged over the 4 corners of the face: 1/(r_face) because grad_theta = (1/r) d_theta phi.
    const Real dth_p =
        (p(i, j + 1) + p(i + 1, j + 1) - p(i, j - 1) - p(i + 1, j - 1)) * (Real(0.25) * idth);
    const Real dth_m =
        (p(i - 1, j + 1) + p(i, j + 1) - p(i - 1, j - 1) - p(i, j - 1)) * (Real(0.25) * idth);
    // (1/r_i)(1/dr) [ r_{i+1/2} F_cross_p - r_{i-1/2} F_cross_m ], F_cross = a_rt (1/r_face) d_theta phi.
    // r_face * (1/r_face) = 1: the r_face metric cancels the 1/r_face of the azimuthal gradient -> factor 1.
    out += (art_p * dth_p - art_m * dth_m) * (idr / ri);
  }
  return out;
}

/// AZIMUTHAL FACE contribution to the diagonal + cross stencil, at (i, j). Returns the local L_int of
/// the azimuthal term. Face a_tt coefficients (arithmetic); metric 1/(r_i^2); cross term a_tr d_r phi
/// at face j+-1/2 (d_r averaged over 4 corners).
ADC_HD inline Real polar_azimuthal_div(const ConstArray4& p, const ConstArray4& att, bool htr,
                                       const ConstArray4& atr, int i, int j, Real ri, Real idr,
                                       Real idth) {
  const Real inv_r2 = Real(1) / (ri * ri);
  const Real att_p = Real(0.5) * (att(i, j) + att(i, j + 1));
  const Real att_m = Real(0.5) * (att(i, j) + att(i, j - 1));
  // DIAGONAL azimuthal term: a_tt^{j+-1/2} (phi_{j+1}-2 phi+phi_{j-1}) / (r_i^2 dtheta^2), flux form.
  Real out =
      (att_p * (p(i, j + 1) - p(i, j)) - att_m * (p(i, j) - p(i, j - 1))) * (idth * idth * inv_r2);
  if (htr) {  // CROSS azimuthal term a_tr d_r phi at face j+-1/2 (d_r averaged over 4 corners).
    const Real atr_p = Real(0.5) * (atr(i, j) + atr(i, j + 1));
    const Real atr_m = Real(0.5) * (atr(i, j) + atr(i, j - 1));
    const Real dr_p =
        (p(i + 1, j) + p(i + 1, j + 1) - p(i - 1, j) - p(i - 1, j + 1)) * (Real(0.25) * idr);
    const Real dr_m =
        (p(i + 1, j - 1) + p(i + 1, j) - p(i - 1, j - 1) - p(i - 1, j)) * (Real(0.25) * idr);
    // (1/r_i)(1/dtheta) [ F_cross_p - F_cross_m ], F_cross = a_tr d_r phi. Metric 1/r_i (face weight dr).
    out += (atr_p * dr_p - atr_m * dr_m) * (idth / ri);
  }
  return out;
}

/// Diagonal (coefficient of phi_{i,j}) of the diagonal POLAR stencil (radial + azimuthal), for the
/// Jacobi preconditioner. Cross terms EXCLUDED from the diagonal (they do not touch phi_{i,j}: the
/// corners i+-1, j+-1 are off-diagonal). Returns the (NEGATIVE) value of the diagonal coefficient of
/// L_int (sum of -face weights), like the scalar stencil (diag < 0).
ADC_HD inline Real polar_diag(const ConstArray4& arr, const ConstArray4& att, int i, int j, Real ri,
                              Real rfm, Real rfp, Real idr, Real idth) {
  const Real arr_p = Real(0.5) * (arr(i, j) + arr(i + 1, j));
  const Real arr_m = Real(0.5) * (arr(i, j) + arr(i - 1, j));
  const Real att_p = Real(0.5) * (att(i, j) + att(i, j + 1));
  const Real att_m = Real(0.5) * (att(i, j) + att(i, j - 1));
  const Real inv_r2 = Real(1) / (ri * ri);
  const Real rad = (rfp * arr_p + rfm * arr_m) * (idr * idr / ri);
  const Real azi = (att_p + att_m) * (idth * idth * inv_r2);
  return -(rad + azi);  // diagonal coefficient of L_int (the neighbors have positive coefficients)
}

/// L_int(phi) = div(A grad phi) in polar (apply). NAMED device-clean functor (recipe #93: first-
/// instantiated cross-TU kernel -> no extended lambda under nvcc). arr/att always provided (at least
/// a_rr=a_tt=1); art/atr optional (cross terms). r_min/dr/dtheta passed as scalars (PolarGeometry
/// ADC_HD accessors recomputed in the kernel).
struct PolarApplyKernel {
  ConstArray4 p;
  Array4 L;
  ConstArray4 arr, att;
  bool hrt, htr;
  ConstArray4 art, atr;
  Real r_min, dr, idr, idth;
  ADC_HD void operator()(int i, int j) const {
    const Real ri = r_min + (i + Real(0.5)) * dr;  // r_cell(i)
    const Real rfm = r_min + i * dr;               // r_face(i)   = r_{i-1/2}
    const Real rfp = r_min + (i + 1) * dr;         // r_face(i+1) = r_{i+1/2}
    L(i, j) = polar_radial_div(p, arr, hrt, art, i, j, ri, rfm, rfp, idr, idth) +
              polar_azimuthal_div(p, att, htr, atr, i, j, ri, idr, idth);
  }
};

/// out = (f - L0 phi) / |diag| -- one Jacobi iteration (point-by-point relaxation) on the DIAGONAL
/// polar stencil (cross terms excluded from the Jacobi splitting: they stay on the right-hand side via
/// the residual). out is WRITTEN. NAMED device-clean functor. Used by the Jacobi preconditioner:
/// applied to a residual z, returns M^{-1} z = diag^{-1} z (simple form, one sweep). idiag = 1/diag
/// stored separately.
struct PolarJacobiApplyKernel {
  ConstArray4 z;
  Array4 out;
  ConstArray4 idiag;  // 1 / diagonal coefficient of L_int (negative), precomputed
  ADC_HD void operator()(int i, int j) const { out(i, j) = z(i, j) * idiag(i, j); }
};

/// Copy component 0 (dst <- src). Local NAMED device-clean functor (the header does not include
/// geometric_mg.hpp: we do NOT depend on the MG V-cycle, deliberately discarded in polar).
struct PolarCopyKernel {
  Array4 d;
  ConstArray4 s;
  ADC_HD void operator()(int i, int j) const { d(i, j) = s(i, j, 0); }
};

/// Computes idiag = 1 / diag of the diagonal polar stencil (for Jacobi). diag = polar_diag (< 0).
struct PolarInvDiagKernel {
  ConstArray4 arr, att;
  Array4 idiag;
  Real r_min, dr, idr, idth;
  ADC_HD void operator()(int i, int j) const {
    const Real ri = r_min + (i + Real(0.5)) * dr;
    const Real rfm = r_min + i * dr;
    const Real rfp = r_min + (i + 1) * dr;
    const Real d = polar_diag(arr, att, i, j, ri, rfm, rfp, idr, idth);
    idiag(i, j) = d != Real(0) ? Real(1) / d : Real(0);
  }
};

}  // namespace detail

/// Applies L_int(phi) = div(A grad phi) in polar over the whole MultiFab. Ghosts of @p phi assumed
/// filled (theta periodic, r physical). a_rr/a_tt: diagonal coefficients (1 component, centers;
/// nullptr -> uniform coefficient 1, isotropic). a_rt/a_tr: cross terms (nullptr -> absent).
inline void apply_polar_tensor(const MultiFab& phi, const PolarGeometry& geom, MultiFab& lap,
                               const MultiFab* a_rr, const MultiFab* a_tt, const MultiFab* a_rt,
                               const MultiFab* a_tr) {
  // CONTRACT: a_rr/a_tt required (the diagonal coefficients of the stencil are always read). An
  // isotropic case provides fields CONSTANT at 1 (PolarTensorKrylovSolver does this via its internal
  // stores). a_rt/a_tr optional (cross terms). We cannot deref a nullptr in the kernel -> guard at entry.
  if (!a_rr || !a_tt)
    throw std::runtime_error(
        "apply_polar_tensor: a_rr and a_tt required (fields at 1 if isotropic)");
  const Real dr = geom.dr();
  const Real idr = Real(1) / dr;
  const Real idth = Real(1) / geom.dtheta();
  for (int li = 0; li < phi.local_size(); ++li) {
    const ConstArray4 p = phi.fab(li).const_array();
    Array4 L = lap.fab(li).array();
    const Box2D v = lap.box(li);
    const ConstArray4 arr = a_rr->fab(li).const_array();
    const ConstArray4 att = a_tt->fab(li).const_array();
    const bool hrt = a_rt != nullptr;
    const bool htr = a_tr != nullptr;
    const ConstArray4 art = hrt ? a_rt->fab(li).const_array() : ConstArray4{};
    const ConstArray4 atr = htr ? a_tr->fab(li).const_array() : ConstArray4{};
    for_each_cell(
        v, detail::PolarApplyKernel{p, L, arr, att, hrt, htr, art, atr, geom.r_min, dr, idr, idth});
  }
}

/// Contract of the iterative POLAR elliptic operators: same shape as PolarEllipticSolver (cf.
/// polar_poisson_solver.hpp) + a tolerance variant solve(rel_tol, max_iters) (polar counterpart of the
/// cartesian LinearSolver). We do NOT redefine PolarEllipticSolver (defined with PolarPoissonSolver):
/// it is included indirectly via the same members. Return type of solve(tol,it) is NON void.
template <class S>
concept PolarLinearSolver = requires(S s, Real tol, int it) {
  { s.rhs() } -> std::same_as<MultiFab&>;
  { s.phi() } -> std::same_as<MultiFab&>;
  s.solve();
  { s.residual() } -> std::convertible_to<Real>;
  { s.geom() } -> std::convertible_to<const PolarGeometry&>;
  s.solve(tol, it);
  requires !std::same_as<decltype(s.solve(tol, it)), void>;
};

/// Result of a polar BiCGStab solve: iterations, relative residual, convergence. (Same shape as the
/// cartesian KrylovResult -- we reuse the type if already included; otherwise we declare the local
/// counterpart.)
struct PolarKrylovResult {
  int iters = 0;
  Real rel_residual = 0;
  bool converged = false;
};

/// Choice of the SIMPLE BiCGStab PRECONDITIONER (NO MG V-cycle -- stagnation on polar 1/r^2, cf.
/// header). Two options, both "simple" (no multigrid hierarchy):
///   - Jacobi: pure diagonal M^{-1} = diag^{-1}. The simplest, but the iteration count GROWS like
///     1/h^2 (ill-conditioned Laplacian): useful as a sanity check, poor on a fine grid.
///   - RadialLine: inverts EXACTLY the RADIAL tridiagonal of the diagonal block per theta line
///     (Thomas, like the direct solver), the azimuthal block diagonal being lumped. Attacks the
///     STRONGLY coupled radial direction (and the 1/r^2 anisotropy via the lumped diagonal), leaving
///     to Krylov only the residual azimuthal coupling: iteration count NEARLY INDEPENDENT of h.
///     Stays "simple" (no MG, no coarse grid). DEFAULT.
enum class PolarPrecond { Jacobi, RadialLine };

/// MATRIX-FREE BiCGStab Krylov solver for the FULL-tensor POLAR elliptic operator
/// L_int(phi) = div(A grad phi), A = [[a_rr, a_rt], [a_tr, a_tt]] possibly NON symmetric. SIMPLE
/// preconditioner (diagonal Jacobi or RadialLine = radial Thomas per theta line). NO MG V-cycle
/// (stagnation on polar 1/r^2, cf. header and PolarPrecond).
///
/// LIFE CYCLE: built on a PolarGeometry + BoxArray + BCRec (radial). In multi-rank MPI, the BoxArray
/// must be split in THETA ONLY (each box covers the full radial range) for the RadialLine
/// preconditioner (cf. scope + check_radial_columns); the Jacobi fallback has no such constraint.
/// rhs()/phi() are the solve fields (warm start on phi()). set_coefficients(...) sets the tensor
/// (pointers to coefficient fields at the CENTER, ghosts filled by the caller before solve). In the
/// isotropic case, call without a_rt/a_tr and provide a_rr/a_tt = fields at 1 (fill_one helper).
class PolarTensorKrylovSolver {
 public:
  /// @param geom ring (r, theta); @param ba BoxArray (single box or THETA splitting in MPI/multi-box);
  /// @param bc radial BC (xlo/xhi), theta periodic.
  /// @param precond simple preconditioner (RadialLine by default; Jacobi as fallback/sanity check).
  PolarTensorKrylovSolver(const PolarGeometry& geom, const BoxArray& ba, const BCRec& bc,
                          PolarPrecond precond = PolarPrecond::RadialLine)
      : geom_(geom),
        bc_(force_theta_periodic(bc)),
        precond_(precond),
        dm_(ba.size(), n_ranks()),
        phi_(ba, dm_, 1, 1),
        rhs_(ba, dm_, 1, 0),
        r_(ba, dm_, 1, 0),
        rhat_(ba, dm_, 1, 0),
        p_(ba, dm_, 1, 0),
        v_(ba, dm_, 1, 0),
        s_(ba, dm_, 1, 0),
        t_(ba, dm_, 1, 0),
        phat_(ba, dm_, 1, 1),
        shat_(ba, dm_, 1, 1),
        idiag_(ba, dm_, 1, 0),
        op_offset_(ba, dm_, 1, 0),
        a_rr_store_(ba, dm_, 1, 1),
        a_tt_store_(ba, dm_, 1, 1) {
    // LAYOUT GUARD (replaces the single-rank guard): the RadialLine preconditioner solves a RADIAL
    // tridiagonal per theta line via a SEQUENTIAL Thomas sweep in r. This sweep must stay LOCAL to a
    // box (it cannot cross a box/rank boundary). We therefore REQUIRE that each box of the BoxArray
    // cover the FULL RADIAL RANGE [domain.lo[0], domain.hi[0]] (splitting in theta only). The Jacobi
    // fallback (per cell) has no such constraint -> no check. Single-rank single box: the box covers
    // the whole ring, the check passes trivially (path unchanged).
    if (precond_ == PolarPrecond::RadialLine)
      check_radial_columns(ba);
    // Default diagonal coefficients = 1 (isotropic): internal fields at 1, ghosts filled Foextrap.
    a_rr_store_.set_val(Real(1));
    a_tt_store_.set_val(Real(1));
    a_rr_ = &a_rr_store_;
    a_tt_ = &a_tt_store_;
  }

  // RULE OF FIVE (C.21): the current pointers a_rr_/a_tt_ alias the internal stores
  // a_rr_store_/a_tt_store_ or external fields. A DEFAULT copy/move would leave these pointers aiming
  // at the SOURCE object's stores (dangling/UB). The solver is ALWAYS used as a LOCAL scope variable
  // (never copied, moved, stored in a container, nor returned by value): we DELETE the four
  // operations rather than write a move re-pointing the stores (useless here).
  PolarTensorKrylovSolver(const PolarTensorKrylovSolver&) = delete;
  PolarTensorKrylovSolver& operator=(const PolarTensorKrylovSolver&) = delete;
  PolarTensorKrylovSolver(PolarTensorKrylovSolver&&) = delete;
  PolarTensorKrylovSolver& operator=(PolarTensorKrylovSolver&&) = delete;

  // --- PolarEllipticSolver / PolarLinearSolver contract ---
  MultiFab& rhs() { return rhs_; }
  MultiFab& phi() { return phi_; }
  const PolarGeometry& geom() const { return geom_; }

  /// Sets the tensor A coefficients. a_rr/a_tt at the CENTER (internal default = 1 if not called);
  /// a_rt/a_tr cross terms (nullptr -> absent). The ghosts of the provided fields are filled HERE
  /// (coeff_bc: theta periodic preserved, radial Foextrap). The caller keeps ownership of the fields
  /// (pointers).
  void set_coefficients(MultiFab* a_rr, MultiFab* a_tt, MultiFab* a_rt = nullptr,
                        MultiFab* a_tr = nullptr) {
    a_rr_ = a_rr ? a_rr : &a_rr_store_;
    a_tt_ = a_tt ? a_tt : &a_tt_store_;
    a_rt_ = a_rt;
    a_tr_ = a_tr;
    fill_coeff_ghosts();
    coeffs_ready_ = false;  // idiag to be recomputed
  }

  /// Current GLOBAL L2 residual ||rhs - L_int(phi)|| (collective). Prepares once if needed.
  Real residual() {
    ensure_coeffs();
    apply_operator(phi_, r_);
    lincomb(r_, Real(1), rhs_, Real(-1), r_);
    return l2_norm(r_);
  }

  void solve() { solve(Real(1e-10), 400); }

  /// MATRIX-FREE BiCGStab preconditioned by precond_ (RadialLine by default, Jacobi as fallback);
  /// fixes the gauge (project_mean) when pin_gauge_ (singular pure Neumann/periodic case). phi() =
  /// unknown (warm start), rhs() = right-hand side. Returns iterations + relative residual + convergence.
  PolarKrylovResult solve(Real rel_tol, int max_iters) {
    ensure_coeffs();
    prepare_offset();  // c_bc = apply_operator(0) (inhomogeneous part of Dirichlet boundary) once
    PolarKrylovResult res;
    // MULTI-RANK MPI: ALL ranks (including those without a box, local_size()==0) execute the SAME
    // BiCGStab body. The per-fab operations (lincomb/saxpy/copy/apply_precond/apply_polar_tensor) are
    // no-ops on an empty rank; the COLLECTIVES (dot/l2_norm/project_mean -> all_reduce_sum) are called
    // by ALL ranks in the SAME order (an empty rank contributes 0). The stopping criteria rely only on
    // these GLOBAL scalars (identical everywhere) -> no desynchronization, no separate "run_empty_rank"
    // path to maintain in parallel. fill_ghosts (fill_boundary) is a PAIRWISE exchange (paired
    // Isend/Irecv, not a collective): an empty rank posts nothing and nobody sends to it -> no deadlock.
    // r0 = rhs - L_int(phi) (AFFINE operator: the Dirichlet data is folded into the residual).
    apply_operator(phi_, v_);
    lincomb(r_, Real(1), rhs_, Real(-1), v_);
    // SINGULAR OPERATOR (pure radial Neumann + periodic theta, no reaction): the constant is in the
    // kernel of L_int. BiCGStab then diverges (the residual keeps an undamped constant component). We
    // FIX THE GAUGE by projection onto the zero-mean subspace: we remove the mean of r0 (the RHS must
    // be compatible: zero mean in the FV sense) and of phi (gauge), then of each preconditioned
    // correction direction in the loop. This is the iterative counterpart of the mode-0 pinning of the
    // direct PolarPoissonSolver, without perturbing the stencil. Dirichlet case (>= one boundary):
    // pin_gauge_ stays false -> PATH UNCHANGED, bit-identical.
    if (pin_gauge_) {
      project_mean(r_);
      project_mean(phi_);
    }
    const Real bnorm = l2_norm(rhs_);
    const Real norm0 = bnorm > Real(0) ? bnorm : Real(1);
    Real rnorm = l2_norm(r_);
    res.rel_residual = rnorm / norm0;
    if (rnorm <= rel_tol * norm0) {
      res.converged = true;
      return res;
    }

    copy_into(rhat_, r_);
    p_.set_val(Real(0));
    v_.set_val(Real(0));
    Real rho_prev = Real(1), alpha = Real(1), omega = Real(1);

    for (int k = 1; k <= max_iters; ++k) {
      const Real rho = dot(rhat_, r_);  // COLLECTIVE
      if (std::fabs(rho) < kTiny || std::fabs(omega) < kTiny) {
        res.iters = k - 1;
        res.rel_residual = rnorm / norm0;
        return res;
      }
      const Real beta = (rho / rho_prev) * (alpha / omega);
      lincomb(p_, Real(1), p_, -omega, v_);  // p <- p - omega v
      lincomb(p_, beta, p_, Real(1), r_);    // p <- r + beta p
      apply_precond(p_, phat_);              // phat = M^{-1} p
      if (pin_gauge_)
        project_mean(phat_);          // gauge: zero-mean correction direction
      apply_operator_lin(phat_, v_);  // v = L_lin(phat) (LINEAR matvec)
      const Real rhat_dot_v = dot(rhat_, v_);
      if (std::fabs(rhat_dot_v) < kTiny) {
        res.iters = k - 1;
        res.rel_residual = rnorm / norm0;
        return res;
      }
      alpha = rho / rhat_dot_v;
      lincomb(s_, Real(1), r_, -alpha, v_);  // s <- r - alpha v
      saxpy(phi_, alpha, phat_);             // phi <- phi + alpha phat
      const Real snorm = l2_norm(s_);
      if (snorm <= rel_tol * norm0) {
        rnorm = snorm;
        res.iters = k;
        res.rel_residual = rnorm / norm0;
        res.converged = true;
        return res;
      }
      apply_precond(s_, shat_);  // shat = M^{-1} s
      if (pin_gauge_)
        project_mean(shat_);          // gauge: zero-mean correction direction
      apply_operator_lin(shat_, t_);  // t = L_lin(shat)
      const Real tt = dot(t_, t_);
      omega = tt > kTiny ? dot(t_, s_) / tt : Real(0);
      saxpy(phi_, omega, shat_);             // phi <- phi + omega shat
      lincomb(r_, Real(1), s_, -omega, t_);  // r <- s - omega t
      rnorm = l2_norm(r_);
      res.iters = k;
      res.rel_residual = rnorm / norm0;
      if (rnorm <= rel_tol * norm0) {
        res.converged = true;
        return res;
      }
      rho_prev = rho;
    }
    return res;  // max_iters reached: best effort (converged=false)
  }

 private:
  static constexpr Real kTiny = Real(1e-300);

  /// LAYOUT guard of the RadialLine preconditioner: each box must cover the FULL radial range of the
  /// domain (splitting in THETA only), otherwise the Thomas sweep in r would cross a box boundary.
  /// Throws on ALL ranks (the BoxArray is replicated, cf. DistributionMapping) -> no deadlock.
  /// Single-rank single box: the box == domain, check trivially true.
  void check_radial_columns(const BoxArray& ba) const {
    const Box2D dom = geom_.domain;
    for (int b = 0; b < ba.size(); ++b) {
      if (ba[b].lo[0] != dom.lo[0] || ba[b].hi[0] != dom.hi[0])
        throw std::runtime_error(
            "PolarTensorKrylovSolver (precond RadialLine): the BoxArray must be split in THETA "
            "only "
            "(each box covers the full radial range). The Thomas sweep in r cannot cross a box "
            "boundary. Use an azimuthal splitting, or the PolarPrecond::Jacobi fallback (per cell, "
            "without layout constraint) if r must be cut.");
    }
  }

  /// theta is PERIODIC by contract (ring, cf. header "BOUNDARY CONDITIONS"); we IMPOSE it at
  /// construction. A caller setting a physical azimuthal BC (System::poisson_bc puts Dirichlet on all
  /// FOUR faces) would otherwise pollute the inhomogeneous matvec (fill_ghosts of the candidate
  /// solution with bc_ -> odd reflection at the theta=0/2pi seam): wrong operator at the junction.
  /// Solver-side counterpart of the stepper phi_bc fix (a spurious azimuthal-seam drift; see
  /// docs/validation/HEADER_PROVENANCE.md).
  static BCRec force_theta_periodic(const BCRec& bc) {
    BCRec b = bc;
    b.ylo = BCType::Periodic;
    b.yhi = BCType::Periodic;
    b.ylo_val = Real(0);
    b.yhi_val = Real(0);
    return b;
  }

  /// BC of the coefficient fields: periodic preserved (theta), physical radial boundary -> Foextrap.
  BCRec coeff_bc() const {
    auto fo = [](BCType t) { return t == BCType::Periodic ? t : BCType::Foextrap; };
    BCRec b;
    b.xlo = fo(bc_.xlo);
    b.xhi = fo(bc_.xhi);
    b.ylo = BCType::Periodic;
    b.yhi = BCType::Periodic;  // theta always periodic
    return b;
  }

  void fill_coeff_ghosts() {
    const BCRec eb = coeff_bc();
    device_fence();
    fill_ghosts(*a_rr_, geom_.domain, eb);
    fill_ghosts(*a_tt_, geom_.domain, eb);
    if (a_rt_)
      fill_ghosts(*a_rt_, geom_.domain, eb);
    if (a_tr_)
      fill_ghosts(*a_tr_, geom_.domain, eb);
  }

  void ensure_coeffs() {
    if (coeffs_ready_)
      return;
    // idiag = 1/diag of the diagonal stencil (for Jacobi).
    const Real dr = geom_.dr();
    const Real idr = Real(1) / dr;
    const Real idth = Real(1) / geom_.dtheta();
    for (int li = 0; li < idiag_.local_size(); ++li) {
      for_each_cell(idiag_.box(li), detail::PolarInvDiagKernel{
                                        a_rr_->fab(li).const_array(), a_tt_->fab(li).const_array(),
                                        idiag_.fab(li).array(), geom_.r_min, dr, idr, idth});
    }
    if (precond_ == PolarPrecond::RadialLine)
      build_radial_lines();
    coeffs_ready_ = true;
  }

  /// Precomputes, for the RadialLine preconditioner, the coefficients of the RADIAL tridiagonal of the
  /// DIAGONAL block (per theta line j and per radius i). This is on the HOST (the precond Thomas sweeps
  /// read these arrays host-side, like PolarPoissonSolver). The a_rr coefficient is averaged in r (the
  /// two radial faces i+-1/2); the azimuthal term of the diagonal block is LUMPED into the diagonal
  /// b_ij (so that M approximates the full diag of L_int). Radial BC folded in HOMOGENEOUSLY (the
  /// precond acts on directions/residuals): Dirichlet b -= a/c, Neumann b += a/c.
  /// MULTI-BOX / MPI (theta-only splitting): we loop over the LOCAL boxes (local_size()). Each box
  /// covers the FULL radial range (check_radial_columns guard), so nr is global and the Thomas sweep
  /// stays box-local. The arrays are stored PER LOCAL BOX: line_b_[li] indexes [jl*nr + i] where jl is
  /// the theta index LOCAL to the box. Single-rank single box: only one box -> identical to the old
  /// [j*nr + i] layout over the whole domain (bit-identical path).
  void build_radial_lines() {
    a_rr_->sync_host();
    a_tt_->sync_host();
    const int nr = geom_.domain.nx();
    const Box2D dom = geom_.domain;
    const Real dr = geom_.dr();
    const Real idr2 = Real(1) / (dr * dr);
    const Real idth2 = Real(1) / (geom_.dtheta() * geom_.dtheta());
    const bool dir_lo = bc_.xlo == BCType::Dirichlet;
    const bool dir_hi = bc_.xhi == BCType::Dirichlet;
    nr_ = nr;
    const int nloc = idiag_.local_size();
    line_b_.resize(static_cast<std::size_t>(nloc));
    line_sub_.resize(static_cast<std::size_t>(nloc));
    line_sup_.resize(static_cast<std::size_t>(nloc));
    for (int li = 0; li < nloc; ++li) {
      const Box2D vb = idiag_.box(li);  // local VALID box (full r range, theta sub-range)
      const int nth_l = vb.ny();        // number of theta lines LOCAL to this box
      const ConstArray4 arr = a_rr_->fab(li).const_array();
      const ConstArray4 att = a_tt_->fab(li).const_array();
      std::vector<Real>& lb = line_b_[li];
      std::vector<Real>& lsub = line_sub_[li];
      std::vector<Real>& lsup = line_sup_[li];
      lb.assign(static_cast<std::size_t>(nth_l) * static_cast<std::size_t>(nr), Real(0));
      lsub.assign(static_cast<std::size_t>(nth_l) * static_cast<std::size_t>(nr), Real(0));
      lsup.assign(static_cast<std::size_t>(nth_l) * static_cast<std::size_t>(nr), Real(0));
      for (int jl = 0; jl < nth_l; ++jl) {
        const int jg = vb.lo[1] + jl;  // GLOBAL theta index
        for (int i = 0; i < nr; ++i) {
          const int ig = dom.lo[0] + i;
          const Real ri = geom_.r_cell(i);
          const Real rfm = geom_.r_face(i);
          const Real rfp = geom_.r_face(i + 1);
          const Real arr_p = Real(0.5) * (arr(ig, jg) + arr(ig + 1, jg));
          const Real arr_m = Real(0.5) * (arr(ig, jg) + arr(ig - 1, jg));
          const Real att_p = Real(0.5) * (att(ig, jg) + att(ig, jg + 1));
          const Real att_m = Real(0.5) * (att(ig, jg) + att(ig, jg - 1));
          const Real ai = rfm * arr_m * (idr2 / ri);  // coeff of p_{i-1} in L_int
          const Real ci = rfp * arr_p * (idr2 / ri);  // coeff of p_{i+1}
          const Real azi_diag =
              (att_p + att_m) * (idth2 / (ri * ri));  // azimuthal part of -diag (lumped)
          Real bi = -(ai + ci) - azi_diag;            // FULL diagonal of L_int (radial + azimuthal)
          Real sub = ai, sup = ci;
          if (i ==
              0) {  // HOMOGENEOUS low BC fold: Dirichlet b -= a, Neumann b += a; sub-diag zeroed
            bi += dir_lo ? -ai : ai;
            sub = Real(0);
          }
          if (i == nr - 1) {  // HOMOGENEOUS high BC fold
            bi += dir_hi ? -ci : ci;
            sup = Real(0);
          }
          const std::size_t idx = static_cast<std::size_t>(jl) * static_cast<std::size_t>(nr) +
                                  static_cast<std::size_t>(i);
          lb[idx] = bi;
          lsub[idx] = sub;
          lsup[idx] = sup;
        }
      }
    }
  }

  /// INHOMOGENEOUS MATRIX-FREE matvec: out = L_int(in), ghosts of in filled with bc_ (FULL BC). AFFINE
  /// in in under nonzero Dirichlet (constant term c_bc folded). Used for r0 / residual().
  void apply_operator(MultiFab& in, MultiFab& out) {
    device_fence();
    fill_ghosts(in, geom_.domain, bc_);
    apply_polar_tensor(in, geom_, out, a_rr_, a_tt_, a_rt_, a_tr_);
  }

  /// LINEAR MATRIX-FREE matvec: out = L_int(in) - c_bc. BiCGStab applies the matvec to correction
  /// DIRECTIONS -> the operator must be linear there (we subtract the boundary offset).
  void apply_operator_lin(MultiFab& in, MultiFab& out) {
    apply_operator(in, out);
    if (has_op_offset_)
      lincomb(out, Real(1), out, Real(-1), op_offset_);
  }

  /// SIMPLE preconditioner: Jacobi (diagonal) or RadialLine (radial Thomas per theta line). LINEAR,
  /// without inhomogeneous BC (acts on directions/residuals). SIMPLE counterpart of the cartesian
  /// Krylov M^{-1} (which does N MG V-cycles; here NO MG: polar 1/r^2 stagnation).
  void apply_precond(MultiFab& in, MultiFab& out) {
    if (precond_ == PolarPrecond::Jacobi) {
      for (int li = 0; li < out.local_size(); ++li)
        for_each_cell(out.box(li),
                      detail::PolarJacobiApplyKernel{in.fab(li).const_array(), out.fab(li).array(),
                                                     idiag_.fab(li).const_array()});
      return;
    }
    apply_precond_radial_line(in, out);
  }

  /// RadialLine: for each theta line j, solves the radial tridiagonal (sub/diag/sup precomputed in
  /// build_radial_lines) by Thomas, right-hand side = in(., j). out(., j) receives the correction.
  /// HOST (sequential Thomas in r) like PolarPoissonSolver; sync_host before reading in.
  /// MULTI-BOX / MPI: loop over the LOCAL boxes (each box covers the full r range -> the sweep stays
  /// box-local); line_b_/sub_/sup_[li] are indexed by the theta index LOCAL to the box. No MPI
  /// exchange (M^{-1} is BLOCK-DIAGONAL per box, by construction of the theta splitting) -> no
  /// deadlock. Single-rank single box: one box -> bit-identical path to the old fab(0).
  void apply_precond_radial_line(MultiFab& in, MultiFab& out) {
    if (in.local_size() == 0)
      return;
    in.sync_host();
    const int r_lo = geom_.domain.lo[0];
    const std::size_t N = static_cast<std::size_t>(nr_);
    cthom_.assign(N, Real(0));  // Thomas working super-diagonal
    xthom_.assign(N, Real(0));  // working solution
    for (int li = 0; li < in.local_size(); ++li) {
      const Box2D vb = in.box(li);
      const ConstArray4 z = in.fab(li).const_array();
      Array4 o = out.fab(li).array();
      const std::vector<Real>& lb = line_b_[li];
      const std::vector<Real>& lsub = line_sub_[li];
      const std::vector<Real>& lsup = line_sup_[li];
      const int nth_l = vb.ny();
      for (int jl = 0; jl < nth_l; ++jl) {
        const int jg = vb.lo[1] + jl;
        const std::size_t base = static_cast<std::size_t>(jl) * N;
        // Thomas in r: a = lsub, b = lb, c = lsup (base + i), rhs = z(., jg).
        Real beta = lb[base + 0];
        xthom_[0] = (beta != Real(0) ? z(r_lo, jg) / beta : Real(0));
        for (std::size_t i = 1; i < N; ++i) {
          cthom_[i] = lsup[base + i - 1] / beta;
          beta = lb[base + i] - lsub[base + i] * cthom_[i];
          const Real zi = z(r_lo + static_cast<int>(i), jg);
          xthom_[i] = (beta != Real(0) ? (zi - lsub[base + i] * xthom_[i - 1]) / beta : Real(0));
        }
        for (int i = static_cast<int>(N) - 2; i >= 0; --i)
          xthom_[static_cast<std::size_t>(i)] -=
              cthom_[static_cast<std::size_t>(i + 1)] * xthom_[static_cast<std::size_t>(i + 1)];
        for (std::size_t i = 0; i < N; ++i)
          o(r_lo + static_cast<int>(i), jg) = xthom_[i];
      }
    }
  }

  /// Prepares the offset c_bc = apply_operator(0) once per solve. Zero Dirichlet BC -> zero offset
  /// (has_op_offset_ = false), path unchanged. Also detects the SINGULAR operator (pure radial
  /// Neumann: no Dirichlet boundary) -> pin_gauge_: the constant is in the kernel, we will fix the
  /// gauge by zero-mean projection (cf. solve). At least one Dirichlet boundary => invertible =>
  /// pin_gauge_=false.
  void prepare_offset() {
    has_op_offset_ = (bc_.xlo == BCType::Dirichlet && bc_.xlo_val != Real(0)) ||
                     (bc_.xhi == BCType::Dirichlet && bc_.xhi_val != Real(0));
    pin_gauge_ = (bc_.xlo != BCType::Dirichlet) && (bc_.xhi != BCType::Dirichlet);
    if (has_op_offset_) {
      // Called on ALL ranks (apply_operator is a no-op on an empty rank except for the pairwise
      // fill_ghosts): no local_size() branch that would unpair the cross-rank halo exchange.
      phat_.set_val(Real(0));
      apply_operator(phat_, op_offset_);  // op_offset_ <- L_int(0) = c_bc
    }
  }

  /// Removes the FV mean (weighted by the volume r_i dr dtheta) from @p x (projection onto the
  /// zero-mean subspace, orthogonal to the constant kernel of L_int). Building block of the gauge
  /// pinning (singular pure Neumann operator). HOST (the weighted mean is a global scalar).
  /// MULTI-RANK MPI: the numerator (sum) and the denominator (vol) are first accumulated over the
  /// LOCAL cells (all the boxes of this rank), then all_reduced (MPI_SUM) -> the SAME GLOBAL mean on
  /// each rank; we then subtract it over all the local cells. COLLECTIVE: all_reduce_sum is called on
  /// EACH rank (including a box-less rank, which contributes 0) -> no deadlock. Single-rank:
  /// all_reduce_sum = identity, only one box -> bit-identical path.
  void project_mean(MultiFab& x) {
    x.sync_host();
    const int r_lo = geom_.domain.lo[0];
    const Real dr = geom_.dr(), dth = geom_.dtheta();
    Real sum = 0, vol = 0;
    for (int li = 0; li < x.local_size(); ++li) {
      const Box2D vb = x.box(li);
      const ConstArray4 a = x.fab(li).const_array();
      for (int j = vb.lo[1]; j <= vb.hi[1]; ++j)
        for (int i = vb.lo[0]; i <= vb.hi[0]; ++i) {
          const Real w = geom_.r_cell(i - r_lo) * dr * dth;  // i - r_lo = 0-based radial offset
          sum += a(i, j) * w;
          vol += w;
        }
    }
    // COLLECTIVE all-reduce (over all ranks, including empty ones): GLOBAL mean identical everywhere.
    sum = static_cast<Real>(all_reduce_sum(static_cast<double>(sum)));
    vol = static_cast<Real>(all_reduce_sum(static_cast<double>(vol)));
    const Real mean = vol > Real(0) ? sum / vol : Real(0);
    for (int li = 0; li < x.local_size(); ++li) {
      const Box2D vb = x.box(li);
      Array4 a = x.fab(li).array();
      for (int j = vb.lo[1]; j <= vb.hi[1]; ++j)
        for (int i = vb.lo[0]; i <= vb.hi[0]; ++i)
          a(i, j) -= mean;
    }
  }

  Real l2_norm(const MultiFab& x) { return std::sqrt(dot(x, x)); }

  void copy_into(MultiFab& dst, const MultiFab& src) {
    for (int li = 0; li < dst.local_size(); ++li) {
      Array4 d = dst.fab(li).array();
      const ConstArray4 s = src.fab(li).const_array();
      for_each_cell(dst.box(li), detail::PolarCopyKernel{d, s});
    }
  }

  PolarGeometry geom_;
  BCRec bc_;
  PolarPrecond precond_;
  DistributionMapping dm_;
  MultiFab phi_, rhs_;
  MultiFab r_, rhat_, p_, v_, s_, t_;
  MultiFab phat_, shat_;
  MultiFab idiag_;      ///< 1/diag (Jacobi), recomputed at each set_coefficients
  MultiFab op_offset_;  ///< c_bc = apply_operator(0) (inhomogeneous part of Dirichlet boundary)
  MultiFab a_rr_store_, a_tt_store_;  ///< default diagonal coefficients (= 1, isotropic)
  MultiFab* a_rr_ = nullptr;  ///< current coefficients (point to the store or the external one)
  MultiFab* a_tt_ = nullptr;
  MultiFab* a_rt_ = nullptr;  ///< cross terms (nullptr -> absent)
  MultiFab* a_tr_ = nullptr;
  bool coeffs_ready_ = false;
  bool has_op_offset_ = false;
  bool pin_gauge_ =
      false;  ///< singular operator (pure Neumann): fix the gauge (zero-mean projection)
  // RadialLine preconditioner: radial tridiagonal of the diagonal block per theta line. PER LOCAL BOX
  // (line_b_/sub_/sup_[li]), each stored [jl*nr + i] where jl = theta index LOCAL to the box, i = global
  // radius 0..nr-1 (a_rr may depend on theta -> coeffs per (i, jl)). The splitting being in THETA only,
  // M^{-1} is block-diagonal per box (no cross-box radial coupling) -> each box is inverted
  // independently. cthom_/xthom_ = Thomas working buffers (reused, length nr). HOST.
  std::vector<std::vector<Real>> line_b_, line_sub_,
      line_sup_;                             ///< tridiag per local box [li][jl*nr+i]
  mutable std::vector<Real> cthom_, xthom_;  ///< Thomas precond buffers (length nr)
  int nr_ = 0;
};

static_assert(PolarLinearSolver<PolarTensorKrylovSolver>,
              "PolarTensorKrylovSolver must model PolarLinearSolver");

}  // namespace adc
