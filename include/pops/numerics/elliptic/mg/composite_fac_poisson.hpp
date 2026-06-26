#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/layout/refinement.hpp>                    // average_down, coarsen_index
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>  // coarse solver (geometric multigrid)
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>  // apply_laplacian (residual, reads the already-filled ghosts)
#include <pops/numerics/time/amr/levels/amr_patch_range.hpp>  // PatchRange, CoverageMask (coarse footprint of a patch)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

/// @file
/// @brief CompositeFacPoisson: 2-level AMR COMPOSITE elliptic solver (Fast Adaptive Composite,
///        FAC) for the SCALAR Poisson Lap phi = f on a coarse level + ONE fine patch (ratio 2).
///
/// MOTIVATION (amr-schur path). The current AMR Poisson (Option A) solves the elliptic only on the
/// coarse level then injects grad phi (piecewise constant) onto the fine patches: the patches refine
/// the TRANSPORT but NOT the elliptic coupling. A COMPOSITE solver makes the fine patch ACTUALLY
/// REFINE the elliptic solution (more accurate phi/grad phi near the patch). This is the AMR fidelity
/// lock (the composite Poisson coupling (FAC) that amr_reflux.hpp explicitly leaves to this solver).
///
/// 2-LEVEL FAC ALGORITHM (McCormick), one fine patch INTERIOR to the coarse domain. Composite solution
/// phi = phi_f on the patch, phi_c elsewhere:
///   0. initial coarse solve: GeometricMG(Lap phi_c = f_c, Dirichlet);
///   it. repeat:
///      1. C-F ghosts: fill the patch ghost ring by BILINEAR INTERPOLATION of phi_c (order
///         2 vs the constant injection of Option A) -> cell-centered C-F Dirichlet condition;
///      2. fine solve: red-black GS on the patch with FROZEN ghosts (Lap phi_f = f_f);
///      3. average_down phi_f -> phi_c on the COVERED coarse cells (consistency);
///      4. composite coarse residual: r_c = f_c - Lap phi_c (NON covered cells), 0 on covered ones,
///         + C-F FLUX CORRECTION: on the coarse cells BORDERING the patch, the flux through the
///         C-F face is replaced by the FINE flux (conservative sum of the 2 fine faces) -> two-way coupling;
///      5. coarse correction: GeometricMG(Lap e_c = r_c, homogeneous Dirichlet); phi_c += e_c (non covered);
///   until ||r_c|| (composite residual norm) below tolerance.
///
/// SCOPE (Phase 4a, multi fine patch). Cartesian, 2 levels, 1..N disjoint fine patches strictly
/// interior, aligned (lo even / hi odd) and SEPARATED by at least one coarse cell (NON adjacent),
/// ratio 2, REPLICATED MONO-BOX coarse (serial / single-rank). N == 1 -> mono-patch path bit-identical to
/// Phase 1 (ctor delegates, per-patch loops degenerate to a single patch). The fine-fine join (ADJACENT
/// patches), MPI and > 2 levels are Phase 4b. The MMS test validates that the fine patch REDUCES the
/// elliptic error near the patch vs coarse-only.
///
/// MULTI-PATCH (Phase 4a). Each fine patch has its own box (fine BoxArray); the FINE operations (bilinear
/// C-F ghosts, SOR, C-F flux correction) loop OVER EACH local patch. The coarse coverage
/// (CoverageMask) is the UNION of the coarse footprints of all patches: it tells which
/// coarse cells are shadowed (residual set to 0, average_down) and lets us skip a bordering
/// cell covered by ANOTHER patch. The separation of at least one coarse cell (ctor guard)
/// guarantees that no fine face is SHARED between two patches: each patch border is a true
/// coarse-fine join, so the bilinear C-F ghost (read from the coarse) and the flux correction are
/// exact patch by patch -- no fine-fine exchange needed.

namespace pops {

namespace detail {

/// BILINEAR interpolation of the coarse potential (cell-centered, @p C with ghosts) at the CENTER of the
/// fine cell (i, j). Ratio @p r. The fine center has abscissa (i+0.5)/r in coarse-step units, i.e.
/// the coarse center-index fx = (i+0.5)/r - 0.5; we interpolate the 4 surrounding coarse centers.
/// INTERIOR patch -> Ic, Ic+1, Jc, Jc+1 are in the coarse domain (ghosts included).
POPS_HD inline Real fac_bilerp_coarse(const ConstArray4& C, int i, int j, int r) {
  const Real fx = (Real(i) + Real(0.5)) / Real(r) - Real(0.5);
  const Real fy = (Real(j) + Real(0.5)) / Real(r) - Real(0.5);
  const int Ic = static_cast<int>(std::floor(fx));
  const int Jc = static_cast<int>(std::floor(fy));
  const Real tx = fx - Real(Ic), ty = fy - Real(Jc);
  const Real c00 = C(Ic, Jc, 0), c10 = C(Ic + 1, Jc, 0);
  const Real c01 = C(Ic, Jc + 1, 0), c11 = C(Ic + 1, Jc + 1, 0);
  return (Real(1) - tx) * (Real(1) - ty) * c00 + tx * (Real(1) - ty) * c10 +
         (Real(1) - tx) * ty * c01 + tx * ty * c11;
}

}  // namespace detail

/// 2-level COMPOSITE FAC Poisson solver (scalar). Built on the coarse layout (replicated mono-box)
/// + the fine patch (mono-box). The caller provides f_c (coarse) and f_f (fine); the solver returns
/// phi_c (coarse, covered = average_down of the fine) and phi_f (fine).
class CompositeFacPoisson {
 public:
  /// MONO-PATCH CTOR (Phase 1): DELEGATES to the multi-patch ctor with a fine BoxArray of a single box, so
  /// BIT-IDENTICAL to the old path. Kept for existing callers (AmrCouplerMP Option-A composite,
  /// mono-patch MMS tests).
  /// @p geom_c: coarse geometry (whole domain). @p ba_c: coarse BoxArray (mono-box covering
  ///             the domain). @p bc: domain BC (Dirichlet for this milestone). @p fine_box: box of the
  ///             fine patch (FINE index space, ratio 2, strictly interior). @p ratio: 2.
  CompositeFacPoisson(const Geometry& geom_c, const BoxArray& ba_c, const BCRec& bc,
                      const Box2D& fine_box, int ratio = 2)
      : CompositeFacPoisson(geom_c, ba_c, bc, BoxArray(std::vector<Box2D>{fine_box}), ratio) {}

  /// MULTI-PATCH CTOR (Phase 4a). @p fine_boxes: tiling of the fine level (1..N disjoint patches, FINE
  /// index space, ratio 2, strictly interior, aligned lo even / hi odd, SEPARATED by at least one
  /// coarse cell). The coarse stays replicated mono-box (single-rank). N == 1 -> mono-patch path.
  CompositeFacPoisson(const Geometry& geom_c, const BoxArray& ba_c, const BCRec& bc,
                      const BoxArray& fine_boxes, int ratio = 2)
      : geom_c_(geom_c),
        geom_f_(geom_c.refine(ratio)),
        ba_c_(ba_c),
        dm_c_(ba_c.size(), n_ranks()),
        bc_(bc),
        ratio_(ratio),
        ba_f_(fine_boxes),
        dm_f_(fine_boxes.size(), n_ranks()),
        mg_(geom_c, ba_c, bc, {}, /*replicated=*/true),
        phi_c_(ba_c, dm_c_, 1, 1),
        phi_f_(ba_f_, dm_f_, 1, 1),
        f_c_(ba_c, dm_c_, 1, 0),
        f_f_(ba_f_, dm_f_, 1, 0),
        res_c_(ba_c, dm_c_, 1, 0),
        eps_c_(ba_c, dm_c_, 1, 1),
        eps_f_(ba_f_, dm_f_, 1, 1),
        axy_c_(ba_c, dm_c_, 1, 1),
        ayx_c_(ba_c, dm_c_, 1, 1),
        axy_f_(ba_f_, dm_f_, 1, 1),
        ayx_f_(ba_f_, dm_f_, 1, 1),
        cov_(Box2D::from_extents(geom_c.domain.nx(), geom_c.domain.ny())) {
    require_separated_patches(
        fine_boxes);  // guard: NON adjacent patches (fine-fine join = Phase 4b)
    // coarse footprints (covered cells) PER PATCH: PatchRange (lo/2 .. (hi-1)/2). The global coarse
    // coverage = UNION of the footprints (any gap between disjoint patches stays NON covered).
    for (int g = 0; g < fine_boxes.size(); ++g)
      patch_coarse_.push_back(PatchRange(fine_boxes[g]).box());
    for (const Box2D& pc : patch_coarse_)
      cov_.mark(pc);
    phi_c_.set_val(Real(0));
    phi_f_.set_val(Real(0));
    eps_c_.set_val(Real(1));  // default permittivity 1 -> operator = Laplacian (scalar)
    eps_f_.set_val(Real(1));
    axy_c_.set_val(Real(0));  // default cross terms 0 -> diagonal block only
    ayx_c_.set_val(Real(0));
    axy_f_.set_val(Real(0));
    ayx_f_.set_val(Real(0));
  }

  MultiFab& rhs_coarse() {
    return f_c_;
  }  ///< coarse right-hand side f_c (div(eps grad phi_c) = f_c)
  MultiFab& rhs_fine() { return f_f_; }  ///< fine right-hand side f_f (div(eps grad phi_f) = f_f)
  MultiFab& phi_coarse() { return phi_c_; }
  MultiFab& phi_fine() { return phi_f_; }
  /// VARIABLE permittivity eps (at cell centers) PER LEVEL. Fill + use_variable_coefficient(true)
  /// to go from Lap phi = f to div(eps grad phi) = f -- the condensed Schur operator at B_z = 0
  /// (eps = 1 + theta^2 dt^2 alpha rho). eps unfilled / not enabled -> scalar (Phase 1), bit-identical.
  MultiFab& eps_coarse() { return eps_c_; }
  MultiFab& eps_fine() { return eps_f_; }
  void use_variable_coefficient(bool v) { has_eps_ = v; }
  /// Cross terms a_xy / a_yx (at cell centers) PER LEVEL: FULL tensor A = diag(eps,eps) +
  /// [[0,a_xy],[a_yx,0]]. This is the condensed Schur operator at B_z != 0 (a_xy = c rho w/det,
  /// a_yx = -a_xy, w = theta dt B_z) -- antisymmetric, NON self-adjoint. Small for the Schur step
  /// (c = theta^2 dt^2 alpha) -> convergent SOR/V-cycle (EXPLICIT cross terms). Not enabled -> diagonal
  /// block only (Phase 3a/1), bit-identical. Requires use_variable_coefficient(true) (the diagonal block).
  MultiFab& a_xy_coarse() { return axy_c_; }
  MultiFab& a_yx_coarse() { return ayx_c_; }
  MultiFab& a_xy_fine() { return axy_f_; }
  MultiFab& a_yx_fine() { return ayx_f_; }
  void use_cross_terms(bool v) { has_cross_ = v; }
  /// Coarse footprint of the FIRST fine patch (mono-patch compat). Multi-patch: see patch_coarse(g).
  const Box2D& patch_coarse() const { return patch_coarse_[0]; }
  /// Coarse footprint of fine patch @p g (0 <= g < n_fine_patches()).
  const Box2D& patch_coarse(int g) const { return patch_coarse_[g]; }
  /// Number of fine patches (size of the fine BoxArray).
  int n_fine_patches() const { return ba_f_.size(); }

  void set_verbose(bool v) { verbose_ = v; }
  /// true: iterate the FAC two-way coupling (C-F flux correction + coarse correction). false:
  /// ONE-WAY path (coarse solve + fine solve with bilinear C-F ghosts) -- the patch refines locally.
  void set_two_way(bool v) { two_way_ = v; }

  /// Solves the composite system. @return the final max composite residual.
  /// @p max_iters FAC iterations (two-way); @p fine_sweeps SOR sweeps per fine solve; @p tol tolerance.
  Real solve(int max_iters = 30, int fine_sweeps = 400, Real tol = 1e-9) {
    // VARIABLE COEFFICIENT (condensed Schur operator B_z=0): sets eps on the coarse solver and
    // fills the eps ghosts PER LEVEL. eps_c ghosts = zero-gradient (coeff_bc Foextrap, like the
    // Schur builder); eps_f C-F ghosts = bilerp of eps_c (consistency of the coefficient flux across
    // the interface). Without variable coefficient -> scalar Laplacian operator (Phase 1, bit-identical).
    if (has_eps_) {
      device_fence();
      fill_ghosts(eps_c_, geom_c_.domain, coeff_bc(bc_));
      fill_cf_coarse_to_fine(eps_c_, eps_f_);
      mg_.set_epsilon(eps_c_);
    }
    if (has_cross_) {  // FULL tensor (Schur B_z != 0): cross terms on both solvers + ghosts.
      device_fence();
      fill_ghosts(axy_c_, geom_c_.domain, coeff_bc(bc_));
      fill_ghosts(ayx_c_, geom_c_.domain, coeff_bc(bc_));
      fill_cf_coarse_to_fine(axy_c_, axy_f_);
      fill_cf_coarse_to_fine(ayx_c_, ayx_f_);
      mg_.set_cross_terms(axy_c_, ayx_c_);
    }
    // 0) initial coarse solve (gives a phi_c for the 1st C-F ghost).
    copy0(mg_.rhs(), f_c_);
    mg_.phi().set_val(Real(0));
    mg_.solve(Real(1e-12), 100);
    copy0(phi_c_, mg_.phi());

    // 1) bilinear C-F ghosts + fine solve (base ONE-WAY).
    refresh_fine(fine_sweeps);

    Real rnorm = composite_coarse_residual();
    if (verbose_ && my_rank() == 0)
      std::fprintf(stderr, "[FAC] init r_c=%.4e\n", rnorm);
    if (!two_way_) {
      last_residual_ = rnorm;
      return rnorm;
    }

    // 2) FAC two-way iterations: coarse correction (C-F flux) then re-solve fine.
    for (int it = 0; it < max_iters; ++it) {
      if (rnorm < tol)
        break;
      // coarse correction: Lap e_c = r_c (homogeneous Dirichlet), phi_c += e_c (non covered).
      copy0(mg_.rhs(), res_c_);
      mg_.phi().set_val(Real(0));
      mg_.solve(Real(1e-12), 100);
      add_uncovered(phi_c_, mg_.phi());
      // re-ghost + re-solve fine on the corrected phi_c.
      refresh_fine(fine_sweeps);
      rnorm = composite_coarse_residual();
      if (verbose_ && my_rank() == 0)
        std::fprintf(stderr, "[FAC] it=%d r_c=%.4e\n", it, rnorm);
    }
    last_residual_ = rnorm;
    return rnorm;
  }

  Real last_residual() const { return last_residual_; }

 private:
  /// dst <- src (component 0, valid cells).
  void copy0(MultiFab& dst, const MultiFab& src) {
    device_fence();
    for (int li = 0; li < dst.local_size(); ++li) {
      Array4 d = dst.fab(li).array();
      const ConstArray4 s = src.fab(li).const_array();
      const Box2D b = dst.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          d(i, j, 0) = s(i, j, 0);
    }
  }

  /// phi_c += e_c on the NON covered cells (the correction does not touch the covered = average_down).
  void add_uncovered(MultiFab& phi, const MultiFab& e) {
    device_fence();
    for (int li = 0; li < phi.local_size(); ++li) {
      Array4 p = phi.fab(li).array();
      const ConstArray4 ec = e.fab(li).const_array();
      const Box2D b = phi.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          if (!cov_.covered(i, j))
            p(i, j, 0) += ec(i, j, 0);
    }
  }

  /// Phase 4a guard: the fine patches must be disjoint AND separated by at least ONE coarse
  /// cell (PatchRange coarse footprints not even adjacent). Otherwise a fine face would be
  /// SHARED between two patches and wrongly treated as a coarse-fine border (bilinear C-F ghost +
  /// flux correction): the fine-fine join (ADJACENT multi-patch) is Phase 4b. We test: coarse
  /// footprint of patch g GROWN by one cell intersects footprint of patch h -> insufficient separation.
  static void require_separated_patches(const BoxArray& fine_boxes) {
    const int N = fine_boxes.size();
    for (int g = 0; g < N; ++g) {
      const Box2D ag = PatchRange(fine_boxes[g]).box();
      for (int h = g + 1; h < N; ++h) {
        const Box2D bh = PatchRange(fine_boxes[h]).box();
        if (!ag.grow(1).intersect(bh).empty())
          throw std::runtime_error(
              "CompositeFacPoisson: adjacent or overlapping fine patches (coarse footprints "
              "separated by less than one cell); the multi-patch fine-fine join is Phase 4b -- "
              "require disjoint patches separated by at least one coarse cell.");
      }
    }
  }

  /// Fills the ghost ring of EACH fine patch by bilerp of phi_c (cell-centered C-F Dirichlet).
  /// Since the patches are separated by at least one coarse cell, the ghost ring of a patch never
  /// overlaps the valid cells of another -> read from the coarse only (no fine-fine exchange).
  void fill_cf_ghosts() {
    const ConstArray4 C = phi_c_.fab(0).const_array();  // replicated mono-box coarse
    const int ng = phi_f_.n_grow();
    for (int li = 0; li < phi_f_.local_size(); ++li) {
      Array4 F = phi_f_.fab(li).array();
      const Box2D vb = phi_f_.box(li);
      for (int j = vb.lo[1] - ng; j <= vb.hi[1] + ng; ++j)
        for (int i = vb.lo[0] - ng; i <= vb.hi[0] + ng; ++i) {
          const bool inside = (i >= vb.lo[0] && i <= vb.hi[0] && j >= vb.lo[1] && j <= vb.hi[1]);
          if (inside)
            continue;  // ghosts only
          F(i, j, 0) = detail::fac_bilerp_coarse(C, i, j, ratio_);
        }
    }
  }

  /// Fills the ghosts of a fine COEFFICIENT field (@p fine) by bilerp of the coarse field (@p coarse):
  /// coefficient consistency at the C-F interface (the coefficient face at the patch border mixes the fine
  /// interior coeff and the injected coarse coeff). Generic (eps, a_xy, a_yx).
  void fill_cf_coarse_to_fine(const MultiFab& coarse, MultiFab& fine) {
    const ConstArray4 C = coarse.fab(0).const_array();  // replicated mono-box coarse
    const int ng = fine.n_grow();
    for (int li = 0; li < fine.local_size(); ++li) {
      Array4 F = fine.fab(li).array();
      const Box2D vb = fine.box(li);
      for (int j = vb.lo[1] - ng; j <= vb.hi[1] + ng; ++j)
        for (int i = vb.lo[0] - ng; i <= vb.hi[0] + ng; ++i) {
          const bool inside = (i >= vb.lo[0] && i <= vb.hi[0] && j >= vb.lo[1] && j <= vb.hi[1]);
          if (inside)
            continue;
          F(i, j, 0) = detail::fac_bilerp_coarse(C, i, j, ratio_);
        }
    }
  }

  /// Coefficient (eps) BC: periodic preserved, physical border -> zero-gradient (Foextrap), like
  /// the Schur builder (coeff_bc) -- the coefficient carries no Dirichlet.
  static BCRec coeff_bc(const BCRec& b) {
    auto fo = [](BCType t) { return t == BCType::Periodic ? t : BCType::Foextrap; };
    BCRec c;
    c.xlo = fo(b.xlo);
    c.xhi = fo(b.xhi);
    c.ylo = fo(b.ylo);
    c.yhi = fo(b.yhi);
    return c;
  }

  /// SOR over-relaxation factor ~ optimal for a patch (2/(1+sin(pi/N))) -> O(N) sweeps convergence
  /// instead of O(N^2) for GS. N = largest side of box @p b (computed per patch in multi-patch).
  Real sor_omega(const Box2D& b) const {
    const int N = std::max(b.nx(), b.ny());
    return Real(2) / (Real(1) + std::sin(Real(kPi_) / Real(N)));
  }

  /// Re-fills the bilinear C-F ghosts from phi_c then relaxes EACH fine patch (SOR) with FROZEN ghosts.
  void refresh_fine(int sweeps) {
    device_fence();
    fill_ghosts(phi_c_, geom_c_.domain,
                bc_);  // phi_c physical ghosts (the bilerp reads up to the border)
    fill_cf_ghosts();
    fine_sor(sweeps);
    average_down(phi_f_, phi_c_,
                 ratio_);  // consistency: coarse covered = fine average (multi-box OK)
  }

  /// Red-black SOR over EACH fine patch: div(eps grad phi_f) = f_f (eps = face harmonic), FROZEN
  /// ghosts (no re-filling). eps == 1 everywhere (scalar) -> Laplacian, bit-identical to Phase 1.
  /// The over-relaxation factor is computed PER PATCH (own size). Since the patches are separated, the
  /// 9-point stencil of a patch never reads the valid cells of another (frozen ghosts only).
  void fine_sor(int sweeps) {
    const Real idx2 = Real(1) / (geom_f_.dx() * geom_f_.dx());
    const Real idy2 = Real(1) / (geom_f_.dy() * geom_f_.dy());
    const bool he = has_eps_;
    const bool hc = has_cross_;
    const Real idx = Real(1) / geom_f_.dx(), idy = Real(1) / geom_f_.dy();  // cross_div: 1/dx, 1/dy
    for (int li = 0; li < phi_f_.local_size(); ++li) {
      const Box2D vb = phi_f_.box(li);
      const Real omega = sor_omega(vb);
      Array4 P = phi_f_.fab(li).array();
      const ConstArray4 Pc =
          phi_f_.fab(li).const_array();  // const view (same memory) for cross stencil
      const ConstArray4 F = f_f_.fab(li).const_array();
      const ConstArray4 E = eps_f_.fab(li).const_array();
      const ConstArray4 AXY = axy_f_.fab(li).const_array();
      const ConstArray4 AYX = ayx_f_.fab(li).const_array();
      for (int s = 0; s < sweeps; ++s)
        for (int color = 0; color < 2; ++color)
          for (int j = vb.lo[1]; j <= vb.hi[1]; ++j)
            for (int i = vb.lo[0]; i <= vb.hi[0]; ++i) {
              if (((i + j) & 1) != color)
                continue;
              // FACE permittivities (harmonic mean of the 2 centers); eps==1 -> faces == 1.
              const Real exm = he ? eps_harmonic(E(i, j, 0), E(i - 1, j, 0)) : Real(1);
              const Real exp = he ? eps_harmonic(E(i, j, 0), E(i + 1, j, 0)) : Real(1);
              const Real eym = he ? eps_harmonic(E(i, j, 0), E(i, j - 1, 0)) : Real(1);
              const Real eyp = he ? eps_harmonic(E(i, j, 0), E(i, j + 1, 0)) : Real(1);
              const Real diag = (exm + exp) * idx2 + (eym + eyp) * idy2;
              const Real nb = (exm * P(i - 1, j, 0) + exp * P(i + 1, j, 0)) * idx2 +
                              (eym * P(i, j - 1, 0) + eyp * P(i, j + 1, 0)) * idy2;
              // EXPLICIT cross terms (9 points, read from the current P): div(A grad phi) =
              // diag_block + cross. We solve diag_block(P) + cross(P) = f -> P = (nb + cross - f)/diag.
              const Real cross =
                  hc ? detail::cross_div(Pc, true, AXY, true, AYX, i, j, idx, idy) : Real(0);
              const Real pgs = (nb + cross - F(i, j, 0)) / diag;
              P(i, j, 0) = (Real(1) - omega) * P(i, j, 0) +
                           omega * pgs;  // over-relax (under-relax if strong)
            }
    }
  }

  /// Composite coarse residual: r_c = f_c - div(eps grad phi_c) (non covered), 0 (covered), + C-F
  /// FLUX correction on the cells bordering the patch. @return ||r_c||_inf (NON covered cells).
  Real composite_coarse_residual() {
    device_fence();
    fill_ghosts(phi_c_, geom_c_.domain, bc_);
    // r_c = f_c - div(A grad phi_c) (apply_laplacian reads the already-filled ghosts; eps + cross if active).
    // The cross terms are read also on the COVERED cells (= fine average after average_down) -> the
    // 9-point stencil stays consistent at the interface; only the NORMAL flux is explicitly joined C-F
    // (the cross flux, tangential and small for the Schur step, is carried by the volume stencil).
    MultiFab lap(ba_c_, dm_c_, 1, 0);
    apply_laplacian(phi_c_, geom_c_, lap, /*coef=*/nullptr, has_eps_ ? &eps_c_ : nullptr,
                    /*kappa=*/nullptr, /*eps_y=*/nullptr, has_cross_ ? &axy_c_ : nullptr,
                    has_cross_ ? &ayx_c_ : nullptr);
    device_fence();
    Array4 R = res_c_.fab(0).array();
    const ConstArray4 LAP = lap.fab(0).const_array();
    const ConstArray4 FC = f_c_.fab(0).const_array();
    const Box2D b = res_c_.box(0);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        R(i, j, 0) = cov_.covered(i, j) ? Real(0) : (FC(i, j, 0) - LAP(i, j, 0));

    // C-F FLUX CORRECTION, PER FINE PATCH. On each coarse cell BORDERING a patch (non covered,
    // covered neighbor), we REPLACE the contribution of the C-F face in div(eps grad phi_c) by the
    // FINE contribution (conservative sum of the r fine faces, harmonic face eps): r_c += (coarse
    // - fine). Since the patches are separated by at least one coarse cell, each border is a TRUE
    // coarse-fine join; the test !cov_.covered(I, J) defensively skips a bordering cell that would be
    // covered by ANOTHER patch (impossible under the guard, but robust: a covered bordering
    // cell is already interior to another patch, its residual stays 0). A cell SEPARATING two
    // patches (right border of one, left border of the other) gets TWO corrections, one per face: correct.
    const ConstArray4 PC = phi_c_.fab(0).const_array();
    const ConstArray4 EC = eps_c_.fab(0).const_array();
    const bool he = has_eps_;
    const Real idx2 = Real(1) / (geom_c_.dx() * geom_c_.dx());
    const Real idy2 = Real(1) / (geom_c_.dy() * geom_c_.dy());
    const int r = ratio_;
    for (int g = 0; g < phi_f_.local_size(); ++g) {
      const ConstArray4 PF = phi_f_.fab(g).const_array();
      const ConstArray4 EF = eps_f_.fab(g).const_array();
      const int Ic0 = patch_coarse_[g].lo[0], Ic1 = patch_coarse_[g].hi[0];
      const int Jc0 = patch_coarse_[g].lo[1], Jc1 = patch_coarse_[g].hi[1];
      // Faces NORMAL TO X: bordering columns I = Ic0-1 (covered +x face) and I = Ic1+1 (-x face).
      for (int J = Jc0; J <= Jc1; ++J) {
        if (!cov_.covered(Ic0 - 1, J)) {  // left: cell (Ic0-1, J), fine face at i = r*Ic0.
          const int I = Ic0 - 1;
          const Real efc = he ? eps_harmonic(EC(I, J, 0), EC(I + 1, J, 0)) : Real(1);
          const Real coarse_c = efc * (PC(I + 1, J, 0) - PC(I, J, 0)) * idx2;
          Real fine_sum = Real(0);
          for (int t = 0; t < r; ++t) {
            const int jf = r * J + t;
            const Real eff =
                he ? eps_harmonic(EF(r * Ic0 - 1, jf, 0), EF(r * Ic0, jf, 0)) : Real(1);
            fine_sum += eff * (PF(r * Ic0, jf, 0) - PF(r * Ic0 - 1, jf, 0));  // interior - ghost
          }
          R(I, J, 0) += coarse_c - fine_sum * idx2;
        }
        if (!cov_.covered(Ic1 + 1, J)) {  // right: cell (Ic1+1, J), fine faces at i = r*Ic1+r.
          const int I = Ic1 + 1;
          const Real efc = he ? eps_harmonic(EC(I, J, 0), EC(I - 1, J, 0)) : Real(1);
          const Real coarse_c = efc * (PC(I - 1, J, 0) - PC(I, J, 0)) * idx2;
          Real fine_sum = Real(0);
          for (int t = 0; t < r; ++t) {
            const int jf = r * J + t;
            const Real eff =
                he ? eps_harmonic(EF(r * Ic1 + r - 1, jf, 0), EF(r * Ic1 + r, jf, 0)) : Real(1);
            fine_sum += eff * (PF(r * Ic1 + r - 1, jf, 0) - PF(r * Ic1 + r, jf, 0));
          }
          R(I, J, 0) += coarse_c - fine_sum * idx2;
        }
      }
      // Faces NORMAL TO Y: bordering rows J = Jc0-1 (+y face) and J = Jc1+1 (-y face).
      for (int I = Ic0; I <= Ic1; ++I) {
        if (!cov_.covered(I, Jc0 - 1)) {
          const int J = Jc0 - 1;
          const Real efc = he ? eps_harmonic(EC(I, J, 0), EC(I, J + 1, 0)) : Real(1);
          const Real coarse_c = efc * (PC(I, J + 1, 0) - PC(I, J, 0)) * idy2;
          Real fine_sum = Real(0);
          for (int t = 0; t < r; ++t) {
            const int iff = r * I + t;
            const Real eff =
                he ? eps_harmonic(EF(iff, r * Jc0 - 1, 0), EF(iff, r * Jc0, 0)) : Real(1);
            fine_sum += eff * (PF(iff, r * Jc0, 0) - PF(iff, r * Jc0 - 1, 0));
          }
          R(I, J, 0) += coarse_c - fine_sum * idy2;
        }
        if (!cov_.covered(I, Jc1 + 1)) {
          const int J = Jc1 + 1;
          const Real efc = he ? eps_harmonic(EC(I, J, 0), EC(I, J - 1, 0)) : Real(1);
          const Real coarse_c = efc * (PC(I, J - 1, 0) - PC(I, J, 0)) * idy2;
          Real fine_sum = Real(0);
          for (int t = 0; t < r; ++t) {
            const int iff = r * I + t;
            const Real eff =
                he ? eps_harmonic(EF(iff, r * Jc1 + r - 1, 0), EF(iff, r * Jc1 + r, 0)) : Real(1);
            fine_sum += eff * (PF(iff, r * Jc1 + r - 1, 0) - PF(iff, r * Jc1 + r, 0));
          }
          R(I, J, 0) += coarse_c - fine_sum * idy2;
        }
      }
    }

    // inf norm of the residual over the NON covered cells.
    Real nrm = Real(0);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        if (!cov_.covered(i, j))
          nrm = std::fmax(nrm, std::fabs(R(i, j, 0)));
    return nrm;
  }

  Geometry geom_c_, geom_f_;
  BoxArray ba_c_;
  DistributionMapping dm_c_;
  BCRec bc_;
  int ratio_;
  BoxArray ba_f_;
  DistributionMapping dm_f_;
  GeometricMG mg_;  ///< coarse solver (initial + corrections), homogeneous Dirichlet
  MultiFab phi_c_, phi_f_, f_c_, f_f_, res_c_;
  MultiFab eps_c_, eps_f_;  ///< variable permittivity per level (condensed Schur operator B_z=0)
  MultiFab axy_c_, ayx_c_, axy_f_, ayx_f_;  ///< cross terms per level (full tensor, Schur B_z!=0)
  std::vector<Box2D> patch_coarse_;  ///< covered coarse footprint PER fine patch (multi-patch)
  CoverageMask cov_;
  Real last_residual_ = 0;
  bool has_eps_ = false;    ///< true: div(eps grad phi) operator; false: scalar Laplacian (Phase 1)
  bool has_cross_ = false;  ///< true: adds the cross terms a_xy/a_yx (full tensor, Schur B_z!=0)
  bool verbose_ = false;
  bool two_way_ = true;
  static constexpr Real kPi_ = Real(3.14159265358979323846);
};

}  // namespace pops
