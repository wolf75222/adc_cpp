#pragma once

/// @file
/// @brief Free functions of the elliptic operator: apply_laplacian (matvec), poisson_residual
///        (residual), gs_color/gs_smooth (red-black Gauss-Seidel smoother), zero_conductor (embedded Dirichlet).
///
/// Layer: `include/adc/numerics/elliptic/poisson`.
/// Role: low-level bricks of the geometric multigrid (geometric_mg.hpp) and matvec of the Krylov
/// solver (krylov_solver.hpp). The operator is provided as FREE FUNCTIONS (not a type): no concept
/// constrains them. GLOBAL convention: we solve L(phi) = -div(A grad phi) + kappa phi = f_phys;
/// internally the kernels assemble L_int = div(A grad phi) - kappa phi and poisson_residual returns
/// res = f - L_int.
/// Contract: all optional coefficients default to nullptr and THEN give back EXACTLY the bit-identical
/// historical path -- mask (embedded boundary, pins phi=0 in the conductor), coef
/// (Shortley-Weller cut-cell weights, order 2 at the boundary), eps/eps_y (variable permittivity, isotropic or
/// diagonal anisotropic, harmonic face mean), kappa (reaction term), a_xy/a_yx (FULL tensor,
/// EXPLICIT cross terms, A possibly non-symmetric).
///
/// Invariants:
/// - FACE permittivity = HARMONIC mean of the two adjacent centers (continuous normal flux, correct
///   even for discontinuous eps); the cross term uses the ARITHMETIC mean (not a normal flux);
/// - the gs_smooth smoother stays 5 POINTS (diagonal block): the cross terms a_xy/a_yx are EXPLICIT,
///   carried only by the residual -> for strongly non-symmetric A the GS V-cycle may NOT
///   converge (a Krylov solver is then required, cf. krylov_solver.hpp);
/// - the kernels are NAMED FUNCTORS (and not ADC_HD lambdas) because they are first-instantiated from an
///   external TU: an extended lambda would break the device kernel emission under nvcc;
/// - the red-black sweep is parallelizable (a red cell depends only on black cells).

#include <adc/core/types.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>

namespace adc {

// Harmonic mean of two center permittivities -> face permittivity.
// Guard against division by 0 if both centers are zero (inactive cell).
ADC_HD inline Real eps_harmonic(Real ec, Real ev) {
  const Real s = ec + ev;
  return s > Real(0) ? Real(2) * ec * ev / s : Real(0);
}

namespace detail {
// Weights of the FOUR faces (xm, xp, ym, yp) at (i,j) for the FACE permittivity path (he), with or
// without cut-cell. Each face = HARMONIC mean of the two adjacent centers (eps_x for x faces,
// eps_y for y faces); in cut-cell (hc) the face Shortley-Weller weight is multiplied by its
// permittivity, otherwise 1/h^2 (idx2 / idy2) is applied. Free ADC_HD functor (device-clean) shared
// by the three he kernels (apply / residual / smoother); body STRICTLY identical to the three
// original copies -> bit-identical output. Output by reference (wxm/wxp/wym/wyp).
ADC_HD inline void face_weights(const ConstArray4& ep, const ConstArray4& ey, int i, int j,
                                Real idx2, Real idy2, bool hc, const ConstArray4& cf, Real& wxm,
                                Real& wxp, Real& wym, Real& wyp) {
  const Real ec = ep(i, j);   // eps_x at center (x faces)
  const Real ecy = ey(i, j);  // eps_y at center (y faces); == ec when isotropic
  const Real exm = eps_harmonic(ec, ep(i - 1, j));
  const Real exp = eps_harmonic(ec, ep(i + 1, j));
  const Real eym = eps_harmonic(ecy, ey(i, j - 1));
  const Real eyp = eps_harmonic(ecy, ey(i, j + 1));
  if (hc) {  // cut-cell: eps_face multiplies each Shortley-Weller weight
    wxm = cf(i, j, 0) * exm;
    wxp = cf(i, j, 1) * exp;
    wym = cf(i, j, 2) * eym;
    wyp = cf(i, j, 3) * eyp;
  } else {  // 5-point stencil with variable face coefficient
    wxm = exm * idx2;
    wxp = exp * idx2;
    wym = eym * idy2;
    wyp = eyp * idy2;
  }
}

// Divergence of the CROSS FLUXES of the full tensor at (i,j): d_x(Axy d_y phi) + d_y(Ayx d_x phi),
// discretized by finite volumes (9-point stencil) as described in the header. Returns the
// contribution to be ADDED to div(A grad phi). Free ADC_HD functor (device-clean) shared by
// apply_laplacian and poisson_residual; face coefficient = ARITHMETIC mean. hxy/hyx
// guard each half-term: an ABSENT coefficient (field not provided) contributes 0 without deref.
ADC_HD inline Real cross_div(const ConstArray4& p, bool hxy, const ConstArray4& axy, bool hyx,
                             const ConstArray4& ayx, int i, int j, Real idx, Real idy) {
  Real out = Real(0);
  if (hxy) {  // x faces: cross flux = Axy_face * (d_y phi)_face, tangential averaged over 4 corners.
    const Real axy_xp = Real(0.5) * (axy(i, j) + axy(i + 1, j));
    const Real axy_xm = Real(0.5) * (axy(i, j) + axy(i - 1, j));
    const Real dyf_xp =
        (p(i, j + 1) + p(i + 1, j + 1) - p(i, j - 1) - p(i + 1, j - 1)) * (Real(0.25) * idy);
    const Real dyf_xm =
        (p(i - 1, j + 1) + p(i, j + 1) - p(i - 1, j - 1) - p(i, j - 1)) * (Real(0.25) * idy);
    out += (axy_xp * dyf_xp - axy_xm * dyf_xm) * idx;
  }
  if (hyx) {  // y faces: cross flux = Ayx_face * (d_x phi)_face.
    const Real ayx_yp = Real(0.5) * (ayx(i, j) + ayx(i, j + 1));
    const Real ayx_ym = Real(0.5) * (ayx(i, j) + ayx(i, j - 1));
    const Real dxf_yp =
        (p(i + 1, j) + p(i + 1, j + 1) - p(i - 1, j) - p(i - 1, j + 1)) * (Real(0.25) * idx);
    const Real dxf_ym =
        (p(i + 1, j - 1) + p(i + 1, j) - p(i - 1, j - 1) - p(i - 1, j)) * (Real(0.25) * idx);
    out += (ayx_yp * dxf_yp - ayx_ym * dxf_ym) * idy;
  }
  return out;
}

// NAMED FUNCTORS (and not ADC_HD lambdas) for the Poisson operator and Gauss-Seidel smoother kernels.
// Same reasons as the rest of the elliptic path (#93, recipe #64): these kernels are
// first-instantiated from the MG V-cycle pulled from an external TU (harness / native loader); an extended
// lambda there breaks the device kernel emission under nvcc (null kernel-stub -> Cuda segfault in
// Release -O without -g). Body STRICTLY identical to the former lambdas (same he/hc/hk branches,
// same stencil) -> bit-identical residual and potential on CPU and device.

// L = div(A grad phi) - kappa phi (apply_laplacian). cf/ep/ey/ka unused if the flag is false.
// hxy/hyx => FULL tensor: we ADD the cross fluxes d_x(Axy d_y phi) + d_y(Ayx d_x phi) (idx/idy
// = 1/dx, 1/dy; axy/ayx = off-diagonal coefficients at center). hxy=hyx=false => bit-identical.
struct ApplyLaplacianKernel {
  ConstArray4 p;
  Array4 L;
  Real idx2, idy2, idx, idy;
  bool hc;
  ConstArray4 cf;
  bool he;
  ConstArray4 ep, ey;
  bool hk;
  ConstArray4 ka;
  bool hxy, hyx;
  ConstArray4 axy, ayx;
  ADC_HD void operator()(int i, int j) const {
    if (he) {  // face permittivity (harmonic), with or without cut-cell
      Real wxm, wxp, wym, wyp;
      face_weights(ep, ey, i, j, idx2, idy2, hc, cf, wxm, wxp, wym, wyp);
      L(i, j) = wxp * p(i + 1, j) + wxm * p(i - 1, j) + wyp * p(i, j + 1) + wym * p(i, j - 1) -
                (wxm + wxp + wym + wyp) * p(i, j);
    } else if (hc)
      L(i, j) = cf(i, j, 1) * p(i + 1, j) + cf(i, j, 0) * p(i - 1, j) + cf(i, j, 3) * p(i, j + 1) +
                cf(i, j, 2) * p(i, j - 1) - cf(i, j, 4) * p(i, j);
    else
      L(i, j) = (p(i + 1, j) - 2 * p(i, j) + p(i - 1, j)) * idx2 +
                (p(i, j + 1) - 2 * p(i, j) + p(i, j - 1)) * idy2;
    // FULL block: ADDITIVE cross fluxes (after the diagonal stencil). hxy=hyx=false => +0, bit-identical.
    if (hxy || hyx)
      L(i, j) += cross_div(p, hxy, axy, hyx, ayx, i, j, idx, idy);
    // Helmholtz / screened operator: L phi = div(A grad phi) - kappa phi.
    if (hk)
      L(i, j) -= ka(i, j) * p(i, j);
  }
};

// res = f - L phi on active cells, 0 on conductor cells (poisson_residual).
// hx => FULL tensor: ADDITIVE cross fluxes (cf. ApplyLaplacianKernel). hx=false => bit-identical.
struct PoissonResidualKernel {
  ConstArray4 p, ff;
  Array4 r;
  Real idx2, idy2, idx, idy;
  bool hm;
  ConstArray4 mk;
  bool hc;
  ConstArray4 cf;
  bool he;
  ConstArray4 ep, ey;
  bool hk;
  ConstArray4 ka;
  bool hxy, hyx;
  ConstArray4 axy, ayx;
  ADC_HD void operator()(int i, int j) const {
    if (hm && mk(i, j) == Real(0)) {
      r(i, j) = 0;
      return;
    }
    Real lap;
    if (he) {  // face permittivity (harmonic), with or without cut-cell
      Real wxm, wxp, wym, wyp;
      face_weights(ep, ey, i, j, idx2, idy2, hc, cf, wxm, wxp, wym, wyp);
      lap = wxp * p(i + 1, j) + wxm * p(i - 1, j) + wyp * p(i, j + 1) + wym * p(i, j - 1) -
            (wxm + wxp + wym + wyp) * p(i, j);
    } else if (hc)
      lap = cf(i, j, 1) * p(i + 1, j) + cf(i, j, 0) * p(i - 1, j) + cf(i, j, 3) * p(i, j + 1) +
            cf(i, j, 2) * p(i, j - 1) - cf(i, j, 4) * p(i, j);
    else
      lap = (p(i + 1, j) - 2 * p(i, j) + p(i - 1, j)) * idx2 +
            (p(i, j + 1) - 2 * p(i, j) + p(i, j - 1)) * idy2;
    // FULL block: ADDITIVE cross fluxes (after the diagonal stencil). hxy=hyx=false => +0, bit-identical.
    if (hxy || hyx)
      lap += cross_div(p, hxy, axy, hyx, ayx, i, j, idx, idy);
    // res = f - L phi, L phi = div(A grad phi) - kappa phi = lap - kappa phi.
    r(i, j) = ff(i, j) - lap + (hk ? ka(i, j) * p(i, j) : Real(0));
  }
};
}  // namespace detail

// a_xy/a_yx: off-diagonal coefficients (FULL tensor). nullptr => cross term absent
// (bit-identical diagonal/Poisson operator). Ghosts (1 layer) assumed filled by the caller.
inline void apply_laplacian(const MultiFab& phi, const Geometry& geom, MultiFab& lap,
                            const MultiFab* coef = nullptr, const MultiFab* eps = nullptr,
                            const MultiFab* kappa = nullptr, const MultiFab* eps_y = nullptr,
                            const MultiFab* a_xy = nullptr, const MultiFab* a_yx = nullptr) {
  const Real idx2 = Real(1) / (geom.dx() * geom.dx());
  const Real idy2 = Real(1) / (geom.dy() * geom.dy());
  const Real idx = Real(1) / geom.dx();
  const Real idy = Real(1) / geom.dy();
  for (int li = 0; li < phi.local_size(); ++li) {
    const ConstArray4 p = phi.fab(li).const_array();
    Array4 L = lap.fab(li).array();
    const Box2D v = lap.box(li);
    const bool hc = coef != nullptr;
    const ConstArray4 cf = hc ? coef->fab(li).const_array() : ConstArray4{};
    const bool he = eps != nullptr;
    const ConstArray4 ep = he ? eps->fab(li).const_array() : ConstArray4{};
    // eps_y==nullptr => isotropic: y faces read the same field as the x faces (eps_x).
    const ConstArray4 ey = (he && eps_y) ? eps_y->fab(li).const_array() : ep;
    const bool hk = kappa != nullptr;  // reaction term -kappa phi
    const ConstArray4 ka = hk ? kappa->fab(li).const_array() : ConstArray4{};
    const bool hxy = a_xy != nullptr;  // Axy cross half-term (x faces)
    const bool hyx = a_yx != nullptr;  // Ayx cross half-term (y faces)
    const ConstArray4 axy = hxy ? a_xy->fab(li).const_array() : ConstArray4{};
    const ConstArray4 ayx = hyx ? a_yx->fab(li).const_array() : ConstArray4{};
    for_each_cell(v, detail::ApplyLaplacianKernel{p, L, idx2, idy2, idx, idy, hc, cf, he, ep, ey,
                                                  hk, ka, hxy, hyx, axy, ayx});
  }
}

// res = f - div(A grad phi) on active cells, 0 on conductor cells.
// a_xy/a_yx: off-diagonal coefficients (cf. apply_laplacian). nullptr => bit-identical.
inline void poisson_residual(MultiFab& phi, const MultiFab& f, const Geometry& geom,
                             const BCRec& bc, MultiFab& res, const MultiFab* mask = nullptr,
                             const MultiFab* coef = nullptr, const MultiFab* eps = nullptr,
                             const MultiFab* kappa = nullptr, const MultiFab* eps_y = nullptr,
                             const MultiFab* a_xy = nullptr, const MultiFab* a_yx = nullptr) {
  device_fence();  // GPU: phi may have been written by a kernel (smoother); we
                   // wait before the host read in fill_ghosts.
  fill_ghosts(phi, geom.domain, bc);
  const Real idx2 = Real(1) / (geom.dx() * geom.dx());
  const Real idy2 = Real(1) / (geom.dy() * geom.dy());
  const Real idx = Real(1) / geom.dx();
  const Real idy = Real(1) / geom.dy();
  for (int li = 0; li < phi.local_size(); ++li) {
    const ConstArray4 p = phi.fab(li).const_array();
    const ConstArray4 ff = f.fab(li).const_array();
    Array4 r = res.fab(li).array();
    const Box2D v = res.box(li);
    const bool hm = mask != nullptr;
    const ConstArray4 mk = hm ? mask->fab(li).const_array() : ConstArray4{};
    const bool hc = coef != nullptr;
    const ConstArray4 cf = hc ? coef->fab(li).const_array() : ConstArray4{};
    const bool he = eps != nullptr;
    const ConstArray4 ep = he ? eps->fab(li).const_array() : ConstArray4{};
    // eps_y==nullptr => isotropic: y faces read the same field as the x faces (eps_x).
    const ConstArray4 ey = (he && eps_y) ? eps_y->fab(li).const_array() : ep;
    const bool hk = kappa != nullptr;  // reaction term -kappa phi
    const ConstArray4 ka = hk ? kappa->fab(li).const_array() : ConstArray4{};
    const bool hxy = a_xy != nullptr;  // Axy cross half-term (x faces)
    const bool hyx = a_yx != nullptr;  // Ayx cross half-term (y faces)
    const ConstArray4 axy = hxy ? a_xy->fab(li).const_array() : ConstArray4{};
    const ConstArray4 ayx = hyx ? a_yx->fab(li).const_array() : ConstArray4{};
    for_each_cell(v,
                  detail::PoissonResidualKernel{p,  ff, r,  idx2, idy2, idx, idy, hm,  mk,  hc,
                                                cf, he, ep, ey,   hk,   ka,  hxy, hyx, axy, ayx});
  }
}

namespace detail {
// Red-black Gauss-Seidel smoother on one color (gs_color). p is WRITTEN in place. Body identical to
// the former lambda -> bit-identical. See the comment of the other kernels (#93) for the motivation
// of the named functor.
struct GsColorKernel {
  Array4 p;
  ConstArray4 ff;
  Real idx2, idy2, diag0;
  int color;
  bool hm;
  ConstArray4 mk;
  bool hc;
  ConstArray4 cf;
  bool he;
  ConstArray4 ep, ey;
  bool hk;
  ConstArray4 ka;
  ADC_HD void operator()(int i, int j) const {
    if (((i + j) & 1) != color)
      return;
    if (hm && mk(i, j) == Real(0))
      return;  // conductor: pins phi=0
    Real off, diag;
    if (he) {  // face permittivity (harmonic), with or without cut-cell
      Real wxm, wxp, wym, wyp;
      face_weights(ep, ey, i, j, idx2, idy2, hc, cf, wxm, wxp, wym, wyp);
      off = wxp * p(i + 1, j) + wxm * p(i - 1, j) + wyp * p(i, j + 1) + wym * p(i, j - 1);
      diag = wxm + wxp + wym + wyp;
    } else if (
        hc) {  // cut-cell stencil (Shortley-Weller); conductor neighbor = phi=0 on the circle
      off = cf(i, j, 1) * p(i + 1, j) + cf(i, j, 0) * p(i - 1, j) + cf(i, j, 3) * p(i, j + 1) +
            cf(i, j, 2) * p(i, j - 1);
      diag = cf(i, j, 4);
    } else {
      off = (p(i + 1, j) + p(i - 1, j)) * idx2 + (p(i, j + 1) + p(i, j - 1)) * idy2;
      diag = diag0;
    }
    // Reaction term: the operator becomes div(eps grad phi) - kappa phi, so the
    // diagonal gains +kappa (kappa >= 0 => more diagonally dominant, MG converges better).
    p(i, j) = (off - ff(i, j)) / (diag + (hk ? ka(i, j) : Real(0)));
  }
};

inline void gs_color(MultiFab& phi, const MultiFab& f, const Geometry& geom, int color,
                     const MultiFab* mask, const MultiFab* coef, const MultiFab* eps,
                     const MultiFab* kappa = nullptr, const MultiFab* eps_y = nullptr) {
  const Real idx2 = Real(1) / (geom.dx() * geom.dx());
  const Real idy2 = Real(1) / (geom.dy() * geom.dy());
  const Real diag0 = 2 * idx2 + 2 * idy2;
  for (int li = 0; li < phi.local_size(); ++li) {
    Array4 p = phi.fab(li).array();
    const ConstArray4 ff = f.fab(li).const_array();
    const Box2D v = phi.box(li);
    const bool hm = mask != nullptr;
    const ConstArray4 mk = hm ? mask->fab(li).const_array() : ConstArray4{};
    const bool hc = coef != nullptr;
    const ConstArray4 cf = hc ? coef->fab(li).const_array() : ConstArray4{};
    const bool he = eps != nullptr;
    const ConstArray4 ep = he ? eps->fab(li).const_array() : ConstArray4{};
    // eps_y==nullptr => isotropic: y faces read the same field as the x faces (eps_x).
    const ConstArray4 ey = (he && eps_y) ? eps_y->fab(li).const_array() : ep;
    const bool hk = kappa != nullptr;  // reaction term -kappa phi (Helmholtz / screened)
    const ConstArray4 ka = hk ? kappa->fab(li).const_array() : ConstArray4{};
    for_each_cell(
        v, GsColorKernel{p, ff, idx2, idy2, diag0, color, hm, mk, hc, cf, he, ep, ey, hk, ka});
  }
}
}  // namespace detail

inline void gs_rb_sweep(MultiFab& phi, const MultiFab& f, const Geometry& geom, const BCRec& bc,
                        const MultiFab* mask = nullptr, const MultiFab* coef = nullptr,
                        const MultiFab* eps = nullptr, const MultiFab* kappa = nullptr,
                        const MultiFab* eps_y = nullptr) {
  device_fence();  // wait for the previous kernel before the host read of the halos
  fill_ghosts(phi, geom.domain, bc);
  detail::gs_color(phi, f, geom, 0, mask, coef, eps, kappa, eps_y);  // red (GPU kernel)
  device_fence();  // the black sweep reads the red values via host fill_ghosts
  fill_ghosts(phi, geom.domain, bc);
  detail::gs_color(phi, f, geom, 1, mask, coef, eps, kappa, eps_y);  // black
}

inline void gs_smooth(MultiFab& phi, const MultiFab& f, const Geometry& geom, const BCRec& bc,
                      int nsweeps, const MultiFab* mask = nullptr, const MultiFab* coef = nullptr,
                      const MultiFab* eps = nullptr, const MultiFab* kappa = nullptr,
                      const MultiFab* eps_y = nullptr) {
  for (int s = 0; s < nsweeps; ++s)
    gs_rb_sweep(phi, f, geom, bc, mask, coef, eps, kappa, eps_y);
}

namespace detail {
// Pins phi=0 in the conductor cells (mask==0). Named functor (#93); body identical.
struct ZeroConductorKernel {
  Array4 p;
  ConstArray4 mk;
  ADC_HD void operator()(int i, int j) const {
    if (mk(i, j) == Real(0))
      p(i, j) = 0;
  }
};
}  // namespace detail

// Forces phi=0 in the conductor cells (mask==0).
inline void zero_conductor(MultiFab& phi, const MultiFab& mask) {
  for (int li = 0; li < phi.local_size(); ++li) {
    Array4 p = phi.fab(li).array();
    const ConstArray4 mk = mask.fab(li).const_array();
    const Box2D v = phi.box(li);
    for_each_cell(v, detail::ZeroConductorKernel{p, mk});
  }
}

}  // namespace adc
