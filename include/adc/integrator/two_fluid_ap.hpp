#pragma once

#include <adc/elliptic/elliptic_solver.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>

// Pas deux-fluides isotherme 2D asymptotic-preserving, PORTABLE GPU. Meme schema
// que test_two_fluid_ap_2d_mf, mais tous les kernels passent par for_each_cell avec
// des lambdas ADC_HD (device-callables) : pas de lambda imbriquee, et abs/max via
// ternaires (std::fabs/std::fmax ne sont pas device-safe ; std::sqrt l'est et n'est
// utilise qu'en hote). Le champ est delegue a un EllipticSolver template (PoissonFFT
// en CPU, GeometricMG entierement on-device en GPU) : avec beta0 constant la
// reformulation est juste lap(phi) = (ne* - ni*)/(1+beta0).
//
// Etat par espece : MultiFab a 3 composantes (n, mx, my). z_e = -1, z_i = +1, n0 = 1.

namespace adc {
namespace tfap {
ADC_HD inline Real ab(Real x) { return x < 0 ? -x : x; }       // |x| device-safe
ADC_HD inline Real mx2(Real a, Real b) { return a > b ? a : b; }  // max device-safe
}  // namespace tfap

// m* = m + dt*(-div F_mom) : Rusanov dimensionnellement scinde sur les equations de
// quantite de mouvement (Euler isotherme). Lit n, mx, my avec 1 ghost.
inline void tfap_mstar(const MultiFab& sp, MultiFab& ms, Real c2, Real dt,
                       const Box2D& dom, Real dx, Real dy) {
  ConstArray4 s = sp.fab(0).const_array();
  Array4 m = ms.fab(0).array();
  const Real cs = std::sqrt(c2);
  for_each_cell(dom, [=] ADC_HD(int i, int j) {
    using tfap::ab;
    using tfap::mx2;
    const Real n0 = s(i, j, 0), x0 = s(i, j, 1), y0 = s(i, j, 2);
    const Real nE = s(i + 1, j, 0), xE = s(i + 1, j, 1), yE = s(i + 1, j, 2);
    const Real nW = s(i - 1, j, 0), xW = s(i - 1, j, 1), yW = s(i - 1, j, 2);
    const Real nN = s(i, j + 1, 0), xN = s(i, j + 1, 1), yN = s(i, j + 1, 2);
    const Real nS = s(i, j - 1, 0), xS = s(i, j - 1, 1), yS = s(i, j - 1, 2);
    // interfaces en x (Fxx = mx^2/n + c^2 n, Fyx = mx my / n)
    const Real aR = mx2(ab(x0 / n0) + cs, ab(xE / nE) + cs);
    const Real aL = mx2(ab(xW / nW) + cs, ab(x0 / n0) + cs);
    const Real fxxR = 0.5 * (x0 * x0 / n0 + c2 * n0 + xE * xE / nE + c2 * nE) - 0.5 * aR * (xE - x0);
    const Real fxxL = 0.5 * (xW * xW / nW + c2 * nW + x0 * x0 / n0 + c2 * n0) - 0.5 * aL * (x0 - xW);
    const Real fyxR = 0.5 * (x0 * y0 / n0 + xE * yE / nE) - 0.5 * aR * (yE - y0);
    const Real fyxL = 0.5 * (xW * yW / nW + x0 * y0 / n0) - 0.5 * aL * (y0 - yW);
    // interfaces en y (Fxy = mx my / n, Fyy = my^2/n + c^2 n)
    const Real bU = mx2(ab(y0 / n0) + cs, ab(yN / nN) + cs);
    const Real bD = mx2(ab(yS / nS) + cs, ab(y0 / n0) + cs);
    const Real fxyU = 0.5 * (x0 * y0 / n0 + xN * yN / nN) - 0.5 * bU * (xN - x0);
    const Real fxyD = 0.5 * (xS * yS / nS + x0 * y0 / n0) - 0.5 * bD * (x0 - xS);
    const Real fyyU = 0.5 * (y0 * y0 / n0 + c2 * n0 + yN * yN / nN + c2 * nN) - 0.5 * bU * (yN - y0);
    const Real fyyD = 0.5 * (yS * yS / nS + c2 * nS + y0 * y0 / n0 + c2 * n0) - 0.5 * bD * (y0 - yS);
    m(i, j, 0) = x0 - dt * ((fxxR - fxxL) / dx + (fxyU - fxyD) / dy);
    m(i, j, 1) = y0 - dt * ((fyxR - fyxL) / dx + (fyyU - fyyD) / dy);
  });
}

// ntilde = n - dt div(m), divergence centree (m : mx en comp 0, my en comp 1).
inline void tfap_div_update(const MultiFab& nsrc, const MultiFab& m, MultiFab& nout,
                            Real dt, const Box2D& dom, Real dx, Real dy) {
  ConstArray4 s = nsrc.fab(0).const_array(), mm = m.fab(0).const_array();
  Array4 o = nout.fab(0).array();
  for_each_cell(dom, [=] ADC_HD(int i, int j) {
    const Real d = (mm(i + 1, j, 0) - mm(i - 1, j, 0)) / (2 * dx) +
                   (mm(i, j + 1, 1) - mm(i, j - 1, 1)) / (2 * dy);
    o(i, j, 0) = s(i, j, 0) - dt * d;
  });
}

// E = -grad phi (centre, phi avec 1 ghost) ; Ex en comp 0, Ey en comp 1.
inline void tfap_efield(const MultiFab& phi, MultiFab& E, const Box2D& dom, Real dx,
                        Real dy) {
  ConstArray4 p = phi.fab(0).const_array();
  Array4 e = E.fab(0).array();
  for_each_cell(dom, [=] ADC_HD(int i, int j) {
    e(i, j, 0) = -(p(i + 1, j, 0) - p(i - 1, j, 0)) / (2 * dx);
    e(i, j, 1) = -(p(i, j + 1, 0) - p(i, j - 1, 0)) / (2 * dy);
  });
}

// m^{n+1} = m* + dt z coup E (Lorentz implicite, coefficient n0 = 1).
inline void tfap_lorentz(const MultiFab& ms, const MultiFab& E, MultiFab& mn, Real z,
                         Real coup, Real dt, const Box2D& dom) {
  ConstArray4 m = ms.fab(0).const_array(), ef = E.fab(0).const_array();
  Array4 o = mn.fab(0).array();
  for_each_cell(dom, [=] ADC_HD(int i, int j) {
    o(i, j, 0) = m(i, j, 0) + dt * z * coup * ef(i, j, 0);
    o(i, j, 1) = m(i, j, 1) + dt * z * coup * ef(i, j, 1);
  });
}

// Solveur AP 2D deux-fluides porte MultiFab, template sur l'EllipticSolver.
template <class Elliptic>
struct TwoFluidAP2D {
  static_assert(EllipticSolver<Elliptic>, "Elliptic doit modeler EllipticSolver");

  int n;
  Real L, dx, dy, cse2, csi2, ce, ci;  // ce = wpe^2, ci = wpi^2
  Box2D dom;
  Geometry geom;
  BoxArray ba;
  DistributionMapping dm;
  MultiFab e, ion, mse, msi, nte, nti, Ef, mne, mni;
  Elliptic ell;
  Periodicity per{true, true};

  TwoFluidAP2D(int n_, Real L_, Real cse2_, Real csi2_, Real wpe, Real wpi)
      : n(n_), L(L_), dx(L_ / n_), dy(L_ / n_), cse2(cse2_), csi2(csi2_),
        ce(wpe * wpe), ci(wpi * wpi),
        dom(Box2D::from_extents(n_, n_)),
        geom{dom, 0.0, L_, 0.0, L_},
        ba(BoxArray::from_domain(dom, n_)),
        dm(ba.size(), n_ranks()),
        e(ba, dm, 3, 1), ion(ba, dm, 3, 1),
        mse(ba, dm, 2, 1), msi(ba, dm, 2, 1),
        nte(ba, dm, 1, 0), nti(ba, dm, 1, 0),
        Ef(ba, dm, 2, 0), mne(ba, dm, 2, 1), mni(ba, dm, 2, 1),
        ell(geom, ba, BCRec{}) {}

  void init(Real eps) {
    e.set_val(0);
    ion.set_val(0);
    Array4 ae = e.fab(0).array(), ai = ion.fab(0).array();
    const Real k = 2 * Real(3.14159265358979323846) / L;  // CI en boucle hote (memoire unifiee)
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
        ae(i, j, 0) = Real(1) + eps * std::cos(k * geom.x_cell(i) + k * geom.y_cell(j));
        ai(i, j, 0) = Real(1);
      }
  }

  void step(Real dt, bool stabilize) {
    fill_boundary(e, dom, per);
    fill_boundary(ion, dom, per);
    tfap_mstar(e, mse, cse2, dt, dom, dx, dy);
    tfap_mstar(ion, msi, csi2, dt, dom, dx, dy);
    fill_boundary(mse, dom, per);
    fill_boundary(msi, dom, per);
    tfap_div_update(e, mse, nte, dt, dom, dx, dy);
    tfap_div_update(ion, msi, nti, dt, dom, dx, dy);
    const Real beta0 = stabilize ? dt * dt * (ce + ci) : Real(0);  // n0 = 1
    {  // RHS du Poisson reformule : lap(phi) = (ne* - ni*)/(1+beta0)
      ConstArray4 ne = nte.fab(0).const_array(), ni = nti.fab(0).const_array();
      Array4 r = ell.rhs().fab(0).array();
      const Real inv = Real(1) / (Real(1) + beta0);
      for_each_cell(dom, [=] ADC_HD(int i, int j) {
        r(i, j, 0) = (ne(i, j, 0) - ni(i, j, 0)) * inv;
      });
    }
    ell.solve();
    fill_boundary(ell.phi(), dom, per);
    tfap_efield(ell.phi(), Ef, dom, dx, dy);
    tfap_lorentz(mse, Ef, mne, Real(-1), ce, dt, dom);  // electron
    tfap_lorentz(msi, Ef, mni, Real(+1), ci, dt, dom);  // ion
    fill_boundary(mne, dom, per);
    fill_boundary(mni, dom, per);
    tfap_div_update(e, mne, e, dt, dom, dx, dy);   // n_e^{n+1} (n source = n_e^n)
    tfap_div_update(ion, mni, ion, dt, dom, dx, dy);
    copy_mom(e, mne);  // recopie mx, my finaux dans l'etat
    copy_mom(ion, mni);
  }

  static void copy_mom(MultiFab& sp, const MultiFab& mn) {
    Array4 s = sp.fab(0).array();
    ConstArray4 m = mn.fab(0).const_array();
    for_each_cell(sp.box(0), [=] ADC_HD(int i, int j) {
      s(i, j, 1) = m(i, j, 0);
      s(i, j, 2) = m(i, j, 1);
    });
  }
};

}  // namespace adc
