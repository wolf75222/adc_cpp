// TRANSPORT FLUIDE ISOTHERME en geometrie POLAIRE (Voie A etape 1). Verifie l'operateur
// assemble_rhs_polar sur la brique fluide IsothermalFluxPolar (3 var : rho, rho v_r, rho v_theta) :
//
//   (A) EQUILIBRE ROTATIF (preuve du terme GEOMETRIQUE de courbure). On construit un etat
//       STATIONNAIRE EXACT de la PDE continue : rotation rigide v_theta = Omega r, v_r = 0, et
//       densite isotherme en equilibre cyclostrophique cs2 d_r(ln rho) = v_theta^2/r, soit
//         rho(r) = rho0 exp( Omega^2 (r^2 - r_ref^2) / (2 cs2) ).
//       Avec ce profil, d_t(rho v) = 0 EXACTEMENT (la pression radiale equilibre la force
//       centrifuge). Le residu discret R = assemble_rhs_polar(IsothermalFluxPolar) doit donc
//       CONVERGER VERS 0 a l'ordre du schema. C'EST LA PREUVE que le terme geometrique
//       S_geom = (0, (rho v_theta^2 + p)/r, -(rho v_r v_theta)/r) est correct : sans lui,
//       l'equation radiale donnerait d_t m_r = -(d_r p + p/r) != 0.
//
//   (A') CONTROLE NEGATIF : le MEME etat, le MEME operateur, mais la brique IsothermalFlux SANS
//       terme geometrique (polar_geom_source absent -> retombe sur 0 via PolarHasGeomSource). Le
//       residu radial ne s'annule PAS (il reste O(1), independant de la resolution). Cela isole le
//       role du terme de courbure : la divergence conservative SEULE ne preserve pas l'equilibre.
//
//   (B) CONVERGENCE MMS du systeme fluide complet (3 var) sous une AVANCE en temps SSPRK3 vers un
//       etat stationnaire manufacture (v_r != 0, v_theta != 0, rho et p variables), terme source
//       manufacture S = div_polar(F) - S_geom (closes, stencils d'ordre 4). L'erreur L2 de la
//       solution vs l'exact converge a l'ordre 2 (divergence FV polaire d'ordre 2, cf.
//       test_polar_transport_mms). Confirme que le transport RADIAL + AZIMUTAL des 3 variables,
//       metrique 1/r ET terme geometrique compris, converge proprement.
//
//   (C) CONSERVATION DE LA MASSE : sur une avance SSPRK3 avec PAROI radiale (wall_radial), la masse
//       Sum_ij rho_ij r_i dr dtheta est conservee a ~machine (le terme geometrique n'agit QUE sur
//       la quantite de mouvement, sa composante 0 est nulle -> il ne cree ni ne detruit de masse).
//
// Host / Serial-safe (UNE box, n_ranks()==1 : non enregistre MPI, comme les autres tests polaires).

#include <pops/core/state/state.hpp>
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/numerics/fv/reconstruction.hpp>
#include <pops/numerics/spatial/operators/polar_operator.hpp>
#include <pops/numerics/time/integrators/time_steppers.hpp>
#include <pops/physics/bricks/hyperbolic.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace pops;

static constexpr double kPiL = 3.14159265358979323846;
static constexpr double kRmin = 0.30;
static constexpr double kRmax = 1.00;
static constexpr double kCs2 = 0.7;  // vitesse du son au carre (isotherme p = cs2 rho)

// =====================================================================================
// (A) EQUILIBRE ROTATIF : rotation rigide v_theta = Omega r, v_r = 0, densite cyclostrophique.
// =====================================================================================
static constexpr double kOmega = 0.8;   // vitesse angulaire de la rotation rigide
static constexpr double kRho0 = 1.0;    // densite de reference a r = r_ref
static constexpr double kRref = kRmin;  // rayon de reference

// rho(r) = rho0 exp( Omega^2 (r^2 - r_ref^2) / (2 cs2) ) : equilibre isotherme cyclostrophique
// (cs2 d_r ln rho = Omega^2 r = v_theta^2 / r). Strictement positif, lisse, croissant en r.
static double eq_rho(double r) {
  return kRho0 * std::exp(kOmega * kOmega * (r * r - kRref * kRref) / (2.0 * kCs2));
}
static double eq_vth(double r) {
  return kOmega * r;
}  // rotation rigide

// Remplit U (rho, rho v_r=0, rho v_theta) sur TOUTE la boite (valides + ghosts) avec l'etat exact :
// l'equilibre est stationnaire et a symetrie azimutale, donc les ghosts radiaux/azimutaux sont exacts.
static void fill_equilibrium(MultiFab& U, const PolarGeometry& g) {
  Array4 u = U.fab(0).array();
  const Box2D gb = U.fab(0).grown_box();
  for (int j = gb.lo[1]; j <= gb.hi[1]; ++j)
    for (int i = gb.lo[0]; i <= gb.hi[0]; ++i) {
      const double r = g.r_cell(i);
      const double rho = eq_rho(r);
      u(i, j, 0) = rho;
      u(i, j, 1) = 0.0;              // rho v_r = 0
      u(i, j, 2) = rho * eq_vth(r);  // rho v_theta
    }
}

// Norme Linf du residu radial (composante 1) sur les cellules valides : c'est la quantite qui doit
// converger vers 0 AVEC le terme geometrique et rester O(1) SANS. Ponderation aucune (Linf).
template <class Model>
static double equilibrium_residual_radial(int nr, int nth, const Model& model) {
  Box2D dom = Box2D::from_extents(nr, nth);
  PolarGeometry g{dom, kRmin, kRmax};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, n_ranks());

  const int ng = Weno5::n_ghost;
  MultiFab U(ba, dm, Model::n_vars, ng);
  MultiFab aux(ba, dm, kAuxBaseComps,
               ng);  // aux non utilise par le flux isotherme (pression interne)
  MultiFab R(ba, dm, Model::n_vars, 0);
  U.set_val(0.0);
  aux.set_val(0.0);
  fill_equilibrium(U, g);

  // recon_prim=true : reconstruction en (rho, v_r, v_theta) (positivite). wall_radial=false : on veut
  // le residu interieur PUR (pas de paroi qui annulerait le flux de bord et masquerait la troncature).
  assemble_rhs_polar<Weno5, RusanovFlux>(model, U, aux, g, R, /*recon_prim=*/true,
                                         /*wall_radial=*/false);
  sync_host();
  const ConstArray4 r = R.fab(0).const_array();
  double linf = 0.0;
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
      const double e = std::fabs(r(i, j, 1));  // residu radial (quantite de mouvement v_r)
      if (e > linf)
        linf = e;
    }
  return linf;
}

// =====================================================================================
// (B) MMS du systeme fluide complet (3 var), avance en temps vers un etat stationnaire manufacture.
// =====================================================================================
static constexpr int kMode = 2;       // mode azimutal
static constexpr double kVr = 0.25;   // v_r constant NON NUL (exerce le flux radial)
static constexpr double kVth0 = 0.5;  // base de v_theta
static constexpr double kTfinal = 0.25;

// Solution exacte STATIONNAIRE manufacturee (lisse, strictement positive, periodique en theta) :
//   rho     = (1 + 0.4 sin(pi (r-rmin)/(rmax-rmin))) * (2 + cos(m theta))
//   v_r     = kVr (constant)
//   v_theta = kVth0 * r           (pour que v_theta varie en r ET exerce la courbure croisee)
static double mms_fr(double r) {
  return 1.0 + 0.4 * std::sin(kPiL * (r - kRmin) / (kRmax - kRmin));
}
static double mms_rho(double r, double th) {
  return mms_fr(r) * (2.0 + std::cos(kMode * th));
}
static double mms_vr() {
  return kVr;
}
static double mms_vth(double r) {
  return kVth0 * r;
}
static double mms_p(double r, double th) {
  return kCs2 * mms_rho(r, th);
}

// Composantes de l'etat conservatif exact.
static double mms_U0(double r, double th) {
  return mms_rho(r, th);
}
static double mms_U1(double r, double th) {
  return mms_rho(r, th) * mms_vr();
}
static double mms_U2(double r, double th) {
  return mms_rho(r, th) * mms_vth(r);
}

// Flux physiques (base locale) : F_r = (rho v_r, rho v_r^2 + p, rho v_r v_theta),
//                                F_theta = (rho v_theta, rho v_r v_theta, rho v_theta^2 + p).
static double mms_Fr(int c, double r, double th) {
  const double rho = mms_rho(r, th), vr = mms_vr(), vth = mms_vth(r), p = mms_p(r, th);
  if (c == 0)
    return rho * vr;
  if (c == 1)
    return rho * vr * vr + p;
  return rho * vr * vth;
}
static double mms_Fth(int c, double r, double th) {
  const double rho = mms_rho(r, th), vr = mms_vr(), vth = mms_vth(r), p = mms_p(r, th);
  if (c == 0)
    return rho * vth;
  if (c == 1)
    return rho * vr * vth;
  return rho * vth * vth + p;
}
// Terme geometrique de courbure analytique : S_geom = (0, (rho v_theta^2 + p)/r, -(rho v_r v_theta)/r).
static double mms_Sgeom(int c, double r, double th) {
  const double rho = mms_rho(r, th), vr = mms_vr(), vth = mms_vth(r), p = mms_p(r, th);
  if (c == 0)
    return 0.0;
  if (c == 1)
    return (rho * vth * vth + p) / r;
  return -(rho * vr * vth) / r;
}
// Terme source manufacture S = div_polar(F) - S_geom (pour d_t U = -div_polar F + S_geom + S = 0).
// div_polar(F)[c] = (1/r) d_r(r F_r[c]) + (1/r) d_theta(F_theta[c]), stencils centraux d'ordre 4.
static double mms_source(int c, double r, double th) {
  const double h = 1e-5;
  auto rFr = [c](double rr, double tt) { return rr * mms_Fr(c, rr, tt); };
  const double drFr =
      (-rFr(r + 2 * h, th) + 8 * rFr(r + h, th) - 8 * rFr(r - h, th) + rFr(r - 2 * h, th)) /
      (12 * h);
  const double dthFth = (-mms_Fth(c, r, th + 2 * h) + 8 * mms_Fth(c, r, th + h) -
                         8 * mms_Fth(c, r, th - h) + mms_Fth(c, r, th - 2 * h)) /
                        (12 * h);
  const double divF = (drFr + dthFth) / r;
  return divF - mms_Sgeom(c, r, th);
}

// Brique fluide polaire LOCALE AU TEST : IsothermalFluxPolar (flux + courbure de production) + un
// terme source MANUFACTURE a 3 composantes, injecte SANS code de prod via le canal aux. Le flux
// isotherme n'utilise PAS l'aux (la pression est interne, p = cs2 rho) : les 3 slots de BASE
// (phi=0, grad_x=1, grad_y=2) sont donc LIBRES sur ce chemin. On y range S directement :
//   S[0] (masse)    -> aux[2] (slot grad_y, libre)
//   S[1] (radial)   -> aux[3] (B_z, canal extra, charge par load_aux si n_aux>=4)
//   S[2] (azimutal) -> aux[4] (T_e, canal extra, charge par load_aux si n_aux>=5)
// n_aux = 5 -> load_aux remplit Aux.grad_y, Aux.B_z, Aux.T_e ; source() les relit tels quels. Aucun
// code de production ne fait cela : la brique et le marshaling du source vivent dans ce fichier.
struct MmsFluidPolar : IsothermalFluxPolar {
  static constexpr int n_aux = 5;  // base 0..2 + B_z(3) + T_e(4)
  POPS_HD StateVec<3> source(const StateVec<3>&, const Aux& a) const {
    StateVec<3> s{};
    s[0] = a.grad_y;  // S[0] (masse)    range dans le slot grad_y (libre : flux isotherme sans aux)
    s[1] = a.B_z;     // S[1] (radial)   range dans le canal extra B_z
    s[2] = a.T_e;     // S[2] (azimutal) range dans le canal extra T_e
    return s;
  }
};

static void fill_mms_state(MultiFab& U, const PolarGeometry& g, const Box2D& dom) {
  sync_host();
  Array4 u = U.fab(0).array();
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
      const double r = g.r_cell(i), th = g.theta_cell(j);
      u(i, j, 0) = mms_U0(r, th);
      u(i, j, 1) = mms_U1(r, th);
      u(i, j, 2) = mms_U2(r, th);
    }
}

static void fill_mms_radial_ghosts(MultiFab& U, const PolarGeometry& g, const Box2D& dom) {
  Array4 u = U.fab(0).array();
  const Box2D gb = U.fab(0).grown_box();
  for (int j = gb.lo[1]; j <= gb.hi[1]; ++j) {
    const double th = g.theta_cell(j);
    for (int i = gb.lo[0]; i <= gb.hi[0]; ++i) {
      if (i >= dom.lo[0] && i <= dom.hi[0])
        continue;
      const double r = g.r_cell(i);
      u(i, j, 0) = mms_U0(r, th);
      u(i, j, 1) = mms_U1(r, th);
      u(i, j, 2) = mms_U2(r, th);
    }
  }
}

// aux : S[0]->grad_y(2), S[1]->B_z(3), S[2]->T_e(4) (slots libres, le flux isotherme n'utilise pas aux).
static void fill_mms_aux(MultiFab& aux, const PolarGeometry& g) {
  Array4 a = aux.fab(0).array();
  const Box2D gb = aux.fab(0).grown_box();
  for (int j = gb.lo[1]; j <= gb.hi[1]; ++j)
    for (int i = gb.lo[0]; i <= gb.hi[0]; ++i) {
      const double r = g.r_cell(i), th = g.theta_cell(j);
      a(i, j, 0) = 0.0;                   // phi (inutilise)
      a(i, j, 1) = 0.0;                   // grad_x (inutilise)
      a(i, j, 2) = mms_source(0, r, th);  // S[0] (masse) -> grad_y
      a(i, j, 3) = mms_source(1, r, th);  // S[1] (radial) -> B_z
      a(i, j, 4) = mms_source(2, r, th);  // S[2] (azimutal) -> T_e
    }
}

static double mms_l2(const MultiFab& U, const PolarGeometry& g, const Box2D& dom) {
  sync_host();
  const ConstArray4 u = U.fab(0).const_array();
  const double dr = g.dr(), dth = g.dtheta();
  double num = 0.0, vol = 0.0;
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
      const double rc = g.r_cell(i), th = g.theta_cell(j);
      const double e0 = u(i, j, 0) - mms_U0(rc, th);
      const double e1 = u(i, j, 1) - mms_U1(rc, th);
      const double e2 = u(i, j, 2) - mms_U2(rc, th);
      const double w = rc * dr * dth;
      num += (e0 * e0 + e1 * e1 + e2 * e2) * w;
      vol += w;
    }
  return std::sqrt(num / vol);
}

template <class Limiter>
static double run_mms_fluid(int nr, int nth) {
  Box2D dom = Box2D::from_extents(nr, nth);
  PolarGeometry g{dom, kRmin, kRmax};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, n_ranks());

  BCRec bc;
  bc.xlo = bc.xhi = BCType::Foextrap;  // r : ghosts radiaux ecrases par l'exact (Dirichlet-MMS)
  bc.ylo = bc.yhi = BCType::Periodic;  // theta periodique

  const int ng = Limiter::n_ghost;
  MultiFab U(ba, dm, 3, ng);
  MultiFab aux(ba, dm, 5, ng);  // 5 canaux : phi, grad_x, S2(grad_y), S0(B_z), S1(T_e)
  U.set_val(0.0);
  aux.set_val(0.0);

  MmsFluidPolar model;
  model.cs2 = kCs2;

  fill_mms_aux(aux, g);
  fill_ghosts(aux, dom, bc);  // enroulement azimutal ; radial deja exact (statique)

  fill_mms_state(U, g, dom);

  const double dr = g.dr();
  const double ds_min = std::min(dr, kRmin * g.dtheta());
  // vitesse de signal max ~ |v| + cs ; v_theta max = kVth0 * rmax
  const double vmax = std::max(std::fabs(kVr), kVth0 * kRmax) + std::sqrt(kCs2);
  const double dt = 0.25 * ds_min / vmax;
  const int nsteps = static_cast<int>(std::ceil(kTfinal / dt));
  const double dt_eff = kTfinal / nsteps;

  for (int s = 0; s < nsteps; ++s) {
    SSPRK3Step{}.take_step(
        [&](MultiFab& stage, MultiFab& R) {
          fill_ghosts(stage, dom, bc);
          fill_mms_radial_ghosts(stage, g, dom);
          assemble_rhs_polar<Limiter, RusanovFlux>(model, stage, aux, g, R, /*recon_prim=*/true);
        },
        U, static_cast<Real>(dt_eff));
  }
  return mms_l2(U, g, dom);
}

// =====================================================================================
// (C) CONSERVATION DE LA MASSE sur une avance avec paroi radiale.
// =====================================================================================
static double total_mass(const MultiFab& U, const PolarGeometry& g, const Box2D& dom) {
  sync_host();
  const ConstArray4 u = U.fab(0).const_array();
  const double dr = g.dr(), dth = g.dtheta();
  double m = 0.0;
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
      m += u(i, j, 0) * g.r_cell(i) * dr * dth;
  return m;
}

static double run_mass_conservation() {
  const int nr = 48, nth = 96;
  Box2D dom = Box2D::from_extents(nr, nth);
  PolarGeometry g{dom, kRmin, kRmax};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, n_ranks());

  BCRec bc;
  bc.xlo = bc.xhi = BCType::Foextrap;
  bc.ylo = bc.yhi = BCType::Periodic;

  const int ng = Weno5::n_ghost;
  MultiFab U(ba, dm, 3, ng);
  MultiFab aux(ba, dm, kAuxBaseComps, ng);
  U.set_val(0.0);
  aux.set_val(0.0);

  // Etat non trivial : densite modulee en r et theta, v_r != 0 (poussee vers les parois -> teste que
  // wall_radial annule le flux radial de bord et conserve la masse), v_theta != 0.
  {
    Array4 u = U.fab(0).array();
    const Box2D gb = U.fab(0).box();
    for (int j = gb.lo[1]; j <= gb.hi[1]; ++j)
      for (int i = gb.lo[0]; i <= gb.hi[0]; ++i) {
        const double r = g.r_cell(i), th = g.theta_cell(j);
        const double rho =
            1.0 + 0.3 * std::cos(kMode * th) * std::sin(kPiL * (r - kRmin) / (kRmax - kRmin));
        u(i, j, 0) = rho;
        u(i, j, 1) = rho * 0.2;      // v_r = 0.2 (vers la paroi)
        u(i, j, 2) = rho * 0.3 * r;  // v_theta = 0.3 r
      }
  }

  IsothermalFluxPolar model;
  model.cs2 = kCs2;

  const double m0 = total_mass(U, g, dom);
  const double dr = g.dr();
  const double ds_min = std::min(dr, kRmin * g.dtheta());
  const double vmax = (0.3 * kRmax + 0.2) + std::sqrt(kCs2);
  const double dt = 0.2 * ds_min / vmax;
  const int nsteps = 30;

  for (int s = 0; s < nsteps; ++s) {
    SSPRK3Step{}.take_step(
        [&](MultiFab& stage, MultiFab& R) {
          fill_ghosts(stage, dom, bc);
          // wall_radial=true : paroi solide aux 2 bords -> flux radial nul -> masse conservee a la machine.
          assemble_rhs_polar<Weno5, RusanovFlux>(model, stage, aux, g, R, /*recon_prim=*/true,
                                                 /*wall_radial=*/true);
        },
        U, dt);
  }
  const double m1 = total_mass(U, g, dom);
  const double rel = std::fabs(m1 - m0) / std::fabs(m0);
  std::printf("[masse] initiale=%.15e finale=%.15e  ecart relatif=%.3e (K=%d pas)\n", m0, m1, rel,
              nsteps);
  return rel;
}

int main() {
  std::printf("=== TRANSPORT FLUIDE ISOTHERME POLAIRE (Voie A etape 1) ===\n");
  std::printf("Anneau r in [%.2f, %.2f], theta in [0, 2pi), cs2=%.2f\n", kRmin, kRmax, kCs2);
  bool ok = true;

  // --- (A) Equilibre rotatif : residu radial -> 0 AVEC le terme geometrique ---------------------
  std::printf("\n--- (A) Equilibre rotatif (rotation rigide Omega=%.2f, rho cyclostrophique) ---\n",
              kOmega);
  std::printf(
      "    AVEC terme geometrique de courbure (IsothermalFluxPolar) : residu radial -> 0\n");
  const int res[3] = {48, 96, 192};
  IsothermalFluxPolar fp;
  fp.cs2 = kCs2;
  double eq[3];
  for (int k = 0; k < 3; ++k) {
    eq[k] = equilibrium_residual_radial(res[k], 2 * res[k], fp);
    std::printf("    nr=%-4d nth=%-4d : |R_radial|_inf = %.6e\n", res[k], 2 * res[k], eq[k]);
  }
  const double peq1 = std::log2(eq[0] / eq[1]);
  const double peq2 = std::log2(eq[1] / eq[2]);
  std::printf("    ordre observe (Linf) : %.2f (48->96), %.2f (96->192)\n", peq1, peq2);
  // L'equilibre exact rend le residu -> 0 a l'ordre de l'operateur (FV polaire ~ ordre 2). On exige
  // une DECROISSANCE nette (ordre >= 1.5) ET un residu fin petit : preuve que le terme geometrique
  // equilibre la divergence de pression radiale.
  if (!(peq1 >= 1.5) || !(peq2 >= 1.5)) {
    std::printf(
        "    ECHEC : le residu ne decroit pas a l'ordre attendu (terme geometrique faux ?)\n");
    ok = false;
  } else {
    std::printf("    OK : residu radial -> 0 (ordre >= 1.5) => terme geometrique correct\n");
  }

  // --- (A') Controle negatif : MEME etat, brique SANS terme geometrique -> residu O(1) ----------
  std::printf("\n--- (A') Controle negatif (IsothermalFlux SANS terme geometrique) ---\n");
  IsothermalFlux f_nogeom;
  f_nogeom.cs2 = kCs2;
  double ng_res[2];
  ng_res[0] = equilibrium_residual_radial(48, 96, f_nogeom);
  ng_res[1] = equilibrium_residual_radial(192, 384, f_nogeom);
  std::printf("    nr=48  : |R_radial|_inf = %.6e\n", ng_res[0]);
  std::printf("    nr=192 : |R_radial|_inf = %.6e\n", ng_res[1]);
  // Sans le terme geometrique le residu est domine par rho v_theta^2/r (O(1)), il NE decroit PAS avec
  // la resolution : on exige qu'il reste GRAND (>> le residu AVEC geometrie a meme resolution).
  if (!(ng_res[1] > 0.1) || !(ng_res[1] > 100.0 * eq[2])) {
    std::printf("    ECHEC : le residu sans geometrie devrait rester O(1) et >> le residu avec\n");
    ok = false;
  } else {
    std::printf(
        "    OK : residu O(1) non convergent sans terme geometrique (>> %.2e avec) => le terme\n"
        "         geometrique est INDISPENSABLE (la divergence conservative seule ne suffit pas)\n",
        eq[2]);
  }

  // --- (B) MMS du systeme fluide complet (3 var) : ordre 2 ---------------------------------------
  std::printf(
      "\n--- (B) MMS fluide complet 3 var (v_r != 0, v_theta != 0) : ordre 2 attendu ---\n");
  double e[3];
  for (int k = 0; k < 3; ++k) {
    e[k] = run_mms_fluid<Weno5>(res[k], 2 * res[k]);
    std::printf("    WENO5  nr=%-4d nth=%-4d : L2 = %.6e\n", res[k], 2 * res[k], e[k]);
  }
  const double p1 = std::log2(e[0] / e[1]);
  const double p2 = std::log2(e[1] / e[2]);
  std::printf("    ordre observe WENO5 (L2) : %.2f (48->96), %.2f (96->192)\n", p1, p2);
  const double kSeuil = 1.7;
  if (!(p1 >= kSeuil) || !(p2 >= kSeuil) || !std::isfinite(e[2])) {
    std::printf("    ECHEC : ordre < %.1f (transport fluide polaire non convergent)\n", kSeuil);
    ok = false;
  } else {
    std::printf(
        "    OK : convergence d'ordre >= %.1f (transport fluide 3 var + courbure correct)\n",
        kSeuil);
  }

  // --- (C) Conservation de la masse (paroi radiale) ---------------------------------------------
  std::printf("\n--- (C) Conservation de la masse (paroi radiale, v_r != 0) ---\n");
  const double rel = run_mass_conservation();
  if (rel > 1e-12) {
    std::printf("    ECHEC : ecart de masse %.3e > 1e-12\n", rel);
    ok = false;
  } else {
    std::printf("    OK : masse conservee a ~machine (%.3e <= 1e-12)\n", rel);
  }

  std::printf("\n=== VERDICT : %s ===\n", ok ? "SUCCESS" : "ECHEC");
  if (ok)
    std::printf("OK test_polar_fluid_transport\n");
  return ok ? 0 : 1;
}
