// ETAGE SOURCE condense par Schur (CondensedSchurSourceStepper, niveau 4 de
// docs/SCHUR_CONDENSATION_DESIGN.md) : test SOURCE-ONLY (transport gele) d'un fluide magnetise sous la
// source implicite couplee potentiel / vitesse / Lorentz (Hoffart et al., arXiv:2510.11808). Cf.
// include/pops/coupling/condensed_schur_source_stepper.hpp.
//
// SYSTEME SOURCE (transport gele), constant en espace (rho = rho0, B_z = B0, alpha) :
//   d_t v   = -grad phi + v x Omega   (Omega porte par B_z ; (v x Omega)_x = +B0 v_y, _y = -B0 v_x)
//   d_t(-Lap phi) = -alpha rho0 div v
// Discretisation SPATIALE : gradient et divergence CENTRES d'ordre 2, Laplacien 5 points (memes
// stencils que le batisseur #124 et field_postprocess), CL Dirichlet (phi = 0 au bord -> -Lap
// inversible, pas de noyau). C'est un systeme d'EDO lineaire raide : la frequence cyclotron B0 et le
// couplage plasma alpha rho0 fixent un pas explicite maximal. Au-dela, l'Euler explicite EXPLOSE ;
// l'etage de Schur (theta-schema implicite, theta = 1 = Euler retrograde) reste STABLE sans condition.
//
// On valide TROIS choses :
//   (A) RELATION IMPLICITE reconstruite : apres step(), v^{n+1} satisfait B v^{n+1} = v^n - dt grad
//       phi^{n+1} (B = I - dt [Omega], grad CENTRE), a la tolerance du SOLVE (1e-8). C'est la verif
//       EXACTE que la reconstruction v = B^{-1}(v^n - dt grad phi) est coherente avec le phi resolu.
//   (B) STABILITE vs EXPLOSION explicite, a GRAND dt : on prend dt = 50 x le pas explicite stable
//       estime. L'Euler explicite (un pas) fait EXPLOSER la norme de v (||v|| croit de plusieurs
//       ordres) ; l'etage de Schur garde ||v|| BORNEE (du meme ordre que l'etat initial). On compare
//       quantitativement les deux normes finales.
//   (C) ACCORD AVEC UNE REFERENCE fine-dt : a un dt MODERE (sous le pas explicite stable), on integre
//       le MEME systeme discret par RK4 a pas fin (N sous-pas) -> reference de l'EDO discrete. L'etage
//       de Schur (theta = 1, Euler retrograde, ordre 1 en temps) doit en etre proche : ecart relatif
//       qui DECROIT a l'ordre 1 quand dt diminue (dt, dt/2 -> ratio ~ 2). C'est le garde-fou que
//       l'etage resout le BON systeme (pas seulement "stable").
//
// MPI : binaire rejoue a np=1/2/4 (CMake). Le solve de Krylov est COLLECTIF (dot/all_reduce sur tous
// les rangs, y compris vides) : iterations et resultat invariants au nombre de rangs. Les verifs
// (ecarts MAX, normes) sont reduites par all_reduce_max : un FAIL sur un rang -> FAIL partout.

#include <pops/coupling/schur/source/condensed_schur_source_stepper.hpp>

#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/numerics/linalg/lorentz_eliminator.hpp>
#include <pops/parallel/comm.hpp>

#include <cmath>
#include <cstdio>

using namespace pops;
static constexpr double kPi = 3.14159265358979323846;

// ----------------------------------------------------------------------------------------------
// Outils de test (hote ; foncteurs nommes pour les kernels device-clean).
// ----------------------------------------------------------------------------------------------

// VariableSet d'un fluide minimal : rho, mx, my (+ energie optionnelle si with_E).
static VariableSet fluid_vars(bool with_E) {
  VariableSet vs;
  vs.kind = VariableKind::Conservative;
  if (with_E) {
    vs.names = {"rho", "mx", "my", "E"};
    vs.roles = {VariableRole::Density, VariableRole::MomentumX, VariableRole::MomentumY,
                VariableRole::Energy};
    vs.size = 4;
  } else {
    vs.names = {"rho", "mx", "my"};
    vs.roles = {VariableRole::Density, VariableRole::MomentumX, VariableRole::MomentumY};
    vs.size = 3;
  }
  return vs;
}

struct Setup {
  Box2D dom;
  Geometry geom;
  BoxArray ba;
  DistributionMapping dm;
  BCRec bc;
  Setup(int n)
      : dom(Box2D::from_extents(n, n)),
        geom{dom, 0.0, 1.0, 0.0, 1.0},
        ba(BoxArray::from_domain(dom, n)),
        dm(ba.size(), n_ranks()) {
    bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;  // phi = 0 au bord -> -Lap inversible
  }
};

// Initialise l'etat (rho0, mom = rho0 v0) et phi0 a partir de profils analytiques lisses, nuls au
// bord (compatibles Dirichlet). vx0/vy0/phi0 sont des sin(pi x) sin(pi y) (et derives) -> reguliers.
struct InitKernel {
  Geometry geom;
  Array4 st, phi;
  Real rho0;
  int c_rho, c_mx, c_my, c_E;  // c_E < 0 si pas d'energie
  POPS_HD void operator()(int i, int j) const {
    const Real x = geom.x_cell(i), y = geom.y_cell(j);
    const Real sx = std::sin(Real(kPi) * x), sy = std::sin(Real(kPi) * y);
    const Real vx = Real(0.6) * sx * sy;                            // v0_x
    const Real vy = Real(-0.4) * std::sin(Real(2 * kPi) * x) * sy;  // v0_y
    const Real ph = Real(0.3) * sx * sy;                            // phi0
    st(i, j, c_rho) = rho0;
    st(i, j, c_mx) = rho0 * vx;
    st(i, j, c_my) = rho0 * vy;
    if (c_E >= 0)
      st(i, j, c_E) = Real(1.0) + Real(0.5) * rho0 * (vx * vx + vy * vy);  // E = e_int + KE
    phi(i, j, 0) = ph;
  }
};

// Champ B_z constant (canal aux scalaire) : un seul champ, composante 0.
struct ConstBzKernel {
  Array4 bz;
  Real B0;
  POPS_HD void operator()(int i, int j) const { bz(i, j, 0) = B0; }
};

// norme L2 GLOBALE (sqrt sum x^2) de la VITESSE (mx,my)/rho de l'etat : diagnostic de stabilite.
static double vel_l2(const MultiFab& st, int c_rho, int c_mx, int c_my) {
  // st a pu etre ecrit par un kernel device (make_state / step) : rendre la residence hote valide
  // avant la lecture directe (Kokkos::Cuda = device_fence ; no-op en serie/OpenMP). Sans cela on lit
  // st pendant que le kernel est en vol (memoire unifiee non ordonnee) -> valeurs indefinies (NaN).
  sync_host();
  double s = 0;
  for (int li = 0; li < st.local_size(); ++li) {
    const ConstArray4 u = st.fab(li).const_array();
    const Box2D b = st.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        const double rho = u(i, j, c_rho);
        const double vx = u(i, j, c_mx) / rho, vy = u(i, j, c_my) / rho;
        s += vx * vx + vy * vy;
      }
  }
  return std::sqrt(all_reduce_sum(s));
}

// ecart L2 GLOBAL relatif des VITESSES entre deux etats (sur les cellules valides).
static double vel_rel_diff(const MultiFab& a, const MultiFab& b, int c_rho, int c_mx, int c_my) {
  sync_host();  // a/b ecrits par des kernels device : residence hote valide avant la lecture directe.
  double num = 0, den = 0;
  for (int li = 0; li < a.local_size(); ++li) {
    const ConstArray4 ua = a.fab(li).const_array();
    const ConstArray4 ub = b.fab(li).const_array();
    const Box2D bx = a.box(li);
    for (int j = bx.lo[1]; j <= bx.hi[1]; ++j)
      for (int i = bx.lo[0]; i <= bx.hi[0]; ++i) {
        const double ra = ua(i, j, c_rho), rb = ub(i, j, c_rho);
        const double ax = ua(i, j, c_mx) / ra, ay = ua(i, j, c_my) / ra;
        const double bx2 = ub(i, j, c_mx) / rb, by2 = ub(i, j, c_my) / rb;
        num += (ax - bx2) * (ax - bx2) + (ay - by2) * (ay - by2);
        den += bx2 * bx2 + by2 * by2;
      }
  }
  num = all_reduce_sum(num);
  den = all_reduce_sum(den);
  return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

// ----------------------------------------------------------------------------------------------
// (A) Verif de la RELATION IMPLICITE : ecart MAX |B v^{n+1} - (v^n - dt grad phi^{n+1})|.
// grad phi CENTRE (meme stencil que la reconstruction). v^n est l'etat AVANT step (passe a part).
// ----------------------------------------------------------------------------------------------
static double implicit_residual(const MultiFab& st_new, const MultiFab& vxn, const MultiFab& vyn,
                                MultiFab& phi_new, const Geometry& geom, const BCRec& bc, Real B0,
                                Real dt, int c_rho, int c_mx, int c_my) {
  device_fence();
  fill_ghosts(phi_new, geom.domain, bc);  // grad centre lit phi(i+-1), phi(j+-1)
  device_fence();  // fill_ghosts est un kernel device : on attend avant la lecture hote des ghosts.
  const Real half_idx = Real(1) / (Real(2) * geom.dx());
  const Real half_idy = Real(1) / (Real(2) * geom.dy());
  double d = 0;
  for (int li = 0; li < st_new.local_size(); ++li) {
    const ConstArray4 u = st_new.fab(li).const_array();
    const ConstArray4 vx = vxn.fab(li).const_array();
    const ConstArray4 vy = vyn.fab(li).const_array();
    const ConstArray4 p = phi_new.fab(li).const_array();
    const Box2D b = st_new.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        const Real rho = u(i, j, c_rho);
        const Real vnx = u(i, j, c_mx) / rho, vny = u(i, j, c_my) / rho;  // v^{n+1}
        // B v^{n+1} : B = I - dt [Omega], (B v)_x = vx - dt B0 vy, (B v)_y = vy + dt B0 vx.
        const Real Bvx = vnx - dt * B0 * vny;
        const Real Bvy = vny + dt * B0 * vnx;
        const Real gx = (p(i + 1, j) - p(i - 1, j)) * half_idx;
        const Real gy = (p(i, j + 1) - p(i, j - 1)) * half_idy;
        const Real rhsx = vx(i, j, 0) - dt * gx;  // (v^n - dt grad phi^{n+1})_x
        const Real rhsy = vy(i, j, 0) - dt * gy;
        d = std::fmax(d, std::fmax(std::fabs(Bvx - rhsx), std::fabs(Bvy - rhsy)));
      }
  }
  return all_reduce_max(d);
}

// ----------------------------------------------------------------------------------------------
// REFERENCE fine-dt (C) et EXPLICITE (B) : integration du MEME systeme discret.
// Etat de reference porte v (vx, vy) et phi separement. La derivee de phi exige (-Lap)^{-1} :
//   d_t phi = (-Lap)^{-1}(-alpha rho0 div v)  <=>  -Lap(phi_dot) = -alpha rho0 div v
//   soit, convention GeometricMG L_int = Lap : Lap(phi_dot) = alpha rho0 div v -> rhs = alpha rho0 div v.
// On reutilise GeometricMG (Poisson canonique, Dirichlet) pour ce solve par etage.
// ----------------------------------------------------------------------------------------------

// div v CENTRE -> rhs(i,j) = alpha rho0 div v (pour le solve Lap(phi_dot) = rhs).
struct DivVKernel {
  ConstArray4 vx, vy;
  Array4 rhs;
  Real coef;  // alpha rho0
  Real half_idx, half_idy;
  POPS_HD void operator()(int i, int j) const {
    const Real divv = (vx(i + 1, j, 0) - vx(i - 1, j, 0)) * half_idx +
                      (vy(i, j + 1, 0) - vy(i, j - 1, 0)) * half_idy;
    rhs(i, j, 0) = coef * divv;
  }
};

// derivee de v : dvx = -grad_x phi + B0 vy ; dvy = -grad_y phi - B0 vx (grad CENTRE).
struct DvKernel {
  ConstArray4 vx, vy, phi;
  Array4 dvx, dvy;
  Real B0, half_idx, half_idy;
  POPS_HD void operator()(int i, int j) const {
    const Real gx = (phi(i + 1, j, 0) - phi(i - 1, j, 0)) * half_idx;
    const Real gy = (phi(i, j + 1, 0) - phi(i, j - 1, 0)) * half_idy;
    dvx(i, j, 0) = -gx + B0 * vy(i, j, 0);
    dvy(i, j, 0) = -gy - B0 * vx(i, j, 0);
  }
};

// y <- y + a x (composante 0, cellules valides). Foncteur nomme.
struct AxpyKernel {
  Array4 y;
  ConstArray4 x;
  Real a;
  POPS_HD void operator()(int i, int j) const { y(i, j, 0) += a * x(i, j, 0); }
};
// z <- base + a x (composante 0). Foncteur nomme.
struct ScaleAddKernel {
  Array4 z;
  ConstArray4 base, x;
  Real a;
  POPS_HD void operator()(int i, int j) const { z(i, j, 0) = base(i, j, 0) + a * x(i, j, 0); }
};

// Integrateur de reference : porte (vx, vy, phi), avance par RK4 a pas dt sur le systeme discret.
// phi_dot resolu par Poisson (GeometricMG) a chaque evaluation de la derivee. Sert a la fois pour la
// reference fine (N grand) et pour un pas explicite unique grossier (via euler_step).
struct RefIntegrator {
  Setup& S;
  Real B0, alpha, rho0;
  Real half_idx, half_idy;
  GeometricMG poisson;  // Lap(phi_dot) = alpha rho0 div v (Dirichlet, canonique)
  MultiFab vx, vy, phi;
  // tampons de derivee
  MultiFab dvx, dvy, dphi;
  // tampons RK4
  MultiFab kvx[4], kvy[4], kphi[4], tvx, tvy, tphi;

  RefIntegrator(Setup& s, Real B0_, Real alpha_, Real rho0_)
      : S(s),
        B0(B0_),
        alpha(alpha_),
        rho0(rho0_),
        half_idx(Real(1) / (Real(2) * s.geom.dx())),
        half_idy(Real(1) / (Real(2) * s.geom.dy())),
        poisson(s.geom, s.ba, s.bc),
        vx(s.ba, s.dm, 1, 1),
        vy(s.ba, s.dm, 1, 1),
        phi(s.ba, s.dm, 1, 1),
        dvx(s.ba, s.dm, 1, 0),
        dvy(s.ba, s.dm, 1, 0),
        dphi(s.ba, s.dm, 1, 1),
        tvx(s.ba, s.dm, 1, 1),
        tvy(s.ba, s.dm, 1, 1),
        tphi(s.ba, s.dm, 1, 1) {
    for (int k = 0; k < 4; ++k) {
      kvx[k] = MultiFab(s.ba, s.dm, 1, 0);
      kvy[k] = MultiFab(s.ba, s.dm, 1, 0);
      kphi[k] = MultiFab(s.ba, s.dm, 1, 0);
    }
  }

  // derivee (dvx, dvy, dphi) au point (avx, avy, aphi). Resout Lap(dphi) = alpha rho0 div v.
  void deriv(MultiFab& avx, MultiFab& avy, MultiFab& aphi, MultiFab& odvx, MultiFab& odvy,
             MultiFab& odphi) {
    device_fence();
    fill_ghosts(avx, S.geom.domain, S.bc);
    fill_ghosts(avy, S.geom.domain, S.bc);
    fill_ghosts(aphi, S.geom.domain, S.bc);
    // dv
    for (int li = 0; li < avx.local_size(); ++li)
      for_each_cell(odvx.box(li), DvKernel{avx.fab(li).const_array(), avy.fab(li).const_array(),
                                           aphi.fab(li).const_array(), odvx.fab(li).array(),
                                           odvy.fab(li).array(), B0, half_idx, half_idy});
    // dphi : rhs = alpha rho0 div v, solve Lap(dphi) = rhs (Dirichlet, dphi=0 au bord)
    for (int li = 0; li < poisson.rhs().local_size(); ++li)
      for_each_cell(poisson.rhs().box(li),
                    DivVKernel{avx.fab(li).const_array(), avy.fab(li).const_array(),
                               poisson.rhs().fab(li).array(), alpha * rho0, half_idx, half_idy});
    poisson.phi().set_val(Real(0));
    poisson.solve(Real(1e-12), 200);
    for (int li = 0; li < odphi.local_size(); ++li)
      for_each_cell(odphi.box(li), detail::CopyComp0Kernel{odphi.fab(li).array(),
                                                           poisson.phi().fab(li).const_array()});
  }

  // un pas RK4 de taille h (in-place sur vx, vy, phi).
  void rk4_step(Real h) {
    deriv(vx, vy, phi, kvx[0], kvy[0], kphi[0]);
    stage(h * Real(0.5), kvx[0], kvy[0], kphi[0]);
    deriv(tvx, tvy, tphi, kvx[1], kvy[1], kphi[1]);
    stage(h * Real(0.5), kvx[1], kvy[1], kphi[1]);
    deriv(tvx, tvy, tphi, kvx[2], kvy[2], kphi[2]);
    stage(h, kvx[2], kvy[2], kphi[2]);
    deriv(tvx, tvy, tphi, kvx[3], kvy[3], kphi[3]);
    // y <- y + h/6 (k0 + 2 k1 + 2 k2 + k3)
    combine(vx, kvx, h);
    combine(vy, kvy, h);
    combine(phi, kphi, h);
  }

  // un pas EULER EXPLICITE de taille h (in-place) : y <- y + h f(y). Sert au cas (B) instable.
  void euler_step(Real h) {
    deriv(vx, vy, phi, kvx[0], kvy[0], kphi[0]);
    axpy(vx, h, kvx[0]);
    axpy(vy, h, kvy[0]);
    axpy(phi, h, kphi[0]);
  }

 private:
  // tvx/tvy/tphi <- (vx,vy,phi) + a (kx,ky,kphi)
  void stage(Real a, MultiFab& kx, MultiFab& ky, MultiFab& kp) {
    for (int li = 0; li < tvx.local_size(); ++li) {
      for_each_cell(tvx.box(li), ScaleAddKernel{tvx.fab(li).array(), vx.fab(li).const_array(),
                                                kx.fab(li).const_array(), a});
      for_each_cell(tvy.box(li), ScaleAddKernel{tvy.fab(li).array(), vy.fab(li).const_array(),
                                                ky.fab(li).const_array(), a});
      for_each_cell(tphi.box(li), ScaleAddKernel{tphi.fab(li).array(), phi.fab(li).const_array(),
                                                 kp.fab(li).const_array(), a});
    }
  }
  void axpy(MultiFab& y, Real a, MultiFab& x) {
    for (int li = 0; li < y.local_size(); ++li)
      for_each_cell(y.box(li), AxpyKernel{y.fab(li).array(), x.fab(li).const_array(), a});
  }
  // y <- y + h/6 (k0 + 2 k1 + 2 k2 + k3)
  void combine(MultiFab& y, MultiFab (&k)[4], Real h) {
    const Real s = h / Real(6);
    sync_host();  // k0..k3 ecrits par derivative (kernels device) : residence hote valide avant lecture.
    for (int li = 0; li < y.local_size(); ++li) {
      Array4 Y = y.fab(li).array();
      const ConstArray4 k0 = k[0].fab(li).const_array(), k1 = k[1].fab(li).const_array(),
                        k2 = k[2].fab(li).const_array(), k3 = k[3].fab(li).const_array();
      const Box2D b = y.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          Y(i, j, 0) +=
              s * (k0(i, j, 0) + Real(2) * k1(i, j, 0) + Real(2) * k2(i, j, 0) + k3(i, j, 0));
    }
  }
};

// Copie l'etat fluide (mom/rho -> v) vers (vx, vy, phi) de l'integrateur de reference, + phi0.
static void load_ref_from_state(RefIntegrator& R, const MultiFab& st, const MultiFab& phi0,
                                int c_rho, int c_mx, int c_my) {
  sync_host();  // st/phi0 ecrits par make_state (kernels device) : residence hote valide avant lecture.
  for (int li = 0; li < st.local_size(); ++li) {
    const ConstArray4 u = st.fab(li).const_array();
    Array4 vx = R.vx.fab(li).array(), vy = R.vy.fab(li).array(), p = R.phi.fab(li).array();
    const ConstArray4 p0 = phi0.fab(li).const_array();
    const Box2D b = st.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        const Real rho = u(i, j, c_rho);
        vx(i, j, 0) = u(i, j, c_mx) / rho;
        vy(i, j, 0) = u(i, j, c_my) / rho;
        p(i, j, 0) = p0(i, j, 0);
      }
  }
}

// ecart relatif des vitesses entre un etat fluide (st) et les champs vx,vy d'un integrateur de ref.
static double vel_rel_state_vs_ref(const MultiFab& st, const RefIntegrator& R, int c_rho, int c_mx,
                                   int c_my) {
  sync_host();  // st/R.vx/R.vy ecrits par des kernels device : residence hote valide avant lecture.
  double num = 0, den = 0;
  for (int li = 0; li < st.local_size(); ++li) {
    const ConstArray4 u = st.fab(li).const_array();
    const ConstArray4 rx = R.vx.fab(li).const_array(), ry = R.vy.fab(li).const_array();
    const Box2D b = st.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        const double rho = u(i, j, c_rho);
        const double ax = u(i, j, c_mx) / rho, ay = u(i, j, c_my) / rho;
        const double bx = rx(i, j, 0), by = ry(i, j, 0);
        num += (ax - bx) * (ax - bx) + (ay - by) * (ay - by);
        den += bx * bx + by * by;
      }
  }
  num = all_reduce_sum(num);
  den = all_reduce_sum(den);
  return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank(), np = n_ranks();
  long fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      if (me == 0)
        std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int n = 32;
  const Real rho0 = 1.5, B0 = 4.0, alpha = 3.0;
  // Pas explicite stable estime : la raideur vient de la cyclotron (B0) et du plasma
  // (sqrt(alpha rho0) x |k|_max ~ alpha rho0 / dx). dt_explicite ~ 1 / max(B0, sqrt(alpha rho0)/dx).
  // n=32 -> dx ~ 1/32. On prend une estimation conservative et on s'en sert pour CALER les regimes.
  const Setup S(n);
  const double dx = static_cast<double>(S.geom.dx());
  const double w_plasma = std::sqrt(static_cast<double>(alpha) * static_cast<double>(rho0)) *
                          (kPi * std::sqrt(2.0));  // |grad| max d'un mode ~ pi sqrt2 (continu)
  const double w_max = std::max(static_cast<double>(B0), w_plasma);
  const double dt_stable = 0.5 / w_max;  // estimation prudente du pas explicite stable
  if (me == 0)
    std::printf("[setup] n=%d dx=%.4f B0=%.1f alpha=%.1f rho0=%.1f -> dt_stable~%.4e\n", n, dx,
                static_cast<double>(B0), static_cast<double>(alpha), static_cast<double>(rho0),
                dt_stable);

  const VariableSet vars = fluid_vars(/*with_E=*/true);
  const int c_rho = vars.index_of(VariableRole::Density);
  const int c_mx = vars.index_of(VariableRole::MomentumX);
  const int c_my = vars.index_of(VariableRole::MomentumY);

  // Etat initial + phi0 + B_z, partages par tous les sous-tests (recharges a chaque fois).
  auto make_state = [&](MultiFab& st, MultiFab& phi) {
    for (int li = 0; li < st.local_size(); ++li)
      for_each_cell(st.box(li), InitKernel{S.geom, st.fab(li).array(), phi.fab(li).array(), rho0,
                                           c_rho, c_mx, c_my, vars.index_of(VariableRole::Energy)});
  };
  MultiFab bz(S.ba, S.dm, 1, 1);
  for (int li = 0; li < bz.local_size(); ++li)
    for_each_cell(bz.box(li), ConstBzKernel{bz.fab(li).array(), B0});

  // ----------------------------------------------------------------------------------------------
  // (A) RELATION IMPLICITE a GRAND dt : B v^{n+1} = v^n - dt grad phi^{n+1} a la tolerance du solve.
  // ----------------------------------------------------------------------------------------------
  {
    const Real dt = Real(50.0 * dt_stable);  // grand pas (regime ou l'explicite explose)
    MultiFab st(S.ba, S.dm, vars.size, 1), phi(S.ba, S.dm, 1, 1);
    make_state(st, phi);
    // v^n fige pour la verif.
    MultiFab vxn(S.ba, S.dm, 1, 0), vyn(S.ba, S.dm, 1, 0);
    for (int li = 0; li < st.local_size(); ++li)
      for_each_cell(st.box(li),
                    detail::ExtractVelocityKernel{st.fab(li).const_array(), vxn.fab(li).array(),
                                                  vyn.fab(li).array(), c_rho, c_mx, c_my});

    CondensedSchurSourceStepper stepper(vars, S.geom, S.ba, S.bc, alpha);
    stepper.step(st, phi, bz, /*c_bz=*/0, /*theta=*/Real(1.0), dt);
    const KrylovResult kr = stepper.last_solve();

    // ATTENTION : step() a EXTRAPOLE (theta=1 -> identite, donc phi/v sont deja a n+1) et l'etat
    // porte v^{n+1}, phi^{n+1}. La relation implicite a verifier est sur le theta-stage = n+1 ici.
    const double rimp =
        implicit_residual(st, vxn, vyn, phi, S.geom, S.bc, B0, dt, c_rho, c_mx, c_my);
    if (me == 0)
      std::printf(
          "(A) implicite : BiCGStab %s en %d iters (rel=%.2e) | max|B v - (v^n - dt grad phi)| = "
          "%.3e\n",
          kr.converged ? "CONVERGE" : "ECHOUE", kr.iters, static_cast<double>(kr.rel_residual),
          rimp);
    chk(kr.converged, "A_solve_converge");
    chk(rimp < 1e-7, "A_relation_implicite");  // tolerance solve 1e-10 propagee par grad -> marge
  }

  // ----------------------------------------------------------------------------------------------
  // (B) STABILITE vs EXPLOSION explicite a GRAND dt, sur K PAS.
  //   Un seul pas explicite ne fait qu'amplifier par ~omega dt (la raideur cyclotron / plasma est
  //   OSCILLATOIRE : |1 + i omega dt| = sqrt(1 + (omega dt)^2)). L'INSTABILITE explicite se voit
  //   sur la COMPOSITION : a dt > dt_stable, chaque pas explicite amplifie par g > 1 et g^K EXPLOSE
  //   geometriquement. On joue donc K pas de taille dt pour les DEUX schemas :
  //   - Euler explicite (K pas) : ||v|| EXPLOSE (g^K, plusieurs ordres) ;
  //   - Schur (K pas, theta=1, Euler retrograde) : ||v|| reste BORNEE (amortissement implicite).
  //   C'est exactement le contraste "explicite instable / implicite stable" a grand dt.
  // ----------------------------------------------------------------------------------------------
  {
    const Real dt =
        Real(8.0 * dt_stable);  // au-dessus du pas stable : l'explicite amplifie a chaque pas
    const int K = 12;           // composition : g^K rend l'explosion explicite visible
    MultiFab st(S.ba, S.dm, vars.size, 1), phi(S.ba, S.dm, 1, 1);
    make_state(st, phi);
    const double v0 = vel_l2(st, c_rho, c_mx, c_my);

    // Schur : K pas de taille dt.
    CondensedSchurSourceStepper stepper(vars, S.geom, S.ba, S.bc, alpha);
    for (int k = 0; k < K; ++k)
      stepper.step(st, phi, bz, 0, Real(1.0), dt);
    const double v_schur = vel_l2(st, c_rho, c_mx, c_my);

    // Euler explicite : K pas du MEME systeme discret.
    MultiFab st0(S.ba, S.dm, vars.size, 1), phi0(S.ba, S.dm, 1, 1);
    make_state(st0, phi0);
    RefIntegrator expl(const_cast<Setup&>(S), B0, alpha, rho0);
    load_ref_from_state(expl, st0, phi0, c_rho, c_mx, c_my);
    for (int k = 0; k < K; ++k)
      expl.euler_step(dt);
    double v_expl = 0;
    {               // norme L2 des vitesses de l'integrateur explicite
      sync_host();  // vx/vy ecrits par euler_step (kernels device) : residence hote valide avant lecture.
      double s = 0;
      for (int li = 0; li < expl.vx.local_size(); ++li) {
        const ConstArray4 vx = expl.vx.fab(li).const_array(), vy = expl.vy.fab(li).const_array();
        const Box2D b = expl.vx.box(li);
        for (int j = b.lo[1]; j <= b.hi[1]; ++j)
          for (int i = b.lo[0]; i <= b.hi[0]; ++i)
            s += static_cast<double>(vx(i, j, 0)) * vx(i, j, 0) +
                 static_cast<double>(vy(i, j, 0)) * vy(i, j, 0);
      }
      v_expl = std::sqrt(all_reduce_sum(s));
    }
    if (me == 0)
      std::printf(
          "(B) dt=%.3e (=%.0fx dt_stable), K=%d pas : ||v0||=%.3e | Schur ||v||=%.3e (x%.2f) | "
          "Euler ||v||=%.3e (x%.2e)\n",
          static_cast<double>(dt), 8.0, K, v0, v_schur, v_schur / v0, v_expl, v_expl / v0);
    // Schur STABLE : la vitesse reste BORNEE (pas d'explosion). On borne genereusement a x5.
    chk(v_schur < 5.0 * v0, "B_schur_stable_borne");
    chk(std::isfinite(v_schur), "B_schur_fini");
    // Euler EXPLOSE : la vitesse depasse de plusieurs ordres l'initiale ET ecrase Schur.
    chk(v_expl > 100.0 * v0, "B_explicite_explose");
    chk(v_expl > 50.0 * v_schur, "B_explicite_pire_que_schur");
  }

  // ----------------------------------------------------------------------------------------------
  // (C) ACCORD avec une REFERENCE fine-dt (RK4) a dt MODERE + ORDRE 1 en dt.
  //   On integre le MEME systeme discret par RK4 a pas fin (reference). Schur (theta=1, Euler
  //   retrograde) a une erreur O(dt) : on la mesure a dt et dt/2 et on exige une DECROISSANCE
  //   (ratio > 1.5, attendu ~2 pour l'ordre 1).
  // ----------------------------------------------------------------------------------------------
  {
    auto schur_vs_ref = [&](Real dt) -> double {
      // reference fine : RK4 a Nsub sous-pas sur [0, dt].
      const int Nsub = 256;
      MultiFab st0(S.ba, S.dm, vars.size, 1), phi0(S.ba, S.dm, 1, 1);
      make_state(st0, phi0);
      RefIntegrator ref(const_cast<Setup&>(S), B0, alpha, rho0);
      load_ref_from_state(ref, st0, phi0, c_rho, c_mx, c_my);
      const Real h = dt / Real(Nsub);
      for (int s = 0; s < Nsub; ++s)
        ref.rk4_step(h);

      // Schur : un pas de taille dt.
      MultiFab st(S.ba, S.dm, vars.size, 1), phi(S.ba, S.dm, 1, 1);
      make_state(st, phi);
      CondensedSchurSourceStepper stepper(vars, S.geom, S.ba, S.bc, alpha);
      stepper.step(st, phi, bz, 0, Real(1.0), dt);
      return vel_rel_state_vs_ref(st, ref, c_rho, c_mx, c_my);
    };

    const Real dtC = Real(0.5 * dt_stable);  // dt MODERE (sous le pas explicite stable)
    const double e1 = schur_vs_ref(dtC);
    const double e2 = schur_vs_ref(Real(0.5) * dtC);
    const double ratio = e2 > 0 ? e1 / e2 : 0;
    if (me == 0)
      std::printf(
          "(C) reference RK4 fine : err(dt)=%.3e err(dt/2)=%.3e ratio=%.2f (ordre 1 attendu ~2)\n",
          e1, e2, ratio);
    chk(e1 < 5e-2,
        "C_accord_reference_modere");  // Euler retrograde ordre 1 a dt modere : ecart faible
    chk(ratio > 1.5,
        "C_ordre_un_decroissance");  // l'erreur DECROIT (convergence vers la reference)
  }

  fails = static_cast<long>(all_reduce_max(static_cast<double>(fails)));
  if (me == 0 && fails == 0)
    std::printf("OK test_condensed_schur_source_stepper (np=%d)\n", np);
  comm_finalize();
  return fails == 0 ? 0 : 1;
}
