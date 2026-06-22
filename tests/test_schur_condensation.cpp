// Tests ANALYTIQUES du batisseur de l'etage source condense par Schur (assemblage SEUL : coefficients
// de l'operateur tensoriel A_op + second membre condense ; PAS de solve, PAS de reconstruction de
// vitesse). Cf. include/adc/coupling/schur_condensation.hpp et docs/SCHUR_CONDENSATION_DESIGN.md.
//
// CONVENTION FIGEE : A_op = I + c rho B^{-1}, c = theta^2 dt^2 alpha ; B^{-1} = (1/det)[[1,w],[-w,1]],
// w = theta dt B_z, det = 1+w^2. RHS = -Lap phi^n - theta dt alpha div(rho B^{-1} v^n), v = (mx,my)/rho.
//
// Tests :
//   (A) rho = const, B_z = const : les coefficients ASSEMBLES eps_x/eps_y/a_xy/a_yx egalent la valeur
//       ANALYTIQUE fermee A = I + c rho B^{-1} (calculee a la main), a la precision machine.
//   (B) rho = const, B_z = const, phi^n et v^n choisis : le RHS assemble egale la valeur ANALYTIQUE
//       -Lap phi^n - theta dt alpha div(rho B^{-1} v^n). phi^n quadratique (Laplacien 5-points EXACT) et
//       flux F = rho B^{-1} v^n LINEAIRE en espace (divergence centree EXACTE) -> match a la precision
//       machine sur les cellules INTERIEURES (le bord depend de la CL des ghosts, hors de ce test).
//   (C) B_z = 0 : A degenere en diag(1 + c rho), a_xy = a_yx = 0 (cas symetrique). Et avec c = 0
//       (alpha = 0) : A = I -> les coefficients sont BIT-IDENTIQUES (eps_x=eps_y=1, a_xy=a_yx=0) au
//       Poisson canonique, et le RHS (sans terme de flux) = -Lap phi^n BIT-IDENTIQUE au Laplacien
//       canonique apply_laplacian (negue). Garde-fou de non-regression.

#include <adc/coupling/schur/schur_condensation.hpp>

#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/layout/distribution_mapping.hpp>
#include <adc/mesh/execution/for_each.hpp>
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/multifab.hpp>
#include <adc/mesh/boundary/physical_bc.hpp>
#include <adc/numerics/elliptic/poisson/poisson_operator.hpp>
#include <adc/numerics/linalg/lorentz_eliminator.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

static double dabs(double x) {
  return x < 0 ? -x : x;
}
static constexpr double EPS = 1e-12;  // precision machine relachee (FMA / association FP)

// VariableSet d'un fluide minimal portant les roles requis : rho, mx, my (Density, MomentumX, MomentumY).
static VariableSet fluid_vars() {
  VariableSet vs;
  vs.kind = VariableKind::Conservative;
  vs.names = {"rho", "mx", "my"};
  vs.roles = {VariableRole::Density, VariableRole::MomentumX, VariableRole::MomentumY};
  vs.size = 3;
  return vs;
}

// Domaine carre unite, mono-box (decompose en MPI par DistributionMapping round-robin).
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
    bc.xlo = bc.xhi = bc.ylo = bc.yhi =
        BCType::Dirichlet;  // bords physiques (RHS teste a l'interieur)
  }
};

// Ecart MAX, sur les cellules VALIDES locales puis reduit en MAX MPI, entre la composante 0 d'un
// MultiFab et une valeur analytique f(i,j). Saut optionnel d'une bordure de @p skip cellules
// (cellules interieures seulement, pour le RHS dont le bord depend de la CL des ghosts).
template <class F>
static double max_gap(const MultiFab& mf, const Box2D& dom, F f, int skip = 0) {
  // mf a ete ecrit par les kernels device du batisseur (assemble_operator / assemble_rhs, qui finissent
  // par fill_ghosts -> kernels for_each_cell). Sous Kokkos::Cuda ces kernels sont ASYNCHRONES : sans
  // rendre la residence hote valide AVANT la lecture directe ci-dessous, on lit la memoire unifiee
  // pendant que le kernel est en vol -> valeurs partielles/indefinies (eps qui derive, RHS faux, les
  // checks bit-identiques C1/C2 cassent). sync_host() = device_fence() cible sous Kokkos, no-op en
  // serie/OpenMP -> comportement hote BIT-IDENTIQUE. Meme idiome que test_condensed_schur_source_stepper.
  sync_host();
  double d = 0;
  for (int li = 0; li < mf.local_size(); ++li) {
    const ConstArray4 a = mf.fab(li).const_array();
    const Box2D b = mf.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        if (i < dom.lo[0] + skip || i > dom.hi[0] - skip || j < dom.lo[1] + skip ||
            j > dom.hi[1] - skip)
          continue;
        d = std::fmax(d, dabs(a(i, j, 0) - f(i, j)));
      }
  }
  return all_reduce_max(d);
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const VariableSet vars = fluid_vars();

  // ----------------------------------------------------------------------------------------------
  // (A) rho = const, B_z = const : coefficients de A_op = I + c rho B^{-1} vs valeur analytique fermee.
  // ----------------------------------------------------------------------------------------------
  {
    const int n = 32;
    Setup S(n);
    const Real rho0 = 1.7, Bz0 = 2.5, alpha = 0.8, theta = 0.5, dt = 0.3;

    MultiFab state(S.ba, S.dm, 3, 0), bz(S.ba, S.dm, 1, 1);
    for (int li = 0; li < state.local_size(); ++li) {
      Array4 u = state.fab(li).array(), b = bz.fab(li).array();
      const Box2D vb = state.box(li);
      for (int j = vb.lo[1]; j <= vb.hi[1]; ++j)
        for (int i = vb.lo[0]; i <= vb.hi[0]; ++i) {
          u(i, j, 0) = rho0;
          u(i, j, 1) = 0;
          u(i, j, 2) = 0;
          b(i, j, 0) = Bz0;
        }
    }

    ElectrostaticLorentzCondensation builder(vars, alpha, theta, dt);
    MultiFab ex(S.ba, S.dm, 1, 1), ey(S.ba, S.dm, 1, 1), axy(S.ba, S.dm, 1, 1),
        ayx(S.ba, S.dm, 1, 1);
    builder.assemble_operator(state, bz, S.geom, S.bc, ex, ey, axy, ayx);

    // valeur analytique fermee (by-hand) : c = theta^2 dt^2 alpha, B^{-1} = (1/det)[[1,w],[-w,1]].
    const double c = theta * theta * dt * dt * alpha;
    const double w = theta * dt * Bz0;
    const double det = 1.0 + w * w;
    const double exa = 1.0 + c * rho0 * (1.0 / det);  // 1 + c rho binv_11
    const double eya = 1.0 + c * rho0 * (1.0 / det);  // 1 + c rho binv_22
    const double axya = c * rho0 * (w / det);         // c rho binv_12
    const double ayxa = c * rho0 * (-w / det);        // c rho binv_21

    chk(max_gap(ex, S.dom, [=](int, int) { return exa; }) < EPS, "A_eps_x_analytique");
    chk(max_gap(ey, S.dom, [=](int, int) { return eya; }) < EPS, "A_eps_y_analytique");
    chk(max_gap(axy, S.dom, [=](int, int) { return axya; }) < EPS, "A_a_xy_analytique");
    chk(max_gap(ayx, S.dom, [=](int, int) { return ayxa; }) < EPS, "A_a_yx_analytique");
    // non symetrie attendue (B_z != 0) : a_xy = -a_yx != 0.
    chk(dabs(axya + ayxa) < EPS && dabs(axya) > 1e-6, "A_non_symetrique_Bz_nonnul");
    std::printf("(A) coeff analytiques : eps=%.6f a_xy=%.6f a_yx=%.6f (c=%.4f w=%.4f)\n", exa, axya,
                ayxa, c, w);
  }

  // ----------------------------------------------------------------------------------------------
  // (B) RHS = -Lap phi^n - theta dt alpha div(rho B^{-1} v^n) vs analytique, cellules INTERIEURES.
  //     phi^n quadratique : phi = px*x^2 + py*y^2 + pxy*x*y -> Lap 5-points EXACT = 2 px + 2 py.
  //     v^n choisi pour que F = rho B^{-1} v = B^{-1}(rho v) = B^{-1}(mx,my) soit LINEAIRE en (x,y),
  //     donc div centree EXACTE. On prend mx, my AFFINES : div(B^{-1} m) analytique ferme.
  // ----------------------------------------------------------------------------------------------
  {
    const int n = 48;
    Setup S(n);
    const Real rho0 = 1.3, Bz0 = 1.8, alpha = 0.9, theta = 0.5, dt = 0.25;
    const double c = theta * theta * dt * dt * alpha;
    (void)c;
    const double w = theta * dt * Bz0;
    const double det = 1.0 + w * w;
    const double g = theta * dt * alpha;  // coefficient du terme div(rho B^{-1} v)

    // phi^n quadratique, mx/my affines.
    const double px = 0.7, py = -0.4, pxy = 0.3;    // phi = px x^2 + py y^2 + pxy x y
    const double mx0 = 0.2, mxx = 0.5, mxy = -0.1;  // mx = mx0 + mxx x + mxy y
    const double my0 = -0.3, myx = 0.4, myy = 0.6;  // my = my0 + myx x + myy y
    auto phi_f = [=](double x, double y) { return px * x * x + py * y * y + pxy * x * y; };
    auto mx_f = [=](double x, double y) { return mx0 + mxx * x + mxy * y; };
    auto my_f = [=](double x, double y) { return my0 + myx * x + myy * y; };

    MultiFab state(S.ba, S.dm, 3, 0), bz(S.ba, S.dm, 1, 1), phi(S.ba, S.dm, 1, 1);
    for (int li = 0; li < state.local_size(); ++li) {
      Array4 u = state.fab(li).array(), b = bz.fab(li).array(), p = phi.fab(li).array();
      const Box2D vb = state.box(li);
      for (int j = vb.lo[1]; j <= vb.hi[1]; ++j)
        for (int i = vb.lo[0]; i <= vb.hi[0]; ++i) {
          const double x = S.geom.x_cell(i), y = S.geom.y_cell(j);
          u(i, j, 0) = rho0;
          u(i, j, 1) = mx_f(x, y);
          u(i, j, 2) = my_f(x, y);
          b(i, j, 0) = Bz0;
          p(i, j, 0) = phi_f(x, y);
        }
    }

    ElectrostaticLorentzCondensation builder(vars, alpha, theta, dt);
    MultiFab rhs(S.ba, S.dm, 1, 0);
    builder.assemble_rhs(phi, state, bz, S.geom, S.bc, rhs);

    // ANALYTIQUE (continu = discret ici) :
    //   -Lap phi = -(2 px + 2 py)  [5-points exact pour un quadratique].
    //   F = B^{-1} (mx, my) ; B^{-1} = (1/det)[[1,w],[-w,1]] :
    //     Fx = (mx + w my)/det,  Fy = (my - w mx)/det.
    //   div F = d_x Fx + d_y Fy = (mxx + w myx)/det + (myy - w mxy)/det  (mx,my affines -> derivees const).
    //   RHS = -(2 px + 2 py) - g * div F.
    const double neg_lap = -(2.0 * px + 2.0 * py);
    const double divF = ((mxx + w * myx) + (myy - w * mxy)) / det;
    const double rhs_a = neg_lap - g * divF;

    // skip = 1 : on saute la bordure (le Laplacien et la div centree y lisent des ghosts poses par la
    // CL, qui ne reproduisent pas l'analytique continu ; le solve PR4 fixera la CL de phi^{n+theta}).
    // Tolerance : le Laplacien 5-points d'un quadratique est EXACT en arithmetique reelle, mais la
    // soustraction p(i+1)-2 p(i)+p(i-1) ~ O(dx^2) sur des termes O(1) annule ~16 chiffres et laisse une
    // erreur d'arrondi O(eps/dx^2). Pour n=48 (dx~1/48) cela vaut ~5e-13 ; on borne a 1e-11 (marge x20).
    const double gap = max_gap(rhs, S.dom, [=](int, int) { return rhs_a; }, 1);
    chk(gap < 1e-11, "B_rhs_analytique_interieur");
    std::printf("(B) RHS analytique=%.6f (neg_lap=%.6f, g*divF=%.6f) | ecart interieur=%.3e\n",
                rhs_a, neg_lap, g * divF, gap);
  }

  // ----------------------------------------------------------------------------------------------
  // (C) B_z = 0 : A diagonal symetrique (a_xy = a_yx = 0, eps_x = eps_y = 1 + c rho).
  //     PUIS c = 0 (alpha = 0) : A = I EXACTEMENT -> bit-identique au Poisson canonique. Le RHS sans
  //     flux (g = 0) = -Lap phi^n BIT-IDENTIQUE a apply_laplacian(phi) negue.
  // ----------------------------------------------------------------------------------------------
  {
    const int n = 32;
    Setup S(n);
    const Real rho0 = 2.1, theta = 0.5, dt = 0.4;

    MultiFab state(S.ba, S.dm, 3, 0), bz(S.ba, S.dm, 1, 1);
    for (int li = 0; li < state.local_size(); ++li) {
      Array4 u = state.fab(li).array(), b = bz.fab(li).array();
      const Box2D vb = state.box(li);
      for (int j = vb.lo[1]; j <= vb.hi[1]; ++j)
        for (int i = vb.lo[0]; i <= vb.hi[0]; ++i) {
          u(i, j, 0) = rho0;
          u(i, j, 1) = 0.5;
          u(i, j, 2) = -0.7;
          b(i, j, 0) = 0.0;  // B_z = 0
        }
    }

    // (C1) B_z = 0, alpha != 0 : A = diag(1 + c rho), croises NULS.
    {
      const Real alpha = 0.6;
      ElectrostaticLorentzCondensation builder(vars, alpha, theta, dt);
      MultiFab ex(S.ba, S.dm, 1, 1), ey(S.ba, S.dm, 1, 1), axy(S.ba, S.dm, 1, 1),
          ayx(S.ba, S.dm, 1, 1);
      builder.assemble_operator(state, bz, S.geom, S.bc, ex, ey, axy, ayx);
      const double c = theta * theta * dt * dt * alpha;
      const double diag = 1.0 + c * rho0;
      chk(max_gap(ex, S.dom, [=](int, int) { return diag; }) < EPS, "C1_eps_x_diag");
      chk(max_gap(ey, S.dom, [=](int, int) { return diag; }) < EPS, "C1_eps_y_diag");
      chk(max_gap(axy, S.dom, [](int, int) { return 0.0; }) == 0.0, "C1_a_xy_zero_bitident");
      chk(max_gap(ayx, S.dom, [](int, int) { return 0.0; }) == 0.0, "C1_a_yx_zero_bitident");
      std::printf("(C1) B_z=0 : A=diag(%.6f), croises=0 (symetrique)\n", diag);
    }

    // (C2) B_z = 0 ET alpha = 0 -> c = 0 : A = I BIT-IDENTIQUE, RHS = -Lap phi^n BIT-IDENTIQUE.
    {
      const Real alpha = 0.0;
      ElectrostaticLorentzCondensation builder(vars, alpha, theta, dt);
      MultiFab ex(S.ba, S.dm, 1, 1), ey(S.ba, S.dm, 1, 1), axy(S.ba, S.dm, 1, 1),
          ayx(S.ba, S.dm, 1, 1);
      builder.assemble_operator(state, bz, S.geom, S.bc, ex, ey, axy, ayx);
      // A = I : eps_x = eps_y = 1 EXACTEMENT (1 + 0), a_xy = a_yx = 0 EXACTEMENT (bit-identique).
      chk(max_gap(ex, S.dom, [](int, int) { return 1.0; }) == 0.0, "C2_eps_x_eq_1_bitident");
      chk(max_gap(ey, S.dom, [](int, int) { return 1.0; }) == 0.0, "C2_eps_y_eq_1_bitident");
      chk(max_gap(axy, S.dom, [](int, int) { return 0.0; }) == 0.0, "C2_a_xy_zero_bitident");
      chk(max_gap(ayx, S.dom, [](int, int) { return 0.0; }) == 0.0, "C2_a_yx_zero_bitident");

      // RHS : g = theta dt alpha = 0 -> pas de terme de flux. rhs = -Lap phi^n. On le compare BIT a BIT
      // a apply_laplacian(phi) negue (le MEME operateur que le batisseur emploie en interne).
      MultiFab phi(S.ba, S.dm, 1, 1);
      for (int li = 0; li < phi.local_size(); ++li) {
        Array4 p = phi.fab(li).array();
        const Box2D vb = phi.box(li);
        for (int j = vb.lo[1]; j <= vb.hi[1]; ++j)
          for (int i = vb.lo[0]; i <= vb.hi[0]; ++i) {
            const double x = S.geom.x_cell(i), y = S.geom.y_cell(j);
            p(i, j, 0) =
                std::sin(3.14159265358979323846 * x) * std::sin(2 * 3.14159265358979323846 * y);
          }
      }
      MultiFab rhs(S.ba, S.dm, 1, 0);
      builder.assemble_rhs(phi, state, bz, S.geom, S.bc, rhs);

      // reference : apply_laplacian sur le MEME phi (memes ghosts deja remplis par assemble_rhs), negue.
      // comparaison directe rhs vs -lap_ref, cellule a cellule (bit-identique attendu).
      MultiFab lap_ref(S.ba, S.dm, 1, 0);
      apply_laplacian(phi, S.geom, lap_ref);
      sync_host();  // rhs (assemble_rhs) et lap_ref (apply_laplacian) ecrits par kernels device :
          // residence hote valide avant la lecture directe (no-op serie/OpenMP, fence Cuda).
      double dmax = 0;
      for (int li = 0; li < rhs.local_size(); ++li) {
        const ConstArray4 r = rhs.fab(li).const_array();
        const ConstArray4 lr = lap_ref.fab(li).const_array();
        const Box2D b = rhs.box(li);
        for (int j = b.lo[1]; j <= b.hi[1]; ++j)
          for (int i = b.lo[0]; i <= b.hi[0]; ++i)
            dmax = std::fmax(dmax, dabs(r(i, j, 0) - (-lr(i, j, 0))));
      }
      dmax = all_reduce_max(dmax);
      chk(dmax == 0.0, "C2_rhs_eq_neg_laplacien_bitident");
      std::printf("(C2) c=0 : A=I bit-identique, RHS=-Lap phi^n bit-identique (dmax=%.3e)\n", dmax);
    }
  }

  if (fails == 0)
    std::printf("OK test_schur_condensation\n");
  return fails == 0 ? 0 : 1;
}
