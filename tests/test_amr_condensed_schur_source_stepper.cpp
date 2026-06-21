// ETAGE SOURCE condense par Schur sur HIERARCHIE AMR (AmrCondensedSchurSourceStepper) : test de PARITE
// mono-niveau. Cf. include/adc/coupling/amr_condensed_schur_source_stepper.hpp.
//
// On valide que l'etage source condense AMR, applique a une hierarchie MONO-NIVEAU (un seul niveau
// couvrant tout le domaine, AUCUN patch fin), est STRICTEMENT identique a l'etage source condense
// uniforme (CondensedSchurSourceStepper, #126) sur la MEME grille. C'est le critere de l'Etape 2 :
// "sur une hierarchie AMR mono-niveau couvrant tout le domaine, un pas AmrCondensedSchurSourceStepper
// doit matcher CondensedSchurSourceStepper uniforme a tolerance stricte." En mono-niveau le grossier
// EST tout le domaine, donc l'etage AMR COMPOSE l'etage uniforme et la parite est bit-pour-bit.
//
// Verifs MONO-NIVEAU :
//   (A) PARITE : ecart MAX sur l'etat (toutes composantes) ET sur phi entre les deux etages < 1e-13.
//   (B) RELATION IMPLICITE : v^{n+1} de l'etage AMR satisfait B v^{n+1} = v^n - dt grad phi^{n+1}
//       (B = I - dt [Omega], grad CENTRE) a la tolerance du solve -> l'etage resout le BON systeme.
//   (C) SOURCE NON TRIVIALE : la quantite de mouvement a CHANGE (la source a bien agi).
//   (D) GARDE Phase 4b : une hierarchie a > 2 niveaux leve une erreur claire.
//
// Verifs MULTI-PATCH (Phase 4a) : 2 niveaux + 2 patchs fins disjoints NON ADJACENTS, dynamique confinee
// dans UN patch (bump gaussien), l'autre distant et quiet :
//   (multi-i)   FINITUDE de l'etat (grossier + 2 patchs).
//   (multi-ii)  PARITE PHYSIQUE FAIBLE : le patch actif ~ le cas mono-patch equivalent (le patch distant
//               perturbe < 5%, seuil RELATIF car le couplage elliptique est GLOBAL -- pas bit-a-bit).
//   (multi-iii) MASSE CONSERVEE : rho gelee sur le grossier + les 2 patchs.
//   (multi-iv)  CHEMIN MONO-PATCH INTACT : average_down coherent (grossier couvert = moyenne 2x2 fine).
//   (multi-v)   REJET du raccord fin-fin : 2 patchs ADJACENTS -> erreur claire (Phase 4b).
//
// Serie (Kokkos OFF) : n_ranks() == 1. L'etage uniforme est par ailleurs deja rejoue MPI np=1/2/4 par
// son propre test ; la parite mono-niveau est une propriete de composition, independante du rang.

#include <adc/coupling/amr_condensed_schur_source_stepper.hpp>
#include <adc/coupling/condensed_schur_source_stepper.hpp>

#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

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

// Initialise l'etat (rho0, mom = rho0 v0) et phi0 a partir de profils lisses, nuls au bord (Dirichlet).
struct InitKernel {
  Geometry geom;
  Array4 st, phi;
  Real rho0;
  int c_rho, c_mx, c_my, c_E;  // c_E < 0 si pas d'energie
  ADC_HD void operator()(int i, int j) const {
    const Real x = geom.x_cell(i), y = geom.y_cell(j);
    const Real sx = std::sin(Real(kPi) * x), sy = std::sin(Real(kPi) * y);
    const Real vx = Real(0.6) * sx * sy;
    const Real vy = Real(-0.4) * std::sin(Real(2 * kPi) * x) * sy;
    const Real ph = Real(0.3) * sx * sy;
    st(i, j, c_rho) = rho0;
    st(i, j, c_mx) = rho0 * vx;
    st(i, j, c_my) = rho0 * vy;
    if (c_E >= 0)
      st(i, j, c_E) = Real(1.0) + Real(0.5) * rho0 * (vx * vx + vy * vy);
    phi(i, j, 0) = ph;
  }
};

struct ConstBzKernel {
  Array4 bz;
  Real B0;
  ADC_HD void operator()(int i, int j) const { bz(i, j, 0) = B0; }
};

// Vitesse LOCALISEE : bump gaussien centre en (xc, yc), de largeur sigma -> v non nulle SEULEMENT autour
// de ce point. Sert a CONFINER la dynamique dans UN patch fin (l'autre patch, distant, voit v ~ 0).
struct LocalizedInitKernel {
  Geometry geom;
  Array4 st;
  Real rho0, xc, yc, sigma;
  int c_rho, c_mx, c_my, c_E;
  ADC_HD void operator()(int i, int j) const {
    const Real x = geom.x_cell(i), y = geom.y_cell(j);
    const Real dx = x - xc, dy = y - yc;
    const Real bump = std::exp(-(dx * dx + dy * dy) / (Real(2) * sigma * sigma));
    const Real vx = Real(0.6) * bump;
    const Real vy = Real(-0.4) * bump;
    st(i, j, c_rho) = rho0;
    st(i, j, c_mx) = rho0 * vx;
    st(i, j, c_my) = rho0 * vy;
    if (c_E >= 0)
      st(i, j, c_E) = Real(1.0) + Real(0.5) * rho0 * (vx * vx + vy * vy);
  }
};

// vrai si toutes les cellules valides (toutes composantes) sont finies, reduit sur tous les rangs.
static bool all_finite(const MultiFab& m, int nc) {
  sync_host();
  double bad = 0;
  for (int li = 0; li < m.local_size(); ++li) {
    const ConstArray4 u = m.fab(li).const_array();
    const Box2D b = m.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        for (int c = 0; c < nc; ++c)
          if (!std::isfinite(u(i, j, c)))
            bad = 1;
  }
  return all_reduce_max(bad) == 0.0;
}

// ecart MAX sur rho (composante c_rho) entre l'etat @p a et l'etat @p b (cellules valides).
static double max_drho(const MultiFab& a, const MultiFab& b, int c_rho) {
  sync_host();
  double d = 0;
  for (int li = 0; li < a.local_size(); ++li) {
    const ConstArray4 ua = a.fab(li).const_array(), ub = b.fab(li).const_array();
    const Box2D bx = a.box(li);
    for (int j = bx.lo[1]; j <= bx.hi[1]; ++j)
      for (int i = bx.lo[0]; i <= bx.hi[0]; ++i)
        d = std::fmax(d, std::fabs(ua(i, j, c_rho) - ub(i, j, c_rho)));
  }
  return all_reduce_max(d);
}

// dst <- src (toutes composantes valides, copie hote ; tampons mono-box repliques en serie).
static void copy_all(MultiFab& dst, const MultiFab& src, int ncomp) {
  sync_host();
  for (int li = 0; li < dst.local_size(); ++li) {
    Array4 d = dst.fab(li).array();
    const ConstArray4 s = src.fab(li).const_array();
    const Box2D b = dst.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        for (int c = 0; c < ncomp; ++c)
          d(i, j, c) = s(i, j, c);
  }
}

// ecart MAX (sur cellules valides, toutes composantes) entre deux MultiFab.
static double max_diff(const MultiFab& a, const MultiFab& b, int ncomp) {
  sync_host();
  double d = 0;
  for (int li = 0; li < a.local_size(); ++li) {
    const ConstArray4 ua = a.fab(li).const_array();
    const ConstArray4 ub = b.fab(li).const_array();
    const Box2D bx = a.box(li);
    for (int j = bx.lo[1]; j <= bx.hi[1]; ++j)
      for (int i = bx.lo[0]; i <= bx.hi[0]; ++i)
        for (int c = 0; c < ncomp; ++c)
          d = std::fmax(d, std::fabs(ua(i, j, c) - ub(i, j, c)));
  }
  return all_reduce_max(d);
}

// RELATION IMPLICITE : ecart MAX |B v^{n+1} - (v^n - dt grad phi^{n+1})|, grad CENTRE (meme stencil que
// la reconstruction). v^n (vxn, vyn) est l'etat AVANT step (extrait a part).
static double implicit_residual(const MultiFab& st_new, const MultiFab& vxn, const MultiFab& vyn,
                                MultiFab& phi_new, const Geometry& geom, const BCRec& bc, Real B0,
                                Real dt, int c_rho, int c_mx, int c_my) {
  device_fence();
  fill_ghosts(phi_new, geom.domain, bc);
  device_fence();
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
        const Real Bvx = vnx - dt * B0 * vny;
        const Real Bvy = vny + dt * B0 * vnx;
        const Real gx = (p(i + 1, j) - p(i - 1, j)) * half_idx;
        const Real gy = (p(i, j + 1) - p(i, j - 1)) * half_idy;
        const Real rhsx = vx(i, j, 0) - dt * gx;
        const Real rhsy = vy(i, j, 0) - dt * gy;
        d = std::fmax(d, std::fmax(std::fabs(Bvx - rhsx), std::fabs(Bvy - rhsy)));
      }
  }
  return all_reduce_max(d);
}

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank();
  long fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      if (me == 0)
        std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int n = 32;
  const Real rho0 = Real(1.5), B0 = Real(4.0), alpha = Real(3.0);
  const Real theta = Real(1.0);  // Euler retrograde (parite la plus stricte)
  Setup S(n);
  const VariableSet vars = fluid_vars(/*with_E=*/true);
  const int c_rho = vars.index_of(VariableRole::Density);
  const int c_mx = vars.index_of(VariableRole::MomentumX);
  const int c_my = vars.index_of(VariableRole::MomentumY);
  const int c_E = vars.index_of(VariableRole::Energy);

  // Pas RAIDE : au-dela du pas explicite, l'etage de Schur reste stable (theta-schema implicite).
  const Real dt = Real(0.05);

  // --- etat / phi / B_z initiaux, IDENTIQUES pour la reference uniforme et l'etage AMR ---
  MultiFab st_ref(S.ba, S.dm, vars.size, 1), st_amr(S.ba, S.dm, vars.size, 1);
  MultiFab phi_ref(S.ba, S.dm, 1, 1), phi_amr(S.ba, S.dm, 1, 1);
  MultiFab bz(S.ba, S.dm, 1, 1);
  for (int li = 0; li < st_ref.local_size(); ++li) {
    for_each_cell(st_ref.box(li),
                  InitKernel{S.geom, st_ref.fab(li).array(), phi_ref.fab(li).array(), rho0, c_rho,
                             c_mx, c_my, c_E});
    for_each_cell(bz.box(li), ConstBzKernel{bz.fab(li).array(), B0});
  }
  copy_all(st_amr, st_ref, vars.size);
  copy_all(phi_amr, phi_ref, 1);

  // v^n (avant step) pour la relation implicite (B).
  MultiFab vxn(S.ba, S.dm, 1, 0), vyn(S.ba, S.dm, 1, 0);
  for (int li = 0; li < st_ref.local_size(); ++li)
    for_each_cell(st_ref.box(li),
                  detail::ExtractVelocityKernel{st_ref.fab(li).const_array(), vxn.fab(li).array(),
                                                vyn.fab(li).array(), c_rho, c_mx, c_my});

  // mom AVANT (pour verifier que la source agit).
  MultiFab mom0(S.ba, S.dm, vars.size, 1);
  copy_all(mom0, st_ref, vars.size);

  // --- (1) REFERENCE uniforme #126 ---
  CondensedSchurSourceStepper ref(vars, S.geom, S.ba, S.bc, alpha);
  ref.step(st_ref, phi_ref, bz, /*c_bz=*/0, theta, dt);
  const KrylovResult kref = ref.last_solve();

  // --- (2) ETAGE AMR mono-niveau : un seul niveau couvrant tout le domaine ---
  std::vector<AmrLevelMP> levels;
  levels.push_back(AmrLevelMP{std::move(st_amr), &bz, S.geom.dx(), S.geom.dy()});
  AmrCondensedSchurSourceStepper amr(vars, S.geom, S.ba, S.bc, alpha);
  amr.step(levels, phi_amr, bz, /*c_bz=*/0, theta, dt);
  const KrylovResult kamr = amr.last_solve();

  // (A) PARITE bit-pour-bit etat + phi.
  const double dstate = max_diff(levels[0].U, st_ref, vars.size);
  const double dphi = max_diff(phi_amr, phi_ref, 1);
  chk(std::isfinite(dstate) && dstate < 1e-13, "parite etat AMR == uniforme (< 1e-13)");
  chk(std::isfinite(dphi) && dphi < 1e-13, "parite phi AMR == uniforme (< 1e-13)");
  chk(kref.converged && kamr.converged, "les deux solves convergent");
  chk(kref.iters == kamr.iters, "memes iterations BiCGStab");

  // (B) RELATION IMPLICITE de l'etage AMR.
  const double rimp =
      implicit_residual(levels[0].U, vxn, vyn, phi_amr, S.geom, S.bc, B0, dt, c_rho, c_mx, c_my);
  chk(std::isfinite(rimp) && rimp < 1e-7, "relation implicite B v = v^n - dt grad phi (< 1e-7)");

  // (C) la source a CHANGE la quantite de mouvement.
  const double dmom = max_diff(levels[0].U, mom0, vars.size);
  chk(std::isfinite(dmom) && dmom > 1e-6, "la source modifie l'etat (dmom > 1e-6)");

  // (D) GARDE Phase 4b : > 2 niveaux -> erreur claire (l'etage composite est cable pour 2 niveaux ;
  // > 2 niveaux / MPI / multi-blocs sont la Phase 4b). Le multi-patch fin (Phase 4a) est desormais
  // ACCEPTE (cf. section MULTI-PATCH ci-dessous) ; le cas 2-niveaux-1-patch VALIDE est exerce a part
  // dans test_amr_condensed_schur_composite.
  {
    MultiFab f1(S.ba, S.dm, vars.size, 1), f2(S.ba, S.dm, vars.size, 1), cphi(S.ba, S.dm, 1, 1);
    copy_all(f1, mom0, vars.size);
    copy_all(f2, mom0, vars.size);
    copy_all(cphi, phi_ref, 1);
    std::vector<AmrLevelMP> three;  // 3 niveaux -> hors du cadre 2-niveaux du composite
    three.push_back(AmrLevelMP{std::move(levels[0].U), &bz, S.geom.dx(), S.geom.dy()});
    three.push_back(AmrLevelMP{std::move(f1), &bz, S.geom.dx() / 2, S.geom.dy() / 2});
    three.push_back(AmrLevelMP{std::move(f2), &bz, S.geom.dx() / 4, S.geom.dy() / 4});
    bool threw = false;
    try {
      amr.step(three, cphi, bz, 0, theta, dt);
    } catch (const std::exception&) {
      threw = true;
    }
    chk(threw,
        "hierarchie > 2 niveaux -> erreur claire (>2 niveaux / MPI / multi-blocs = Phase 4b)");
  }

  // =========================== MULTI-PATCH FIN (Phase 4a) ===========================
  // Etage source condense COMPOSITE sur 2 niveaux + 2 PATCHS fins disjoints NON ADJACENTS (separes de
  // plusieurs cellules grossieres). La dynamique (vitesse) est CONFINEE dans le patch A (bump gaussien) ;
  // le patch B, distant, voit v ~ 0. On verifie : (i) finitude ; (ii) parite physique faible (le resultat
  // dans le patch A est ~ celui du cas MONO-patch A seul -- le patch B distant et quiet perturbe le patch
  // A bien en-dessous du pourcent) ; (iii) masse conservee (rho gelee) ; (iv) chemin mono-patch intact
  // (average_down coherent) ; (v) rejet d'un raccord fin-fin (patchs ADJACENTS) -> erreur claire.
  {
    // grossier = Setup S (n = 32) ; fin = grossier raffine x2 (domaine fin [0,63]^2).
    const Real rho0m = Real(1.5), B0m = Real(4.0);
    const Geometry geom_c = S.geom;
    const Geometry geom_f = geom_c.refine(2);

    // Patch A : empreinte grossiere [8,15]^2 -> box fine [16,31]^2 (lo pair / hi impair, interieur).
    // Patch B : empreinte grossiere [20,27]^2 -> box fine [40,55]^2. Separation = 4 cellules grossieres.
    Box2D fbA{{16, 16}, {31, 31}};
    Box2D fbB{{40, 40}, {55, 55}};
    BoxArray ba_fAB(std::vector<Box2D>{fbA, fbB});  // pavage 2 patchs
    BoxArray ba_fA(std::vector<Box2D>{fbA});        // mono-patch A (reference)
    DistributionMapping dm_fAB(ba_fAB.size(), n_ranks());
    DistributionMapping dm_fA(ba_fA.size(), n_ranks());

    // centre du bump dans le patch A (region physique [0.25, 0.5]^2) ; sigma petit -> v ~ 0 au patch B.
    const Real xc = Real(0.36), yc = Real(0.36), sigma = Real(0.05);
    auto fill_localized = [&](MultiFab& U, const Geometry& g) {
      for (int li = 0; li < U.local_size(); ++li)
        for_each_cell(U.box(li), LocalizedInitKernel{g, U.fab(li).array(), rho0m, xc, yc, sigma,
                                                     c_rho, c_mx, c_my, c_E});
    };

    // Un pas composite ; rend l'etat grossier + fin et les etats INITIAUX fins (pour mesurer l'effet source).
    auto run = [&](const BoxArray& ba_f, const DistributionMapping& dm_f, MultiFab& Uc_out,
                   MultiFab& Uf_out, MultiFab& Uf_init_out) {
      MultiFab Uc(S.ba, S.dm, vars.size, 1), Uf(ba_f, dm_f, vars.size, 1);
      fill_localized(Uc, geom_c);
      fill_localized(Uf, geom_f);
      Uf_init_out = MultiFab(ba_f, dm_f, vars.size, 1);
      copy_all(Uf_init_out, Uf, vars.size);
      MultiFab cphi(S.ba, S.dm, 1, 1), cbz(S.ba, S.dm, 1, 1), faux(ba_f, dm_f, 1, 1);
      cphi.set_val(0.0);  // phi^n grossier
      faux.set_val(0.0);  // phi^n fin (aux comp 0)
      for (int li = 0; li < cbz.local_size(); ++li)
        for_each_cell(cbz.box(li), ConstBzKernel{cbz.fab(li).array(), B0m});
      std::vector<AmrLevelMP> lv;
      lv.push_back(AmrLevelMP{std::move(Uc), &cphi, geom_c.dx(), geom_c.dy()});
      lv.push_back(AmrLevelMP{std::move(Uf), &faux, geom_f.dx(), geom_f.dy()});
      AmrCondensedSchurSourceStepper st(vars, geom_c, S.ba, S.bc, alpha);
      st.step(lv, cphi, cbz, /*c_bz=*/0, theta, dt);
      Uc_out = std::move(lv[0].U);
      Uf_out = std::move(lv[1].U);
    };

    MultiFab UcAB, UfAB, UfAB0, UcA, UfA, UfA0;
    run(ba_fAB, dm_fAB, UcAB, UfAB, UfAB0);  // multi-patch (A + B)
    run(ba_fA, dm_fA, UcA, UfA, UfA0);       // mono-patch (A seul) -> reference de parite

    // (i) FINITUDE : etat grossier + les deux patchs fins du run multi-patch.
    chk(all_finite(UcAB, vars.size) && all_finite(UfAB, vars.size),
        "(multi-i) etat fini (grossier + 2 patchs fins)");

    // (ii) PARITE PHYSIQUE FAIBLE. Le patch A est fab(0) dans les DEUX runs (meme box, premier du pavage).
    // dmom_A = effet de la source dans le patch A (mono) ; diff_A = perturbation due a l'ajout du patch B.
    // Le patch B etant distant et QUIET (v ~ 0, phi^n induit lisse et petit a sa position, conditions de
    // Dirichlet nulles au bord), son raffinement de l'elliptique perturbe le patch A a ~ la PRECISION DU
    // SOLVE FAC (residu compose ~1e-9) : empiriquement diff_A/dmom_A ~ 5e-8. Ce n'est PAS une parite
    // bit-a-bit (le couplage elliptique est GLOBAL : average_down et correction de flux du patch B touchent
    // le grossier, donc le patch A) -- d'ou un seuil RELATIF, fixe a 1e-5 (~200x de marge sur l'observe,
    // robuste a la variation FP, tout en prouvant que le patch distant est quasi un no-op).
    {
      sync_host();
      const ConstArray4 A_multi = UfAB.fab(0).const_array();
      const ConstArray4 A_mono = UfA.fab(0).const_array();
      const ConstArray4 A_init = UfA0.fab(0).const_array();
      double diff_A = 0, dmom_A = 0;
      const Box2D b = UfA.box(0);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          for (int c = c_mx; c <= c_my; ++c) {
            diff_A = std::fmax(diff_A, std::fabs(A_multi(i, j, c) - A_mono(i, j, c)));
            dmom_A = std::fmax(dmom_A, std::fabs(A_mono(i, j, c) - A_init(i, j, c)));
          }
      diff_A = all_reduce_max(diff_A);
      dmom_A = all_reduce_max(dmom_A);
      if (me == 0)
        std::printf("  [multi] dmom_A=%.3e diff_A=%.3e (rel=%.2e)\n", dmom_A, diff_A,
                    diff_A / std::fmax(dmom_A, 1e-30));
      chk(dmom_A > 1e-3,
          "(multi-ii) la source agit reellement dans le patch A (dynamique non triviale)");
      chk(std::isfinite(diff_A) && diff_A < Real(1e-5) * dmom_A,
          "(multi-ii) parite physique : patch A ~ cas mono-patch (perturbation du patch B distant "
          "< 1e-5)");
    }

    // (iii) MASSE CONSERVEE : rho GELEE par la source sur le grossier ET les deux patchs fins. On compare
    // a un etat initial rho0 uniforme (rho ne doit pas bouger ; le grossier couvert = average_down de
    // rho0 = rho0).
    {
      MultiFab UcAB0(S.ba, S.dm, vars.size, 1);
      fill_localized(UcAB0, geom_c);  // meme init -> rho0 partout
      chk(max_drho(UcAB, UcAB0, c_rho) < 1e-12 && max_drho(UfAB, UfAB0, c_rho) < 1e-12,
          "(multi-iii) masse conservee : rho gelee (grossier + 2 patchs fins)");
    }

    // (iv) CHEMIN MONO-PATCH INTACT : sur le run A seul, les cellules grossieres COUVERTES = moyenne 2x2
    // des cellules fines (cascade average_down coherente, invariant #169). Empreinte de A = [8,15]^2.
    {
      sync_host();
      const ConstArray4 UC = UcA.fab(0).const_array();
      const ConstArray4 UF = UfA.fab(0).const_array();
      double dcov = 0;
      for (int J = 8; J <= 15; ++J)
        for (int I = 8; I <= 15; ++I)
          for (int c = c_mx; c <= c_my; ++c) {
            const double avg = 0.25 * (UF(2 * I, 2 * J, c) + UF(2 * I + 1, 2 * J, c) +
                                       UF(2 * I, 2 * J + 1, c) + UF(2 * I + 1, 2 * J + 1, c));
            dcov = std::fmax(dcov, std::fabs(UC(I, J, c) - avg));
          }
      chk(all_reduce_max(dcov) < 1e-12,
          "(multi-iv) mono-patch intact : grossier couvert = moyenne 2x2 fine (average_down)");
    }

    // (v) REJET du raccord fin-fin (Phase 4b) : deux patchs ADJACENTS (empreintes grossieres [8,15] et
    // [16,23], separation NULLE) -> le ctor du FAC leve une erreur claire. On ne tente PAS de coupler des
    // faces fines partagees en silence.
    {
      Box2D adjA{{16, 16}, {31, 31}};  // empreinte [8,15]
      Box2D adjB{{32, 32}, {47, 47}};  // empreinte [16,23] -> ADJACENTE a adjA
      BoxArray ba_adj(std::vector<Box2D>{adjA, adjB});
      DistributionMapping dm_adj(ba_adj.size(), n_ranks());
      MultiFab Uc(S.ba, S.dm, vars.size, 1), Uf(ba_adj, dm_adj, vars.size, 1);
      fill_localized(Uc, geom_c);
      fill_localized(Uf, geom_f);
      MultiFab cphi(S.ba, S.dm, 1, 1), cbz(S.ba, S.dm, 1, 1), faux(ba_adj, dm_adj, 1, 1);
      cphi.set_val(0.0);
      faux.set_val(0.0);
      for (int li = 0; li < cbz.local_size(); ++li)
        for_each_cell(cbz.box(li), ConstBzKernel{cbz.fab(li).array(), B0m});
      std::vector<AmrLevelMP> lv;
      lv.push_back(AmrLevelMP{std::move(Uc), &cphi, geom_c.dx(), geom_c.dy()});
      lv.push_back(AmrLevelMP{std::move(Uf), &faux, geom_f.dx(), geom_f.dy()});
      AmrCondensedSchurSourceStepper st(vars, geom_c, S.ba, S.bc, alpha);
      bool threw = false;
      try {
        st.step(lv, cphi, cbz, /*c_bz=*/0, theta, dt);
      } catch (const std::exception&) {
        threw = true;
      }
      chk(threw, "(multi-v) patchs fins ADJACENTS -> erreur claire (raccord fin-fin = Phase 4b)");
    }
  }

  fails = static_cast<long>(all_reduce_max(static_cast<double>(fails)));
  if (me == 0 && fails == 0)
    std::printf("OK test_amr_condensed_schur_source_stepper\n");
  comm_finalize();
  return fails == 0 ? 0 : 1;
}
