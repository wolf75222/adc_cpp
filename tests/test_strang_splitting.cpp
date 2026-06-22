// Ordre TEMPOREL du splitting d'operateur ORCHESTRE PAR LE STEPPER DE SYSTEME (SystemStepper).
//
// On teste le VRAI orchestrateur (SystemStepper<Impl>::step pour Lie, ::step_strang pour Strang),
// pas la brique splitting.hpp (couverte par test_splitting). SystemStepper est un gabarit sur Impl :
// on l'instancie avec un Impl JOUET minimal qui satisfait son contrat structurel (sp, solve_fields,
// t, macro_step_, couplings, fields_.ell_phi(), aux, geom/pgeom_/polar_ pour la CFL). Aucun AMReX
// lourd, aucun Schur : l'etage source emprunte le hook GENERIQUE BlockState::source_step.
//
// Systeme jouet 2x2 NON commutant, de flot exact connu (matrice exp) -> on mesure UNIQUEMENT
// l'ordre temporel du splitting, sans erreur spatiale :
//   dU/dt = (A + B) U,  U = (x, y),  A = [[0,1],[0,0]], B = [[0,0],[1,0]]
//   H (transport, exp(A h)) : x += h y
//   S (source,    exp(B h)) : y += h g, ou g est le champ "resolu" par solve_fields (g <- x courant)
//   [A,B] != 0 -> vraie erreur de commutation ; exact : (x,y) = (x0 ch t + y0 sh t, x0 sh t + y0 ch t)
//
// La dependance de S au champ g (peuple par solve_fields) rend les RE-RESOLUTIONS de solve_fields du
// chemin Strang LOAD-BEARING : sans le solve_fields RE-RESOLU avant la source (et avant la 2nde
// demi-avance), S lirait un g PERIME et l'ordre 2 serait rompu. C'est la consistance phi documentee
// dans docs/HOFFART_STEP_SEQUENCE.md, verifiee ici par l'ordre observe ET par le compte d'appels.
//
// Attendu : Lie ordre ~1 (erreur /2 quand dt /2), Strang ordre ~2 (erreur /4).

#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/layout/distribution_mapping.hpp>
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/multifab.hpp>
#include <adc/runtime/system/system_block_store.hpp>
#include <adc/runtime/system/system_stepper.hpp>
// run_source_stage (gabarit, instancie via SystemStepper<MockImpl>::step) DEREFERENCE les types Schur
// dans des branches MORTES ici (s.schur / s.schur_polar restent nullptr ; on passe par source_step).
// Le member-access ->step(...) exige neanmoins le type COMPLET a l'instanciation -> on inclut les deux
// en-tetes Schur (comme python/system.cpp). On n'en CONSTRUIT aucun objet : le test reste leger.
#include <adc/coupling/schur/source/condensed_schur_source_stepper.hpp>
#include <adc/coupling/schur/source/polar_condensed_schur_source_stepper.hpp>

#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

using namespace adc;

// --- Impl JOUET : satisfait le contrat structurel de SystemStepper<Impl> -----------------------------
// Membres lus par le gabarit : Species (= SystemBlockStore::BlockState), sp, solve_fields(), t,
// macro_step_, couplings, fields_ (avec ell_phi()), aux, geom/pgeom_/polar_ (CFL, jamais exercee ici).
struct MockImpl {
  using Species = SystemBlockStore::BlockState;

  // fields_ minimal : seul ell_phi() est reference par run_source_stage (branche Schur, non prise ici
  // car on passe par source_step). Il doit neanmoins exister et rendre un MultiFab&.
  struct Fields {
    MultiFab phi;
    MultiFab& ell_phi() { return phi; }
  };

  std::vector<Species> sp;
  std::vector<std::function<void(Real)>> couplings;
  double t = 0.0;
  int macro_step_ = 0;
  bool polar_ = false;
  // Geometrie de transport EMBEDDED-BOUNDARY (chantier T5-PR3) : le stepper lit geometry_mode_ / eb_set_
  // pour aiguiller l'avance de transport. None + !eb_set_ -> le toy emprunte s.advance (chemin plein),
  // donc l'ordre temporel observe (Lie vs Strang) est INCHANGE (ce test ne fixe aucun domaine).
  GeometryMode geometry_mode_ = GeometryMode::None;
  bool eb_set_ = false;
  Geometry geom;
  PolarGeometry pgeom_;
  Fields fields_;
  MultiFab aux;

  // Champ "resolu" partage, lu par l'etage source (proxy de grad phi). Compte des resolutions.
  Real g = Real(0);
  int solve_calls = 0;

  MockImpl(const BoxArray& ba, const DistributionMapping& dm)
      : geom{Box2D::from_extents(4, 4), 0.0, 1.0, 0.0, 1.0},
        pgeom_{Box2D::from_extents(4, 4), Real(1), Real(2)},
        aux(ba, dm, 1, 0) {
    fields_.phi = MultiFab(ba, dm, 1, 0);
  }

  // solve_fields() JOUET : peuple le champ g a partir de la densite-proxy x du bloc COURANT (cellule
  // (0,0)). C'est le couplage champ<-densite que le transport (via la source) doit voir COHERENT.
  void solve_fields() {
    ++solve_calls;
    if (sp.empty() || sp[0].U.local_size() == 0)
      return;  // rang sans fab local (MPI mono-box) : no-op
    const ConstArray4 a = sp[0].U.fab(0).const_array();
    g = a(0, 0, 0);  // g <- x courant
  }
};

using Stepper = adc::stepper::SystemStepper<MockImpl>;

static void fill_ic(MultiFab& U, Real x0, Real y0) {
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 a = U.fab(li).array();
    const Box2D b = U.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        a(i, j, 0) = x0;
        a(i, j, 1) = y0;
      }
  }
}

// Construit un Impl jouet a UN bloc avec le transport H (x += h y). L'etage source generique
// (source_step) est branche PAR L'APPELANT apres que impl a son adresse FINALE (la fermeture capture
// &impl->g, peuple par solve_fields). eff_dt/n sous-pas non utiles ici (operateur lineaire, n=1).
static MockImpl make_impl(Real x0, Real y0) {
  const int N = 4;
  Box2D dom = Box2D::from_extents(N, N);
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, 1);
  MockImpl impl(ba, dm);

  MockImpl::Species s;
  s.name = "toy";
  s.U = MultiFab(ba, dm, 2, 0);
  s.ncomp = 2;
  s.substeps = 1;
  s.evolve = true;
  s.stride = 1;
  s.gamma = 1.4;
  fill_ic(s.U, x0, y0);
  // H : transport exp(A h) -> x += h y (independant de g ; n sous-pas equivalents pour cet operateur lineaire).
  s.advance = [](MultiFab& U, Real h, int /*n*/) {
    for (int li = 0; li < U.local_size(); ++li) {
      Array4 a = U.fab(li).array();
      const Box2D b = U.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          a(i, j, 0) += h * a(i, j, 1);
    }
  };
  impl.sp.push_back(std::move(s));
  return impl;
}

// Branche l'etage source generique S (exp(B h) -> y += h g, g = champ resolu par solve_fields) sur le
// bloc 0 de @p impl (capture l'adresse FINALE de impl).
static void attach_source(MockImpl& impl) {
  MockImpl* self = &impl;
  impl.sp[0].source_step = [self](MultiFab& U, Real h) {
    const Real g = self->g;  // champ resolu (g <- x courant) par le dernier solve_fields
    for (int li = 0; li < U.local_size(); ++li) {
      Array4 a = U.fab(li).array();
      const Box2D b = U.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          a(i, j, 1) += h * g;
    }
  };
}

// Erreur max vs solution exacte a T, en n macro-pas, schema strang/lie.
static double run(bool strang, int n, double T, Real x0, Real y0) {
  MockImpl impl = make_impl(x0, y0);
  attach_source(impl);
  Stepper st(&impl);
  st.set_scheme(strang ? adc::stepper::SplitScheme::Strang : adc::stepper::SplitScheme::Lie);

  const double dt = T / n;
  st.advance(dt, n);

  const double xe = (double)x0 * std::cosh(T) + (double)y0 * std::sinh(T);
  const double ye = (double)x0 * std::sinh(T) + (double)y0 * std::cosh(T);
  double err = 0;
  const ConstArray4 a = impl.sp[0].U.fab(0).const_array();
  const Box2D b = impl.sp[0].U.box(0);
  for (int j = b.lo[1]; j <= b.hi[1]; ++j)
    for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
      err = std::fmax(err, std::fabs(a(i, j, 0) - xe));
      err = std::fmax(err, std::fabs(a(i, j, 1) - ye));
    }
  return err;
}

// Compte des solve_fields sur k macro-pas pour le schema donne (structure de la consistance phi).
static int count_solves(bool strang, int k) {
  MockImpl impl = make_impl(Real(1), Real(0));
  attach_source(impl);
  Stepper st(&impl);
  st.set_scheme(strang ? adc::stepper::SplitScheme::Strang : adc::stepper::SplitScheme::Lie);
  st.advance(0.01, k);
  return impl.solve_calls;
}

// Lit l'etat final (x, y) d'un run a UN seul macro-pas, avec source et/ou transport desactives, pour
// les cas degeneres (Strang == transport pur si source nulle ; Strang == source pure si transport nul).
static void run_one_step(bool with_transport, bool with_source, double dt, Real x0, Real y0,
                         double& xf, double& yf) {
  MockImpl impl = make_impl(x0, y0);
  if (!with_transport)
    impl.sp[0].advance = [](MultiFab&, Real, int) {};  // H = identite
  if (with_source)
    attach_source(impl);  // sinon source_step reste nullptr -> S = no-op (etage absent)

  Stepper st(&impl);
  st.set_scheme(adc::stepper::SplitScheme::Strang);
  st.step(dt);
  const ConstArray4 a = impl.sp[0].U.fab(0).const_array();
  xf = a(0, 0, 0);
  yf = a(0, 0, 1);
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const double T = 0.8;
  const Real x0 = 1.0, y0 = 0.0;

  // --- (A) ORDRE TEMPOREL observe : Strang ~2, Lie ~1 -------------------------------------------------
  const double sE1 = run(true, 20, T, x0, y0), sE2 = run(true, 40, T, x0, y0);
  const double lE1 = run(false, 20, T, x0, y0), lE2 = run(false, 40, T, x0, y0);
  const double sOrder = std::log2(sE1 / sE2);
  const double lOrder = std::log2(lE1 / lE2);
  std::printf("Strang : err(20)=%.3e err(40)=%.3e ordre=%.2f\n", sE1, sE2, sOrder);
  std::printf("Lie    : err(20)=%.3e err(40)=%.3e ordre=%.2f\n", lE1, lE2, lOrder);
  chk(sOrder > 1.8 && sOrder < 2.2, "strang_ordre_2");
  chk(lOrder > 0.8 && lOrder < 1.3, "lie_ordre_1");
  chk(sE1 < lE1, "strang_plus_precis_que_lie");

  // --- (B) NON-REGRESSION Lie : le chemin Lie est INCHANGE (= H(dt) puis S(dt), 1 seul solve) ---------
  // On reproduit le pas Lie a la main (H(dt) ; S(dt) avec g <- x apres H) et on compare au stepper Lie.
  {
    const double dt = 0.05;
    double xs, ys;
    {  // stepper Lie, 1 pas
      MockImpl impl = make_impl(x0, y0);
      attach_source(impl);
      Stepper st(&impl);
      st.set_scheme(adc::stepper::SplitScheme::Lie);
      st.step(dt);
      const ConstArray4 a = impl.sp[0].U.fab(0).const_array();
      xs = a(0, 0, 0);
      ys = a(0, 0, 1);
    }
    // Reference manuelle Lie : solve (g=x0) ; H : x = x0 + dt*y0 ; S : y = y0 + dt*g(=x0).
    const double xref = (double)x0 + dt * (double)y0;
    const double yref = (double)y0 + dt * (double)x0;
    chk(std::fabs(xs - xref) < 1e-14, "lie_x_inchange");
    chk(std::fabs(ys - yref) < 1e-14, "lie_y_inchange");
    chk(count_solves(false, 5) == 5, "lie_un_solve_par_pas");  // 1 solve_fields / macro-pas
  }

  // --- (C) CONSISTANCE phi : Strang RE-RESOUT solve_fields entre les etages (3 / macro-pas) -----------
  chk(count_solves(true, 5) == 15, "strang_trois_solves_par_pas");  // 3 solve_fields / macro-pas

  // --- (D) DEGENERESCENCE source nulle : Strang(H(dt/2) S0 H(dt/2)) == transport pur H(dt) ------------
  {
    const double dt = 0.3;
    double xf, yf;
    run_one_step(/*transport*/ true, /*source*/ false, dt, x0, y0, xf, yf);
    // H seul : 2 demi-avances de transport pur = H(dt/2) puis H(dt/2). y inchange ; x += dt*y0.
    const double xref = (double)x0 + dt * (double)y0;
    chk(std::fabs(xf - xref) < 1e-14 && std::fabs(yf - (double)y0) < 1e-14,
        "strang_source_nulle_eq_transport");
  }

  // --- (E) DEGENERESCENCE transport nul : Strang(H0 S(dt) H0) == source pure S(dt) --------------------
  {
    const double dt = 0.3;
    double xf, yf;
    run_one_step(/*transport*/ false, /*source*/ true, dt, x0, y0, xf, yf);
    // H = identite -> g = x0 a chaque solve. S(dt) plein : y = y0 + dt*x0 ; x inchange.
    const double xref = (double)x0;
    const double yref = (double)y0 + dt * (double)x0;
    chk(std::fabs(xf - xref) < 1e-14 && std::fabs(yf - yref) < 1e-14,
        "strang_transport_nul_eq_source");
  }

  if (fails == 0)
    std::printf("OK test_strang_splitting\n");
  return fails == 0 ? 0 : 1;
}
