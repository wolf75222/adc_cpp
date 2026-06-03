#include <adc/runtime/system.hpp>

#include <adc/core/variables.hpp>  // VariableSet + VariableRole : descripteur a roles porte par chaque bloc
#include <adc/runtime/block_builder.hpp>  // GridContext + make_block/make_max_speed (fermetures compilees)
#include <adc/runtime/model_factory.hpp>  // detail::dispatch_model + briques compilees
#include <adc/numerics/elliptic/geometric_mg.hpp>
#include <adc/numerics/elliptic/poisson_fft_solver.hpp>
#include <adc/numerics/time/implicit_stepper.hpp>   // backward_euler_source
#include <adc/numerics/time/time_steppers.hpp>      // ForwardEuler, SSPRK2Step (math RK du coeur)
#include <adc/numerics/spatial_operator.hpp>     // assemble_rhs, SourceFreeModel, max_wave_speed_mf, load_state

#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>  // sum
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>  // fill_ghosts, fill_boundary
#include <adc/runtime/dynamic_model.hpp>  // IModel : modele charge a l'execution (bloc dynamique)

#include <algorithm>
#include <cmath>
#include <dlfcn.h>  // dlopen/dlsym : chargement d'une brique generee (.so)
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <variant>

namespace adc {

// Residu hote R = -div F* + S(U, aux) (Rusanov, a_max GLOBAL comme adc.PythonFlux, periodique)
// calcule via un IModel : sert au bloc DYNAMIQUE (modele charge a l'execution, dispatch virtuel,
// hors GPU). @p recon = ordre de reconstruction MUSCL des etats de face sur les variables
// conservatives : 0 = aucune (ordre 1), 1 = minmod, 2 = van Leer (ordre 2, TVD). Le flux ET la
// source recoivent l'aux par cellule (phi, grad phi) : un modele couple (transport ExB, force) marche,
// pas seulement Euler. @p AUX est l'aux du systeme (3 comp phi/grad_x/grad_y, composante-majeur ; vide
// => nul). U et le retour en disposition composante-majeur (c*n*n + j*n + i).
namespace {
ADC_HD inline double limited_slope(double am, double ap, int recon) {
  if (recon == 1) {  // minmod : TVD, robuste
    if (am * ap <= 0) return 0.0;
    return (std::fabs(am) < std::fabs(ap)) ? am : ap;
  }
  if (recon == 2) {  // van Leer : plus lisse aux extrema
    const double ab = am * ap;
    if (ab <= 0) return 0.0;
    return 2.0 * ab / (am + ap);
  }
  return 0.0;  // ordre 1 (pas de pente)
}

template <int NV>
std::vector<double> host_residual(const IModel<NV>& m, const std::vector<double>& U,
                                  const std::vector<double>& AUX, int n, double dx, int recon) {
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  auto idx = [&](int i, int j) {
    return static_cast<std::size_t>((j + n) % n) * n + ((i + n) % n);
  };
  auto cell = [&](int i, int j) {
    StateVec<NV> u;
    for (int c = 0; c < NV; ++c) u[c] = U[static_cast<std::size_t>(c) * nn + idx(i, j)];
    return u;
  };
  auto aux_at = [&](int i, int j) {  // aux par cellule, periodique ; vide => nul
    Aux a{};
    if (AUX.size() >= 3 * nn) {
      const std::size_t k = idx(i, j);
      a.phi = AUX[k];
      a.grad_x = AUX[nn + k];
      a.grad_y = AUX[2 * nn + k];
      if (AUX.size() >= 4 * nn) a.B_z = AUX[3 * nn + k];  // champ extra B_z si marshale (n_aux > 3)
    }
    return a;
  };
  // pente limitee de la cellule (i,j) dans la direction dir (sur les variables conservatives)
  auto slope = [&](int i, int j, int dir) {
    StateVec<NV> s{};
    if (recon == 0) return s;
    StateVec<NV> Uc = cell(i, j);
    StateVec<NV> Um = (dir == 0) ? cell(i - 1, j) : cell(i, j - 1);
    StateVec<NV> Up = (dir == 0) ? cell(i + 1, j) : cell(i, j + 1);
    for (int c = 0; c < NV; ++c) s[c] = limited_slope(Uc[c] - Um[c], Up[c] - Uc[c], recon);
    return s;
  };
  double amax = 0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      StateVec<NV> u = cell(i, j);
      Aux a = aux_at(i, j);
      double s = std::max(m.max_wave_speed(u, a, 0), m.max_wave_speed(u, a, 1));
      if (s > amax) amax = s;
    }
  // flux numerique a la face +dir de la cellule (i,j) : etats MUSCL (cellule + voisine) puis Rusanov
  auto face_flux = [&](int i, int j, int dir) {
    const int in = (dir == 0) ? i + 1 : i, jn = (dir == 0) ? j : j + 1;
    StateVec<NV> Uc = cell(i, j), Un = cell(in, jn);
    StateVec<NV> sc = slope(i, j, dir), sn = slope(in, jn, dir), L, R;
    for (int c = 0; c < NV; ++c) {
      L[c] = Uc[c] + 0.5 * sc[c];  // extrapolation vers la face depuis chaque cote
      R[c] = Un[c] - 0.5 * sn[c];
    }
    StateVec<NV> FL = m.flux(L, aux_at(i, j), dir), FR = m.flux(R, aux_at(in, jn), dir), f;
    for (int c = 0; c < NV; ++c) f[c] = 0.5 * (FL[c] + FR[c]) - 0.5 * amax * (R[c] - L[c]);
    return f;
  };
  std::vector<double> Rout(static_cast<std::size_t>(NV) * nn, 0.0);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      StateVec<NV> Uc = cell(i, j);
      StateVec<NV> Fxr = face_flux(i, j, 0), Fxl = face_flux(i - 1, j, 0);
      StateVec<NV> Fyr = face_flux(i, j, 1), Fyl = face_flux(i, j - 1, 1);
      StateVec<NV> S = m.source(Uc, aux_at(i, j));  // terme source genere (force, etc.)
      for (int c = 0; c < NV; ++c)
        Rout[static_cast<std::size_t>(c) * nn + static_cast<std::size_t>(j) * n + i] =
            -((Fxr[c] - Fxl[c]) + (Fyr[c] - Fyl[c])) / dx + S[c];
    }
  return Rout;
}

// Indice de la composante portant @p role dans @p vs, ou @p fallback si le bloc ne renseigne
// pas ce role (bloc dynamique / compile : descripteur sans roles). Permet aux couplages de viser
// une composante par son SENS sans coder l'indice en dur, tout en restant retro-compatible.
int role_index(const VariableSet& vs, VariableRole role, int fallback) {
  const int c = vs.index_of(role);
  return c >= 0 ? c : fallback;
}
}  // namespace

struct System::Impl {
  // Fermetures compilees figees a l'ajout du bloc (modele compose + schema spatial + temps).
  // Type-erased SEULEMENT au niveau de la liste de blocs ; le noyau reste compile.
  struct Species {
    std::string name;
    MultiFab U;
    int ncomp;
    int substeps;                                             // sous-pas statiques (add_block)
    bool evolve;                                              // false = espece gelee (fond fixe non avance)
    double gamma;                                             // pour l'energie au repos (4 var)
    std::function<void(MultiFab&, Real, int)> advance;        // (U, dt, n) : n sous-pas de dt/n
    std::function<void(MultiFab&, MultiFab&)> rhs_into;        // R <- -div F + S (Poisson fige)
    std::function<Real(const MultiFab&)> max_speed;           // max |vitesse d'onde| du bloc
    std::function<void(const MultiFab&, MultiFab&)> add_poisson_rhs;  // += elliptic_rhs(U)
    // Descripteur des variables conservatives / primitives (noms + ROLES physiques) du bloc.
    // Les roles (fournis par M::conservative_vars()) permettent aux couplages inter-especes de
    // viser une composante par son SENS (qte de mvt, energie) au lieu d'un indice u[1]/u[3] en dur.
    VariableSet cons_vars, prim_vars;
  };

  SystemConfig cfg;
  Geometry geom;
  BoxArray ba;
  DistributionMapping dm;
  BCRec bc_;        // CL transport (periodique ou Foextrap selon cfg.periodic)
  Box2D dom;
  Periodicity per_;
  bool periodic_;
  MultiFab aux;
  int aux_ncomp_ = kAuxBaseComps;     // largeur du canal aux PARTAGE (max des blocs ; >= 3)
  std::vector<Real> bz_field_;        // champ B_z(x) n*n row-major (vide si non fourni)
  int te_src_ = -1;                   // indice du bloc fluide source de T_e (-1 = aucune)
  std::vector<Species> sp;
  double t = 0;
  std::vector<std::function<void(Real)>> couplings;  // sources couplees inter-especes (splitting)

  // Configuration Poisson (solveur elliptique construit paresseusement).
  std::string p_rhs = "charge_density";
  std::string p_solver = "geometric_mg";
  std::string p_bc = "auto";
  std::string p_wall = "none";
  double p_wall_radius = 0.0;
  Real p_eps_ = 1;  // permittivite CONSTANTE : div(eps grad phi) = f <=> lap phi = f/eps
  bool has_eps_field_ = false;          // permittivite VARIABLE eps(x) fournie (porte par l'operateur)
  std::vector<double> p_eps_field_;     // champ eps(x), n*n row-major (si has_eps_field_)
  std::optional<std::variant<GeometricMG, PoissonFFTSolver>> ell_;

  explicit Impl(const SystemConfig& c)
      : cfg(c),
        geom{Box2D::from_extents(c.n, c.n), 0.0, c.L, 0.0, c.L},
        ba(std::vector<Box2D>{Box2D::from_extents(c.n, c.n)}),
        dm(1, n_ranks()),
        bc_(make_bc(c)),
        dom(Box2D::from_extents(c.n, c.n)),
        per_{c.periodic, c.periodic},
        periodic_(c.periodic),
        aux(ba, dm, kAuxBaseComps, 1) {}

  // Garantit une largeur aux >= ncomp (canal PARTAGE). Reallouer l'aux GARDE son adresse (membre :
  // les fermetures de bloc capturent &aux via grid_ctx) et re-applique B_z. No-op si deja assez large.
  void ensure_aux_width(int ncomp) {
    if (ncomp <= aux_ncomp_) return;
    aux_ncomp_ = ncomp;
    aux = MultiFab(ba, dm, aux_ncomp_, 1);
    apply_bz();
    apply_te();
  }

  // Peuple la composante B_z (indice kAuxBaseComps) du canal partage depuis bz_field_, sur les
  // cellules valides. No-op si B_z non fourni ou si aucun bloc ne le lit (largeur de base). Les
  // halos de B_z sont remplis par solve_fields (comme grad) ; field_postprocess n'ecrit que comp 0..2.
  void apply_bz() {
    if (bz_field_.empty() || aux_ncomp_ <= kAuxBaseComps) return;
    const int n = cfg.n;
    Array4 a = aux.fab(0).array();
    const Box2D v = aux.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        a(i, j, kAuxBaseComps) = bz_field_[static_cast<std::size_t>(j) * n + i];
  }

  // Composante canonique de T_e (apres phi/grad/B_z) ; cf. adc::Aux et AUX_CANONICAL cote DSL.
  static constexpr int kTeComp = kAuxBaseComps + 1;  // = 4

  // Peuple la composante T_e (temperature electronique) = p/rho du bloc fluide source te_src_.
  // RECALCULEE a chaque solve_fields (T_e varie avec le fluide, contrairement a B_z statique).
  // No-op si aucune source ou si aucun bloc ne lit T_e (largeur insuffisante). Le bloc source est
  // compressible (4 var) ; p = (gamma-1)(E - 0.5 rho|v|^2), T = p/rho.
  void apply_te() {
    if (te_src_ < 0 || aux_ncomp_ <= kTeComp) return;
    const Species& s = sp[static_cast<std::size_t>(te_src_)];
    const Real gm1 = Real(s.gamma) - Real(1);
    const ConstArray4 us = s.U.fab(0).const_array();
    Array4 a = aux.fab(0).array();
    const Box2D v = aux.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        const Real rho = us(i, j, 0), mx = us(i, j, 1), my = us(i, j, 2), E = us(i, j, 3);
        const Real p = gm1 * (E - Real(0.5) * (mx * mx + my * my) / rho);
        a(i, j, kTeComp) = p / rho;  // T = p / rho
      }
  }

  static BCRec make_bc(const SystemConfig& c) {
    BCRec b;  // periodique par defaut
    if (!c.periodic) b.xlo = b.xhi = b.ylo = b.yhi = BCType::Foextrap;
    return b;
  }

  Species& find(const std::string& name) {
    for (auto& s : sp)
      if (s.name == name) return s;
    throw std::runtime_error("System : bloc inconnu '" + name + "'");
  }
  const Species& find(const std::string& name) const {
    for (auto& s : sp)
      if (s.name == name) return s;
    throw std::runtime_error("System : bloc inconnu '" + name + "'");
  }
  int index(const std::string& name) const {
    for (std::size_t k = 0; k < sp.size(); ++k)
      if (sp[k].name == name) return static_cast<int>(k);
    throw std::runtime_error("System : bloc inconnu '" + name + "'");
  }

  // Sources de COUPLAGE inter-especes : appliquees par SPLITTING (un pas additif explicite de dt)
  // APRES le transport de chaque bloc. Chaque couplage est un for_each_cell (kernel DEVICE) lisant /
  // mettant a jour plusieurs blocs au meme point ; ils s'ordonnent apres le transport sur le meme
  // espace d'execution, donc plus de device_fence prealable (plus d'acces hote).
  void apply_couplings(Real dt) {
    if (couplings.empty()) return;
    for (auto& c : couplings) c(dt);
  }

  // --- solveur elliptique (Poisson de systeme) -----------------------------
  BCRec poisson_bc() {
    std::string mode = p_bc;
    if (mode == "auto") mode = (p_wall == "circle" || !cfg.periodic) ? "dirichlet" : "periodic";
    BCRec b;
    if (mode == "periodic") return b;
    if (mode == "dirichlet") {
      b.xlo = b.xhi = b.ylo = b.yhi = BCType::Dirichlet;
      return b;
    }
    if (mode == "neumann") {
      b.xlo = b.xhi = b.ylo = b.yhi = BCType::Foextrap;
      return b;
    }
    throw std::runtime_error("System::set_poisson : bc inconnu '" + mode + "'");
  }
  std::function<bool(Real, Real)> wall_active() {
    if (p_wall == "none") return {};
    if (p_wall == "circle") {
      const double cx = 0.5 * cfg.L, cy = 0.5 * cfg.L, R = p_wall_radius;
      return [cx, cy, R](Real x, Real y) { return std::hypot(x - cx, y - cy) < R; };
    }
    throw std::runtime_error("System::set_poisson : wall inconnu '" + p_wall + "'");
  }
  void ensure_elliptic() {
    if (ell_) return;
    if (p_rhs != "charge_density")
      throw std::runtime_error("System::set_poisson : rhs '" + p_rhs +
                               "' inconnu (seul 'charge_density')");
    const BCRec pbc = poisson_bc();
    std::function<bool(Real, Real)> active = wall_active();
    if (p_solver == "fft") {
      if (active)
        throw std::runtime_error("System : solver 'fft' incompatible avec une paroi -> 'geometric_mg'");
      if (has_eps_field_)
        throw std::runtime_error("System : solver 'fft' a coefficient CONSTANT, incompatible avec un "
                                 "champ eps(x) variable -> utiliser solver='geometric_mg'");
      ell_.emplace(std::in_place_type<PoissonFFTSolver>, geom, ba, pbc, active);
    } else if (p_solver == "geometric_mg") {
      ell_.emplace(std::in_place_type<GeometricMG>, geom, ba, pbc, std::move(active));
      if (has_eps_field_) apply_epsilon_field();  // operateur div(eps grad phi) a eps(x) variable
    } else {
      throw std::runtime_error("System::set_poisson : solver '" + p_solver +
                               "' inconnu (geometric_mg|fft)");
    }
  }
  // Installe le champ eps(x) (n*n row-major) sur le GeometricMG : l'operateur passe a
  // div(eps grad phi) = f, eps PORTE PAR L'OPERATEUR (coefficient de face harmonique, ordre 2),
  // sans mise a l'echelle 1/eps du second membre. Seul GeometricMG supporte ce coefficient variable.
  void apply_epsilon_field() {
    GeometricMG& mg = std::get<GeometricMG>(*ell_);
    MultiFab eps_fine(ba, dm, 1, 0);
    Array4 e = eps_fine.fab(0).array();
    const Box2D v = eps_fine.box(0);
    const int n = cfg.n;
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        e(i, j, 0) = static_cast<Real>(p_eps_field_[static_cast<std::size_t>(j) * n + i]);
    mg.set_epsilon(eps_fine);  // copie sur le niveau fin + restriction (average_down) aux grossiers
  }
  MultiFab& ell_rhs() {
    return std::visit([](auto& e) -> MultiFab& { return e.rhs(); }, *ell_);
  }
  MultiFab& ell_phi() {
    return std::visit([](auto& e) -> MultiFab& { return e.phi(); }, *ell_);
  }
  void ell_solve() {
    std::visit([](auto& e) { e.solve(); }, *ell_);
  }

  // --- schemas spatiaux compiles -------------------------------------------
  // Evaluateur methode-des-lignes d'un bloc (L/F/Model figes) : ghosts puis R = -div F + S.
  // Construction des fermetures de bloc (avance + residu + Poisson) deplacee en en-tete
  // (adc/runtime/block_builder.hpp : make_block / make_max_speed / make_poisson_rhs) afin que le
  // chemin template de production soit instanciable hors de cette unite (compilation AOT d'un
  // modele genere). Ici on ne fournit que le contexte de grille a leur passer.
  GridContext grid_ctx() { return GridContext{dom, bc_, geom, &aux}; }

  void solve_fields() {
    ensure_elliptic();
    MultiFab& rhs = ell_rhs();
    rhs.set_val(Real(0));
    for (auto& s : sp) s.add_poisson_rhs(s.U, rhs);  // f = Sum_s elliptic_rhs_s(u_s)
    // Permittivite CONSTANTE : div(eps grad phi) = f <=> lap phi = f/eps, donc on met le rhs a
    // l'echelle 1/eps. Avec un champ eps(x) VARIABLE on NE le fait PAS : l'operateur GeometricMG
    // porte eps directement (cf. apply_epsilon_field), le rhs reste f tel quel.
    if (!has_eps_field_ && p_eps_ != Real(1)) {
      const Real inv = Real(1) / p_eps_;
      for (int li = 0; li < rhs.local_size(); ++li) {
        Array4 r = rhs.fab(li).array();
        const Box2D v = rhs.box(li);
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i) r(i, j, 0) *= inv;
      }
    }
    ell_solve();
    device_fence();
    const Real dx = geom.dx(), dy = geom.dy();
    const ConstArray4 p = ell_phi().fab(0).const_array();
    Array4 a = aux.fab(0).array();
    const Box2D v = aux.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        a(i, j, 0) = p(i, j);
        a(i, j, 1) = (p(i + 1, j) - p(i - 1, j)) / (2 * dx);
        a(i, j, 2) = (p(i, j + 1) - p(i, j - 1)) / (2 * dy);
      }
    apply_te();  // T_e = p/rho du bloc fluide source, recalculee a chaque solve (B_z, comp 3, preservee)
    if (periodic_)
      fill_boundary(aux, dom, per_);
    else
      fill_ghosts(aux, dom, bc_);  // extrapolation au bord (paroi / sortie libre)
  }

  std::vector<double> copy_comp0(const MultiFab& mf) const {
    device_fence();
    const ConstArray4 u = mf.fab(0).const_array();
    const Box2D v = mf.box(0);
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(u(i, j, 0));
    return out;
  }
  std::vector<double> copy_state(const MultiFab& mf, int ncomp) const {
    device_fence();
    const ConstArray4 u = mf.fab(0).const_array();
    const Box2D v = mf.box(0);
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(ncomp) * v.nx() * v.ny());
    for (int c = 0; c < ncomp; ++c)
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(u(i, j, c));
    return out;
  }
  void write_state(MultiFab& mf, int ncomp, const std::vector<double>& in) {
    const Box2D v = mf.box(0);
    const std::size_t need = static_cast<std::size_t>(ncomp) * v.nx() * v.ny();
    if (in.size() != need)
      throw std::runtime_error("System::set_state : taille != ncomp*n*n");
    Array4 u = mf.fab(0).array();
    std::size_t k = 0;
    for (int c = 0; c < ncomp; ++c)
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) u(i, j, c) = in[k++];
  }

  // Construit un bloc DYNAMIQUE (modele IModel<NV> charge depuis le .so @p h) et l'ajoute. Le
  // shared_ptr possede le modele : il appelle adc_destroy_model puis ferme le .so a la destruction.
  template <int NV>
  static void push_dynamic(Impl* P, const std::string& name, void* h, int substeps,
                           std::vector<std::string> names, int recon) {
    auto mk = reinterpret_cast<void* (*)()>(dlsym(h, "adc_make_model"));
    auto del = reinterpret_cast<void (*)(void*)>(dlsym(h, "adc_destroy_model"));
    if (!mk || !del) {
      dlclose(h);
      throw std::runtime_error("add_dynamic_block : adc_make_model / adc_destroy_model absents du .so");
    }
    std::shared_ptr<IModel<NV>> im(static_cast<IModel<NV>*>(mk()),
                                   [del, h](IModel<NV>* p) { del(p); dlclose(h); });
    // Le modele charge peut lire des champs aux supplementaires (n_aux > 3, p.ex. B_z) : on
    // elargit le canal aux PARTAGE pour que set_magnetic_field le peuple et que le marshaling hote
    // les transporte. Modele de base (3) -> no-op. Les fermetures lisent P->aux_ncomp_ a l'appel.
    P->ensure_aux_width(im->n_aux());
    const int n = P->cfg.n;
    const double dx = P->cfg.L / P->cfg.n;

    std::function<void(MultiFab&, MultiFab&)> rhs_into = [P, im, n, dx, recon](MultiFab& U,
                                                                               MultiFab& R) {
      P->write_state(R, NV, host_residual<NV>(*im, P->copy_state(U, NV),
                                              P->copy_state(P->aux, P->aux_ncomp_), n, dx, recon));
    };
    std::function<Real(const MultiFab&)> max_speed = [P, im, n](const MultiFab& U) -> Real {
      std::vector<double> u = P->copy_state(U, NV), aux = P->copy_state(P->aux, P->aux_ncomp_);
      const std::size_t nn = static_cast<std::size_t>(n) * n;
      Real mx = 0;
      for (std::size_t c0 = 0; c0 < nn; ++c0) {
        StateVec<NV> s;
        for (int c = 0; c < NV; ++c) s[c] = u[static_cast<std::size_t>(c) * nn + c0];
        Aux a{};
        if (aux.size() >= 3 * nn) {
          a.phi = aux[c0];
          a.grad_x = aux[nn + c0];
          a.grad_y = aux[2 * nn + c0];
          if (aux.size() >= 4 * nn) a.B_z = aux[3 * nn + c0];
        }
        Real v = std::max(im->max_wave_speed(s, a, 0), im->max_wave_speed(s, a, 1));
        if (v > mx) mx = v;
      }
      return mx;
    };
    std::function<void(MultiFab&, Real, int)> advance = [P, im, n, dx, recon](MultiFab& U, Real dt,
                                                                              int nsub) {
      const Real hh = dt / nsub;
      const std::vector<double> aux = P->copy_state(P->aux, P->aux_ncomp_);  // aux gelee (splitting)
      for (int s = 0; s < nsub; ++s) {  // Euler explicite par sous-pas (chemin hote, prototype)
        std::vector<double> u = P->copy_state(U, NV);
        std::vector<double> res = host_residual<NV>(*im, u, aux, n, dx, recon);
        for (std::size_t k = 0; k < u.size(); ++k) u[k] += hh * res[k];
        P->write_state(U, NV, u);
      }
    };
    // Contribution du bloc dynamique au Poisson de systeme : rhs += elliptic_rhs(U) par cellule.
    // Modele sans elliptique => elliptic_rhs vaut 0 (aucun effet), donc retro-compatible.
    std::function<void(const MultiFab&, MultiFab&)> add_poisson =
        [P, im, n](const MultiFab& U, MultiFab& rhs) {
          std::vector<double> u = P->copy_state(U, NV);
          const std::size_t nn = static_cast<std::size_t>(n) * n;
          Array4 r = rhs.fab(0).array();
          const Box2D v = rhs.box(0);
          for (int j = v.lo[1]; j <= v.hi[1]; ++j)
            for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
              const std::size_t k = static_cast<std::size_t>(j - v.lo[1]) * n + (i - v.lo[0]);
              StateVec<NV> s;
              for (int c = 0; c < NV; ++c) s[c] = u[static_cast<std::size_t>(c) * nn + k];
              r(i, j, 0) += im->elliptic_rhs(s);
            }
        };
    if (names.empty())
      for (int c = 0; c < NV; ++c) names.push_back("u" + std::to_string(c));

    Species block{name, MultiFab(P->ba, P->dm, NV, 2), NV, substeps, true, 1.4,
                  std::move(advance), std::move(rhs_into), std::move(max_speed), std::move(add_poisson)};
    // Bloc dynamique : descripteur a noms seuls (pas de roles ; les couplages retombent alors
    // sur les indices historiques via role_index).
    block.cons_vars = {VariableKind::Conservative, names, NV, {}};
    block.prim_vars = {VariableKind::Primitive, names, NV, {}};
    P->sp.push_back(std::move(block));
    P->sp.back().U.set_val(Real(0));
  }
};

System::System(const SystemConfig& c) : p_(std::make_unique<Impl>(c)) {}
System::~System() = default;
System::System(System&&) noexcept = default;
System& System::operator=(System&&) noexcept = default;

void System::add_block(const std::string& name, const ModelSpec& model,
                       const std::string& limiter, const std::string& riemann,
                       const std::string& recon, const std::string& time, int substeps,
                       bool evolve) {
  Impl* P = p_.get();
  if (substeps < 1) throw std::runtime_error("System::add_block : substeps >= 1");
  if (time != "explicit" && time != "imex")
    throw std::runtime_error("System::add_block : time 'explicit' | 'imex' (recu '" + time +
                             "')");
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error("System::add_block : recon 'conservative' | 'primitive' (recu '" +
                             recon + "')");
  const bool imex = (time == "imex");
  const bool recon_prim = (recon == "primitive");

  int ncomp = 1;
  BlockClosures clo;
  std::function<Real(const MultiFab&)> max_speed;
  std::function<void(const MultiFab&, MultiFab&)> add_poisson_rhs;
  VariableSet cons_vs, prim_vs;
  const GridContext ctx = P->grid_ctx();
  // Le modele est compose a partir des briques designees par la spec ; le visiteur cable les
  // fermetures (constructeurs en en-tete, instanciables AOT). ncomp = n_vars du modele compose ;
  // set_density s'y adapte. Les noms de variables viennent du descripteur Variables porte par le
  // modele (brique Vars), source unique de verite.
  detail::dispatch_model(model, [&](auto m) {
    using M = decltype(m);
    ncomp = M::n_vars;
    clo = make_block(m, limiter, riemann, ctx, imex, recon_prim);
    max_speed = make_max_speed(m, ctx);
    add_poisson_rhs = make_poisson_rhs(m);
    cons_vs = M::conservative_vars();  // noms + ROLES physiques (source unique de verite)
    prim_vs = M::primitive_vars();
  });
  // Installation commune (meme chemin que add_compiled_model pour un modele genere par le DSL) :
  // les fermetures tournent sur les MultiFab REELS du System (halos MPI via fill_boundary, device
  // via Kokkos), sans recopie.
  install_block(name, ncomp, cons_vs, prim_vs, model.gamma, std::move(clo), std::move(max_speed),
                std::move(add_poisson_rhs), substeps, evolve);
}

// Contexte de grille reel (maillage + CL + aux) : sert au gabarit add_compiled_model pour fabriquer
// les fermetures d'un modele compile AOT sur les vrais champs du System (parite native, sans marshaling).
GridContext System::grid_context() { return p_->grid_ctx(); }

// Installe un bloc a partir de fermetures deja fabriquees (par dispatch_model cote add_block, ou par
// block_builder cote add_compiled_model). Centralise la creation de l'espece (U, noms, schema).
void System::install_block(const std::string& name, int ncomp,
                           const VariableSet& cons_vars, const VariableSet& prim_vars, double gamma,
                           BlockClosures closures, std::function<Real(const MultiFab&)> max_speed,
                           std::function<void(const MultiFab&, MultiFab&)> poisson_rhs,
                           int substeps, bool evolve) {
  Impl* P = p_.get();
  P->sp.push_back(Impl::Species{name, MultiFab(P->ba, P->dm, ncomp, 2), ncomp, substeps, evolve,
                                gamma, std::move(closures.advance), std::move(closures.rhs_into),
                                std::move(max_speed), std::move(poisson_rhs)});
  P->sp.back().U.set_val(Real(0));
  P->sp.back().cons_vars = cons_vars;
  P->sp.back().prim_vars = prim_vars;
}

void System::add_dynamic_block(const std::string& name, const std::string& so_path, int substeps,
                               const std::vector<std::string>& names, const std::string& recon) {
  if (substeps < 1) throw std::runtime_error("System::add_dynamic_block : substeps >= 1");
  int recon_id = 0;  // ordre de reconstruction MUSCL des etats de face (conservatif)
  if (recon == "none") recon_id = 0;
  else if (recon == "minmod") recon_id = 1;
  else if (recon == "vanleer") recon_id = 2;
  else throw std::runtime_error("System::add_dynamic_block : recon 'none' | 'minmod' | 'vanleer' "
                                "(recu '" + recon + "')");
  void* h = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!h) {
    const char* e = dlerror();
    throw std::runtime_error("add_dynamic_block : dlopen('" + so_path + "') : " +
                             std::string(e ? e : "?"));
  }
  auto nv_fn = reinterpret_cast<int (*)()>(dlsym(h, "adc_model_nvars"));
  if (!nv_fn) {
    dlclose(h);
    throw std::runtime_error("add_dynamic_block : adc_model_nvars absent du .so");
  }
  const int nv = nv_fn();
  switch (nv) {
    case 1: Impl::push_dynamic<1>(p_.get(), name, h, substeps, names, recon_id); break;
    case 3: Impl::push_dynamic<3>(p_.get(), name, h, substeps, names, recon_id); break;
    case 4: Impl::push_dynamic<4>(p_.get(), name, h, substeps, names, recon_id); break;
    default:
      dlclose(h);
      throw std::runtime_error("add_dynamic_block : n_vars=" + std::to_string(nv) +
                               " non supporte (1, 3, 4)");
  }
}

void System::add_compiled_block(const std::string& name, const std::string& so_path,
                                const std::string& limiter, const std::string& riemann,
                                const std::string& recon, const std::string& time, int substeps,
                                const std::vector<std::string>& names) {
  Impl* P = p_.get();
  if (substeps < 1) throw std::runtime_error("System::add_compiled_block : substeps >= 1");
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error("System::add_compiled_block : recon 'conservative' | 'primitive'");
  if (time != "explicit" && time != "imex")
    throw std::runtime_error("System::add_compiled_block : time 'explicit' | 'imex'");
  const int recon_prim = (recon == "primitive") ? 1 : 0;
  const int imex = (time == "imex") ? 1 : 0;

  void* h = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!h) {
    const char* e = dlerror();
    throw std::runtime_error("add_compiled_block : dlopen('" + so_path + "') : " +
                             std::string(e ? e : "?"));
  }
  // ABI extern "C" du bloc compile (compiled_block_abi.hpp). Le .so tourne le chemin de production
  // (assemble_rhs<Limiter, Flux>, SSPRK2/IMEX) sur le modele genere ; seuls des tableaux plats
  // transitent (aucun objet C++ partage a travers le dlopen, donc RTLD_LOCAL sans risque d'ABI).
  using nv_fn_t = int (*)();
  using res_fn_t = void (*)(const double*, double*, const double*, int, double, double, int,
                            const char*, const char*, int);
  using adv_fn_t = void (*)(double*, const double*, int, double, double, int, const char*,
                            const char*, int, int, double, int);
  using max_fn_t = double (*)(const double*, const double*, int, double, double, int);
  using poi_fn_t = void (*)(const double*, double*, int);
  auto nv_fn = reinterpret_cast<nv_fn_t>(dlsym(h, "adc_model_nvars"));
  auto res_fn = reinterpret_cast<res_fn_t>(dlsym(h, "adc_compiled_residual"));
  auto adv_fn = reinterpret_cast<adv_fn_t>(dlsym(h, "adc_compiled_advance"));
  auto max_fn = reinterpret_cast<max_fn_t>(dlsym(h, "adc_compiled_max_speed"));
  auto poi_fn = reinterpret_cast<poi_fn_t>(dlsym(h, "adc_compiled_poisson_rhs"));
  if (!nv_fn || !res_fn || !adv_fn || !max_fn || !poi_fn) {
    dlclose(h);
    throw std::runtime_error("add_compiled_block : ABI bloc compile absente du .so (regenerer via "
                             "dsl.compile_aot / compile_or_jit(mode='compile'))");
  }
  const int nv = nv_fn();
  std::shared_ptr<void> lib(h, [](void* p) { dlclose(p); });  // ferme le .so a la mort des fermetures
  const int n = P->cfg.n;
  const double dx = P->geom.dx(), dy = P->geom.dy();
  const int per = P->periodic_ ? 1 : 0;
  const std::string lim = limiter, riem = riemann;

  std::function<void(MultiFab&, MultiFab&)> rhs_into =
      [P, lib, res_fn, nv, n, dx, dy, per, lim, riem, recon_prim](MultiFab& U, MultiFab& R) {
        std::vector<double> u = P->copy_state(U, nv), a = P->copy_state(P->aux, 3);
        std::vector<double> r(static_cast<std::size_t>(nv) * n * n, 0.0);
        res_fn(u.data(), r.data(), a.data(), n, dx, dy, per, lim.c_str(), riem.c_str(), recon_prim);
        P->write_state(R, nv, r);
      };
  std::function<void(MultiFab&, Real, int)> advance =
      [P, lib, adv_fn, nv, n, dx, dy, per, lim, riem, recon_prim, imex](MultiFab& U, Real dt,
                                                                        int nsub) {
        std::vector<double> u = P->copy_state(U, nv), a = P->copy_state(P->aux, 3);
        adv_fn(u.data(), a.data(), n, dx, dy, per, lim.c_str(), riem.c_str(), recon_prim, imex,
               static_cast<double>(dt), nsub);
        P->write_state(U, nv, u);
      };
  std::function<Real(const MultiFab&)> max_speed =
      [P, lib, max_fn, nv, n, dx, dy, per](const MultiFab& U) -> Real {
        std::vector<double> u = P->copy_state(U, nv), a = P->copy_state(P->aux, 3);
        return max_fn(u.data(), a.data(), n, dx, dy, per);
      };
  std::function<void(const MultiFab&, MultiFab&)> add_poisson =
      [P, lib, poi_fn, nv, n](const MultiFab& U, MultiFab& rhs) {
        std::vector<double> u = P->copy_state(U, nv);
        std::vector<double> pr(static_cast<std::size_t>(n) * n, 0.0);
        poi_fn(u.data(), pr.data(), n);
        Array4 r = rhs.fab(0).array();
        const Box2D v = rhs.box(0);
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i)
            r(i, j, 0) += pr[static_cast<std::size_t>(j - v.lo[1]) * n + (i - v.lo[0])];
      };

  std::vector<std::string> nm = names;
  if (nm.empty())
    for (int c = 0; c < nv; ++c) nm.push_back("u" + std::to_string(c));
  Impl::Species block{name, MultiFab(P->ba, P->dm, nv, 2), nv, substeps, true, 1.4,
                      std::move(advance), std::move(rhs_into), std::move(max_speed),
                      std::move(add_poisson)};
  // Bloc compile AOT : descripteur a noms seuls (roles non transmis par l'ABI extern "C").
  block.cons_vars = {VariableKind::Conservative, nm, nv, {}};
  block.prim_vars = {VariableKind::Primitive, nm, nv, {}};
  P->sp.push_back(std::move(block));
  P->sp.back().U.set_val(Real(0));
}

void System::set_poisson(const std::string& rhs, const std::string& solver,
                         const std::string& bc, const std::string& wall, double wall_radius,
                         double epsilon) {
  if (epsilon == 0.0) throw std::runtime_error("System::set_poisson : epsilon != 0 requis");
  p_->p_rhs = rhs;
  p_->p_solver = solver;
  p_->p_bc = bc;
  p_->p_wall = wall;
  p_->p_wall_radius = wall_radius;
  p_->p_eps_ = static_cast<Real>(epsilon);
  p_->ell_.reset();
}

void System::set_epsilon_field(const std::vector<double>& eps) {
  const int n = p_->cfg.n;
  if (static_cast<int>(eps.size()) != n * n)
    throw std::runtime_error("System::set_epsilon_field : taille != n*n");
  for (double e : eps)
    if (!(e > 0.0))
      throw std::runtime_error("System::set_epsilon_field : permittivite eps(x) > 0 requise");
  p_->p_eps_field_ = eps;
  p_->has_eps_field_ = true;
  p_->ell_.reset();  // l'operateur sera reconstruit avec le champ eps au prochain solve_fields
}

void System::ensure_aux_width(int ncomp) { p_->ensure_aux_width(ncomp); }

void System::set_magnetic_field(const std::vector<double>& bz) {
  const int n = p_->cfg.n;
  if (static_cast<int>(bz.size()) != n * n)
    throw std::runtime_error("System::set_magnetic_field : taille != n*n");
  p_->bz_field_.assign(bz.begin(), bz.end());
  p_->apply_bz();  // applique tout de suite si un bloc lit deja B_z ; sinon conserve pour ensure_aux_width
}

void System::set_electron_temperature_from(const std::string& name) {
  const int idx = p_->index(name);  // leve si bloc inconnu
  if (p_->sp[static_cast<std::size_t>(idx)].ncomp != 4)
    throw std::runtime_error("System::set_electron_temperature_from : le bloc '" + name +
                             "' doit etre compressible (4 var : rho, rho u, rho v, E) pour T = p/rho");
  p_->te_src_ = idx;
  // T_e (comp canonique 4) DERIVE : recalcule a chaque solve_fields. Inerte tant qu'aucun bloc ne
  // lit T_e (n_aux=5 -> ensure_aux_width(5)), comme set_magnetic_field pour B_z.
  p_->apply_te();
}

void System::add_ionization(const std::string& electron, const std::string& ion,
                            const std::string& neutral, double rate) {
  Impl* P = p_.get();
  const int ie = P->index(electron), ii = P->index(ion), ig = P->index(neutral);
  const Real k = static_cast<Real>(rate);
  // Densite resolue par ROLE (comme add_collision / add_thermal_exchange), fallback comp 0 si le
  // bloc ne renseigne pas ses roles (bloc dynamique / compile). Un bloc rangeant sa densite ailleurs
  // que l'indice 0 reste correctement couple.
  const int de = role_index(P->sp[ie].cons_vars, VariableRole::Density, 0);
  const int di = role_index(P->sp[ii].cons_vars, VariableRole::Density, 0);
  const int dg = role_index(P->sp[ig].cons_vars, VariableRole::Density, 0);
  // Ionisation (operator-split, sur la densite) : taux r = k n_e n_g. Un neutre disparait, un ion et
  // un electron apparaissent : n_g -= dt r, n_i += dt r, n_e += dt r. La masse est transferee du
  // neutre vers l'ion (n_i + n_g conserve). Premiere brique de couplage ; le transfert de quantite
  // de mouvement / energie (especes fluides) est un raffinement ulterieur.
  P->couplings.push_back([P, ie, ii, ig, k, de, di, dg](Real dt) {
    Array4 ue = P->sp[ie].U.fab(0).array();
    Array4 ui = P->sp[ii].U.fab(0).array();
    Array4 ug = P->sp[ig].U.fab(0).array();
    for_each_cell(P->sp[ie].U.box(0), [=] ADC_HD(int i, int j) {  // sur device (lit n_e, n_g)
      const Real dn = dt * k * ue(i, j, de) * ug(i, j, dg);
      ug(i, j, dg) -= dn;
      ui(i, j, di) += dn;
      ue(i, j, de) += dn;
    });
  });
}

void System::add_collision(const std::string& a, const std::string& b, double rate) {
  Impl* P = p_.get();
  const int ia = P->index(a), ib = P->index(b);
  if (P->sp[ia].ncomp < 3 || P->sp[ib].ncomp < 3)
    throw std::runtime_error("System::add_collision : les deux blocs doivent porter une quantite "
                             "de mouvement (transport fluide >= 3 variables)");
  const Real k = static_cast<Real>(rate);
  // Composantes resolues par ROLE (qte de mvt x/y, densite) plutot que par indice litteral : un
  // bloc qui range ses variables autrement reste correctement couple. Fallback aux indices
  // historiques (1, 2, 0) si le bloc ne renseigne pas ses roles (bloc dynamique / compile).
  const VariableSet& va_set = P->sp[ia].cons_vars;
  const VariableSet& vb_set = P->sp[ib].cons_vars;
  const int mxa = role_index(va_set, VariableRole::MomentumX, 1);
  const int mya = role_index(va_set, VariableRole::MomentumY, 2);
  const int da = role_index(va_set, VariableRole::Density, 0);
  const int mxb = role_index(vb_set, VariableRole::MomentumX, 1);
  const int myb = role_index(vb_set, VariableRole::MomentumY, 2);
  const int db = role_index(vb_set, VariableRole::Density, 0);
  // Friction inter-especes (operator-split) : force F = k (u_a - u_b) sur la quantite de
  // mouvement, opposee sur chaque espece (qte de mvt totale conservee) ; les vitesses relaxent
  // l'une vers l'autre. L'echauffement par friction (energie) est un raffinement ulterieur
  // (neglige : convient aux especes isothermes, sans eq. d'energie).
  P->couplings.push_back([P, ia, ib, k, mxa, mya, da, mxb, myb, db](Real dt) {
    Array4 ua = P->sp[ia].U.fab(0).array();
    Array4 ub = P->sp[ib].U.fab(0).array();
    for_each_cell(P->sp[ia].U.box(0), [=] ADC_HD(int i, int j) {  // sur device
      const Real fx = dt * k * (ua(i, j, mxa) / ua(i, j, da) - ub(i, j, mxb) / ub(i, j, db));
      ua(i, j, mxa) -= fx; ub(i, j, mxb) += fx;
      const Real fy = dt * k * (ua(i, j, mya) / ua(i, j, da) - ub(i, j, myb) / ub(i, j, db));
      ua(i, j, mya) -= fy; ub(i, j, myb) += fy;
    });
  });
}

void System::add_thermal_exchange(const std::string& a, const std::string& b, double rate) {
  Impl* P = p_.get();
  const int ia = P->index(a), ib = P->index(b);
  if (P->sp[ia].ncomp != 4 || P->sp[ib].ncomp != 4)
    throw std::runtime_error("System::add_thermal_exchange : les deux blocs doivent porter une "
                             "energie (Euler compressible, 4 variables)");
  const Real k = static_cast<Real>(rate);
  const Real ga = static_cast<Real>(P->sp[ia].gamma), gb = static_cast<Real>(P->sp[ib].gamma);
  // Composantes resolues par ROLE (energie, qte de mvt x/y, densite) plutot que par indice litteral.
  // Fallback aux indices historiques (3, 1, 2, 0) si le bloc ne renseigne pas ses roles.
  const VariableSet& va_set = P->sp[ia].cons_vars;
  const VariableSet& vb_set = P->sp[ib].cons_vars;
  const int ea = role_index(va_set, VariableRole::Energy, 3);
  const int mxa = role_index(va_set, VariableRole::MomentumX, 1);
  const int mya = role_index(va_set, VariableRole::MomentumY, 2);
  const int da = role_index(va_set, VariableRole::Density, 0);
  const int eb = role_index(vb_set, VariableRole::Energy, 3);
  const int mxb = role_index(vb_set, VariableRole::MomentumX, 1);
  const int myb = role_index(vb_set, VariableRole::MomentumY, 2);
  const int db = role_index(vb_set, VariableRole::Density, 0);
  // Echange thermique (operator-split) : flux de chaleur q = k (T_a - T_b) sur l'energie, oppose
  // sur chaque espece (energie totale conservee) ; les temperatures relaxent. T = p/rho (a une
  // constante pres), p = (gamma-1)(E - 1/2 rho |u|^2). Transfere l'energie INTERNE (u inchange).
  P->couplings.push_back([P, ia, ib, k, ga, gb, ea, mxa, mya, da, eb, mxb, myb, db](Real dt) {
    Array4 ua = P->sp[ia].U.fab(0).array();
    Array4 ub = P->sp[ib].U.fab(0).array();
    for_each_cell(P->sp[ia].U.box(0), [=] ADC_HD(int i, int j) {  // sur device
      const Real ra = ua(i, j, da), rb = ub(i, j, db);
      const Real pa = (ga - Real(1)) * (ua(i, j, ea) -
          Real(0.5) * (ua(i, j, mxa) * ua(i, j, mxa) + ua(i, j, mya) * ua(i, j, mya)) / ra);
      const Real pb = (gb - Real(1)) * (ub(i, j, eb) -
          Real(0.5) * (ub(i, j, mxb) * ub(i, j, mxb) + ub(i, j, myb) * ub(i, j, myb)) / rb);
      const Real q = dt * k * (pa / ra - pb / rb);  // k (T_a - T_b), T = p/rho
      ua(i, j, ea) -= q;
      ub(i, j, eb) += q;
    });
  });
}

void System::set_density(const std::string& name, const std::vector<double>& rho) {
  Impl::Species& s = p_->find(name);
  const int n = p_->cfg.n;
  if (static_cast<int>(rho.size()) != n * n)
    throw std::runtime_error("System::set_density : taille != n*n");
  const Real gm1 = Real(s.gamma) - Real(1);
  Array4 u = s.U.fab(0).array();
  const Box2D v = s.U.box(0);
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
      const Real r = rho[static_cast<std::size_t>(j) * n + i];
      u(i, j, 0) = r;
      if (s.ncomp >= 3) { u(i, j, 1) = 0; u(i, j, 2) = 0; }  // qte de mvt au repos
      if (s.ncomp == 4) u(i, j, 3) = r / gm1;                // E = p/(g-1), p = rho
    }
}

void System::solve_fields() { p_->solve_fields(); }

void System::step(double dt) {
  p_->solve_fields();
  for (auto& s : p_->sp)
    if (s.evolve) s.advance(s.U, Real(dt), s.substeps);  // bloc gele : non avance
  p_->apply_couplings(Real(dt));  // sources couplees inter-especes (splitting), apres transport
  p_->t += dt;
}
void System::advance(double dt, int nsteps) {
  for (int s = 0; s < nsteps; ++s) step(dt);
}
double System::step_cfl(double cfl) {
  p_->solve_fields();
  Real wmax = Real(1e-30);
  for (auto& s : p_->sp)
    if (s.evolve) wmax = std::max(wmax, s.max_speed(s.U));  // bloc gele : ne contraint pas le pas
  const Real h = std::min(p_->geom.dx(), p_->geom.dy());
  const double dt = cfl * static_cast<double>(h) / static_cast<double>(wmax);
  for (auto& s : p_->sp)
    if (s.evolve) s.advance(s.U, Real(dt), s.substeps);
  p_->apply_couplings(Real(dt));
  p_->t += dt;
  return dt;
}
double System::step_adaptive(double cfl) {
  p_->solve_fields();
  // Multirate : macro-pas = pas stable du bloc le plus LENT ; chaque bloc plus rapide est
  // sous-cycle n_b = ceil(w_b / w_min). aux fige sur le macro-pas (couplage once-per-step).
  Real wmin = Real(1e30);
  std::vector<Real> wb;
  wb.reserve(p_->sp.size());
  for (auto& s : p_->sp) {
    const Real w = s.evolve ? s.max_speed(s.U) : Real(0);  // bloc gele : hors cadence
    wb.push_back(w);
    if (s.evolve) wmin = std::min(wmin, w);
  }
  if (wmin >= Real(1e30)) wmin = Real(1e-30);  // aucun bloc evolutif (tous geles)
  const Real h = std::min(p_->geom.dx(), p_->geom.dy());
  const double macro_dt = cfl * static_cast<double>(h) / static_cast<double>(wmin);
  for (std::size_t b = 0; b < p_->sp.size(); ++b) {
    if (!p_->sp[b].evolve) continue;  // bloc gele : non avance
    int n = static_cast<int>(std::ceil(static_cast<double>(wb[b] / wmin)));
    if (n < 1) n = 1;
    p_->sp[b].advance(p_->sp[b].U, Real(macro_dt), n);
  }
  p_->apply_couplings(Real(macro_dt));
  p_->t += macro_dt;
  return macro_dt;
}

std::vector<double> System::eval_rhs(const std::string& name) {
  Impl::Species& s = p_->find(name);
  MultiFab R(p_->ba, p_->dm, s.ncomp, 0);
  s.rhs_into(s.U, R);
  return p_->copy_state(R, s.ncomp);
}
std::vector<double> System::get_state(const std::string& name) {
  Impl::Species& s = p_->find(name);
  return p_->copy_state(s.U, s.ncomp);
}
void System::set_state(const std::string& name, const std::vector<double>& u) {
  Impl::Species& s = p_->find(name);
  p_->write_state(s.U, s.ncomp, u);
}
int System::n_vars(const std::string& name) const { return p_->find(name).ncomp; }
std::vector<std::string> System::variable_names(const std::string& name,
                                               const std::string& kind) const {
  const Impl::Species& s = p_->find(name);
  if (kind == "conservative") return s.cons_vars.names;
  if (kind == "primitive") return s.prim_vars.names;
  throw std::runtime_error("System::variable_names : kind 'conservative' | 'primitive' (recu '" +
                           kind + "')");
}
std::vector<std::string> System::variable_roles(const std::string& name,
                                               const std::string& kind) const {
  const Impl::Species& s = p_->find(name);
  const VariableSet* vs = nullptr;
  if (kind == "conservative") vs = &s.cons_vars;
  else if (kind == "primitive") vs = &s.prim_vars;
  else throw std::runtime_error("System::variable_roles : kind 'conservative' | 'primitive' (recu '" +
                                kind + "')");
  std::vector<std::string> out;
  out.reserve(static_cast<std::size_t>(vs->size));
  for (int i = 0; i < vs->size; ++i) out.push_back(role_name(vs->at(i).role));  // 'custom' si absent
  return out;
}

int System::nx() const { return p_->cfg.n; }
double System::time() const { return p_->t; }
int System::n_species() const { return static_cast<int>(p_->sp.size()); }
double System::mass(const std::string& name) const { return sum(p_->find(name).U, 0); }
std::vector<double> System::density(const std::string& name) const {
  return p_->copy_comp0(p_->find(name).U);
}
std::vector<double> System::potential() {
  p_->ensure_elliptic();
  device_fence();
  const ConstArray4 ph = p_->ell_phi().fab(0).const_array();
  const Box2D v = p_->aux.box(0);
  std::vector<double> out;
  out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(ph(i, j));
  return out;
}

}  // namespace adc
