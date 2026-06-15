#pragma once

#include <adc/core/state.hpp>       // StateVec, Aux, ADC_AUX_FIELDS, kAuxBaseComps
#include <adc/core/types.hpp>       // ADC_HD, Real
#include <adc/core/variables.hpp>   // VariableSet + VariableKind + VariableRole + role_from_name
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/runtime/abi_key.hpp>          // adc::abi_key (garde-fou ABI du loader natif)
#include <adc/runtime/dynamic_model.hpp>    // IModel : modele charge a l'execution (bloc dynamique)
#include <adc/runtime/grid_context.hpp>     // GridContext
#include <adc/runtime/system.hpp>           // adc::System (install_block / grid_context / ensure_aux_width)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <adc/runtime/dynlib.hpp>  // couche portable dlopen<->LoadLibraryW (ADC-99) ; inclut <dlfcn.h> sur POSIX
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

/// @file
/// @brief NativeLoader : chargement d'un .so genere par le DSL (chemins JIT / AOT / natif) +
///        garde-fou ABI. Extrait VERBATIM de python/system.cpp (PR extraction #1, plan
///        docs/SYSTEM_CPP_EXTRACTION_PLAN.md). STRICTEMENT bit-identique : aucun changement de
///        comportement, de logique, ou d'ordre d'evaluation -- le code est deplace tel quel.
///
/// Les fermetures des trois chemins accedent aux internes de System::Impl (copy_state / write_state
/// / aux / sp / ensure_aux_width) et a des methodes hors-ligne de System (install_block /
/// grid_context). Pour rester sans dependance circulaire sur la definition de Impl (qui reste
/// PRIVEE a python/system.cpp), les fonctions de chargement sont des TEMPLATES parametrees sur le
/// type Impl reel : python/system.cpp les instancie avec System::Impl apres avoir defini Impl. Les
/// helpers purs (host_residual, split, BlockMeta, ...) ne touchent pas Impl et restent des fonctions
/// libres `inline` (ODR-safe en en-tete).

namespace adc {
namespace native_loader {

/// Pente limitee MUSCL d'une cellule depuis differences arriere @p am et avant @p ap. ADC_HD
/// (device-callable, sans std::). @p recon : 0 = ordre 1 (pente nulle), 1 = minmod (TVD), 2 = van Leer.
/// Brique ponctuelle du residu hote host_residual ; pendant hote du limiteur compile (reconstruction.hpp).
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

/// Residu hote R = -div F* + S(U, aux) (Rusanov, a_max GLOBAL comme adc.PythonFlux, periodique) calcule
/// via un IModel : sert au bloc DYNAMIQUE (modele charge a l'execution, dispatch virtuel, hors GPU).
/// @p recon = ordre de reconstruction MUSCL des etats de face sur les variables conservatives (cf.
/// limited_slope). Le flux ET la source recoivent l'aux par cellule (phi, grad phi) : un modele couple
/// (transport ExB, force) marche, pas seulement Euler. @p AUX est l'aux du systeme (>= 3 comp
/// phi/grad_x/grad_y, composante-majeur ; vide => nul). U et le retour en disposition composante-majeur
/// (c*n*n + j*n + i).
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
      // Champs extra marshales depuis la SOURCE UNIQUE ADC_AUX_FIELDS (adc/core/state.hpp) :
      // meme table que load_aux cote device. Une composante extra n'est lue que si le canal est
      // assez large ((idx+1)*nn elements). Ajouter un champ aux => 1 ligne dans ADC_AUX_FIELDS,
      // ce site (et l'autre, plus bas) le transporte AUTOMATIQUEMENT. Ferme le trou #51.
#define ADC_AUX_MARSHAL(name, idx) \
  if (AUX.size() >= ((idx) + 1) * nn) a.name = AUX[(idx) * nn + k];
      ADC_AUX_FIELDS(ADC_AUX_MARSHAL)
#undef ADC_AUX_MARSHAL
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

/// Decoupe @p s sur @p sep (champs vides conserves). Brique de parsing des metadonnees TEXTE
/// transportees par l'ABI du .so (noms / roles en CSV).
inline std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == sep) { out.push_back(cur); cur.clear(); }
    else cur += c;
  }
  out.push_back(cur);
  return out;
}

/// Metadonnees OPTIONNELLES lues par dlsym sur un .so genere (noms / roles / gamma). Toutes
/// optionnelles : un vieux .so sans ces symboles donne meta vide -> les chemins add_compiled_block /
/// add_dynamic_block retombent sur leur fallback (noms u0.., roles vides, gamma 1.4). Aucune dependance
/// d'objet C++ : seules des chaines et un double transitent par dlsym (ABI plate, comme le reste du bloc).
struct BlockMeta {
  bool has_gamma = false;
  double gamma = 1.4;
  VariableSet cons{VariableKind::Conservative, {}, 0, {}};
  VariableSet prim{VariableKind::Primitive, {}, 0, {}};
};

/// Construit un VariableSet a partir des CSV noms / roles (parallele a names ; roles VIDE = non
/// renseignes -> VariableSet sans roles, fallback indices cote couplages). @p names_csv vide -> set vide.
inline VariableSet parse_var_set(VariableKind kind, const std::string& names_csv,
                                 const std::string& roles_csv) {
  VariableSet vs{kind, {}, 0, {}};
  if (names_csv.empty()) return vs;  // pas de noms transportes : set vide (le caller mettra son fallback)
  vs.names = split(names_csv, ',');
  vs.size = static_cast<int>(vs.names.size());
  if (!roles_csv.empty()) {
    std::vector<std::string> rs = split(roles_csv, ',');
    for (const std::string& r : rs) vs.roles.push_back(role_from_name(r));
  }
  return vs;
}

/// Lit (par dlsym, tous OPTIONNELS) les symboles de metadonnees d'un .so deja ouvert (@p h). Renvoie
/// noms+roles deserialises et gamma s'il est present. Les symboles absents laissent les champs vides /
/// has_gamma=false : le caller decide alors du fallback. Format des chaines : "cons_csv|prim_csv".
inline BlockMeta read_block_meta(adc::dynlib::handle h) {
  BlockMeta m;
  auto names_fn = reinterpret_cast<const char* (*)()>(adc::dynlib::sym(h,"adc_compiled_var_names"));
  auto roles_fn = reinterpret_cast<const char* (*)()>(adc::dynlib::sym(h,"adc_compiled_roles"));
  auto gamma_fn = reinterpret_cast<double (*)()>(adc::dynlib::sym(h,"adc_compiled_gamma"));
  std::string names = names_fn ? std::string(names_fn()) : std::string();
  std::string roles = roles_fn ? std::string(roles_fn()) : std::string();
  // "cons|prim" : indice 0 = jeu conservatif, indice 1 = jeu primitif (chacun eventuellement vide).
  std::vector<std::string> nparts = split(names, '|');
  std::vector<std::string> rparts = split(roles, '|');
  auto part = [](const std::vector<std::string>& v, std::size_t i) {
    return i < v.size() ? v[i] : std::string();
  };
  m.cons = parse_var_set(VariableKind::Conservative, part(nparts, 0), part(rparts, 0));
  m.prim = parse_var_set(VariableKind::Primitive, part(nparts, 1), part(rparts, 1));
  if (gamma_fn) { m.has_gamma = true; m.gamma = gamma_fn(); }
  return m;
}

/// Construit un bloc DYNAMIQUE (modele IModel<NV> charge depuis le .so @p h) et l'ajoute. Le
/// shared_ptr possede le modele : il appelle adc_destroy_model puis ferme le .so a la destruction.
/// VERBATIM de l'ancien Impl::push_dynamic ; @p ImplT est System::Impl (instancie cote system.cpp).
template <typename ImplT, int NV>
void push_dynamic(ImplT* P, const std::string& name, adc::dynlib::handle h, int substeps,
                  std::vector<std::string> names, int recon) {
  auto mk = reinterpret_cast<void* (*)()>(adc::dynlib::sym(h,"adc_make_model"));
  auto del = reinterpret_cast<void (*)(void*)>(adc::dynlib::sym(h,"adc_destroy_model"));
  if (!mk || !del) {
    adc::dynlib::close(h);
    throw std::runtime_error("add_dynamic_block : adc_make_model / adc_destroy_model absents du .so");
  }
  std::shared_ptr<IModel<NV>> im(static_cast<IModel<NV>*>(mk()),
                                 [del, h](IModel<NV>* p) { del(p); adc::dynlib::close(h); });
  // Le modele charge peut lire des champs aux supplementaires (n_aux > 3, p.ex. B_z) : on
  // elargit le canal aux PARTAGE pour que set_magnetic_field le peuple et que le marshaling hote
  // les transporte. Modele de base (3) -> no-op. Les fermetures lisent P->aux_ncomp_ a l'appel.
  P->ensure_aux_width(im->n_aux());
  const int n = P->cfg.n;
  const double dx = P->cfg.L / P->cfg.n;

  std::function<void(MultiFab&, MultiFab&)> rhs_into = [P, im, n, dx, recon](MultiFab& U,
                                                                             MultiFab& R) {
    // Chemin HOTE (bloc dynamique) : meme garde MPI que add_poisson. Sur un rang sans box locale
    // (local_size()==0 a np>1) il n'y a rien a marshaler ; le rang proprietaire porte la physique
    // complete. Sans cette garde, copy_state(U) / write_state(R) dereferenceraient fab(0) inexistant.
    // Pas d'operation MPI collective ici -> no-op unilateral, aucun risque d'interblocage.
    if (U.local_size() == 0) return;
    P->write_state(R, NV, host_residual<NV>(*im, P->copy_state(U, NV),
                                            P->copy_state(P->aux, P->aux_ncomp_), n, dx, recon));
  };
  std::function<Real(const MultiFab&)> max_speed = [P, im, n](const MultiFab& U) -> Real {
    // Meme garde MPI : rang vide -> vitesse locale 0 (l'all_reduce_max en aval prend le max global,
    // le proprietaire contribue la vraie valeur). Pas de collectif ici -> no-op sans interblocage.
    if (U.local_size() == 0) return Real(0);
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
        // Champs extra : meme SOURCE UNIQUE ADC_AUX_FIELDS que load_aux et que l'autre site
        // de marshaling (host_residual). cf. note la-haut ; ajout d'un champ = 1 ligne.
#define ADC_AUX_MARSHAL(name, idx) \
  if (aux.size() >= ((idx) + 1) * nn) a.name = aux[(idx) * nn + c0];
        ADC_AUX_FIELDS(ADC_AUX_MARSHAL)
#undef ADC_AUX_MARSHAL
      }
      Real v = std::max(im->max_wave_speed(s, a, 0), im->max_wave_speed(s, a, 1));
      if (v > mx) mx = v;
    }
    return mx;
  };
  std::function<void(MultiFab&, Real, int)> advance = [P, im, n, dx, recon](MultiFab& U, Real dt,
                                                                            int nsub) {
    // Meme garde MPI : rang vide -> no-op (rien a marshaler). Pas de collectif -> sans interblocage.
    if (U.local_size() == 0) return;
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
        // Chemin HOTE (bloc dynamique .so prototype) : il marshale la box vers un tableau plein
        // n*n et la traite comme repliquee. Sur un rang sans box locale (System repartit UNE box,
        // donc local_size()==0 a np>1 sur les rangs non proprietaires) il n'y a rien a marshaler :
        // on saute, le rang proprietaire porte la contribution complete au second membre. Sans
        // cette garde, copy_state(U) / rhs.fab(0) dereferenceraient un fab inexistant (crash hote).
        if (rhs.local_size() == 0) return;
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
  // Metadonnees OPTIONNELLES (noms / roles / gamma) portees par l'ABI etendue du .so. Symetrique
  // du chemin AOT. Absentes d'un vieux .so -> meta vide -> fallback (noms u0.. / pas de roles /
  // gamma 1.4). PRIORITE au nom explicite (names=), puis meta, puis fallback ; roles + primitif
  // ne viennent QUE de l'ABI (l'API ne les expose pas).
  const BlockMeta meta = read_block_meta(h);
  VariableSet cons_vs = meta.cons, prim_vs = meta.prim;
  if (!names.empty()) {
    if (static_cast<int>(names.size()) != NV)
      throw std::runtime_error("System::add_dynamic_block : names= a " +
                               std::to_string(names.size()) + " noms mais le bloc '" + name +
                               "' a " + std::to_string(NV) + " variables");
    cons_vs.names = names;
    cons_vs.size = static_cast<int>(names.size());
  }
  if (cons_vs.names.empty()) {
    for (int c = 0; c < NV; ++c) cons_vs.names.push_back("u" + std::to_string(c));
    cons_vs.size = NV;
  }
  if (prim_vs.names.empty()) prim_vs = {VariableKind::Primitive, cons_vs.names, cons_vs.size, {}};
  const double gamma = meta.has_gamma ? meta.gamma : 1.4;

  typename ImplT::Species block{name, MultiFab(P->ba, P->dm, NV, 2), NV, substeps, true, /*stride=*/1, gamma,
                std::move(advance), std::move(rhs_into), std::move(max_speed), std::move(add_poisson)};
  block.cons_vars = std::move(cons_vs);
  block.prim_vars = std::move(prim_vs);
  // Conversions PONCTUELLES cons <-> prim DU MODELE (set/get_primitive_state) : forwardees par
  // l'interface virtuelle IModel<NV> (ModelAdapter delegue a M.to_primitive/to_conservative quand
  // M les expose, sinon identite). Un vieux .so dont le modele n'a pas de conversion retombe donc
  // sur l'identite -- exact pour un scalaire (prim == cons). StateVec<NV> partage la largeur NV.
  block.prim_to_cons = [im](const double* in, double* out) {
    StateVec<NV> p{};
    for (int c = 0; c < NV; ++c) p[c] = static_cast<Real>(in[c]);
    const StateVec<NV> u = im->to_conservative(p);
    for (int c = 0; c < NV; ++c) out[c] = static_cast<double>(u[c]);
  };
  block.cons_to_prim = [im](const double* in, double* out) {
    StateVec<NV> u{};
    for (int c = 0; c < NV; ++c) u[c] = static_cast<Real>(in[c]);
    const StateVec<NV> p = im->to_primitive(u);
    for (int c = 0; c < NV; ++c) out[c] = static_cast<double>(p[c]);
  };
  P->sp.push_back(std::move(block));
  P->sp.back().U.set_val(Real(0));
}

/// Corps de System::add_dynamic_block. VERBATIM ; @p self = le System appelant, @p P = self->p_.get().
template <typename ImplT>
void add_dynamic_block(System* self, ImplT* P, const std::string& name, const std::string& so_path,
                       int substeps, const std::vector<std::string>& names,
                       const std::string& recon) {
  (void)self;
  if (substeps < 1) throw std::runtime_error("System::add_dynamic_block : substeps >= 1");
  int recon_id = 0;  // ordre de reconstruction MUSCL des etats de face (conservatif)
  if (recon == "none") recon_id = 0;
  else if (recon == "minmod") recon_id = 1;
  else if (recon == "vanleer") recon_id = 2;
  else throw std::runtime_error("System::add_dynamic_block : recon 'none' | 'minmod' | 'vanleer' "
                                "(recu '" + recon + "')");
  adc::dynlib::handle h = adc::dynlib::open(so_path);
  if (!h) {
    const std::string e = adc::dynlib::last_error();
    throw std::runtime_error("add_dynamic_block : dlopen('" + so_path + "') : " +
                             (e.empty() ? std::string("?") : e));
  }
  auto nv_fn = reinterpret_cast<int (*)()>(adc::dynlib::sym(h,"adc_model_nvars"));
  if (!nv_fn) {
    adc::dynlib::close(h);
    throw std::runtime_error("add_dynamic_block : adc_model_nvars absent du .so");
  }
  const int nv = nv_fn();
  switch (nv) {
    case 1: push_dynamic<ImplT, 1>(P, name, h, substeps, names, recon_id); break;
    case 3: push_dynamic<ImplT, 3>(P, name, h, substeps, names, recon_id); break;
    case 4: push_dynamic<ImplT, 4>(P, name, h, substeps, names, recon_id); break;
    default:
      adc::dynlib::close(h);
      throw std::runtime_error("add_dynamic_block : n_vars=" + std::to_string(nv) +
                               " non supporte (1, 3, 4)");
  }
}

/// Corps de System::add_compiled_block. VERBATIM ; @p self = le System appelant, @p P = self->p_.get().
template <typename ImplT>
void add_compiled_block(System* self, ImplT* P, const std::string& name, const std::string& so_path,
                        const std::string& limiter, const std::string& riemann,
                        const std::string& recon, const std::string& time, int substeps,
                        const std::vector<std::string>& names, double pos_floor = 0) {
  (void)self;
  if (substeps < 1) throw std::runtime_error("System::add_compiled_block : substeps >= 1");
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error("System::add_compiled_block : recon 'conservative' | 'primitive'");
  // NB : le chemin AOT (.so) marshale time->imex sans schema RK explicite : seul SSPRK2 est cable
  // dans l'ABI extern "C" du .so. "ssprk3" et "euler" ne sont donc PAS supportes ici (l'avance
  // resterait SSPRK2, silencieusement) -> on les rejette ; les exposer exigerait d'etendre l'ABI du
  // .so (hors scope ADC-174). Ils sont portes par add_block natif ET backend='production'
  // (add_native_block : le gabarit marshale method jusqu'au make_block du loader).
  if (time != "explicit" && time != "imex")
    throw std::runtime_error("System::add_compiled_block : time 'explicit' | 'imex' (ssprk3, euler et "
                             "la famille IMEX-RK ARS(2,2,2) -> add_block natif ou backend='production' ; "
                             "le chemin AOT n'expose que SSPRK2 + backward-Euler local)");
  // WENO5 (stencil 5 points, 3 ghosts) est desormais EXPOSE par le chemin AOT : la grille locale du
  // .so alloue block_n_ghost(limiter) (compiled_block_abi.hpp), 3 pour weno5, donc assemble_rhs ne lit
  // pas hors bornes. limiter est valide par make_block dans le .so (none|minmod|vanleer|weno5).
  const int recon_prim = (recon == "primitive") ? 1 : 0;
  const int imex = (time == "imex") ? 1 : 0;

  adc::dynlib::handle h = adc::dynlib::open(so_path);
  if (!h) {
    const std::string e = adc::dynlib::last_error();
    throw std::runtime_error("add_compiled_block : dlopen('" + so_path + "') : " +
                             (e.empty() ? std::string("?") : e));
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
  auto nv_fn = reinterpret_cast<nv_fn_t>(adc::dynlib::sym(h,"adc_model_nvars"));
  auto res_fn = reinterpret_cast<res_fn_t>(adc::dynlib::sym(h,"adc_compiled_residual"));
  auto adv_fn = reinterpret_cast<adv_fn_t>(adc::dynlib::sym(h,"adc_compiled_advance"));
  auto max_fn = reinterpret_cast<max_fn_t>(adc::dynlib::sym(h,"adc_compiled_max_speed"));
  auto poi_fn = reinterpret_cast<poi_fn_t>(adc::dynlib::sym(h,"adc_compiled_poisson_rhs"));
  if (!nv_fn || !res_fn || !adv_fn || !max_fn || !poi_fn) {
    adc::dynlib::close(h);
    throw std::runtime_error("add_compiled_block : ABI bloc compile absente du .so (regenerer via "
                             "dsl.compile_aot / compile_or_jit(mode='compile'))");
  }
  const int nv = nv_fn();
  // Largeur du canal aux que le modele compile LIT (B_z, T_e...). Symetrique du chemin JIT
  // (IModel::n_aux). Optionnel : un vieux .so sans ce symbole retombe sur le contrat de base (3).
  auto naux_fn = reinterpret_cast<nv_fn_t>(adc::dynlib::sym(h,"adc_compiled_naux"));
  const int naux = naux_fn ? naux_fn() : kAuxBaseComps;
  // PARAMS RUNTIME (P7-b) : variantes SUFFIXEES `_p` qui prennent un bloc plat (const double*, int) de
  // valeurs de parametres runtime, injectees dans le modele avant l'execution. OPTIONNELLES : un .so
  // genere avant ce chantier (ou un modele sans param runtime) ne les expose pas / declare nparams=0 ->
  // on garde les symboles historiques (chemin params-const, bit-identique). On SEEDE le bloc de valeurs
  // aux defauts de declaration (adc_compiled_param_defaults) pour qu'un set_param ulterieur n'ecrase
  // qu'une entree sans remettre les autres a zero.
  auto nparams_fn = reinterpret_cast<nv_fn_t>(adc::dynlib::sym(h,"adc_compiled_nparams"));
  const int nparams = nparams_fn ? nparams_fn() : 0;
  using res_p_fn_t = void (*)(const double*, double*, const double*, int, double, double, int,
                              const char*, const char*, int, const double*, int, double);
  using adv_p_fn_t = void (*)(double*, const double*, int, double, double, int, const char*,
                              const char*, int, int, double, int, const double*, int, double);
  using max_p_fn_t = double (*)(const double*, const double*, int, double, double, int,
                                const double*, int);
  using poi_p_fn_t = void (*)(const double*, double*, int, const double*, int);
  auto res_p_fn = reinterpret_cast<res_p_fn_t>(adc::dynlib::sym(h,"adc_compiled_residual_p"));
  auto adv_p_fn = reinterpret_cast<adv_p_fn_t>(adc::dynlib::sym(h,"adc_compiled_advance_p"));
  auto max_p_fn = reinterpret_cast<max_p_fn_t>(adc::dynlib::sym(h,"adc_compiled_max_speed_p"));
  auto poi_p_fn = reinterpret_cast<poi_p_fn_t>(adc::dynlib::sym(h,"adc_compiled_poisson_rhs_p"));
  // Bloc PARTAGE des valeurs courantes : capture par les fermetures ET enregistre dans P->block_params_
  // (set_block_params y ecrit -> les fermetures voient la nouvelle valeur au prochain pas). Vide si le
  // bloc n'a aucun param runtime ou si l'ABI `_p` est absente (vieux .so) : les fermetures appellent
  // alors les symboles historiques (chemin const). pv non nul declenche le chemin `_p`.
  std::shared_ptr<std::vector<double>> pv;
  const bool use_params = (nparams > 0 && res_p_fn && adv_p_fn && max_p_fn && poi_p_fn);
  // LIMITEUR DE POSITIVITE (ADC-76) : pos_floor > 0 passe par les variantes `_p` (seules signatures
  // qui le transportent). Un vieux .so sans `_p` ne peut pas l'appliquer -> erreur claire (jamais un
  // chemin silencieux sans positivite). pos_floor <= 0 : chemins historiques, bit-identiques.
  if (pos_floor > 0 && (!res_p_fn || !adv_p_fn))
    throw std::runtime_error(
        "add_compiled_block : positivity_floor > 0 exige l'ABI `_p` du .so (regenerer le module "
        "compile avec les en-tetes courants)");
  if (use_params) {
    pv = std::make_shared<std::vector<double>>(static_cast<std::size_t>(nparams), 0.0);
    auto defs_fn = reinterpret_cast<void (*)(double*)>(adc::dynlib::sym(h,"adc_compiled_param_defaults"));
    if (defs_fn) defs_fn(pv->data());  // seed aux defauts de declaration
  }  // enregistrement dans P->block_params_ DIFFERE juste avant push_back (apres les validations qui
     // peuvent lever : evite une entree orpheline sans bloc associe si l'ajout echoue).
  // Metadonnees OPTIONNELLES (noms / roles / gamma) transportees par l'ABI etendue du .so. Absentes
  // d'un vieux .so -> meta vide, on retombe sur le fallback (noms u0.. / pas de roles / gamma 1.4).
  const BlockMeta meta = read_block_meta(h);
  // Elargit le canal aux PARTAGE pour que set_magnetic_field/T_e le peuplent et que le marshaling
  // transporte les composantes extra vers le .so. Modele de base (3) -> no-op (bit-identique).
  P->ensure_aux_width(naux);
  std::shared_ptr<void> lib(h, [](void* p) {  // ferme le .so a la mort des fermetures
    adc::dynlib::close(static_cast<adc::dynlib::handle>(p));
  });
  const int n = P->cfg.n;
  const double dx = P->geom.dx(), dy = P->geom.dy();
  const int per = P->periodic_ ? 1 : 0;
  const std::string lim = limiter, riem = riemann;

  std::function<void(MultiFab&, MultiFab&)> rhs_into =
      [P, lib, res_fn, res_p_fn, pv, nv, n, dx, dy, per, lim, riem, recon_prim,
       pos_floor](MultiFab& U, MultiFab& R) {
        // Chemin HOTE (bloc compile .so) : meme garde MPI que add_poisson. Sur un rang sans box
        // locale (local_size()==0 a np>1) il n'y a rien a marshaler ; le rang proprietaire porte
        // la physique complete. Pas de collectif ici -> no-op unilateral, aucun interblocage.
        if (U.local_size() == 0) return;
        std::vector<double> u = P->copy_state(U, nv), a = P->copy_state(P->aux, P->aux_ncomp_);
        std::vector<double> r(static_cast<std::size_t>(nv) * n * n, 0.0);
        if (pv || pos_floor > 0)  // variante `_p` : params RUNTIME (P7-b) et/ou positivite (ADC-76)
          res_p_fn(u.data(), r.data(), a.data(), n, dx, dy, per, lim.c_str(), riem.c_str(),
                   recon_prim, pv ? pv->data() : nullptr, pv ? static_cast<int>(pv->size()) : 0,
                   pos_floor);
        else
          res_fn(u.data(), r.data(), a.data(), n, dx, dy, per, lim.c_str(), riem.c_str(), recon_prim);
        P->write_state(R, nv, r);
      };
  std::function<void(MultiFab&, Real, int)> advance =
      [P, lib, adv_fn, adv_p_fn, pv, nv, n, dx, dy, per, lim, riem, recon_prim, imex,
       pos_floor](MultiFab& U, Real dt, int nsub) {
        // Meme garde MPI : rang vide -> no-op. Pas de collectif -> sans interblocage.
        if (U.local_size() == 0) return;
        std::vector<double> u = P->copy_state(U, nv), a = P->copy_state(P->aux, P->aux_ncomp_);
        if (pv || pos_floor > 0)  // variante `_p` : params RUNTIME (P7-b) et/ou positivite (ADC-76)
          adv_p_fn(u.data(), a.data(), n, dx, dy, per, lim.c_str(), riem.c_str(), recon_prim, imex,
                   static_cast<double>(dt), nsub, pv ? pv->data() : nullptr,
                   pv ? static_cast<int>(pv->size()) : 0, pos_floor);
        else
          adv_fn(u.data(), a.data(), n, dx, dy, per, lim.c_str(), riem.c_str(), recon_prim, imex,
                 static_cast<double>(dt), nsub);
        P->write_state(U, nv, u);
      };
  std::function<Real(const MultiFab&)> max_speed =
      [P, lib, max_fn, max_p_fn, pv, nv, n, dx, dy, per](const MultiFab& U) -> Real {
        // Meme garde MPI : rang vide -> vitesse locale 0. L'all_reduce_max en aval prend le max global.
        if (U.local_size() == 0) return Real(0);
        std::vector<double> u = P->copy_state(U, nv), a = P->copy_state(P->aux, P->aux_ncomp_);
        if (pv)  // params RUNTIME (P7-b)
          return max_p_fn(u.data(), a.data(), n, dx, dy, per, pv->data(),
                          static_cast<int>(pv->size()));
        return max_fn(u.data(), a.data(), n, dx, dy, per);
      };
  std::function<void(const MultiFab&, MultiFab&)> add_poisson =
      [P, lib, poi_fn, poi_p_fn, pv, nv, n](const MultiFab& U, MultiFab& rhs) {
        // Chemin HOTE (bloc compile .so prototype) : meme garde MPI que le bloc dynamique. Sur un
        // rang sans box locale (local_size()==0 a np>1) il n'y a rien a marshaler -> on saute, le
        // rang proprietaire porte la contribution complete. Sans cela copy_state(U) / rhs.fab(0)
        // dereferenceraient un fab inexistant (crash hote).
        if (rhs.local_size() == 0) return;
        std::vector<double> u = P->copy_state(U, nv);
        std::vector<double> pr(static_cast<std::size_t>(n) * n, 0.0);
        if (pv)  // params RUNTIME (P7-b)
          poi_p_fn(u.data(), pr.data(), n, pv->data(), static_cast<int>(pv->size()));
        else
          poi_fn(u.data(), pr.data(), n);
        Array4 r = rhs.fab(0).array();
        const Box2D v = rhs.box(0);
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i)
            r(i, j, 0) += pr[static_cast<std::size_t>(j - v.lo[1]) * n + (i - v.lo[0])];
      };

  // Descripteurs de variables : PRIORITE au nom explicite passe par l'appelant (names=), sinon aux
  // NOMS portes par l'ABI etendue du .so (meta), sinon fallback u0.. . Les ROLES et le PRIMITIF ne
  // transitent QUE par l'ABI (l'API n'a pas de moyen de les fournir) : on les prend de meta tels quels.
  VariableSet cons_vs = meta.cons, prim_vs = meta.prim;
  if (!names.empty()) {                 // override explicite des noms conservatifs
    if (static_cast<int>(names.size()) != nv)
      throw std::runtime_error("System::add_compiled_block : names= a " +
                               std::to_string(names.size()) + " noms mais le bloc '" + name +
                               "' a " + std::to_string(nv) + " variables");
    cons_vs.names = names;
    cons_vs.size = static_cast<int>(names.size());
  }
  if (cons_vs.names.empty()) {          // ni names= ni meta : fallback historique u0..
    for (int c = 0; c < nv; ++c) cons_vs.names.push_back("u" + std::to_string(c));
    cons_vs.size = nv;
  }
  if (prim_vs.names.empty()) prim_vs = {VariableKind::Primitive, cons_vs.names, cons_vs.size, {}};
  // gamma : porte par l'ABI si le modele le declare (adc_compiled_gamma), sinon defaut historique 1.4.
  const double gamma = meta.has_gamma ? meta.gamma : 1.4;
  typename ImplT::Species block{name, MultiFab(P->ba, P->dm, nv, 2), nv, substeps, true, /*stride=*/1, gamma,
                      std::move(advance), std::move(rhs_into), std::move(max_speed),
                      std::move(add_poisson)};
  block.cons_vars = std::move(cons_vs);
  block.prim_vars = std::move(prim_vs);
  // Conversions cons <-> prim DU MODELE via l'ABI etendue du .so (set/get_primitive_state). Les
  // symboles operent sur des tableaux plats composante-majeur c*nn+k : appeles avec n=1 (donc nn=1),
  // ils convertissent UNE cellule (in/out = nv doubles), ce qui est exactement le contrat CellConvert.
  // OPTIONNELS : un .so genere avant ce chantier ne les expose pas -> conversion vide -> identite (le
  // chemin set/get_primitive_state retombe alors sur prim == cons, exact pour un scalaire).
  using cv_fn_t = void (*)(const double*, double*, int);
  auto p2c_fn = reinterpret_cast<cv_fn_t>(adc::dynlib::sym(h,"adc_compiled_to_conservative"));
  auto c2p_fn = reinterpret_cast<cv_fn_t>(adc::dynlib::sym(h,"adc_compiled_to_primitive"));
  // P7-b : si le bloc a des params runtime, les conversions cons<->prim doivent les voir aussi (la
  // conversion peut lire un param runtime). On prefere alors les variantes `_p` avec le bloc PARTAGE.
  using cv_p_fn_t = void (*)(const double*, double*, int, const double*, int);
  auto p2c_p_fn = reinterpret_cast<cv_p_fn_t>(adc::dynlib::sym(h,"adc_compiled_to_conservative_p"));
  auto c2p_p_fn = reinterpret_cast<cv_p_fn_t>(adc::dynlib::sym(h,"adc_compiled_to_primitive_p"));
  if (pv && p2c_p_fn)
    block.prim_to_cons = [lib, p2c_p_fn, pv](const double* in, double* out) {
      p2c_p_fn(in, out, 1, pv->data(), static_cast<int>(pv->size()));
    };
  else if (p2c_fn)
    block.prim_to_cons = [lib, p2c_fn](const double* in, double* out) { p2c_fn(in, out, 1); };
  if (pv && c2p_p_fn)
    block.cons_to_prim = [lib, c2p_p_fn, pv](const double* in, double* out) {
      c2p_p_fn(in, out, 1, pv->data(), static_cast<int>(pv->size()));
    };
  else if (c2p_fn)
    block.cons_to_prim = [lib, c2p_fn](const double* in, double* out) { c2p_fn(in, out, 1); };
  // P7-b : enregistrer le bloc PARTAGE des params runtime APRES les validations (toutes passees ici) :
  // set_block_params le retrouvera par nom, et les fermetures du bloc partagent le meme shared_ptr.
  if (pv) P->block_params_[name] = pv;
  P->sp.push_back(std::move(block));
  P->sp.back().U.set_val(Real(0));
}

/// Corps de System::add_native_block. VERBATIM ; @p self = le System appelant (et le `this` marshale
/// vers le loader natif), @p P = self->p_.get().
template <typename ImplT>
void add_native_block(System* self, ImplT* P, const std::string& name, const std::string& so_path,
                      const std::string& limiter, const std::string& riemann,
                      const std::string& recon, const std::string& time, double gamma,
                      int substeps, bool evolve, int stride, double pos_floor = 0) {
  (void)P;
  if (substeps < 1) throw std::runtime_error("System::add_native_block : substeps >= 1");
  if (stride < 1) throw std::runtime_error("System::add_native_block : stride >= 1");
  // Validation AMONT du schema (comme add_block / add_compiled_block) : add_compiled_model interprete
  // imex = (time=="imex"), recon_prim = (recon=="primitive") et le schema RK explicite
  // method = ssprk3 | euler | ssprk2 selon time ; une chaine inconnue retomberait SILENCIEUSEMENT sur
  // explicit/ssprk2/conservatif. On rejette donc une faute de frappe ICI plutot que de tourner un
  // schema different de celui demande. "ssprk3" (ordre 3, a apparier a weno5) et "euler" (ForwardEuler,
  // ordre 1 : fidelite aux references premier ordre, validation -- ADC-174) sont ACCEPTES : le gabarit
  // add_compiled_model les marshale jusqu'au make_block du .so, comme le chemin natif add_block. limiter/riemann sont valides par make_block dans le loader (exception
  // claire, ABI partagee verifiee plus bas).
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error("System::add_native_block : recon 'conservative' | 'primitive' (recu '" +
                             recon + "')");
  if (time != "explicit" && time != "ssprk3" && time != "euler" && time != "imex")
    throw std::runtime_error("System::add_native_block : time 'explicit' | 'ssprk3' | 'euler' | 'imex' "
                             "(recu '" + time + "' ; la famille IMEX-RK ARS(2,2,2) n'est cablee que sur "
                             "un modele compose adc.Model(...) -> add_block natif)");
  // WENO5 (stencil 5 points, 3 ghosts) est desormais EXPOSE par le chemin natif : le loader inline
  // add_compiled_model qui, apres install_block, reallue l'etat du bloc a block_n_ghost(limiter) (3
  // pour weno5) -- MEME mecanisme qu'add_block. assemble_rhs ne lit donc pas hors bornes. limiter est
  // valide par make_block dans le loader (none|minmod|vanleer|weno5).
  // Chemin "production" du DSL : le loader .so genere (emit_cpp_native_loader) inline le gabarit
  // en-tete add_compiled_model<ProdModel>, qui fabrique les fermetures sur le CONTEXTE REEL du
  // System (grid_context) et installe le bloc via install_block -- chemin NATIF, zero-copie, MEMES
  // MultiFab que add_block (pas de marshaling de tableaux plats comme add_compiled_block). Le loader
  // appelle donc des methodes hors-ligne de adc::System DEFINIES dans CE module ; il faut les
  // resoudre a travers le dlopen contre le module _adc deja charge.
  // PORTABILITE ELF (Linux) : CPython charge _adc en RTLD_LOCAL, donc ses symboles ne sont PAS dans
  // la portee globale et le loader ne pourrait pas les resoudre. On PROMEUT donc le module courant en
  // portee globale (RTLD_NOLOAD = sans le recharger, juste promouvoir ; RTLD_GLOBAL OR'e dans les
  // flags de l'objet deja charge). Le chemin du module est trouve par dladdr sur un symbole exporte
  // (adc::abi_key). Sur macOS c'est inoffensif (le loader resout deja par dynamic_lookup).
#if defined(_WIN32)
  // Windows (ADC-100) : PAS de RTLD_GLOBAL. La .dll generee est liee a la COMPILATION contre les
  // import libraries : _adc.lib (symboles System ADC_EXPORT : install_block/grid_context/...) et
  // kokkoscore.lib (runtime Kokkos PARTAGE en DLL). Ses indefinis sont donc resolus par l'OS-loader
  // contre _adc.pyd et kokkos*.dll deja charges -- pas besoin de promouvoir un scope global. On
  // charge simplement la .dll et on resout adc_install_native.
  adc::dynlib::handle h = adc::dynlib::open(so_path);
  if (!h)
    throw std::runtime_error("add_native_block : LoadLibrary('" + so_path + "') : " +
                             adc::dynlib::last_error() +
                             " (la .dll doit etre liee contre _adc.lib + kokkoscore.lib ; cf. ADC-100)");
  {
    auto key_fn = reinterpret_cast<const char* (*)()>(adc::dynlib::sym(h, "adc_native_abi_key"));
    if (!key_fn) {
      adc::dynlib::close(h);
      throw std::runtime_error("add_native_block : adc_native_abi_key absent de la .dll");
    }
    const std::string loader_key = key_fn();
    const std::string module_key = abi_key();
    if (loader_key != module_key) {
      adc::dynlib::close(h);
      throw std::runtime_error("add_native_block : ABI incompatible -- cle loader '" + loader_key +
                               "' != cle module '" + module_key + "'");
    }
    using install_fn_t = void (*)(void*, const char*, const char*, const char*, const char*,
                                  const char*, double, int, int, int, double);
    auto install = reinterpret_cast<install_fn_t>(adc::dynlib::sym(h, "adc_install_native"));
    if (!install) {
      adc::dynlib::close(h);
      throw std::runtime_error("add_native_block : adc_install_native absent de la .dll");
    }
    install(static_cast<void*>(self), name.c_str(), limiter.c_str(), riemann.c_str(), recon.c_str(),
            time.c_str(), gamma, substeps, evolve ? 1 : 0, stride, pos_floor);
  }
  // .dll laissee chargee pour la duree du process (le bloc pointe du code qui y vit).
#else
  {
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&adc::abi_key), &info) && info.dli_fname)
      dlopen(info.dli_fname, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
  }
  // RTLD_GLOBAL : place les symboles du loader dans la portee globale ET autorise le loader a resoudre
  // ses indefinis (les methodes System exportees ADC_EXPORT) contre les images deja chargees. RTLD_NOW :
  // resolution immediate -> un symbole System manquant (module sans ADC_EXPORT) echoue ICI, pas en vol.
  void* h = dlopen(so_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!h) {
    const char* e = dlerror();
    throw std::runtime_error("add_native_block : dlopen('" + so_path + "') : " +
                             std::string(e ? e : "?") +
                             " (les symboles adc::System (install_block/grid_context/"
                             "ensure_aux_width) doivent etre exportes ET le module _adc charge "
                             "globalement ; cf. ADC_EXPORT)");
  }
  // GARDE-FOU ABI EXPLICITE : la cle baked dans le loader (a SA compilation) doit egaler la cle du
  // module (a SA compilation). Un ecart = en-tetes / compilateur / standard divergents -> agencement
  // memoire de System/GridContext/BlockClosures potentiellement different a travers la frontiere ->
  // UB. On leve une erreur CLAIRE plutot que de laisser passer un loader incompatible.
  auto key_fn = reinterpret_cast<const char* (*)()>(adc::dynlib::sym(h,"adc_native_abi_key"));
  if (!key_fn) {
    adc::dynlib::close(h);
    throw std::runtime_error("add_native_block : adc_native_abi_key absent du .so (regenerer via "
                             "dsl.compile_native / compile(backend='production'))");
  }
  const std::string loader_key = key_fn();
  const std::string module_key = abi_key();
  if (loader_key != module_key) {
    adc::dynlib::close(h);
    throw std::runtime_error("add_native_block : ABI incompatible -- cle du loader '" + loader_key +
                             "' != cle du module '" + module_key +
                             "'. Recompiler le loader avec le MEME compilateur, standard C++ et "
                             "en-tetes adc que le module _adc.");
  }
  // Installateur natif du loader : reinterpret_cast<System*>(this) puis add_compiled_model<ProdModel>.
  // Schema (limiter/riemann/recon/time/gamma/substeps) marshale en arguments plats extern "C" : le
  // loader reconstruit imex/recon_prim et appelle le gabarit. evolve est passe pour qu'un bloc fond
  // fixe (non avance) soit possible comme via add_block.
  using install_fn_t = void (*)(void*, const char*, const char*, const char*, const char*,
                                const char*, double, int, int, int, double);
  auto install = reinterpret_cast<install_fn_t>(adc::dynlib::sym(h,"adc_install_native"));
  if (!install) {
    adc::dynlib::close(h);
    throw std::runtime_error("add_native_block : adc_install_native absent du .so (regenerer via "
                             "dsl.compile_native / compile(backend='production'))");
  }
  // NB signature : adc_install_native transporte desormais pos_floor (limiteur de positivite,
  // ADC-76) en argument plat final. Un loader genere AVANT cet ajout a une autre signature C --
  // il est REJETE en amont par la cle d'ABI (ADC_HEADER_SIG change avec les en-tetes), jamais
  // appele avec un layout d'arguments errone.
  install(static_cast<void*>(self), name.c_str(), limiter.c_str(), riemann.c_str(), recon.c_str(),
          time.c_str(), gamma, substeps, evolve ? 1 : 0, stride, pos_floor);
  // Le .so reste charge (RTLD_GLOBAL) pour la duree du process : le bloc installe pointe du code
  // (fermetures de block_builder, foncteurs nommes) qui y vit. On NE le ferme PAS (pas de propriete
  // partagee a accrocher : les fermetures sont copiees dans le registre du System mais leur CODE est
  // dans le .so). Coherent avec un binaire de production ou le modele est lie a vie.
#endif  // _WIN32 (chemin production POSIX-only ; Windows = throw, cf. ADC-100)
}

}  // namespace native_loader
}  // namespace adc
