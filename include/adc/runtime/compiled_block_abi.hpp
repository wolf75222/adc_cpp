#pragma once

#include <adc/runtime/block_builder.hpp>

#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence : barriere avant lecture hote (memoire unifiee)
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>

#include <adc/core/physical_model.hpp>  // aux_comps<Model> : largeur du canal aux du modele genere
#include <adc/runtime/runtime_params.hpp>  // RuntimeParams : params RUNTIME (P7-b) transportes par l'ABI

#include <string>
#include <type_traits>
#include <utility>

/// @file
/// @brief ABI C d'un bloc COMPILE AOT : un modele genere par le DSL, compile ahead-of-time en .so,
///        expose son residu / avance / vitesse d'onde / second membre Poisson via des fonctions
///        extern "C". Le .so EXECUTE le chemin de PRODUCTION (block_builder.hpp : assemble_rhs<Limiter,
///        Flux>, SSPRK2/IMEX du coeur) sur le CompositeModel<Gen...> connu a SA compilation : la
///        numerique est inlinee, identique a un bloc natif add_block. Seul le maillage transite par
///        des tableaux plats (marshaling cote System, comme add_dynamic_block) : aucune dependance de
///        symbole C++ a travers le dlopen (ABI propre). La version zero-copie / GPU (modele compile
///        dans le build Kokkos, sans marshaling) est un chemin distinct.
///
/// Disposition des tableaux : composante-majeur c*n*n + j*n + i (comme System::copy_state). Un .so
/// genere se reduit a : inclure la brique + cet en-tete, poser
///   namespace adc_generated { using Model = adc::CompositeModel<...>; }
/// puis ADC_DEFINE_COMPILED_BLOCK(adc_generated::Model).

namespace adc::compiled_block {

/// Maillage local mono-grille reconstruit dans le .so + aux remplie depuis le tableau du System.
struct LocalGrid {
  Box2D dom;
  Geometry geom;
  BoxArray ba;
  DistributionMapping dm;
  BCRec bc;
  bool periodic;
  MultiFab aux;  // naux comp (phi, grad phi, [B_z, T_e...]), 1 ghost
};

/// @p naux = largeur du canal aux que le modele LIT (aux_comps<Model>(), >= 3). Le tableau plat
/// @p aux_in porte EXACTEMENT @p naux composantes (marshale par le System via copy_state) ; on les
/// copie toutes dans l'aux interne pour que assemble_rhs (load_aux<aux_comps<Model>()>) lise B_z/T_e.
/// Les ghosts (B_z/T_e inclus) sont remplis par les memes CL que le System.
inline LocalGrid make_grid(int n, double dx, double dy, bool periodic, const double* aux_in,
                           int naux) {
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, dx * n, 0.0, dy * n};
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), 1);  // .so mono-rang (bloc CPU prototype)
  BCRec bc;
  if (!periodic) bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Foextrap;
  MultiFab aux(ba, dm, naux, 1);
  aux.set_val(0.0);
  if (aux_in) {  // aux interieur depuis le tableau, puis ghosts (memes CL que le System)
    Array4 a = aux.fab(0).array();
    const std::size_t nn = static_cast<std::size_t>(n) * n;
    for (int c = 0; c < naux; ++c)
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
          a(i, j, c) = aux_in[static_cast<std::size_t>(c) * nn + static_cast<std::size_t>(j) * n + i];
    if (periodic)
      fill_boundary(aux, dom, Periodicity{true, true});
    else
      fill_ghosts(aux, dom, bc);
  }
  return LocalGrid{dom, geom, ba, dm, bc, periodic, std::move(aux)};
}

/// Recopie @p nv composantes du tableau plat composante-majeur @p in vers l'INTERIEUR du fab(0) (sans
/// toucher les ghosts). Pendant entree du marshaling : disposition c*n*n + j*n + i (cf. System::copy_state).
inline void fill_interior(MultiFab& mf, const double* in, int n, int nv) {
  Array4 a = mf.fab(0).array();
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  for (int c = 0; c < nv; ++c)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        a(i, j, c) = in[static_cast<std::size_t>(c) * nn + static_cast<std::size_t>(j) * n + i];
}

/// Recopie l'INTERIEUR du fab(0) (@p nv composantes) vers le tableau plat composante-majeur @p out.
/// Pendant sortie de fill_interior. device_fence() PREALABLE : les kernels d'assemble_rhs / d'avance
/// sont ASYNC, on attend avant de lire la memoire unifiee cote hote (sinon course -> resultat partiel).
inline void extract(const MultiFab& mf, double* out, int n, int nv) {
  device_fence();  // assemble_rhs / l'avance ont lance des kernels ASYNC ; attendre avant la
                   // lecture hote de la memoire unifiee (sinon course -> resultat partiel sur GPU).
  const ConstArray4 a = mf.fab(0).const_array();
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  for (int c = 0; c < nv; ++c)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        out[static_cast<std::size_t>(c) * nn + static_cast<std::size_t>(j) * n + i] = a(i, j, c);
}

/// Detecte (SFINAE) si une brique B expose un membre `params` de type adc::RuntimeParams (les briques
/// DSL n'en portent un que si une formule lit un parametre runtime, P7-b). Une brique native ou une
/// brique DSL SANS param runtime n'en a pas -> les helpers ci-dessous sont alors des no-op (bit-identite).
template <class B, class = void>
struct HasRuntimeParams : std::false_type {};
template <class B>
struct HasRuntimeParams<B, std::void_t<decltype(std::declval<B&>().params)>>
    : std::is_same<std::decay_t<decltype(std::declval<B&>().params)>, RuntimeParams> {};

/// Ecrit @p rp dans le membre `params` de la brique @p b si elle en a un (P7-b) ; sinon no-op. La
/// brique se comporte alors comme avec ces valeurs de parametres a la place de ses defauts de
/// declaration.
template <class B>
inline void apply_runtime_params(B& b, const RuntimeParams& rp) {
  if constexpr (HasRuntimeParams<B>::value) b.params = rp;
}

/// Construit un Model (CompositeModel<Hyp, Src, Ell>) et y injecte les @p npar valeurs de parametres
/// runtime @p pvals dans CHAQUE brique qui porte un membre `params` (P7-b). @p pvals == nullptr ou
/// @p npar == 0 -> aucune injection : le Model garde ses defauts de declaration (donc identique a un
/// param const). Centralise le seul point de l'ABI AOT qui depend des params runtime ; les briques
/// natives / sans param runtime sont inchangees (apply_runtime_params est un no-op chez elles).
template <class Model>
Model make_model_with_params(const double* pvals, int npar) {
  Model model{};
  if (pvals && npar > 0) {
    RuntimeParams rp;
    rp.count = npar > kMaxRuntimeParams ? kMaxRuntimeParams : npar;
    for (int k = 0; k < rp.count; ++k) rp.values[k] = static_cast<Real>(pvals[k]);
    apply_runtime_params(model.hyp, rp);
    apply_runtime_params(model.src, rp);
    apply_runtime_params(model.ell, rp);
  }
  return model;
}

/// Nombre de parametres runtime que le membre `params` d'une brique @p b porte (0 si la brique n'en a
/// pas). Sert a model_nparams pour reporter au loader la taille attendue du bloc de parametres.
template <class B>
inline int brick_nparams(const B& b) {
  if constexpr (HasRuntimeParams<B>::value) return b.params.count;
  else { (void)b; return 0; }
}

/// Bloc `params` (RuntimeParams complet) du modele : la PREMIERE brique qui en porte un (toutes
/// portent le meme bloc complet quand le modele a des params runtime). RuntimeParams vide (count=0)
/// si aucune brique n'a de param runtime.
template <class Model>
inline RuntimeParams model_params(const Model& model) {
  if constexpr (HasRuntimeParams<std::decay_t<decltype(model.hyp)>>::value) return model.hyp.params;
  else if constexpr (HasRuntimeParams<std::decay_t<decltype(model.src)>>::value) return model.src.params;
  else if constexpr (HasRuntimeParams<std::decay_t<decltype(model.ell)>>::value) return model.ell.params;
  else { (void)model; return RuntimeParams{}; }
}

/// Nombre de parametres runtime declares par @p Model (P7-b) : max des `params.count` de ses 3 briques
/// (chacune porte le MEME bloc complet quand le modele a des params runtime ; 0 partout sinon). Expose
/// par adc_compiled_nparams() pour que le loader dimensionne le bloc de valeurs qu'il transmet.
template <class Model>
inline int model_nparams() {
  Model model{};
  int a = brick_nparams(model.hyp), b = brick_nparams(model.src), c = brick_nparams(model.ell);
  int m = a > b ? a : b;
  return m > c ? m : c;
}

/// Ecrit les valeurs de DECLARATION des parametres runtime de @p Model dans @p out (au moins
/// model_nparams<Model>() doubles). Sert au loader a SEEDER son bloc de valeurs : un set_param ulterieur
/// n'ecrase qu'une entree, les autres gardent leur defaut de declaration (pas de remise a zero).
template <class Model>
void model_param_defaults(double* out) {
  Model model{};
  const RuntimeParams rp = model_params(model);
  for (int k = 0; k < rp.count; ++k) out[k] = static_cast<double>(rp.values[k]);
}

/// Residu -div F + S du chemin de production (assemble_rhs<Limiter, Flux>) sur le modele genere.
/// GHOSTS : on alloue l'etat local avec block_n_ghost(lim) (3 pour weno5, son stencil 5 points,
/// sinon 2 MUSCL) -- MEME mecanisme que System::add_block (set_block_ghosts, PR #88). Sans cette
/// largeur, assemble_rhs lirait hors bornes de la grille locale du .so. none/minmod/vanleer (<=2
/// ghosts) : allocation et resultat bit-identiques a l'historique (le surplus ghost est un no-op).
template <class Model>
void residual(const double* U, double* R, const double* aux_in, int n, double dx, double dy,
              bool periodic, const std::string& lim, const std::string& riem, bool recon_prim,
              const double* pvals = nullptr, int npar = 0, double pos_floor = 0) {
  LocalGrid lg = make_grid(n, dx, dy, periodic, aux_in, aux_comps<Model>());
  MultiFab Umf(lg.ba, lg.dm, Model::n_vars, block_n_ghost(lim)), Rmf(lg.ba, lg.dm, Model::n_vars, 0);
  fill_interior(Umf, U, n, Model::n_vars);
  const GridContext ctx{lg.dom, lg.bc, lg.geom, &lg.aux};
  Model model = make_model_with_params<Model>(pvals, npar);  // P7-b : params runtime (no-op si const)
  BlockClosures clo = make_block(model, lim, riem, ctx, /*imex=*/false, recon_prim, "ssprk2", {}, {},
                                 nullptr, static_cast<Real>(pos_floor));
  clo.rhs_into(Umf, Rmf);
  extract(Rmf, R, n, Model::n_vars);
}

/// Avance de @p nsub sous-pas (SSPRK2 explicite, ou ForwardEuler + source implicite si imex).
template <class Model>
void advance(double* U, const double* aux_in, int n, double dx, double dy, bool periodic,
             const std::string& lim, const std::string& riem, bool recon_prim, bool imex,
             double dt, int nsub, const double* pvals = nullptr, int npar = 0,
             double pos_floor = 0) {
  LocalGrid lg = make_grid(n, dx, dy, periodic, aux_in, aux_comps<Model>());
  // block_n_ghost(lim) : 3 pour weno5 (stencil 5 points), 2 MUSCL sinon. cf. residual ci-dessus.
  MultiFab Umf(lg.ba, lg.dm, Model::n_vars, block_n_ghost(lim));
  fill_interior(Umf, U, n, Model::n_vars);
  const GridContext ctx{lg.dom, lg.bc, lg.geom, &lg.aux};
  Model model = make_model_with_params<Model>(pvals, npar);  // P7-b : params runtime (no-op si const)
  BlockClosures clo = make_block(model, lim, riem, ctx, imex, recon_prim, "ssprk2", {}, {}, nullptr,
                                 static_cast<Real>(pos_floor));
  clo.advance(Umf, dt, nsub);
  extract(Umf, U, n, Model::n_vars);
}

/// Vitesse d'onde max du bloc (pour le pas CFL cote System) via make_max_speed sur le modele genere.
template <class Model>
double max_speed(const double* U, const double* aux_in, int n, double dx, double dy, bool periodic,
                 const double* pvals = nullptr, int npar = 0) {
  LocalGrid lg = make_grid(n, dx, dy, periodic, aux_in, aux_comps<Model>());
  MultiFab Umf(lg.ba, lg.dm, Model::n_vars, 2);
  fill_interior(Umf, U, n, Model::n_vars);
  const GridContext ctx{lg.dom, lg.bc, lg.geom, &lg.aux};
  Model model = make_model_with_params<Model>(pvals, npar);  // P7-b : params runtime (no-op si const)
  return make_max_speed(model, ctx)(Umf);
}

/// Conversion PRIMITIF -> CONSERVATIF cellule par cellule (M.to_conservative) sur des tableaux plats
/// ncomp*n*n composante-majeur. Sert a System::set_primitive_state via le bloc AOT (init depuis les
/// primitives). Modele sans conversion (scalaire) -> identite (HasPrimitiveVars faux). Pas de
/// maillage / ghosts : conversion PONCTUELLE, donc une simple boucle sur les n*n cellules.
template <class Model>
void to_conservative(const double* P, double* U, int n, const double* pvals = nullptr, int npar = 0) {
  constexpr int NV = Model::n_vars;
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  Model model = make_model_with_params<Model>(pvals, npar);  // P7-b : params runtime (no-op si const)
  for (std::size_t k = 0; k < nn; ++k) {
    if constexpr (HasPrimitiveVars<Model>) {
      typename Model::Prim p{};
      for (int c = 0; c < NV; ++c) p[c] = static_cast<Real>(P[static_cast<std::size_t>(c) * nn + k]);
      const typename Model::State u = model.to_conservative(p);
      for (int c = 0; c < NV; ++c) U[static_cast<std::size_t>(c) * nn + k] = static_cast<double>(u[c]);
    } else {
      for (int c = 0; c < NV; ++c) U[static_cast<std::size_t>(c) * nn + k] = P[static_cast<std::size_t>(c) * nn + k];
    }
  }
}

/// Conversion CONSERVATIF -> PRIMITIF cellule par cellule (M.to_primitive). Pendant diagnostic de
/// to_conservative (System::get_primitive_state via le bloc AOT). Identite si le modele n'expose pas
/// de conversion.
template <class Model>
void to_primitive(const double* U, double* P, int n, const double* pvals = nullptr, int npar = 0) {
  constexpr int NV = Model::n_vars;
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  Model model = make_model_with_params<Model>(pvals, npar);  // P7-b : params runtime (no-op si const)
  for (std::size_t k = 0; k < nn; ++k) {
    if constexpr (HasPrimitiveVars<Model>) {
      typename Model::State u{};
      for (int c = 0; c < NV; ++c) u[c] = static_cast<Real>(U[static_cast<std::size_t>(c) * nn + k]);
      const typename Model::Prim p = model.to_primitive(u);
      for (int c = 0; c < NV; ++c) P[static_cast<std::size_t>(c) * nn + k] = static_cast<double>(p[c]);
    } else {
      for (int c = 0; c < NV; ++c) P[static_cast<std::size_t>(c) * nn + k] = U[static_cast<std::size_t>(c) * nn + k];
    }
  }
}

/// PROJECTION PONCTUELLE post-pas (ADC-177) : U(k) <- model.project(U(k), aux(k)) sur les n*n
/// cellules, tableaux plats composante-majeur (meme disposition que to_conservative). PONCTUELLE :
/// aucun maillage ni ghost, simple boucle de cellules ; l'aux est marshale par cellule depuis le
/// tableau plat (composantes canoniques via ADC_AUX_FIELDS + champs nommes, miroir de load_aux).
/// Modele SANS trait HasPointwiseProjection -> no-op (le loader ne devrait jamais l'appeler :
/// adc_compiled_has_projection() rend 0).
template <class Model>
void pointwise_project(double* U, const double* aux_in, int n, const double* pvals = nullptr,
                       int npar = 0) {
  if constexpr (HasPointwiseProjection<Model>) {
    constexpr int NV = Model::n_vars;
    constexpr int naux = aux_comps<Model>();
    const std::size_t nn = static_cast<std::size_t>(n) * n;
    Model model = make_model_with_params<Model>(pvals, npar);  // P7-b : params runtime (no-op si const)
    for (std::size_t k = 0; k < nn; ++k) {
      typename Model::State u{};
      for (int c = 0; c < NV; ++c)
        u[c] = static_cast<Real>(U[static_cast<std::size_t>(c) * nn + k]);
      Aux a{};
      if (aux_in) {
        a.phi = static_cast<Real>(aux_in[k]);
        a.grad_x = static_cast<Real>(aux_in[nn + k]);
        a.grad_y = static_cast<Real>(aux_in[2 * nn + k]);
        // Champs extra canoniques depuis la SOURCE UNIQUE ADC_AUX_FIELDS (cf. native_loader) : une
        // composante n'est lue que si le modele la consomme (naux) -- le System marshale toujours au
        // moins naux composantes (ensure_aux_width a l'ajout du bloc).
#define ADC_AUX_MARSHAL(name, idx) \
  if constexpr (naux > (idx)) a.name = static_cast<Real>(aux_in[(idx) * nn + k]);
        ADC_AUX_FIELDS(ADC_AUX_MARSHAL)
#undef ADC_AUX_MARSHAL
        // Champs aux NOMMES (aux_field, ADC-70) : composantes a partir de kAuxNamedBase, miroir de
        // load_aux (borne deroulee et clampee a kAuxMaxExtra, jamais d'acces hors tableau).
        if constexpr (naux > kAuxNamedBase) {
          constexpr int n_extra =
              (naux - kAuxNamedBase) < kAuxMaxExtra ? (naux - kAuxNamedBase) : kAuxMaxExtra;
          for (int x = 0; x < n_extra; ++x)
            a.extra[x] = static_cast<Real>(aux_in[static_cast<std::size_t>(kAuxNamedBase + x) * nn + k]);
        }
      }
      const typename Model::State p = model.project(u, a);
      for (int c = 0; c < NV; ++c)
        U[static_cast<std::size_t>(c) * nn + k] = static_cast<double>(p[c]);
    }
  } else {
    (void)U; (void)aux_in; (void)n; (void)pvals; (void)npar;
  }
}

/// Second membre elliptique f(U) du bloc (le System l'ajoute a sa somme de Poisson).
template <class Model>
void poisson_rhs(const double* U, double* rhs_out, int n, const double* pvals = nullptr,
                 int npar = 0) {
  Box2D dom = Box2D::from_extents(n, n);
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), 1);
  MultiFab Umf(ba, dm, Model::n_vars, 2), rhsmf(ba, dm, 1, 0);
  fill_interior(Umf, U, n, Model::n_vars);
  rhsmf.set_val(0.0);
  Model model = make_model_with_params<Model>(pvals, npar);  // P7-b : params runtime (no-op si const)
  make_poisson_rhs(model)(Umf, rhsmf);
  const ConstArray4 r = rhsmf.fab(0).const_array();
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  (void)nn;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) rhs_out[static_cast<std::size_t>(j) * n + i] = r(i, j, 0);
}

}  // namespace adc::compiled_block

/// Definit l'ABI extern "C" du bloc compile pour le modele @p MODEL (un alias SANS virgule de
/// niveau superieur : poser `using Model = adc::CompositeModel<...>;` puis passer cet alias).
///
/// P7-b (PARAMS RUNTIME). En PLUS des symboles historiques (inchanges, donc bit-identiques pour un
/// modele sans param runtime), on emet des variantes SUFFIXEES `_p` qui prennent un bloc plat de
/// parametres runtime (const double* pvals, int npar) injecte dans le modele avant l'execution, et
/// `adc_compiled_nparams()` (nb de params runtime du modele, 0 si aucun). Le loader (native_loader.hpp)
/// PREFERE les symboles `_p` quand ils sont presents ; un vieux .so sans eux retombe sur les symboles
/// historiques (aucun param runtime). Les symboles historiques delegent aux memes templates avec
/// pvals=nullptr/npar=0 -> bit-identite stricte du chemin params-const.
///
/// ADC-177 (PROJECTION PONCTUELLE post-pas) : symboles OPTIONNELS adc_compiled_has_projection()
/// (1 si le modele declare le trait HasPointwiseProjection, 0 sinon) et adc_compiled_project_p
/// (U <- project(U, aux) sur les tableaux plats). Un vieux .so ne les expose pas -> le loader
/// n'installe aucune projection (bit-identique). Un .so NEUF charge par un VIEUX module : la
/// projection serait ignoree (le vieux loader n'interroge pas ces symboles) -- combinaison exclue en
/// pratique, dsl.py et le module _adc etant livres ensemble (cf. docs/DSL_API.md).
#define ADC_DEFINE_COMPILED_BLOCK(MODEL)                                                            \
  extern "C" int adc_model_nvars() { return MODEL::n_vars; }                                        \
  extern "C" int adc_compiled_naux() { return adc::aux_comps<MODEL>(); }                            \
  extern "C" int adc_compiled_nparams() { return adc::compiled_block::model_nparams<MODEL>(); }     \
  extern "C" void adc_compiled_param_defaults(double* out) {                                        \
    adc::compiled_block::model_param_defaults<MODEL>(out);                                          \
  }                                                                                                 \
  extern "C" void adc_compiled_residual(const double* U, double* R, const double* aux, int n,       \
                                        double dx, double dy, int periodic, const char* lim,        \
                                        const char* riem, int recon_prim) {                         \
    adc::compiled_block::residual<MODEL>(U, R, aux, n, dx, dy, periodic != 0, lim, riem,            \
                                         recon_prim != 0);                                          \
  }                                                                                                 \
  extern "C" void adc_compiled_residual_p(const double* U, double* R, const double* aux, int n,     \
                                          double dx, double dy, int periodic, const char* lim,      \
                                          const char* riem, int recon_prim, const double* pvals,    \
                                          int npar, double pos_floor) {                             \
    adc::compiled_block::residual<MODEL>(U, R, aux, n, dx, dy, periodic != 0, lim, riem,            \
                                         recon_prim != 0, pvals, npar, pos_floor);                  \
  }                                                                                                 \
  extern "C" void adc_compiled_advance(double* U, const double* aux, int n, double dx, double dy,   \
                                       int periodic, const char* lim, const char* riem,             \
                                       int recon_prim, int imex, double dt, int nsub) {             \
    adc::compiled_block::advance<MODEL>(U, aux, n, dx, dy, periodic != 0, lim, riem,                \
                                        recon_prim != 0, imex != 0, dt, nsub);                      \
  }                                                                                                 \
  extern "C" void adc_compiled_advance_p(double* U, const double* aux, int n, double dx, double dy, \
                                         int periodic, const char* lim, const char* riem,           \
                                         int recon_prim, int imex, double dt, int nsub,             \
                                         const double* pvals, int npar, double pos_floor) {         \
    adc::compiled_block::advance<MODEL>(U, aux, n, dx, dy, periodic != 0, lim, riem,                \
                                        recon_prim != 0, imex != 0, dt, nsub, pvals, npar,          \
                                        pos_floor);                                                 \
  }                                                                                                 \
  extern "C" double adc_compiled_max_speed(const double* U, const double* aux, int n, double dx,    \
                                           double dy, int periodic) {                               \
    return adc::compiled_block::max_speed<MODEL>(U, aux, n, dx, dy, periodic != 0);                 \
  }                                                                                                 \
  extern "C" double adc_compiled_max_speed_p(const double* U, const double* aux, int n, double dx,  \
                                             double dy, int periodic, const double* pvals,          \
                                             int npar) {                                            \
    return adc::compiled_block::max_speed<MODEL>(U, aux, n, dx, dy, periodic != 0, pvals, npar);    \
  }                                                                                                 \
  extern "C" void adc_compiled_poisson_rhs(const double* U, double* rhs, int n) {                   \
    adc::compiled_block::poisson_rhs<MODEL>(U, rhs, n);                                             \
  }                                                                                                 \
  extern "C" void adc_compiled_poisson_rhs_p(const double* U, double* rhs, int n,                   \
                                             const double* pvals, int npar) {                       \
    adc::compiled_block::poisson_rhs<MODEL>(U, rhs, n, pvals, npar);                                \
  }                                                                                                 \
  extern "C" void adc_compiled_to_conservative(const double* P, double* U, int n) {                 \
    adc::compiled_block::to_conservative<MODEL>(P, U, n);                                           \
  }                                                                                                 \
  extern "C" void adc_compiled_to_conservative_p(const double* P, double* U, int n,                 \
                                                 const double* pvals, int npar) {                   \
    adc::compiled_block::to_conservative<MODEL>(P, U, n, pvals, npar);                              \
  }                                                                                                 \
  extern "C" void adc_compiled_to_primitive(const double* U, double* P, int n) {                    \
    adc::compiled_block::to_primitive<MODEL>(U, P, n);                                              \
  }                                                                                                 \
  extern "C" void adc_compiled_to_primitive_p(const double* U, double* P, int n,                    \
                                              const double* pvals, int npar) {                      \
    adc::compiled_block::to_primitive<MODEL>(U, P, n, pvals, npar);                                 \
  }                                                                                                 \
  extern "C" int adc_compiled_has_projection() {                                                    \
    return adc::HasPointwiseProjection<MODEL> ? 1 : 0;                                              \
  }                                                                                                 \
  extern "C" void adc_compiled_project_p(double* U, const double* aux, int n,                       \
                                         const double* pvals, int npar) {                           \
    adc::compiled_block::pointwise_project<MODEL>(U, aux, n, pvals, npar);                          \
  }
