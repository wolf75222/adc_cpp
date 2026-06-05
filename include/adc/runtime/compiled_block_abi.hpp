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

#include <string>
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

inline void fill_interior(MultiFab& mf, const double* in, int n, int nv) {
  Array4 a = mf.fab(0).array();
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  for (int c = 0; c < nv; ++c)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        a(i, j, c) = in[static_cast<std::size_t>(c) * nn + static_cast<std::size_t>(j) * n + i];
}

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

/// Residu -div F + S du chemin de production (assemble_rhs<Limiter, Flux>) sur le modele genere.
/// GHOSTS : on alloue l'etat local avec block_n_ghost(lim) (3 pour weno5, son stencil 5 points,
/// sinon 2 MUSCL) -- MEME mecanisme que System::add_block (set_block_ghosts, PR #88). Sans cette
/// largeur, assemble_rhs lirait hors bornes de la grille locale du .so. none/minmod/vanleer (<=2
/// ghosts) : allocation et resultat bit-identiques a l'historique (le surplus ghost est un no-op).
template <class Model>
void residual(const double* U, double* R, const double* aux_in, int n, double dx, double dy,
              bool periodic, const std::string& lim, const std::string& riem, bool recon_prim) {
  LocalGrid lg = make_grid(n, dx, dy, periodic, aux_in, aux_comps<Model>());
  MultiFab Umf(lg.ba, lg.dm, Model::n_vars, block_n_ghost(lim)), Rmf(lg.ba, lg.dm, Model::n_vars, 0);
  fill_interior(Umf, U, n, Model::n_vars);
  const GridContext ctx{lg.dom, lg.bc, lg.geom, &lg.aux};
  Model model{};
  BlockClosures clo = make_block(model, lim, riem, ctx, /*imex=*/false, recon_prim);
  clo.rhs_into(Umf, Rmf);
  extract(Rmf, R, n, Model::n_vars);
}

/// Avance de @p nsub sous-pas (SSPRK2 explicite, ou ForwardEuler + source implicite si imex).
template <class Model>
void advance(double* U, const double* aux_in, int n, double dx, double dy, bool periodic,
             const std::string& lim, const std::string& riem, bool recon_prim, bool imex,
             double dt, int nsub) {
  LocalGrid lg = make_grid(n, dx, dy, periodic, aux_in, aux_comps<Model>());
  // block_n_ghost(lim) : 3 pour weno5 (stencil 5 points), 2 MUSCL sinon. cf. residual ci-dessus.
  MultiFab Umf(lg.ba, lg.dm, Model::n_vars, block_n_ghost(lim));
  fill_interior(Umf, U, n, Model::n_vars);
  const GridContext ctx{lg.dom, lg.bc, lg.geom, &lg.aux};
  Model model{};
  BlockClosures clo = make_block(model, lim, riem, ctx, imex, recon_prim);
  clo.advance(Umf, dt, nsub);
  extract(Umf, U, n, Model::n_vars);
}

template <class Model>
double max_speed(const double* U, const double* aux_in, int n, double dx, double dy, bool periodic) {
  LocalGrid lg = make_grid(n, dx, dy, periodic, aux_in, aux_comps<Model>());
  MultiFab Umf(lg.ba, lg.dm, Model::n_vars, 2);
  fill_interior(Umf, U, n, Model::n_vars);
  const GridContext ctx{lg.dom, lg.bc, lg.geom, &lg.aux};
  Model model{};
  return make_max_speed(model, ctx)(Umf);
}

/// Conversion PRIMITIF -> CONSERVATIF cellule par cellule (M.to_conservative) sur des tableaux plats
/// ncomp*n*n composante-majeur. Sert a System::set_primitive_state via le bloc AOT (init depuis les
/// primitives). Modele sans conversion (scalaire) -> identite (HasPrimitiveVars faux). Pas de
/// maillage / ghosts : conversion PONCTUELLE, donc une simple boucle sur les n*n cellules.
template <class Model>
void to_conservative(const double* P, double* U, int n) {
  constexpr int NV = Model::n_vars;
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  Model model{};
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
void to_primitive(const double* U, double* P, int n) {
  constexpr int NV = Model::n_vars;
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  Model model{};
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

/// Second membre elliptique f(U) du bloc (le System l'ajoute a sa somme de Poisson).
template <class Model>
void poisson_rhs(const double* U, double* rhs_out, int n) {
  Box2D dom = Box2D::from_extents(n, n);
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), 1);
  MultiFab Umf(ba, dm, Model::n_vars, 2), rhsmf(ba, dm, 1, 0);
  fill_interior(Umf, U, n, Model::n_vars);
  rhsmf.set_val(0.0);
  Model model{};
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
#define ADC_DEFINE_COMPILED_BLOCK(MODEL)                                                            \
  extern "C" int adc_model_nvars() { return MODEL::n_vars; }                                        \
  extern "C" int adc_compiled_naux() { return adc::aux_comps<MODEL>(); }                            \
  extern "C" void adc_compiled_residual(const double* U, double* R, const double* aux, int n,       \
                                        double dx, double dy, int periodic, const char* lim,        \
                                        const char* riem, int recon_prim) {                         \
    adc::compiled_block::residual<MODEL>(U, R, aux, n, dx, dy, periodic != 0, lim, riem,            \
                                         recon_prim != 0);                                          \
  }                                                                                                 \
  extern "C" void adc_compiled_advance(double* U, const double* aux, int n, double dx, double dy,   \
                                       int periodic, const char* lim, const char* riem,             \
                                       int recon_prim, int imex, double dt, int nsub) {             \
    adc::compiled_block::advance<MODEL>(U, aux, n, dx, dy, periodic != 0, lim, riem,                \
                                        recon_prim != 0, imex != 0, dt, nsub);                      \
  }                                                                                                 \
  extern "C" double adc_compiled_max_speed(const double* U, const double* aux, int n, double dx,    \
                                           double dy, int periodic) {                               \
    return adc::compiled_block::max_speed<MODEL>(U, aux, n, dx, dy, periodic != 0);                 \
  }                                                                                                 \
  extern "C" void adc_compiled_poisson_rhs(const double* U, double* rhs, int n) {                   \
    adc::compiled_block::poisson_rhs<MODEL>(U, rhs, n);                                             \
  }                                                                                                 \
  extern "C" void adc_compiled_to_conservative(const double* P, double* U, int n) {                 \
    adc::compiled_block::to_conservative<MODEL>(P, U, n);                                           \
  }                                                                                                 \
  extern "C" void adc_compiled_to_primitive(const double* U, double* P, int n) {                    \
    adc::compiled_block::to_primitive<MODEL>(U, P, n);                                              \
  }
