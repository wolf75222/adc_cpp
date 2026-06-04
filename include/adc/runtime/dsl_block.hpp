#pragma once

#include <adc/core/physical_model.hpp>  // aux_comps<Model> : largeur aux demandee par le modele
#include <adc/runtime/block_builder.hpp>
#include <adc/runtime/system.hpp>

#include <functional>
#include <string>
#include <utility>

/// @file
/// @brief add_compiled_model : branche un modele COMPILE (un CompositeModel, typiquement genere par
///        le DSL puis inclus au moment de la COMPILATION) comme bloc NATIF du System.
///
/// Difference avec System::add_compiled_block (.so + ABI extern "C" + marshaling de tableaux plats,
/// pour le prototypage RUNTIME cote Python) : ici le modele est connu a la compilation, donc
/// block_builder fabrique les fermetures sur le CONTEXTE DE GRILLE REEL du System (grid_context) et le
/// bloc tourne EXACTEMENT le chemin de production -- le residu fait fill_boundary (halos MPI) +
/// assemble_rhs (Kokkos) sur les vrais MultiFab du System, SANS recopie : le MEME make_block que
/// add_block. Parite validee sur CPU (build/ ET backend Kokkos Serial) : eval_rhs bit-identique a
/// add_block (cf. tests/test_compiled_model_parity.cpp).
///
/// DEVICE (backend Kokkos Cuda) : tout le chemin make_block est desormais bati sur des FONCTEURS
/// NOMMES (build_block via AdvanceExplicit/AdvanceImex/RhsInto/BlockRhsEval, make_max_speed via
/// MaxSpeed ; cf. block_builder.hpp), au lieu des lambdas premiere-instanciees depuis cette TU
/// appelante. nvcc emet alors fiablement le kernel device imbrique (AssembleRhsKernel) ; le crash
/// device a l'execution (Heisenbug : OK Serial + compute-sanitizer, segfault Cuda) est leve, et la
/// parite A==B (dres=0) est obtenue sur GH200. C'est le backend "compile" de l'ideal
/// m.compile_or_jit() pour un binaire de production.

namespace adc {

/// Ajoute @p model (CompositeModel) comme bloc natif de @p sys avec le schema (limiter x riemann,
/// reconstruction, traitement temporel) demande. @p gamma sert a set_density (energie au repos, 4 var).
template <class Model>
void add_compiled_model(System& sys, const std::string& name, Model model,
                        const std::string& limiter = "minmod",
                        const std::string& riemann = "rusanov",
                        const std::string& recon = "conservative",
                        const std::string& time = "explicit", double gamma = 1.4,
                        int substeps = 1, bool evolve = true) {
  const bool imex = (time == "imex");
  const bool recon_prim = (recon == "primitive");
  // Le bloc peut lire des champs auxiliaires supplementaires (aux_comps<Model> > 3, p.ex. B_z d'une
  // source magnetisee) : on elargit le canal aux PARTAGE du System AVANT de capturer son adresse,
  // pour que la fermeture lise un aux assez large. Modele de base (3) -> no-op, inchange.
  sys.ensure_aux_width(aux_comps<Model>());
  const GridContext ctx = sys.grid_context();
  BlockClosures clo = make_block(model, limiter, riemann, ctx, imex, recon_prim);
  std::function<Real(const MultiFab&)> ms = make_max_speed(model, ctx);
  std::function<void(const MultiFab&, MultiFab&)> pr = make_poisson_rhs(model);
  sys.install_block(name, Model::n_vars, Model::conservative_vars(), Model::primitive_vars(),
                    gamma, std::move(clo), std::move(ms), std::move(pr), substeps, evolve);
}

}  // namespace adc
