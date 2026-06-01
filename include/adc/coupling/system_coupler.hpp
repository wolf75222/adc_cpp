#pragma once

#include <adc/core/coupled_system.hpp>
#include <adc/core/types.hpp>
#include <adc/coupling/coupled_source.hpp>
#include <adc/coupling/elliptic_rhs.hpp>
#include <adc/elliptic/elliptic_problem.hpp>
#include <adc/elliptic/elliptic_solver.hpp>
#include <adc/elliptic/geometric_mg.hpp>
#include <adc/integrator/implicit_stepper.hpp>
#include <adc/integrator/scheduler.hpp>
#include <adc/integrator/time_steppers.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/operator/spatial_operator.hpp>
#include <adc/parallel/comm.hpp>

#include <functional>
#include <type_traits>
#include <utility>

// Coupleur de systeme mono-niveau.
//
// Le Coupler historique avance un seul PhysicalModel. SystemCoupler est la couche
// au-dessus : un CoupledSystem contient plusieurs EquationBlock, le RHS elliptique
// peut lire tous les blocs, et chaque bloc porte sa discretisation spatiale et sa
// politique de temps.
//
// C'est volontairement un orchestrateur : il ne remplace ni PhysicalModel
// (formules locales), ni assemble_rhs (operateur spatial), ni GeometricMG
// (elliptique). Il connecte ces briques pour rendre explicite le niveau
// d'abstraction voulu par la discussion tuteur :
//   electrons : modele + schema + implicite + sous-pas
//   ions      : modele + schema + explicite
//   Poisson   : assembleur global f(U_e, U_i, ...)

namespace adc {

namespace detail {
template <class>
inline constexpr bool always_false_v = false;

template <class Block>
struct ScopedBlockState {
  Block& block;
  MultiFab* old_state;

  ScopedBlockState(Block& b, MultiFab& stage_state)
      : block(b), old_state(b.state) {
    block.state = &stage_state;
  }

  ~ScopedBlockState() { block.state = old_state; }
};
}  // namespace detail

template <CoupledSystemLike System, class RhsAssembler,
          class Elliptic = GeometricMG>
class SystemCoupler {
  static_assert(EllipticSolver<Elliptic>,
                "le backend elliptique doit modeler EllipticSolver");

 public:
  SystemCoupler(System system, const Geometry& geom, const BoxArray& ba,
                const BCRec& bcPhi, RhsAssembler rhs_assembler,
                std::function<bool(Real, Real)> active = {})
      : system_(std::move(system)),
        rhs_assembler_(std::move(rhs_assembler)),
        geom_(geom),
        ba_(ba),
        dm_(ba.size(), n_ranks()),
        bcPhi_(bcPhi),
        aux_bc_(derive_aux_bc(bcPhi)),
        mg_(geom, ba, bcPhi, std::move(active)),
        aux_(ba, dm_, 3, 1) {}

  System& system() { return system_; }
  const System& system() const { return system_; }

  MultiFab& phi() { return mg_.phi(); }
  const MultiFab& aux() const { return aux_; }

  void solve_fields() {
    rhs_assembler_(system_, mg_.rhs());
    mg_.solve();
    derive_aux();
  }

  // Avance les blocs selon leur TimePolicy.
  //
  // Les blocs explicites SSPRK2/SSPRK3 sont executes ici. Les blocs implicites
  // ou IMEX sont delegues au callback : c'est l'endroit ou brancher Newton,
  // solveur lineaire, solveur collisionnel, etc., sans imposer une API prematuree
  // au coeur.
  template <class ImplicitAdvance>
  void step(Real dt, ImplicitAdvance&& implicit_advance) {
    ImplicitAdvance& advance_implicit = implicit_advance;
    advance_subcycled(system_, dt, [&](auto& block, Real h, int s, int n) {
      using Block = std::decay_t<decltype(block)>;
      constexpr TimeTreatment treatment = block_time_treatment_v<Block>;
      if constexpr (treatment == TimeTreatment::Explicit) {
        advance_explicit_block(block, h);
      } else if constexpr (treatment == TimeTreatment::Implicit ||
                           treatment == TimeTreatment::IMEX) {
        solve_fields();
        // IMEX = vrai forward-backward (revue Codex 9.1) : le coeur avance le TRANSPORT
        // explicite (−div F, source-free), puis le callback traite la SOURCE en implicite.
        // Implicite pur : pas de transport, tout au callback.
        if constexpr (treatment == TimeTreatment::IMEX) explicit_transport(block, h);
        advance_implicit(*this, block, h, s, n);
      }
    });
  }

  // Surcharge pratique pour un systeme entierement explicite.
  void step(Real dt) {
    step(dt, [](auto&, auto& block, Real, int, int) {
      using Block = std::decay_t<decltype(block)>;
      static_assert(detail::always_false_v<Block>,
                    "SystemCoupler::step(dt) ne peut pas avancer un bloc "
                    "implicite/IMEX sans callback");
    });
  }

  // Pas de SOURCE DE COUPLAGE inter-especes (splitting forward-Euler) : on
  // rafraichit phi (aux) puis on laisse la source lire tous les blocs et les
  // mettre a jour. C'est l'endroit distinct de model.source (locale au bloc) :
  // ici S_e peut dependre de U_i et de phi. NoCoupledSource => no-op.
  template <class CoupledSource>
  void coupled_source_step(CoupledSource&& src, Real dt) {
    static_assert(CoupledSourceFor<std::decay_t<CoupledSource>, System>,
                  "coupled_source_step attend une CoupledSource : "
                  "apply(system, aux, dt)");
    solve_fields();
    src.apply(system_, aux_, dt);
  }

 private:
  static BCRec derive_aux_bc(const BCRec& b) {
    auto t = [](BCType x) {
      return x == BCType::Periodic ? BCType::Periodic : BCType::Foextrap;
    };
    BCRec a;
    a.xlo = t(b.xlo);
    a.xhi = t(b.xhi);
    a.ylo = t(b.ylo);
    a.yhi = t(b.yhi);
    return a;
  }

  void derive_aux() {
    fill_ghosts(mg_.phi(), geom_.domain, bcPhi_);
    const Real cx = Real(1) / (2 * geom_.dx());
    const Real cy = Real(1) / (2 * geom_.dy());
    field_postprocess(mg_.phi(), aux_, cx, cy,
                      FieldPostProcess{FieldPostProcess::GradSign::Plus, true});
    fill_ghosts(aux_, geom_.domain, aux_bc_);
  }

  template <class Limiter, class NumericalFlux, class Block>
  void stage_rhs(Block& block, MultiFab& state, MultiFab& R,
                 bool recompute_aux) {
    if (recompute_aux) {
      detail::ScopedBlockState<Block> swap(block, state);
      solve_fields();
    }
    fill_ghosts(state, geom_.domain, block.bc);
    assemble_rhs<Limiter, NumericalFlux>(block.model, state, aux_, geom_, R);
  }

  // Avance explicite d'un bloc : on DELEGUE le schema a un objet TimeStepper du coeur
  // (SSPRK2Step / SSPRK3Step) en lui passant l'evaluateur de residu du bloc (fill_ghosts +
  // re-solve aux par etage + assemble_rhs<Spatial>). Bit-identique a l'ancien SSPRK inline.
  template <class Block>
  void advance_explicit_block(Block& block, Real dt) {
    using Time = TimePolicyTraits<typename Block::Time>;
    using Method = typename Time::Method;
    using Limiter = typename Block::Spatial::Limiter;
    using NumericalFlux = typename Block::Spatial::NumericalFlux;
    static_assert(Time::treatment == TimeTreatment::Explicit,
                  "advance_explicit_block attend un bloc explicite");

    auto rhs_eval = [&](MultiFab& stage, MultiFab& R) {
      stage_rhs<Limiter, NumericalFlux>(block, stage, R, /*recompute_aux=*/true);
    };
    // Method = tag du coeur (SSPRK2/SSPRK3) -> objet stepper correspondant ; OU un
    // TimeStepper fourni par l'utilisateur (son propre take_step), instancie ici.
    if constexpr (std::is_same_v<Method, SSPRK3>)
      SSPRK3Step{}.take_step(rhs_eval, block.U(), dt);
    else if constexpr (std::is_same_v<Method, SSPRK2>)
      SSPRK2Step{}.take_step(rhs_eval, block.U(), dt);
    else if constexpr (TimeStepper<Method>)
      Method{}.take_step(rhs_eval, block.U(), dt);
    else
      static_assert(detail::always_false_v<Method>,
                    "Method explicite doit etre SSPRK2, SSPRK3, ou un TimeStepper "
                    "(objet a take_step(rhs_eval, U, dt)) fourni par l'utilisateur");
  }

  // Demi-pas EXPLICITE d'un bloc IMEX : transport seul (−div F, source-free), Euler avant.
  // La source (raide) est traitee separement en implicite par le callback. aux suppose a
  // jour (solve_fields appele juste avant dans step).
  template <class Block>
  void explicit_transport(Block& block, Real dt) {
    using Model = typename Block::Model;
    using Limiter = typename Block::Spatial::Limiter;
    using NumericalFlux = typename Block::Spatial::NumericalFlux;
    const SourceFreeModel<Model> sf{block.model};
    MultiFab R(ba_, dm_, Model::n_vars, 0);
    fill_ghosts(block.U(), geom_.domain, block.bc);
    assemble_rhs<Limiter, NumericalFlux>(sf, block.U(), aux_, geom_, R);
    saxpy(block.U(), dt, R);
  }

  System system_;
  RhsAssembler rhs_assembler_;
  Geometry geom_;
  BoxArray ba_;
  DistributionMapping dm_;
  BCRec bcPhi_, aux_bc_;
  Elliptic mg_;
  MultiFab aux_;
};

}  // namespace adc
