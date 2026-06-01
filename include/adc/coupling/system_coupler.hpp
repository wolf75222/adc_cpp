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

// Systeme couple mono-niveau : DEUX responsabilites, DEUX classes (retour tuteur §8.2 B).
//
//   SystemAssembler : ASSEMBLE. Couple l'hyperbolique et l'elliptique : RHS de systeme
//     (f = Sum_s q_s n_s), Poisson, aux = (phi, grad phi), et l'evaluateur de residu d'un
//     bloc R = −div F + S. Il ne fait AUCUN pas de temps.
//   SystemDriver    : AVANCE. Porte le schedule (sous-cyclage par espece, implicite/IMEX
//     delegue) et appelle un TimeStepper (take_step). « Avancer un coupleur » etait bancal :
//     ici un Assembleur assemble, un Driver avance. Le Driver POSSEDE un Assembleur.
//
// SystemCoupler reste comme ALIAS de SystemDriver (compat : tests, facades adc_cases).
//
// Aucune des deux ne remplace PhysicalModel (formules locales), assemble_rhs (operateur
// spatial), ni GeometricMG (elliptique) : elles les connectent.

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

// === ASSEMBLEUR : champs (Poisson de systeme + aux) + residu de bloc. Aucun pas. ======
template <CoupledSystemLike System, class RhsAssembler,
          class Elliptic = GeometricMG>
class SystemAssembler {
  static_assert(EllipticSolver<Elliptic>,
                "le backend elliptique doit modeler EllipticSolver");

 public:
  SystemAssembler(System system, const Geometry& geom, const BoxArray& ba,
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
  MultiFab& aux() { return aux_; }
  const MultiFab& aux() const { return aux_; }
  const Geometry& geom() const { return geom_; }
  const BoxArray& ba() const { return ba_; }
  const DistributionMapping& dm() const { return dm_; }

  // Resout le RHS de systeme (Sum_s q_s n_s), Poisson, puis aux = (phi, grad phi).
  void solve_fields() {
    rhs_assembler_(system_, mg_.rhs());
    mg_.solve();
    derive_aux();
  }

  // Residu d'un bloc a un etage : R = −div F + S (+ aux re-resolu si recompute_aux). C'est
  // l'evaluateur (la fleche methode-des-lignes) que le Driver passe au TimeStepper.
  template <class Limiter, class NumericalFlux, class Block>
  void block_residual(Block& block, MultiFab& state, MultiFab& R,
                      bool recompute_aux) {
    if (recompute_aux) {
      detail::ScopedBlockState<Block> swap(block, state);
      solve_fields();
    }
    fill_ghosts(state, geom_.domain, block.bc);
    assemble_rhs<Limiter, NumericalFlux>(block.model, state, aux_, geom_, R);
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

  System system_;
  RhsAssembler rhs_assembler_;
  Geometry geom_;
  BoxArray ba_;
  DistributionMapping dm_;
  BCRec bcPhi_, aux_bc_;
  Elliptic mg_;
  MultiFab aux_;
};

// === DRIVER : avance le systeme. Possede un Assembleur, lui delegue les champs. =========
template <CoupledSystemLike System, class RhsAssembler,
          class Elliptic = GeometricMG>
class SystemDriver {
 public:
  SystemDriver(System system, const Geometry& geom, const BoxArray& ba,
               const BCRec& bcPhi, RhsAssembler rhs_assembler,
               std::function<bool(Real, Real)> active = {})
      : asm_(std::move(system), geom, ba, bcPhi, std::move(rhs_assembler),
             std::move(active)) {}

  // Acces delegues a l'assembleur (compat avec l'ancienne API SystemCoupler).
  System& system() { return asm_.system(); }
  const System& system() const { return asm_.system(); }
  MultiFab& phi() { return asm_.phi(); }
  const MultiFab& aux() const { return asm_.aux(); }
  void solve_fields() { asm_.solve_fields(); }
  SystemAssembler<System, RhsAssembler, Elliptic>& assembler() { return asm_; }

  // Avance les blocs selon leur TimePolicy. Explicites SSPRK2/3 (ou TimeStepper utilisateur)
  // executes via take_step ; implicites/IMEX delegues au callback (Newton, collisions, ...).
  // IMEX = vrai forward-backward : transport explicite par le coeur, source au callback.
  template <class ImplicitAdvance>
  void step(Real dt, ImplicitAdvance&& implicit_advance) {
    ImplicitAdvance& advance_implicit = implicit_advance;
    // macro_step_ : un bloc de cadence `stride` n'avance qu'1 macro-pas sur stride
    // (alors d'un pas effectif stride*dt). stride=1 -> chaque pas (historique).
    advance_subcycled(asm_.system(), dt, macro_step_, [&](auto& block, Real h, int s, int n) {
      using Block = std::decay_t<decltype(block)>;
      constexpr TimeTreatment treatment = block_time_treatment_v<Block>;
      if constexpr (treatment == TimeTreatment::Explicit) {
        advance_explicit_block(block, h);
      } else if constexpr (treatment == TimeTreatment::Implicit ||
                           treatment == TimeTreatment::IMEX) {
        asm_.solve_fields();
        if constexpr (treatment == TimeTreatment::IMEX) explicit_transport(block, h);
        advance_implicit(*this, block, h, s, n);
      }
    });
    ++macro_step_;
  }

  // Surcharge pratique pour un systeme entierement explicite.
  void step(Real dt) {
    step(dt, [](auto&, auto& block, Real, int, int) {
      using Block = std::decay_t<decltype(block)>;
      static_assert(detail::always_false_v<Block>,
                    "SystemDriver::step(dt) ne peut pas avancer un bloc "
                    "implicite/IMEX sans callback");
    });
  }

  // Source de couplage inter-especes (splitting forward-Euler) : rafraichit phi (aux) puis
  // laisse la source lire tous les blocs + aux. Distinct de model.source (local au bloc).
  template <class CoupledSource>
  void coupled_source_step(CoupledSource&& src, Real dt) {
    static_assert(CoupledSourceFor<std::decay_t<CoupledSource>, System>,
                  "coupled_source_step attend une CoupledSource : "
                  "apply(system, aux, dt)");
    asm_.solve_fields();
    src.apply(asm_.system(), asm_.aux(), dt);
  }

 private:
  // Avance explicite d'un bloc : DELEGUE le schema a un objet TimeStepper (SSPRK2/3 du coeur
  // ou integrateur utilisateur), en lui passant l'evaluateur de residu de l'assembleur.
  template <class Block>
  void advance_explicit_block(Block& block, Real dt) {
    using Time = TimePolicyTraits<typename Block::Time>;
    using Method = typename Time::Method;
    using Limiter = typename Block::Spatial::Limiter;
    using NumericalFlux = typename Block::Spatial::NumericalFlux;
    static_assert(Time::treatment == TimeTreatment::Explicit,
                  "advance_explicit_block attend un bloc explicite");

    auto rhs_eval = [&](MultiFab& stage, MultiFab& R) {
      asm_.template block_residual<Limiter, NumericalFlux>(block, stage, R,
                                                           /*recompute_aux=*/true);
    };
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
  // La source raide est traitee separement en implicite par le callback. aux suppose a jour.
  template <class Block>
  void explicit_transport(Block& block, Real dt) {
    using Model = typename Block::Model;
    using Limiter = typename Block::Spatial::Limiter;
    using NumericalFlux = typename Block::Spatial::NumericalFlux;
    const SourceFreeModel<Model> sf{block.model};
    MultiFab R(asm_.ba(), asm_.dm(), Model::n_vars, 0);
    fill_ghosts(block.U(), asm_.geom().domain, block.bc);
    assemble_rhs<Limiter, NumericalFlux>(sf, block.U(), asm_.aux(), asm_.geom(), R);
    saxpy(block.U(), dt, R);
  }

  SystemAssembler<System, RhsAssembler, Elliptic> asm_;
  int macro_step_ = 0;  // compteur de macro-pas (cadence stride par bloc)
};

// Compat / nommage historique : SystemCoupler == le Driver (qui avance). On garde l'alias
// SystemCoupler (tests, facade MultiSpeciesSolver) ET SystemDriver (le nom « qui avance »).
template <CoupledSystemLike System, class RhsAssembler, class Elliptic = GeometricMG>
using SystemCoupler = SystemDriver<System, RhsAssembler, Elliptic>;

}  // namespace adc
