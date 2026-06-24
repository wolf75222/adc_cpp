#pragma once

#include <adc/core/state/state.hpp>  // kAuxBaseComps (default aux channel of the Schur stage: B_z)
#include <adc/core/foundation/types.hpp>  // Real
#include <adc/core/state/variables.hpp>   // VariableSet (role descriptor carried by each block)
#include <adc/mesh/index/box2d.hpp>       // Box2D
#include <adc/mesh/execution/for_each.hpp>  // device_fence (marshaling synchronizes the device before reading the host)
#include <adc/mesh/storage/multifab.hpp>  // MultiFab, Array4, ConstArray4

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

/// @file
/// @brief SystemBlockStore: the BLOCK MANAGEMENT responsibility extracted from the god-class System::Impl
///        (audit Lot B.3, last P0 extraction; follows SystemFieldSolver #176 and SystemStepper).
///        Extracted VERBATIM from python/system.cpp: no change to numerics, layout, iteration order,
///        indexing, or error message. STRICTLY bit-identical -- the code is moved as is.
///
/// CONTRACT / INVARIANTS
/// - OWNS: the BlockState struct (formerly Species: state U + descriptors + closures of the block) and the
///   ordered registry of blocks (vector<BlockState>), the UNIQUE source of truth populated by all
///   add paths (add_block / install_block; add_dynamic_block / add_compiled_block / add_native_block
///   via native_loader).
/// - EXPOSES: index(name) (0-based index or error), find(name) (const + non const, reference to the block or
///   error), and the state MARSHALING helpers copy_comp0 / copy_state / write_state (host <-> MultiFab copy,
///   device_fence included). The insertion ORDER fixes indexing and thus iteration: it is
///   PRESERVED (push_back at the tail, never sorted nor reshuffled).
/// - ERROR MESSAGES UNCHANGED: "System: bloc inconnu '...'" (index/find) and
///   "System::set_state: taille != ncomp*n*n" (write_state) are kept VERBATIM (the non-regression of the
///   facade tests depends on it).
/// - DOES NOT OWN: the domain (ba/dm/dom/geom/pgeom_), the aux and its width, the Poisson/elliptic, the
///   couplings, t/macro_step_. These concerns stay in System::Impl (or in SystemFieldSolver /
///   SystemStepper); the store knows nothing about them.
///
/// The `blocks` registry is PUBLIC: System::Impl exposes it as is via a reference member `sp` (alias),
/// so that the already-extracted header templates (SystemFieldSolver, SystemStepper, native_loader) that
/// iterate `owner_->sp` / `P->sp` and name `Impl::Species` stay UNCHANGED and bit-identical. The
/// struct is named BlockState (clearer meaning than the historical "Species"); Impl keeps the alias
/// `using Species = SystemBlockStore::BlockState;` for template compatibility.

namespace adc {

/// Forward-declaration: BlockState carries a shared_ptr<CondensedSchurSourceStepper> (condensed source
/// stage, OPT-IN). The pointer alone is enough here; the definition lives in the coupling header, included
/// where the stage is actually built (python/system.cpp::set_source_stage).
class CondensedSchurSourceStepper;
/// Forward-declaration: POLAR counterpart of the condensed source stage (ring (r, theta)). A BlockState
/// carries AT MOST ONE of the two steppers (Cartesian OR polar), depending on the System geometry, chosen by
/// set_source_stage. The pointer alone is enough here (cf. PolarCondensedSchurSourceStepper, Voie A step 2c).
class PolarCondensedSchurSourceStepper;

/// ORDERED registry of the System blocks + state marshaling helpers. See contract above (OWNS
/// BlockState + the vector; EXPOSES index/find + copy/write_state; DOES NOT OWN the domain/aux/Poisson).
class SystemBlockStore {
 public:
  /// Type-erase of the POINTWISE (one cell) cons <-> prim conversion of a block: in/out are
  /// arrays of ncomp doubles. SAME type as System::CellConvert (identical std::function): assignment
  /// from set_block_conversion / native_loader stays a trivial move.
  using CellConvert = std::function<void(const double* in, double* out)>;

  /// Compiled closures frozen at block add time (composite model + spatial scheme + time).
  /// Type-erased ONLY at the block list level; the kernel stays compiled.
  /// MEMBER ORDER FROZEN: install_block (python/system.cpp) and native_loader (push_dynamic /
  /// add_compiled_model) initialize this struct by positional AGGREGATE
  /// {name, U, ncomp, substeps, evolve, stride, gamma, advance, rhs_into, max_speed, add_poisson_rhs};
  /// do not reorder these members nor insert any before add_poisson_rhs.
  struct BlockState {
    std::string name;
    MultiFab U;
    int ncomp;
    int substeps;    // static substeps (add_block)
    bool evolve;     // false = frozen species (fixed background, not advanced)
    int stride = 1;  // cadence: advance once every stride macro-steps
    double gamma;    // for the rest energy (4 var)
    std::function<void(MultiFab&, Real, int)> advance;   // (U, dt, n): n substeps of dt/n
    std::function<void(MultiFab&, MultiFab&)> rhs_into;  // R <- -div F + S (Poisson frozen)
    std::function<Real(const MultiFab&)> max_speed;      // max |wave speed| of the block
    std::function<void(const MultiFab&, MultiFab&)> add_poisson_rhs;  // += elliptic_rhs(U)
    // Descriptor of the conservative / primitive variables (names + physical ROLES) of the block.
    // The roles (provided by M::conservative_vars()) let inter-species couplings target a component
    // by its MEANING (momentum, energy) instead of a hard-coded index u[1]/u[3].
    VariableSet cons_vars, prim_vars;
    // POINTWISE cons <-> prim conversions OF THE BLOCK MODEL (one cell, ncomp doubles in/out).
    // Set at add time (install_block / push_dynamic) from the real model; empty -> identity (the
    // model exposes no conversion, e.g. pure scalar or .so generated before this work).
    // Consumed by set_primitive_state / get_primitive_state (init/diagnostic in primitive).
    CellConvert prim_to_cons, cons_to_prim;
    // Schur-CONDENSED SOURCE STAGE (OPT-IN, adc.Split(source=CondensedSchur), cf. set_source_stage).
    // nullptr (default) = no condensed source stage: the block advances EXACTLY as before
    // (bit-identical). Non null = after the hyperbolic transport, the step runs the standalone source stage
    // (CondensedSchurSourceStepper, #126) in place of the explicit / IMEX source. shared_ptr:
    // keeps BlockState MOVABLE (the stepper carries a GeometricMG, neither copyable nor trivially movable).
    std::shared_ptr<CondensedSchurSourceStepper> schur;
    // POLAR counterpart of the condensed source stage (ring (r, theta), Voie A step 2c). Exclusive with
    // schur above: set_source_stage builds ONE or the OTHER depending on the System geometry (polar
    // -> schur_polar, Cartesian -> schur). run_source_stage dispatches on whichever is non null. The
    // Cartesian path stays BIT-IDENTICAL (schur_polar == nullptr when the System is Cartesian).
    std::shared_ptr<PolarCondensedSchurSourceStepper> schur_polar;
    double schur_theta = 0.5;  // theta-scheme of the source stage (0.5 = Crank-Nicolson)
    // Component of the aux channel read as magnetic field Omega = B_z by the Schur stage (audit
    // wave 2: field TRANSPORTED in the ABI instead of the frozen literal kAuxBaseComps). Default =
    // kAuxBaseComps (canonical B_z channel), bit-identical; set_source_stage may redirect it.
    int schur_bz_comp = kAuxBaseComps;
    // GENERIC SOURCE STAGE (optional): a callable (U, dt) that advances IN PLACE the source stage of the
    // block. nullptr (default) = no generic source stage (BIT-IDENTICAL path). run_source_stage runs it
    // ONLY if no condensed Schur stage (schur / schur_polar) is wired, so it changes
    // NOTHING in the production Schur path. Used for generic splitting (adc.Strang on an arbitrary
    // source stage) and for the stepper time-order tests (non-commuting toy operators). Trailing
    // + nullptr default: the positional aggregate init of the other members stays unchanged.
    std::function<void(MultiFab&, Real)> source_step;
    // DISK TRANSPORT ADVANCES (work T5-PR3, OPT-IN). Empty (default) -> no disk routing:
    // the stepper advances via `advance` (full Cartesian path, BIT-IDENTICAL). Non empty, they MIMIC
    // `advance` (same RK / IMEX scheme, same limiter / flux) but dispatch the transport residual
    // to the disk operator, and are SELECTED only if the System is in Staircase mode (resp.
    // CutCell) AND a disk is set (set_disc_domain). Built at block add time (build_block)
    // AT THE SAME TIME as `advance`, they read the System mask / level set by pointer at step time
    // (stable address): the order add_block / set_disc_domain is irrelevant. Trailing + empty default:
    // the positional aggregate init of the other members stays unchanged.
    std::function<void(MultiFab&, Real, int)>
        advance_masked;  // residual via assemble_rhs_masked (Staircase)
    std::function<void(MultiFab&, Real, int)> advance_eb;  // residual via assemble_rhs_eb (CutCell)
    // dt_hotspot DIAGNOSTIC (ADC-182): (U, w, i, j) -> GLOBAL cell dominating the transport CFL bound
    // of the block + its speed w = max(wx, wy). ON DEMAND only (System::dt_hotspot):
    // never queried by step/step_cfl (hot path bit-identical). Trailing + empty default.
    std::function<void(const MultiFab&, Real&, int&, int&)> hotspot;
    // OPTIONAL STEP BOUNDS of the block (audit 2026-06, step_cfl work). EMPTY (default) -> the
    // stepper does not query them: STRICTLY historical step policy (transport only,
    // bit-identical). Set by set_block_dt_bounds when the model declares the traits
    // HasSourceFrequency / HasStabilityDt (cf. core/physical_model.hpp for the semantics).
    // Trailing + empty default: the positional aggregate init of the other members stays unchanged.
    std::function<Real(const MultiFab&)>
        source_frequency;  // max over cells of mu [1/s] (0 = no constraint)
    std::function<Real(const MultiFab&)>
        stability_dt;  // min over cells of the admissible step (0 = no constraint)
    // PROJECTION PONCTUELLE post-pas (ADC-177) : U <- project(U, aux) sur les cellules VALIDES,
    // appliquee par le stepper a la FIN de chaque macro-pas ENTIER (apres transport + etage source +
    // couplages ; jamais par etage RK). VIDE (defaut) -> jamais interrogee (cout nul, chemin
    // bit-identique). Trailing + defaut vide : l'init par agregat positionnel reste inchangee.
    std::function<void(MultiFab&)> project;
    // FLUX-ONLY residual R <- -div F(U) (NO default/composite source), Poisson frozen (ADC-425). The
    // SAME transport assembly as rhs_into evaluated on SourceFreeModel<Model> (zero source), so the
    // flux / ghost / geometry handling is bit-identical -- only the source is dropped (with
    // limiter='none'; the HLL wave-speed cache -- rejected on the aot/production backends compiled
    // Programs use -- is the only path where cached cell-center speeds differ from the per-face
    // reconstruction). Read by
    // System::block_neg_div_flux_into, which a compiled time Program's hyperbolic stage calls so a
    // Lie/Strang split assembles "flux but no source" (spec criterion 17). EMPTY (default) for paths
    // that do not build it (the host .so prototype loader); block_neg_div_flux_into fails loud then.
    // Trailing + empty default: the positional aggregate init of the other members stays unchanged.
    std::function<void(MultiFab&, MultiFab&)> rhs_flux_only;
    // NAMED elliptic-field RHS closures (ADC-428): field name -> (+= elliptic_field_rhs(U)). A model
    // declaring m.elliptic_field("phi2", rhs=...) carries here a SECOND Poisson right-hand side
    // (distinct from add_poisson_rhs, the default Poisson coupling), assembled the same way (host loop,
    // += per cell). EMPTY (default) -> no named elliptic field: bit-identical to the historical block.
    // The SystemFieldSolver gathers these per field (sum over blocks) into a SEPARATE elliptic solve
    // whose phi/grad are written to the field's OWN aux channel. Trailing + empty default: the
    // positional aggregate init of the other members stays unchanged.
    std::map<std::string, std::function<void(const MultiFab&, MultiFab&)>> named_poisson_rhs;
  };

  /// ORDERED registry of the blocks (UNIQUE source of truth). PUBLIC: Impl aliases it as `sp` for the
  /// already-extracted templates (SystemFieldSolver / SystemStepper / native_loader) that iterate owner_->sp.
  std::vector<BlockState> blocks;

  // --- access by NAME (0-based indexing, insertion order) ------------------------------------------
  /// Reference to block @p name (for writing). @throws std::runtime_error "System: bloc inconnu '...'".
  BlockState& find(const std::string& name) {
    for (auto& s : blocks)
      if (s.name == name)
        return s;
    throw std::runtime_error("System : bloc inconnu '" + name + "'");
  }
  /// Reference to block @p name (for reading). @throws std::runtime_error "System: bloc inconnu '...'".
  const BlockState& find(const std::string& name) const {
    for (auto& s : blocks)
      if (s.name == name)
        return s;
    throw std::runtime_error("System : bloc inconnu '" + name + "'");
  }
  /// 0-based index of block @p name (insertion order). @throws std::runtime_error if unknown.
  int index(const std::string& name) const {
    for (std::size_t k = 0; k < blocks.size(); ++k)
      if (blocks[k].name == name)
        return static_cast<int>(k);
    throw std::runtime_error("System : bloc inconnu '" + name + "'");
  }

  /// Number of registered blocks.
  int size() const { return static_cast<int>(blocks.size()); }
  /// Block names, in insertion order (UNIQUE source: all add paths appear here).
  std::vector<std::string> names() const {
    // Reads the UNIQUE block registry, populated by all add paths: a block loaded via
    // add_dynamic_block / add_compiled_block (.so) appears just like an add_block.
    std::vector<std::string> out;
    out.reserve(blocks.size());
    for (const auto& s : blocks)
      out.push_back(s.name);
    return out;
  }

  // --- state marshaling (host <-> MultiFab copy; device_fence included) ---------------------------
  /// Copies component 0 of fab(0) (density) in row-major (j slow, i fast). device_fence beforehand:
  /// the marshaling reads the host, so the device must have finished writing U.
  std::vector<double> copy_comp0(const MultiFab& mf) const {
    device_fence();
    // MPI single-box: the box lives on the owner rank (rank 0). A rank without a box (local_size()==0)
    // has NO fab(0) -> return EMPTY rather than an OUT-OF-BOUNDS access (UB). Single-rank: local_size()
    // is always 1, behavior UNCHANGED. For the global multi-rank field: System::density_global.
    if (mf.local_size() == 0)
      return {};
    const ConstArray4 u = mf.fab(0).const_array();
    const Box2D v = mf.box(0);
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        out.push_back(u(i, j, 0));
    return out;
  }
  /// Copies the ncomp components of fab(0) in component-major layout (c slow, then j, then i).
  std::vector<double> copy_state(const MultiFab& mf, int ncomp) const {
    device_fence();
    // Rank without a box (MPI single-box, non-owner): return EMPTY (no fab(0)). Cf. copy_comp0;
    // the global multi-rank field goes through System::state_global (collective gather).
    if (mf.local_size() == 0)
      return {};
    const ConstArray4 u = mf.fab(0).const_array();
    const Box2D v = mf.box(0);
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(ncomp) * v.nx() * v.ny());
    for (int c = 0; c < ncomp; ++c)
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          out.push_back(u(i, j, c));
    return out;
  }
  /// Writes the ncomp components into fab(0) from a component-major buffer (same layout as
  /// copy_state). @throws std::runtime_error "System::set_state: taille != ncomp*n*n" if the size
  /// does not match ncomp*nx*ny (message unchanged).
  void write_state(MultiFab& mf, int ncomp, const std::vector<double>& in) {
    // Rank without a box (MPI single-box, non-owner): NO-OP (no fab(0) to write). Lets
    // sim.set_state / sim.restart be called on ALL ranks with the GLOBAL field: only the
    // owner rank (rank 0, box = full domain) writes, the others do nothing. Single-rank:
    // local_size()==1, validation + write UNCHANGED (bit-identical).
    if (mf.local_size() == 0)
      return;
    const Box2D v = mf.box(0);
    const std::size_t need = static_cast<std::size_t>(ncomp) * v.nx() * v.ny();
    if (in.size() != need)
      throw std::runtime_error("System::set_state : taille != ncomp*n*n");
    Array4 u = mf.fab(0).array();
    std::size_t k = 0;
    for (int c = 0; c < ncomp; ++c)
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          u(i, j, c) = in[k++];
  }
};

}  // namespace adc
