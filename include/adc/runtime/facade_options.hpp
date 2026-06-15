#pragma once

#include <string>
#include <vector>

/// @file
/// @brief OPTIONS PODs for the public facades (System / AmrSystem), grouping the long families of
///        HOMOGENEOUS parameters that posed an ordering footgun (C++ Core Guidelines I.23).
///
/// Layer: `include/adc/runtime`.
/// Role: carry, in a single named aggregate, the settings of a Schur-condensed source stage and the
///   bytecode description of an inter-species coupled source. These families were previously flat
///   lists of parameters of the SAME type (several adjacent `std::string`, several parallel
///   `std::vector<int>`) -- silently swappable at the call site. Grouping them into a named POD makes
///   the call self-documenting (designated initializers) and removes the swap risk.
/// Contract: flat POD crossing the bindings without friction. The in-class DEFAULTS reproduce EXACTLY
///   the old defaults of the flat parameters -> no behavior change.
///
/// Invariants:
/// - each field keeps the name, type and default of the old flat parameter of the same name;
/// - these PODs live ABOVE the ABI layer (compiled_block_abi.hpp / native_loader.hpp): they never
///   cross the extern "C" boundary of a .so loader. The SEMANTIC extern "C" ABI (residual / advance,
///   structs crossing the loader) therefore stays UNCHANGED. On the other hand the abi_key() LITERAL
///   CHANGES: it embeds the token headers=ADC_HEADER_SIG (conservative sha256 of the path and content
///   of EVERY header under include/, cf. abi_key.hpp and python/CMakeLists.txt); merely ADDING this
///   header and EDITING system.hpp / amr_system.hpp shifts ADC_HEADER_SIG. This is EXPECTED and
///   harmless: no semantic ABI changes, but add_native_block will reject the AOT .so generated before
///   this change (divergent signature) -> a one-time regeneration of the stale .so.

namespace adc {

/// @brief Settings of the Schur-condensed SOURCE STAGE (cf. System::set_source_stage /
///        AmrSystem::set_source_stage). Groups the Krylov solve and the field DESCRIPTORS --
///        a family of four adjacent `std::string` that were swappable at the call site.
///
/// Usage: built by the facade (or by the bindings, from the flat Python kwargs) then passed to
///   set_source_stage. All defaults (empty POD) = bit-identical historical behavior.
/// Contract: the DEFAULTS reproduce the old defaults of the flat parameters.
///  - krylov_tol / krylov_max_iters: tolerance and budget of the stage Krylov solve (BiCGStab).
///    <= 0 (default) = historical stepper constants (1e-10; 400 cartesian, 600 polar).
///  - density / momentum_x / momentum_y / energy: DESCRIPTORS of the stage fields. EMPTY string
///    (default) = canonical role (Density / MomentumX / MomentumY / optional Energy), bit-identical.
///    Otherwise: a stable ROLE NAME ("density", "momentum_x", ...) or a VARIABLE NAME of the block.
///    energy == "none" disables the energy update.
///  - bz_aux_component: aux-channel component read as the magnetic field Omega. < 0 (default) =
///    canonical B_z channel (kAuxBaseComps), bit-identical. (Ignored by AmrSystem: mono-block stage
///    on the canonical channel.)
struct SourceStageOptions {
  double krylov_tol = 0.0;
  int krylov_max_iters = 0;
  std::string density = "";
  std::string momentum_x = "";
  std::string momentum_y = "";
  std::string energy = "";
  int bz_aux_component = -1;
};

/// @brief BYTECODE description of a generic inter-species COUPLED SOURCE (cf.
///        System::add_coupled_source / AmrSystem::add_coupled_source). Groups the FLAT arrays
///        of the bytecode ABI -- six `std::vector` (four of block/role descriptors, two+ of stack
///        machine program) swappable at the call site -- into a single named aggregate.
///
/// Usage: built by the facade (or by the bindings, from the flat Python kwargs) then passed to
///   add_coupled_source with the frequency and the label kept flat (a double and a string, distinct
///   types, outside the homogeneous footgun). A malformed shape raises an EXPLICIT error on add.
/// Contract: FLAT ABI -- no C++ object crosses the boundary; this POD is only a facade-side carrier of
///   arrays. The DEFAULTS reproduce the old defaults of the flat parameters (the EMPTY per-cell
///   frequency programs = constant frequency alone, bit-identical).
///  - in_blocks / in_roles: blocks read as input and their roles (one per input register).
///  - consts: constants (.param()), loaded after the inputs.
///  - out_blocks / out_roles: target block and target role of each source term.
///  - prog_ops / prog_args: concatenated opcodes of ALL terms (stack machine) and their parallel
///    arguments (register index for PushReg).
///  - prog_lens: program length of each term (segments prog_ops / prog_args in order).
///  - freq_prog_ops / freq_prog_args: OPTIONAL program of a PER-CELL frequency mu(U) (same stack
///    machine, SAME register table). EMPTY (default) = constant frequency alone.
struct CoupledSourceProgram {
  std::vector<std::string> in_blocks;
  std::vector<std::string> in_roles;
  std::vector<double> consts;
  std::vector<std::string> out_blocks;
  std::vector<std::string> out_roles;
  std::vector<int> prog_ops;
  std::vector<int> prog_args;
  std::vector<int> prog_lens;
  std::vector<int> freq_prog_ops;
  std::vector<int> freq_prog_args;
};

}  // namespace adc
