#include "../bindings_detail.hpp"

// ADC-365: the System runtime-composition facade bindings.
void init_system(py::module_& m) {
  py::class_<System>(m, "System")
      .def(py::init<const SystemConfig&>())
      // Per-block composition: model (bricks) + spatial scheme (limiter/riemann) + time
      // (explicit/imex) + substeps. Python says WHAT, the compiled C++ does the compute.
      // ADC-214: the Python SURFACE is UNCHANGED (same flat newton_* kwargs, same defaults). The
      // lambda receives them flat and BUILDS the NewtonOptions POD internally before calling the
      // new C++ method (which groups these homogeneous parameters). adc_cases sees no change.
      .def(
          "add_block",
          [](System& s, const std::string& name, const ModelSpec& model, const std::string& limiter,
             const std::string& riemann, const std::string& recon, const std::string& time,
             int substeps, bool evolve, int stride, const std::vector<std::string>& implicit_vars,
             const std::vector<std::string>& implicit_roles, int newton_max_iters,
             double newton_rel_tol, double newton_abs_tol, double newton_fd_eps,
             bool newton_diagnostics, double newton_damping, const std::string& newton_fail_policy,
             double positivity_floor, bool wave_speed_cache) {
            NewtonOptions newton;
            newton.max_iters = newton_max_iters;
            newton.rel_tol = static_cast<Real>(newton_rel_tol);
            newton.abs_tol = static_cast<Real>(newton_abs_tol);
            newton.fd_eps = static_cast<Real>(newton_fd_eps);
            newton.damping = static_cast<Real>(newton_damping);
            newton.fail_policy =
                newton_fail_policy_from_string(newton_fail_policy, "System::add_block");
            s.add_block(name, model, limiter, riemann, recon, time, substeps, evolve, stride,
                        implicit_vars, implicit_roles, newton, newton_diagnostics, positivity_floor,
                        wave_speed_cache);
          },
          py::arg("name"), py::arg("model"), py::arg("limiter") = "minmod",
          py::arg("riemann") = "rusanov", py::arg("recon") = "conservative",
          py::arg("time") = "explicit", py::arg("substeps") = 1, py::arg("evolve") = true,
          py::arg("stride") = 1,
          // Implicit mask CARRIED BY THE BLOCK (IMEX): conserved variables treated implicitly by
          // NAME (implicit_vars) or by physical ROLE (implicit_roles). Empty (default) -> model default,
          // bit-identical. Resolved on the C++ side against the block's names/roles (error on a missing name/role).
          py::arg("implicit_vars") = std::vector<std::string>{},
          py::arg("implicit_roles") = std::vector<std::string>{},
          // Options of the implicit IMEX source Newton (defaults = historical constants 2 / 1e-7,
          // bit-identical). newton_diagnostics=True enables the report (newton_report(name)).
          py::arg("newton_max_iters") = 2, py::arg("newton_rel_tol") = 0.0,
          py::arg("newton_abs_tol") = 0.0, py::arg("newton_fd_eps") = 1e-7,
          py::arg("newton_diagnostics") = false, py::arg("newton_damping") = 1.0,
          py::arg("newton_fail_policy") = "none",
          // Zhang-Shu POSITIVITY limiter (ADC-76): density floor of the reconstructed face states
          // (conservative scaling toward the cell mean). 0 (default) = inactive,
          // bit-identical path. Requires a model exposing the Density role.
          py::arg("positivity_floor") = 0.0,
          // HLL wave speed cache (opt-in): evaluates model.wave_speeds once per cell instead of per
          // face. riemann='hll' + explicit only (explicit error otherwise). NoSlope + conservative
          // recon -> bit-identical to the per-face path. False (default) = path unchanged.
          py::arg("wave_speed_cache") = false)
      // Newton report (IMEX diagnostics OPT-IN): dict {enabled, converged, max_residual,
      // max_iters_used, n_failed, failed_cell, failed_component}, aggregated over the substeps of the
      // LAST advance of the block. failed_cell = (i, j) of ONE faulty cell or None.
      .def(
          "newton_report",
          [](const System& s, const std::string& name) {
            const System::SourceNewtonReport r = s.newton_report(name);
            py::dict d;
            d["enabled"] = r.enabled;
            d["converged"] = r.converged;
            d["max_residual"] = r.max_residual;
            d["max_iters_used"] = r.max_iters_used;
            d["n_failed"] = r.n_failed;
            if (r.failed_i >= 0)
              d["failed_cell"] =
                  py::make_tuple(static_cast<int>(r.failed_i), static_cast<int>(r.failed_j));
            else
              d["failed_cell"] = py::none();
            d["failed_component"] = static_cast<int>(r.failed_comp);
            return d;
          },
          py::arg("name"))
      // Block whose model is loaded at runtime from a .so generated by the DSL (host path).
      .def("add_dynamic_block", &System::add_dynamic_block, py::arg("name"), py::arg("so_path"),
           py::arg("substeps") = 1, py::arg("names") = std::vector<std::string>{},
           py::arg("recon") = "none")
      .def("add_compiled_block", &System::add_compiled_block, py::arg("name"), py::arg("so_path"),
           py::arg("limiter") = "minmod", py::arg("riemann") = "rusanov",
           py::arg("recon") = "conservative", py::arg("time") = "explicit", py::arg("substeps") = 1,
           py::arg("names") = std::vector<std::string>{}, py::arg("positivity_floor") = 0.0)
      // P7-b: changes the RUNTIME parameters of an AOT block WITHOUT recompiling the .so. values =
      // whole block (sorted name order on the DSL side). cf. System::set_block_params.
      .def("set_block_params", &System::set_block_params, py::arg("name"), py::arg("values"))
      // NATIVE block loaded from a .so loader generated by the DSL (backend "production",
      // dsl.compile_native): the .so inlines add_compiled_model<ProdModel> -> zero-copy block on the
      // real System context, ABI key verified. cf. System::add_native_block.
      .def("add_native_block", &System::add_native_block, py::arg("name"), py::arg("so_path"),
           py::arg("limiter") = "minmod", py::arg("riemann") = "rusanov",
           py::arg("recon") = "conservative", py::arg("time") = "explicit", py::arg("gamma") = 1.4,
           py::arg("substeps") = 1, py::arg("evolve") = true, py::arg("stride") = 1,
           py::arg("positivity_floor") = 0.0)
      // Compiled time Program (epic ADC-399 / ADC-401): dlopen a generated problem.so, verify its
      // ABI key against this module (fail-loud -> RuntimeError), and install its macro-step body. The
      // block(s) must already exist (add_equation); the Program drives sim.step(dt) via ProgramContext.
      .def("install_program", &System::install_program, py::arg("so_path"))
      // Compiled-Program macro-step cadence (ADC-411): SYSTEM-level substeps + stride around the
      // installed program closure (cf. SystemStepper::step). Separate from install_program so the .so
      // ABI is untouched; CompiledTime(substeps=, stride=) threads through here. Both must be >= 1.
      .def("set_program_cadence", &System::set_program_cadence, py::arg("substeps"),
           py::arg("stride"))
      // ADC-406b: IR hash of the installed compiled Program (the .so's adc_program_hash), or "" if
      // none. sim.checkpoint records it; sim.restart rejects a restart against a DIFFERENT Program.
      .def("installed_program_hash", &System::installed_program_hash)
      // ADC-414 (spec op 23): scalar diagnostics a compiled Program records via P.record_scalar,
      // retrievable AFTER sim.step. program_diagnostic(name) reads one (raises if never recorded);
      // program_diagnostics() returns the whole name -> value dict.
      .def("program_diagnostic", &System::program_diagnostic, py::arg("name"))
      .def("program_diagnostics", &System::program_diagnostics)
      // ADC-435 (spec ctx.param): RUNTIME parameters a compiled Program reads via ctx.param. set_param
      // changes the value WITHOUT recompiling the .so (only the NAME is in the program source / cache
      // key); param(name) reads one (raises if never set); params() returns the whole name -> value dict.
      .def("set_param", &System::set_param, py::arg("name"), py::arg("value"))
      .def("param", &System::param, py::arg("name"))
      .def("params", &System::params)
      // Multistep history checkpoint/restart seam (ADC-406b): the facade gathers/restores the
      // System-owned rings DIRECTLY (no .so checkpoint_extra ABI). history_global mirrors state_global
      // (collective gather, component-major); restore_history mirrors set_state (owner-rank scatter).
      .def("history_names", &System::history_names)
      .def("history_depth", &System::history_depth, py::arg("name"))
      .def("history_ncomp", &System::history_ncomp, py::arg("name"))
      .def(
          "history_global",
          [](const System& s, const std::string& name, int slot) {
            return to_3d(s.history_global(name, slot), s.history_ncomp(name), s.ny(), s.nx());
          },
          py::arg("name"), py::arg("slot"))
      .def("history_initialized", &System::history_initialized, py::arg("name"))
      .def(
          "restore_history",
          [](System& s, const std::string& name, int slot,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.restore_history(name, slot, flat(arr));
          },
          py::arg("name"), py::arg("slot"), py::arg("values"))
      .def("set_history_initialized", &System::set_history_initialized, py::arg("name"),
           py::arg("initialized"))
      .def("add_ionization", &System::add_ionization, py::arg("electron"), py::arg("ion"),
           py::arg("neutral"), py::arg("rate"))
      .def("add_collision", &System::add_collision, py::arg("a"), py::arg("b"), py::arg("rate"))
      .def("add_thermal_exchange", &System::add_thermal_exchange, py::arg("a"), py::arg("b"),
           py::arg("rate"))
      // Schur-condensed source stage (OPT-IN, adc.Split(source=adc.CondensedSchur(...))): replaces
      // the block's explicit / IMEX source with the C++ condensed stage (CondensedSchurSourceStepper, #126)
      // after the hyperbolic transport. kind='electrostatic_lorentz'. Default (without the call) unchanged.
      // ADC-214: Python surface UNCHANGED (same flat krylov_* kwargs / descriptors, same
      // defaults). The lambda receives them flat and BUILDS the SourceStageOptions POD before the C++ call.
      .def(
          "set_source_stage",
          [](System& s, const std::string& name, const std::string& kind, double theta,
             double alpha, double krylov_tol, int krylov_max_iters, const std::string& density,
             const std::string& momentum_x, const std::string& momentum_y,
             const std::string& energy, int bz_aux_component) {
            SourceStageOptions opts;
            opts.krylov_tol = krylov_tol;
            opts.krylov_max_iters = krylov_max_iters;
            opts.density = density;
            opts.momentum_x = momentum_x;
            opts.momentum_y = momentum_y;
            opts.energy = energy;
            opts.bz_aux_component = bz_aux_component;
            s.set_source_stage(name, kind, theta, alpha, opts);
          },
          py::arg("name"), py::arg("kind"), py::arg("theta"), py::arg("alpha"),
          // Tolerance / budget of the stage's Krylov solve (audit 2026-06): <= 0 = historical
          // stepper defaults (1e-10; 400 Cartesian / 600 polar).
          py::arg("krylov_tol") = 0.0, py::arg("krylov_max_iters") = 0,
          // Field descriptors (wave 2 audit: roles carried in the ABI): "" = canonical
          // role (bit-identical); otherwise a stable role name or a block variable name.
          // bz_aux_component < 0 = canonical B_z channel. Honored in Cartesian as in polar.
          py::arg("density") = "", py::arg("momentum_x") = "", py::arg("momentum_y") = "",
          py::arg("energy") = "", py::arg("bz_aux_component") = -1)
      // Time splitting policy: "lie" (default, bit-identical) or "strang" (H(dt/2) S(dt)
      // H(dt/2), 2nd order). Cf. System::set_time_scheme / SystemStepper::step_strang.
      .def("set_time_scheme", &System::set_time_scheme, py::arg("scheme"))
      // (System) -- see also AmrSystem.add_coupled_source below for the AMR counterpart.
      // GLOBAL time-step bound (step_cfl audit): fn() evaluated ONCE per step (host) by
      // step_cfl / step_adaptive; dt <= fn() when fn() > 0 and finite. Hook for non
      // cell-local constraints (coupling, Schur/Poisson, scheduler, user ramp). A Python
      // callback is acceptable here (never per cell).
      .def("add_dt_bound", &System::add_dt_bound, py::arg("label"), py::arg("fn"))
      // ACTIVE bound of the last step_cfl: "transport:<block>" | "source_frequency:<block>" |
      // "stability_dt:<block>" | "global:<label>" | "degenerate" | "" (no CFL step yet).
      .def("last_dt_bound", &System::last_dt_bound)
      // Clock (IO v1): macro_step exposed + restoration (t, macro_step) for the restart -- the
      // stride cadence depends on macro_step % stride, not only on t.
      .def("macro_step", &System::macro_step)
      .def("set_clock", &System::set_clock, py::arg("t"), py::arg("macro_step"))
      .def("set_potential", &System::set_potential, py::arg("phi"))
      // Gauss law policy (R0, Hoffart repro): "restart" (default, re-solves Poisson every
      // step, bit-identical) or "evolve" (after phi^0, no more re-solve; the Schur stage evolves phi
      // without restart, like the paper). Cf. System::set_gauss_policy.
      .def("set_gauss_policy", &System::set_gauss_policy, py::arg("policy"))
      // Generic COUPLED source (adc.dsl.CoupledSource, P5): flat ABI (postfix bytecode). Reads
      // fields (block, role) and writes source terms compiled into a stack machine, applied by
      // explicit splitting after the transport (same seam as add_ionization). Without the call, unchanged.
      // ADC-214: Python surface UNCHANGED (same flat kwargs in_blocks/.../freq_prog_args, same
      // defaults). The lambda assembles the CoupledSourceProgram POD before the C++ call (frequency / label
      // stay flat, distinct types outside the homogeneous family).
      .def(
          "add_coupled_source",
          [](System& s, const std::vector<std::string>& in_blocks,
             const std::vector<std::string>& in_roles, const std::vector<double>& consts,
             const std::vector<std::string>& out_blocks, const std::vector<std::string>& out_roles,
             const std::vector<int>& prog_ops, const std::vector<int>& prog_args,
             const std::vector<int>& prog_lens, double frequency, const std::string& label,
             const std::vector<int>& freq_prog_ops, const std::vector<int>& freq_prog_args) {
            CoupledSourceProgram prog{in_blocks,     in_roles,      consts,    out_blocks,
                                      out_roles,     prog_ops,      prog_args, prog_lens,
                                      freq_prog_ops, freq_prog_args};
            s.add_coupled_source(prog, frequency, label);
          },
          py::arg("in_blocks"), py::arg("in_roles"), py::arg("consts"), py::arg("out_blocks"),
          py::arg("out_roles"), py::arg("prog_ops"), py::arg("prog_args"), py::arg("prog_lens"),
          // CONSTANT declared frequency mu of the coupling (CoupledSource.frequency, wave 3): step
          // bound dt <= cfl/mu on the macro-step; <= 0 = no bound (historical).
          py::arg("frequency") = 0.0, py::arg("label") = "coupled_source",
          // Optional PER-CELL frequency mu(U): bytecode program (same stack machine / register
          // table as the terms). EMPTY (default) = constant frequency only, bit-identical.
          py::arg("freq_prog_ops") = std::vector<int>{},
          py::arg("freq_prog_args") = std::vector<int>{})
      .def("variable_names", &System::variable_names,
           "Variable names of a block (introspection). kind = 'conservative' | 'primitive'.",
           py::arg("name"), py::arg("kind") = "conservative")
      .def("variable_roles", &System::variable_roles,
           "PHYSICAL roles of a block's variables, parallel to variable_names: 'density', "
           "'momentum_x', 'energy', ... or 'custom' if the block does not declare its roles. This "
           "is what "
           "the inter-species couplings resolve (index_of(role)). kind = 'conservative' | "
           "'primitive'.",
           py::arg("name"), py::arg("kind") = "conservative")
      .def("block_gamma", &System::block_gamma, py::arg("name"))
      .def("set_poisson", &System::set_poisson,
           "Configures the shared system Poisson. rhs: 'charge_density' | 'composite' (labels "
           "of the SAME right-hand side f = sum of the elliptic bricks per block; charge_density = "
           "historical "
           "alias). solver: 'geometric_mg' (any case, wall included) | 'fft' (periodic, "
           "discrete stencil; n = 2^k for the fast FFT, otherwise direct DFT O(n^2)) | "
           "'fft_spectral' "
           "(periodic, continuous symbol -(kx^2+ky^2)). bc: 'auto' | 'periodic' | 'dirichlet' | "
           "'neumann'. wall: 'none' | "
           "'circle' (conducting wall centered at (L/2, L/2), radius wall_radius). epsilon: "
           "CONSTANT permittivity of div(eps grad phi) = f (for variable eps(x): "
           "set_epsilon_field). "
           "abs_tol: absolute floor of the GeometricMG V-cycle stopping criterion (0 = relative "
           "criterion, "
           "historical; no effect on FFT).",
           py::arg("rhs") = "charge_density", py::arg("solver") = "geometric_mg",
           py::arg("bc") = "auto", py::arg("wall") = "none", py::arg("wall_radius") = 0.0,
           py::arg("epsilon") = 1.0, py::arg("abs_tol") = 0.0)
      // DISC transport domain (T2 / T5-PR3 work): materializes a cell-centered 0/1 mask (cell
      // active if its center is in hypot(x-cx, y-cy) - R < 0) and WIRES the transport according to
      // mode=: 'none' (default, full Cartesian transport, bit-identical even with the disc set),
      // 'staircase' (assemble_rhs_masked, 0/1 face gate), 'cutcell' (assemble_rhs_eb, cut-cell EB,
      // apertures + kappa). Honored under Lie AND Strang (set_time_scheme). cf. System::set_disc_domain.
      .def("set_disc_domain", &System::set_disc_domain, py::arg("cx"), py::arg("cy"), py::arg("R"),
           py::arg("mode") = "none")
      // Toggles ONLY the disc transport mode ('none'|'staircase'|'cutcell') without (re)defining the
      // disc. A mode != 'none' requires a disc already set (set_disc_domain) -> error otherwise.
      .def("set_geometry_mode", &System::set_geometry_mode, py::arg("mode"))
      // Domain 0/1 mask (ny, nx) row-major (diagnostic / contract verification). All 1.0 without
      // set_disc_domain.
      .def("disc_mask", [](const System& s) { return to_2d(s.disc_mask(), s.ny(), s.nx()); })
      .def(
          "set_epsilon_field",
          [](System& s, py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_epsilon_field(flat(arr));
          },
          py::arg("eps"))
      .def(
          "set_epsilon_anisotropic_field",
          [](System& s, py::array_t<double, py::array::c_style | py::array::forcecast> eps_x,
             py::array_t<double, py::array::c_style | py::array::forcecast> eps_y) {
            s.set_epsilon_anisotropic_field(flat(eps_x), flat(eps_y));
          },
          py::arg("eps_x"), py::arg("eps_y"))
      .def(
          "set_reaction_field",
          [](System& s, py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_reaction_field(flat(arr));
          },
          py::arg("kappa"))
      .def(
          "set_magnetic_field",
          [](System& s, py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_magnetic_field(flat(arr));
          },
          py::arg("bz"))
      // NAMED aux fields (ADC-70 phase 1): by canonical COMPONENT (>= 5). The name -> comp
      // resolution lives in the Python facade (adc.System.set_aux_field), which calls these two methods.
      .def(
          "set_aux_field_component",
          [](System& s, int comp,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_aux_field_component(comp, flat(arr));
          },
          py::arg("comp"), py::arg("field"))
      // ADC-369: per-field aux halo policy (bc_type = adc::BCType Foextrap=1 / Dirichlet=2). The Python
      // facade (adc.System.set_aux_field(..., halo=adc.AuxHalo(...))) resolves name -> comp and calls this.
      .def(
          "set_aux_field_halo_component",
          [](System& s, int comp, int bc_type, double value) {
            s.set_aux_field_halo_component(comp, bc_type, value);
          },
          py::arg("comp"), py::arg("bc_type"), py::arg("value"))
      .def(
          "aux_field_component",
          [](const System& s, int comp) {
            return to_2d(s.aux_field_component(comp), s.ny(), s.nx());
          },
          py::arg("comp"))
      .def("set_electron_temperature_from", &System::set_electron_temperature_from, py::arg("name"))
      .def(
          "set_density",
          [](System& s, const std::string& name,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_density(name, flat(arr));
          },
          py::arg("name"), py::arg("rho"))
      // Init from the PRIMITIVES: prim = array (ncomp, n, n) component-major in the order of
      // primitive_vars(name); converted to conservative by the block's model. The Python facade
      // (adc.System.set_primitive_state(**prims)) assembles this array from the named kwargs.
      .def(
          "set_primitive_state",
          [](System& s, const std::string& name,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_primitive_state(name, flat(arr));
          },
          py::arg("name"), py::arg("prim"))
      // Diagnostic: conservative state -> primitive (ncomp, n, n), order of primitive_vars(name).
      .def(
          "get_primitive_state",
          [](System& s, const std::string& name) {
            return to_3d(s.get_primitive_state(name), s.n_vars(name), s.ny(), s.nx());
          },
          py::arg("name"))
      .def("solve_fields", &System::solve_fields)
      .def("step", &System::step, py::arg("dt"))
      .def("advance", &System::advance, py::arg("dt"), py::arg("nsteps"))
      .def("step_cfl", &System::step_cfl,
           "Advances by ONE step at dt = cfl * h / max wave speed of the system (also honors the "
           "optional bounds: substeps, stride, source_frequency, couplings, add_dt_bound). Returns "
           "the dt "
           "used.",
           py::arg("cfl"))
      .def("dt_hotspot", &System::dt_hotspot,
           "Diagnostic (ADC-182): (w, i, j) of the GLOBAL cell that dominates the transport CFL "
           "bound "
           "of block 'name' -- to locate a collapsing dt. On demand, off the hot path.",
           py::arg("name"))
      .def("step_adaptive", &System::step_adaptive,
           "Advances by ONE MULTIRATE macro-step: the slowest block sets the macro-step, each "
           "faster "
           "block is sub-cycled n = ceil(w_block / w_min) times. Returns the macro-step.",
           py::arg("cfl"))
      // Primitives for a CUSTOM time integrator in Python (take_step):
      .def(
          "eval_rhs",
          [](System& s, const std::string& name) {
            return to_3d(s.eval_rhs(name), s.n_vars(name), s.ny(), s.nx());
          },
          py::arg("name"))
      .def(
          "get_state",
          [](System& s, const std::string& name) {
            return to_3d(s.get_state(name), s.n_vars(name), s.ny(), s.nx());
          },
          py::arg("name"))
      .def(
          "set_state",
          [](System& s, const std::string& name,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_state(name, flat(arr));
          },
          py::arg("name"), py::arg("u"))
      .def("n_vars", &System::n_vars, py::arg("name"))
      .def("nx", &System::nx)
      .def("ny", &System::ny)
      .def("time", &System::time)
      .def("n_species", &System::n_species)
      .def("block_names", &System::block_names)
      .def("mass", &System::mass, py::arg("name"))
      .def(
          "density",
          [](const System& s, const std::string& name) {
            return to_2d(s.density(name), s.ny(), s.nx());
          },
          py::arg("name"))
      .def("potential", [](System& s) { return to_2d(s.potential(), s.ny(), s.nx()); })
      // GLOBAL accessors (MPI-safe collectives): multi-rank outputs / checkpoint (IO v1). Each
      // rank MUST call them (internal all_reduce); they return the COMPLETE field (rank-0 gather
      // implicit via all_reduce_sum) -- single-rank: bit-identical to density / get_state / potential.
      // The sim.write / sim.checkpoint facade uses them then writes the file only on rank 0.
      .def(
          "density_global",
          [](const System& s, const std::string& name) {
            return to_2d(s.density_global(name), s.ny(), s.nx());
          },
          py::arg("name"))
      .def(
          "state_global",
          [](const System& s, const std::string& name) {
            return to_3d(s.state_global(name), s.n_vars(name), s.ny(), s.nx());
          },
          py::arg("name"))
      .def("potential_global",
           [](System& s) { return to_2d(s.potential_global(), s.ny(), s.nx()); })
      // LOCAL per-fab accessors (NOT collective): PARALLEL HDF5 writing by hyperslabs (PR-IO-3,
      // sim.write(format='hdf5', parallel=True)). local_boxes returns the list of local boxes
      // (ilo, jlo, ihi, jhi) in GLOBAL indices; local_state returns the state of fab li reshaped
      // (n_vars, bny, bnx) for a hyperslab dset[:, jlo:jhi+1, ilo:ihi+1]. A rank without a box returns an
      // empty list. Since the System is single-box, real parallelism only appears on a multi-box
      // geometry (cf. AMR); the API stays correct in the general case.
      .def("local_boxes", &System::local_boxes, py::arg("name"))
      .def(
          "local_state",
          [](const System& s, const std::string& name, int li) {
            const auto boxes = s.local_boxes(name);
            if (li < 0 || li >= static_cast<int>(boxes.size()))
              throw std::out_of_range("System.local_state: local fab index out of bounds");
            const int bnx = boxes[li][2] - boxes[li][0] + 1;  // ihi - ilo + 1
            const int bny = boxes[li][3] - boxes[li][1] + 1;  // jhi - jlo + 1
            return to_3d(s.local_state(name, li), s.n_vars(name), bny, bnx);
          },
          py::arg("name"), py::arg("li"))
      .def_static("abi_key", &System::abi_key,
                  "Module ABI key (cf. adc.abi_key); compared to that of a native loader.");
}
