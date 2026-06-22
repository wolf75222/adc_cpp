#include "../bindings_detail.hpp"

// ADC-365: the AMR (AmrSystemConfig + AmrSystem) bindings.
void init_amr(py::module_& m) {
  // --- AMR: single-species composition on multi-patch AMR (generic composable brick) ---
  // adc_cases DRIVES it from Python (no C++ on the cases side) just like System.
  //
  // NB: the two-fluid AP integrator (BESPOKE asymptotic-preserving scheme, not composable
  // block by block) has left the core: it is not a generic brick but a SCENARIO. It now lives
  // in adc_cases (cf. adc_cases/two_fluid_ap/), compiled on the fly against the generic
  // headers of adc_cpp; it is no longer exposed by the _adc module.

  // AmrSystem: generic single-species composition on AMR.
  py::class_<AmrSystemConfig>(m, "AmrSystemConfig")
      .def(py::init<>())
      .def_readwrite("n", &AmrSystemConfig::n)
      .def_readwrite("L", &AmrSystemConfig::L)
      .def_readwrite("regrid_every", &AmrSystemConfig::regrid_every)
      .def_readwrite("periodic", &AmrSystemConfig::periodic)
      .def_readwrite("distribute_coarse", &AmrSystemConfig::distribute_coarse)
      .def_readwrite("coarse_max_grid", &AmrSystemConfig::coarse_max_grid);

  py::class_<AmrSystem>(m, "AmrSystem")
      .def(py::init<const AmrSystemConfig&>())
      // ADC-214: Python surface UNCHANGED (same flat newton_* kwargs, same defaults). The lambda
      // assembles the NewtonOptions POD before the C++ call (parity with System.add_block).
      .def(
          "add_block",
          [](AmrSystem& s, const std::string& name, const ModelSpec& model,
             const std::string& limiter, const std::string& riemann, const std::string& recon,
             const std::string& time, int substeps, int stride,
             const std::vector<std::string>& implicit_vars,
             const std::vector<std::string>& implicit_roles, int newton_max_iters,
             double newton_rel_tol, double newton_abs_tol, double newton_fd_eps,
             double newton_damping, const std::string& newton_fail_policy, bool newton_diagnostics,
             double positivity_floor) {
            NewtonOptions newton;
            newton.max_iters = newton_max_iters;
            newton.rel_tol = static_cast<Real>(newton_rel_tol);
            newton.abs_tol = static_cast<Real>(newton_abs_tol);
            newton.fd_eps = static_cast<Real>(newton_fd_eps);
            newton.damping = static_cast<Real>(newton_damping);
            newton.fail_policy =
                newton_fail_policy_from_string(newton_fail_policy, "AmrSystem::add_block");
            s.add_block(name, model, limiter, riemann, recon, time, substeps, stride, implicit_vars,
                        implicit_roles, newton, newton_diagnostics, positivity_floor);
          },
          py::arg("name"), py::arg("model"), py::arg("limiter") = "minmod",
          py::arg("riemann") = "rusanov", py::arg("recon") = "conservative",
          py::arg("time") = "explicit", py::arg("substeps") = 1, py::arg("stride") = 1,
          // Partial IMEX mask CARRIED BY THE BLOCK (capstone vii): conserved variables treated
          // implicitly by NAME (implicit_vars) or by physical ROLE (implicit_roles). Empty (default)
          // -> full backward-Euler. Only meaningful with time="imex" and MULTI-BLOCK (cf. add_block).
          py::arg("implicit_vars") = std::vector<std::string>{},
          py::arg("implicit_roles") = std::vector<std::string>{},
          // IMEX Newton options (wave 3, System parity): OPTIONS wired in MONO-BLOCK (coupler)
          // AND MULTI-BLOCK (engine). newton_diagnostics (newton_report report): native MULTI-BLOCK
          // only (mono-block rejected at build; .so loaders rejected at the Python facade).
          py::arg("newton_max_iters") = 2, py::arg("newton_rel_tol") = 0.0,
          py::arg("newton_abs_tol") = 0.0, py::arg("newton_fd_eps") = 1e-7,
          py::arg("newton_damping") = 1.0, py::arg("newton_fail_policy") = "none",
          py::arg("newton_diagnostics") = false,
          // Zhang-Shu positivity floor (ADC-259): Density-role face-state + C/F-ghost-mean floor on
          // the AMR transport. 0 (default) = inactive, bit-identical. Marshaled from spatial.positivity_floor
          // by the AmrSystem.add_block / add_equation Python facade.
          py::arg("positivity_floor") = 0.0)
      // Newton report (IMEX diagnostics OPT-IN, native MULTI-BLOCK): dict {enabled, converged,
      // max_residual, max_iters_used, n_failed, failed_cell, failed_component}, aggregated over the
      // levels/substeps of the LAST advance of the block. failed_cell = (i, j) or None. EXACT shape of
      // the System.newton_report binding (parity, including failed_cell tuple/None).
      .def(
          "newton_report",
          [](AmrSystem& s, const std::string& name) {
            const AmrSystem::SourceNewtonReport r = s.newton_report(name);
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
      // NATIVE AMR block loaded from a .so loader generated by the DSL (backend "production",
      // target="amr_system"): the .so inlines add_compiled_model(AmrSystem&) -> native block on the
      // AMR hierarchy (reflux, regrid), ABI key verified. cf. AmrSystem::add_native_block. NO
      // evolve (mono-block AMR). The AMR LIMITS (primitive/roe/hllc/weno5) are guarded on the Python
      // facade side (AmrSystem.add_equation) before this binding.
      .def("add_native_block", &AmrSystem::add_native_block, py::arg("name"), py::arg("so_path"),
           py::arg("limiter") = "minmod", py::arg("riemann") = "rusanov",
           py::arg("recon") = "conservative", py::arg("time") = "explicit", py::arg("gamma") = 1.4,
           py::arg("substeps") = 1,
           // Zhang-Shu positivity floor (ADC-322): marshaled down the regenerated .so loader
           // (adc_install_native_amr). 0 (default) = inactive, bit-identical.
           py::arg("positivity_floor") = 0.0)
      // Regrid criterion: refine where the SELECTED variable exceeds threshold. Default = component 0
      // (historical density), bit-identical 1e30 no-op. ADC-296: select it PER BLOCK by NAME (variable=)
      // or physical ROLE (role=); a block lacking it raises at build (no silent comp-0 fallback). A
      // non-default selector is MULTI-BLOCK only (mono-block / compiled .so refine on component 0).
      .def("set_refinement", &AmrSystem::set_refinement, py::arg("threshold"),
           py::arg("variable") = "", py::arg("role") = "",
           "Refine where the selected conserved variable exceeds threshold. variable=/role= pick "
           "it per "
           "block by name or physical role (default: component 0, the historical density). "
           "Selecting by "
           "name and role at once, or a name/role absent from a block, raises. Non-default "
           "selector is "
           "multi-block only.")
      // PHI tag on |grad phi| (D4) added to the union of regrid tags: also refines where the
      // norm of the potential gradient exceeds grad_threshold (diocotron ring edge). MULTI-BLOCK
      // + regrid_every > 0. <= 0 (default) -> phi DISABLED (bit-identical). cf. AmrSystem::set_phi_refinement.
      .def("set_phi_refinement", &AmrSystem::set_phi_refinement, py::arg("grad_threshold"))
      .def(
          "set_poisson", &AmrSystem::set_poisson,
          "Configures the coarse Poisson of the AMR hierarchy (cf. System.set_poisson). On AMR the "
          "solver is ALWAYS GeometricMG and the right-hand side ALWAYS the sum of the elliptic "
          "bricks. rhs: 'charge_density' | 'composite'. solver: 'geometric_mg' only (no "
          "FFT on the hierarchy). bc: 'auto' | 'periodic' | 'dirichlet' | 'neumann'. wall: "
          "'none' | 'circle' (circular conducting wall, requires wall_radius > 0).",
          py::arg("rhs") = "charge_density", py::arg("solver") = "geometric_mg",
          py::arg("bc") = "auto", py::arg("wall") = "none", py::arg("wall_radius") = 0.0)
      // GLOBAL step bound + ACTIVE bound (AMR StabilityPolicy, System.add_dt_bound parity).
      .def("add_dt_bound", &AmrSystem::add_dt_bound, py::arg("label"), py::arg("fn"))
      .def("last_dt_bound", &AmrSystem::last_dt_bound)
      // amr-schur PATH (AMR counterpart of System.set_magnetic_field / set_source_stage / set_time_scheme).
      // GLOBAL Schur-condensed source stage (electrostatic/Lorentz) on the mono-block hierarchy,
      // instead of the local IMEX source. B_z (Lorentz term) accepts a flattened numpy (n, n).
      .def(
          "set_magnetic_field",
          [](AmrSystem& s, py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_magnetic_field(flat(arr));
          },
          py::arg("bz"))
      // ADC-291: model-NAMED aux field at a resolved channel component (>= kAuxNamedBase). The Python
      // facade (AmrSystem.set_aux_field) resolves the name -> comp and reshapes (n, n) -> flat n*n.
      .def(
          "set_aux_field_component",
          [](AmrSystem& s, int comp,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_aux_field_component(comp, flat(arr));
          },
          py::arg("comp"), py::arg("field"))
      // ADC-369: per-field aux halo policy (bc_type = adc::BCType Foextrap=1 / Dirichlet=2).
      .def(
          "set_aux_field_halo_component",
          [](AmrSystem& s, int comp, int bc_type, double value) {
            s.set_aux_field_halo_component(comp, bc_type, value);
          },
          py::arg("comp"), py::arg("bc_type"), py::arg("value"))
      // ADC-214: Python surface UNCHANGED (same flat krylov_* kwargs / descriptors, same
      // defaults; no bz_aux_component on the AMR side). The lambda assembles the SourceStageOptions POD.
      .def(
          "set_source_stage",
          [](AmrSystem& s, const std::string& name, const std::string& kind, double theta,
             double alpha, double krylov_tol, int krylov_max_iters, const std::string& density,
             const std::string& momentum_x, const std::string& momentum_y,
             const std::string& energy) {
            SourceStageOptions opts;
            opts.krylov_tol = krylov_tol;
            opts.krylov_max_iters = krylov_max_iters;
            opts.density = density;
            opts.momentum_x = momentum_x;
            opts.momentum_y = momentum_y;
            opts.energy = energy;
            s.set_source_stage(name, kind, theta, alpha, opts);
          },
          py::arg("name"), py::arg("kind"), py::arg("theta"), py::arg("alpha"),
          // Carried settings (wave 3, System parity): Krylov tolerances of the coarse solve
          // (<= 0 = defaults 1e-10/400) + field descriptors ("" = canonical role).
          py::arg("krylov_tol") = 0.0, py::arg("krylov_max_iters") = 0, py::arg("density") = "",
          py::arg("momentum_x") = "", py::arg("momentum_y") = "", py::arg("energy") = "")
      .def("set_time_scheme", &AmrSystem::set_time_scheme, py::arg("scheme"))
      .def(
          "set_density",
          [](AmrSystem& s, const std::string& name,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_density(name, flat(arr));
          },
          py::arg("name"), py::arg("rho"))
      // Full initial conservative state (ncomp, n, n) -> starts the AMR from the paper's drift
      // state (rho, rho*u, rho*v) instead of m=0. Keeps ndim==3 EXPLICIT: flat() flattens
      // any C-contiguous array, so a 2D density (n, n) passed by mistake would become a
      // 1-component state (comp 0 = density, momentum left at 0) -- a silent density masquerade
      // with the wrong physics. We require (ncomp, n, n). flat() then flattens in
      // component-major c*n*n + j*n + i (same convention as to_3d / set_state).
      .def(
          "set_conservative_state",
          [](AmrSystem& s, const std::string& name,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            if (arr.ndim() != 3)
              throw std::runtime_error(
                  "AmrSystem.set_conservative_state: state expected of shape (ncomp, n, n); got "
                  "a " +
                  std::to_string(arr.ndim()) +
                  "D array (a 2D density? use "
                  "set_density)");
            s.set_conservative_state(name, flat(arr));
          },
          py::arg("name"), py::arg("U"))
      // Inter-species COUPLED source (compiled adc.dsl.CoupledSource, P5 bytecode), MULTI-BLOCK on the
      // SHARED AMR hierarchy: applied after the transport at each macro-step, by explicit
      // splitting, level by level + fine -> coarse cascade (consistent covered cells). SAME
      // flat ABI as System.add_coupled_source. Without the call, unchanged. cf. AmrSystem::add_coupled_source.
      // ADC-214: Python surface UNCHANGED (same flat kwargs, same defaults). The lambda assembles the
      // CoupledSourceProgram POD before the C++ call (parity with System.add_coupled_source).
      .def(
          "add_coupled_source",
          [](AmrSystem& s, const std::vector<std::string>& in_blocks,
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
          py::arg("frequency") = 0.0, py::arg("label") = "coupled_source",
          // Optional PER-CELL frequency mu(U): evaluated on the coarse level (cf. System).
          py::arg("freq_prog_ops") = std::vector<int>{},
          py::arg("freq_prog_args") = std::vector<int>{})
      .def("step", &AmrSystem::step, py::arg("dt"))
      .def("advance", &AmrSystem::advance, py::arg("dt"), py::arg("nsteps"))
      .def("step_cfl", &AmrSystem::step_cfl,
           "Advances by one AMR macro-step at dt = cfl * dx_coarse / max wave speed (also honors "
           "the "
           "substeps/stride cadence in multi-block and the optional bounds). Returns the dt used.",
           py::arg("cfl"))
      .def("nx", &AmrSystem::nx)
      .def("time", &AmrSystem::time)
      // AMR clock (IO v1, System parity): macro-step counter + restoration (t, macro_step) ->
      // the regrid/stride cadence resumes exactly after a set_clock. Prerequisite PR-IO-3.
      .def("macro_step", &AmrSystem::macro_step)
      .def("set_clock", &AmrSystem::set_clock, py::arg("t"), py::arg("macro_step"))
      .def("n_blocks", &AmrSystem::n_blocks)
      .def("block_names", &AmrSystem::block_names)
      .def("n_patches", &AmrSystem::n_patches)
      // Index-space footprints of the fine patches: list of tuples (level, ilo, jlo, ihi, jhi), INCLUSIVE
      // corners, in the index space of the level (n << level cells/direction, ratio 2). SAME
      // source as n_patches() (the GLOBAL fine BoxArray) -> rank-independent, MPI-safe. Query between
      // steps, zero cost on the hot path. The Python wrapper converts to [0, L]^2 (it knows n via nx() and
      // L); cf. AmrSystem.patch_rectangles() on the facade side.
      .def("patch_boxes",
           [](AmrSystem& s) {
             py::list out;
             for (const adc::PatchBox& b : s.patch_boxes())
               out.append(py::make_tuple(b.level, b.ilo, b.jlo, b.ihi, b.jhi));
             return out;
           })
      // COARSE-level (base) box counts (ADC-319, MPI ownership diagnostic): coarse_local_boxes() = base
      // boxes OWNED by this rank (level-0 local_size()); coarse_total_boxes() = total base boxes (BoxArray
      // size, identical on all ranks). distribute_coarse=True -> local < total per rank (distributed
      // coarse transport); replicated / single-box -> local == total. Query between steps, no hot cost.
      .def("coarse_local_boxes", &AmrSystem::coarse_local_boxes)
      .def("coarse_total_boxes", &AmrSystem::coarse_total_boxes)
      // mass / density: overload by BLOCK NAME (multi-block; empty name -> 1st block, mono-block
      // compat or cosmetic name). The name INDEXES the block in multi-block (each block has its mass /
      // density, conserved PER BLOCK at reflux). Without argument -> 1st block (mono-block back-compat).
      .def("mass", [](AmrSystem& s) { return s.mass(); })
      .def(
          "mass", [](AmrSystem& s, const std::string& name) { return s.mass(name); },
          py::arg("name"))
      // AMR: SQUARE domain (n x n), no polar geometry -> rows == cols == nx() (unchanged).
      .def("density", [](AmrSystem& s) { return to_2d(s.density(), s.nx(), s.nx()); })
      .def(
          "density",
          [](AmrSystem& s, const std::string& name) {
            return to_2d(s.density(name), s.nx(), s.nx());
          },
          py::arg("name"))
      // phi of the coarse (base) level, (n, n). SAME observable as System.potential(): level 0
      // covers the whole domain -> enough to sample a median circle (azimuthal FFT). In
      // multi-block, phi results from the SYSTEM Poisson (Sum_b q_b n_b co-located), shared by all.
      .def("potential", [](AmrSystem& s) { return to_2d(s.potential(), s.nx(), s.nx()); })
      // AMR CHECKPOINT / RESTART single-rank (ADC-65): full conservative state per level + phi
      // (warm-start) + imposition of the saved fine hierarchy. SERIAL MONO-BLOCK (multi-block: C++
      // rejection; np>1: facade rejection -- per-level gather = future). level_state / level_potential return
      // FLAT fields (c*nf*nf + j*nf + i / nf*nf, nf = nx << k); the facade reshapes. set_*
      // flatten any C-contiguous array (flat). set_hierarchy: list of tuples
      // (level, ilo, jlo, ihi, jhi) like patch_boxes() (the coupler filters level 1).
      .def("n_levels", &AmrSystem::n_levels)
      .def("n_vars", [](AmrSystem& s) { return s.n_vars(); })
      .def(
          "level_state", [](AmrSystem& s, int k) { return s.level_state(k); }, py::arg("k"))
      .def(
          "set_level_state",
          [](AmrSystem& s, int k,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_level_state(k, flat(arr));
          },
          py::arg("k"), py::arg("state"))
      .def(
          "level_potential", [](AmrSystem& s, int k) { return s.level_potential(k); }, py::arg("k"))
      .def(
          "set_level_potential",
          [](AmrSystem& s, int k,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_level_potential(k, flat(arr));
          },
          py::arg("k"), py::arg("phi"))
      .def(
          "set_hierarchy",
          [](AmrSystem& s, const std::vector<std::tuple<int, int, int, int, int>>& boxes) {
            std::vector<adc::PatchBox> bx;
            bx.reserve(boxes.size());
            for (const auto& b : boxes)
              bx.push_back(adc::PatchBox{std::get<0>(b), std::get<1>(b), std::get<2>(b),
                                         std::get<3>(b), std::get<4>(b)});
            s.set_hierarchy(bx);
          },
          py::arg("boxes"));
}
