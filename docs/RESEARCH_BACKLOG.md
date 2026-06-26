# Research / external backlog (items NOT auto-completable)

Scope by multi-agent workflow (June 2026). These items are numerical RESEARCH or EXTERNAL INTEGRATION:
they do not "finish" like an agent PR. Each one has a factual status, a concrete next step,
and a NO-GO / decision-gate criterion. To be picked up by the owner when relevant.

## Tensor AP under strong field -- RESEARCH

- Status: the condensed Schur (schur_condensation.hpp, condensed_schur_source_stepper.hpp) ALREADY handles the
  strong field UNCONDITIONALLY (exact 2x2 inversion LorentzEliminator). It is stable, but does not neutralize
  the growth of the coefficients in omega_c*dt. A tensor AP would be an EFFICIENCY gain, NOT a stability one
  (already acquired). The two-fluid AP exists only as a scalar modification of the RHS (two_fluid_ap.md), not as
  a tensor-operator reformulation.
- Next step (math study, no code): asymptotic expansion of the condensed Schur eqs. in the limit
  omega_c >> omega_d (analogous to two_fluid_ap.md sec.3); identify whether a dimensionless scaling factor
  emerges; 1D toy (isothermal plane-parallel, uniform B_z) comparing current Schur vs tensor AP; PR
  only if validated.
- NO-GO: if the AP requires a NON-LOCAL operator incompatible with the roles/DSL -> defer indefinitely.

## Full-device perf scaling -- NEEDS ROMEO FIRST

- Status: the COARSE mesh is REPLICATED by CHOICE (amr_coupler_mp.hpp:224-234 coupler_make_coarse_layout,
  amr_dsl_block.hpp:159-188; replicated_coarse=true). The DISTRIBUTED mode exists (distribute_coarse=true) but
  measures 3-5x SLOWER (705-1403 ms/step vs 222-278 replicated): the MG V-cycle exchanges cross-rank halos at
  each coarse level (~7 levels, fill_boundary latency-bound on 2x2 boxes). Poisson dominates 96-99.9%.
- Next step: ROMEO multi-GPU profile (instrument GeometricMG::vcycle_rec with Kokkos::fence + per-level
  timing) on np=2/4 GH200 -> isolate where the 3-5x slowdown happens.
- DECISION GATE: if the latency > 50% of the coarse time -> HYBRID MG (distribute the fine, gather for the
  bottom-solve) is worth 2-3 weeks; otherwise coarse replication is the RIGHT trade-off -> close the lot.

## SAMRAI integration -- EXTERNAL-BIG, DEFERRED

- Status: pops has a COMPLETE IN-HOUSE AMR stack (Phase 1 frozen hierarchy: substeps #175, coupled sources #179,
  local IMEX #184; Phase 2 union-tags regrid #199; reflux/FluxRegister, multi-bloc, validated CPU Serial/
  OpenMP/MPI np=1/2/4 + GPU Cuda GH200). SAMRAI (LLNL) = external AMR framework.
- VERDICT: DEFER as long as the in-house AMR covers the science path (polar diocotron convergent in
  resolution via union-tags regrid). REOPENING criterion: a need that the in-house AMR does NOT cover
  (e.g. mature distributed multi-level MG, multi-node hero-run scaling) AND an integration cost that is justified
  (C++ dependency, structure mapping, Python binding, maintenance).

## P7-a implicit-total -- RESEARCH

- Status: today Lie splitting (explicit-transport SSPRK3 + implicit-source-Schur theta). "implicit-total"
  = a TOTALLY implicit scheme (transport + source), a big numerical work item (full Jacobian, nonlinear
  solver). P7-b (DSL runtime params) is SEPARATE and DOABLE (PR in progress).
- Next step: scheme study (research); not auto-completable. Only launch if the order-1 Lie splitting
  becomes a measured limiting factor (otherwise the simpler order-2 Strang is enough, cf. roadmap).
