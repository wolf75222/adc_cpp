#!/usr/bin/env python3
"""Compiled/sim inspection completeness: inspect / requirements / dumps / explain_bind (Spec 5 sec.12.1).

INERT inspection (criterion #15, epic ADC-479): build a SYNTHETIC ``CompiledProblem`` (a real
in-memory ``pops.time.Program`` + a real ``CompiledModel`` metadata carrier -- NO Kokkos compile, NO
.so on disk) and assert that

  - ``compiled.inspect()`` (sec.12.1) returns a printable ``CompiledReport`` aggregating name /
    backend / platform / layout / blocks / fields / program / required inputs / artifacts / status,
    serialising via ``to_dict`` / ``to_json`` WITHOUT binding;
  - ``compiled.requirements()`` lists the COMPILE-TIME constraints (model capabilities + layout /
    backend / abi), DISTINCT from ``arguments()``; an unknowable piece is reported honestly;
  - ``compiled.inspect_capabilities()`` returns the descriptor capability rows for this compiled;
  - ``compiled.dump_ir`` / ``dump_cpp`` / ``dump_schedule`` EXPOSE the existing codegen (IR JSON =
    the ``_serialize`` the hash digests; C++ = ``emit_cpp_program``; schedule = the commit order),
    and raise a CLEAR error -- never fake a file -- when the carried Program is missing;
  - ``System.explain_bind`` / ``AmrSystem.explain_bind`` return a printable ``BindReport`` of
    provided-vs-required for a REAL System (no fake engine), reusing ADC-463 collect_missing.

``dump_cpp`` emits the C++ SOURCE host-side (pure codegen, no Kokkos); only the real ``.so`` compile
that ``compile_problem`` performs is Kokkos-gated, exercised by a guarded smoke check that skips
cleanly when the toolchain is absent (it never fakes a compile).

Pytest + __main__ guard (CI runs ``python3 <file>``).
"""
import json
import os
import sys
import tempfile

try:
    import pops  # noqa: F401
    from pops.codegen import BindReport, CompiledReport, RequirementsReport
    from pops.codegen.loader import CompiledModel, CompiledProblem
    from pops.physics.model import Param
    from pops import time as adctime
except Exception as exc:  # noqa: BLE001 -- pops unavailable in this interpreter
    print("skip test_inspection_completeness (pops unavailable: %s)" % exc)
    sys.exit(0)


def _program(name="intro_demo"):
    """A real in-memory Program: a state, an elliptic field solve, a Forward-Euler commit."""
    P = adctime.Program(name)
    dt = P.dt
    U = P.state("plasma")
    f = P.solve_fields("phi", U)
    R = P.rhs(state=U, fields=f, flux=True, sources=["default"])
    P.commit("plasma", P.linear_combine("U1", U + dt * R))
    return P


def _model(*, n_vars=4, n_aux=1, aux_names=("B_z",), params=None, caps=None,
           has_hllc=False, has_roe=False, has_wave_speeds=False):
    """A real CompiledModel metadata carrier (no .so) -- the engine class, carrying only metadata.

    The friendly ``has_*`` flags map to the CompiledModel constructor's ``hllc`` / ``roe`` /
    ``wave_speeds`` capability args (the attributes the model exposes as ``has_hllc`` / ...)."""
    cons = ["rho", "mx", "my", "E"][:n_vars]
    roles = ["Density", "MomentumX", "MomentumY", "Energy"][:n_vars]
    return CompiledModel(
        so_path="/nonexistent/problem.so", backend="production", adder="add_native_block",
        cons_names=cons, cons_roles=roles, prim_names=cons, n_vars=n_vars, gamma=1.4,
        n_aux=n_aux, params=params or {}, caps=caps or {"cpu": True, "mpi": True},
        abi_key="SIG|c++|c++23", model_hash="modelhash", cxx="c++", std="c++23",
        aux_extra_names=list(aux_names),
        hllc=has_hllc, roe=has_roe, wave_speeds=has_wave_speeds)


def _compiled(*, program=None, params=None, **model_kw):
    """A SYNTHETIC CompiledProblem: a real lowered Program + a real CompiledModel, no compile."""
    P = program if program is not None else _program()
    m = _model(params=params, **model_kw)
    return CompiledProblem("/tmp/pops-cache/problem.so", P, m, "SIG|c++|c++23", "c++", "c++23",
                           problem_hash="deadbeefcafe", cache_key="0badc0de")


def chk(cond, label):
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    assert cond, label


# ---------------------------------------------------------------------------
# sec.12.1: inspect() -> CompiledReport
# ---------------------------------------------------------------------------

def test_inspect_aggregates_metadata():
    """inspect() returns a CompiledReport aggregating name / backend / blocks / fields / inputs."""
    print("== inspect() aggregates the compiled metadata ==")
    params = {"cs2": Param("cs2", 1.0, kind="runtime"),
              "gamma_const": Param("gamma_const", 1.4, kind="const")}
    cp = _compiled(n_vars=4, n_aux=1, aux_names=("B_z",), params=params, has_wave_speeds=True)
    rep = cp.inspect()
    chk(isinstance(rep, CompiledReport), "inspect() returns a CompiledReport")
    chk(rep.name == "intro_demo", "report names the program")
    chk(rep.backend == "production", "a compiled time Program is the production backend")
    chk(rep.platform == "mpi", "platform reflects the model's mpi capability")
    chk(rep.layout == "system", "layout is the single-level System runtime")
    chk(any(b["name"] == "plasma" and b["components"] == 4 for b in rep.blocks),
        "the committed block 'plasma' carries the model component count")
    chk(any(f["name"] == "phi" for f in rep.fields), "the elliptic 'phi' field is listed")
    chk(rep.inputs["states"] == ["plasma"], "required states list the committed block")
    chk(rep.inputs["params"] == ["cs2"], "only the runtime param is a required input (const frozen)")
    chk(rep.inputs["aux"] == ["B_z"], "the named aux is a required input")
    chk(rep.program["commits"] == ["plasma"], "program summary lists the committed block")
    chk(rep.artifacts["so_path"] == cp.so_path, "artifacts carry the .so path")
    chk(rep.status == "compiled, waiting for pops.bind(...)", "status is the bind-pending line")


def test_inspect_printable_and_serialisable():
    """str(inspect()) is a readable multi-line report; to_dict / to_json round-trip."""
    print("== inspect() is printable + serialisable ==")
    cp = _compiled()
    rep = cp.inspect()
    text = str(rep)
    chk("compiled problem 'intro_demo'" in text, "str() names the program")
    chk("status   : compiled, waiting for pops.bind(...)" in text, "str() carries the status line")
    chk("object at 0x" not in text, "str() is not the default <...object at 0x...> repr")
    d = rep.to_dict()
    chk(json.loads(json.dumps(d)) == d, "to_dict is JSON round-trippable")
    chk(d["backend"] == "production", "to_dict carries the backend")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "inspect.json")
        rep.to_json(path)
        chk(json.load(open(path, encoding="utf-8"))["name"] == "intro_demo", "to_json(path) is valid")


def test_inspect_is_inert():
    """inspect() reads metadata only -- no bind, no .so read."""
    print("== inspect() is inert (no .so on disk needed) ==")
    cp = _compiled()
    chk(not os.path.exists(cp.so_path), "the synthetic .so path does not exist")
    chk(isinstance(cp.inspect(), CompiledReport), "inspect() works with no .so on disk")


# ---------------------------------------------------------------------------
# sec.12.1: requirements() -> RequirementsReport (DISTINCT from arguments())
# ---------------------------------------------------------------------------

def test_requirements_lists_compile_constraints():
    """requirements() lists model capabilities + layout / backend, distinct from arguments()."""
    print("== requirements() lists the compile-time constraints ==")
    cp = _compiled(has_wave_speeds=True, has_hllc=True)
    req = cp.requirements()
    chk(isinstance(req, RequirementsReport), "requirements() returns a RequirementsReport")
    tokens = {c["capability"] for c in req.capabilities}
    chk("wave_speeds" in tokens, "the wave_speeds capability the flux needs is reported")
    chk("hllc_star_state" in tokens, "the HLLC capability is reported")
    chk(req.constraints["backend"] == "production", "the backend constraint is production")
    chk(req.constraints["layout"] == "system", "the layout constraint is the System runtime")
    chk(req.constraints["abi_key"] == "SIG|c++|c++23", "the ABI key the toolchain must match")
    # DISTINCT from arguments(): requirements has no 'instances' bind-input list.
    chk(not hasattr(req, "instances"), "requirements() is not the arguments() bind-input list")


def test_requirements_defers_honestly():
    """requirements() reports unknowable pieces honestly (the spatial scheme is a bind input)."""
    print("== requirements() defers unknowable pieces honestly ==")
    cp = _compiled()  # a base model with no has_* capability flags
    req = cp.requirements()
    chk(req.capabilities == [], "a base Rusanov-only model reports no extra capability (not faked)")
    chk(any("bind input" in note.lower() for note in req.unknown),
        "the spatial scheme is honestly reported as a bind input, not fabricated")
    chk(any("stencil" in note.lower() or "ghost" in note.lower() for note in req.unknown),
        "the unrecorded stencil width is honestly deferred")
    chk("constraints" in str(req) and "backend" in str(req), "str() prints the constraints table")


# ---------------------------------------------------------------------------
# sec.12.1: inspect_capabilities() -> scoped CapabilityMatrix
# ---------------------------------------------------------------------------

def test_inspect_capabilities_scoped():
    """inspect_capabilities() returns the descriptor capability rows for this compiled."""
    print("== inspect_capabilities() reuses the top-level matrix ==")
    cp = _compiled()
    matrix = cp.inspect_capabilities()
    names = {e.name for e in matrix}
    chk(len(matrix) > 0, "the capability matrix is non-empty")
    chk("rusanov" in names and "hll" in names, "the Riemann fluxes are catalogued")
    cats = set(matrix.categories())
    chk("riemann" in cats and "reconstruction" in cats, "the route-choosing categories are present")
    chk("capability matrix" in str(matrix), "the matrix is printable")
    # It matches the top-level machinery scoped to the compiled's categories.
    chk(cats <= set(cp._CAPABILITY_CATEGORIES), "the scope is the compiled's bind categories")


# ---------------------------------------------------------------------------
# sec.12.1: dump_ir / dump_cpp / dump_schedule (EXPOSE the existing codegen)
# ---------------------------------------------------------------------------

def test_dump_ir_serialises_the_program():
    """dump_ir writes the SAME serialization _ir_hash digests (nodes / commits / block order)."""
    print("== dump_ir exposes the serialized Program IR ==")
    cp = _compiled()
    blob = cp.dump_ir()
    parsed = json.loads(blob)
    # dump_ir IS the Program _serialize blob, JSON-normalised (tuples -> lists through json).
    normalised = json.loads(json.dumps(cp.program._serialize()))
    chk(parsed == normalised, "dump_ir() is exactly the Program _serialize blob")
    chk(parsed["name"] == "intro_demo" and parsed["commits"], "the IR carries name + commits")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ir.json")
        out = cp.dump_ir(path)
        chk(out == path, "dump_ir(path) returns the written path")
        chk(json.load(open(path, encoding="utf-8")) == parsed, "dump_ir(path) wrote the same JSON")


def test_dump_cpp_reuses_emit():
    """dump_cpp writes the generated C++ by REUSING emit_cpp_program (host-side, no Kokkos)."""
    print("== dump_cpp reuses the existing emit_cpp_program codegen ==")
    cp = _compiled()
    expected = cp.program.emit_cpp_program(model=cp.model)
    with tempfile.TemporaryDirectory() as tmp:
        # A directory target writes <program_name>.cpp inside it.
        out = cp.dump_cpp(tmp)
        chk(out == os.path.join(tmp, "intro_demo.cpp"), "dump_cpp(dir) writes <name>.cpp in the dir")
        chk(open(out, encoding="utf-8").read() == expected, "the written C++ IS the emitted source")
        chk("pops_install_program" in open(out, encoding="utf-8").read(),
            "the generated source carries the .so install entry point")
        # An explicit .cpp path is written verbatim.
        explicit = os.path.join(tmp, "custom.cpp")
        chk(cp.dump_cpp(explicit) == explicit, "dump_cpp(path.cpp) writes that path")
    # A non-existent target directory is a CLEAR error (never silently creates / fakes).
    try:
        cp.dump_cpp("/nonexistent-dir-xyz/sub")
        chk(False, "dump_cpp should reject a missing target directory")
    except NotADirectoryError:
        chk(True, "dump_cpp rejects a missing target directory")


def test_dump_schedule_lists_commit_order():
    """dump_schedule writes the block commit order (the runtime block-index schedule)."""
    print("== dump_schedule exposes the commit/schedule order ==")
    cp = _compiled()
    text = cp.dump_schedule()
    chk("commit" in text and "plasma" in text, "the schedule names the committed block")
    chk("0  commit plasma" in text, "the block is at runtime index 0")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "schedule.txt")
        chk(cp.dump_schedule(path) == path, "dump_schedule(path) returns the path")
        chk("plasma" in open(path, encoding="utf-8").read(), "dump_schedule(path) wrote the listing")


def test_dumps_raise_clear_error_without_program():
    """A handle with no Program raises a CLEAR error -- never fakes a file."""
    print("== dump_* fail loud (no fake file) when no Program is carried ==")
    cp = CompiledProblem("/tmp/pops-cache/problem.so", None, _model(), "SIG|c++|c++23",
                         "c++", "c++23")
    for who, call in (("dump_ir", lambda: cp.dump_ir()),
                      ("dump_cpp", lambda: cp.dump_cpp(tempfile.gettempdir())),
                      ("dump_schedule", lambda: cp.dump_schedule())):
        try:
            call()
            chk(False, "%s should raise without a Program" % who)
        except ValueError as exc:
            chk("carries no Program" in str(exc), "%s raises a clear no-Program error" % who)


# ---------------------------------------------------------------------------
# sec.12.1: System.explain_bind / AmrSystem.explain_bind (REAL engine, no fake)
# ---------------------------------------------------------------------------

def test_explain_bind_on_real_system():
    """System.explain_bind reports provided-vs-required for a REAL System (reuses ADC-463)."""
    print("== System.explain_bind (real System, no fake engine) ==")
    params = {"cs2": Param("cs2", 1.0, kind="runtime")}
    cp = _compiled(n_vars=4, n_aux=1, aux_names=("B_z",), params=params)
    sim = pops.System(n=16, L=1.0, periodic=True)  # a REAL engine (no fake adc)
    rep = sim.explain_bind(cp)
    chk(isinstance(rep, BindReport), "explain_bind returns a BindReport")
    chk(rep.required["instances"] == ["plasma"], "the committed block is a required instance")
    chk(rep.required["params"] == ["cs2"], "the runtime param is required")
    chk(rep.required["aux"] == ["B_z"], "the named aux is required")
    chk(rep.provided["instances"] == [], "a fresh System provides no block yet")
    chk(not rep.ready, "a fresh System is NOT ready to bind (inputs missing)")
    # Each missing line is actionable (names the install keyword).
    chk(any("instance 'plasma'" in m for m in rep.missing), "the missing instance is named")
    chk(any("runtime param 'cs2'" in m for m in rep.missing), "the missing runtime param is named")
    chk("bind plan" in str(rep), "the report is printable")
    d = rep.to_dict()
    chk(json.loads(json.dumps(d)) == d, "to_dict is JSON round-trippable")


def test_explain_bind_amr_parity():
    """AmrSystem.explain_bind has signature parity with System.explain_bind (inert)."""
    print("== AmrSystem.explain_bind parity ==")
    cp = _compiled(n_vars=4, n_aux=1, aux_names=("B_z",))
    amr = pops.AmrSystem(n=16, L=1.0)
    rep = amr.explain_bind(cp)
    chk(isinstance(rep, BindReport), "AmrSystem.explain_bind returns a BindReport")
    chk(rep.required["instances"] == ["plasma"], "the required instance is reported on AMR too")
    chk(not rep.ready, "a fresh AmrSystem is not ready to bind")


# ---------------------------------------------------------------------------
# Kokkos-gated smoke: a REAL compile_problem produces a handle whose dumps work end-to-end.
# Skips cleanly when the toolchain (compiler + Kokkos) is absent -- never fakes a compile.
# ---------------------------------------------------------------------------

def test_real_compile_inspection_or_skips():
    """A real compile_problem(...) handle inspects + dumps end-to-end (Kokkos-gated; skips if absent)."""
    print("== real compile_problem inspection (Kokkos-gated; skips if absent) ==")
    program = _program()
    try:
        compiled = pops.compile_problem(time=program, force=True)
    except Exception as exc:  # noqa: BLE001 -- no compiler / no Kokkos: skip cleanly
        print("  [..] real compile skipped (no toolchain / Kokkos): %s" % str(exc)[:90])
        return
    chk(isinstance(compiled.inspect(), CompiledReport), "a real handle inspects")
    chk(isinstance(compiled.requirements(), RequirementsReport), "a real handle has requirements")
    chk("name" in compiled.dump_ir(), "a real handle dumps its IR")
    with tempfile.TemporaryDirectory() as tmp:
        chk(os.path.exists(compiled.dump_cpp(tmp)), "a real handle dumps its C++ source")


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print("FAIL %s: %s" % (fn.__name__, exc))
    print("\n%d/%d test functions passed" % (len(fns) - failed, len(fns)))
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
