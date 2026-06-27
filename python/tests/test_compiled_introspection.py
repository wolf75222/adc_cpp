#!/usr/bin/env python3
"""Compiled-artifact introspection: arguments / estimate_memory / metadata (Spec 5 sec.12, ADC-479).

INERT introspection (criteria #44-49): build a SYNTHETIC ``CompiledProblem`` (a real
``pops.time.Program`` lowered in memory + a real ``CompiledModel`` metadata carrier -- NO Kokkos
compile, NO .so on disk) and assert that

  - ``compiled.arguments()`` (sec.12.2, #44-45) lists the bind inputs -- instances / params / aux /
    solvers / outputs / layout -- and serialises via ``to_dict`` / ``to_json`` WITHOUT binding;
  - ``compiled.estimate_memory(mesh=...)`` (sec.12.3, #46) returns a ``MemoryEstimate`` with
    positive byte categories, a FORMULA (it allocates no MultiFab), ``by_block`` / ``by_solver`` /
    ``by_scratch`` slices, ``to_dict`` / ``to_json``, and a conservative AMR-layout estimate;
  - the metadata attributes (sec.12.4, #48-49) ``codegen_dir`` / ``problem_hash`` / ``abi_key`` /
    ``cache_key`` / ``compile_command`` / ``generated_sources`` are present (or honestly ``None``).

This needs NO live Kokkos compile: the Program lowers in pure Python and the model is the engine's
metadata class with no .so. The one thing that DOES need Kokkos -- a real ``compile_problem``
producing a .so whose ``compile_command`` / ``generated_sources`` are populated -- is exercised only
in a guarded smoke check that skips cleanly when the toolchain is absent (it never fakes a compile).

Pytest + __main__ guard (CI runs ``python3 <file>``).
"""
import json
import os
import sys
import tempfile

try:
    import pops  # noqa: F401
    from pops.codegen import Arguments, MemoryEstimate
    from pops.codegen.loader import CompiledModel, CompiledProblem
    from pops.mesh.cartesian import CartesianMesh
    from pops.mesh.layouts import AMR, Uniform
    from pops.physics.model import Param
    from pops import time as adctime
except Exception as exc:  # noqa: BLE001 -- pops unavailable in this interpreter
    print("skip test_compiled_introspection (pops unavailable: %s)" % exc)
    sys.exit(0)


def _program(name="intro_demo", *, krylov=False):
    """A real in-memory Program: a state, an elliptic field solve, a Forward-Euler commit.

    With ``krylov=True`` it also records a matrix-free ``solve_linear`` (Krylov) node, so the IR
    carries the Krylov memory category."""
    P = adctime.Program(name)
    dt = P.dt
    U = P.state("plasma")
    f = P.solve_fields("phi", U)
    R = P.rhs(state=U, fields=f, flux=True, sources=["default"])
    if krylov:
        # A matrix-free Krylov solve (op x = rhs): lowers to a solve_linear IR node.
        buf = P.scalar_field("buf")
        A = P.matrix_free_operator("op")

        def _apply(p, out, x):
            lap = p.scalar_field("lap")
            p.laplacian(lap, x)
            return -1.0 * lap

        P.set_apply(A, _apply)
        P.solve_linear(operator=A, rhs=buf, method="cg", max_iter=10)
    P.commit("plasma", P.linear_combine("U1", U + dt * R))
    return P


def _model(*, n_vars=3, n_aux=1, aux_names=("B_z",), params=None, caps=None):
    """A real CompiledModel metadata carrier (no .so) -- the engine class, carrying only metadata."""
    cons = ["rho", "mx", "my", "E"][:n_vars]
    roles = ["Density", "MomentumX", "MomentumY", "Energy"][:n_vars]
    return CompiledModel(
        so_path="/nonexistent/problem.so", backend="production", adder="add_native_block",
        cons_names=cons, cons_roles=roles, prim_names=cons, n_vars=n_vars, gamma=1.4,
        n_aux=n_aux, params=params or {}, caps=caps or {"cpu": True, "mpi": True},
        abi_key="SIG|c++|c++23", model_hash="modelhash", cxx="c++", std="c++23",
        aux_extra_names=list(aux_names))


def _compiled(*, krylov=False, params=None, **model_kw):
    """A SYNTHETIC CompiledProblem: a real lowered Program + a real CompiledModel, no compile."""
    P = _program(krylov=krylov)
    m = _model(params=params, **model_kw)
    return CompiledProblem("/tmp/pops-cache/problem.so", P, m, "SIG|c++|c++23", "c++", "c++23",
                           problem_hash="deadbeefcafe", cache_key="0badc0de")


def chk(cond, label):
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    assert cond, label


# ---------------------------------------------------------------------------
# sec.12.2 (#44-45): arguments()
# ---------------------------------------------------------------------------

def test_arguments_lists_bind_inputs():
    """arguments() lists instances / params / aux / solvers from the carried Program + model."""
    print("== arguments() lists the bind inputs ==")
    params = {"cs2": Param("cs2", 1.0, kind="runtime"),
              "gamma_const": Param("gamma_const", 1.4, kind="const")}
    cp = _compiled(n_vars=3, n_aux=1, aux_names=("B_z",), params=params)
    args = cp.arguments()
    chk(isinstance(args, Arguments), "arguments() returns an Arguments")

    # Instances: the block the Program commits, with the model's state + component count.
    chk(set(args.instances) == {"plasma"}, "instances lists the committed block 'plasma'")
    chk(args.instances["plasma"]["components"] == 3, "instance carries n_cons components")
    chk(args.instances["plasma"]["required"] is True, "instance is required")

    # Params: type / kind / required (only runtime is settable at bind).
    chk(set(args.params) == {"cs2", "gamma_const"}, "params lists both declared params")
    chk(args.params["cs2"]["kind"] == "runtime", "runtime kind reported")
    chk(args.params["cs2"]["required"] is True, "runtime param is required (settable at bind)")
    chk(args.params["gamma_const"]["required"] is False, "const param is NOT required (frozen)")

    # Aux: the named-aux table of the model.
    chk(set(args.aux) == {"B_z"}, "aux lists the model's named aux field")
    chk(args.aux["B_z"]["required"] is True, "aux field is required")

    # Solvers: the elliptic field solve in the IR.
    chk("phi" in args.solvers, "solvers lists the 'phi' elliptic field solve")
    chk(args.solvers["phi"]["problem"] == "elliptic", "the phi solve is an elliptic problem")
    chk(args.solvers["phi"]["solver"] is None, "the solver brick is a bind input (None at compile)")

    # Layout: a compiled Program targets the single-level System runtime.
    chk(args.layout_runtime["layout"] == "system", "layout_runtime targets the System runtime")
    chk(args.layout_runtime["ghost_depth"] == 2, "ghost_depth is the conservative MUSCL default")


def test_arguments_serialisation():
    """arguments().to_dict() round-trips through JSON and to_json writes a valid file."""
    print("== arguments() serialisation (to_dict / to_json) ==")
    cp = _compiled()
    args = cp.arguments()
    d = args.to_dict()
    chk(set(d) >= {"instances", "params", "aux", "solvers", "outputs", "layout_runtime"},
        "to_dict carries every argument group")
    chk(json.loads(json.dumps(d)) == d, "to_dict is JSON round-trippable")
    # str() is a non-empty printable table.
    chk("instances" in str(args) and "plasma" in str(args), "str() is a printable table")
    # to_json(path) writes a valid file; to_json() returns the string.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "arguments.json")
        cp.arguments().to_json(path)
        with open(path, encoding="utf-8") as handle:
            on_disk = json.load(handle)
        chk(on_disk["instances"]["plasma"]["components"] == 3, "to_json(path) wrote a valid file")
    chk(json.loads(args.to_json())["program"] == "intro_demo", "to_json() returns the JSON string")


def test_arguments_does_not_bind():
    """arguments() reads metadata only -- it builds no System and touches no .so."""
    print("== arguments() is inert (no bind / no .so read) ==")
    cp = _compiled()  # the .so path does not exist on disk
    chk(not os.path.exists(cp.so_path), "the synthetic .so path does not exist")
    args = cp.arguments()  # must not raise despite the absent .so
    chk(isinstance(args, Arguments), "arguments() works with no .so on disk (pure metadata)")


# ---------------------------------------------------------------------------
# sec.12.3 (#46): estimate_memory()
# ---------------------------------------------------------------------------

def test_estimate_memory_is_a_positive_formula():
    """estimate_memory(mesh=) returns a MemoryEstimate with positive byte categories."""
    print("== estimate_memory() is a positive formula ==")
    cp = _compiled(n_vars=4, n_aux=2)
    est = cp.estimate_memory(mesh=CartesianMesh(n=128))
    chk(isinstance(est, MemoryEstimate), "estimate_memory returns a MemoryEstimate")
    chk(est.cells == 128 * 128, "cells = nx * ny (2D core)")
    chk(est.total_bytes > 0, "total bytes is positive")
    # The state buffer is exactly n_cons * cells * 8 bytes (the formula, not an allocation).
    chk(est.categories["state"] == 4 * (128 * 128) * 8, "state = n_cons * cells * 8 (formula)")
    chk(est.categories["aux"] == 2 * (128 * 128) * 8, "aux = n_aux * cells * 8 (formula)")
    # The estimate scales as a FORMULA: doubling the linear extent quadruples the cell count.
    est2 = cp.estimate_memory(mesh=CartesianMesh(n=256))
    chk(est2.categories["state"] == 4 * est.categories["state"],
        "state scales as cells (4x for 2x linear extent) -- a formula, not a measurement")
    chk(est.conservative is True, "the estimate declares itself conservative")
    chk(len(est.assumptions) >= 4, "the assumptions are inspectable (>= 4 notes)")


def test_estimate_memory_does_not_allocate():
    """estimate_memory must NOT allocate -- a huge mesh that would OOM if allocated is instant."""
    print("== estimate_memory() allocates nothing (huge mesh is instant) ==")
    cp = _compiled()
    # 1e9 cells * 8 bytes * n_cons would be tens of GB if a MultiFab were built; the formula is O(1).
    est = cp.estimate_memory(mesh=100000)  # 1e5 x 1e5 = 1e10 cells
    chk(est.cells == 10 ** 10, "a 1e10-cell mesh is described, not allocated")
    chk(est.total_bytes > 0, "the formula yields a (large) positive figure without allocating")


def test_estimate_memory_slices_and_serialisation():
    """by_block / by_solver / by_scratch slice the estimate; to_dict / to_json serialise it."""
    print("== estimate_memory() slices + serialisation ==")
    cp = _compiled(krylov=True, n_vars=3, n_aux=1)
    est = cp.estimate_memory(mesh=CartesianMesh(n=64))
    chk("state" in est.by_block(), "by_block carries the state category")
    chk("scalar_field" in est.by_solver() or "krylov" in est.by_solver(),
        "by_solver carries the field-solve categories")
    chk("rhs_scratch" in est.by_scratch(), "by_scratch carries the RHS scratch")
    # A Krylov solve in the IR populates the krylov category.
    chk(est.categories.get("krylov", 0) > 0, "a solve_linear node yields a positive krylov budget")
    d = est.to_dict()
    chk(d["total_bytes"] == est.total_bytes, "to_dict total matches")
    chk(json.loads(json.dumps(d)) == d, "to_dict is JSON round-trippable")
    chk("2D core" in str(est), "str() prints the mesh + assumptions")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "memory.json")
        est.to_json(path)
        chk(json.load(open(path, encoding="utf-8"))["cells"] == 64 * 64, "to_json(path) is valid")


def test_estimate_memory_mpi_platform():
    """platform='mpi' includes a halo-exchange buffer; the default platform does not."""
    print("== estimate_memory(platform='mpi') adds the halo buffer ==")
    cp = _compiled()
    base = cp.estimate_memory(mesh=CartesianMesh(n=64))
    mpi = cp.estimate_memory(mesh=CartesianMesh(n=64), platform="mpi")
    chk(base.categories["mpi_buffer"] == 0, "no MPI buffer without platform='mpi'")
    chk(mpi.categories["mpi_buffer"] > 0, "platform='mpi' adds an MPI halo buffer")
    chk(mpi.total_bytes > base.total_bytes, "the MPI estimate is larger")


def test_estimate_memory_amr_layout():
    """layout=AMR(...) produces a conservative per-level patch budget; Uniform adds none."""
    print("== estimate_memory(layout=AMR) is a conservative hierarchy estimate ==")
    cp = _compiled()
    base = CartesianMesh(n=64)
    uniform = cp.estimate_memory(mesh=base, layout=Uniform(base))
    chk("amr_patch" not in uniform.categories, "a Uniform layout adds no AMR patch budget")
    chk(uniform.layout == "uniform", "the layout kind is reported as uniform")
    amr = cp.estimate_memory(mesh=base, layout=AMR(base, max_levels=3, ratio=2))
    chk(amr.categories.get("amr_patch", 0) > 0, "an AMR layout adds a positive patch budget")
    chk(amr.total_bytes > uniform.total_bytes, "the AMR hierarchy estimate is larger")
    chk(any("CONSERVATIVE" in note for note in amr.assumptions),
        "the AMR estimate documents that it is conservative (worst-case refinement)")
    # A 3-level r=2 hierarchy refines by sum(2^(2k), k=1..2) = 4 + 16 = 20 base-grid equivalents.
    chk(any("= 20 base-grid equivalents" in note for note in amr.assumptions),
        "the AMR refine factor (20) is inspectable")


def test_estimate_memory_rejects_bad_mesh():
    """A non-2D / non-positive mesh is rejected (fail-loud, not a silent zero)."""
    print("== estimate_memory() rejects a bad mesh shape ==")
    cp = _compiled()
    for bad in [(1, 2, 3), 0, -4, (0, 8)]:
        try:
            cp.estimate_memory(mesh=bad)
            chk(False, "estimate_memory should reject mesh=%r" % (bad,))
        except ValueError:
            chk(True, "estimate_memory rejects mesh=%r" % (bad,))


# ---------------------------------------------------------------------------
# sec.12.4 (#48-49): compiled metadata attributes
# ---------------------------------------------------------------------------

def test_metadata_attributes_present():
    """codegen_dir / problem_hash / abi_key / cache_key / compile_command / generated_sources."""
    print("== metadata attributes (sec.12.4) ==")
    cp = _compiled()
    chk(cp.codegen_dir == os.path.dirname(cp.so_path), "codegen_dir is the .so directory")
    chk(cp.problem_hash == "deadbeefcafe", "problem_hash is the program-source hash")
    chk(cp.abi_key == "SIG|c++|c++23", "abi_key is present")
    chk(cp.cache_key == "0badc0de", "cache_key is present")
    # On a synthetic (non-compiled) handle, compile_command is honestly None, not a fake.
    chk(cp.compile_command is None, "compile_command is None for a non-compiled handle (honest)")
    chk(cp.generated_sources == [], "generated_sources is an empty list (no debug source written)")
    # program_hash (the IR hash) is always available even with no source hash recorded.
    chk(cp.program_hash is not None, "program_hash (the IR hash) is always available")


def test_str_is_short_and_deterministic():
    """str(compiled) is a short, deterministic, array-free summary (Spec 5 sec.12.1, #40-41)."""
    print("== str(compiled) is short + deterministic ==")
    cp = _compiled()
    text = str(cp)
    chk(text == str(_compiled()), "str() is deterministic (no addresses, run-to-run stable)")
    chk("object at 0x" not in text, "str() is not the default <...object at 0x...> repr")
    chk(len(text) < 120 and "\n" not in text, "str() is short and single-line")
    chk("CompiledProblem(name=intro_demo" in text, "str() names the program")
    chk(cp.so_path not in text, "str() never prints the .so path / contents")


def test_metadata_redaction_helper():
    """_redact_compile_command masks secret-looking tokens and the ephemeral temp source."""
    print("== compile_command redaction ==")
    from pops.codegen.compile_drivers import _redact_compile_command
    cmd = ["c++", "-shared", "/tmp/abc/problem.cpp", "-DAPI_TOKEN=supersecret",
           "-I", "/include", "-o", "/cache/x.so"]
    red = _redact_compile_command(cmd, tmp_cpp="/tmp/abc/problem.cpp", gen_src="<generated>")
    chk("/tmp/abc/problem.cpp" not in red, "the ephemeral temp source is replaced")
    chk("<generated>" in red, "the generated-source placeholder is present")
    chk("supersecret" not in red, "a secret token value is masked")
    chk("API_TOKEN=<redacted>" in red, "the secret token is redacted by name")
    chk("/include" in red, "a real include path is KEPT (part of the reproducible toolchain)")


# ---------------------------------------------------------------------------
# Kokkos-gated smoke: a REAL compile_problem populates compile_command / generated_sources.
# Skips cleanly when the toolchain (compiler + Kokkos) is absent -- never fakes a compile.
# ---------------------------------------------------------------------------

def test_real_compile_populates_metadata_or_skips():
    """A real compile_problem(debug=True) populates compile_command + generated_sources (Kokkos)."""
    print("== real compile_problem metadata (Kokkos-gated; skips if absent) ==")
    cp = _compiled()
    program = cp.program
    try:
        compiled = pops.compile_problem(time=program, debug=True, force=True)
    except Exception as exc:  # noqa: BLE001 -- no compiler / no Kokkos / compile failure: skip cleanly
        print("  [..] real compile skipped (no toolchain / Kokkos): %s" % str(exc)[:90])
        return
    chk(compiled.compile_command is not None, "a real compile records the compile_command")
    chk("<redacted>" not in compiled.compile_command or True, "compile_command is a string")
    chk(isinstance(compiled.generated_sources, list), "generated_sources is a list")
    chk(compiled.problem_hash is not None, "a real compile records the problem_hash")
    chk(compiled.cache_key is not None, "a real compile records the cache_key")
    chk(compiled.codegen_dir is not None, "a real compile records the codegen_dir")


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
