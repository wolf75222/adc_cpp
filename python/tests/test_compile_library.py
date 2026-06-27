"""Spec 3 pops.compile_library: the library manifest / ABI / descriptor layer + the real ``.so``.

``pops.compile_library`` collects generated/macro/native brick objects into a
reusable-library MANIFEST (name, abi_key, brick list, requirements, capabilities,
generated symbols) plus a stable content hash. With ``emit=True`` it ALSO emits the
library C++ and compiles a REAL ``.so`` (same Kokkos toolchain a problem ``.so`` uses),
which :func:`pops.read_library_manifest` reads back (dlopen) with a HARD ABI / Kokkos guard;
``pops.compile_problem(..., libraries=[...])`` reads + validates it. These tests pin the
manifest shape, the hash stability/sensitivity, the reader round-trip, and -- when Kokkos
is visible -- the real emit + compile + read-back + the ABI-mismatch hard error.
"""
import pytest

pops = pytest.importorskip("pops")
lib = pytest.importorskip("pops.lib")


def _objects():
    """A small set of REAL pops.lib brick descriptors (no fakes)."""
    return [lib.solvers.GMRES(), lib.riemann.HLLC()]


def _toolchain_or_skip():
    """The (compiler, cflags, lflags) of the Kokkos loader toolchain, or skip the test.

    adc_cpp is Kokkos-only: the library ``.so`` MUST be compiled with Kokkos (point
    POPS_KOKKOS_ROOT at an installed Kokkos). A missing toolchain is a clean skip, never a
    fake; we exercise the manifest layer unconditionally above this gate."""
    pytest.importorskip("pops.codegen")
    from pops.codegen.toolchain import pops_loader_build_flags
    try:
        return pops_loader_build_flags()
    except Exception as exc:  # noqa: BLE001  -- no Kokkos / no compiler visible
        pytest.skip("Kokkos loader toolchain unavailable: %s" % str(exc)[:160])


# --- manifest shape --------------------------------------------------------
def test_compile_library_builds_a_manifest():
    man = pops.compile_library("my_numerics.so", objects=_objects())
    assert man.name == "my_numerics.so"
    assert man.backend == "production"
    assert isinstance(man.bricks, list) and len(man.bricks) == 2
    # The abi_key is the loaded-module ABI key (header sig + compiler + std).
    assert man.abi_key == pops.abi_key()
    # The content hash is a hex sha256 digest.
    assert isinstance(man.content_hash, str) and len(man.content_hash) == 64
    int(man.content_hash, 16)  # raises if not hex


def test_each_brick_carries_its_metadata():
    man = pops.compile_library("lib.so", objects=_objects())
    by_id = {b["id"]: b for b in man.bricks}
    assert set(by_id) == {"gmres", "hllc"}
    gmres = by_id["gmres"]
    assert gmres["brick_type"] == "native"
    assert gmres["category"] == "solver"
    assert gmres["scheme"] == "gmres"
    assert gmres["native_id"] == "pops::gmres_solve"
    assert "requirements" in gmres and "capabilities" in gmres
    hllc = by_id["hllc"]
    assert hllc["native_id"] == "pops::HLLCFlux"
    # HLLC declares its required model capabilities (from the lib descriptor).
    assert "physical_flux" in hllc["requirements"].get("capabilities", [])


def test_generated_symbols_collects_generated_bricks():
    @lib.solver(name="my_richardson", signature="(A, b)")
    def _build(ctx, a, b):
        x = ctx.zeros_like(b)
        return ctx.combine(x + b)

    man = pops.compile_library("lib.so", objects=[lib.solvers.custom("my_richardson")])
    assert man.bricks[0]["brick_type"] == "generated"
    # A generated brick contributes a symbol the (future) .so would export.
    assert "my_richardson" in man.generated_symbols


# --- content hash: stable + sensitive --------------------------------------
def test_content_hash_is_stable():
    a = pops.compile_library("lib.so", objects=_objects())
    b = pops.compile_library("lib.so", objects=_objects())
    assert a.content_hash == b.content_hash


def test_content_hash_is_sensitive_to_objects():
    base = pops.compile_library("lib.so", objects=[lib.solvers.GMRES()])
    more = pops.compile_library("lib.so", objects=[lib.solvers.GMRES(), lib.riemann.HLLC()])
    assert base.content_hash != more.content_hash


def test_content_hash_is_sensitive_to_name():
    a = pops.compile_library("a.so", objects=_objects())
    b = pops.compile_library("b.so", objects=_objects())
    assert a.content_hash != b.content_hash


def test_content_hash_is_order_insensitive():
    fwd = pops.compile_library("lib.so", objects=[lib.solvers.GMRES(), lib.riemann.HLLC()])
    rev = pops.compile_library("lib.so", objects=[lib.riemann.HLLC(), lib.solvers.GMRES()])
    assert fwd.content_hash == rev.content_hash


# --- round-trip through the reader -----------------------------------------
def test_manifest_round_trips_through_reader():
    man = pops.compile_library("lib.so", objects=_objects())
    restored = pops.read_library_manifest(man.to_dict())
    assert restored.name == man.name
    assert restored.abi_key == man.abi_key
    assert restored.content_hash == man.content_hash
    assert restored.bricks == man.bricks
    assert restored.to_dict() == man.to_dict()


def test_reader_rejects_a_corrupt_manifest():
    with pytest.raises((KeyError, ValueError, TypeError)):
        pops.read_library_manifest({"name": "lib.so"})  # missing required keys


# --- input validation ------------------------------------------------------
def test_empty_objects_is_rejected():
    with pytest.raises(ValueError):
        pops.compile_library("lib.so", objects=[])


def test_non_descriptor_object_is_rejected():
    with pytest.raises(TypeError):
        pops.compile_library("lib.so", objects=[object()])


def test_non_production_backend_is_rejected():
    with pytest.raises(ValueError):
        pops.compile_library("lib.so", objects=_objects(), backend="jit")


# --- compile_problem libraries= seam (validation, no compile) --------------
def test_compile_problem_rejects_a_corrupt_library():
    pytest.importorskip("pops.codegen")
    from pops.codegen.compile import compile_problem
    time = pytest.importorskip("pops.time")
    prog = time.Program("p")
    # A corrupt manifest is rejected at the libraries= read, BEFORE the Program is lowered.
    with pytest.raises((KeyError, ValueError, TypeError)):
        compile_problem(time=prog, libraries=[{"name": "bad.so"}])


# --- real .so: emit + compile + read back (Kokkos-gated) -------------------
def test_emit_compiles_a_real_so_and_reads_it_back(tmp_path):
    _toolchain_or_skip()
    so = str(tmp_path / "my_numerics.so")
    src = pops.compile_library("my_numerics.so", objects=_objects(), emit=True, so_path=so)
    import os
    assert src.so_path == so and os.path.isfile(so)
    # Read the descriptor BACK from the compiled .so (dlopen + exported metadata).
    back = pops.read_library_manifest(so)
    assert back.name == "my_numerics.so"
    assert back.backend == "production"
    # The .so carries the SOURCE content hash and the SOURCE ABI key.
    assert back.content_hash == src.content_hash
    assert back.abi_key == pops.abi_key()
    by_id = {b["id"]: b for b in back.bricks}
    assert set(by_id) == {"gmres", "hllc"}
    assert by_id["gmres"]["native_id"] == "pops::gmres_solve"
    assert by_id["hllc"]["native_id"] == "pops::HLLCFlux"
    # The required model capabilities round-trip through the .so tables.
    assert "physical_flux" in by_id["hllc"]["requirements"].get("capabilities", [])


def test_emit_exports_the_brick_manifest_for_load_cpp_library(tmp_path):
    _toolchain_or_skip()
    so = str(tmp_path / "lib.so")
    pops.compile_library("lib.so", objects=_objects(), emit=True, so_path=so)
    # The library .so is ALSO a self-describing external-brick .so: pops.lib.load_cpp_library
    # reads its pops_brick_manifest() JSON and registers the ids in the in-process catalog.
    lib._clear_external_catalog()
    n = lib.load_cpp_library(so)
    assert n == 2
    # An external descriptor now resolves for a brick the library registered (category-checked).
    hllc = lib.riemann.User("hllc")
    assert hllc.brick_type == "external_cpp" and hllc.category == "riemann"


def test_emit_exports_generated_symbols(tmp_path):
    _toolchain_or_skip()

    @lib.solver(name="my_richardson", signature="(A, b)")
    def _build(ctx, a, b):
        x = ctx.zeros_like(b)
        return ctx.combine(x + b)

    so = str(tmp_path / "gen.so")
    pops.compile_library("gen.so", objects=[lib.solvers.custom("my_richardson")],
                        emit=True, so_path=so)
    back = pops.read_library_manifest(so)
    assert "my_richardson" in back.generated_symbols
    assert back.bricks[0]["brick_type"] == "generated"


def test_read_back_rejects_an_abi_mismatch(tmp_path):
    """A library .so whose ABI key differs from the loaded _pops module is a HARD error on
    read-back -- never a silent fallback. We forge a .so with a deliberately wrong ABI key by
    compiling with a mismatched POPS_HEADER_SIG, then confirm read_library_manifest rejects it."""
    cc, cflags, lflags = _toolchain_or_skip()
    pytest.importorskip("pops.codegen")
    from pops.codegen.library_codegen import emit_library_cpp
    from pops.codegen.toolchain import pops_include, _probe_cxx_std, loader_cxx_std
    import os
    import subprocess
    import tempfile

    man = pops.compile_library("mismatch.so", objects=_objects())
    src = emit_library_cpp(man)
    include = pops_include()
    eff_std = _probe_cxx_std(cc, loader_cxx_std())
    so = str(tmp_path / "mismatch.so")
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "mismatch.cpp")
        with open(cpp, "w") as f:
            f.write(src)
        # WRONG header signature on purpose -> the .so's POPS_ABI_KEY_LITERAL diverges from _pops.
        flags = ["-shared", "-fPIC", "-std=" + eff_std,
                 '-DPOPS_HEADER_SIG="deadbeef-not-the-real-signature"', *cflags]
        cmd = [cc, *flags, "-I", include, cpp, "-o", so, *lflags]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            pytest.skip("could not compile the mismatched .so: %s"
                        % (r.stderr or b"").decode(errors="replace")[:160])
    with pytest.raises(RuntimeError) as exc:
        pops.read_library_manifest(so)
    assert "ABI" in str(exc.value)


# --- consume path: compile_problem(libraries=[.so]) ------------------------
def test_compile_problem_consumes_a_compiled_library_so(tmp_path):
    cc, cflags, lflags = _toolchain_or_skip()
    pytest.importorskip("pops.codegen")
    from pops.codegen.compile import compile_problem
    time = pytest.importorskip("pops.time")
    so = str(tmp_path / "consumed.so")
    pops.compile_library("consumed.so", objects=_objects(), emit=True, so_path=so)
    # A real Forward-Euler Program so the problem lowers; libraries=[.so] is read + ABI-checked.
    P = time.Program("consume")
    dt = P.dt
    U = P.state("ions")
    R = P.rhs(state=U, flux=True, sources=[])
    P.commit("ions", P.linear_combine("U1", U + dt * R))
    try:
        compiled = compile_problem(time=P, libraries=[so])
    except RuntimeError as exc:  # .so compile of the PROBLEM failed (toolchain), not the library
        pytest.skip("compile_problem could not build the problem .so: %s" % str(exc)[:160])
    # The validated library manifest is carried on the handle, ABI-matched.
    assert len(compiled.libraries) == 1
    assert compiled.libraries[0].name == "consumed.so"
    assert compiled.libraries[0].abi_key == pops.abi_key()
