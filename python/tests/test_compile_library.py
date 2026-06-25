"""Spec 3 adc.compile_library: the library manifest / ABI / descriptor layer.

``adc.compile_library`` collects generated/macro/native brick objects into a
reusable-library MANIFEST (name, abi_key, brick list, requirements, capabilities,
generated symbols) plus a stable content hash. It NEVER computes numerics and, in
this slice, NEVER emits a compiled ``.so`` -- that is the deferred ADC-464 C++
follow-up. These tests pin the manifest shape, the hash stability/sensitivity, the
round-trip through the reader, and the honest ``emit=True`` deferral.
"""
import pytest

adc = pytest.importorskip("adc")
lib = pytest.importorskip("adc.lib")


def _objects():
    """A small set of REAL adc.lib brick descriptors (no fakes)."""
    return [lib.solvers.GMRES(), lib.riemann.HLLC()]


# --- manifest shape --------------------------------------------------------
def test_compile_library_builds_a_manifest():
    man = adc.compile_library("my_numerics.so", objects=_objects())
    assert man.name == "my_numerics.so"
    assert man.backend == "production"
    assert isinstance(man.bricks, list) and len(man.bricks) == 2
    # The abi_key is the loaded-module ABI key (header sig + compiler + std).
    assert man.abi_key == adc.abi_key()
    # The content hash is a hex sha256 digest.
    assert isinstance(man.content_hash, str) and len(man.content_hash) == 64
    int(man.content_hash, 16)  # raises if not hex


def test_each_brick_carries_its_metadata():
    man = adc.compile_library("lib.so", objects=_objects())
    by_id = {b["id"]: b for b in man.bricks}
    assert set(by_id) == {"gmres", "hllc"}
    gmres = by_id["gmres"]
    assert gmres["brick_type"] == "native"
    assert gmres["category"] == "solver"
    assert gmres["scheme"] == "gmres"
    assert gmres["native_id"] == "adc::gmres_solve"
    assert "requirements" in gmres and "capabilities" in gmres
    hllc = by_id["hllc"]
    assert hllc["native_id"] == "adc::HLLCFlux"
    # HLLC declares its required model capabilities (from the lib descriptor).
    assert "physical_flux" in hllc["requirements"].get("capabilities", [])


def test_generated_symbols_collects_generated_bricks():
    @lib.solver(name="my_richardson", signature="(A, b)")
    def _build(ctx, a, b):
        x = ctx.zeros_like(b)
        return ctx.combine(x + b)

    man = adc.compile_library("lib.so", objects=[lib.solvers.custom("my_richardson")])
    assert man.bricks[0]["brick_type"] == "generated"
    # A generated brick contributes a symbol the (future) .so would export.
    assert "my_richardson" in man.generated_symbols


# --- content hash: stable + sensitive --------------------------------------
def test_content_hash_is_stable():
    a = adc.compile_library("lib.so", objects=_objects())
    b = adc.compile_library("lib.so", objects=_objects())
    assert a.content_hash == b.content_hash


def test_content_hash_is_sensitive_to_objects():
    base = adc.compile_library("lib.so", objects=[lib.solvers.GMRES()])
    more = adc.compile_library("lib.so", objects=[lib.solvers.GMRES(), lib.riemann.HLLC()])
    assert base.content_hash != more.content_hash


def test_content_hash_is_sensitive_to_name():
    a = adc.compile_library("a.so", objects=_objects())
    b = adc.compile_library("b.so", objects=_objects())
    assert a.content_hash != b.content_hash


def test_content_hash_is_order_insensitive():
    fwd = adc.compile_library("lib.so", objects=[lib.solvers.GMRES(), lib.riemann.HLLC()])
    rev = adc.compile_library("lib.so", objects=[lib.riemann.HLLC(), lib.solvers.GMRES()])
    assert fwd.content_hash == rev.content_hash


# --- round-trip through the reader -----------------------------------------
def test_manifest_round_trips_through_reader():
    man = adc.compile_library("lib.so", objects=_objects())
    restored = adc.read_library_manifest(man.to_dict())
    assert restored.name == man.name
    assert restored.abi_key == man.abi_key
    assert restored.content_hash == man.content_hash
    assert restored.bricks == man.bricks
    assert restored.to_dict() == man.to_dict()


def test_reader_rejects_a_corrupt_manifest():
    with pytest.raises((KeyError, ValueError, TypeError)):
        adc.read_library_manifest({"name": "lib.so"})  # missing required keys


# --- input validation ------------------------------------------------------
def test_empty_objects_is_rejected():
    with pytest.raises(ValueError):
        adc.compile_library("lib.so", objects=[])


def test_non_descriptor_object_is_rejected():
    with pytest.raises(TypeError):
        adc.compile_library("lib.so", objects=[object()])


def test_non_production_backend_is_rejected():
    with pytest.raises(ValueError):
        adc.compile_library("lib.so", objects=_objects(), backend="jit")


# --- honest deferral: emitting the .so raises ------------------------------
def test_emitting_the_so_raises_adc464():
    with pytest.raises(NotImplementedError) as exc:
        adc.compile_library("lib.so", objects=_objects(), emit=True)
    assert "ADC-464" in str(exc.value)


# --- compile_problem libraries= seam ---------------------------------------
def test_compile_problem_accepts_and_defers_libraries():
    dsl = pytest.importorskip("adc.dsl")  # numpy-backed; skip if absent
    time = pytest.importorskip("adc.time")
    man = adc.compile_library("lib.so", objects=_objects())
    prog = time.Program("p")
    # The manifest is validated (round-tripped) but the link is the deferred C++ follow-up.
    with pytest.raises(NotImplementedError) as exc:
        dsl.compile_problem(time=prog, libraries=[man])
    assert "ADC-464" in str(exc.value)


def test_compile_problem_rejects_a_corrupt_library():
    dsl = pytest.importorskip("adc.dsl")
    time = pytest.importorskip("adc.time")
    prog = time.Program("p")
    with pytest.raises((KeyError, ValueError, TypeError)):
        dsl.compile_problem(time=prog, libraries=[{"name": "bad.so"}])
