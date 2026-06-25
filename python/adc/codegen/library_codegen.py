"""adc.library_codegen -- the C++ emitter for a compiled brick library ``.so`` (Spec 3 section 21).

``adc.compile_library(..., emit=True)`` lowers a :class:`adc.library.LibraryManifest`
to a single C++ translation unit and compiles it (via the same Kokkos toolchain a
problem ``.so`` uses, :func:`adc.dsl.adc_loader_build_flags`). This module owns the
SOURCE emission; :func:`adc.library.compile_library` owns the compile + cache.

The emitted ``.so`` exports a stable, ABI-keyed descriptor of the library:

* ``adc_library_abi_key()`` -- the ``ADC_ABI_KEY_LITERAL`` of the TU (compiler + std +
  header signature + kokkos + stdlib). The library reader compares it against the loaded
  ``_adc`` module's ``abi_key()``; a mismatch is a HARD error (never a silent fallback);
* ``adc_library_name()`` / ``adc_library_backend()`` / ``adc_library_content_hash()`` --
  the manifest identity, so a compiled ``.so`` round-trips back into a ``LibraryManifest``;
* ``adc_library_brick_count()`` and per-brick string tables (id / type / category / scheme
  / native id / requirements JSON / capabilities JSON / available) -- the full brick list
  with its signatures, requirements and capabilities;
* ``adc_library_generated_symbol_*`` -- the generated brick ids the ``.so`` carries;
* ``adc_brick_manifest()`` -- the SAME JSON contract :func:`adc.lib.load_cpp_library`
  already reads, so the library ``.so`` is also a self-describing external-brick ``.so``,
  registered into ``BrickRegistry`` at static-init via ``ADC_REGISTER_BRICK``.

It compiles against ``external_brick.hpp`` + ``abi_key.hpp`` only -- no numerics, no
device kernels. A generated brick (e.g. an ``@adc.lib.solver`` solver) contributes its id
to the generated-symbol table; lowering its kernel body to a callable C++ symbol is the
deferred ADC-462 codegen follow-up and is recorded here as a metadata-only entry, never a
fabricated symbol.
"""
import json

__all__ = ["emit_library_cpp"]


def _str_table(accessor, values):
    """A ``const char* adc_library_<accessor>(int i)`` switch over @p values (JSON-escaped
    C string literals), returning ``""`` out of range. Mirrors the program codegen's table()
    so the read-back side is a uniform ``(count, name)`` string-table scan."""
    cases = "".join('    case %d: return %s;\n' % (i, json.dumps(v))
                    for i, v in enumerate(values))
    return ('extern "C" const char* adc_library_%s(int i) {\n'
            '  switch (i) {\n%s    default: return "";\n  }\n}\n' % (accessor, cases))


def _requirements_json(brick):
    """The brick's requirements as compact JSON (the wire form the reader parses back).

    Mirrors :meth:`adc.library.LibraryManifest` entries: a dict of named requirement lists
    (e.g. ``{"capabilities": ["physical_flux", "wave_speeds"]}``). Empty dict -> ``{}``."""
    return json.dumps(brick.get("requirements", {}), sort_keys=True)


def _capabilities_json(brick):
    """The brick's capabilities as compact JSON (the wire form the reader parses back)."""
    return json.dumps(brick.get("capabilities", {}), sort_keys=True)


def _manifest_csv(value_map, key):
    """Flatten a requirements/capabilities dict's lists into the CSV the external-brick
    manifest (``BrickRegistry`` / ``adc_brick_manifest``) uses. The native bricks carry
    ``{"capabilities": [...]}`` (required model capabilities) and the external bricks carry
    ``{"provides": [...]}``; both flatten to a flat token CSV so the existing
    :func:`adc.lib.load_cpp_library` parser reads them unchanged."""
    tokens = []
    for vals in value_map.values():
        if isinstance(vals, (list, tuple)):
            tokens.extend(str(v) for v in vals if str(v))
        elif vals:
            tokens.append(str(vals))
    return ",".join(tokens)


def _register_brick_lines(bricks):
    """``BrickRegistry::instance().register_brick({...})`` calls at static-init time, one per
    brick, so the library ``.so`` IS a self-describing external-brick ``.so`` (its bricks are
    catalogued the same way a hand-written external brick is). The id / category / requirement
    CSV / capability CSV mirror ``adc_brick_manifest`` below; registration order is the brick
    order."""
    lines = []
    for b in bricks:
        req_csv = _manifest_csv(b.get("requirements", {}), "capabilities")
        cap_csv = _manifest_csv(b.get("capabilities", {}), "provides")
        lines.append(
            "  ::adc::runtime::program::BrickRegistry::instance().register_brick(\n"
            "      {%s, %s, %s, %s});" % (
                json.dumps(b["id"]), json.dumps(b["category"]),
                json.dumps(req_csv), json.dumps(cap_csv)))
    return "\n".join(lines)


def _brick_manifest_json(name, bricks):
    """The JSON ``adc_brick_manifest()`` returns -- the SAME contract
    :func:`adc.lib.load_cpp_library` parses. ``{"library", "bricks": [{"id", "category",
    "requirements" CSV, "capabilities" CSV}, ...]}``. The ``library`` key is informational;
    the reader keys on ``bricks``."""
    entries = []
    for b in bricks:
        entries.append({
            "id": b["id"],
            "category": b["category"],
            "requirements": _manifest_csv(b.get("requirements", {}), "capabilities"),
            "capabilities": _manifest_csv(b.get("capabilities", {}), "provides"),
        })
    return json.dumps({"library": name, "bricks": entries}, sort_keys=True)


def emit_library_cpp(manifest):
    """C++ source of a compiled brick library ``.so`` for @p manifest (a
    :class:`adc.library.LibraryManifest`).

    The TU includes only ``external_brick.hpp`` (the ``BrickRegistry`` + ``ADC_REGISTER_BRICK``
    machinery) and ``abi_key.hpp`` (the ``ADC_ABI_KEY_LITERAL``); it has no numerics and no
    device code, so it compiles fast and is ABI-keyed exactly like a problem ``.so``. It
    exports the library identity, the full brick list (id / type / category / scheme / native
    id / requirements / capabilities / available), the generated-symbol table, and the
    ``adc_brick_manifest()`` JSON the external-brick loader reads.
    """
    bricks = list(manifest.bricks)
    gen = list(manifest.generated_symbols)
    manifest_json = _brick_manifest_json(manifest.name, bricks)
    return _LIBRARY_CPP_TEMPLATE.format(
        name=json.dumps(manifest.name),
        backend=json.dumps(manifest.backend),
        content_hash=json.dumps(manifest.content_hash),
        brick_count=len(bricks),
        gen_count=len(gen),
        manifest_json=json.dumps(manifest_json),  # the JSON STRING, escaped as a C literal
        register=_register_brick_lines(bricks),
        ids=_str_table("brick_id", [b["id"] for b in bricks]),
        types=_str_table("brick_type", [b["brick_type"] for b in bricks]),
        categories=_str_table("brick_category", [b["category"] for b in bricks]),
        schemes=_str_table("brick_scheme",
                           ["" if b["scheme"] is None else str(b["scheme"]) for b in bricks]),
        native_ids=_str_table("brick_native_id", [b["native_id"] for b in bricks]),
        availables=_str_table("brick_available",
                              ["1" if b["available"] else "0" for b in bricks]),
        requirements=_str_table("brick_requirements", [_requirements_json(b) for b in bricks]),
        capabilities=_str_table("brick_capabilities", [_capabilities_json(b) for b in bricks]),
        gen_symbols=_str_table("generated_symbol", gen))


# Source of a compiled brick library .so (ADC-464, Spec 3 section 21). It includes only the
# host-side brick registry + the ABI-key literal: no numerics, no device kernels. {name}/{backend}/
# {content_hash}/{manifest_json} are JSON-escaped C string literals; the literal braces of the C++
# scaffold are doubled for str.format.
_LIBRARY_CPP_TEMPLATE = '''\
// GENERATED by adc.library_codegen.emit_library_cpp (epic ADC-464, Spec 3 section 21). Do not edit.
// A compiled brick library: it exports an ABI-keyed descriptor (library identity + brick list +
// requirements/capabilities + generated symbols) read back by adc.read_library_manifest, and the
// adc_brick_manifest() JSON adc.lib.load_cpp_library consumes. No numerics, no device code.
#include <adc/runtime/program/external_brick.hpp>  // BrickRegistry / ADC_REGISTER_BRICK
#include <adc/runtime/dynamic/abi_key.hpp>          // ADC_ABI_KEY_LITERAL (ABI / Kokkos guard)

// --- library identity (round-trips back into a LibraryManifest) -----------------------------------
extern "C" const char* adc_library_abi_key() {{ return ADC_ABI_KEY_LITERAL; }}
extern "C" const char* adc_library_name() {{ return {name}; }}
extern "C" const char* adc_library_backend() {{ return {backend}; }}
extern "C" const char* adc_library_content_hash() {{ return {content_hash}; }}
extern "C" int adc_library_brick_count() {{ return {brick_count}; }}
extern "C" int adc_library_generated_symbol_count() {{ return {gen_count}; }}

// --- per-brick metadata tables (id / type / category / scheme / native id / reqs / caps / avail) --
{ids}{types}{categories}{schemes}{native_ids}{availables}{requirements}{capabilities}{gen_symbols}
// --- external-brick manifest (the JSON adc.lib.load_cpp_library reads) ----------------------------
extern "C" const char* adc_brick_manifest() {{ return {manifest_json}; }}

// Register every brick into the process-global BrickRegistry at static-init time, so this library
// .so is ALSO a self-describing external-brick .so (adc.lib.riemann.User(id) resolves after load).
namespace {{
const bool adc_library_registered = [] {{
{register}
  return true;
}}();
}}  // namespace
'''
