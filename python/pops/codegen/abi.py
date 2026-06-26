"""pops.codegen.abi : ABI guards for compiled DSL .so files.

Extracted verbatim from pops.dsl (bodies byte-for-byte); only import lines adjusted.
Public API re-exported from pops.codegen.__init__.
"""

from .toolchain import pops_header_signature, _pops_module, pops_include  # noqa: F401


def module_header_signature():
    """Header signature BAKED into the loaded _pops module (token headers= of abi_key()), or None
    if the module is not loadable / the key is absent ("unknown" from a manual build -> None too)."""
    mod = _pops_module()
    abi = getattr(mod, "abi_key", None) if mod is not None else None
    if not callable(abi):
        return None
    try:
        key = abi()
    except Exception:
        return None
    for tok in str(key).split(";"):
        if tok.startswith("headers="):
            sig = tok[len("headers="):]
            return sig if sig and sig != "unknown" else None
    return None


def check_compiled_matches_module(abi_key):
    """PRE-DLOPEN guard at the WIRING of a CompiledModel (add_equation -> add_native_block).

    INDISPENSABLE COMPLEMENT of _check_headers_match_module : on a cache HIT, compile_native does not
    run (the .so comes out of the cache) and the compilation guard therefore does not protect this path --
    a stale _pops module would dlopen the .so and fail on the cryptic 'symbol not found'.
    Here we compare the header signature EMBEDDED in the Python key of the .so ('<sig>|<cxx>|<std>',
    cf. _abi_key_python, recomputed on the CURRENT include tree at each compile()) with the one baked
    into the module. Pure STRING comparison (no re-hash). No-op if either of the two is missing
    (old module, CompiledModel built by hand)."""
    baked = module_header_signature()
    if baked is None or not abi_key:
        return
    so_sig = str(abi_key).split("|", 1)[0]
    if so_sig and so_sig != baked:
        raise RuntimeError(
            "pops : the compiled model (.so) was produced against pops headers DIFFERENT from those "
            "of the loaded _pops module (signature %s... vs %s... baked in the module).\n"
            "Typical cause : stale _pops module (built before a `git pull`) while the .so has just "
            "been (re)compiled on the up-to-date headers -- the dlopen would fail with a cryptic "
            "'symbol not found'.\n"
            "Remedy : REBUILD the module then re-run :\n"
            "  cmake --preset python && cmake --build --preset python\n"
            "Full diagnostic : python -c \"import pops; pops.doctor()\"."
            % (so_sig[:12], baked[:12]))


def _abi_key_python(include, cxx, std):
    """ABI key on the Python side, MIRROR of pops::detail::abi_key_string (compiler + standard +
    header signature). Makes the verification + diagnostic available on the Python side BEFORE
    loading the .so (the native path compares its own on the C++ side). Stable and readable form:
    "<header sig>|<cxx>|<std>". include absent -> empty signature (degraded diagnostic, no UB)."""
    import os
    sig = pops_header_signature(include) if include and os.path.isdir(include) else ""
    return "%s|%s|%s" % (sig, cxx or "", std or "")
