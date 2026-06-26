"""Spec 3 section 21 / ADC-464: compile a reusable brick library to a real ``.so``.

``pops.compile_library("my_numerics.so", objects=[...], backend="production", emit=True)``
collects a set of brick descriptors (native / generated / external) into a library and
compiles a REAL ``.so`` with the same Kokkos toolchain a problem ``.so`` uses. The ``.so``
exports an ABI-keyed descriptor: the library name, the ABI key, the brick list (id / type /
category / scheme / native id / requirements / capabilities / available) and the generated
symbols, plus the ``pops_brick_manifest()`` JSON the external-brick loader reads.

``pops.read_library_manifest("my_numerics.so")`` dlopens the ``.so`` and reads that descriptor
back, with a HARD ABI / Kokkos guard (a library compiled against another toolchain is rejected,
never silently loaded). ``pops.compile_problem(..., libraries=["my_numerics.so"])`` reads +
validates it (the consume path).

This script always prints the manifest (pure Python). The real emit + compile + read-back is
guarded on Kokkos being visible (``export POPS_KOKKOS_ROOT=/path/to/kokkos``); without it the
compile section is skipped honestly rather than faked.

Run: python3 examples/spec3/compile_library_so.py
     POPS_KOKKOS_ROOT=/opt/homebrew/anaconda3 python3 examples/spec3/compile_library_so.py
"""
import os
import tempfile

import pops
import pops.lib as lib


def build_manifest():
    """A small reusable library: a native GMRES solver and a native HLLC Riemann brick."""
    objects = [lib.solvers.GMRES(), lib.riemann.HLLC()]
    return pops.compile_library("my_numerics.so", objects=objects, backend="production")


def print_manifest(man):
    print("library:", man.name, "backend=%r" % man.backend)
    print("abi_key:", man.abi_key)
    print("content_hash:", man.content_hash[:16], "...")
    print("bricks:")
    for b in man.bricks:
        print("  - %-14s type=%-9s category=%-12s native_id=%s"
              % (b["id"], b["brick_type"], b["category"], b["native_id"] or "(none)"))
    print("generated_symbols:", man.generated_symbols or "(none)")


def main():
    # 1) Build the manifest (pure Python, no compiler) and print it.
    man = build_manifest()
    print("== library manifest (pure Python) ==")
    print_manifest(man)

    # 2) Emit + compile a real .so (Kokkos-gated). adc_cpp is Kokkos-only, so the .so MUST be
    #    compiled with Kokkos; point POPS_KOKKOS_ROOT at an installed Kokkos.
    print("\n== compile a real library .so (needs Kokkos) ==")
    with tempfile.TemporaryDirectory() as tmp:
        so = os.path.join(tmp, "my_numerics.so")
        try:
            compiled = pops.compile_library("my_numerics.so", objects=[
                lib.solvers.GMRES(), lib.riemann.HLLC()],
                backend="production", emit=True, so_path=so)
        except (RuntimeError, OSError) as exc:
            print("compile requires Kokkos + a matching toolchain (set POPS_KOKKOS_ROOT):")
            print("  ", str(exc).splitlines()[0])
            print("\nOK: the manifest layer ran; the real .so compile is honestly skipped here.")
            return

        print("compiled:", compiled.so_path)

        # 3) Read the descriptor BACK from the compiled .so (dlopen + ABI guard).
        back = pops.read_library_manifest(so)
        print("\n== descriptor read back from the .so ==")
        print_manifest(back)
        assert back.content_hash == compiled.content_hash, "the .so carries the source hash"
        assert back.abi_key == pops.abi_key(), "the .so ABI key matches the loaded module"

        # 4) The library .so is ALSO a self-describing external-brick .so: load_cpp_library reads
        #    its pops_brick_manifest() JSON and registers the ids in the in-process catalog.
        lib._clear_external_catalog()
        n = lib.load_cpp_library(so)
        print("\nadc.lib.load_cpp_library registered %d bricks; riemann.User('hllc') -> %r"
              % (n, lib.riemann.User("hllc")))

    print("\nOK: a reusable brick library was emitted, compiled, and read back across the ABI.")


if __name__ == "__main__":
    main()
