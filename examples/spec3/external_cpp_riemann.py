"""Spec 3 section 36 / criterion 20: an EXTERNAL C++ Riemann brick in the same type system.

An advanced user can ship a Riemann solver as a standalone ``.so`` and select it wherever a
native (``pops::HLLCFlux``) or generated brick is selected -- it is the fourth Spec 3 brick kind
(native / generated / macro / external-C++). The C++ side registers a manifest entry at static-init
time and exports an ``pops_brick_manifest()`` reader; the Python side dlopens it and surfaces an
``external_cpp`` :class:`pops.lib.BrickDescriptor`. This example builds an Euler-like board, selects
the external brick, and lowers the IR -- all pure Python. The compiled-run section (the real dlopen
+ the AOT Kokkos build) is guarded: it requires the brick ``.so`` and a matching toolchain (ROMEO).

The C++ an advanced user writes (built into ``my_riemann.so`` with the pops headers)::

    #include <pops/runtime/program/external_brick.hpp>

    struct MyRiemann {
      // The numerical flux kernel the codegen wires in (host/device). It reads the left/right
      // states and the physical flux, and returns the interface flux -- same signature contract
      // as the native pops::HLLCFlux. (Kernel body omitted; it is ordinary finite-volume code.)
    };

    // Register the brick id, its catalog slot, and the model capabilities it requires (CSV).
    // The 3-argument POPS_REGISTER_BRICK macro leaves the PROVIDES field empty.
    POPS_REGISTER_BRICK("my_riemann", "riemann", "pressure,wave_speeds");

    // The host the user links exports the manifest reader load_cpp_library() calls:
    extern "C" const char* pops_brick_manifest();  // JSON over BrickRegistry::instance()

Run: python3 examples/spec3/external_cpp_riemann.py
"""
import json
import os

import pops.lib as lib
from pops.physics import Model

# The manifest the user's ``pops_brick_manifest()`` would return for the brick above. The example
# falls back to registering it through the real ``pops.lib._register_manifest`` seam (the same code
# ``load_cpp_library`` runs after dlopen) so the authoring path is exercised WITHOUT a compiled
# ``.so``; it is never a fake descriptor.
_MANIFEST = json.dumps({"bricks": [{
    "id": "my_riemann",
    "category": "riemann",
    "requirements": "pressure,wave_speeds",
    "capabilities": "physical_flux",
}]})


def register_external_brick():
    """Register the external brick, preferring a real ``.so`` dlopen when one is available.

    If ``POPS_BRICK_SO`` points at a loadable brick library, :func:`pops.lib.load_cpp_library`
    dlopens it and reads its manifest for real. Otherwise the example registers the same manifest
    through the manifest-parsing seam (the path ``load_cpp_library`` itself runs after dlopen),
    so the descriptor and lowering are exercised on any host. Returns whether a real ``.so`` loaded.
    """
    so_path = os.environ.get("POPS_BRICK_SO")
    if so_path and os.path.exists(so_path):
        lib.load_cpp_library(so_path)
        return True
    lib._register_manifest(_MANIFEST)
    return False


def euler_with_external_riemann(riemann):
    """A role-tagged Euler board whose finite-volume rate uses the external @p riemann brick."""
    m = Model("euler")
    U = m.state("U", components=["rho", "mx", "my", "E"],
                roles={"rho": "density", "mx": "momentum_x", "my": "momentum_y", "E": "energy"})
    rho, mx, my, E = U
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    g = m.param("gamma", 1.4)
    p = m.primitive("p", (g - 1.0) * (E - 0.5 * (mx * mx + my * my) / rho))
    m.flux("F",
           x=[mx, mx * u + p, my * u, (E + p) * u],
           y=[my, mx * v, my * v + p, (E + p) * v])
    # The external brick is selected exactly like a native one (m.riemann("hllc")); it lives in
    # the SAME type system -- finite_volume_rate reads its descriptor scheme and records it.
    m.finite_volume_rate("advance", riemann=riemann)
    return m


def main():
    # 1) An unloaded id is rejected -- a descriptor is NEVER fabricated for an unregistered brick.
    try:
        lib.riemann.User("my_riemann")
    except LookupError as exc:
        print("unloaded id rejected:", str(exc).splitlines()[0])

    # 2) Register the brick (real .so dlopen when POPS_BRICK_SO is set, else the manifest seam).
    loaded_from_so = register_external_brick()
    print("brick source:", ".so dlopen" if loaded_from_so else "manifest seam (no .so on host)")

    # 3) ``riemann.User`` surfaces an external_cpp descriptor in the riemann category, carrying the
    # manifest's requirements/capabilities -- the same metadata shape as a native descriptor.
    riemann = lib.riemann.User("my_riemann")
    print("descriptor:", riemann, "type=%r category=%r" % (riemann.brick_type, riemann.category))
    print("native_id:", riemann.native_id, "requirements:", riemann.requirements)

    # 4) Build the board with the external brick and lower it -- pure Python, builds cleanly.
    m = euler_with_external_riemann(riemann)
    assert m.check()
    print("\noperator-first IR (the external brick is selected like a native one):")
    print(m.dump_module_ir())

    # 5) The compiled run (the real dlopen of the brick kernel + the AOT Kokkos build) needs the
    # brick .so AND a matching toolchain; honestly deferred to ROMEO rather than faked here.
    try:
        m.compile(backend="production")
        print("\ncompiled run: OK")
    except (RuntimeError, NotImplementedError, OSError) as exc:
        print("\ncompiled run requires the brick .so + a matching Kokkos AOT toolchain (ROMEO):")
        print("  ", str(exc).splitlines()[0])

    print("\nOK: the external C++ brick is one type system with native/generated/macro bricks.")


if __name__ == "__main__":
    main()
