"""Spec 5 (sec.4 / 5 / 5.15 / 16): the central packages are top-level, not under pops.lib.

Spec 5 homes the generic building blocks in top-level packages and reserves ``pops.lib``
for ready-to-use presets (``lib.time`` / ``lib.models``). These checks assert that end
state structurally (source-only; they do not import ``pops`` / ``_pops``).
"""
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
POPS = REPO_ROOT / "python" / "pops"

# Spec 5 top-level central packages (sec.4 / 5).
CENTRAL_PACKAGES = (
    "numerics",      # discretisation descriptors (riemann/reconstruction/projections)
    "moments",       # moment-model toolkit
    "diagnostics",   # reduction catalog
    "mesh",          # mesh/layout/AMR descriptors
    "params",        # typed scalar params
    "output",        # output/checkpoint policies
    "external",      # compiled-brick references
    "fields",        # typed elliptic field-problem authoring (Spec 5 Phase E)
)

# Catalogs Spec 5 moves OUT of pops.lib (no longer their own modules under lib/).
MOVED_OUT_OF_LIB = ("riemann", "reconstruction", "moments", "diagnostics", "operators")


def test_central_packages_are_top_level():
    for pkg in CENTRAL_PACKAGES:
        init = POPS / pkg / "__init__.py"
        assert init.is_file(), "Spec 5 central package missing: python/pops/%s/__init__.py" % pkg


def test_shared_descriptor_module_is_top_level():
    assert (POPS / "descriptors.py").is_file(), (
        "Spec 5: the shared BrickDescriptor + Descriptor base must live in "
        "python/pops/descriptors.py")


def test_moved_catalogs_are_gone_from_lib():
    offenders = []
    for name in MOVED_OUT_OF_LIB:
        if (POPS / "lib" / name).exists():
            offenders.append("python/pops/lib/%s" % name)
    assert not offenders, (
        "Spec 5 sec.5.15: these central catalogs must move out of pops.lib:\n  "
        + "\n  ".join(offenders))


def test_lib_keeps_only_presets():
    # pops.lib keeps the presets (time, models). The transitional spatial/fields/solvers
    # catalogs are the Phase A2 carve-out (held for the in-flight install path) -- allowed
    # for now; everything else under lib must be a preset or that carve-out.
    allowed = {"time", "models", "spatial", "fields", "solvers",
               "__init__.py", "descriptors.py", "__pycache__"}
    unexpected = []
    for child in (POPS / "lib").iterdir():
        if child.name not in allowed and not child.name.endswith(".pyc"):
            unexpected.append(child.name)
    assert not unexpected, (
        "Spec 5 sec.5.15: pops.lib should hold only presets (time/models) + the Phase A2 "
        "carve-out (spatial/fields/solvers); unexpected: %s" % unexpected)
