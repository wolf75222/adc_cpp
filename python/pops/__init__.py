"""pops : Python bindings for the adc_cpp library.

The core exposes generic compiled BRICKS (transport, source, elliptic right-hand
side) ; a MODEL is a composition of bricks, named on the application side. Python
composes the bricks (objects), the cell-by-cell computation stays in compiled C++ (no
numpy, GPU/MPI preserved).

    import pops
    sim = pops.System(n=192, periodic=False)
    sim.add_block(
        "ne",
        model=pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0),
                        source=pops.NoSource(), elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0)),
        spatial=pops.Spatial(minmod=True), time=pops.Explicit())
    sim.set_poisson(bc="dirichlet", wall="circle", wall_radius=0.40)
    sim.set_density("ne", ne_numpy)
    sim.step_cfl(0.4)

The scenario names (diocotron, electron_euler...) are compositions on the
application side (see adc_cases). No scenario name here.
"""
# Load the _pops extension (RTLD_GLOBAL so the DSL production .so resolves C++ symbols).
from pops import _bootstrap  # noqa: F401  (import side effect: loads _pops with the right flags)
from pops._bootstrap import (SystemConfig, ModelSpec, _System,  # noqa: F401
                             AmrSystemConfig, _AmrSystem, abi_key)
from pops._version import __version__  # noqa: F401

# Runtime layer (the ONLY importer of _pops): systems, parallelism, doctor, mesh, bricks, host flux.
from pops.runtime.system import System, AmrSystem  # noqa: F401
from pops.runtime.threading import set_threads, has_kokkos, parallel_info  # noqa: F401
from pops.runtime.doctor import doctor, capabilities  # noqa: F401
from pops.runtime.mesh import CartesianMesh, PolarMesh, AuxHalo  # noqa: F401
from pops.runtime.python_flux import PythonFlux  # noqa: F401
from pops.runtime.bricks import (  # noqa: F401
    Scalar, FluidState, ExB, CompressibleFlux, IsothermalFlux,
    NoSource, PotentialForce, GravityForce, MagneticLorentzForce, PotentialMagneticForce,
    ChargeDensity, BackgroundDensity, GravityCoupling,
    Model, CompositeModel, _native_to_brick,
    DivEpsGrad, CompositeRhs, ChargeDensitySource, ElectricFieldFromPotential, EllipticModel,
    div_eps_grad, charge_density, composite_rhs, electric_field_from_potential, elliptic,
    EllipticSolver,
    Ionization, Collision, ThermalExchange,
    Spatial, FiniteVolume, Explicit, _role_to_stable, _norm_implicit,
    IMEX, SourceImplicit, SourceImplicitBE, IMEXRK, Implicit, Role,
    CondensedSchur, Split, Strang,
)

__all__ = [
    "System", "SystemConfig", "AmrSystem", "AmrSystemConfig", "Model", "CompositeModel",
    "CartesianMesh", "PolarMesh", "AuxHalo",
    "Scalar", "FluidState", "ExB", "CompressibleFlux", "IsothermalFlux",
    "NoSource", "PotentialForce", "GravityForce", "MagneticLorentzForce", "PotentialMagneticForce",
    "ChargeDensity", "BackgroundDensity", "GravityCoupling",
    "Spatial", "FiniteVolume", "Explicit", "IMEX", "IMEXRK", "SourceImplicit", "SourceImplicitBE",
    "Implicit", "Split", "Strang", "CondensedSchur", "Role", "integrate",
    "elliptic", "div_eps_grad", "charge_density", "composite_rhs",
    "electric_field_from_potential", "EllipticSolver", "EllipticModel",
    "Ionization", "Collision", "ThermalExchange",
    "PythonFlux", "time", "model", "math", "physics", "lib", "mesh",
    "params", "output", "external", "fields", "linalg", "solvers",
    "abi_key", "capabilities", "inspect_capabilities", "inspect_amr",
    "set_threads", "has_kokkos", "parallel_info", "doctor",
    "compile_problem", "CompiledProblem", "CompiledTime",
    "compile_library", "read_library_manifest", "LibraryManifest",
]


# Lower / authoring layers + the moved integrate (re-exported, surface unchanged).
from pops.runtime import integrate  # noqa: E402,F401  (pops.integrate name preserved; without numpy)
from . import time  # noqa: E402  (pops.time.Program IR; pure stdlib, no numpy/_pops dependency)
from . import model  # noqa: E402  (pops.model operator-first type system; pure stdlib, Spec 2)
from . import math  # noqa: E402  (pops.math board operators; pure stdlib, Spec 3, dsl lazy)
from . import lib  # noqa: E402  (pops.lib typed-brick descriptor catalog; pure stdlib, Spec 3)
from . import physics  # noqa: E402  (pops.physics board model authoring; numpy-free import, Spec 3)
from . import mesh  # noqa: E402  (pops.mesh typed mesh/layout/AMR descriptors; pure stdlib, Spec 5)
from . import params  # noqa: E402  (pops.params typed scalar params; pure stdlib, Spec 5)
from . import output  # noqa: E402  (pops.output typed output/checkpoint policies; pure stdlib, Spec 5)
from . import external  # noqa: E402  (pops.external compiled-brick references; pure stdlib, Spec 5)
from . import fields  # noqa: E402  (pops.fields typed elliptic field-problem authoring; pure stdlib, Spec 5)
from . import linalg  # noqa: E402  (pops.linalg abstract algebra: names A x = b; pure stdlib, Spec 5)
from . import solvers  # noqa: E402  (pops.solvers linear/nonlinear/elliptic solver catalog; pure stdlib, Spec 5)
from .codegen.library import (  # noqa: E402,F401  (re-export: brick-library manifest API, Spec 3 section 21)
    LibraryManifest, compile_library, read_library_manifest)
from .time import CompiledTime  # noqa: E402,F401  (re-export: compiled-Program time policy)
from ._capabilities import inspect_capabilities, inspect_amr  # noqa: E402,F401  (Spec 5: descriptor-sourced matrix + AMR report)


# LAZY pops.compile_problem / pops.CompiledProblem (PEP 562): the codegen engine pulls numpy at
# import (host evaluator of the prototype IR), whereas the native path (System/add_block) and the
# production backend do not need it. Exposing these top-level names LAZILY keeps `import pops`
# numpy-free until the DSL/compile path is first used; numpy's absence then gives a targeted message
# (doctor too).
def __getattr__(name):
    if name == "compile_problem":
        from .codegen.compile import compile_problem
        return compile_problem
    if name == "CompiledProblem":
        from .codegen.loader import CompiledProblem
        return CompiledProblem
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
