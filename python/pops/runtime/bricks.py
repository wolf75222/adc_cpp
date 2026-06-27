"""Composable bricks + scheme/time policies (Spec-4 PR-F facade).

Public façade that re-exports the value objects split across ``_bricks_model`` (state /
transport / source / elliptic bricks + Model composer + hybrid composition + EPM),
``_bricks_scheme`` (couplings + Spatial / FiniteVolume + Explicit) and ``_bricks_time`` (the
implicit / split time policies + Role + the mask helpers). Split into three modules to satisfy
the <=500-line cap ; this module keeps ``from pops.runtime.bricks import *`` working as the
single import point. ``abi_key`` (the module ABI key on the extension) is re-exported here too,
surfaced through the runtime layer (it comes from ``pops._bootstrap``).
"""

from pops._bootstrap import abi_key

from pops.runtime._bricks_model import (
    Scalar, FluidState, ExB, CompressibleFlux, IsothermalFlux,
    NoSource, PotentialForce, GravityForce, MagneticLorentzForce, PotentialMagneticForce,
    ChargeDensity, BackgroundDensity, GravityCoupling,
    Model, CompositeModel, _native_to_brick,
    DivEpsGrad, CompositeRhs, ChargeDensitySource, ElectricFieldFromPotential, EllipticModel,
    div_eps_grad, charge_density, composite_rhs, electric_field_from_potential, elliptic,
    EllipticSolver,
)
from pops.runtime._bricks_scheme import (
    Ionization, Collision, ThermalExchange,
    Spatial, FiniteVolume, Explicit,
)
from pops.runtime._bricks_time import (
    _role_to_stable, _norm_implicit,
    IMEX, SourceImplicit, SourceImplicitBE, IMEXRK, Implicit, Role,
    CondensedSchur, Split, Strang,
)

__all__ = [
    "abi_key",
    "Scalar", "FluidState", "ExB", "CompressibleFlux", "IsothermalFlux",
    "NoSource", "PotentialForce", "GravityForce", "MagneticLorentzForce", "PotentialMagneticForce",
    "ChargeDensity", "BackgroundDensity", "GravityCoupling",
    "Model", "CompositeModel", "_native_to_brick",
    "DivEpsGrad", "CompositeRhs", "ChargeDensitySource", "ElectricFieldFromPotential",
    "EllipticModel", "div_eps_grad", "charge_density", "composite_rhs",
    "electric_field_from_potential", "elliptic", "EllipticSolver",
    "Ionization", "Collision", "ThermalExchange",
    "Spatial", "FiniteVolume", "Explicit", "_role_to_stable", "_norm_implicit",
    "IMEX", "SourceImplicit", "SourceImplicitBE", "IMEXRK", "Implicit", "Role",
    "CondensedSchur", "Split", "Strang",
]
