"""Mesh / geometry objects (Spec-4 PR-F).

The CHOICE of geometry lives in a MESH object, not in the scheme : pops.FiniteVolume stays
reconstruction + Riemann flux + variables (no geometry argument). The mesh is passed to the
system via pops.System(mesh=...). pops.CartesianMesh is the implicit default (square domain,
numerics STRICTLY unchanged, bit-identical). pops.PolarMesh describes a global ring (r, theta).
pops.AuxHalo declares a per-field aux ghost boundary policy.
"""


class CartesianMesh:
    """CARTESIAN mesh (implicit default) : square domain [0, L]^2, n x n cells.

    This is the historical geometry : pops.System(mesh=pops.CartesianMesh(n, L, periodic)) is STRICTLY
    equivalent (bit-identical) to pops.System(n=n, L=L, periodic=periodic). Provided for symmetry with
    pops.PolarMesh (the geometry choice is explicit on both sides)."""

    def __init__(self, n=64, L=1.0, periodic=True):
        self.n = int(n)
        self.L = float(L)
        self.periodic = bool(periodic)
    def _apply(self, config):
        config.geometry = "cartesian"
        config.n = self.n
        config.L = self.L
        config.periodic = self.periodic


class PolarMesh:
    """GLOBAL ANNULAR POLAR mesh ("polar diocotron grid" workstream, Phase 1): domain
    r in [r_min, r_max] x theta in [0, 2pi), nr x ntheta cells. theta is PERIODIC, r carries a
    PHYSICAL boundary condition (wall / outlet). Axis convention: direction 0 = radial,
    direction 1 = azimuthal (cf. PolarGeometry / assemble_rhs_polar on the C++ side).

    The Phase-0 prototype quantified that the Cartesian grid diffuses the RADIAL gradient of a ring in
    azimuthal rotation (ratio 73 vs polar): carrying the radial direction on a grid axis lifts
    this structural lock of the diocotron.

    SCOPE (audit update 2026-06): the polar path is WIRED into System.step (polar transport
    assemble_rhs_polar + polar Poisson + aux drift in the local basis (e_r, e_theta)).
    pops.System(mesh=pops.PolarMesh(...)) builds a global ring and advances on it. THREE levels not
    to confuse:

    - polar transport: scalar ExB AND isothermal fluid (IsothermalFluxPolar); Riemann flux
      'rusanov' (default, all transport) AND 'hll' (isothermal fluid only -- gated on model.wave_speeds,
      identical to the Cartesian one; scalar ExB does not provide wave_speeds -> 'hll' raises a
      clear rejection). 'hllc'/'roe' remain rejected on the C++ side (Euler 4 vars, no polar brick);
    - DIRECT polar Poisson (PolarPoissonSolver): single-rank, one box covering the ring;
    - TENSORIAL polar Schur stage (PolarCondensedSchurSourceStepper, via pops.Split/CondensedSchur):
      the C++ solver is multi-rank/multi-box (theta split).

    THETA SPLIT OF THE TRANSPORT (theta_boxes, ADC-67). theta_boxes=1 (default) = single-box,
    STRICTLY bit-identical to history. theta_boxes>1 = the ring is split into theta BANDS
    (each box covers the whole radius and one azimuthal band; theta_boxes must DIVIDE ntheta and
    stay <= ntheta) and the polar TRANSPORT (assemble_rhs_polar + collective fill_ghosts) runs
    multi-box. MATRIX of multi-box capabilities:

    - polar TRANSPORT (System transport, get/set state, eval_rhs, density): multi-box OK
      (per-box assembly + collective halos; the global state is reconstructed on read);
    - DIRECT polar Poisson (PolarPoissonSolver): SINGLE-BOX ONLY. A System with theta_boxes>1 that
      solves the direct field (solve_fields / step / potential, e.g. a coupled scalar ExB block)
      raises a clear UPSTREAM error (the direct solver requires complete theta rows + r columns on
      one box): use theta_boxes=1 OR the tensorial Schur stage;
    - tensorial polar Schur stage (pops.Split + pops.CondensedSchur): multi-box (multi-box C++
      solver; the theta split is now driven by theta_boxes).

    The DIRECT polar Poisson refuses MPI (single-rank). No Cartesian<->polar coupling (global
    ring). Optional step bounds (stability_speed/stability_dt/source_frequency) ARE wired on the
    polar path (default without trait = max_wave_speed, bit-identical). Cf. docs/GENERICITY_2026-06.md
    section 3 and pops.capabilities()['geometry'] / ['stability_policy']['system_polar']."""

    def __init__(self, r_min, r_max, nr, ntheta, theta_boxes=1):
        if not (r_max > r_min >= 0.0):
            raise ValueError("PolarMesh: requires r_max > r_min >= 0 (ring)")
        # nr >= 3: the radial drift of the aux (System.solve_fields_polar) uses a 2nd-order ONE-SIDED
        # stencil at both walls on phi (without ghost); nr < 3 would read phi out of bounds. A global
        # ring always has nr >= 3. ntheta >= 1 (the azimuthal drift wraps the periodic index).
        if nr < 3:
            raise ValueError("PolarMesh: nr >= 3 (2nd-order one-sided radial stencil at the walls)")
        if ntheta < 1:
            raise ValueError("PolarMesh: ntheta >= 1")
        # theta_boxes: split of the transport into theta bands (1 = single-box, default). We validate HERE
        # (Python side, clear message) AND on the C++ side (check_geometry, for a SystemConfig built by
        # hand): 1 <= theta_boxes <= ntheta AND theta_boxes DIVIDES ntheta (equal azimuthal bands).
        tb = int(theta_boxes)
        if tb < 1:
            raise ValueError("PolarMesh: theta_boxes >= 1 (1 = single-box)")
        if tb > int(ntheta):
            raise ValueError("PolarMesh: theta_boxes <= ntheta (at least one azimuthal cell per band)")
        if int(ntheta) % tb != 0:
            raise ValueError("PolarMesh: theta_boxes must DIVIDE ntheta (equal azimuthal bands)")
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.nr = int(nr)
        self.ntheta = int(ntheta)
        self.theta_boxes = tb

    def _apply(self, config):
        config.geometry = "polar"
        config.nr = self.nr
        config.ntheta = self.ntheta
        config.r_min = self.r_min
        config.r_max = self.r_max
        config.theta_boxes = self.theta_boxes
        config.n = self.nr  # n serves as the default size for the rest of the config (diagnostics)


class AuxHalo:
    """Per-field aux halo/ghost boundary policy, passed to set_aux_field(..., halo=pops.AuxHalo(...)).

    A model-NAMED aux field (m.aux_field(name)) normally inherits the SHARED aux ghost behavior derived
    from the potential phi (periodic preserved, otherwise zero-gradient). pops.AuxHalo lets that ONE
    field declare its own boundary policy instead, applied AFTER the shared fill to its component only:

    - ``kind='foextrap'`` : zero-gradient (ghost = mirror interior cell);
    - ``kind='dirichlet'`` : fixed boundary value (ghost = 2*value - interior), ``value`` the imposed value.

    The policy is applied UNIFORMLY to the NON-PERIODIC faces; periodic faces (a fully periodic domain,
    the polar theta direction) keep their wrap, so a per-field policy never breaks the domain's periodic
    structure. Works on System (Cartesian + polar) and the AMR coarse level. No halo (default) -> the
    shared aux BC, strictly bit-identical. Per-face asymmetric BC is a follow-up.
    """

    # Mirrors pops::BCType on the C++ side: Periodic=0, Foextrap=1, Dirichlet=2.
    _KINDS = {"foextrap": 1, "dirichlet": 2}

    def __init__(self, kind, value=0.0):
        if kind not in self._KINDS:
            raise ValueError("AuxHalo: kind must be 'foextrap' or 'dirichlet' (got %r)" % (kind,))
        self.kind = kind
        self.bc_type = self._KINDS[kind]
        self.value = float(value)

    def __repr__(self):
        return "AuxHalo(%r, value=%g)" % (self.kind, self.value)
