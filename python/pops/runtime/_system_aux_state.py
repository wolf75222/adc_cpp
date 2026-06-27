"""System aux/state mixin (Spec-4 PR-F): named aux fields, disc domain, primitive state.

Named-aux resolution + set/get, the disc transport-domain controls, and the primitive-variable
state helpers of :class:`pops.runtime.system.System`. Mixed in via inheritance; methods operate
on ``self._s`` and ``self._aux_field_index``.
"""


class _SystemAuxState:
    """Named aux + disc domain + primitive state methods of System."""

    def _resolve_aux_field(self, block, name):
        """Resolve (block, NAMED aux field name) -> canonical component of the aux channel (ADC-70 phase 1).
        Resolution rule: a CANONICAL name (phi/grad/B_z/T_e) is REJECTED here -- these fields have
        their dedicated paths (B_z -> set_magnetic_field, T_e -> set_electron_temperature_from, phi/grad
        derived by solve_fields). Otherwise look it up in the block table (filled at add_equation from
        the compiled model). Raises ValueError with an actionable message on unknown block/name."""
        from pops.physics.aux import AUX_CANONICAL  # late import (physics <-> __init__ cycle)
        if name == "B_z":
            raise ValueError(
                "set_aux_field: 'B_z' (magnetic field) is set via sim.set_magnetic_field(Bz), "
                "NOT via set_aux_field (B_z is a canonical aux field, not a named field).")
        if name == "T_e":
            raise ValueError(
                "set_aux_field: 'T_e' (electron temperature) is DERIVED from a fluid block via "
                "sim.set_electron_temperature_from(block), NOT set via set_aux_field.")
        if name in AUX_CANONICAL:
            raise ValueError(
                "set_aux_field: '%s' is a CANONICAL aux field (derived by the solver, not settable); "
                "set_aux_field only carries the NAMED fields declared by m.aux_field(...)." % name)
        table = self._aux_field_index.get(block)
        if table is None:
            raise ValueError(
                "set_aux_field: block '%s' unknown (or added without a named aux field); add the block "
                "via add_equation(model=...) with a model declaring m.aux_field('%s')." % (block, name))
        if name not in table:
            known = sorted(table) if table else "(none)"
            raise ValueError(
                "set_aux_field: aux field '%s' not declared by block '%s'; known named fields: %s"
                % (name, block, known))
        return table[name]

    def set_aux_field(self, block, name, field, halo=None):
        """Set a NAMED aux field (ADC-70 phase 1) of a block: @p name must have been declared by the
        model via m.aux_field(name) (and the block added via add_equation). @p field: 2D array (ny, nx)
        or flat (n*n), row-major. The field is STATIC (user-supplied, like B_z) and PERSISTS
        from one step to the next (solve_fields never rewrites named components). For B_z / T_e,
        use their dedicated paths (set_magnetic_field / set_electron_temperature_from).

        @p halo (ADC-369): an optional pops.AuxHalo declaring this field's own ghost boundary policy
        (foextrap / dirichlet), applied to the non-periodic faces after the shared aux fill. Default
        None inherits the shared aux BC (bit-identical)."""
        import numpy as np
        comp = self._resolve_aux_field(block, name)
        arr = np.asarray(field, dtype=float)
        self._s.set_aux_field_component(comp, arr.reshape(-1))
        if halo is not None:
            self._s.set_aux_field_halo_component(comp, halo.bc_type, halo.value)

    def aux_field(self, block, name):
        """Read a NAMED aux field (ADC-70 phase 1) of a block -> 2D array (ny, nx). Equals 0 everywhere as
        long as no set_aux_field has written it (aux channel initialized to zero, never rewritten by
        solve_fields beyond the derived components). @p name: declared by m.aux_field(name)."""
        import numpy as np
        comp = self._resolve_aux_field(block, name)
        return np.asarray(self._s.aux_field_component(comp), dtype=float)

    def set_disc_domain(self, cx, cy, R, mode="none"):
        """Set the TRANSPORT DOMAIN as a DISC of center (cx, cy) and radius R, and WIRE the
        transport according to mode= (T2 / T5-PR3 work). Materializes a 0/1 cell-centered mask (cell
        active when its center is inside the disc, level set hypot(x-cx, y-cy) - R < 0, SAME convention
        as the conducting wall of Poisson). This is the finite-volume counterpart of the elliptic wall: the paper
        (Hoffart et al., arXiv:2510.11808) transports on a REAL disc whereas PoPS transports on the
        full Cartesian square, the circle acting only in the Poisson wall (lock from the Cartesian ring
        edges, cf. docs/HOFFART_FIDELITY.md).

        The ``mode`` parameter wires the transport:

        - 'none' (default): the mask is materialized (queryable via disc_mask()) but the transport
          stays FULL Cartesian (assemble_rhs) -> step() BIT-IDENTICAL even with the disc set;
        - 'staircase': conservative masked transport (assemble_rhs_masked, 0/1 face gate);
        - 'cutcell': cut-cell / embedded-boundary transport (assemble_rhs_eb, apertures alpha_f +
          volume fraction kappa, smooth boundary, order 2 inside the disc).

        The mode is honored under Lie AND Strang (cf. Split / Strang). R > 0; Cartesian only (the
        polar one already bounds the ring by its radial walls -> explicit error)."""
        self._s.set_disc_domain(cx, cy, R, mode)

    def set_geometry_mode(self, mode):
        """Switch ONLY the disc transport mode ('none'|'staircase'|'cutcell') without (re)defining the
        disc. A mode != 'none' requires a disc already set (set_disc_domain) -> error otherwise. Setting
        back to 'none' restores the full Cartesian transport (bit-identical)."""
        self._s.set_geometry_mode(mode)

    def disc_mask(self):
        """0/1 cell-centered domain mask, array (ny, nx) (diagnostic / contract
        verification). All 1.0 as long as set_disc_domain has not been called (subdomain = whole
        domain, default path)."""
        return self._s.disc_mask()

    def set_primitive_state(self, name, **prims):
        """Initialize a block from its PRIMITIVE variables, named (rho/u/v/p ...):

            sim.set_primitive_state("electrons", rho=rho0, u=u0, v=v0, p=p0)

        Each primitive is an (n, n) array. The expected names are those of
        variable_names(name, "primitive") (the order of the block model). The (ncomp, n, n) array is
        assembled in that order, then CONVERTED to conservative variables by the block model (on the
        C++ side: compressible E = p/(g-1) + 1/2 rho|v|^2; isothermal rho u; scalar identity) and written
        to the state. Ergonomic counterpart of set_density (which only sets the density, leaving it at rest).

        Raises a clear error if a primitive name is unknown for the block, or if one is missing."""
        import numpy as np  # local: numpy is only required for this host assembly

        names = list(self._s.variable_names(name, "primitive"))
        n = self.nx()
        unknown = [k for k in prims if k not in names]
        if unknown:
            raise ValueError(
                "set_primitive_state: unknown primitive(s) %r for block '%s'; "
                "expected primitives: %r" % (unknown, name, names))
        missing = [k for k in names if k not in prims]
        if missing:
            raise ValueError(
                "set_primitive_state: missing primitive(s) %r for block '%s'; "
                "provide all the primitives: %r" % (missing, name, names))
        # Assemble (ncomp, n, n) in the model ORDER (primitive_vars), not the kwargs order.
        prim = np.empty((len(names), n, n), dtype=np.float64)
        for c, nm in enumerate(names):
            arr = np.asarray(prims[nm], dtype=np.float64)
            if arr.shape != (n, n):
                raise ValueError(
                    "set_primitive_state: primitive '%s' of shape %r, expected (%d, %d)"
                    % (nm, tuple(arr.shape), n, n))
            prim[c] = arr
        self._s.set_primitive_state(name, prim)

    def get_primitive_state(self, name):
        """Read the conservative state of a block and return it in PRIMITIVE variables (diagnostic):

            P = sim.get_primitive_state("electrons")   # {"rho": ..., "u": ..., "v": ..., "p": ...}

        Returns a dict {primitive_name: array (n, n)} in the order of variable_names(name,
        "primitive"). Inverse of set_primitive_state (exact round-trip to machine precision, the
        model cons <-> prim conversion being consistent)."""
        names = list(self._s.variable_names(name, "primitive"))
        prim = self._s.get_primitive_state(name)  # (ncomp, n, n)
        return {nm: prim[c] for c, nm in enumerate(names)}
