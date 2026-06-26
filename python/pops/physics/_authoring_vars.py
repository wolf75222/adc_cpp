"""Authoring mixin: variable / primitive / aux declarations.

Split out of the monolithic :class:`~pops.physics.model.HyperbolicModel` so no
single file exceeds the Spec-4 500-line bound. The mixin holds only methods; the
instance attributes they touch (``cons_names`` / ``prim_defs`` / ``aux_names``
/ ``aux_extra_names`` / ``prim_state`` / ``cons_from`` / ``prim_roles``) are
created by ``HyperbolicModel.__init__`` (see :mod:`pops.physics.model`).

Imports only :mod:`pops.ir` and the package-level aux constants: this layer is
codegen-free and ``_pops``-free (Spec-4 import-graph rule).
"""
from pops.ir import Var, _wrap

from .aux import AUX_CANONICAL, AUX_NAMED_MAX, aux_total_n_aux


class _VariablesMixin:
    """Conservative / primitive / auxiliary variable declarations."""

    def cons(self, name):
        self.cons_names.append(name)
        return Var(name, "cons")

    def conservative_vars(self, *names, roles=None):
        """Declare the conservative variables. @p roles (optional): list of the same length explicitly
        setting the physical role of each component (string 'Density'/'MomentumX'... or None
        to fall back on the canonical mapping of the name); useful for a non-standard layout where the names
        do not suffice to deduce the meaning. Without roles, the canonical name -> role mapping applies."""
        if roles is not None and len(roles) != len(names):
            raise ValueError("conservative_vars: %d roles for %d variables" % (len(roles), len(names)))
        self.cons_roles = list(roles) if roles is not None else None
        return tuple(self.cons(n) for n in names)

    def primitive(self, name, expr):
        """Define a primitive by its formula (in terms of the cons / previous primitives)."""
        self.prim_defs[name] = _wrap(expr)
        return Var(name, "prim")

    def aux(self, name):
        """CANONICAL auxiliary field (e.g. grad_x, grad_y, B_z, T_e) provided at execution. The name
        MUST be a key of AUX_CANONICAL. For an arbitrary NAMED field, see aux_field."""
        self.aux_names.append(name)
        return Var(name, "aux")

    def aux_field(self, name):
        """NAMED auxiliary field (ADC-70 phase 1) provided at execution per block via
        System.set_aux_field(bloc, name, array). Unlike aux(...) (CANONICAL components
        phi/grad/B_z/T_e), name is ARBITRARY: the k-th call reserves component
        AUX_NAMED_BASE + k of the aux channel (read in C++ via aux.extra_field(k)). Returns a Var
        usable in flux / source / eigenvalues / elliptic_rhs like any other aux variable.

        At most AUX_NAMED_MAX named fields per model (FIXED bound on the C++ side, Aux POD). A name already
        canonical (B_z, T_e, phi...) is REJECTED: those fields have their dedicated paths (aux('B_z') +
        set_magnetic_field, etc.); a duplicate named name is also rejected."""
        # The name becomes a C++ LOCAL in the generated formula (cf. _aux_locals_lines) AND the key of the
        # facade table: it must be a valid C++ identifier (letters/digits/_, not a
        # leading digit). Explicit rejection rather than a .so that does not compile.
        if not (isinstance(name, str) and name.isidentifier()):
            raise ValueError("aux_field(%r): invalid name (C++ identifier expected: "
                             "letters/digits/_, without a leading digit)" % (name,))
        if name in AUX_CANONICAL:
            raise ValueError(
                "aux_field('%s') : '%s' is a CANONICAL aux field; use aux('%s') (and the "
                "dedicated path, e.g. set_magnetic_field for B_z, set_electron_temperature_from "
                "for T_e)" % (name, name, name))
        if name in self.aux_extra_names:
            raise ValueError("aux_field('%s') : field already declared" % name)
        if len(self.aux_extra_names) >= AUX_NAMED_MAX:
            raise ValueError("aux_field('%s') : at most %d named aux fields per model "
                             "(kAuxMaxExtra bound on the C++ side)" % (name, AUX_NAMED_MAX))
        self.aux_extra_names.append(name)
        return Var(name, "aux")

    def _aux_locals_lines(self):
        """C++ locals for the aux fields read in a formula: canonical '<n>' <- a.<n> ;
        named '<n>' <- a.extra_field(k) (k = position in aux_extra_names). The local name is
        IDENTICAL to the one the Expr emits (Var.to_cpp), so the formula references it directly."""
        lines = ["    const pops::Real %s = a.%s;" % (n, n) for n in self.aux_names]
        lines += ["    const pops::Real %s = a.extra_field(%d);" % (n, k)
                  for k, n in enumerate(self.aux_extra_names)]
        return lines

    def _reads_aux(self):
        """True if a formula reads an aux field (canonical or named): drives the naming of the Aux
        parameter ('a' vs anonymous) so as not to trigger an unused-parameter warning."""
        return bool(self.aux_names) or bool(self.aux_extra_names)

    def _total_n_aux(self):
        """TOTAL width of the model's aux channel (canonical + named fields)."""
        return aux_total_n_aux(self.aux_names, self.aux_extra_names)

    def set_primitive_state(self, *vars_or_names, roles=None):
        """Declares the ORDERED layout of the primitive state (Prim): component names, in order.
        Necessary for the brick codegen (to_primitive fills Prim in this order). Each name must
        be a conservative variable or an already-defined primitive. @p roles (optional): same
        convention as conservative_vars (explicit per-component override, None = canonical mapping)."""
        self.prim_state = [v.name if isinstance(v, Var) else str(v) for v in vars_or_names]
        if roles is not None and len(roles) != len(self.prim_state):
            raise ValueError("set_primitive_state : %d roles for %d variables"
                             % (len(roles), len(self.prim_state)))
        self.prim_roles = list(roles) if roles is not None else None

    def set_conservative_from(self, exprs):
        """Formulas of the conservative state as a function of the primitives (one per conservative
        variable, in conservative_vars order). Used to generate to_conservative: the DSL cannot invert
        the primitives symbolically, so the user provides the inverse explicitly."""
        self.cons_from = [_wrap(e) for e in exprs]

    @property
    def n_vars(self): return len(self.cons_names)
