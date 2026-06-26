"""Generic coupled inter-species source (explicit splitting).

:class:`CoupledSource` describes an ARBITRARY coupling between species as
FORMULAS, beyond the named couplings (ionization / collision / thermal exchange).
It reads fields ``(block, role)`` as INPUT and WRITES source terms given by
symbolic expressions; ``compile(backend)`` lowers each expr to postfix BYTECODE
(stack machine) that C++ evaluates per cell -- NO ``.so`` nor Python callback.

Import-graph rule (Spec 4): pure :mod:`pops.ir` + stdlib. The compile path is a
pure-Python bytecode emitter, so no codegen / ``_pops`` import is needed at all.
"""
from pops.ir import Var, _wrap, Expr, Const, Add, Sub, Mul, Div, Pow, Neg, Sqrt, sqrt  # noqa: F401
from pops.ir.visitors import _key  # noqa: F401

from .model import Param  # CoupledSource.param wraps a const Param (acyclic: model never imports this)


# --- Generic COUPLED inter-species source (P5 phase 1, EXPLICIT splitting) -----------------------
#
# pops.dsl.CoupledSource describes an ARBITRARY coupling between species as FORMULAS, beyond the named
# couplings (Ionization / Collision / ThermalExchange) which freeze a formula. We read fields (block,
# role) as INPUT and WRITE source terms (block, role) given by symbolic expressions
# (same Expr as the models DSL: +, *, -, /, **, sqrt, params). compile(backend) compiles each
# expr into postfix BYTECODE (stack machine) that C++ evaluates in the same for_each_cell device as the
# named couplings -- NO .so nor Python callback per cell. Applied AFTER transport (split).

# Inverse of pops::role_name: DSL role name (CamelCase, cf. CANONICAL_ROLES) -> canonical lowercase
# name expected by pops::role_from_name (C++ boundary). Single source of the correspondence.
_ROLE_TO_CANONICAL = {
    "Density": "density",
    "MomentumX": "momentum_x", "MomentumY": "momentum_y", "MomentumZ": "momentum_z",
    "Energy": "energy",
    "VelocityX": "velocity_x", "VelocityY": "velocity_y", "VelocityZ": "velocity_z",
    "Pressure": "pressure", "Temperature": "temperature", "Scalar": "scalar",
}


def _role_canonical(role):
    """Canonical role name (lowercase, C++ boundary) for a DSL role. Accepts already-canonical."""
    if role in _ROLE_TO_CANONICAL:
        return _ROLE_TO_CANONICAL[role]
    if role in _ROLE_TO_CANONICAL.values():
        return role
    raise ValueError("CoupledSource: unknown role %r (roles: %s)"
                     % (role, ", ".join(sorted(_ROLE_TO_CANONICAL))))


# Stack machine opcodes: MIRROR of pops::CsOp (coupled_source_program.hpp). FROZEN values
# (transported as-is by the Python -> C++ ABI).
_CS_PUSHREG = 0
_CS_ADD = 1
_CS_SUB = 2
_CS_MUL = 3
_CS_DIV = 4
_CS_NEG = 5
_CS_POW = 6
_CS_SQRT = 7

# FROZEN capacities, mirror of coupled_source_program.hpp (kCsMaxReg / kCsMaxTerms / kCsMaxProg). We
# diagnose on the Python side (clear error) before reaching the C++ boundary.
_CS_MAX_REG = 32
_CS_MAX_TERMS = 16
_CS_MAX_PROG = 256


class _CsField(Var):
    """Symbolic handle of a (block, role): it is a Var (hence a full Expr) whose environment NAME
    '<block>::<role>' indexes both the numpy eval (env) and the input register at the
    bytecode codegen. Subclassing Var directly gives operators / _wrap / to_cpp / eval / deps, so
    `+k * ne * ng` builds the expected Expr tree without delegation."""

    def __init__(self, block, role):
        super().__init__("%s::%s" % (block, role), "coupled_field")
        self.block = block
        self.role = role

    def __repr__(self): return "_CsField(%r, %r)" % (self.block, self.role)


class _CsBlock:
    """Construction helper: `src.block("electrons").role("density")` -> _CsField. Records the
    (block, role) requested on the source to fix the order of the input registers."""

    def __init__(self, src, name):
        self._src = src
        self.name = name

    def role(self, role):
        return self._src._field(self.name, role)


class CompiledCoupledSource:
    """Result of CoupledSource.compile(...): packages the FLAT ABI (bytecode) ready for
    System.add_coupled_source, + the REFERENCE numpy evaluator (same Expr) for tests / a Python
    integrator. No .so: the coupling is interpreted on the C++ side (device stack machine)."""

    def __init__(self, name, backend, in_blocks, in_roles, consts, out_blocks, out_roles,
                 prog_ops, prog_args, prog_lens, terms, reg_order, frequency=0.0,
                 freq_prog_ops=None, freq_prog_args=None, frequency_expr=None):
        self.name = name
        self.backend = backend
        self.frequency = float(frequency)  # mu [1/s] declared CONSTANT (0 = no constant bound)
        self.in_blocks = list(in_blocks)
        self.in_roles = list(in_roles)        # canonical (lowercase, C++ boundary)
        self.consts = list(consts)
        self.out_blocks = list(out_blocks)
        self.out_roles = list(out_roles)      # canonical
        self.prog_ops = list(prog_ops)
        self.prog_args = list(prog_args)
        self.prog_lens = list(prog_lens)
        # PER-CELL frequency mu(U): bytecode (same inputs/constants as the source). EMPTY =
        # constant frequency only. Transported to System/AmrSystem.add_coupled_source.
        self.freq_prog_ops = list(freq_prog_ops) if freq_prog_ops else []
        self.freq_prog_args = list(freq_prog_args) if freq_prog_args else []
        self._frequency_expr = frequency_expr  # reference Expr (numpy eval for tests); None if constant
        self._terms = list(terms)             # [(block, role_canonical, Expr)]: numpy reference
        self._reg_order = list(reg_order)     # env names '<block>::<role>' in register order

    def __repr__(self):
        return ("CompiledCoupledSource(name=%r, backend=%r, n_in=%d, n_const=%d, n_terms=%d)"
                % (self.name, self.backend, len(self.in_blocks), len(self.consts),
                   len(self.out_blocks)))

    def reference_terms(self, fields):
        """Evaluates the source terms on numpy arrays (REFERENCE for tests). @p fields:
        dict (block, role_canonical) -> array; returns [(block, role_canonical, dS)] with dS = S
        (the evaluated symbolic term), BEFORE multiplication by dt. Same Expr as the C++ codegen."""
        env = {}
        for (block, role), arr in fields.items():
            env["%s::%s" % (block, _role_canonical(role))] = arr
        return [(b, r, e.eval(env)) for (b, r, e) in self._terms]

    def reference_frequency(self, fields):
        """Evaluates the PER-CELL frequency mu(U) on numpy arrays (REFERENCE for tests):
        same Expr / register table as the C++ bytecode. @p fields: dict (block, role_canonical) ->
        array; returns the mu array (same formulas). Returns None if the frequency is CONSTANT
        (no Expr) -- use .frequency in that case."""
        if self._frequency_expr is None:
            return None
        env = {}
        for (block, role), arr in fields.items():
            env["%s::%s" % (block, _role_canonical(role))] = arr
        return self._frequency_expr.eval(env)


class CoupledSource:
    """Generic COUPLED inter-species source (pops.dsl), Phase 1 (EXPLICIT splitting). Reuses the
    Expr of the models DSL for the source formulas; no coupling is hard-coded.

        src = pops.dsl.CoupledSource("ionization")
        ne = src.block("electrons").role("density")
        ng = src.block("neutrals").role("density")
        k  = src.param("Kiz", 1.0)
        src.add("electrons", role="density", expr=+k * ne * ng)
        src.add("neutrals",  role="density", expr=-k * ne * ng)
        sim.add_coupling(src.compile(backend="production"))

    compile(backend) -> CompiledCoupledSource: flat ABI (bytecode) consumed by
    System.add_coupled_source (C++ side: stack machine evaluated in a for_each_cell device, MPI-safe,
    named functor). The backend (production / prototype) does NOT change the numerics: the bytecode is
    interpreted on the C++ side in both cases (no .so per coupling); it is kept for introspection
    and API parity with the models DSL."""

    def __init__(self, name="coupled_source"):
        self.name = name
        self._fields = {}    # '<block>::<role>' -> _CsField (single input register per (block, role))
        self._reg_order = []  # order of appearance of the input fields (-> register order)
        self._params = {}    # name -> Param
        self._terms = []     # [(block, role_canonical, Expr)]
        # Indices (in self._terms) of the terms EMITTED BY add_pair, by pair: [(idx_gain, idx_loss)].
        # add_pair guarantees that the two terms carry EXACTLY the same evaluated Expr, one +expr,
        # the other -expr (Neg) -> conservative exchange by construction. verify_conservation=True
        # revisits them at compile time to CHECK the property (and detect a breach on the manual add side).
        self._pairs = []
        self._frequency = 0.0  # mu [1/s] declared CONSTANT (coupling step bound; 0 = no bound)
        # optional PER-CELL mu(U): an Expr (same vocabulary as the terms: block().role() +
        # param()) emitted into bytecode at compile() against the SAME register table. None = constant.
        self._frequency_expr = None

    def frequency(self, mu):
        """Declared coupling FREQUENCY mu [1/s] (vague 3 audit): step bound dt <= cfl / mu
        aggregated by System/AmrSystem::step_cfl (reason 'coupled_source:<name>'). Couplings are
        applied ONCE per MACRO-step (splitting, apply_couplings(dt)): the bound applies to the
        macro-dt, WITHOUT a substeps/stride factor. mu <= 0 = no bound (historical).

        @p mu accepts TWO forms:
          - a number (float / int) -> CONSTANT frequency (historical path, bit-identical);
          - an Expr of the SAME vocabulary as the terms (fields block().role() + param()) ->
            PER-CELL frequency mu(U), emitted into bytecode at compile() and evaluated per cell on the
            C++ side (MAX + all_reduce_max -> dt <= cfl / max(mu)). The referenced fields MUST be
            declared via .block(...).role(...) (as for the terms); otherwise compile() raises an
            EXPLICIT ValueError (field used without .block(...).role(...)).

        Returns self (chainable)."""
        # An Expr/_CsField/Param -> per-cell frequency (bytecode); a scalar -> constant.
        if isinstance(mu, (int, float)) and not isinstance(mu, bool):
            self._frequency = float(mu)
            self._frequency_expr = None
        else:
            self._frequency_expr = _wrap(mu)  # Expr / _CsField / Param -> tree node (cf. _wrap)
            self._frequency = 0.0             # the bytecode carries the bound; no duplicate constant
        return self

    # --- symbolic construction ----------------------------------------------------------------
    def block(self, name):
        """Handle of a block: .role(role) derives a symbolic field (block, role) from it."""
        return _CsBlock(self, name)

    def _field(self, block, role):
        canon = _role_canonical(role)
        key = "%s::%s" % (block, canon)
        if key not in self._fields:
            f = _CsField(block, canon)
            self._fields[key] = f
            self._reg_order.append(key)
        return self._fields[key]

    def param(self, name, value):
        """NAMED constant parameter, usable like an Expr (inlines as a real in the bytecode)."""
        p = Param(name, value, kind="const")
        self._params[name] = p
        return p

    def add(self, block, role=None, expr=None):
        """Adds a source TERM: d_t (block.role) += expr. @p expr is an Expr / _CsField / Param /
        number. Several adds on the same (block, role) ADD UP (sum of source terms)."""
        if role is None:
            raise ValueError("CoupledSource.add: role= required")
        if expr is None:
            raise ValueError("CoupledSource.add: expr= required")
        e = expr if isinstance(expr, Expr) else _wrap(expr)  # _CsField / Var are already Expr; Param/scalar -> _wrap
        self._terms.append((block, _role_canonical(role), e))
        return self

    def add_pair(self, block_a, block_b, role=None, expr=None):
        """Adds a CONSERVATIVE EXCHANGE of the quantity @p role between @p block_a and @p block_b, described
        by a SINGLE expression @p expr (Expr / _CsField / Param / number).

        Sign convention (to remember): @p block_a GAINS +expr, @p block_b LOSES -expr, on the SAME
        evaluated value of @p expr. In other words:

            d_t (block_a.role) += +expr
            d_t (block_b.role) += -expr

        It is the DSL equivalent of the NAMED C++ couplings (add_collision / add_thermal_exchange), which
        compute ONE value and apply it with two opposite signs: the sum over the two blocks of the
        exchanged term is zero at each cell and at each step, so the total quantity sum(role) over
        (block_a, block_b) is CONSERVED by construction -- independently of the chosen formula, the dt
        and the state. Choose @p expr >= 0 for a transfer from B to A (A gains, B loses); a negative
        sign of expr simply reverses the transfer direction (conservation holds in all cases).

        Contrast with two hand-written .add(...) (+expr on A, -expr on B): add_pair guarantees that
        the TWO legs carry the SAME Expr (the second is exactly Neg of the first), whereas by hand
        nothing prevents writing two slightly different formulas by mistake -> conservation broken
        silently. add_pair removes this risk; compile(verify_conservation=True) also checks it
        for hand-written couplings.

        @p block_a and @p block_b must be distinct. add_pair is purely ADDITIVE on top of .add:
        the manual API stays available and unchanged. Returns self (chainable)."""
        if role is None:
            raise ValueError("CoupledSource.add_pair: role= required")
        if expr is None:
            raise ValueError("CoupledSource.add_pair: expr= required")
        if block_a == block_b:
            raise ValueError("CoupledSource.add_pair: block_a and block_b must be distinct "
                             "(received %r for both)" % (block_a,))
        canon = _role_canonical(role)
        gain = expr if isinstance(expr, Expr) else _wrap(expr)  # +expr (gaining leg)
        loss = Neg(gain)                                        # -expr: SAME subtree, opposite sign
        idx_gain = len(self._terms)
        self._terms.append((block_a, canon, gain))
        idx_loss = len(self._terms)
        self._terms.append((block_b, canon, loss))
        self._pairs.append((idx_gain, idx_loss))
        return self

    # --- bytecode codegen -----------------------------------------------------------------------
    def _emit_program(self, expr, reg_index):
        """Compile @p expr (Expr tree) into postfix bytecode (parallel ops/args lists) against the
        register table @p reg_index (env name '<bloc>::<role>' OR constant value -> register index).
        Constants (Const / inline Param) become a dedicated constant register. Postfix traversal
        (recursion over the tree structure): a Var pushes its register, a binary emits its two
        subtrees then the opcode, etc. -- exactly the semantics of CsProgram::eval on the C++ side."""
        ops, args = [], []

        def emit(node):
            if isinstance(node, Var):
                if node.name not in reg_index:
                    raise ValueError("CoupledSource: field %r used without .block(...).role(...)"
                                     % node.name)
                ops.append(_CS_PUSHREG)
                args.append(reg_index[node.name])
            elif isinstance(node, Const):
                ops.append(_CS_PUSHREG)
                args.append(self._const_reg(node.value, reg_index))
            elif isinstance(node, Neg):
                emit(node.a)
                ops.append(_CS_NEG)
                args.append(0)
            elif isinstance(node, Sqrt):
                emit(node.a)
                ops.append(_CS_SQRT)
                args.append(0)
            elif isinstance(node, Add):
                emit(node.a)
                emit(node.b)
                ops.append(_CS_ADD)
                args.append(0)
            elif isinstance(node, Sub):
                emit(node.a)
                emit(node.b)
                ops.append(_CS_SUB)
                args.append(0)
            elif isinstance(node, Mul):
                emit(node.a)
                emit(node.b)
                ops.append(_CS_MUL)
                args.append(0)
            elif isinstance(node, Div):
                emit(node.a)
                emit(node.b)
                ops.append(_CS_DIV)
                args.append(0)
            elif isinstance(node, Pow):
                emit(node.a)
                emit(node.b)
                ops.append(_CS_POW)
                args.append(0)
            else:
                raise TypeError("CoupledSource: expression node not supported in Phase 1: %r "
                                "(supported: +, -, *, /, **, unary -, sqrt, field, constant)"
                                % type(node).__name__)

        emit(expr)
        return ops, args

    def _const_reg(self, value, reg_index):
        """Register index of a constant @p value (deduplicated). Constants occupy the
        registers AFTER the input fields (cf. CoupledSourceKernel: r[n_in + c] = consts[c])."""
        key = ("const", float(value))
        if key not in reg_index:
            reg_index[key] = len(self._reg_order) + len(self._consts)
            self._consts.append(float(value))
        return reg_index[key]

    @staticmethod
    def _signed_key(expr):
        """SIGNED STRUCTURAL key of @p expr: (sign, body_key), where sign is +1 / -1 and
        body_key is the structural key (_key) of the expression stripped of ALL its leading Neg
        (the sign folds in each peeled Neg). Two expressions are structurally OPPOSITE iff they have
        the SAME body_key and opposite signs (e.g. E and Neg(E), or -E and Neg(-E)=E). Peeling
        ALL leading Neg makes the key robust to the +expr / -expr pair from add_pair EVEN when expr is
        already a Neg (add_pair sets loss = Neg(gain), hence one more Neg). We do NOT normalize the
        internal algebra (k*ne vs ne*k): at worst a false 'non-conservative' on differently written
        forms, NEVER a false 'conservative' (the check stays conservative, hence sound)."""
        sign = 1
        while isinstance(expr, Neg):
            sign = -sign
            expr = expr.a
        return (sign, _key(expr))

    def _verify_conservation(self):
        """Verify that, role by role, the sum of the source terms CANCELS structurally: each
        contribution +E on one block is compensated by a contribution -E (same structural body) on
        another block. Raises an EXPLICIT ValueError otherwise. This is exactly the property add_pair
        guarantees by construction; this check extends it to hand-written couplings (two .add) and detects
        a break (slightly different formulas, forgotten sign, orphan term). Purely symbolic
        (no numerical evaluation): same structural key as the codegen CSE."""
        from collections import Counter
        per_role = {}
        for (_block, role, expr) in self._terms:
            sign, body = self._signed_key(expr)
            # Signed counter per structural body: +1 for +E, -1 for -E. Everything cancels => conservative.
            c = per_role.setdefault(role, Counter())
            c[body] += sign
        offenders = []
        for role in sorted(per_role):
            for body, net in per_role[role].items():
                if net != 0:
                    offenders.append((role, body, net))
        if offenders:
            details = "; ".join(
                "role '%s': term %r not compensated (net=%+d)" % (role, body, net)
                for (role, body, net) in offenders)
            raise ValueError(
                "CoupledSource.compile(verify_conservation=True): NON-conservative coupling. "
                "Each contribution +E on one block must be compensated by -E (same expression) "
                "on another block (use add_pair to guarantee it). Uncompensated terms: "
                + details)

    def compile(self, backend="production", verify_conservation=False):
        """Compile the source into a CompiledCoupledSource (flat bytecode ABI). @p backend documents
        the intent (API parity with the model DSL); the numerics are identical (C++ interpreter).

        @p verify_conservation (opt-in, default False): SYMBOLIC check that the coupling conserves
        each quantity (role) -- the sum of the source terms of a same role cancels structurally
        (each +E compensated by a -E on another block). add_pair satisfies this property by
        construction; this mode extends it to hand-written couplings (two .add) and raises an EXPLICIT
        ValueError if a term is not compensated (divergent formula, forgotten sign, orphan term).
        Off by default: a deliberately NON-conservative coupling (net creation/destruction, e.g.
        ionization creating an e/i pair) stays legal without passing the flag."""
        if not self._terms:
            raise ValueError("CoupledSource.compile: no term (.add(...) required)")
        if verify_conservation:
            self._verify_conservation()
        # Register table: input fields first (order of appearance), constants next.
        reg_index = {key: i for i, key in enumerate(self._reg_order)}
        self._consts = []
        prog_ops, prog_args, prog_lens = [], [], []
        out_blocks, out_roles = [], []
        for (block, role, expr) in self._terms:
            ops, args = self._emit_program(expr, reg_index)
            if len(ops) > _CS_MAX_PROG:
                raise ValueError("CoupledSource: program of term (%s.%s) too long (%d > %d)"
                                 % (block, role, len(ops), _CS_MAX_PROG))
            prog_ops += ops
            prog_args += args
            prog_lens.append(len(ops))
            out_blocks.append(block)
            out_roles.append(role)
        # Optional PER-CELL FREQUENCY: its program is emitted AFTER the terms, against the SAME
        # register table (reg_index) and the SAME constant list (self._consts) -- the referenced fields
        # must be declared via .block().role() (otherwise _emit_program raises: field used
        # without .block(...).role(...)). The frequency's own constants are appended after those of the
        # terms; on the C++ side they occupy the same registers r[n_in ..] (CoupledFreqKernel loads them
        # like the source). Constant frequency (or none) -> empty program (historical path).
        freq_prog_ops, freq_prog_args = [], []
        if self._frequency_expr is not None:
            freq_prog_ops, freq_prog_args = self._emit_program(self._frequency_expr, reg_index)
            if len(freq_prog_ops) > _CS_MAX_PROG:
                raise ValueError("CoupledSource: frequency program too long (%d > %d)"
                                 % (len(freq_prog_ops), _CS_MAX_PROG))
        n_reg = len(self._reg_order) + len(self._consts)
        if n_reg > _CS_MAX_REG:
            raise ValueError("CoupledSource: too many registers (inputs + constants = %d > %d)"
                             % (n_reg, _CS_MAX_REG))
        if len(out_blocks) > _CS_MAX_TERMS:
            raise ValueError("CoupledSource: too many source terms (%d > %d)"
                             % (len(out_blocks), _CS_MAX_TERMS))
        in_blocks = [self._fields[key].block for key in self._reg_order]
        in_roles = [self._fields[key].role for key in self._reg_order]
        return CompiledCoupledSource(
            name=self.name, backend=backend, in_blocks=in_blocks, in_roles=in_roles,
            consts=list(self._consts), out_blocks=out_blocks, out_roles=out_roles,
            prog_ops=prog_ops, prog_args=prog_args, prog_lens=prog_lens,
            terms=self._terms, reg_order=self._reg_order, frequency=self._frequency,
            freq_prog_ops=freq_prog_ops, freq_prog_args=freq_prog_args,
            frequency_expr=self._frequency_expr)
