"""pops.problem -- the top-level compilable assembly (Spec 5 sec.5.16 / sec.11).

:class:`Problem` is the inert, typed top-level assembly a user authors before lowering:
a mesh ``layout``, one or more physics ``block`` declarations, the elliptic ``field``
problems, the runtime ``param`` declarations, the static ``aux`` inputs, the ``output``
policies and the ``time`` scheme. Like :class:`pops.fields.FieldProblem` it is a
:class:`pops.descriptors.Descriptor`: it declares its requirements / capabilities /
options and answers ``available(context)`` / :meth:`validate` with an EXPLAINABLE status
before the runtime is ever touched. It computes nothing.

``pops.compile(problem, time=...)`` lowers the assembly through the EXISTING codegen
(``compile_problem``) and ``pops.bind(compiled, ...)`` wires it onto the EXISTING runtime
(``System`` / ``AmrSystem`` via the unified ``install``). The :class:`Problem` here owns no
codegen and no runtime of its own; the heavy ``.so`` compile + install + run path is
Kokkos-gated and validated on CI / ROMEO. Every not-yet-wired route fails LOUD (a clear
``NotImplementedError``), never silently.
"""
from pops.descriptors import Availability, Descriptor
from pops.fields import FieldProblem
from pops.mesh.cartesian import CartesianMesh
from pops.mesh.layouts import AMR, Uniform

# Field names the default native Poisson route already serves (Spec 3 / the install seam:
# pops.runtime._system_unified_install._install_solver). A field problem named anything else
# needs the deferred multi-elliptic runtime, so the assembly refuses it at validate().
_POISSON_FIELD_NAMES = ("phi", "poisson", "charge_density", "default")


class _AMRPolicyHandle:
    """The ``problem.amr`` refinement-policy handle (only valid for an ``AMR`` layout).

    A thin authoring shim that records the refinement criteria on the problem's ``AMR``
    layout descriptor and returns the :class:`Problem` so it chains. It owns no AMR runtime;
    the policies it stores (``pops.mesh.amr.Refine`` / ``TagUnion`` / ``RegridEvery`` ...) are
    inert descriptors the deferred AMR route will materialise.
    """

    def __init__(self, problem):
        self._problem = problem

    @property
    def _layout(self):
        return self._problem._layout

    def refine(self, criterion=None, *, regrid=None, nesting=None, patches=None):
        """Record the refinement criterion / regrid / nesting / patch policies (chains)."""
        layout = self._layout
        if criterion is not None:
            layout.refine = criterion
        if regrid is not None:
            layout.regrid = regrid
        if nesting is not None:
            layout.nesting = nesting
        if patches is not None:
            layout.patches = patches
        return self._problem


class Problem(Descriptor):
    """A typed, inert top-level assembly: layout + blocks + fields + params + aux + outputs.

    ``Problem(layout=Uniform(CartesianMesh()), name="plasma")`` then chained::

        prob = (pops.Problem(name="plasma")
                .block("ne", physics=model, spatial=pops.FiniteVolume())
                .field(pops.fields.FieldProblem(unknown=phi, equation=eq, solver=mg))
                .time(pops.time.Program(...)))

    Each assembly setter RETURNS the problem so calls chain. The default ``layout`` is a
    single-level :class:`~pops.mesh.layouts.Uniform` over a default
    :class:`~pops.mesh.cartesian.CartesianMesh`. :meth:`validate` runs structural checks and
    raises a LOUD ``NotImplementedError`` for the deferred routes (more than one block, a
    non-Poisson field, a non-empty output policy); ``compile`` / ``bind`` lower the rest.
    """

    category = "problem"

    def __init__(self, layout=None, name=None):
        self._name = str(name) if name else "Problem"
        self._layout = layout if layout is not None else Uniform(CartesianMesh())
        self._blocks = {}    # name -> {"physics": <model>, "spatial": <brick or None>}
        self._fields = {}    # field.name -> FieldProblem
        self._params = {}    # param_name -> {"default": value, "kind": str}
        self._aux = {}       # aux_name -> value/descriptor
        self._outputs = []   # output / checkpoint policies (stored; lowering deferred)
        self._time = None    # optional pops.time.Program (attaches at compile time)

    @property
    def name(self):
        return self._name

    # --- assembly (chaining setters) ----------------------------------------
    def block(self, name, physics, spatial=None):
        """Declare a physics block ``name`` (its ``physics`` model is REQUIRED). Chains."""
        key = str(name)
        if physics is None:
            raise ValueError("Problem.block(%r): a physics model is required" % key)
        if key in self._blocks:
            raise ValueError("Problem.block(%r): a block of that name already exists" % key)
        self._blocks[key] = {"physics": physics, "spatial": spatial}
        return self

    def field(self, field_problem):
        """Register an elliptic :class:`~pops.fields.FieldProblem` (keyed on its name). Chains."""
        if not isinstance(field_problem, FieldProblem):
            raise TypeError("Problem.field: expected a pops.fields.FieldProblem; got %r"
                            % type(field_problem).__name__)
        key = field_problem.name
        if key in self._fields:
            raise ValueError("Problem.field: a field named %r already exists" % key)
        self._fields[key] = field_problem
        return self

    def param(self, name, default=None, kind="const"):
        """Declare a runtime/const parameter and its default value. Chains."""
        self._params[str(name)] = {"default": default, "kind": str(kind)}
        return self

    def aux(self, name, value=None):
        """Declare a static aux input ``name`` (e.g. a background field). Chains."""
        self._aux[str(name)] = value
        return self

    def output(self, policy):
        """Attach an output / checkpoint policy (stored; lowering is deferred). Chains."""
        self._outputs.append(policy)
        return self

    def time(self, program):
        """Attach the time scheme (a ``pops.time.Program``) used at compile. Chains."""
        self._time = program
        return self

    # --- layout / amr access -------------------------------------------------
    @property
    def layout(self):
        return self._layout

    @property
    def amr(self):
        """The AMR refinement-policy handle (raises ``ValueError`` if layout is not AMR)."""
        if not isinstance(self._layout, AMR):
            raise ValueError(
                "Problem.amr: only available with layout=AMR(...); this problem has layout %r"
                % type(self._layout).__name__)
        return _AMRPolicyHandle(self)

    # --- DescriptorProtocol surface (pure Python; no runtime, no codegen) ----
    def options(self):
        return {"name": self._name, "layout": self._layout.name,
                "n_blocks": len(self._blocks), "n_fields": len(self._fields),
                "n_params": len(self._params), "n_aux": len(self._aux),
                "n_outputs": len(self._outputs), "has_time": self._time is not None}

    def requirements(self):
        req = dict(self._layout.requirements())
        if self._fields:
            req["elliptic_solve"] = True
        req["time_scheme"] = True
        return req

    def capabilities(self):
        caps = dict(self._layout.capabilities())
        caps["blocks"] = sorted(self._blocks)
        caps["fields"] = sorted(self._fields)
        return caps

    def available(self, context=None):
        """An EXPLAINABLE availability status, computed from the parts (no runtime)."""
        layout_status = self._layout.available(context)
        if not layout_status.ok:
            return layout_status
        if not self._blocks:
            return Availability.no("problem has no block; add one with .block(name, physics)",
                                   missing=["block"])
        for name, spec in self._blocks.items():
            if spec.get("physics") is None:
                return Availability.no("block %r has no physics model" % name,
                                       missing=["physics"])
        for field in self._fields.values():
            status = field.available(context)
            if not status.ok:
                return status
        collisions = set(self._blocks) & set(self._fields)
        if collisions:
            return Availability.no(
                "block and field share name(s): %s" % ", ".join(sorted(collisions)),
                missing=list(collisions))
        return Availability.yes()

    def validate(self, context=None):
        """Structural validation; raise LOUD for the deferred routes (never fake success).

        Checks the layout, that there is at least one block each with a physics model, that
        every field problem is itself valid, and that no block and field share a name. Then
        rejects -- with a clear ``NotImplementedError`` -- the routes not wired in this PR:
        more than one block, a field whose name the default Poisson route does not serve, and
        a non-empty output policy. These are HONEST deferrals, not silent no-ops.
        """
        self._layout.validate(context)
        if not self._blocks:
            raise ValueError(
                "Problem.validate: no block declared; add one with .block(name, physics)")
        for name, spec in self._blocks.items():
            if spec.get("physics") is None:
                raise ValueError("Problem.validate: block %r has no physics model" % name)
        for field in self._fields.values():
            field.validate(context)
        collisions = set(self._blocks) & set(self._fields)
        if collisions:
            raise ValueError("Problem.validate: block and field share name(s): %s"
                             % ", ".join(sorted(collisions)))

        # Deferred routes (PR-2+) -- fail loud, never silently truncate or ignore.
        if len(self._blocks) > 1:
            raise NotImplementedError(
                "Problem.validate: multi-block assembly is deferred; declare exactly one block "
                "(got %d: %s)" % (len(self._blocks), ", ".join(sorted(self._blocks))))
        for field_name in self._fields:
            if field_name not in _POISSON_FIELD_NAMES:
                raise NotImplementedError(
                    "Problem.validate: a non-Poisson field (%r) is deferred; only the default "
                    "Poisson field (one of %s) is wired today"
                    % (field_name, ", ".join(_POISSON_FIELD_NAMES)))
        if self._outputs:
            raise NotImplementedError(
                "Problem.validate: lowering an output/checkpoint policy is deferred; the "
                "assembly stores %d policy(ies) but compile/bind do not consume them yet"
                % len(self._outputs))
        return True

    def inspect(self):
        info = super().inspect()
        info["layout"] = self._layout.inspect()
        info["blocks"] = {
            name: {"physics": getattr(spec["physics"], "name", repr(spec["physics"])),
                   "spatial": getattr(spec["spatial"], "name", spec["spatial"])}
            for name, spec in self._blocks.items()}
        info["fields"] = {name: fp.inspect() for name, fp in self._fields.items()}
        info["params"] = dict(self._params)
        info["aux"] = sorted(self._aux)
        info["outputs"] = [getattr(p, "name", repr(p)) for p in self._outputs]
        info["time"] = getattr(self._time, "name", None) if self._time is not None else None
        return info

    def __str__(self):
        return ("%s [%s] layout=%s | blocks=%d | fields=%d | params=%d | aux=%d | time=%s"
                % (self._name, self.category, self._layout.name, len(self._blocks),
                   len(self._fields), len(self._params), len(self._aux),
                   "set" if self._time is not None else "none"))

    def __repr__(self):
        return ("Problem(name=%r, layout=%s, blocks=%s, fields=%s)"
                % (self._name, self._layout.name, sorted(self._blocks), sorted(self._fields)))


__all__ = ["Problem"]
