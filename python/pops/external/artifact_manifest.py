"""pops.external.artifact_manifest -- the rich manifest of a compiled artifact (Spec 5 sec.13.12).

Spec 5 sec.13.12 (acceptance criterion #36, epic ADC-479) widens the compiled-artifact manifest
from the thin brick-id / category list :class:`pops.external.manifests.CompiledManifest` carries
into the FULL self-description a runtime needs to bind a ``.so`` safely: its ABI identity, the
model name, the blocks / variables / roles, the required aux, the const / runtime params, the
ghost depth, the field outputs, and the ``supports_*`` capability flags (uniform / AMR / MPI /
GPU / stride / partial-IMEX mask / named fields) plus the native entrypoints.

This module reads that metadata from a :class:`pops.codegen.loader.CompiledProblem` (its
``arguments()`` bind table, its carried physical model, its ``abi_key``). It is INERT: it
binds nothing, dlopens nothing, runs no kernel. Where a field is GENUINELY available from
today's metadata it is populated; where the C++ codegen does NOT yet emit a capability flag the
field is honestly ``None`` (UNKNOWN), never a fabricated ``True`` / ``False`` -- the
:meth:`CompiledArtifactManifest.needs_cpp_followup` list reports exactly which flags need a C++
codegen follow-up to become real.

The :func:`build_compiled_manifest` builder reads the passed ``compiled`` object's public
surface only (no module-scope ``pops.codegen`` import), so the ``external`` layer stays at the
bottom of the import graph (cf. tests/architecture/test_import_graph.py). It pulls in no numpy /
``_pops`` at module scope.
"""

# Capability flags Spec 5 sec.13.12 enumerates. The first four are GENUINELY derivable from the
# backend capability dict the compiled model carries (CompiledModel.caps = {cpu, mpi, amr, gpu});
# the last three have NO emitted source in today's C++ codegen, so they are reported UNKNOWN
# (None) and listed by needs_cpp_followup() rather than fabricated.
_SUPPORTS_FROM_CAPS = ("supports_uniform", "supports_amr", "supports_mpi", "supports_gpu")
_SUPPORTS_UNKNOWN = ("supports_stride", "supports_partial_imex_mask", "supports_named_fields")
_SUPPORTS_FLAGS = _SUPPORTS_FROM_CAPS + _SUPPORTS_UNKNOWN


class CompiledArtifactManifest:
    """The rich, self-describing manifest of a compiled artifact (Spec 5 sec.13.12, #36).

    A plain, inert value carrying what a runtime needs to bind the ``.so`` safely. The fields:

      - ``abi_version`` / ``abi_key`` / ``required_headers_sig``: the ABI identity. ``abi_key`` is
        the ``<headers>|<cxx>|<std>`` cache key; ``required_headers_sig`` is its header token;
        ``abi_version`` is the discrete numeric ABI revision, ``None`` until the C++ emits one.
      - ``model_name``: the compiled program / model name.
      - ``blocks``: the physics blocks the artifact commits (bind ``instances=``).
      - ``variables`` / ``roles``: the conservative state component names + their physical roles
        (``roles`` is ``None`` when the carried model does not record them).
      - ``aux_required``: the static aux inputs the model declares.
      - ``params_const`` / ``params_runtime``: the const (frozen at compile) and runtime (settable
        at bind) parameter names.
      - ``ghost_depth``: the halo depth the artifact assumes.
      - ``field_outputs``: the elliptic / diagnostic field outputs the Program records.
      - ``supports_uniform`` / ``supports_amr`` / ``supports_mpi`` / ``supports_gpu``: the layout /
        backend capability flags, read from the model's backend capability dict (``None`` when the
        carried model records no caps).
      - ``supports_stride`` / ``supports_partial_imex_mask`` / ``supports_named_fields``: capability
        flags the C++ codegen does NOT yet emit -- honestly ``None`` (UNKNOWN), reported by
        :meth:`needs_cpp_followup`.
      - ``native_entrypoints``: the native symbols the artifact exports; ``[]`` until the C++ emits
        them in the manifest.

    A flag that is ``None`` is UNKNOWN, not ``False``: a validator must not hard-reject on it (cf.
    :func:`check_layout_supported`). :meth:`to_dict` / :meth:`__str__` serialise / print it.
    """

    def __init__(self, *, model_name=None, abi_key=None, abi_version=None,
                 required_headers_sig=None, blocks=None, variables=None, roles=None,
                 aux_required=None, params_const=None, params_runtime=None, ghost_depth=None,
                 field_outputs=None, supports_uniform=None, supports_amr=None, supports_mpi=None,
                 supports_gpu=None, supports_stride=None, supports_partial_imex_mask=None,
                 supports_named_fields=None, native_entrypoints=None):
        self.model_name = model_name
        self.abi_key = abi_key
        self.abi_version = abi_version
        self.required_headers_sig = required_headers_sig
        self.blocks = list(blocks or [])
        self.variables = list(variables or [])
        self.roles = list(roles) if roles is not None else None
        self.aux_required = list(aux_required or [])
        self.params_const = list(params_const or [])
        self.params_runtime = list(params_runtime or [])
        self.ghost_depth = ghost_depth
        self.field_outputs = list(field_outputs or [])
        # supports_* flags: True / False when GENUINELY known, None when the C++ does not emit it.
        self.supports_uniform = supports_uniform
        self.supports_amr = supports_amr
        self.supports_mpi = supports_mpi
        self.supports_gpu = supports_gpu
        self.supports_stride = supports_stride
        self.supports_partial_imex_mask = supports_partial_imex_mask
        self.supports_named_fields = supports_named_fields
        self.native_entrypoints = list(native_entrypoints or [])

    def supports(self):
        """The ``{flag: True/False/None}`` capability map (``None`` = honestly unknown)."""
        return {name: getattr(self, name) for name in _SUPPORTS_FLAGS}

    def needs_cpp_followup(self):
        """The capability flags that are UNKNOWN today and need a C++ codegen follow-up to be real.

        A flag is listed when its value is ``None`` -- either because the carried model recorded no
        backend caps (the caps-sourced flags) or because the C++ codegen does not yet emit it
        (``supports_stride`` / ``supports_partial_imex_mask`` / ``supports_named_fields``, and
        ``native_entrypoints`` when empty). It is the HONEST record of what is not yet wired, so a
        caller never mistakes a missing flag for a ``False``."""
        pending = [name for name in _SUPPORTS_FLAGS if getattr(self, name) is None]
        if not self.native_entrypoints:
            pending.append("native_entrypoints")
        return pending

    def to_dict(self):
        """A plain-dict view of every manifest field (JSON-ready; ``None`` flags stay ``None``)."""
        out = {"model_name": self.model_name, "abi_key": self.abi_key,
               "abi_version": self.abi_version, "required_headers_sig": self.required_headers_sig,
               "blocks": list(self.blocks), "variables": list(self.variables),
               "roles": list(self.roles) if self.roles is not None else None,
               "aux_required": list(self.aux_required),
               "params_const": list(self.params_const),
               "params_runtime": list(self.params_runtime), "ghost_depth": self.ghost_depth,
               "field_outputs": list(self.field_outputs),
               "native_entrypoints": list(self.native_entrypoints)}
        out.update(self.supports())
        return out

    def __str__(self):
        def _flag(value):
            return "unknown" if value is None else ("yes" if value else "no")

        lines = ["compiled-artifact manifest %r (Spec 5 sec.13.12)"
                 % (self.model_name or "problem")]
        lines.append("  abi          : key=%s version=%s headers=%s"
                     % (_short(self.abi_key), self.abi_version,
                        _short(self.required_headers_sig)))
        lines.append("  blocks       : %s" % (", ".join(self.blocks) or "(none)"))
        lines.append("  variables    : %s" % (", ".join(self.variables) or "(none)"))
        lines.append("  roles        : %s"
                     % ("(unknown)" if self.roles is None else
                        (", ".join(self.roles) or "(none)")))
        lines.append("  aux_required : %s" % (", ".join(self.aux_required) or "(none)"))
        lines.append("  params       : const=[%s] runtime=[%s]"
                     % (", ".join(self.params_const), ", ".join(self.params_runtime)))
        lines.append("  ghost_depth  : %s" % self.ghost_depth)
        lines.append("  field_outputs: %s" % (", ".join(self.field_outputs) or "(none)"))
        lines.append("  supports     :")
        for name in _SUPPORTS_FLAGS:
            lines.append("    %-26s %s" % (name, _flag(getattr(self, name))))
        lines.append("  native_entrypoints: %s"
                     % (", ".join(self.native_entrypoints) or "(none)"))
        pending = self.needs_cpp_followup()
        if pending:
            lines.append("  needs C++ follow-up (UNKNOWN, not fabricated): %s"
                         % ", ".join(pending))
        return "\n".join(lines)

    def __repr__(self):
        return ("CompiledArtifactManifest(model_name=%r, abi_key=%r, blocks=%r)"
                % (self.model_name, _short(self.abi_key), self.blocks))


def _short(value):
    """A short (12-char) head of an abi key / signature for printing (``None`` stays ``None``)."""
    if not value:
        return value
    text = str(value)
    return text if len(text) <= 12 else text[:12] + "..."


def _headers_sig(abi_key):
    """The header-signature token of an ``<headers>|<cxx>|<std>`` abi key (``None`` if absent).

    Mirrors :func:`pops.codegen.abi.module_header_signature`'s ``|`` split: the abi key's first
    pipe-delimited field is the header signature. ``None`` when there is no key or no signature."""
    if not abi_key:
        return None
    head = str(abi_key).split("|", 1)[0].strip()
    return head or None


def _caps_flags(model):
    """The ``{supports_uniform/amr/mpi/gpu: bool}`` flags from a model's backend caps (else None).

    The compiled model carries ``caps = {cpu, mpi, amr, gpu}`` (CompiledModel, the backend
    capability dict). ``supports_uniform`` mirrors ``cpu`` (a CPU-capable artifact runs on a
    single uniform grid). When the carried model records NO caps (e.g. a bare facade model), every
    flag is ``None`` -- honestly unknown, never fabricated as ``True``/``False``."""
    caps = getattr(model, "caps", None)
    if not caps:
        return {name: None for name in _SUPPORTS_FROM_CAPS}
    return {"supports_uniform": bool(caps.get("cpu", False)),
            "supports_amr": bool(caps.get("amr", False)),
            "supports_mpi": bool(caps.get("mpi", False)),
            "supports_gpu": bool(caps.get("gpu", False))}


def build_compiled_manifest(compiled):
    """Build the rich :class:`CompiledArtifactManifest` of a compiled artifact (Spec 5 sec.13.12).

    Reads, all from the ``compiled`` object's public surface (no compile / bind / runtime read):

      - the ABI identity from ``compiled.abi_key`` (and its header token);
      - the model name from ``compiled.program_name`` (or the carried model's ``name``);
      - the blocks / params / aux / ghost depth / field outputs from ``compiled.arguments()`` (the
        bind table built by the codegen introspection -- the single source of those groups);
      - the conservative variable names + roles from the carried physical model;
      - the ``supports_*`` capability flags from the model's backend caps where genuinely known,
        else honestly ``None`` (see :func:`_caps_flags` and :meth:`needs_cpp_followup`).

    The ``arguments()`` call is the only behaviour and it is itself inert (it reads metadata). A
    ``compiled`` handle that exposes no ``arguments`` (a degraded handle) yields a manifest with
    the ABI / model fields populated and the bind groups empty rather than raising."""
    model = getattr(compiled, "model", None)
    abi_key = getattr(compiled, "abi_key", None)
    model_name = (getattr(compiled, "program_name", None)
                  or getattr(model, "name", None))

    args = compiled.arguments() if hasattr(compiled, "arguments") else None
    if args is not None:
        blocks = sorted(args.instances)
        aux_required = sorted(args.aux)
        params_const = sorted(n for n, s in args.params.items() if s.get("kind") != "runtime")
        params_runtime = sorted(n for n, s in args.params.items() if s.get("kind") == "runtime")
        field_outputs = sorted(set(args.solvers) | set(args.outputs))
        ghost_depth = args.layout_runtime.get("ghost_depth")
    else:
        blocks = aux_required = params_const = params_runtime = field_outputs = []
        ghost_depth = None

    variables = list(getattr(model, "cons_names", []) or [])
    raw_roles = getattr(model, "cons_roles", None)
    roles = list(raw_roles) if raw_roles else None

    caps_flags = _caps_flags(model)

    return CompiledArtifactManifest(
        model_name=model_name, abi_key=abi_key, abi_version=None,
        required_headers_sig=_headers_sig(abi_key), blocks=blocks, variables=variables,
        roles=roles, aux_required=aux_required, params_const=params_const,
        params_runtime=params_runtime, ghost_depth=ghost_depth, field_outputs=field_outputs,
        supports_stride=None, supports_partial_imex_mask=None, supports_named_fields=None,
        native_entrypoints=[], **caps_flags)


# Spec 5 sec.13.12 maps each layout kind to the capability flag a compiled artifact must support
# to bind under it. ``"amr"`` -> supports_amr, ``"uniform"`` -> supports_uniform.
_LAYOUT_SUPPORT_FLAG = {"amr": "supports_amr", "uniform": "supports_uniform",
                        "system": "supports_uniform"}


def check_layout_supported(manifest, layout_kind):
    """Validate a compiled artifact against a target layout via its capability flags (sec.13.12).

    Spec 5 sec.13.12 error shape: a manifest whose layout flag is GENUINELY ``False`` rejects with
    "Compiled artifact cannot be used with layout=AMR(...): supports_amr=false". This raises
    :class:`ValueError` ONLY when the flag is known-``False``; a flag that is ``None`` (UNKNOWN --
    the C++ has not emitted it) is NOT a rejection -- it returns the (truthy) "unknown" status so a
    working route is never broken by a missing check (the no-false-positive discipline: a false
    rejection is worse than a missing one). An unrecognised @p layout_kind returns "unknown" too.

    Returns an :class:`pops.descriptors.Availability` -- ``yes`` when the flag is ``True``,
    ``partial`` (truthy is ``False``, but carries the reason) when ``None``/unrecognised, and
    raises on a known ``False``. The caller wires it on the layout-bind branch."""
    from pops.descriptors import Availability  # lazy: keeps the external layer's module scope clean
    flag_name = _LAYOUT_SUPPORT_FLAG.get(str(layout_kind).lower())
    if flag_name is None:
        return Availability.partial(
            "layout %r has no known capability flag; not validated" % (layout_kind,))
    value = getattr(manifest, flag_name, None)
    if value is None:
        return Availability.partial(
            "%s is unknown (the C++ codegen does not emit it yet); layout %r not validated -- "
            "not rejecting on an unknown flag" % (flag_name, layout_kind))
    if value is False:
        raise ValueError(
            "Compiled artifact cannot be used with layout=%s(...): %s=false"
            % (layout_kind, flag_name))
    return Availability.yes("%s=true" % flag_name)


__all__ = ["CompiledArtifactManifest", "build_compiled_manifest", "check_layout_supported"]
