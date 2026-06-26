"""Aux-channel layout, physical roles, and runtime-param bound.

These constants and helpers are the single Python-side source of the canonical
aux-channel layout (mirror of ``POPS_AUX_FIELDS`` in
``include/pops/core/state.hpp``), the canonical name->role mapping, and the
runtime-parameter bound. They are pure data + pure functions: this module
imports nothing above the IR layer (stdlib only), so the whole ``pops.physics``
package can depend on it without pulling in codegen or ``_pops``.

Kept inside ``pops.physics`` (rather than promoted to ``pops.ir``) to avoid
scope-creep across the codegen consumers that already import these names
LAZILY from ``pops.physics`` (e.g. ``codegen.compile``'s lazy
``from pops.physics import Model, AUX_CANONICAL``); see the Spec-4 blueprint
punch-list P5.
"""

# (cf. pops::Aux / kAuxBaseComps on the C++ side). phi/grad_x/grad_y = BASE contract (3 components);
# the following ones (B_z, ...) WIDEN the channel -> the generated brick then declares n_aux so that
# the system sizes and populates the shared channel (cf. CompositeModel::n_aux, ensure_aux_width).
#
# INHERENT C++ <-> Python DUPLICATION: the table below MUST stay the MIRROR of the single C++
# source POPS_AUX_FIELDS (include/pops/core/state.hpp), from which load_aux (device read)
# and the host marshaling (python/system.cpp) are generated. Python does not read the C++ headers, so
# we cannot generate it: adding an extra aux field = 1 line here AND 1 line in POPS_AUX_FIELDS,
# with the SAME {name, index}. This is the only remaining duplication; the 3 C++ sites are now unified.
AUX_CANONICAL = {"phi": 0, "grad_x": 1, "grad_y": 2, "B_z": 3, "T_e": 4}
AUX_BASE_COMPS = 3

# Aux fields NAMED by the model (ADC-70 phase 1): m.aux_field("name"). Components starting from
# AUX_NAMED_BASE (= 5, just after T_e=4) -- the k-th declared name is component AUX_NAMED_BASE + k,
# read in C++ via aux.extra_field(k). MIRRORS of kAuxNamedBase / kAuxMaxExtra (include/pops/core/state.hpp,
# single C++ source). Decouples the user names from the canonical channel: B_z / T_e keep their indices
# 3 / 4 and their dedicated paths (set_magnetic_field / set_electron_temperature_from).
AUX_NAMED_BASE = 5
AUX_NAMED_MAX = 4  # maximum number of named aux fields per model (= kAuxMaxExtra on the C++ side)

# Bound on the number of RUNTIME parameters per block (P7-b). MIRROR of kMaxRuntimeParams
# (include/pops/runtime/runtime_params.hpp): the C++ carrier RuntimeParams has an array of this FIXED
# size (device-copiable without allocation), so a model exceeding the bound is rejected at codegen.
_K_MAX_RUNTIME_PARAMS = 32


def aux_n_aux(aux_names):
    """Aux channel width required by these CANONICAL fields: max(3, largest index + 1).
    Raises ValueError on an unknown name (a canonical aux field MUST be a component of pops::Aux)."""
    w = AUX_BASE_COMPS
    for nm in aux_names:
        if nm not in AUX_CANONICAL:
            raise ValueError("unknown aux field '%s': expected %s (components of pops::Aux)"
                             % (nm, sorted(AUX_CANONICAL)))
        w = max(w, AUX_CANONICAL[nm] + 1)
    return w


def aux_total_n_aux(aux_names, aux_extra_names):
    """TOTAL width of the aux channel: max of the canonical width (aux_n_aux) and, if NAMED fields
    (aux_field) are declared, AUX_NAMED_BASE + number of names (the last name = component
    AUX_NAMED_BASE + len-1). Without a named field -> aux_n_aux (historical path, bit-identical)."""
    w = aux_n_aux(aux_names)
    if aux_extra_names:
        w = max(w, AUX_NAMED_BASE + len(aux_extra_names))
    return w


# --- Physical roles: variable name -> VariableRole -------------------------
# CANONICAL mapping name -> physical role (cf. pops::VariableRole / role_name on the C++ side). Lets a
# generated brick DECLARE the MEANING of its components (density, momentum, energy...) instead of
# empty roles, so that inter-species couplings (System::add_collision / add_thermal_exchange)
# resolve via index_of(role) rather than via a literal index. The usual names of fluid models
# (rho, rho_u, u, p, E, n...) are recognized; an unknown name stays 'Custom'. A model can impose
# its roles explicitly (conservative_vars(..., roles=[...]) / set_primitive_state(..., roles=[...]))
# for a non-standard layout. Key = EXACT variable name, value = member of pops::VariableRole.
CANONICAL_ROLES = {
    "rho": "Density", "n": "Density", "density": "Density",
    "rho_u": "MomentumX", "rhou": "MomentumX", "mom_x": "MomentumX", "mx": "MomentumX",
    "rho_v": "MomentumY", "rhov": "MomentumY", "mom_y": "MomentumY", "my": "MomentumY",
    "rho_w": "MomentumZ", "rhow": "MomentumZ", "mom_z": "MomentumZ", "mz": "MomentumZ",
    "E": "Energy", "rho_E": "Energy", "ener": "Energy", "energy": "Energy",
    "u": "VelocityX", "v": "VelocityY", "w": "VelocityZ",
    "vx": "VelocityX", "vy": "VelocityY", "vz": "VelocityZ",
    "p": "Pressure", "pressure": "Pressure",
    "T": "Temperature", "temperature": "Temperature",
}


def role_of(name):
    """CANONICAL physical role of name @p name (member of pops::VariableRole), 'Custom' if unknown."""
    return CANONICAL_ROLES.get(name, "Custom")


def roles_for(names, override=None):
    """List of roles (pops::VariableRole members) parallel to @p names. @p override (optional):
    list of the same length explicitly fixing the roles (string 'Density'... or None to fall back
    on the canonical mapping of the name). Used for non-standard layouts where names are not enough."""
    if override is None:
        return [role_of(nm) for nm in names]
    if len(override) != len(names):
        raise ValueError("roles: %d roles for %d variables" % (len(override), len(names)))
    return [(r if r is not None else role_of(nm)) for nm, r in zip(names, override, strict=True)]


