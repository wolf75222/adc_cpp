#pragma once

#include <stdexcept>
#include <string>
#include <vector>

/// @file
/// @brief Descriptor of a model's variables (Vars). Carried by the HYPERBOLIC brick (along with the
///        flux and the conversions), because variables and flux are physically linked; this is NOT a
///        standalone brick that can be combined freely.
///
/// `Variables` DESCRIBES the variables (conservative or primitive): kind, names, size. It is HOST
/// metadata (it does not drive the computation, which works per component via the cons<->prim
/// conversions), but it is a MANDATORY CONTRACT of the hyperbolic model (HyperbolicModel concept):
/// conservative_vars() and primitive_vars(). Used for introspection, named diagnostics, and labelled
/// output.

namespace pops {

/// Kind of a variable set: conserved (U) or primitive (W).
/// Used as a tag in VariableSet; do not use it to dispatch numerical logic.
enum class VariableKind { Conservative, Primitive };

/// PHYSICAL role of a component. Lets you address a component by its MEANING
/// (index_of(MomentumX)) rather than by a magic index u[1]: a coupled source can target
/// "the momentum of a given species" without hard-coding the index. Custom = role not provided.
enum class VariableRole {
  Density,
  MomentumX,
  MomentumY,
  MomentumZ,
  Energy,
  VelocityX,
  VelocityY,
  VelocityZ,
  Pressure,
  Temperature,
  Scalar,
  Custom
};

/// Forward declaration: VariableSet::index_of(const std::string&) resolves a canonical role NAME via
/// role_from_name (defined below) before matching a user-defined role label.
inline VariableRole role_from_name(const std::string& s);

/// A variable: name, physical role, component index in the state.
struct Variable {
  std::string name;
  VariableRole role;
  int component;
};

/// A model's variable set: kind (cons/prim), names, size, canonical `roles` (optional, parallel to
/// `names`; absent -> Custom), and `user_roles` (optional string labels parallel to `names`, for
/// components whose role is OUTSIDE the canonical enum). Existing calls `{kind, names, size}` and
/// `{kind, names, size, roles}` stay valid (user_roles empty). index_of(role) gives the index of the
/// component carrying that role (-1 if absent).
struct VariableSet {
  VariableKind kind;
  std::vector<std::string> names;
  int size;
  std::vector<VariableRole> roles{};      ///< parallel to `names`; empty = roles not provided
  std::vector<std::string> user_roles{};  ///< parallel to `names`; per-component user-defined role
                                          ///< label (Custom role); empty entry = canonical role

  /// Index of the component carrying @p role (first occurrence), -1 if absent.
  int index_of(VariableRole role) const {
    for (int i = 0; i < static_cast<int>(roles.size()); ++i)
      if (roles[i] == role)
        return i;
    return -1;
  }
  /// Index of the component carrying @p role addressed BY NAME: a canonical role name
  /// (role_from_name) first, else a user-defined role label (user_roles). -1 if absent. Resolving a
  /// user label by string removes the first-occurrence ambiguity of several `Custom` components. An
  /// EMPTY @p role is never a valid target (it would otherwise match the empty user_roles slot of a
  /// canonical component on a mixed block) and returns -1.
  int index_of(const std::string& role) const {
    if (role.empty())
      return -1;
    const VariableRole r = role_from_name(role);
    if (r != VariableRole::Custom)
      return index_of(r);
    for (int i = 0; i < static_cast<int>(user_roles.size()); ++i)
      if (user_roles[i] == role)
        return i;
    return -1;
  }
  /// Full descriptor of component @p i (Custom role if not provided).
  Variable at(int i) const {
    return {names[i], i < static_cast<int>(roles.size()) ? roles[i] : VariableRole::Custom, i};
  }
};

/// Human-readable name of a role (introspection, Python binding). Stable: used as a key on the
/// application side.
inline const char* role_name(VariableRole r) {
  switch (r) {
    case VariableRole::Density:
      return "density";
    case VariableRole::MomentumX:
      return "momentum_x";
    case VariableRole::MomentumY:
      return "momentum_y";
    case VariableRole::MomentumZ:
      return "momentum_z";
    case VariableRole::Energy:
      return "energy";
    case VariableRole::VelocityX:
      return "velocity_x";
    case VariableRole::VelocityY:
      return "velocity_y";
    case VariableRole::VelocityZ:
      return "velocity_z";
    case VariableRole::Pressure:
      return "pressure";
    case VariableRole::Temperature:
      return "temperature";
    case VariableRole::Scalar:
      return "scalar";
    case VariableRole::Custom:
      return "custom";
  }
  return "custom";
}

/// Inverse of role_name: physical role from its stable name (Custom if unknown). Used to
/// reconstruct a VariableSet with roles from TEXT metadata (e.g. the string carried by a compiled /
/// dynamic .so: the extern "C" ABI carries only strings, not the enum).
inline VariableRole role_from_name(const std::string& s) {
  if (s == "density")
    return VariableRole::Density;
  if (s == "momentum_x")
    return VariableRole::MomentumX;
  if (s == "momentum_y")
    return VariableRole::MomentumY;
  if (s == "momentum_z")
    return VariableRole::MomentumZ;
  if (s == "energy")
    return VariableRole::Energy;
  if (s == "velocity_x")
    return VariableRole::VelocityX;
  if (s == "velocity_y")
    return VariableRole::VelocityY;
  if (s == "velocity_z")
    return VariableRole::VelocityZ;
  if (s == "pressure")
    return VariableRole::Pressure;
  if (s == "temperature")
    return VariableRole::Temperature;
  if (s == "scalar")
    return VariableRole::Scalar;
  return VariableRole::Custom;
}

/// CSV of a VariableSet's names (separator ','). Building block of the TEXT metadata that a generated
/// .so exposes: the extern "C" ABI does not carry a C++ object, so we serialize to a string.
inline std::string names_csv(const VariableSet& vs) {
  std::string s;
  for (std::size_t i = 0; i < vs.names.size(); ++i) {
    if (i)
      s += ',';
    s += vs.names[i];
  }
  return s;
}

/// CSV of a VariableSet's roles (role_name, separator ','). A component carrying a user-defined role
/// label (user_roles, Custom role) emits its LABEL instead of "custom", so the user role round-trips
/// through the .so ABI (parse_roles_into is the inverse). EMPTY if the model does not provide its
/// roles (vs.roles empty): the consumer then falls back to indices (backward compatibility).
inline std::string roles_csv(const VariableSet& vs) {
  std::string s;
  for (std::size_t i = 0; i < vs.roles.size(); ++i) {
    if (i)
      s += ',';
    if (i < vs.user_roles.size() && !vs.user_roles[i].empty())
      s += vs.user_roles[i];
    else
      s += role_name(vs.roles[i]);
  }
  return s;
}

/// Inverse of roles_csv: fill @p vs.roles (and @p vs.user_roles for any NON-canonical token) from a
/// roles CSV. A canonical token (role_from_name) maps to its enum with an empty user label; a
/// non-canonical token maps to VariableRole::Custom keeping the token as its user-role label, so a
/// user role survives the .so ABI round-trip. user_roles stays EMPTY when every token is canonical
/// (bit-identical to historical roleless / canonical blocks). Empty @p csv leaves both empty.
inline void parse_roles_into(VariableSet& vs, const std::string& csv) {
  if (csv.empty())
    return;
  std::vector<std::string> labels;
  bool any_user = false;
  std::size_t start = 0;
  for (;;) {
    const std::size_t comma = csv.find(',', start);
    const std::string tok =
        csv.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
    const VariableRole r = role_from_name(tok);
    vs.roles.push_back(r);
    const bool is_user = (r == VariableRole::Custom && tok != role_name(VariableRole::Custom));
    labels.push_back(is_user ? tok : std::string());
    any_user = any_user || is_user;
    if (comma == std::string::npos)
      break;
    start = comma + 1;
  }
  if (any_user)
    vs.user_roles = std::move(labels);
}

/// Resolve a REQUIRED canonical @p role to its component in @p vs, for a NAMED coupling
/// (add_collision / add_thermal_exchange / ionization) that historically targeted the canonical
/// layout. A genuinely ROLELESS set (`roles` empty: a legacy / dynamic block that declares no roles)
/// returns the canonical @p fallback -- backward compatible. A ROLES-BEARING set that declares some
/// roles but NOT @p role THROWS: a silent fallback to @p fallback would apply the coupling to the
/// WRONG component. @p origin / @p block name the error.
inline int coupling_role_index(const VariableSet& vs, VariableRole role, int fallback,
                               const char* origin, const std::string& block) {
  if (vs.roles.empty())
    return fallback;  // roleless legacy block: keep the canonical fallback
  const int c = vs.index_of(role);
  if (c >= 0)
    return c;
  throw std::runtime_error(std::string(origin) + " : block '" + block + "' declares roles (" +
                           roles_csv(vs) + ") but not the role '" + role_name(role) +
                           "' this coupling requires (no silent fallback to component " +
                           std::to_string(fallback) + ")");
}

/// A model's "names" metadata: "cons_csv|prim_csv" (separator '|' between the two sets). Read
/// as-is by the consumer (System) via the optional symbol pops_compiled_var_names.
template <class Model>
std::string var_names_meta() {
  return names_csv(Model::conservative_vars()) + "|" + names_csv(Model::primitive_vars());
}

/// A model's "roles" metadata: "cons_roles_csv|prim_roles_csv" (empty side = roles not provided).
template <class Model>
std::string roles_meta() {
  return roles_csv(Model::conservative_vars()) + "|" + roles_csv(Model::primitive_vars());
}

/// Old name (compat): VariableSet used to be `Variables`. Kept for existing and generated code.
using Variables = VariableSet;

}  // namespace pops

/// Exports the OPTIONAL "names + roles" metadata of a .so block via extern "C" symbols read by
/// dlsym on the System side. SHARED by the two generated backends (AOT compiled_block and JIT
/// dynamic_model): the lost metadata (names/roles) is carried without breaking the flat ABI.
/// BACKWARD-COMPATIBLE: a .so that does not define these symbols (generated before this work) stays
/// valid -- the System does not find the symbol and falls back (names u0.., no roles).
/// @p MODEL = type of the model (carries conservative_vars / primitive_vars).
#define POPS_EXPORT_BLOCK_METADATA(MODEL)                       \
  extern "C" const char* pops_compiled_var_names() {            \
    static const std::string s = pops::var_names_meta<MODEL>(); \
    return s.c_str();                                          \
  }                                                            \
  extern "C" const char* pops_compiled_roles() {                \
    static const std::string s = pops::roles_meta<MODEL>();     \
    return s.c_str();                                          \
  }

/// Exports the block's gamma (adiabatic index) via the optional symbol pops_compiled_gamma, read by
/// the System's inter-species couplings (collision, thermal exchange, T_e). EMITTED ONLY if the
/// model declares a gamma: otherwise the symbol stays absent and the System keeps its default 1.4.
#define POPS_EXPORT_BLOCK_GAMMA(GAMMA)      \
  extern "C" double pops_compiled_gamma() { \
    return (GAMMA);                        \
  }
