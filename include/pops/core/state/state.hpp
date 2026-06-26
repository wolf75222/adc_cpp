/// @file
/// @brief Pointwise types of the physics layer: StateVec<N> (conserved state) and Aux (auxiliary
///        fields from the elliptic solver). The canonical indices of Aux are defined here
///        and must stay in sync with AUX_CANONICAL in python/pops/dsl.py (inherent
///        duplication: Python does not read C++ headers).

#pragma once

#include <pops/core/foundation/types.hpp>

// State and Aux: the two pointwise types handled by the physics layer.
//
// Architecture rule: these are trivially copyable aggregates (POD).
// A PhysicalModel never sees anything else. No box, no MPI rank,
// no Kokkos view enters here. This is what makes the physics portable:
// the same code compiles for CPU and device because it touches no
// parallelism.

namespace pops {

/// Conserved state vector of fixed size, known at compile time.
///
/// Examples: StateVec<1> for a scalar (advection), StateVec<4> for 2D Euler.
/// The value N drives PhysicalModel::n_vars.
///
/// device INVARIANT: raw C array (`Real v[N]`), not std::array; trivially
/// copyable, device-clean (POPS_HD). No non-trivial constructor.
template <int N>
struct StateVec {
  Real v[N]{};  // C array: trivially usable on device (not std::array)

  POPS_HD Real& operator[](int i) { return v[i]; }
  POPS_HD Real operator[](int i) const { return v[i]; }

  POPS_HD static constexpr int size() { return N; }
};

/// @name Arithmetic operators for StateVec (POPS_HD, device-clean).
/// @{
template <int N>
POPS_HD StateVec<N> operator+(StateVec<N> a, const StateVec<N>& b) {
  for (int i = 0; i < N; ++i)
    a[i] += b[i];
  return a;
}

template <int N>
POPS_HD StateVec<N> operator-(StateVec<N> a, const StateVec<N>& b) {
  for (int i = 0; i < N; ++i)
    a[i] -= b[i];
  return a;
}

template <int N>
POPS_HD StateVec<N> operator*(Real s, StateVec<N> a) {
  for (int i = 0; i < N; ++i)
    a[i] *= s;
  return a;
}
/// @}

// Auxiliary fields derived from the elliptic solve: the potential and
// its gradient at the point. This is the single channel through which the coupling enters
// the physics. It feeds BOTH the flux (drift transport: the
// E x B velocity comes from grad phi) and the source (self-gravitating
// compressible fluid: S = -rho grad phi). This duality is what lets a single
// spatial operator serve both target problems.
//
// EXTENSIBLE aux channel (cf. aux_comps()/load_aux in spatial_operator.hpp). The first three
// components are the BASE contract, identical to the legacy layout:
//   [0] = phi, [1] = grad_x, [2] = grad_y.
// The following ones are ADDITIONAL auxiliary fields, optional, in a fixed
// canonical order ([3] = B_z, [4] = T_e, ...). A model declares how many components it reads
// via a static member n_aux (default kAuxBaseComps = 3); a model without n_aux never reads
// the extra fields and stays strictly bit-identical. The extra fields default to 0:
// load_aux overwrites them only if the model requests them.
//
/// SINGLE SOURCE of the layout of the EXTRA aux fields (X-macro). This is the ONLY place
/// listing {member, index} for the fields beyond the base contract.
/// Python-C++ INVARIANT: the indices here must stay identical to AUX_CANONICAL in
/// python/pops/dsl.py ({"phi":0,"grad_x":1,"grad_y":2,"B_z":3,"T_e":4}). Modifying one
/// requires modifying the other at the same time. The duplication is inherent: Python does not
/// read C++ headers. load_aux (device read, spatial_operator.hpp) AND the host marshaling
/// (python/system.cpp) are GENERATED from it, so adding an extra aux field is done HERE and
/// NOWHERE ELSE. This closes the historical gap (#51: T_e added to the struct + load_aux but
/// forgotten in the JIT marshaling -> read as 0 silently). Each entry: X(member, index). The
/// index MUST be >= 3 (components 0..2 are phi/grad_x/grad_y, wired in the base constructor) and
/// follow the canonical layout shared with AUX_CANONICAL on the DSL side (python/pops/dsl.py),
/// inherent duplication: Python does not read C++ headers. To add a field:
/// 1 line here (and the mirror line in AUX_CANONICAL). Pure preprocessor -> device-clean
/// (nvcc/Kokkos), no C++26 reflection.
#define POPS_AUX_FIELDS(X) \
  X(B_z, 3)               \
  X(T_e, 4)

// MAXIMUM number of NAMED aux fields (declared by a model via aux_field("...") on the DSL side,
// ADC-70 phase 1) that an Aux can carry. FIXED bound: Aux stays a trivially copyable POD (raw C
// array, device-clean) -- no allocation, no std::vector. Update HERE and AUX_NAMED_MAX
// on the DSL side (python/pops/dsl.py) if more than four named fields per model are wanted.
inline constexpr int kAuxMaxExtra = 4;

/// @brief POINTWISE auxiliary fields shared with the physics: single coupling channel.
///
/// Role: carry to the point the outputs of the elliptic solver and the fields provided by the system,
/// which the physics consumes BOTH in the flux (drift transport: E x B velocity = grad phi)
/// and in the source (self-gravitating fluid: S = -rho grad phi).
///
/// Usage: a PhysicalModel reads phi/grad_x/grad_y (BASE contract, components 0..2, always
/// present). The canonical EXTRA fields (B_z = comp 3, T_e = comp 4) and the fields NAMED by
/// the model (extra[k] <-> component kAuxNamedBase + k) are loaded ONLY if the model declares
/// a static member n_aux large enough; a model without n_aux stays strictly bit-identical.
/// Read a named field via extra_field(k) (bounded read, returns 0 out of bounds).
///
/// Contract: trivially copyable POD, device-clean (raw C array, no allocation);
/// the layout of the EXTRA fields is generated from the X-macro POPS_AUX_FIELDS (single source).
///
/// Invariants:
/// - all fields beyond the base contract default to 0 (load_aux overwrites them only
///   on the model's request);
/// - the indices must stay in sync with AUX_CANONICAL / AUX_NAMED_BASE on the DSL side
///   (python/pops/dsl.py): inherent duplication, Python does not read C++ headers;
/// - kAuxMaxExtra bounds the number of named fields to keep the POD at fixed size.
struct Aux {
  Real phi{};     // potential       (aux component 0)
  Real grad_x{};  // d phi / d x     (aux component 1)
  Real grad_y{};  // d phi / d y     (aux component 2)
  // EXTRA members generated from POPS_AUX_FIELDS (single source). B_z = out-of-plane B field
  // provided by the system (comp 3); T_e = electron temperature p/rho of a fluid block
  // (comp 4). All optional, defaulting to 0.
#define POPS_AUX_DECL(name, idx) Real name{};
  POPS_AUX_FIELDS(POPS_AUX_DECL)
#undef POPS_AUX_DECL
  // Aux fields NAMED by the model (ADC-70 phase 1). Components of the aux channel starting at
  // kAuxNamedBase (= 5, right after T_e): extra[k] <-> component (kAuxNamedBase + k). Defaulting to
  // 0; load_aux loads them ONLY if the model declares n_aux > kAuxNamedBase (if constexpr,
  // zero cost by default). Read in a DSL formula via aux.extra_field(k).
  Real extra[kAuxMaxExtra]{};

  /// BOUNDED read of a named aux field (component kAuxNamedBase + k). Returns 0 out of bounds: the
  /// generated brick never calls extra_field with a k that the model did not declare, but the
  /// guard makes the access safe (still device-clean, no dynamic branch on a k known at codegen).
  POPS_HD Real extra_field(int k) const { return (k >= 0 && k < kAuxMaxExtra) ? extra[k] : Real(0); }
};

// Width of the aux channel of the base contract (phi, grad phi). A model reading additional
// fields declares a larger n_aux; cf. aux_comps()/load_aux().
inline constexpr int kAuxBaseComps = 3;

// First component of the NAMED aux fields (ADC-70 phase 1): right AFTER the canonical fields
// B_z (3) and T_e (4), so index 5. A model declaring K named fields sets n_aux = kAuxNamedBase +
// K; extra[k] is component (kAuxNamedBase + k). Placed AFTER the canonical channel so that user
// names never encroach on B_z / T_e (which keep their dedicated paths
// set_magnetic_field / set_electron_temperature_from). Python MIRROR: AUX_NAMED_BASE (dsl.py).
inline constexpr int kAuxNamedBase = kAuxBaseComps + 2;  // = 5 (after B_z=3, T_e=4)

// Safeguard: the base of the named fields must be STRICTLY beyond the last canonical extra
// field (the largest index of POPS_AUX_FIELDS + 1). If a canonical field is added beyond T_e,
// this static_assert forces raising kAuxNamedBase (and AUX_NAMED_BASE on the DSL side) accordingly.
#define POPS_AUX_NAMED_BASE_CHECK(name, idx) \
  static_assert(kAuxNamedBase > (idx),      \
                "kAuxNamedBase must be beyond the canonical aux field '" #name "'");
POPS_AUX_FIELDS(POPS_AUX_NAMED_BASE_CHECK)
#undef POPS_AUX_NAMED_BASE_CHECK

// Safeguard: the indices declared in POPS_AUX_FIELDS are strictly EXTRA (>= base) and
// start right after the base contract. Checked at compile time that the table stays
// consistent with kAuxBaseComps (the 1st extra field is at index kAuxBaseComps).
#define POPS_AUX_IDX_CHECK(name, idx)    \
  static_assert((idx) >= kAuxBaseComps, \
                "extra aux field '" #name "': index must be >= kAuxBaseComps (3)");
POPS_AUX_FIELDS(POPS_AUX_IDX_CHECK)
#undef POPS_AUX_IDX_CHECK

// EXPLICIT total cap of the aux channel (ADC-291): the maximum n_aux a model may declare is the
// named base plus the bounded number of model-NAMED fields. NAMING the cap (rather than leaving the
// bound an ad-hoc literal scattered across sites) makes the only remaining compile-time aux limit
// DECLARATIVE; it is mirrored by AUX_NAMED_MAX on the DSL side (python/pops/dsl.py) and reported by
// pops.capabilities() (the _pops module exposes kAuxMaxExtra/kAuxNamedBase so Python cannot silently
// drift). A model declaring n_aux > kAuxMaxComps is out of contract.
inline constexpr int kAuxMaxComps = kAuxNamedBase + kAuxMaxExtra;

// The named-aux bound must allow at least one field, and the Aux POD storage must match it EXACTLY:
// extra[] is the only place named fields live, so a mismatch between the declared limit and the array
// would silently truncate (read-as-0). These pin the limit at compile time (tested via test_aux_names
// and the Python<->C++ parity in test_capabilities.py).
static_assert(kAuxMaxExtra >= 1, "kAuxMaxExtra must allow at least one model-named aux field");
static_assert(static_cast<int>(sizeof(Aux::extra) / sizeof(Real)) == kAuxMaxExtra,
              "Aux::extra[] size must equal kAuxMaxExtra (the declared named-aux limit)");

}  // namespace pops
