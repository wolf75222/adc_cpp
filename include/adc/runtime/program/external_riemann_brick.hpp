#pragma once

// Static dispatch of an EXTERNAL C++ Riemann brick (Spec 3 section 21-22, criterion 20, ADC-463).
//
// `external_brick.hpp` owns the HOST IDENTITY catalog: `ADC_REGISTER_BRICK` records a brick's id +
// requirements, `adc_brick_manifest()` exports them, and `adc.lib.load_cpp_library` surfaces a
// requirement-carrying `riemann.User(id)` descriptor. This header owns the NUMERICAL half: how the
// brick's flux is actually DISPATCHED into the finite-volume machinery without a per-cell string
// lookup.
//
// The flux of an external brick is a `NumericalFlux` policy (numerics/fv/numerical_flux.hpp) living
// in a SEPARATE `.so`, so it can never be a compile-time template parameter of the host's pre-built
// `make_block` (whose `if (riem == "hllc") build_block<..., HLLCFlux>` ladder is closed over the
// native fluxes). Instead the `.so` ITSELF performs the static instantiation: the
// `ADC_DEFINE_EXTERNAL_RIEMANN_BRICK` macro emits an `extern "C"` entry point that calls
// `build_block<Limiter, UserFlux>(...)` -- the user flux is a compile-time template parameter inside
// the `.so`, fully inlined, exactly like a native flux's `build_block` leaf. The host dlopens the
// `.so`, resolves that entry-point function pointer ONCE at install time, and calls it; the per-cell
// kernel then runs the statically-instantiated `UserFlux` functor with NO string comparison on the
// hot path. The only string is the limiter (a 4-way `if` resolved once per install, mirroring the
// native AOT block in compiled_block_abi.hpp).
//
// ABI (flat double arrays, component-major c*n*n + j*n + i, like System::copy_state): identical to
// the AOT compiled block (compiled_block_abi.hpp) so the host marshals an external brick the same
// way it marshals a generated one. Only the flux is fixed at the `.so`'s compile time instead of
// dispatched by string.

#include <adc/runtime/program/external_brick.hpp>

#include <adc/runtime/builders/compiled/compiled_block_abi.hpp>  // compiled_block::{make_grid,...}
#include <adc/runtime/builders/block/block_builder.hpp>  // build_block<Limiter, Flux>, block_n_ghost
#include <adc/runtime/config/dispatch_tags.hpp>          // validate_limiter
#include <adc/numerics/fv/reconstruction.hpp>            // NoSlope / Minmod / VanLeer / Weno5

#include <adc/runtime/dynamic/dynlib.hpp>  // portable dlopen<->LoadLibraryW (ADC-99)

#include <stdexcept>
#include <string>
#include <vector>

namespace adc::runtime::program {

namespace detail {

// Builds the block closures for the external flux @p Flux at limiter @p lim. The flux is a
// COMPILE-TIME template parameter of build_block (the same leaf the native string ladder routes to),
// so it is fully inlined; the only runtime branch is the limiter (resolved ONCE here, not per cell).
// Mirrors make_block's limiter ladder but with the flux fixed -- no riemann string comparison.
template <class Model, class Flux>
BlockClosures external_make_block(const Model& m, const std::string& lim, const GridContext& ctx,
                                  bool recon_prim, Real pos_floor) {
  validate_limiter(lim, "external riemann brick");
  if (lim == "none")
    return build_block<NoSlope, Flux>(m, ctx, /*imex=*/false, recon_prim, "ssprk2", {}, {}, nullptr,
                                      pos_floor);
  if (lim == "minmod")
    return build_block<Minmod, Flux>(m, ctx, /*imex=*/false, recon_prim, "ssprk2", {}, {}, nullptr,
                                     pos_floor);
  if (lim == "vanleer")
    return build_block<VanLeer, Flux>(m, ctx, /*imex=*/false, recon_prim, "ssprk2", {}, {}, nullptr,
                                      pos_floor);
  if (lim == "weno5")
    return build_block<Weno5, Flux>(m, ctx, /*imex=*/false, recon_prim, "ssprk2", {}, {}, nullptr,
                                    pos_floor);
  throw_registry_dispatch_mismatch("external riemann brick", "limiteur", lim);
}

// One explicit residual R = -div F(U) + S evaluated with the external flux @p Flux on @p Model. Same
// marshaling as compiled_block::residual (flat arrays, local single-grid mesh, aux from the host) --
// only the flux is the user's, instantiated statically by build_block. Used by the macro's
// extern "C" adc_brick_residual entry point.
template <class Model, class Flux>
void external_residual(const double* U, double* R, const double* aux_in, int n, double dx,
                       double dy, bool periodic, const std::string& lim, bool recon_prim,
                       double pos_floor) {
  compiled_block::LocalGrid lg =
      compiled_block::make_grid(n, dx, dy, periodic, aux_in, aux_comps<Model>());
  MultiFab Umf(lg.ba, lg.dm, Model::n_vars, block_n_ghost(lim)),
      Rmf(lg.ba, lg.dm, Model::n_vars, 0);
  compiled_block::fill_interior(Umf, U, n, Model::n_vars);
  const GridContext ctx{lg.dom, lg.bc, lg.geom, &lg.aux};
  Model model{};
  BlockClosures clo =
      external_make_block<Model, Flux>(model, lim, ctx, recon_prim, static_cast<Real>(pos_floor));
  clo.rhs_into(Umf, Rmf);
  compiled_block::extract(Rmf, R, n, Model::n_vars);
}

}  // namespace detail

// The host-side handle to a loaded external Riemann brick `.so`: dlopen the library, read its
// manifest, and resolve the typed entry-point function pointers ONCE. After construction the brick
// is dispatched by calling the resolved residual() pointer -- a direct C call into the `.so`'s
// statically-instantiated flux, never a per-cell string lookup. The manifest is also registered in
// the process catalog (BrickRegistry) so the brick's id + requirements are visible to a later host
// query (mirroring what adc.lib.load_cpp_library does on the Python side).
//
// This is the C++ counterpart of adc.lib.load_cpp_library: the Python path surfaces the descriptor
// (requirements/capabilities) for the board/install layer; this path resolves the numerical entry
// point for a host that drives the brick from C++. A brick `.so` not exporting the expected symbols
// is rejected with a clear error (it is not an adc external Riemann brick `.so`).
class ExternalBrickHandle {
 public:
  // Function-pointer type of the brick's residual entry point (ADC_DEFINE_EXTERNAL_RIEMANN_BRICK).
  using ResidualFn = void (*)(const double*, double*, const double*, int, double, double, int,
                              const char*, int, double);

  // dlopen @p so_path, read + register its manifest, and resolve the entry points for brick @p id.
  // Throws std::runtime_error if the library cannot be opened, does not export adc_brick_manifest /
  // adc_brick_residual (not an adc external Riemann brick), or does not register @p id as a riemann
  // brick (a clear, actionable message naming the id and the loaded ids).
  ExternalBrickHandle(const std::string& so_path, const std::string& id) : id_(id) {
    handle_ = dynlib::open(so_path);
    if (!dynlib::valid(handle_))
      throw std::runtime_error("external riemann brick: cannot dlopen '" + so_path +
                               "': " + dynlib::last_error());
    auto manifest_fn =
        reinterpret_cast<const char* (*)()>(dynlib::sym(handle_, "adc_brick_manifest"));
    if (manifest_fn == nullptr)
      throw std::runtime_error(
          "external riemann brick '" + so_path +
          "' does not export adc_brick_manifest(); it is not an adc brick .so");
    // The .so's static initializers already ran ADC_REGISTER_BRICK against the registry of the .so's
    // OWN image; reading its manifest and re-registering here makes the ids visible in THIS image's
    // process catalog too (RTLD_LOCAL keeps the .so's statics private otherwise). Done via the same
    // entries() the Python seam parses -- no behavioral divergence.
    register_manifest_json(manifest_fn());
    const BrickManifestEntry* entry = BrickRegistry::instance().lookup(id_);
    if (entry == nullptr)
      throw std::runtime_error("external riemann brick '" + id_ +
                               "' not found in the manifest of '" + so_path + "'");
    if (entry->category != "riemann")
      throw std::runtime_error("external brick '" + id_ + "' is registered as category '" +
                               entry->category + "', not 'riemann'");
    residual_ = reinterpret_cast<ResidualFn>(dynlib::sym(handle_, "adc_brick_residual"));
    if (residual_ == nullptr)
      throw std::runtime_error("external riemann brick '" + id_ +
                               "' does not export adc_brick_residual(); rebuild the .so with "
                               "ADC_DEFINE_EXTERNAL_RIEMANN_BRICK");
    requirements_ = entry->requirements;
  }

  ExternalBrickHandle(const ExternalBrickHandle&) = delete;
  ExternalBrickHandle& operator=(const ExternalBrickHandle&) = delete;
  ~ExternalBrickHandle() {
    if (dynlib::valid(handle_))
      dynlib::close(handle_);
  }

  // The resolved residual entry point: a direct call into the `.so`'s statically-instantiated flux.
  ResidualFn residual() const { return residual_; }

  // The CSV of model capabilities the brick requires (from its manifest); "" when none.
  const std::string& requirements() const { return requirements_; }

  const std::string& id() const { return id_; }

 private:
  // Minimal manifest reader: parse the {"bricks":[{"id","category","requirements",...}]} the `.so`
  // exports and register each entry. A header-only sibling of lib.py's _register_manifest; it accepts
  // exactly the JSON to_json() emits (flat string fields, no nesting). A malformed manifest throws.
  static void register_manifest_json(const char* json) {
    if (json == nullptr)
      throw std::runtime_error("external riemann brick: adc_brick_manifest() returned NULL");
    const std::string s = json;
    // to_json() emits {"bricks":[{...},{...}]}: skip past the array's '[' (the top-level object's '{'
    // precedes it), then read each {...} brick record until the array closes. An empty array ([]) or a
    // manifest with no bricks array registers nothing (valid).
    const std::size_t arr = s.find('[');
    if (arr == std::string::npos)
      return;
    std::size_t pos = arr + 1;
    while (true) {
      const std::size_t obj = s.find('{', pos);
      if (obj == std::string::npos)
        break;
      const std::size_t end = s.find('}', obj);
      if (end == std::string::npos)
        break;
      const std::string rec = s.substr(obj, end - obj + 1);
      BrickManifestEntry e;
      e.id = field(rec, "id");
      e.category = field(rec, "category");
      e.requirements = field(rec, "requirements");
      e.capabilities = field(rec, "capabilities");
      if (!e.id.empty())
        BrickRegistry::instance().register_brick(e);
      pos = end + 1;
    }
  }

  // Extracts the value of "key":"value" from one manifest record (the fields to_json() emits are flat
  // quoted strings; this is a targeted reader, not a general JSON parser). It skips backslash-escaped
  // characters when scanning for the closing quote (so an escaped `\"` inside the value does not end
  // it early) and json_unescape's the result. "" when the key is absent.
  static std::string field(const std::string& rec, const std::string& key) {
    const std::string pat = "\"" + key + "\":\"";
    const std::size_t k = rec.find(pat);
    if (k == std::string::npos)
      return "";
    const std::size_t start = k + pat.size();
    std::size_t end = start;
    while (end < rec.size() && rec[end] != '"') {
      end += (rec[end] == '\\' && end + 1 < rec.size()) ? 2 : 1;  // skip an escaped pair atomically
    }
    if (end >= rec.size())
      return "";
    return json_unescape(rec.substr(start, end - start));
  }

  dynlib::handle handle_ = nullptr;
  ResidualFn residual_ = nullptr;
  std::string id_;
  std::string requirements_;
};

}  // namespace adc::runtime::program

// Defines the static-dispatch ABI of an external Riemann brick `.so`: registers its identity in the
// host catalog AND emits the entry point the host calls. Use ONCE at namespace scope:
//   struct MyRiemann { template <class M> ADC_HD typename M::State operator()(...) const {...} };
//   ADC_DEFINE_EXTERNAL_RIEMANN_BRICK("my_riemann", MyRiemann,
//                                     adc::CompositeModel<adc::Euler, ...>, "pressure,wave_speeds");
//   ADC_DEFINE_BRICK_MANIFEST();  // exports the manifest reader (once per .so)
//
// @p id          the brick id a user selects via adc.lib.riemann.User(id);
// @p Flux        the NumericalFlux policy (numerics/fv/numerical_flux.hpp contract);
// @p Model       a TOP-LEVEL ALIAS of the CompositeModel the .so instantiates the flux against (write
//                `using Model = adc::CompositeModel<...>;` first and pass the alias -- a bare
//                CompositeModel<A, B, C> has a comma the preprocessor would split, exactly like
//                ADC_DEFINE_COMPILED_BLOCK(MODEL));
// @p reqs_csv    the CSV of model capabilities the brick requires (surfaced in the manifest).
//
// The emitted adc_brick_residual instantiates build_block<Limiter, Flux> at the .so's compile time:
// the flux is a STATIC template argument, never a per-cell string lookup. adc_brick_nvars /
// adc_brick_naux let the host size its marshaling arrays (same role as adc_compiled_nvars/_naux).
//
// ABI WARNING: the brick `.so` MUST be compiled against the SAME Kokkos backend and version (and the
// same adc headers) as the host binary that dlopens it -- the residual runs the host's Kokkos
// runtime. A mismatched `.so` may dlopen yet fail unpredictably. There is no load-time Kokkos-ABI
// check yet (a future safeguard); for now this is the caller's contract, mirroring the AOT
// compiled-block path (ADC_DEFINE_COMPILED_BLOCK), which carries the same requirement.
#define ADC_DEFINE_EXTERNAL_RIEMANN_BRICK(id, Flux, Model, reqs_csv)                       \
  ADC_REGISTER_BRICK(id, "riemann", reqs_csv);                                             \
  extern "C" int adc_brick_nvars() {                                                       \
    return Model::n_vars;                                                                  \
  }                                                                                        \
  extern "C" int adc_brick_naux() {                                                        \
    return adc::aux_comps<Model>();                                                        \
  }                                                                                        \
  extern "C" void adc_brick_residual(const double* U, double* R, const double* aux, int n, \
                                     double dx, double dy, int periodic, const char* lim,  \
                                     int recon_prim, double pos_floor) {                   \
    ::adc::runtime::program::detail::external_residual<Model, Flux>(                       \
        U, R, aux, n, dx, dy, periodic != 0, lim, recon_prim != 0, pos_floor);             \
  }
