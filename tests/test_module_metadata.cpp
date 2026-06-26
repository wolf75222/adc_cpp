// Locks the GeneratedModule metadata reader (include/pops/runtime/program/module_metadata.hpp,
// Spec 2 / ADC-442): the typed operator registry a problem.so exports for introspection and
// install-time validation. OperatorId is the registration index; the reader degrades gracefully on
// a pre-Spec-2 .so (no pops_module_* symbols) by returning present=false. Deliberately light: it does
// NOT build a .so (the end-to-end read of a real generated .so is validated on the Kokkos/AOT path,
// ROMEO); it pins the struct semantics + the absence handling the install path relies on.
#include <pops/runtime/program/module_metadata.hpp>

#include <dlfcn.h>

#include <cstdio>
#include <string>
#include <vector>

using namespace pops::runtime::program;

namespace {

int failures = 0;

void check(bool cond, const char* what) {
  if (cond) {
    std::printf("  [OK ] %s\n", what);
  } else {
    std::printf("  [FAIL] %s\n", what);
    ++failures;
  }
}

}  // namespace

int main() {
  // (1) Default-constructed descriptor: not present, empty, find() == nullptr.
  ModuleMetadata empty;
  check(!empty.present, "default ModuleMetadata is not present");
  check(empty.operators.empty(), "default ModuleMetadata has no operators");
  check(empty.find("anything") == nullptr, "find on an empty descriptor returns nullptr");

  // (2) A hand-built descriptor: OperatorId is the index; find() resolves by name and carries kind.
  ModuleMetadata m;
  m.present = true;
  m.operators.push_back({0U, "fields_from_state", "field_operator", "(U) -> Fields", "{}"});
  m.operators.push_back({1U, "explicit_rhs", "local_rate", "(U, Fields) -> Rate(U)",
                         "{\"kind\":\"local_rate\"}"});
  m.state_spaces.push_back("U");
  m.field_spaces.push_back("fields");
  check(m.find("explicit_rhs") != nullptr, "find resolves a known operator");
  check(m.find("explicit_rhs")->id == 1U, "OperatorId is the registration index");
  check(m.find("explicit_rhs")->kind == "local_rate", "operator kind is carried");
  check(m.find("nope") == nullptr, "find on an unknown operator returns nullptr");

  // (3) Reading a null handle, or a handle that exports no pops_module_* symbols (the running program
  // itself), yields a not-present descriptor -- the backward-compatible / graceful path the install
  // routine takes for a pre-Spec-2 .so.
  check(!read_module_metadata(nullptr).present, "read_module_metadata(nullptr) is not present");
  void* self = dlopen(nullptr, RTLD_NOW | RTLD_LOCAL);
  check(!read_module_metadata(self).present,
        "reading a handle without pops_module_* symbols is not present");
  if (self != nullptr) {
    dlclose(self);
  }

  // (4) required_aux parses the "aux" array of an operator's requirements JSON (ADC-446, the
  // install-time validation input). A flat, closed vocabulary, scanned without a JSON library.
  check(required_aux("{\"kind\":\"local_source\",\"aux\":[\"grad_x\",\"grad_y\"]}") ==
            std::vector<std::string>({"grad_x", "grad_y"}),
        "required_aux extracts a two-name aux array");
  check(required_aux("{\"kind\":\"local_linear_operator\",\"aux\":[\"B_z\"]}") ==
            std::vector<std::string>({"B_z"}),
        "required_aux extracts a single-name aux array");
  check(required_aux("{\"kind\":\"local_rate\"}").empty(),
        "required_aux on requirements without an aux key is empty");
  check(required_aux("{\"kind\":\"field_operator\",\"aux\":[]}").empty(),
        "required_aux on an empty aux array is empty");
  check(required_aux("").empty(), "required_aux on an empty string is empty");

  // (5) required_solver reads the scalar "solver" requirement (Spec criterion 24, ADC-466), and the
  // KEY match is anchored: an aux field literally named "solver", or any value equal to "solver",
  // must NOT be misread as a solver requirement (else a valid install is wrongly rejected).
  check(required_solver("{\"kind\":\"field_operator\",\"solver\":\"geometric_mg\"}") == "geometric_mg",
        "required_solver extracts the solver requirement");
  check(required_solver("{\"kind\":\"local_rate\"}").empty(),
        "required_solver without a solver key is empty");
  check(required_solver("{\"aux\":[\"solver\"],\"foo\":\"bar\"}").empty(),
        "required_solver does not misread an aux field named 'solver' (anchored key match)");
  check(required_solver("{\"aux\":[\"solver\"],\"kind\":\"field_operator\"}").empty(),
        "required_solver does not read the 'kind' value when 'solver' is only an aux element");
  check(required_blocks("{\"kind\":\"local_source\",\"block\":[\"ions\"]}") ==
            std::vector<std::string>{"ions"},
        "required_blocks extracts the block-instance requirement");
  check(required_blocks("{\"kind\":\"field_operator\",\"aux\":[\"B_z\"]}").empty(),
        "required_blocks without a block key is empty");

  std::printf(failures == 0 ? "OK  test_module_metadata\n" : "FAILED test_module_metadata\n");
  return failures == 0 ? 0 : 1;
}
