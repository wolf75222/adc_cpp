# STL & Boost Audit (ADC-192)

Date: 2026-06-15. Scope: STL idiom across the host/orchestration code and a
definite decision on whether to introduce Boost. No prior report covers this --
`docs/TOOLCHAIN_ROBUSTESSE_AUDIT_2026-06-10.md` (the nearest mapped report) is
exclusively about build/install robustness (compiler resolution, ABI guards,
dlopen/cache, conda) and contains zero STL/Boost content.

## Boost decision: NO

Do not introduce Boost. Boost appears nowhere in the tree (a grep of `include/`,
`python/`, `CMakeLists.txt`, `cmake/` returns 0 hits), which is the correct
deliberate stance for a header-only, light-CMake, device-portable
(Kokkos/CUDA) project. Every host-side need found is already covered by the STL
the project already uses (it includes <algorithm> and <numeric> and uses
std::sort/std::find/std::iota/std::min/std::max). No isolated, compelling Boost
win exists: candidates one might reach for (Boost.Container small_vector,
Boost.Algorithm, Boost.Hana, mp11) would each add a heavy dependency for what
one or two STL lines already do, and pulling Boost into a header-only core that
must stay device-clean is a real portability cost. The core compiles in C++20,
so std::ranges::find / std::any_of are available without any new dependency.

## STL verdict

The host/orchestration code is mostly clean -- no raw-loop epidemic. The vast
majority of for-loops in python/bindings/system/base/system.cpp, python/bindings/amr/amr_system.cpp and
coupling/*.hpp iterate Kokkos field cells (i,j,c marshaling) or drive a bytecode
VM (coupled_source_program.hpp); these are correctly OUT of scope (device/hot,
STL unavailable on device, no clean primitive). A small number of genuine
host-side hand-rolled loops over plain std::vector/std::string would be clearer
with an STL primitive.

Important lint nuance: `.clang-tidy` enables modernize-*/readability-* (which
would flag readability-use-anyofallof / modernize-use-ranges /
modernize-loop-convert), BUT its HeaderFilterRegex is `include/adc/.*`, so the
`python/*.cpp` pybind translation units are NEVER linted -- the findings there
are net-new and will not be auto-surfaced. The single `include/adc/**` site
(load_balance.hpp) IS inside the regex and would be flagged once a tidy pass
exercises it.

Deliberately NOT reported (to avoid weak findings): the eps/kappa validation
loops in system.cpp throw on first failure (a side effect), so
readability-use-anyofallof would not fire and an std::any_of+throw rewrite is
not strictly clearer.

## Findings

| Sev | Where | Issue |
|-----|-------|-------|
| Low | include/adc/parallel/load_balance.hpp:108 (in make_knapsack_distribution) | Hand-rolled min-element scan (`if (load[q] < load[r]) r = q;`) over the load vector -- exactly std::min_element. Host orchestration (LPT knapsack, once per regrid). File already includes <algorithm> and <numeric>; this site IS inside the clang-tidy HeaderFilterRegex. |
| Low | python/bindings/system/base/system.cpp (resolve_implicit_components name search) | Hand-rolled linear name search over cons.names by index; a textbook std::find / std::ranges::find on a std::vector<std::string>. Host-side block setup. TU already includes <algorithm>. NOT covered by .clang-tidy (python/*.cpp excluded), so net-new. |
| Low | python/bindings/amr/amr_system.cpp (resolve_implicit_components name search) | Exact copy of the system.cpp linear name search plus the same index-based CSV build. Same STL-find opportunity, same uncovered-by-tidy status. The two copies are themselves a small duplication: both could call one shared host helper. |

### Proposed actions (all bit-identical, behavior-preserving)

- load_balance.hpp: replace the manual scan with
  `const int r = static_cast<int>(std::min_element(load.begin(), load.end()) - load.begin());`
  std::min_element returns the first minimum, matching the strict-`<` tie-break
  of the current loop. No numerical-parity change.
- system.cpp: replace with `std::find` (or std::ranges::find), keeping the
  "absent -> throw with CSV of available names" behavior.
- amr_system.cpp: inline std::find as above, or (preferred) extract a single
  host-only helper `int component_index(const VariableSet&, std::string_view)`
  reused by both TUs, implementing the lookup with std::find internally so the
  duplication collapses to one site.

## Conclusion

ADC-192 is satisfied: Boost = NO (definite), STL idiom is healthy, and the three
Low hand-rolled-loop sites are documented here and tracked as a single small
refactor issue. No mass correction is applied; the core stays device-clean.