#pragma once

/// @file
/// @brief ADC_EXPORT: force DEFAULT VISIBILITY on an out-of-line symbol, even when the unit is
///        compiled with -fvisibility=hidden (case of the pybind11 _adc module).
///
/// Used by the DSL "production" path: a generated .so loader, dlopen-ed at run time
/// (System::add_native_block), includes the add_compiled_model header template which calls
/// OUT-OF-LINE methods of adc::System (install_block / grid_context / ensure_aux_width) DEFINED in the
/// already-loaded _adc module. Without default visibility, these symbols do not appear in the
/// dynamic table of the module and the loader CANNOT resolve them (link failure at dlopen).
/// We therefore export EXACTLY these methods + adc::abi_key (minimal surface). MSVC / Windows: no
/// effect here (POSIX dlopen path; a future Windows port would use __declspec(dllexport)).

// Windows: the _adc module (which DEFINES these symbols) must define ADC_EXPORT_BUILDING_MODULE at its
// compilation -> dllexport; the generated .dll loader that IMPORTS them falls back on dllimport. Unix:
// default visibility (the module is compiled with -fvisibility=hidden). See ADC-99 (portable layer).
#if defined(_WIN32)
#if defined(ADC_EXPORT_BUILDING_MODULE)
#define ADC_EXPORT __declspec(dllexport)
#else
#define ADC_EXPORT __declspec(dllimport)
#endif
#else
#define ADC_EXPORT __attribute__((visibility("default")))
#endif
