#pragma once

/// @file
/// @brief PORTABLE dynamic loading: `dlopen`/`dlsym`/`dlclose` (POSIX) <-> `LoadLibraryW`/
///        `GetProcAddress`/`FreeLibrary` (Windows). Minimal surface for the adc runtime (native
///        loader / DSL `.dll`), ADC-99.
///
/// POSIX: strictly equivalent to the historical usage (`RTLD_NOW | RTLD_LOCAL`). The DSL
/// "production" path has a SPECIAL need (`RTLD_GLOBAL` promotion to resolve the symbols of
/// `_pops` exported as `POPS_EXPORT` through the dlopen) that stays handled at its call site; on the
/// Windows side the equivalent goes through `__declspec(dllexport)` (cf. export.hpp) + import library, ADC-100.

#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace pops {
namespace dynlib {

#if defined(_WIN32)
using handle = HMODULE;
#else
using handle = void*;
#endif

/// Platform dynamic library suffix.
inline const char* suffix() {
#if defined(_WIN32)
  return ".dll";
#elif defined(__APPLE__)
  return ".dylib";
#else
  return ".so";
#endif
}

/// Opens a dynamic library (@p path in UTF-8). Returns a null handle on failure.
inline handle open(const std::string& path) {
#if defined(_WIN32)
  // UTF-8 -> UTF-16 for LoadLibraryW (Unicode paths and paths with spaces).
  const int n = ::MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
  std::wstring w(n > 0 ? n - 1 : 0, L'\0');
  if (n > 0)
    ::MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, w.data(), n);
  return ::LoadLibraryW(w.c_str());
#else
  return ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

/// Resolves @p name in @p h. Returns nullptr if absent.
inline void* sym(handle h, const char* name) {
#if defined(_WIN32)
  return reinterpret_cast<void*>(::GetProcAddress(h, name));
#else
  return ::dlsym(h, name);
#endif
}

/// Closes @p h (no-op on a null handle).
inline void close(handle h) {
#if defined(_WIN32)
  if (h)
    ::FreeLibrary(h);
#else
  if (h)
    ::dlclose(h);
#endif
}

/// true if @p h is a valid handle.
inline bool valid(handle h) {
  return h != handle{};
}

/// Last error message (best-effort, for diagnostics).
inline std::string last_error() {
#if defined(_WIN32)
  const DWORD e = ::GetLastError();
  if (!e)
    return {};
  char* buf = nullptr;
  ::FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, e, 0, reinterpret_cast<char*>(&buf), 0, nullptr);
  std::string m = buf ? buf : "";
  if (buf)
    ::LocalFree(buf);
  return m;
#else
  const char* e = ::dlerror();
  return e ? std::string(e) : std::string();
#endif
}

}  // namespace dynlib
}  // namespace pops
