# Security policy

`adc_cpp` is scientific research code (a numerical solver). It is not meant to process
sensitive data or to be exposed as a service, but reports of vulnerabilities or safety
issues (out-of-bounds read, undefined behavior, denial of service from malformed input,
...) are welcome.

## Reporting

- Preferably via GitHub **private vulnerability reporting** (the *Security* tab of the
  repository, *Report a vulnerability*).
- Otherwise, open an issue; for a sensitive topic, give only the kind of problem and ask
  for a private channel before sharing the details.

Follow-up is **best-effort**: the project is maintained by one person, with no guaranteed
turnaround. Fixes are published as ordinary patches on `master`.

## Scope

The numerical core, the Python bindings and the DSL path (compilation of code generated at
runtime). Third-party dependencies (Kokkos, OpenMPI, pybind11, ...) are the responsibility
of their own maintainers.
