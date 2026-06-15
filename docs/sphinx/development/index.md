# Development

This section is for contributors and maintainers. User documentation (getting started, tutorials,
how-to, concepts, reference) lives in the other sections; here you find how the code is structured,
how it is tested and validated, and how to contribute.

```{toctree}
:maxdepth: 1

documentation
testing
numerical-validation
version-control-and-branches
```

## Canonical design guides

The design documents are the contributor reference and stay at the repository root:

- [ARCHITECTURE](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md): the five
  layers, the modules, the dispatch seam, the AMR engine.
- [ALGORITHMS](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md): the numerical
  methods, fluxes, reconstruction, elliptic solvers, time integration.
- [CHOICES](https://github.com/wolf75222/adc_cpp/blob/master/docs/CHOICES.md): the deliberate design
  trade-offs.
- [BACKEND_COVERAGE](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md): the
  backend and test matrix.

## Process

- [CONTRIBUTING](https://github.com/wolf75222/adc_cpp/blob/master/CONTRIBUTING.md): the build, test
  and pull-request workflow.
- [SECURITY](https://github.com/wolf75222/adc_cpp/blob/master/SECURITY.md): how to report a
  vulnerability.
- [VERSIONING](https://github.com/wolf75222/adc_cpp/blob/master/docs/VERSIONING.md): the public API
  under SemVer and the release steps.
