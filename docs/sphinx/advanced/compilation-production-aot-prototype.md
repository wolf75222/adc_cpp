# Compilation modes

A symbolic [DSL model](../reference/dsl_reference.md) is translated to C++ and compiled to a shared
library that the runtime loads. The compilation runs in one of several modes that trade build time
against the runtime path:

- A prototype mode favors fast iteration while you develop a model.
- A production mode compiles the optimized path used for real runs.

The compiled `.so` is cached, keyed on the toolchain and the build options, so an unchanged model is
not recompiled. For the exact mode names, their flags and the cache key, see the
[DSL reference](../reference/dsl_reference.md).
