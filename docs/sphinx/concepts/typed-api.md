# The typed authoring API

PoPS is authored in Python and executed in C++. The Python side describes a simulation
with typed objects; the C++ core does every cell-by-cell computation. This page states the
rule that shapes the whole `pops` surface (Spec 5): Python describes, C++ executes, and the
description is made of typed objects, never of strings disguised as configuration.

## Python describes, C++ executes

A `pops` program is a composition of typed descriptors. `HLL()`, `MUSCL(limiter=Minmod())`,
`PoissonProblem(...)`, `AMR(...)`, `RuntimeParam(..., domain=Positive())` are values: they
name a brick and carry its metadata (a native C++ id, requirements, capabilities, options),
they validate their own inputs, and they lower to the IR the codegen and runtime consume.
They compute nothing in Python; there is no numpy loop on the hot path, and the GPU / MPI
backends are preserved end to end.

This is the opposite of a YAML config inflated into Python. A dictionary of strings is opaque:
nothing checks that `"hllc"` is spelled right, that the model can supply the contact speed it
needs, or that the requested backend exists, until the run fails deep in C++. A typed object
checks at authoring time, autocompletes in an editor, and points at the one C++ symbol it
selects.

## Strings name, types choose

The dividing line is simple:

- A **string names a user object**: the name of a field (`"phi"`), a runtime parameter
  (`"nu"`), a block, a coupling. These are labels the user invents; a string is the right type.
- A **typed object chooses an algorithm, a backend, or a layout**: the Riemann solver, the
  reconstruction, the limiter, the mesh layout, the output format, the math mode. These are
  fixed choices the library offers; a typed object is the right type.

So `RuntimeParam("nu", ...)` takes a string (the user named the parameter `nu`), but the
Riemann solver is `HLL()`, not `riemann="hll"`.

## No YAML disguised as Python

A string algorithm selector is rejected, not silently accepted. The helper
`pops.descriptors.reject_string_selector` raises with the typed replacement spelled out:

```python
from pops.descriptors import reject_string_selector

reject_string_selector("hll", "riemann", "pops.numerics.riemann.HLL()")
# TypeError: String algorithm selector rejected: riemann='hll'.
#            Use pops.numerics.riemann.HLL().
```

The error names the offending selector and the typed form to use instead, so the fix is
mechanical. See {doc}`../reference/migration-strings-to-typed` for the full old-string ->
typed-object table.

## Availability is explicit

A typed brick that the core does not yet wire reports `available=False` with an empty
`native_id` rather than a fabricated symbol, and a descriptor can answer an
{doc}`Availability <../reference/spec5-packages>` query (`yes` / `no` / `partial`) that names
what is missing. A combination outside the {doc}`capability matrix <../reference/spec5-packages>`
raises an explicit error on the C++ side; it is never silently ignored.

For the per-package surface and runnable examples, see
{doc}`../reference/spec5-packages`.
