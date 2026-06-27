"""Spec 3 multi-output operators: a coupled_rate operator returning a RateBundle.

A coupled operator (a collision term) takes an arbitrary arity of states and returns one typed
rate per participating species. RateBundle.require enforces that each block's rate lives over its
own StateSpace. A coupled_rate operator lowers, via P.call, to a bundle whose per-block rates
compose like any RHS; the coupled-rate KERNEL codegen is the deferred ADC-457 runtime.

Run: python3 examples/spec3/rate_bundle_collisions.py
"""
import pops.model as model
import pops.time as adctime
from pops.ir.expr import Var


def build_bundle():
    """The typed multi-output, of arbitrary arity (here 3 species)."""
    e = model.StateSpace("electron_state", ["ne", "mex", "mey"])
    i = model.StateSpace("ion_state", ["ni", "mix", "miy"])
    n = model.StateSpace("neutral_state", ["nn", "mnx", "mny"])
    collisions = model.RateBundle({
        "electrons": model.Rate(e),
        "ions": model.Rate(i),
        "neutrals": model.Rate(n),
    })
    return collisions, (e, i, n)


def coupled_module():
    """A two-fluid module with a coupled_rate operator (e, i) -> RateBundle."""
    mod = model.Module("two_fluid_collisions")
    e = mod.state_space("electron_state", ("ne", "mex", "mey"))
    i = mod.state_space("ion_state", ("ni", "mix", "miy"))
    bundle = model.RateBundle({"electrons": model.Rate(e), "ions": model.Rate(i)})
    ne, ni = Var("ne", "cons"), Var("ni", "cons")
    mod.operator(name="collision", signature=model.Signature((e, i), bundle),
                 kind="coupled_rate",
                 expr={"electrons": [ni - ne, ne, ne], "ions": [ne - ni, ni, ni]})
    return mod, e, i


def main():
    collisions, (e, i, n) = build_bundle()
    print("RateBundle arity:", len(collisions), "->", collisions.keys())
    for block, state in (("electrons", e), ("ions", i), ("neutrals", n)):
        print("  %-10s -> %r (ok)" % (block, collisions.require(block, state)))
    try:
        collisions.require("electrons", i)  # wrong StateSpace
    except TypeError as exc:
        print("rejected wrong rate on wrong state:", str(exc)[:60], "...")

    # a coupled_rate OPERATOR: P.call returns one rate per species, each composes like any RHS
    mod, es, is_ = coupled_module()
    P = adctime.Program("collision_step").bind_operators(mod)
    dt = P.dt
    e_n, i_n = P.state("electrons", space=es), P.state("ions", space=is_)
    C = P.call("collision", e_n, i_n)
    P.commit_many({"electrons": P.linear_combine("e1", e_n + dt * C["electrons"]),
                   "ions": P.linear_combine("i1", i_n + dt * C["ions"])})
    print("\noperator-first IR (one coupled node + a per-block projection each):")
    print(P.dump_operator_ir())
    try:
        P._check_lowerable(None)
    except NotImplementedError as exc:
        print("\ncoupled-rate codegen deferred (ADC-457):", str(exc)[:72], "...")
    print("\nOK: coupled_rate authored, typed, composed, committed; kernel codegen is ADC-457.")


if __name__ == "__main__":
    main()
