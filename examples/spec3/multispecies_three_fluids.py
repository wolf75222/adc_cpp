"""Spec 3 generic multi-species: three fluids over the operator-first multi-block kernel.

No species is hardcoded: each is a named block of a StateSpace. This builds a 3-species
time step with the operator-first kernel (pops.model multi-state spaces + RateBundle for a
typed multi-output coupling + pops.time multi-block Program + commit_many). It builds the IR
and checks structure; it does not run a simulation. The blackboard sugar for this
(m.species for N>1, m.coupled_rate) and the multi-block field-solve / coupled-rate RUNTIME
are tracked by ADC-457.

Run: python3 examples/spec3/multispecies_three_fluids.py
"""
import pops.model as model
import pops.time as adctime


def species_spaces():
    """Three species, each a StateSpace -- the core knows BlockInstances, not 'electrons'."""
    e = model.StateSpace("electron_state", ["ne", "mex", "mey"],
                         roles={"ne": "Density", "mex": "MomentumX", "mey": "MomentumY"})
    i = model.StateSpace("ion_state", ["ni", "mix", "miy"])
    n = model.StateSpace("neutral_state", ["nn", "mnx", "mny"])
    return e, i, n


def collision_bundle(e, i, n):
    """A coupled collision operator's typed multi-output: one Rate per species (arity 3)."""
    coll = model.RateBundle({"electrons": model.Rate(e), "ions": model.Rate(i),
                             "neutrals": model.Rate(n)})
    # the bundle is typed: a wrong rate on a wrong state is rejected
    coll.require("electrons", e)
    coll.require("ions", i)
    coll.require("neutrals", n)
    return coll


def multi_species_step():
    """A forward step coupling three blocks through a shared field solve + per-species rates,
    committed atomically. IR-level (the coupled field-solve runtime is ADC-457)."""
    P = adctime.Program("three_fluids_step")
    dt = P.dt
    e_n = P.state("electrons")
    i_n = P.state("ions")
    n_n = P.state("neutrals")

    fields = P.solve_fields_from_blocks([e_n, i_n, n_n], name="fields")  # coupled, arity 3
    e1 = P.linear_combine("e1", e_n + dt * P.rhs(name="Re", state=e_n, fields=fields, flux=True))
    i1 = P.linear_combine("i1", i_n + dt * P.rhs(name="Ri", state=i_n, fields=fields, flux=True))
    n1 = P.linear_combine("n1", n_n + dt * P.rhs(name="Rn", state=n_n, fields=fields, flux=True))

    P.commit_many({"electrons": e1, "ions": i1, "neutrals": n1})  # atomic multi-block commit
    return P


def main():
    e, i, n = species_spaces()
    coll = collision_bundle(e, i, n)
    print("species:", [s.name for s in (e, i, n)])
    print("RateBundle arity:", len(coll), "->", coll.keys())
    try:
        coll.require("electrons", i)   # wrong StateSpace
    except TypeError as exc:
        print("typed multi-output rejects a wrong rate:", str(exc)[:70], "...")

    P = multi_species_step()
    assert set(P.commits()) == {"electrons", "ions", "neutrals"}
    fields = next(v for v in P._values if v.op == "solve_fields_from_blocks")
    assert len(fields.inputs) == 3
    print("committed blocks:", sorted(P.commits()))
    print("coupled field solve inputs:", len(fields.inputs))
    print("\nOK: 3 species, no hardcoding, typed RateBundle, atomic commit_many.")


if __name__ == "__main__":
    main()
