"""Spec 3 unified scheduler: a held field solve + sub-cycled transport (authoring).

The elliptic field solve is expensive, so hold it and refresh only every N steps; the cheap
transport runs every step. This shows the schedule AUTHORING surface (pops.time.every/hold/
subcycle), the cacheable-capability validation, and the honest refusal to lower a non-always
schedule (the cache / accumulate_dt / checkpoint RUNTIME is the C++ part of ADC-458). It builds
the IR and inspects it; it does not run a simulation.

Run: python3 examples/spec3/scheduled_fields_subcycled_transport.py
"""
import pops.model as model
import pops.time as adctime
from pops import dsl


def plasma_module():
    mod = model.Module("scheduled_plasma")
    u = mod.state_space("U", ("rho", "mx", "my"))
    fields = mod.field_space("fields", ("phi",))
    rho = dsl.Var("rho", "cons")
    mod.operator(name="fields_from_state", signature=(u,) >> fields,
                 kind="field_operator", expr=rho)
    mod.operator(name="flux", signature=(u,) >> model.Rate(u), kind="grid_operator",
                 expr={"x": [rho, rho, rho], "y": [rho, rho, rho]})
    # the Poisson field solve is cacheable: holding a stale phi between refreshes is allowed
    mod.operator_capabilities("fields_from_state", cacheable=True)
    return mod, u


def main():
    mod, u = plasma_module()

    # the cheap transport refreshes the field every 10 steps and holds it in between
    P = adctime.Program("scheduled_step").bind_operators(mod)
    U = P.state("plasma", space=u)
    fields = P.call("fields_from_state", U, schedule=adctime.every(10).hold())
    transport = P.call("flux", U, schedule=adctime.subcycle(4))
    print("field solve schedule :", fields.attrs["schedule"])
    print("transport schedule   :", transport.attrs["schedule"])
    print("\noperator-first IR (schedules are recorded + inspectable):")
    print(P.dump_operator_ir())

    # holding a NON-cacheable operator is rejected at authoring time
    try:
        P.call("flux", U, schedule=adctime.every(10).hold())
    except ValueError as exc:
        print("\nrejected hold on a non-cacheable operator:\n  ", exc)

    # a non-always schedule is recorded but refuses to silently lower to a no-op (ADC-458)
    try:
        P._check_schedules_lowerable()
    except NotImplementedError as exc:
        print("\nrefuses to lower until the ADC-458 runtime lands:\n  ", str(exc)[:96], "...")

    print("\nOK: schedules authored + validated; runtime is ADC-458.")


if __name__ == "__main__":
    main()
