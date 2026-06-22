"""Named physical-model sources: m.source_term and m.linear_source (ADC-400, Phase 1).

Pure Python (adc.dsl); no compilation. Covers the spec "Extension de adc.dsl.Model" tests 1-8
plus the cache-key contract:

  - source_term / linear_source declaration + validation (dimensions, names, collisions);
  - linear_source coefficients must be linear in U (no conservative/primitive dependency);
  - m.source([...]) stays backward compatible and is equivalent to source_term("default", [...]);
  - an old stepper requesting a total source on a multiple-named-source model is rejected
    (never summed implicitly);
  - the model hash folds named sources in CONDITIONALLY: a model without any named source keeps
    a byte-identical hash to before the feature (golden), while changing a named source or a
    linear_source matrix invalidates the cache.

Run with python3 (PYTHONPATH = built adc package).
"""
import numpy as np

from adc import dsl

# Golden hashes computed on master BEFORE this feature (build() / build(with_source=False)
# below). The feature must NOT perturb the cache key of a model that uses only m.source(...).
GOLDEN_WITH_SOURCE = "bde07b2a0b83e85e65a2c73e839b0abfbb38eae0572fbfddd359a2ac9a92c89e"
GOLDEN_NO_SOURCE = "0bcb5c54153956f207d0d2ff7e2a6c542c6f0cf1741aee03e49bc35daacbe3d1"


def build(with_source=True):
    """Canonical 3-variable electrostatic model (rho, rho_u, rho_v), aux grad_x/grad_y.

    Mirrors the golden computed on master with only m.source(...); used to assert the named-source
    feature leaves the existing cache key untouched."""
    m = dsl.Model("es3")
    rho, mx, my = m.conservative_vars("rho", "rho_u", "rho_v")
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    m.flux(x=[mx, mx * mx / rho, mx * my / rho], y=[my, mx * my / rho, my * my / rho])
    m.wave_speeds(x=(0.0 * rho, 0.0 * rho), y=(0.0 * rho, 0.0 * rho))
    if with_source:
        m.source([0.0 * rho, -rho * gx, -rho * gy])
    return m


def base_model():
    """3-variable carrier with primitives and aux declared (rho, rho_u, rho_v; u, v; grad, B_z)."""
    m = dsl.Model("nm")
    rho, mx, my = m.conservative_vars("rho", "rho_u", "rho_v")
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    bz = m.aux("B_z")
    m.flux(x=[mx, mx * mx / rho, mx * my / rho], y=[my, mx * my / rho, my * my / rho])
    m.wave_speeds(x=(0.0 * rho, 0.0 * rho), y=(0.0 * rho, 0.0 * rho))
    return m, dict(rho=rho, mx=mx, my=my, u=u, v=v, gx=gx, gy=gy, bz=bz)


def expect_error(fn, needle):
    try:
        fn()
    except ValueError as e:
        assert needle in str(e), "wrong message: %r (expected substring %r)" % (str(e), needle)
        return
    raise SystemExit("expected ValueError containing %r, none raised" % needle)


def test_source_term_valid():
    m, V = base_model()
    m.source_term("electric", [0.0 * V["rho"], -V["rho"] * V["gx"], -V["rho"] * V["gy"]])
    assert m.check()
    print("OK  1. source_term valid (3 components, check passes)")


def test_source_term_wrong_dim():
    m, V = base_model()
    expect_error(lambda: m.source_term("electric", [0.0 * V["rho"], -V["rho"] * V["gx"]]),
                 "2 expressions for 3")
    print("OK  2. source_term wrong dimension rejected")


def test_linear_source_valid():
    m, V = base_model()
    bz = V["bz"]
    m.linear_source("lorentz", matrix=[[0.0, 0.0, 0.0],
                                       [0.0, 0.0, bz],
                                       [0.0, -bz, 0.0]])
    assert m.check()
    print("OK  3. linear_source valid (3x3, aux+const coefficients)")


def test_linear_source_non_square():
    m, V = base_model()
    expect_error(lambda: m.linear_source("lorentz", matrix=[[0.0, 0.0],
                                                            [0.0, 0.0],
                                                            [0.0, 0.0]]),
                 "3x3")
    print("OK  4. linear_source non-square rejected")


def test_linear_source_depends_on_cons():
    m, V = base_model()
    rho = V["rho"]
    expect_error(lambda: m.linear_source("bad", matrix=[[0.0, 0.0, 0.0],
                                                        [0.0, 0.0, rho],
                                                        [0.0, -rho, 0.0]]),
                 "must not depend on conservative or primitive variables")
    print("OK  5. linear_source depending on a conservative variable rejected")


def test_linear_source_depends_on_prim():
    m, V = base_model()
    u = V["u"]
    expect_error(lambda: m.linear_source("bad", matrix=[[u, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0]]),
                 "must not depend on conservative or primitive variables")
    print("OK  6. linear_source depending on a primitive variable rejected")


def test_source_backward_compat():
    # Existing m.source(...) path is untouched: a model using only m.source keeps the byte-identical
    # golden hash, and check() still passes.
    m = build(with_source=True)
    assert m.check()
    assert m._model_hash() == GOLDEN_WITH_SOURCE, m._model_hash()
    assert build(with_source=False)._model_hash() == GOLDEN_NO_SOURCE
    print("OK  7. m.source backward compatible (golden cache key preserved)")


def test_old_stepper_multiple_named_sources():
    # Named non-default sources with NO explicit default: an old stepper asking for the total source
    # (the numpy evaluator m.eval_source) must reject rather than silently summing.
    m, V = base_model()
    rho, gx, gy = V["rho"], V["gx"], V["gy"]
    m.source_term("electric", [0.0 * rho, -rho * gx, -rho * gy])
    m.source_term("chemistry", [0.0 * rho, 0.0 * rho, 0.0 * rho])
    U = np.ones((3, 4, 4))
    aux = {"grad_x": np.zeros((4, 4)), "grad_y": np.zeros((4, 4)), "B_z": np.zeros((4, 4))}
    expect_error(lambda: m.eval_source(U, aux),
                 "model has multiple named sources")
    print("OK  8. old stepper + multiple named sources rejected")


def test_source_term_default_equiv_source():
    a = build(with_source=True)
    b = build(with_source=False)
    rho = dsl.Var("rho", "cons")
    gx, gy = dsl.Var("grad_x", "aux"), dsl.Var("grad_y", "aux")
    b.source_term("default", [0.0 * rho, -rho * gx, -rho * gy])
    assert b._model_hash() == a._model_hash() == GOLDEN_WITH_SOURCE
    print("OK  9. source_term('default', ...) == m.source(...) (same cache key)")


def test_hash_invalidation_named_source():
    m1, V1 = base_model()
    m1.source_term("electric", [0.0 * V1["rho"], -V1["rho"] * V1["gx"], -V1["rho"] * V1["gy"]])
    m2, V2 = base_model()
    m2.source_term("electric", [0.0 * V2["rho"], -2.0 * V2["rho"] * V2["gx"], -V2["rho"] * V2["gy"]])
    assert m1._model_hash() != m2._model_hash()
    # adding a named source changes the hash versus the same model without it
    m0, _ = base_model()
    assert m0._model_hash() != m1._model_hash()
    print("OK  10. changing a source_term invalidates the cache")


def test_hash_invalidation_linear_source():
    m1, V1 = base_model()
    m1.linear_source("lorentz", matrix=[[0.0, 0.0, 0.0],
                                        [0.0, 0.0, V1["bz"]],
                                        [0.0, -V1["bz"], 0.0]])
    m2, V2 = base_model()
    m2.linear_source("lorentz", matrix=[[0.0, 0.0, 0.0],
                                        [0.0, 0.0, 2.0 * V2["bz"]],
                                        [0.0, -V2["bz"], 0.0]])
    assert m1._model_hash() != m2._model_hash()
    print("OK  11. changing a linear_source matrix invalidates the cache")


def test_name_collisions():
    m, V = base_model()
    m.source_term("electric", [0.0 * V["rho"], 0.0 * V["rho"], 0.0 * V["rho"]])
    expect_error(lambda: m.source_term("electric", [0.0 * V["rho"], 0.0 * V["rho"], 0.0 * V["rho"]]),
                 "already declared")
    expect_error(lambda: m.linear_source("electric", matrix=[[0.0, 0.0, 0.0],
                                                            [0.0, 0.0, 0.0],
                                                            [0.0, 0.0, 0.0]]),
                 "collides with a source_term")
    print("OK  12. duplicate / colliding names rejected")


def main():
    test_source_term_valid()
    test_source_term_wrong_dim()
    test_linear_source_valid()
    test_linear_source_non_square()
    test_linear_source_depends_on_cons()
    test_linear_source_depends_on_prim()
    test_source_backward_compat()
    test_old_stepper_multiple_named_sources()
    test_source_term_default_equiv_source()
    test_hash_invalidation_named_source()
    test_hash_invalidation_linear_source()
    test_name_collisions()
    print("test_dsl_named_sources : tout est vert")


if __name__ == "__main__":
    main()
