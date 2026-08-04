import jax.numpy as jnp
import pytest

from genepax.gp.functions import (
    _protected_division,
    _protected_power,
    function_set_numeric,
    max_abs_value,
)


@pytest.mark.parametrize("function", function_set_numeric.values())
def test_numeric_gp_functions_are_finite_and_bounded(function):
    x = jnp.asarray([jnp.nan, jnp.inf, -jnp.inf, 0.0, 1e30, -1e30])
    y = jnp.asarray([0.5, 0.0, -0.5, jnp.nan, jnp.inf, -jnp.inf])

    values = function(x, y)

    assert jnp.all(jnp.isfinite(values))
    assert jnp.all(jnp.abs(values) <= max_abs_value)


def test_protected_division_handles_zero_and_extreme_denominators():
    values = _protected_division(
        jnp.asarray([1.0, 1e30, -1e30]),
        jnp.asarray([0.0, 1e-30, -1e-30]),
    )

    assert jnp.array_equal(values, jnp.zeros(3))


def test_protected_power_is_real_for_negative_fractional_base():
    values = _protected_power(
        jnp.asarray([-4.0, -1.0, 0.0, 4.0]),
        jnp.asarray([0.5, -2.5, -10.0, 100.0]),
    )

    assert jnp.all(jnp.isfinite(values))
    assert values[0] < 0
    assert values[1] < 0
    assert values[2] == 0
    assert values[3] <= max_abs_value
