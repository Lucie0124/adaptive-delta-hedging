import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from src.metrics import value_at_risk, expected_shortfall
from src.risk_metrics import paired_bootstrap, bootstrap_interval


@pytest.mark.parametrize("losses,alpha,var,es", [
    ([1, 2, 3, 4], 0.5, 2, 3.5),
    ([1, 2, 3, 4], 0.625, 3, 11/3),  # tail mass 1.5: (4+0.5*3)/1.5
    ([0, 0, 0, 10], 0.5, 0, 5),      # do not include all VaR ties
    ([1, 2, 3, 4], 0.95, 4, 4),
    ([-5, -4, -3, -2], 0.5, -4, -2.5),
    ([7], 0.95, 7, 7),
    (list(range(1, 21)), 0.95, 19, 20),
])
def test_empirical_quantile_and_exact_tail_probability_mass(losses, alpha, var, es):
    assert value_at_risk(losses, alpha) == pytest.approx(var)
    assert expected_shortfall(losses, alpha) == pytest.approx(es)


def test_loss_sign_translation_scale_and_no_double_counting():
    gross = np.array([-3, -1, 0, 2.0])
    cost = np.full(4, 0.5)
    net = gross-cost
    assert expected_shortfall(-net, 0.5) == pytest.approx(2.5)
    assert expected_shortfall(-net+10, 0.5) == pytest.approx(12.5)
    assert expected_shortfall(-net*3, 0.5) == pytest.approx(7.5)
    # A positive P&L is a negative loss, not an absolute error.
    assert expected_shortfall(-np.array([1, 2, 3]), 0.5) < 0


def test_bootstrap_preserves_pairing_and_is_reproducible():
    x = np.linspace(-2, 5, 80)
    losses = np.column_stack([x, x+2])
    costs = np.column_stack([np.abs(x), np.abs(x)+1])
    samples = paired_bootstrap(losses, costs, n_bootstrap=50, seed=15)
    assert_allclose(samples["es"][:, 1]-samples["es"][:, 0], 2)
    assert_allclose(samples["mean_cost"][:, 1]-samples["mean_cost"][:, 0], 1)
    again = paired_bootstrap(losses, costs, n_bootstrap=50, seed=15)
    assert_array_equal(samples["es"], again["es"])
    assert not np.array_equal(samples["es"], paired_bootstrap(losses, costs, n_bootstrap=50, seed=16)["es"])
    interval = bootstrap_interval(samples["es"][:, 1]-samples["es"][:, 0])
    assert interval["ci_low"] == pytest.approx(2)
    assert interval["ci_high"] == pytest.approx(2)


def test_bootstrap_uses_same_es_definition_for_fractional_tail():
    losses = np.array([[0, 2], [1, 3], [2, 4], [9, 11.0]])
    samples = paired_bootstrap(losses, np.zeros_like(losses), confidence=0.625,
                               n_bootstrap=10, seed=8)
    rng = np.random.default_rng(8)
    for i in range(10):
        indices = rng.integers(0, 4, size=4)
        for j in range(2):
            assert samples["es"][i, j] == pytest.approx(expected_shortfall(losses[indices, j], 0.625))


@pytest.mark.parametrize("function", [value_at_risk, expected_shortfall])
@pytest.mark.parametrize("losses,alpha", [([], 0.95), ([np.nan], 0.95), ([[1, 2]], 0.95),
                                        ([1, 2], 0), ([1, 2], 1), ([1, 2], np.nan)])
def test_invalid_risk_inputs(function, losses, alpha):
    with pytest.raises(ValueError):
        function(losses, alpha)


def test_invalid_bootstrap_inputs():
    with pytest.raises(ValueError):
        paired_bootstrap([[1]], [[0]], n_bootstrap=10)
    with pytest.raises(ValueError):
        paired_bootstrap([[1], [2]], [[-1], [0]], n_bootstrap=10)
    with pytest.raises(ValueError):
        paired_bootstrap([[1], [2]], [[0], [0]], n_bootstrap=1)
