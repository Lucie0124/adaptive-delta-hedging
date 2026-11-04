import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from src.black_scholes import bs_call_price
from src.monte_carlo import simulate_gbm_paths, risk_neutral_call_price_mc


def test_shape_initial_price_positivity_and_repeatability():
    kwargs = dict(s0=103.0, n_paths=300, n_steps=36, seed=1729)
    paths = simulate_gbm_paths(**kwargs)
    assert paths.shape == (300, 37)
    assert_array_equal(paths[:, 0], 103.0)
    assert np.all(paths > 0)
    assert_array_equal(paths, simulate_gbm_paths(**kwargs))
    assert not np.array_equal(paths, simulate_gbm_paths(**(kwargs | {"seed": 1730})))


def test_does_not_read_or_modify_global_random_state():
    np.random.seed(13)
    before = np.random.get_state()
    paths = simulate_gbm_paths(n_paths=5, n_steps=2)
    after = np.random.get_state()
    assert before[0] == after[0]
    assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    np.random.seed(99)
    assert_array_equal(paths, simulate_gbm_paths(n_paths=5, n_steps=2))


def test_supplied_shocks_exact_law_and_no_input_mutation():
    shocks = np.array([[0.0, 1.0], [-1.0, 0.5]])
    original = shocks.copy()
    paths = simulate_gbm_paths(n_paths=2, n_steps=2, shocks=shocks)
    expected = 100*np.exp(np.cumsum(0.015+0.2*np.sqrt(0.5)*shocks, axis=1))
    assert_allclose(paths[:, 1:], expected, rtol=1e-14)
    assert_array_equal(shocks, original)
    assert_array_equal(paths, simulate_gbm_paths(n_paths=2, n_steps=2, shocks=shocks, seed=900))


def test_log_return_moments_with_statistical_tolerances():
    n, steps, maturity, mu, vol = 20000, 12, 0.75, 0.07, 0.3
    paths = simulate_gbm_paths(mu=mu, sigma_true=vol, maturity=maturity,
                               n_paths=n, n_steps=steps, seed=812)
    increments = np.diff(np.log(paths), axis=1).ravel()
    dt = maturity / steps
    expected_mean, expected_var = (mu-0.5*vol**2)*dt, vol**2*dt
    size = increments.size
    assert abs(increments.mean()-expected_mean) < 5*np.sqrt(expected_var/size)
    assert abs(increments.var(ddof=1)-expected_var) < 5*expected_var*np.sqrt(2/(size-1))
    # Independent terminal stock-mean check using the GBM variance formula.
    expected_stock = 100*np.exp(mu*maturity)
    stock_se = expected_stock*np.sqrt(np.expm1(vol**2*maturity)/n)
    assert abs(paths[:, -1].mean()-expected_stock) < 5*stock_se


@pytest.mark.parametrize("r", [0.0, 0.05, -0.01])
def test_risk_neutral_call_matches_analytical_price_statistically(r):
    estimate = risk_neutral_call_price_mc(r=r, n_paths=120000, seed=730)
    price = bs_call_price(100, r=r)
    # A five-SE numerical regression gate has a much smaller false failure rate
    # than asserting that every 95% confidence interval must cover the price.
    assert abs(estimate["estimate"]-price) < 5*estimate["standard_error"]
    width = 1.959963984540054 * estimate["standard_error"]
    assert estimate["ci_low"] == pytest.approx(estimate["estimate"]-width)
    assert estimate["ci_high"] == pytest.approx(estimate["estimate"]+width)


def test_deterministic_zero_volatility_and_zero_maturity():
    paths = simulate_gbm_paths(mu=0.03, sigma_true=0, n_paths=3, n_steps=4)
    assert_allclose(paths, np.broadcast_to(100*np.exp(0.03*np.linspace(0, 1, 5)), (3, 5)))
    assert_array_equal(simulate_gbm_paths(maturity=0, n_paths=2, n_steps=2), 100)
    estimate = risk_neutral_call_price_mc(r=0.05, sigma=0, n_paths=10)
    assert estimate["estimate"] == pytest.approx(bs_call_price(100, r=0.05, sigma=0))
    assert estimate["standard_error"] == pytest.approx(0, abs=1e-14)


@pytest.mark.parametrize("kwargs", [
    {"s0": 0}, {"mu": np.inf}, {"sigma_true": -0.2}, {"maturity": -1},
    {"n_paths": 0}, {"n_steps": 0}, {"n_steps": 2.5}, {"n_paths": True},
    {"n_paths": 2, "n_steps": 2, "shocks": [[0, 1]]},
    {"n_paths": 1, "n_steps": 1, "shocks": [[np.nan]]},
])
def test_invalid_simulation_arguments(kwargs):
    with pytest.raises(ValueError):
        simulate_gbm_paths(**kwargs)


@pytest.mark.parametrize("kwargs", [{"n_paths": 1}, {"confidence": 1}, {"strike": 0}])
def test_invalid_pricing_arguments(kwargs):
    with pytest.raises(ValueError):
        risk_neutral_call_price_mc(**kwargs)
