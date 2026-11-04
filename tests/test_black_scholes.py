import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.black_scholes import (
    bs_call_price, bs_call_delta, bs_call_gamma, call_payoff,
)


@pytest.mark.parametrize("r,expected", [(0.0, 7.965567455405804),
                                       (0.05, 10.450583572185565)])
def test_known_prices(r, expected):
    assert bs_call_price(100, 100, r, 0.2, 1) == pytest.approx(expected, abs=1e-11)


def test_known_greeks_and_scalar_type():
    assert bs_call_delta(100) == pytest.approx(0.539827837277029, abs=1e-12)
    assert bs_call_gamma(100) == pytest.approx(0.0198476273738506, abs=1e-12)
    assert isinstance(bs_call_price(100), float)


def test_maturity_and_degenerate_limits_without_warnings():
    spots = np.array([0.0, 80.0, 100.0, 120.0])
    with np.errstate(all="raise"):
        for vol, tau in [(0.2, 0.0), (0.0, 1.0)]:
            assert_allclose(bs_call_price(spots, sigma=vol, tau=tau), [0, 0, 0, 20])
            assert_allclose(bs_call_delta(spots, sigma=vol, tau=tau), [0, 0, 0.5, 1])
            assert_allclose(bs_call_gamma(spots, sigma=vol, tau=tau), 0)
    assert_allclose(call_payoff(spots), [0, 0, 0, 20])
    assert bs_call_price(100, r=0.05, sigma=0) == pytest.approx(100-100*np.exp(-0.05))
    assert bs_call_delta(100, r=0.05, sigma=0) == 1.0


def test_broadcasting_bounds_and_positive_gamma():
    spots = np.array([[70.0], [100.0], [130.0]])
    times = np.array([0.1, 0.5, 1.0])
    price = bs_call_price(spots, r=0.03, tau=times)
    delta = bs_call_delta(spots, r=0.03, tau=times)
    gamma = bs_call_gamma(spots, r=0.03, tau=times)
    assert price.shape == delta.shape == gamma.shape == (3, 3)
    assert np.all((delta >= 0) & (delta <= 1))
    assert np.all(gamma > 0)
    assert np.all(price <= spots)
    assert np.all(price >= np.maximum(spots-100*np.exp(-0.03*times), 0)-1e-12)


def test_mixed_limits_and_tiny_positive_time():
    with np.errstate(divide="raise", invalid="raise"):
        assert_allclose(bs_call_price([0, 100, 120], tau=[1, 0, 1]),
                        [0, 0, bs_call_price(120)])
        delta = bs_call_delta([80, 100, 120], tau=1e-12)
        assert_allclose(delta, [0, 0.5, 1], atol=1e-7)
        assert np.isfinite(bs_call_gamma(100, tau=1e-12))


def test_finite_difference_sensitivities():
    spots = np.array([85.0, 100.0, 120.0])
    h = 0.01
    base = bs_call_price(spots, r=0.04, sigma=0.3, tau=0.7)
    up = bs_call_price(spots+h, r=0.04, sigma=0.3, tau=0.7)
    down = bs_call_price(spots-h, r=0.04, sigma=0.3, tau=0.7)
    assert_allclose((up-down)/(2*h), bs_call_delta(spots, r=0.04, sigma=0.3, tau=0.7),
                    rtol=1e-7, atol=1e-8)
    assert_allclose((up-2*base+down)/h**2, bs_call_gamma(spots, r=0.04, sigma=0.3, tau=0.7),
                    rtol=2e-6, atol=1e-8)


@pytest.mark.parametrize("kwargs", [
    {"spot": -1}, {"strike": 0}, {"sigma": -0.1}, {"tau": -1},
    {"spot": np.nan}, {"r": np.inf}, {"tau": [1.0, np.nan]},
])
@pytest.mark.parametrize("function", [bs_call_price, bs_call_delta, bs_call_gamma])
def test_invalid_inputs(function, kwargs):
    with pytest.raises(ValueError):
        function(**({"spot": 100} | kwargs))


@pytest.mark.parametrize("spot,strike", [(-1, 100), (100, 0), (np.nan, 100)])
def test_invalid_payoff(spot, strike):
    with pytest.raises(ValueError):
        call_payoff(spot, strike)
