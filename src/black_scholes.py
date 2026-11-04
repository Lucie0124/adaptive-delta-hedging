"""No-dividend European calls; time arguments are remaining years.

Inputs broadcast following NumPy rules. At maturity or zero volatility, Delta
at the payoff kink is 0.5 and Gamma is reported as zero, not as a derivative
at that kink. No artificial lower bound is applied to time or volatility.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

FloatArray = NDArray[np.float64]
ScalarOrArray = float | FloatArray


def _output(value: FloatArray) -> ScalarOrArray:
    return float(value) if value.ndim == 0 else value


def _inputs(
    spot: ArrayLike, strike: ArrayLike, r: ArrayLike,
    sigma: ArrayLike, tau: ArrayLike,
) -> tuple[FloatArray, ...]:
    values = tuple(np.broadcast_arrays(*[
        np.asarray(x, dtype=float) for x in (spot, strike, r, sigma, tau)
    ]))
    if any(np.any(~np.isfinite(x)) for x in values):
        raise ValueError("Black-Scholes inputs must be finite.")
    s, k, _, vol, time = values
    if np.any(s < 0) or np.any(k <= 0):
        raise ValueError("spot must be nonnegative and strike strictly positive.")
    if np.any(vol < 0) or np.any(time < 0):
        raise ValueError("sigma and tau must be nonnegative.")
    return values


def call_payoff(spot: ArrayLike, strike: ArrayLike = 100.0) -> ScalarOrArray:
    """Return (spot - strike)^+ for nonnegative spot and positive strike."""
    s, k, _, _, _ = _inputs(spot, strike, 0.0, 0.0, 0.0)
    return _output(np.maximum(s - k, 0.0))


def bs_call_price(
    spot: ArrayLike, strike: ArrayLike = 100.0, r: ArrayLike = 0.0,
    sigma: ArrayLike = 0.20, tau: ArrayLike = 1.0,
) -> ScalarOrArray:
    """Risk-neutral call value; tau=0 returns payoff, sigma=0 its PV limit."""
    s, k, rate, vol, time = _inputs(spot, strike, r, sigma, tau)
    discounted_strike = k * np.exp(-rate * time)
    price = np.array(np.maximum(s - discounted_strike, 0.0), copy=True)
    regular = (s > 0) & (vol > 0) & (time > 0)
    width = vol[regular] * np.sqrt(time[regular])
    d1 = ((np.log(s[regular]) - np.log(k[regular]))
          + (rate[regular] + 0.5 * vol[regular] ** 2) * time[regular]) / width
    price[regular] = (s[regular] * ndtr(d1)
                      - discounted_strike[regular] * ndtr(d1 - width))
    # Suppress tiny negative roundoff when both terms are close to zero.
    return _output(np.maximum(price, 0.0))


def bs_call_delta(
    spot: ArrayLike, strike: ArrayLike = 100.0, r: ArrayLike = 0.0,
    sigma: ArrayLike = 0.20, tau: ArrayLike = 1.0,
) -> ScalarOrArray:
    """Shares per long call; hold positive Delta to hedge one short call.

    At tau=0 or sigma=0 use the step at the discounted strike, with 0.5 at
    equality. At the absorbing spot=0 boundary return zero.
    """
    s, k, rate, vol, time = _inputs(spot, strike, r, sigma, tau)
    boundary = k * np.exp(-rate * time)
    delta = np.where(s > boundary, 1.0, np.where(s == boundary, 0.5, 0.0))
    regular = (s > 0) & (vol > 0) & (time > 0)
    d1 = ((np.log(s[regular]) - np.log(k[regular]))
          + (rate[regular] + 0.5 * vol[regular] ** 2) * time[regular]) / (
              vol[regular] * np.sqrt(time[regular]))
    delta[regular] = ndtr(d1)
    return _output(np.where(s == 0, 0.0, delta))


def bs_call_gamma(
    spot: ArrayLike, strike: ArrayLike = 100.0, r: ArrayLike = 0.0,
    sigma: ArrayLike = 0.20, tau: ArrayLike = 1.0,
) -> ScalarOrArray:
    """Call curvature per share-price unit.

    Report zero at tau=0, sigma=0 or spot=0. The payoff derivative is undefined
    at its kink; the reported zero is a convention, not finite kink curvature.
    """
    s, k, rate, vol, time = _inputs(spot, strike, r, sigma, tau)
    gamma = np.zeros_like(s)
    regular = (s > 0) & (vol > 0) & (time > 0)
    width = vol[regular] * np.sqrt(time[regular])
    d1 = ((np.log(s[regular]) - np.log(k[regular]))
          + (rate[regular] + 0.5 * vol[regular] ** 2) * time[regular]) / width
    gamma[regular] = np.exp(-0.5 * d1 ** 2) / (
        np.sqrt(2.0 * np.pi) * s[regular] * width)
    return _output(gamma)
