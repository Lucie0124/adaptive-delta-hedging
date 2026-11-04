"""Exact GBM sampling and a separate risk-neutral pricing diagnostic."""

from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from .black_scholes import call_payoff


class PricingEstimate(TypedDict):
    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    confidence: float
    n_paths: int


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def simulate_gbm_paths(
    s0: float = 100.0, mu: float = 0.05, sigma_true: float = 0.20,
    maturity: float = 1.0, n_steps: int = 252, n_paths: int = 2000,
    seed: int = 42, *, shocks: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Sample physical-measure GBM with shape (n_paths, n_steps + 1).

    Each row starts at s0. Uses a fresh local Generator, never global random
    state. If shocks is supplied it must have shape (n_paths, n_steps), contain
    finite standard-normal realisations, and replaces random sampling entirely
    (seed is then unused). Zero volatility and zero maturity are supported.
    """
    _positive_integer(n_steps, "n_steps")
    _positive_integer(n_paths, "n_paths")
    if not np.all(np.isfinite([s0, mu, sigma_true, maturity])):
        raise ValueError("GBM parameters must be finite.")
    if s0 <= 0 or sigma_true < 0 or maturity < 0:
        raise ValueError("s0 must be positive; sigma_true and maturity nonnegative.")
    if shocks is None:
        z = np.random.default_rng(seed).standard_normal((n_paths, n_steps))
    else:
        z = np.asarray(shocks, dtype=float)
        if z.shape != (n_paths, n_steps) or not np.all(np.isfinite(z)):
            raise ValueError("shocks must be finite with shape (n_paths, n_steps).")
    dt = maturity / n_steps
    log_increments = (mu - 0.5 * sigma_true ** 2) * dt + sigma_true * np.sqrt(dt) * z
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = s0
    with np.errstate(over="raise", under="raise", invalid="raise"):
        paths[:, 1:] = s0 * np.exp(np.cumsum(log_increments, axis=1))
    if not np.all(np.isfinite(paths)) or np.any(paths <= 0):
        raise FloatingPointError("GBM paths overflowed or underflowed; review parameters.")
    return paths


def risk_neutral_call_price_mc(
    s0: float = 100.0, strike: float = 100.0, r: float = 0.0,
    sigma: float = 0.20, maturity: float = 1.0, n_paths: int = 2000,
    seed: int = 314159, confidence: float = 0.95,
) -> PricingEstimate:
    """Price a call using iid discounted Q payoffs, with a normal-CLT interval.

    GBM's exact terminal law needs one step only; physical drift is not an
    argument. Pricing uses its own paths and seed, separate from hedge scenarios.
    The interval is asymptotic, not a guarantee for every finite sample.
    """
    _positive_integer(n_paths, "n_paths")
    if n_paths < 2 or not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("Need at least two paths and 0 < confidence < 1.")
    if not np.isfinite(r):
        raise ValueError("r must be finite.")
    # Validate strike before doing potentially expensive simulation.
    call_payoff(s0, strike)
    terminal = simulate_gbm_paths(s0, r, sigma, maturity, 1, n_paths, seed)[:, -1]
    values = np.exp(-r * maturity) * np.asarray(call_payoff(terminal, strike))
    estimate = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(n_paths))
    width = float(norm.ppf(0.5 + confidence / 2)) * se
    return PricingEstimate(estimate=estimate, standard_error=se,
                           ci_low=estimate-width, ci_high=estimate+width,
                           confidence=confidence, n_paths=n_paths)
