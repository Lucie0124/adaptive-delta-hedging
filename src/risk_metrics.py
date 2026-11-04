"""Paired path bootstrap for empirical ES and expected execution costs."""

from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .metrics import _tail_mean_columns


class BootstrapSamples(TypedDict):
    es: NDArray[np.float64]
    mean_cost: NDArray[np.float64]


def paired_bootstrap(
    losses: ArrayLike, costs: ArrayLike, *, confidence: float = 0.95,
    n_bootstrap: int = 500, seed: int = 731,
) -> BootstrapSamples:
    """Resample iid path rows jointly across every policy/scenario column.

    Matrices have shape (n_paths, n_candidates). Return bootstrap estimates
    shaped (n_bootstrap, n_candidates); one row-index draw is shared by all
    columns on each replicate. Columns may include different fee cases and
    volatility scenarios ONLY if they use the same underlying shocks/paths.
    No Gaussian return or independent-policy assumption is used. Local RNG.
    """
    loss, cost = np.asarray(losses, float), np.asarray(costs, float)
    if (loss.ndim != 2 or loss.shape[0] < 2 or loss.shape[1] < 1
            or cost.shape != loss.shape or not np.all(np.isfinite(loss))
            or not np.all(np.isfinite(cost)) or np.any(cost < 0)):
        raise ValueError("Need matching finite (paths>=2, candidates>=1) losses and nonnegative costs.")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be in (0,1).")
    if (isinstance(n_bootstrap, (bool, np.bool_))
            or not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 2):
        raise ValueError("n_bootstrap must be an integer >= 2.")
    rng = np.random.default_rng(seed)
    es = np.empty((n_bootstrap, loss.shape[1]))
    mean_cost = np.empty_like(es)
    for b in range(n_bootstrap):
        indices = rng.integers(0, loss.shape[0], size=loss.shape[0])
        es[b] = _tail_mean_columns(loss[indices], confidence)
        mean_cost[b] = cost[indices].mean(axis=0)
    return BootstrapSamples(es=es, mean_cost=mean_cost)


def bootstrap_interval(estimates: ArrayLike, interval_level: float = 0.95) -> dict[str, float]:
    """Percentile interval and bootstrap standard error for a scalar estimate.

    These are pointwise, two-sided percentile intervals, not simultaneous
    confidence guarantees for a searched grid. Their upper endpoint is used
    as an additional conservative feasibility diagnostic, not the primary rule.
    """
    x = np.asarray(estimates, float)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("estimates must contain at least two finite bootstrap replicates.")
    if not np.isfinite(interval_level) or not 0 < interval_level < 1:
        raise ValueError("interval_level must be in (0,1).")
    low, high = np.quantile(x, [(1-interval_level)/2, (1+interval_level)/2])
    return {"ci_low": float(low), "ci_high": float(high),
            "standard_error": float(x.std(ddof=1))}
