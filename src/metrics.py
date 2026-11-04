"""P&L, execution metrics and empirical upper-tail loss measures."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

from .hedging import HedgingResult


def _validate_losses(losses: ArrayLike, confidence: float) -> np.ndarray:
    values = np.asarray(losses, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("losses must be a nonempty finite one-dimensional vector.")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one.")
    return values


def value_at_risk(losses: ArrayLike, confidence: float = 0.95) -> float:
    """Empirical inverse-CDF VaR of LOSSES: order statistic ceil(alpha*n).

    Equal observation weights, no linear quantile interpolation. Positive
    numbers are losses; an all-profitable book may have negative VaR.
    """
    values = _validate_losses(losses, confidence)
    index = int(np.ceil(confidence * values.size))-1
    return float(np.partition(values, index)[index])


def _tail_mean_columns(values: np.ndarray, confidence: float) -> np.ndarray:
    """Quantile-integral ES along axis 0, also shared by paired bootstraps.

    The caller validates finite values and 0 < confidence < 1. Each column
    is a policy and each row an equally weighted path. Partition only the
    required upper tail rather than sorting the entire sample.
    """
    n = values.shape[0]
    mass = (1-confidence)*n
    nearest = round(mass)
    if nearest >= 1 and abs(mass-nearest) < 16*np.finfo(float).eps*n:
        mass = float(nearest)
    full = int(np.floor(mass))
    fraction = mass-full
    boundary = n-full-1
    ordered = np.partition(values, max(boundary, 0), axis=0)
    total = ordered[n-full:].sum(axis=0) if full else np.zeros(values.shape[1:])
    if fraction:
        total = total+fraction*ordered[boundary]
    return np.asarray(total/mass)


def expected_shortfall(losses: ArrayLike, confidence: float = 0.95) -> float:
    """Average worst (1-alpha) probability mass of empirical LOSSES.

    If h=(1-alpha)*n, average the floor(h) largest observations plus the
    fractional part of h times the next largest, divided by h. This equals
    integral_alpha^1 VaR_u du/(1-alpha), including atoms and ties. Averaging
    every value >= VaR is generally wrong for finite/discrete samples.
    Transaction costs already belong in net P&L: supply losses=-pnl_net.
    """
    values = _validate_losses(losses, confidence)
    return float(_tail_mean_columns(values, confidence))


def pnl_statistics(pnl: ArrayLike) -> dict[str, float]:
    """Mean, sample SD, MAE, RMSE and mean SE for finite iid path P&Ls.

    At least two observations are required for sample standard deviation.
    Errors are measured relative to zero P&L, not demeaned P&L.
    """
    values = np.asarray(pnl, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("pnl must be a finite vector with at least two values.")
    sd = float(np.std(values, ddof=1))
    return {"mean_pnl": float(np.mean(values)), "std_pnl": sd,
            "mae": float(np.mean(np.abs(values))),
            "rmse": float(np.sqrt(np.mean(values**2))),
            "se_mean_pnl": float(sd / np.sqrt(values.size))}


def mean_confidence_interval(values: ArrayLike, confidence: float = 0.95) -> dict[str, float]:
    """Asymptotic normal interval for an iid mean, including paired contrasts.

    For two strategies on common paths, supply the per-path differences;
    do not treat the two strategy samples as independent.
    """
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one.")
    stats = pnl_statistics(values)
    mean, se = stats["mean_pnl"], stats["se_mean_pnl"]
    width = float(norm.ppf(0.5+confidence/2))*se
    return {"mean": mean, "standard_error": se, "ci_low": mean-width,
            "ci_high": mean+width}


def summarize_hedging(result: HedgingResult) -> dict[str, str | int | float]:
    """Summarise gross/net P&L, both fee conventions, notional and trade stages.

    All currency metrics are per sold call. Mean rebalances excludes opening
    and liquidation; mean trades includes them. Raw fees differ from their
    terminal value if r is nonzero. Counts refer to executed nonzero trades.
    """
    summary: dict[str, str | int | float] = {
        "strategy": result["strategy"], "n_paths": result["pnl_net"].size,
    }
    for convention in ("gross", "net"):
        for name, value in pnl_statistics(result[f"pnl_{convention}"]).items():
            summary[f"{name}_{convention}"] = value
    for output, source in (
        ("mean_total_transaction_cost", "total_transaction_cost"),
        ("mean_transaction_cost_fv", "transaction_cost_fv"),
        ("mean_traded_notional", "total_traded_notional"),
        ("mean_initial_trades", "n_initial"),
        ("mean_rebalances", "n_rebalances"),
        ("mean_liquidation_trades", "n_liquidation"),
        ("mean_trades", "n_trades"),
    ):
        summary[output] = float(np.mean(result[source]))
    return summary
