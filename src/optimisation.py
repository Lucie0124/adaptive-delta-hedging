"""Transparent finite-candidate cost minimisation under an empirical ES limit.

This module never tunes on evaluation paths. The caller supplies a common
path matrix and freezes calibration selections before generating evaluation
data. Trading and portfolio accounting remain in the validated hedging engine.
"""

from typing import TypedDict

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .hedging import hedge_paths, HedgingResult, Strategy
from .metrics import (summarize_hedging, expected_shortfall, value_at_risk,
                      mean_confidence_interval)
from .risk_metrics import paired_bootstrap, bootstrap_interval, BootstrapSamples


class Candidate(TypedDict):
    policy_id: str
    strategy: Strategy
    epsilon: float | None


def candidate_policies(epsilon_grid: ArrayLike = (0.005, 0.01, 0.02, 0.03, 0.05)) -> list[Candidate]:
    """Daily, weekly and sorted unique adaptive thresholds, always keeping 0.01."""
    grid = np.asarray(epsilon_grid, dtype=float)
    if grid.ndim != 1 or not np.all(np.isfinite(grid)) or np.any(grid < 0):
        raise ValueError("epsilon_grid must be a finite nonnegative vector.")
    policies = [Candidate(policy_id=s, strategy=s, epsilon=None) for s in ("daily", "weekly")]
    for epsilon in sorted(set(grid.tolist()) | {0.01}):
        policies.append(Candidate(policy_id=f"adaptive_{epsilon}", strategy="adaptive", epsilon=epsilon))
    return policies


def apply_transaction_cost_rate(
    result: HedgingResult, kappa: float, reference_kappa: float,
) -> HedgingResult:
    """Exact fee sensitivity for cash-independent policies on identical trades.

    Costs are linear in the proportional rate, including interest to T. The
    validated engine supplies the trades and fee-free shadow; this transformation
    introduces no alternative trading/cash logic. Only compact results are
    accepted to avoid returning an inconsistent historical cash ledger.
    """
    if (not np.all(np.isfinite([kappa, reference_kappa]))
            or kappa < 0 or reference_kappa <= 0):
        raise ValueError("Need kappa >= 0 and reference_kappa > 0, both finite.")
    if result["history"] is not None:
        raise ValueError("Use return_history=False before rescaling costs.")
    scaled = result.copy()
    factor = kappa/reference_kappa
    scaled["total_transaction_cost"] = factor*result["total_transaction_cost"]
    scaled["transaction_cost_fv"] = factor*result["transaction_cost_fv"]
    scaled["pnl_net"] = result["pnl_gross"]-scaled["transaction_cost_fv"]
    scaled["terminal_portfolio_net"] = result["terminal_portfolio_gross"]-scaled["transaction_cost_fv"]
    return scaled


def evaluate_cost_grid(
    paths: np.ndarray, candidates: list[Candidate], cost_grid: ArrayLike, *,
    strike: float = 100.0, maturity: float = 1.0, r: float = 0.0,
    sigma_model: float = 0.20, weekly_interval: int = 5,
    reference_kappa: float = 0.002, confidence: float = 0.95,
    n_bootstrap: int = 500, bootstrap_seed: int = 731,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], BootstrapSamples]:
    """Evaluate all candidates on the SAME supplied paths and bootstrap jointly.

    Return rows ordered by fee rate then candidate, compact path matrices with
    one column per row, and paired bootstrap estimates in that same order.
    Costs/fees are included once in net P&L; losses are exactly -pnl_net.
    No calibration/evaluation selection happens inside this descriptive scorer.
    """
    rates = np.asarray(cost_grid, dtype=float)
    if (rates.ndim != 1 or rates.size == 0 or not np.all(np.isfinite(rates))
            or np.any(rates < 0) or not candidates):
        raise ValueError("Need candidates and a nonempty finite nonnegative cost grid.")
    if len({c["policy_id"] for c in candidates}) != len(candidates):
        raise ValueError("Candidate IDs must be unique.")
    base = {}
    for candidate in candidates:
        result = hedge_paths(paths, candidate["strategy"], strike=strike,
                             maturity=maturity, r=r, sigma_model=sigma_model,
                             kappa=reference_kappa, epsilon=candidate["epsilon"] or 0.0,
                             weekly_interval=weekly_interval, return_history=False)
        if result["stock_paths"] is not paths:
            raise ValueError("Supply a float64 ndarray so every policy shares the same input object.")
        np.testing.assert_allclose(result["pnl_gross"]-result["pnl_net"],
                                   result["transaction_cost_fv"], rtol=0, atol=1e-10)
        base[candidate["policy_id"]] = result
    field_map = {
        "pnl_net": "pnl_net", "pnl_gross": "pnl_gross",
        "cost_fv": "transaction_cost_fv", "cost_raw": "total_transaction_cost",
        "traded_notional": "total_traded_notional", "n_initial": "n_initial",
        "n_rebalances": "n_rebalances", "n_liquidation": "n_liquidation", "n_trades": "n_trades",
    }
    buffers = {name: [] for name in field_map}
    rows = []
    for kappa in sorted(set(rates.tolist())):
        for candidate in candidates:
            result = apply_transaction_cost_rate(base[candidate["policy_id"]], kappa, reference_kappa)
            rows.append({"column_id": len(rows), "policy_id": candidate["policy_id"],
                         "epsilon": candidate["epsilon"], "kappa": kappa,
                         "confidence": confidence, **summarize_hedging(result),
                         "mean_cost": float(result["transaction_cost_fv"].mean()),
                         "var": value_at_risk(-result["pnl_net"], confidence),
                         "es": expected_shortfall(-result["pnl_net"], confidence)})
            for key, source in field_map.items():
                buffers[key].append(result[source])
    arrays = {key: np.column_stack(values) for key, values in buffers.items()}
    bootstrap = paired_bootstrap(-arrays["pnl_net"], arrays["cost_fv"], confidence=confidence,
                                 n_bootstrap=n_bootstrap, seed=bootstrap_seed)
    for i, row in enumerate(rows):
        for metric, samples in (("es", bootstrap["es"]), ("mean_cost", bootstrap["mean_cost"])):
            interval = bootstrap_interval(samples[:, i])
            row[f"{metric}_ci_low"] = interval["ci_low"]
            row[f"{metric}_ci_high"] = interval["ci_high"]
            row[f"{metric}_bootstrap_se"] = interval["standard_error"]
    table = pd.DataFrame(rows)
    table["pareto_efficient"] = False
    for _, group in table.groupby("kappa", sort=False):
        table.loc[group.index, "pareto_efficient"] = pareto_efficient(
            group["mean_cost"].to_numpy(), group["es"].to_numpy())
    return table, arrays, bootstrap


def pareto_efficient(costs: ArrayLike, risks: ArrayLike) -> np.ndarray:
    """True unless another candidate is <= in both and strictly < in either.

    Identical cost/risk points are both retained. This is an empirical frontier
    of point estimates; it is not a statistically certified population frontier.
    """
    c, r = np.asarray(costs, float), np.asarray(risks, float)
    if c.ndim != 1 or c.shape != r.shape or not np.all(np.isfinite(c)) or not np.all(np.isfinite(r)):
        raise ValueError("costs and risks must be matching finite vectors.")
    dominates = ((c[:, None] <= c[None, :]) & (r[:, None] <= r[None, :])
                 & ((c[:, None] < c[None, :]) | (r[:, None] < r[None, :])))
    return ~dominates.any(axis=0)


def select_policy(table: pd.DataFrame, risk_budget: float, *,
                  feasibility_rule: str = "point_estimate") -> dict:
    """Select minimum terminal-valued mean cost among calibration-feasible rows.

    Ties: lower ES, then lexical policy_id. At zero cost every candidate has
    the same objective; this tie-break does not imply unique cost optimality.
    Default eligibility uses point ES <= budget. Optional es_upper_bound uses
    the pointwise percentile interval's upper end, not a simultaneous guarantee.
    """
    if not np.isfinite(risk_budget) or risk_budget < 0:
        raise ValueError("risk_budget must be finite and nonnegative.")
    if feasibility_rule not in ("point_estimate", "es_upper_bound"):
        raise ValueError("Unknown feasibility rule.")
    if table.empty or table["policy_id"].duplicated().any():
        raise ValueError("Supply one fee scenario with unique candidate IDs.")
    fields = ["mean_cost", "es", "es_ci_high"]
    if not np.all(np.isfinite(table[fields].to_numpy(dtype=float))):
        raise ValueError("Selection metrics must be finite.")
    risk_field = "es" if feasibility_rule == "point_estimate" else "es_ci_high"
    feasible = table.loc[table[risk_field] <= risk_budget]
    result = {"risk_budget": float(risk_budget), "feasibility_rule": feasibility_rule,
              "feasible_policies": feasible["policy_id"].tolist(), "n_feasible": len(feasible),
              "selected_policy": None, "status": "no_feasible_strategy"}
    if feasible.empty:
        return result
    selected = feasible.sort_values(["mean_cost", "es", "policy_id"]).iloc[0]
    result.update(selected_policy=str(selected["policy_id"]), status="selected",
                  calibration_mean_cost=float(selected["mean_cost"]),
                  calibration_es=float(selected["es"]),
                  calibration_es_ci_low=float(selected["es_ci_low"]),
                  calibration_es_ci_high=float(selected["es_ci_high"]),
                  calibration_headroom=float(risk_budget-selected["es"]),
                  calibration_upper_bound_headroom=float(risk_budget-selected["es_ci_high"]))
    return result


def paired_comparisons(table: pd.DataFrame, arrays: dict[str, np.ndarray],
                       bootstrap: BootstrapSamples) -> pd.DataFrame:
    """All policy pairs within each fee case, preserving path/bootstrap pairing."""
    from itertools import combinations
    rows = []
    for kappa, group in table.groupby("kappa", sort=False):
        for (_, first), (_, second) in combinations(group.iterrows(), 2):
            i, j = int(first["column_id"]), int(second["column_id"])
            for metric, difference, samples in (
                ("es", float(first["es"]-second["es"]), bootstrap["es"][:, i]-bootstrap["es"][:, j]),
                ("mean_cost", float(first["mean_cost"]-second["mean_cost"]),
                 bootstrap["mean_cost"][:, i]-bootstrap["mean_cost"][:, j]),
            ):
                rows.append({"kappa": kappa, "first": first["policy_id"], "second": second["policy_id"],
                             "metric_first_minus_second": metric, "difference": difference,
                             **bootstrap_interval(samples)})
            interval = mean_confidence_interval(arrays["pnl_net"][:, i]-arrays["pnl_net"][:, j])
            rows.append({"kappa": kappa, "first": first["policy_id"], "second": second["policy_id"],
                         "metric_first_minus_second": "mean_net_pnl", "difference": interval["mean"],
                         **{k: interval[k] for k in ("ci_low", "ci_high", "standard_error")}})
    return pd.DataFrame(rows)
