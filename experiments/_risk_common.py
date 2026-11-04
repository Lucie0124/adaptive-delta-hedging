"""Configuration, independent samples and saved-decision reporting for C/D."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._common import ROOT, write_metadata
from src.monte_carlo import simulate_gbm_paths


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def path_hash(paths: np.ndarray) -> str:
    return hashlib.sha256(paths.tobytes()).hexdigest()


def validate_config(config: dict) -> None:
    """Reject ambiguous sampling, missing reference scenarios and invalid limits."""
    for key in ("n_steps", "n_calibration", "n_evaluation", "n_bootstrap", "weekly_interval"):
        value = config[key]
        minimum = 1 if key in ("n_steps", "weekly_interval") else 2
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}.")
    if config["calibration_seed"] == config["evaluation_seed"]:
        raise ValueError("Calibration and evaluation seeds must be independent/distinct.")
    if config["baseline_kappa"] <= 0 or config["baseline_kappa"] not in config["cost_grid"]:
        raise ValueError("Positive baseline_kappa must be in cost_grid.")
    if config["primary_risk_budget"] not in config["risk_budgets"]:
        raise ValueError("primary_risk_budget must be declared in risk_budgets.")
    if not config["risk_budgets"] or any(not np.isfinite(b) or b < 0 for b in config["risk_budgets"]):
        raise ValueError("Risk budgets must be finite, nonnegative and predefined.")
    if config["sigma_true"] not in config["stress_volatilities"] or 0.25 not in config["stress_volatilities"]:
        raise ValueError("Stress grid must include the baseline and original 25% scenario.")
    if config["premium_convention"] != "fixed_model_volatility":
        raise ValueError("Principal results must retain the fixed model premium.")
    if config["feasibility_rule"] not in ("point_estimate", "es_upper_bound"):
        raise ValueError("Unknown feasibility rule.")
    if not 0 < config["confidence"] < 1:
        raise ValueError("confidence must be in (0,1).")
    for key in ("cost_grid", "epsilon_grid", "stress_volatilities"):
        values = np.asarray(config[key], float)
        if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"Invalid {key}.")


def load_config(path: Path, overrides: dict | None = None) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    config.update({k: v for k, v in (overrides or {}).items() if v is not None})
    validate_config(config)
    return config


def path_sample(config: dict, sample: str, sigma_true: float | None = None) -> np.ndarray:
    """Independent across sample roles; paired standard shocks across volatilities.

    The same seed/dimensions regenerate exactly the same standard-normal shocks
    for different volatility scenarios, while calibration/evaluation use distinct
    seeds. The evaluation helper is only called after selection is frozen.
    """
    validate_config(config)
    if sample not in ("calibration", "evaluation"):
        raise ValueError("sample must be calibration or evaluation.")
    paths = simulate_gbm_paths(
        s0=config["s0"], mu=config["mu"],
        sigma_true=config["sigma_true"] if sigma_true is None else sigma_true,
        maturity=config["maturity"], n_steps=config["n_steps"],
        n_paths=config[f"n_{sample}"], seed=config[f"{sample}_seed"],
    )
    paths.setflags(write=False)
    return paths


def engine_parameters(config: dict) -> dict:
    return {key: config[key] for key in ("strike", "maturity", "r", "sigma_model", "weekly_interval")}


def evaluate_frozen_decisions(table: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    """Evaluate saved choices only; never select the best evaluation/stress row."""
    rows = []
    for decision in frozen["decisions"]:
        row = {"kappa": decision["kappa"], "risk_budget": decision["risk_budget"],
               "selected_policy": decision["selected_policy"],
               "calibration_status": decision["status"]}
        if decision["selected_policy"] is None:
            row.update(status="no_calibration_feasible_strategy", passes_point_constraint=False,
                       passes_upper_bound_constraint=False)
        else:
            match = table[(table["kappa"] == decision["kappa"])
                          & (table["policy_id"] == decision["selected_policy"])]
            if len(match) != 1:
                raise ValueError("Frozen policy missing or ambiguous in evaluation table.")
            stats = match.iloc[0]
            row.update(status="fixed_policy_evaluation", es=float(stats["es"]),
                       es_ci_low=float(stats["es_ci_low"]), es_ci_high=float(stats["es_ci_high"]),
                       mean_cost=float(stats["mean_cost"]),
                       mean_cost_ci_low=float(stats["mean_cost_ci_low"]),
                       mean_cost_ci_high=float(stats["mean_cost_ci_high"]),
                       headroom=float(decision["risk_budget"]-stats["es"]),
                       upper_bound_headroom=float(decision["risk_budget"]-stats["es_ci_high"]),
                       passes_point_constraint=bool(stats["es"] <= decision["risk_budget"]),
                       passes_upper_bound_constraint=bool(stats["es_ci_high"] <= decision["risk_budget"]))
        rows.append(row)
    return pd.DataFrame(rows)


def save_metadata(args: argparse.Namespace, config: dict, **extras) -> None:
    write_metadata(args, {"scope": "Finite-policy ES-constrained simulation study",
                          "study_configuration": config, "configuration_sha256": config_hash(config),
                          "risk_measure": "empirical upper-tail quantile-integral ES of losses=-pnl_net",
                          "uncertainty": "paired iid path percentile bootstrap; pointwise 95% intervals",
                          **extras})
