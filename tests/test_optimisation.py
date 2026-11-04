import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from experiments._risk_common import path_sample, load_config, evaluate_frozen_decisions
from src.hedging import hedge_paths
from src.optimisation import (candidate_policies, pareto_efficient, select_policy,
                              apply_transaction_cost_rate, evaluate_cost_grid)

ROOT = Path(__file__).resolve().parents[1]


def test_candidate_grid_preserves_benchmarks_and_deduplicates():
    grid = candidate_policies()
    assert len(grid) == 7
    assert [c["strategy"] for c in grid[:2]] == ["daily", "weekly"]
    assert [c["epsilon"] for c in grid[2:]] == [0.005, 0.01, 0.02, 0.03, 0.05]
    extended = candidate_policies([0.1, 0.02, 0.02])
    assert [c["epsilon"] for c in extended[2:]] == [0.01, 0.02, 0.1]
    with pytest.raises(ValueError):
        candidate_policies([-1])


def test_pareto_dominance_ties_and_single_strict_improvement():
    assert_array_equal(pareto_efficient([1, 1, 2, 0.5, 1], [2, 3, 1, 4, 2]),
                        [True, False, True, True, True])
    assert_array_equal(pareto_efficient([1, 1], [1, 2]), [True, False])
    assert_array_equal(pareto_efficient([1, 2], [1, 1]), [True, False])


def selection_fixture():
    return pd.DataFrame({"policy_id": ["cheapest", "admissible", "safest"],
                         "mean_cost": [0.2, 0.6, 1.0], "es": [4.0, 2.0, 1.0],
                         "es_ci_low": [3.8, 1.8, 0.9], "es_ci_high": [4.2, 2.2, 1.1]})


def test_minimum_cost_among_feasible_not_minimum_es():
    decision = select_policy(selection_fixture(), 2)
    assert decision["selected_policy"] == "admissible"
    assert decision["feasible_policies"] == ["admissible", "safest"]
    assert decision["calibration_headroom"] == 0
    assert select_policy(selection_fixture(), 0.5)["selected_policy"] is None
    assert select_policy(selection_fixture(), 0.5)["status"] == "no_feasible_strategy"
    assert select_policy(selection_fixture(), 2, feasibility_rule="es_upper_bound")["selected_policy"] == "safest"


def test_zero_cost_tie_break_is_predefined():
    table = selection_fixture()
    table["mean_cost"] = 0
    assert select_policy(table, 5)["selected_policy"] == "safest"


@pytest.mark.parametrize("r", [0, 0.07, -0.03])
@pytest.mark.parametrize("kappa", [0, 0.0001, 0.005])
def test_exact_cost_scaling_matches_full_ledger(r, kappa):
    paths = np.array([[100, 105, 98, 112], [100, 90, 95, 92]], dtype=float)
    base = hedge_paths(paths, "adaptive", r=r, kappa=0.002, return_history=False)
    scaled = apply_transaction_cost_rate(base, kappa, 0.002)
    direct = hedge_paths(paths, "adaptive", r=r, kappa=kappa, return_history=False)
    for key in ("pnl_net", "pnl_gross", "transaction_cost_fv", "total_transaction_cost",
                "terminal_portfolio_net", "total_traded_notional", "n_trades"):
        assert_allclose(scaled[key], direct[key], atol=1e-11)
    assert_array_equal(scaled["n_trades"], base["n_trades"])


def test_sample_independence_repeatability_and_paired_volatility_shocks():
    config = load_config(ROOT/"configs/study.json", {"n_calibration": 20, "n_evaluation": 20, "n_steps": 5})
    cal, evaluation = path_sample(config, "calibration"), path_sample(config, "evaluation")
    assert not np.array_equal(cal[:, 1:], evaluation[:, 1:])
    assert_array_equal(evaluation, path_sample(config, "evaluation"))
    stress = path_sample(config, "evaluation", sigma_true=0.25)
    dt = config["maturity"]/config["n_steps"]
    def recover_shocks(paths, vol):
        return (np.diff(np.log(paths), axis=1)-(config["mu"]-0.5*vol**2)*dt)/(vol*np.sqrt(dt))
    assert_allclose(recover_shocks(evaluation, 0.2), recover_shocks(stress, 0.25), atol=1e-13)
    with pytest.raises(ValueError):
        path_sample(config | {"evaluation_seed": config["calibration_seed"]}, "evaluation")


def test_grid_common_paths_volatility_inputs_and_gross_net(monkeypatch):
    import src.optimisation as module
    original = module.hedge_paths
    paths = np.array([[100, 107, 115], [100, 92, 97], [100, 101, 103]], dtype=float)
    calls = []
    def checked_paths(received, strategy, **kwargs):
        assert received is paths
        assert kwargs["sigma_model"] == 0.17
        calls.append(strategy)
        return original(received, strategy, **kwargs)
    monkeypatch.setattr(module, "hedge_paths", checked_paths)
    table, arrays, _ = evaluate_cost_grid(paths, candidate_policies([0.01]), [0, 0.002],
                                         sigma_model=0.17, n_bootstrap=10)
    assert calls == ["daily", "weekly", "adaptive"]
    assert len(table) == 6
    assert_allclose(arrays["pnl_gross"]-arrays["pnl_net"], arrays["cost_fv"], atol=1e-12)
    assert_array_equal(arrays["n_trades"][:, :3], arrays["n_trades"][:, 3:])
    assert_allclose(arrays["cost_raw"][:, 3:], 0.002*arrays["traded_notional"][:, 3:])


def test_evaluation_cannot_reselect_cheaper_test_winner():
    table = selection_fixture()
    table["kappa"] = 0.002
    table["mean_cost_ci_low"] = table["mean_cost"]-0.01
    table["mean_cost_ci_high"] = table["mean_cost"]+0.01
    frozen = {"decisions": [{"kappa": 0.002, "risk_budget": 2,
                              "selected_policy": "admissible", "status": "selected"}]}
    table.loc[table["policy_id"] == "cheapest", "es"] = 0.1
    table.loc[table["policy_id"] == "admissible", "es"] = 3
    report = evaluate_frozen_decisions(table, frozen)
    assert report.iloc[0]["selected_policy"] == "admissible"
    assert not report.iloc[0]["passes_point_constraint"]
    assert report.iloc[0]["headroom"] == -1
