"""independent calibration/evaluation and finite-policy cost selection."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from experiments._common import ROOT
from experiments._risk_common import (load_config, path_sample, config_hash, path_hash,
                                      engine_parameters, evaluate_frozen_decisions, save_metadata)
from src.optimisation import (candidate_policies, evaluate_cost_grid, select_policy,
                              paired_comparisons)
from src.risk_plotting import plot_frontier, plot_cost_sensitivity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/study.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results/03_cost_risk_optimisation")
    parser.add_argument("--figures-dir", type=Path, default=ROOT/"results/figures")
    for key in ("n-calibration", "n-evaluation", "n-bootstrap"):
        parser.add_argument(f"--{key}", type=int)
    args = parser.parse_args()
    config = load_config(args.config, {name: getattr(args, name) for name in
                                       ("n_calibration", "n_evaluation", "n_bootstrap")})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"predeclared_configuration.json").write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")
    candidates = candidate_policies(config["epsilon_grid"])

    print(f"Calibration: {config['n_calibration']:,} common physical paths; {len(candidates)} policies.", flush=True)
    paths = path_sample(config, "calibration")
    calibration_hash = path_hash(paths)
    calibration, calibration_arrays, calibration_bootstrap = evaluate_cost_grid(
        paths, candidates, config["cost_grid"], **engine_parameters(config),
        reference_kappa=config["baseline_kappa"], confidence=config["confidence"],
        n_bootstrap=config["n_bootstrap"], bootstrap_seed=config["calibration_bootstrap_seed"])
    calibration["sample"] = "calibration"
    calibration.to_csv(args.output_dir/"calibration_summary.csv", index=False)
    np.savez_compressed(args.output_dir/"calibration_path_results.npz", **calibration_arrays)
    np.savez_compressed(args.output_dir/"calibration_bootstrap.npz", **calibration_bootstrap)
    paired_comparisons(calibration, calibration_arrays, calibration_bootstrap).to_csv(
        args.output_dir/"calibration_paired_differences.csv", index=False)
    decisions = []
    for kappa, table in calibration.groupby("kappa", sort=False):
        for budget in config["risk_budgets"]:
            decisions.append({"kappa": float(kappa), **select_policy(
                table, budget, feasibility_rule=config["feasibility_rule"])})
    frozen = {"study_configuration": config, "configuration_sha256": config_hash(config),
              "candidates": candidates, "calibration_paths_sha256": calibration_hash,
              "selection_data": "calibration_only", "decisions": decisions,
              "tie_break": "mean_cost, then ES, then lexical policy_id",
              "evaluation_policy": "evaluate frozen choices; never reselect on evaluation or stress"}
    selection_file = args.output_dir/"frozen_selection.json"
    # This file is durably written BEFORE evaluation shocks are generated.
    selection_file.write_text(json.dumps(frozen, indent=2)+"\n", encoding="utf-8")
    print("Selection frozen before evaluation generation.", flush=True)
    main_decision = next(d for d in decisions if d["kappa"] == config["baseline_kappa"]
                         and d["risk_budget"] == config["primary_risk_budget"])
    baseline = calibration[calibration["kappa"] == config["baseline_kappa"]]
    plot_frontier(baseline, config["primary_risk_budget"], main_decision["selected_policy"],
                   args.figures_dir/"cost_risk_frontier.png")
    plot_cost_sensitivity(calibration, config["baseline_kappa"],
                           args.figures_dir/"transaction_cost_comparison.png")
    del paths

    print(f"Evaluation: {config['n_evaluation']:,} independent common physical paths.", flush=True)
    evaluation_paths = path_sample(config, "evaluation")
    evaluation_hash = path_hash(evaluation_paths)
    if evaluation_hash == calibration_hash:
        raise RuntimeError("Calibration and evaluation path samples are identical.")
    evaluation, evaluation_arrays, evaluation_bootstrap = evaluate_cost_grid(
        evaluation_paths, candidates, config["cost_grid"], **engine_parameters(config),
        reference_kappa=config["baseline_kappa"], confidence=config["confidence"],
        n_bootstrap=config["n_bootstrap"], bootstrap_seed=config["evaluation_bootstrap_seed"])
    evaluation["sample"] = "evaluation"
    evaluation.to_csv(args.output_dir/"evaluation_summary.csv", index=False)
    np.savez_compressed(args.output_dir/"evaluation_path_results.npz", **evaluation_arrays)
    np.savez_compressed(args.output_dir/"evaluation_bootstrap.npz", **evaluation_bootstrap)
    paired_comparisons(evaluation, evaluation_arrays, evaluation_bootstrap).to_csv(
        args.output_dir/"evaluation_paired_differences.csv", index=False)
    report = evaluate_frozen_decisions(evaluation, frozen)
    report.to_csv(args.output_dir/"risk_budget_decisions.csv", index=False)
    pd.DataFrame(decisions).to_csv(args.output_dir/"calibration_decisions.csv", index=False)
    errors = {
        sample: float(np.max(np.abs(arrays["pnl_gross"]-arrays["pnl_net"]-arrays["cost_fv"])))
        for sample, arrays in (("calibration", calibration_arrays), ("evaluation", evaluation_arrays))}
    save_metadata(args, config, calibration_paths_sha256=calibration_hash,
                   evaluation_paths_sha256=evaluation_hash, calibration_evaluation_independent=True,
                   selection_frozen_before_evaluation=True, candidates=candidates,
                   frozen_selection_sha256=hashlib.sha256(selection_file.read_bytes()).hexdigest(),
                   max_gross_net_reconciliation_error=errors,
                   per_path_columns="NPZ column i corresponds to summary column_id i",
                   shared_paths_within_each_sample=True,
                   fee_sensitivity="exact linear cost scaling verified against full ledgers by tests")
    print("\nFrozen decisions at the historical 0.2% cost rate:", flush=True)
    print(report[report["kappa"] == config["baseline_kappa"]].to_string(index=False), flush=True)
    print("\nCalibration frontier:\n", baseline[["policy_id", "mean_cost", "es", "es_ci_low", "es_ci_high", "pareto_efficient"]].to_string(index=False), flush=True)
    print("Saved", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
