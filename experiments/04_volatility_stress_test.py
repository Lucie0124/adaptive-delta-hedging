"""frozen baseline choices under paired true-volatility scenarios."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from experiments._common import ROOT
from experiments._risk_common import (validate_config, config_hash, path_hash, path_sample,
                                      engine_parameters, evaluate_frozen_decisions, save_metadata)
from src.black_scholes import bs_call_price
from src.metrics import expected_shortfall, mean_confidence_interval, pnl_statistics
from src.optimisation import evaluate_cost_grid
from src.risk_metrics import paired_bootstrap, bootstrap_interval
from src.risk_plotting import plot_stress_distributions, plot_stress_es, plot_stress_heatmap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-file", type=Path, default=ROOT/"results/03_cost_risk_optimisation/frozen_selection.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results/04_volatility_stress_test")
    parser.add_argument("--figures-dir", type=Path, default=ROOT/"results/figures")
    args = parser.parse_args()
    frozen = json.loads(args.selection_file.read_text(encoding="utf-8"))
    config = frozen["study_configuration"]
    validate_config(config)
    if frozen["configuration_sha256"] != config_hash(config) or frozen["selection_data"] != "calibration_only":
        raise ValueError("Invalid frozen calibration-selection artifact.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    candidates = frozen["candidates"]
    arrays_by_vol, summaries, path_hashes = {}, [], {}
    max_errors = {}
    for vol in sorted(set(config["stress_volatilities"])):
        print(f"True volatility {vol:.0%}, model {config['sigma_model']:.0%}: fixed calibration policies; {config['n_evaluation']:,} paths.", flush=True)
        paths = path_sample(config, "evaluation", sigma_true=vol)
        path_hashes[str(vol)] = path_hash(paths)
        # Bootstrap once across all volatility columns below to retain stress pairing.
        table, arrays, _ = evaluate_cost_grid(
            paths, candidates, [config["baseline_kappa"]], **engine_parameters(config),
            reference_kappa=config["baseline_kappa"], confidence=config["confidence"],
            n_bootstrap=2, bootstrap_seed=config["stress_bootstrap_seed"])
        table["sigma_true"], table["sigma_model"] = vol, config["sigma_model"]
        table["sample"] = "evaluation_stress_fixed_policy"
        summaries.append(table)
        arrays_by_vol[vol] = arrays
        max_errors[str(vol)] = float(np.max(np.abs(arrays["pnl_gross"]-arrays["pnl_net"]-arrays["cost_fv"])))
        np.savez_compressed(args.output_dir/f"path_results_vol_{vol:g}.npz", **arrays)
        del paths
    table = pd.concat(summaries, ignore_index=True)
    vols = sorted(arrays_by_vol)
    count = len(candidates)
    joint_losses = np.column_stack([-arrays_by_vol[v]["pnl_net"] for v in vols])
    joint_costs = np.column_stack([arrays_by_vol[v]["cost_fv"] for v in vols])
    boot = paired_bootstrap(joint_losses, joint_costs, confidence=config["confidence"],
                             n_bootstrap=config["n_bootstrap"], seed=config["stress_bootstrap_seed"])
    np.savez_compressed(args.output_dir/"paired_volatility_bootstrap.npz", **boot)
    for i in range(len(table)):
        for name, field in (("es", "es"), ("mean_cost", "mean_cost")):
            interval = bootstrap_interval(boot[field][:, i])
            for key in ("ci_low", "ci_high", "standard_error"):
                table.loc[i, f"{name}_{'bootstrap_se' if key == 'standard_error' else key}"] = interval[key]
    table.to_csv(args.output_dir/"stress_summary.csv", index=False)
    baseline_vol = config["sigma_true"]
    original_evaluation = pd.read_csv(args.selection_file.parent/"evaluation_summary.csv")
    saved_metadata = json.loads((args.selection_file.parent/"metadata.json").read_text())
    if hashlib.sha256(args.selection_file.read_bytes()).hexdigest() != saved_metadata["frozen_selection_sha256"]:
        raise RuntimeError("Frozen selection file has changed since the calibration/evaluation run.")
    if path_hashes[str(baseline_vol)] != saved_metadata["evaluation_paths_sha256"]:
        raise RuntimeError("Stress baseline does not reproduce the independent evaluation paths.")
    before = original_evaluation[original_evaluation["kappa"] == config["baseline_kappa"]].set_index("policy_id")
    after = table[table["sigma_true"] == baseline_vol].set_index("policy_id")
    np.testing.assert_allclose(after.loc[before.index, ["mean_pnl_net", "mean_cost", "es"]],
                               before[["mean_pnl_net", "mean_cost", "es"]], rtol=0, atol=1e-10)

    # Only historical-fee decisions apply to this stress, without reoptimisation.
    stress_frozen = dict(frozen, decisions=[d for d in frozen["decisions"] if d["kappa"] == config["baseline_kappa"]])
    decisions = []
    differences = []
    diagnostics = []
    base_offset = vols.index(baseline_vol)*count
    base_premium = bs_call_price(config["s0"], config["strike"], config["r"],
                                  config["sigma_model"], config["maturity"])
    for vol in vols:
        group = table[table["sigma_true"] == vol]
        report = evaluate_frozen_decisions(group, stress_frozen)
        report["sigma_true"] = vol
        decisions.append(report)
        fair_premium = bs_call_price(config["s0"], config["strike"], config["r"], vol, config["maturity"])
        premium_shift = (fair_premium-base_premium)*np.exp(config["r"]*config["maturity"])
        for j, candidate in enumerate(candidates):
            column = vols.index(vol)*count+j
            row = group.iloc[j]
            # Pure premium-funding diagnostic: same trades/fees, deterministic
            # additional initial premium grown at r. It is not a new hedge model.
            shifted = arrays_by_vol[vol]["pnl_net"][:, j]+premium_shift
            diagnostics.append({"sigma_true": vol, "policy_id": candidate["policy_id"],
                                "fixed_model_premium": base_premium, "fair_true_vol_premium": fair_premium,
                                "terminal_premium_shift": premium_shift,
                                "fixed_premium_es": row["es"],
                                "fair_premium_es": expected_shortfall(-shifted, config["confidence"]),
                                "fair_premium_es_ci_low": row["es_ci_low"]-premium_shift,
                                "fair_premium_es_ci_high": row["es_ci_high"]-premium_shift,
                                **pnl_statistics(shifted)})
            if vol != baseline_vol:
                base_row = table.iloc[base_offset+j]
                for metric, point, samples in (
                    ("es", float(row["es"]-base_row["es"]), boot["es"][:, column]-boot["es"][:, base_offset+j]),
                    ("mean_cost", float(row["mean_cost"]-base_row["mean_cost"]),
                     boot["mean_cost"][:, column]-boot["mean_cost"][:, base_offset+j]),
                ):
                    differences.append({"sigma_true": vol, "baseline_sigma_true": baseline_vol,
                                        "policy_id": candidate["policy_id"], "metric_stress_minus_baseline": metric,
                                        "difference": point, **bootstrap_interval(samples)})
                interval = mean_confidence_interval(arrays_by_vol[vol]["pnl_net"][:, j]
                                                     -arrays_by_vol[baseline_vol]["pnl_net"][:, j])
                differences.append({"sigma_true": vol, "baseline_sigma_true": baseline_vol,
                                    "policy_id": candidate["policy_id"], "metric_stress_minus_baseline": "mean_net_pnl",
                                    "difference": interval["mean"],
                                    **{key: interval[key] for key in ("ci_low", "ci_high", "standard_error")}})
    pd.concat(decisions, ignore_index=True).to_csv(args.output_dir/"fixed_policy_budget_stress.csv", index=False)
    pd.DataFrame(differences).to_csv(args.output_dir/"paired_stress_differences.csv", index=False)
    diagnostic = pd.DataFrame(diagnostics)
    diagnostic.to_csv(args.output_dir/"fair_premium_diagnostic.csv", index=False)
    np.testing.assert_allclose(diagnostic["fair_premium_es"],
                               diagnostic["fixed_premium_es"]-diagnostic["terminal_premium_shift"], atol=1e-10)
    main_decision = next(d for d in stress_frozen["decisions"] if d["risk_budget"] == config["primary_risk_budget"])
    displayed = ["daily", "weekly", "adaptive_0.01"]
    if main_decision["selected_policy"] is not None and main_decision["selected_policy"] not in displayed:
        displayed.append(main_decision["selected_policy"])
    plot_stress_distributions(table, arrays_by_vol, displayed, args.figures_dir/"hedging_pnl_distributions.png")
    plot_stress_es(table, config["primary_risk_budget"], args.figures_dir/"expected_shortfall_comparison.png")
    plot_stress_heatmap(table, main_decision["selected_policy"], args.figures_dir/"volatility_stress_heatmap.png")
    save_metadata(args, config, selection_file_sha256=hashlib.sha256(args.selection_file.read_bytes()).hexdigest(),
                   selected_policy_primary=main_decision["selected_policy"], paths_sha256_by_volatility=path_hashes,
                   paired_shocks_across_volatilities=True, reoptimised_under_stress=False,
                   baseline_reproduces_frozen_evaluation=True,
                   max_gross_net_reconciliation_error=max_errors,
                   principal_premium=float(base_premium), fair_premium_is_supplementary_only=True,
                   bootstrap_columns="sorted volatility scenarios, then frozen candidate order")
    print(table[["sigma_true", "policy_id", "mean_pnl_net", "std_pnl_net", "rmse_net", "mean_cost", "es", "es_ci_low", "es_ci_high"]].to_string(index=False), flush=True)
    print("\nFixed baseline-policy stress decisions:\n", pd.concat(decisions).to_string(index=False), flush=True)
    print("Saved", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
