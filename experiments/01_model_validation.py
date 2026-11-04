"""Experiment 1: Q-pricing validation and frictionless frequency convergence.

Run from the project root: python experiments/01_model_validation.py
"""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from experiments._common import parameter_parser, write_metadata
from src.black_scholes import bs_call_price
from src.hedging import hedge_paths
from src.metrics import pnl_statistics, mean_confidence_interval
from src.monte_carlo import risk_neutral_call_price_mc, simulate_gbm_paths
from src.plotting import plot_pricing, plot_convergence


def main() -> None:
    parser = parameter_parser(__doc__, "01_model_validation")
    parser.add_argument("--pricing-seed", type=int, default=314159)
    parser.add_argument("--convergence-seed", type=int, default=271828)
    parser.add_argument("--convergence-paths", type=int, default=None,
                        help="Defaults to --n-paths.")
    parser.add_argument("--convergence-steps", type=int, nargs="+", default=[12, 48, 252, 1008],
                        help="At least two counts, each dividing the finest count.")
    args = parser.parse_args()
    steps = sorted(set(args.convergence_steps))
    if len(steps) < 2 or steps[0] <= 0 or any(steps[-1] % n for n in steps):
        parser.error("Need at least two positive nested step counts dividing the finest count.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    analytical = bs_call_price(args.s0, args.strike, args.r, args.sigma_model, args.maturity)
    price = risk_neutral_call_price_mc(args.s0, args.strike, args.r, args.sigma_model,
                                       args.maturity, args.n_paths, args.pricing_seed)
    error = abs(price["estimate"]-analytical)
    pricing = pd.DataFrame([{
        "analytical_price": analytical, "mc_estimate": price["estimate"],
        "standard_error": price["standard_error"], "ci_low": price["ci_low"],
        "ci_high": price["ci_high"], "absolute_error": error, "n_paths": args.n_paths,
        "price_within_95pct_ci": price["ci_low"] <= analytical <= price["ci_high"],
    }])

    # This is a model diagnostic: use Q paths, true=model volatility and zero
    # fees. It is separate from the baseline P-measure trading comparison.
    convergence_paths = args.n_paths if args.convergence_paths is None else args.convergence_paths
    paths = simulate_gbm_paths(args.s0, args.r, args.sigma_model, args.maturity,
                               steps[-1], convergence_paths, args.convergence_seed)
    rows, squared_errors = [], {}
    for count in steps:
        sample = paths[:, ::steps[-1]//count]
        result = hedge_paths(sample, "daily", strike=args.strike, maturity=args.maturity,
                              r=args.r, sigma_model=args.sigma_model, kappa=0,
                              return_history=False)
        stats = pnl_statistics(result["pnl_gross"])
        rows.append({"n_steps": count, "n_paths": convergence_paths, **stats})
        squared_errors[count] = result["pnl_gross"]**2
    convergence = pd.DataFrame(rows)
    contrasts = pd.DataFrame([
        {"coarse_steps": coarse, "fine_steps": fine,
         **mean_confidence_interval(squared_errors[coarse]-squared_errors[fine])}
        for coarse, fine in zip(steps[:-1], steps[1:])
    ])
    overall = mean_confidence_interval(squared_errors[steps[0]]-squared_errors[steps[-1]])
    deterministic = args.sigma_model == 0
    convergence_passed = (bool(np.max(convergence["rmse"]) < 1e-10) if deterministic
                          else overall["ci_low"] > 0)
    checks = {
        "pricing_error_below_five_standard_errors": error <= max(5*price["standard_error"], 1e-10),
        "coarse_to_fine_mse_improvement_positive_at_95pct": overall["ci_low"] > 0,
        "convergence_diagnostic_passed": convergence_passed,
        "convergence_criterion": "zero error at every mesh" if deterministic else "positive paired coarse-to-fine MSE improvement",
        "overall_paired_mse_improvement": overall,
        "note": "A 95% price CI may miss in a valid sample; the diagnostic price gate uses five SE.",
    }
    pricing.to_csv(args.output_dir / "pricing_validation.csv", index=False)
    convergence.to_csv(args.output_dir / "frictionless_convergence.csv", index=False)
    contrasts.to_csv(args.output_dir / "convergence_paired_mse_improvements.csv", index=False)
    (args.output_dir / "checks.json").write_text(json.dumps(checks, indent=2)+"\n", encoding="utf-8")
    write_metadata(args, {"pricing_measure": "Q", "pricing_drift": args.r,
                         "pricing_volatility": args.sigma_model,
                         "convergence_measure": "Q", "convergence_drift": args.r,
                         "convergence_volatility_true_and_model": args.sigma_model,
                         "convergence_transaction_cost_rate": 0,
                         "convergence_paths_actual": convergence_paths,
                         "unused_common_cli_parameters": ["mu", "sigma_true", "kappa", "epsilon",
                                                           "weekly_interval", "seed", "n_steps"]})
    plot_pricing(pricing, args.output_dir / "pricing_validation.png")
    plot_convergence(convergence, args.output_dir / "frictionless_convergence.png")
    print("Risk-neutral call pricing (currency per option):")
    print(pricing.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nFrictionless coupled-grid validation (correct volatility):")
    print(convergence.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nOverall paired coarse-minus-fine MSE improvement:", overall)
    print("Saved to", args.output_dir)
    if not (checks["pricing_error_below_five_standard_errors"]
            and checks["convergence_diagnostic_passed"]):
        raise RuntimeError("Statistical diagnostic failed. Investigate; do not mark this run validated.")


if __name__ == "__main__":
    main()
