"""Experiment 2: daily, weekly and adaptive on common physical GBM paths.

Run from the project root: python experiments/02_baseline_hedging.py
"""

from pathlib import Path
import hashlib
from itertools import combinations
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

from experiments._common import parameter_parser, hedge_parameters, write_metadata
from src.hedging import hedge_paths
from src.metrics import summarize_hedging, mean_confidence_interval
from src.monte_carlo import simulate_gbm_paths
from src.plotting import plot_pnl_distributions, plot_execution


def main() -> None:
    args = parameter_parser(__doc__, "02_baseline_hedging").parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The only stock-scenario generation in this experiment.
    paths = simulate_gbm_paths(args.s0, args.mu, args.sigma_true, args.maturity,
                               args.n_steps, args.n_paths, args.seed)
    paths.setflags(write=False)
    parameters = hedge_parameters(args)
    summaries, path_tables, contrasts, traces = [], [], [], []
    zero_fee_results = {}
    max_reconciliation_error = 0.0
    for case, kappa in (("no_costs", 0.0), ("with_costs", args.kappa)):
        results = {}
        for strategy in ("daily", "weekly", "adaptive"):
            result = hedge_paths(paths, strategy, **parameters, kappa=kappa, return_history=False)
            results[strategy] = result
            if result["stock_paths"] is not paths:
                raise RuntimeError("Strategy lost the shared underlying paths.")
            assert_allclose(result["pnl_gross"]-result["pnl_net"],
                            result["transaction_cost_fv"], rtol=0, atol=1e-10)
            max_reconciliation_error = max(max_reconciliation_error, float(np.max(np.abs(
                result["pnl_gross"]-result["pnl_net"]-result["transaction_cost_fv"]))))
            if case == "no_costs":
                zero_fee_results[strategy] = result
            else:
                zero = zero_fee_results[strategy]
                assert_allclose(result["pnl_gross"], zero["pnl_net"], rtol=0, atol=1e-10)
                assert_array_equal(result["n_trades"], zero["n_trades"])
                assert_array_equal(result["total_traded_notional"], zero["total_traded_notional"])
            summaries.append({"fee_case": case, "kappa": kappa,
                              **summarize_hedging(result),
                              "se_mean_cost": mean_confidence_interval(result["total_transaction_cost"])["standard_error"]})
            path_tables.append(pd.DataFrame({
                "scenario_id": "physical_baseline", "set_id": "exploratory",
                "path_id": np.arange(args.n_paths), "strategy": strategy,
                "epsilon": args.epsilon if strategy == "adaptive" else np.nan,
                "fee_case": case, "kappa": kappa,
                **{key: result[key] for key in (
                    "premium", "terminal_payoff", "terminal_portfolio_net", "terminal_portfolio_gross",
                    "pnl_gross", "pnl_net", "transaction_cost_fv", "total_transaction_cost",
                    "total_traded_notional", "n_initial", "n_rebalances", "n_liquidation", "n_trades")},
            }))
            # Keep a compact inspectable example ledger. Full histories for any
            # number of paths are available through hedge_paths(return_history=True).
            example = hedge_paths(paths[:1], strategy, **parameters, kappa=kappa)
            trace = pd.DataFrame({"time": example["times"], "stock_price": paths[0],
                                   **{key: value[0] for key, value in example["history"].items()}})
            trace["strategy"], trace["fee_case"], trace["path_id"] = strategy, case, 0
            trace["option_payoff"] = np.nan
            trace["post_settlement_cash"] = np.nan
            trace.loc[args.n_steps, "option_payoff"] = example["terminal_payoff"][0]
            trace.loc[args.n_steps, "post_settlement_cash"] = example["pnl_net"][0]
            traces.append(trace)
        for first, second in combinations(results, 2):
            a, b = results[first], results[second]
            for metric, difference in (
                ("net_pnl", a["pnl_net"]-b["pnl_net"]),
                ("net_absolute_error", np.abs(a["pnl_net"])-np.abs(b["pnl_net"])),
                ("gross_squared_error", a["pnl_gross"]**2-b["pnl_gross"]**2),
                ("total_transaction_cost", a["total_transaction_cost"]-b["total_transaction_cost"]),
            ):
                contrasts.append({"fee_case": case, "first": first, "second": second,
                                  "metric_first_minus_second": metric,
                                  **mean_confidence_interval(difference)})
    summary, path_table = pd.DataFrame(summaries), pd.concat(path_tables, ignore_index=True)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    path_table.to_csv(args.output_dir / "path_results.csv", index=False)
    pd.DataFrame(contrasts).to_csv(args.output_dir / "paired_differences.csv", index=False)
    pd.concat(traces, ignore_index=True).to_csv(args.output_dir / "example_path_ledger.csv", index=False)
    plot_pnl_distributions(path_table, args.output_dir / "pnl_distributions.png")
    plot_execution(summary, args.output_dir / "costs_and_trades.png")
    write_metadata(args, {"measure": "P", "premium_volatility": args.sigma_model,
                         "underlying_paths_sha256": hashlib.sha256(paths.tobytes()).hexdigest(),
                         "shared_paths_all_strategies_and_fee_cases": True,
                         "max_gross_net_reconciliation_error": max_reconciliation_error,
                         "costs_change_neither_trades_nor_gross_pnl": True,
                         "path_results_units": "currency/share units per one sold call; counts per path"})
    (args.output_dir / "checks.json").write_text(json.dumps({
        "shared_paths": True, "fee_invariant_decisions_and_gross_pnl": True,
        "gross_net_identity_passed": True,
        "max_absolute_reconciliation_error": max_reconciliation_error,
        "note": "These run diagnostics complement, and do not replace, the full pytest gate.",
    }, indent=2)+"\n", encoding="utf-8")
    columns = ["fee_case", "strategy", "mean_pnl_net", "std_pnl_net", "mae_net", "rmse_net",
               "mean_total_transaction_cost", "mean_traded_notional", "mean_rebalances", "mean_trades"]
    print(summary[columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nMaximum gross/net reconciliation error:", max_reconciliation_error)
    print("Saved to", args.output_dir)


if __name__ == "__main__":
    main()
