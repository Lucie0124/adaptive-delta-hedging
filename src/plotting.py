"""Matplotlib figures only; financial calculations live in the engine.

Static PNG outputs are requested by the project and work directly in Markdown.
Histograms span the full observed range and share bins and scales across fees.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"daily": "#2459A6", "weekly": "#B65E19", "adaptive": "#087F79"}
STYLES = {"daily": "-", "weekly": "--", "adaptive": ":"}


def _save(fig: plt.Figure, output: Path) -> None:
    fig.savefig(output, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_pricing(table: pd.DataFrame, output: Path) -> None:
    """Analytical price and estimated price with its 95% CLT interval."""
    row = table.iloc[0]
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    ax.axvline(row["analytical_price"], color=COLORS["weekly"], linestyle="--",
               label=f"Black–Scholes: {row['analytical_price']:.4f}")
    ax.errorbar(row["mc_estimate"], 0, xerr=1.959963984540054*row["standard_error"],
                fmt="o", color=COLORS["daily"], capsize=7, label="MC estimate ± 95% interval")
    ax.set(yticks=[], xlabel="Call price (currency per option)",
           title=f"Risk-neutral pricing check · {int(row['n_paths']):,} iid payoffs")
    ax.legend(loc="upper left")
    ax.grid(axis="x", alpha=0.2)
    _save(fig, output)


def plot_convergence(table: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    ax.plot(table["n_steps"], table["rmse"], "o-", color=COLORS["daily"])
    positive_errors = bool((table["rmse"] > 0).all())
    ax.set(xscale="log", yscale="log" if positive_errors else "linear",
           xlabel="Hedging intervals (log scale)",
           ylabel=f"Gross P&L RMSE ({'log scale, ' if positive_errors else ''}currency)",
           title="Frictionless replication · coupled paths, correct volatility")
    ax.set_xticks(table["n_steps"], labels=table["n_steps"].astype(str))
    for _, row in table.iterrows():
        ax.annotate(f"{row['rmse']:.3f}", (row["n_steps"], row["rmse"]),
                    xytext=(6, 5), textcoords="offset points")
    ax.grid(which="both", alpha=0.2)
    _save(fig, output)


def plot_pnl_distributions(paths: pd.DataFrame, output: Path) -> None:
    scenarios = paths["fee_case"].drop_duplicates().tolist()
    low, high = paths["pnl_net"].min(), paths["pnl_net"].max()
    if low == high:
        low, high = low-0.5, high+0.5
    bins = np.linspace(low, high, 71)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(12, 4.5), sharex=True,
                             sharey=True, layout="constrained", squeeze=False)
    for ax, scenario in zip(axes[0], scenarios):
        for strategy in COLORS:
            rows = paths[(paths["fee_case"] == scenario) & (paths["strategy"] == strategy)]
            ax.hist(rows["pnl_net"], bins=bins, density=True, histtype="step", linewidth=1.8,
                     color=COLORS[strategy], linestyle=STYLES[strategy], label=strategy.capitalize())
        ax.axvline(0, color="#64748B", linewidth=0.8)
        rate = paths.loc[paths["fee_case"] == scenario, "kappa"].iloc[0]
        ax.set(title=f"{scenario.replace('_', ' ').capitalize()} · κ = {rate:.3%}",
               xlabel="Terminal bank P&L (currency per option)")
        ax.grid(alpha=0.15)
        ax.legend()
    axes[0, 0].set_ylabel("Probability density (all observations included)")
    fig.suptitle("Same physical GBM paths for every strategy and both fee cases")
    _save(fig, output)


def plot_execution(summary: pd.DataFrame, output: Path) -> None:
    fee_rows = summary[summary["fee_case"] == "with_costs"].set_index("strategy")
    strategies = list(COLORS)
    rows = fee_rows.loc[strategies]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    axes[0].bar(x, rows["mean_total_transaction_cost"], color=list(COLORS.values()),
                 yerr=1.959963984540054*rows["se_mean_cost"], capsize=4)
    for i, value in enumerate(rows["mean_total_transaction_cost"]):
        axes[0].annotate(f"{value:.3f}", (i, value), xytext=(0, 10),
                         textcoords="offset points", ha="center")
    axes[0].set(ylabel="Currency per option", title="Mean total fees · 95% intervals for means")
    axes[0].set_ylim(0, max(0.1, rows["mean_total_transaction_cost"].max()*1.22))
    bottom = np.zeros(3)
    for field, label, color in (
        ("mean_initial_trades", "Opening", "#A8B6C7"),
        ("mean_rebalances", "Interior rebalances", "#2459A6"),
        ("mean_liquidation_trades", "Liquidation", "#B65E19"),
    ):
        axes[1].bar(x, rows[field], bottom=bottom, label=label, color=color)
        bottom += rows[field].to_numpy()
    for i, value in enumerate(rows["mean_trades"]):
        axes[1].annotate(f"{value:.1f}", (i, value), xytext=(0, 5),
                         textcoords="offset points", ha="center")
    axes[1].set(ylabel="Executed trades per path", title="Mean trade count · same for both fee cases")
    axes[1].set_ylim(0, max(1, rows["mean_trades"].max()*1.22))
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(x, [s.capitalize() for s in strategies])
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    _save(fig, output)
