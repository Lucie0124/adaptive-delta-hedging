"""Static publication figures for the simulated C/D study; no financial logic."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .plotting import _save


def policy_label(policy: str) -> str:
    return policy.capitalize() if policy in ("daily", "weekly") else f"Adaptive ε={float(policy.split('_')[1]):g}"


def policy_colors(policies: list[str]) -> dict[str, str]:
    adaptive = [p for p in policies if p.startswith("adaptive")]
    palette = ["#00695C", "#168C83", "#3C726D", "#63568A", "#A04574"]
    return {"daily": "#2459A6", "weekly": "#B65E19",
            **{p: palette[i % len(palette)] for i, p in enumerate(adaptive)}}


def plot_frontier(table: pd.DataFrame, budget: float, selected: str | None, output: Path) -> None:
    colors = policy_colors(table["policy_id"].tolist())
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    frontier = table[table["pareto_efficient"]].sort_values("es")
    ax.plot(frontier["es"], frontier["mean_cost"], color="#536576", linewidth=1,
             linestyle="--", label="Empirical non-dominated frontier")
    markers = ["o", "s", "v", "^", "D", "P", "X"]
    for i, (_, row) in enumerate(table.iterrows()):
        policy = row["policy_id"]
        ax.errorbar(row["es"], row["mean_cost"],
                     xerr=[[max(0, row["es"]-row["es_ci_low"])], [max(0, row["es_ci_high"]-row["es"])]],
                     yerr=[[max(0, row["mean_cost"]-row["mean_cost_ci_low"])],
                           [max(0, row["mean_cost_ci_high"]-row["mean_cost"])]],
                     fmt=markers[i % len(markers)], color=colors[policy], capsize=3,
                     markersize=7, alpha=0.9, label=policy_label(policy))
        if policy == selected:
            ax.scatter(row["es"], row["mean_cost"], marker="o", s=210, facecolors="none",
                        edgecolors="#142745", linewidths=1.5, label="Frozen primary-budget choice")
    ax.axvline(budget, color="#922B3E", linestyle=":", label=f"Predefined budget: {budget:g}")
    ax.set(xlabel="ES95 of net terminal losses (currency per option)",
           ylabel="Mean terminal-valued transaction cost (currency per option)",
           title=f"Calibration cost–risk frontier · κ = {table['kappa'].iloc[0]:.3%}\nPointwise paired-bootstrap 95% intervals")
    ax.set_ylim(bottom=0)
    ax.margins(x=0.20, y=0.12)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower left", fontsize=8.5, ncol=2)
    _save(fig, output)


def plot_cost_sensitivity(table: pd.DataFrame, baseline_kappa: float, output: Path) -> None:
    policies = table["policy_id"].drop_duplicates().tolist()
    colors = policy_colors(policies)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for i, policy in enumerate(policies):
        rows = table[table["policy_id"] == policy].sort_values("kappa")
        axes[0].plot(rows["kappa"]*10000, rows["mean_cost"], marker="o", markersize=4,
                      color=colors[policy], linestyle="--" if i % 2 else "-", label=policy_label(policy))
    axes[0].set(xlabel="Experimental proportional half-spread (basis points)",
                ylabel="Mean terminal-valued fees (currency per option)",
                title="Cost sensitivity on identical stock trades", ylim=(0, None))
    axes[0].legend(fontsize=8)
    rows = table[table["kappa"] == baseline_kappa]
    positions = np.arange(len(rows))
    axes[1].barh(positions, rows["mean_trades"], color=[colors[p] for p in rows["policy_id"]])
    axes[1].set_yticks(positions, [policy_label(p) for p in rows["policy_id"]])
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Mean executed trades (opening + interior + closing)",
                title="Ticket counts differ from traded notional", xlim=(0, None))
    for i, value in enumerate(rows["mean_trades"]):
        axes[1].annotate(f"{value:.1f}", (value, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9)
    axes[1].margins(x=0.12)
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    _save(fig, output)


def plot_stress_distributions(
    summaries: pd.DataFrame, arrays: dict[float, dict[str, np.ndarray]],
    policies: list[str], output: Path,
) -> None:
    scenarios = [float(summaries["sigma_true"].min()), 0.25]
    colors = policy_colors(summaries["policy_id"].drop_duplicates().tolist())
    plotted = []
    for vol in scenarios:
        group = summaries[summaries["sigma_true"] == vol]
        for policy in policies:
            index = int(group.loc[group["policy_id"] == policy, "column_id"].iloc[0])
            plotted.append(arrays[vol]["pnl_net"][:, index])
    low = min(float(x.min()) for x in plotted)
    high = max(float(x.max()) for x in plotted)
    bins = np.linspace(low, high if high > low else low+1, 85)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True, sharey=True, layout="constrained")
    for ax, vol in zip(axes, scenarios):
        group = summaries[summaries["sigma_true"] == vol]
        for i, policy in enumerate(policies):
            index = int(group.loc[group["policy_id"] == policy, "column_id"].iloc[0])
            ax.hist(arrays[vol]["pnl_net"][:, index], bins=bins, density=True, histtype="step",
                     color=colors[policy], linestyle=["-", "--", ":", "-."][i % 4],
                     linewidth=1.6, label=policy_label(policy))
        ax.axvline(0, color="#64748B", linewidth=0.8)
        ax.set(title=f"True volatility {vol:.0%}", xlabel="Net terminal bank P&L (currency per option)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Probability density · full observed range")
    fig.suptitle(f"Frozen policies · model volatility and sold-call premium held at {summaries['sigma_model'].iloc[0]:.0%}")
    _save(fig, output)


def plot_stress_es(table: pd.DataFrame, budget: float, output: Path) -> None:
    policies = table["policy_id"].drop_duplicates().tolist()
    scenarios = sorted(table["sigma_true"].unique())
    colors = policy_colors(policies)
    fig, ax = plt.subplots(figsize=(11, 6.5), layout="constrained")
    height = 0.72/len(scenarios)
    hatches = ["", "///", "xx", "..."]
    for j, vol in enumerate(scenarios):
        rows = table[table["sigma_true"] == vol].set_index("policy_id").loc[policies]
        y = np.arange(len(policies))+(j-(len(scenarios)-1)/2)*height
        ax.barh(y, rows["es"], height=height*0.90, color=[colors[p] for p in policies],
                 hatch=hatches[j % 4], edgecolor="white", linewidth=0.6,
                 xerr=np.maximum(0, np.vstack([rows["es"]-rows["es_ci_low"], rows["es_ci_high"]-rows["es"]])),
                 error_kw={"elinewidth": 0.8, "capsize": 2})
    ax.axvline(budget, color="#922B3E", linestyle=":", label=f"Primary risk budget {budget:g}")
    ax.set_yticks(np.arange(len(policies)), [policy_label(p) for p in policies])
    ax.invert_yaxis()
    ax.set(xlabel="ES95 of net terminal losses (currency per option)", xlim=(0, None),
           title=f"Volatility misspecification · fixed {table['sigma_model'].iloc[0]:.0%} hedge volatility and premium\nPointwise paired-bootstrap 95% intervals")
    handles = [Patch(facecolor="#64748B", hatch=hatches[i % 4], label=f"True σ={vol:.0%}") for i, vol in enumerate(scenarios)]
    handles.append(Line2D([0], [0], color="#922B3E", linestyle=":", label=f"Budget {budget:g}"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    _save(fig, output)


def plot_stress_heatmap(table: pd.DataFrame, selected: str | None, output: Path) -> None:
    policies = table["policy_id"].drop_duplicates().tolist()
    pivot = table.pivot(index="sigma_true", columns="policy_id", values="es").loc[:, policies]
    fig, ax = plt.subplots(figsize=(12, 4.5), layout="constrained")
    values = pivot.to_numpy()
    plot = ax.imshow(values, cmap="YlOrRd", vmin=0, vmax=float(values.max()), aspect="auto")
    ax.set_yticks(np.arange(len(pivot)), [f"{v:.0%}" for v in pivot.index])
    ax.set_xticks(np.arange(len(policies)), [policy_label(p)+( " *" if p == selected else "") for p in policies],
                   rotation=25, ha="right")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i,j]:.2f}", ha="center", va="center",
                     color="white" if values[i,j] > 0.6*values.max() else "#142745")
    ax.set(ylabel="True annual volatility", title="ES95 of net losses under volatility stress\nFixed premium; * frozen primary-budget policy")
    fig.colorbar(plot, ax=ax, label="Currency per sold option")
    _save(fig, output)
