"""Independent ledger, empirical-risk and sample-size validation.

Run: python validation/run_validation.py
"""

from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import platform
import subprocess
import sys
import time
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm, ttest_1samp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.hedging import hedge_paths
from src.monte_carlo import simulate_gbm_paths

OUT = ROOT/"results/validation"
TAB, FIG, DIAG = (OUT/name for name in ("tables", "figures", "diagnostics"))
SEEDS = {"bootstrap": 94001, "fresh_paths": 862202, "fresh_bootstrap": 94002,
         "accounting": 108, "pricing": 94003}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if hasattr(x,"item") else str(x))+"\n", encoding="utf-8")


def snapshot():
    """Hash the self-contained scientific inputs for the current run."""
    files = [p for folder in ("src", "experiments", "configs", "validation")
             for p in (ROOT/folder).glob("*.py" if folder != "configs" else "*.json")]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}


def run_logged(manifest, name, args):
    command = [str(x) for x in args]
    start = time.perf_counter()
    print("Executing", name, flush=True)
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    log = DIAG/f"{name}.txt"
    log.write_text(result.stdout+"\nSTDERR\n"+result.stderr, encoding="utf-8")
    record = {"name": name, "command": command, "exit_code": result.returncode,
              "status": "VERIFIED" if result.returncode == 0 else "NOT VERIFIED",
              "elapsed_seconds": time.perf_counter()-start, "log": str(log.relative_to(ROOT))}
    manifest["executions"].append(record)
    dump(OUT/"validation_manifest.json", manifest)
    if result.returncode:
        print("Execution failure:", name, result.stderr[-1500:], flush=True)
    return result.returncode == 0


def ref_es(losses, alpha=0.95):
    """Independent definition: integrate the empirical inverse CDF with cell weights."""
    x = np.sort(np.asarray(losses, float))
    n = len(x)
    right = np.arange(1, n+1)/n
    weights = np.maximum(0, right-np.maximum(right-1/n, alpha))
    return float(np.dot(x, weights)/(1-alpha))


def ref_stats(pnl, cost):
    x = np.asarray(pnl)
    losses = -x
    return {"mean_pnl": float(x.mean()), "sd_pnl": float(x.std(ddof=1)),
            "mae": float(np.abs(x).mean()), "rmse": float(np.sqrt(np.mean(x*x))),
            "mean_cost": float(np.mean(cost)),
            "var95": float(np.sort(losses)[int(np.ceil(.95*len(x)))-1]), "es95": ref_es(losses)}


def bootstrap_metrics(pnl, costs, b, seed):
    """Independent paired bootstrap; all validation sample counts have an integer 5% tail."""
    x, c = np.asarray(pnl), np.asarray(costs)
    n, k = x.shape
    tail_count = n//20
    assert n % 20 == 0
    rng = np.random.default_rng(seed)
    samples = {name: np.empty((b,k)) for name in ("mean_pnl", "mae", "rmse", "mean_cost", "es95")}
    for rep in range(b):
        ids = rng.integers(n, size=n)
        y = x[ids]
        samples["mean_pnl"][rep] = y.mean(axis=0)
        samples["mae"][rep] = np.abs(y).mean(axis=0)
        samples["rmse"][rep] = np.sqrt((y*y).mean(axis=0))
        samples["mean_cost"][rep] = c[ids].mean(axis=0)
        loss = -y
        samples["es95"][rep] = np.partition(loss, n-tail_count, axis=0)[-tail_count:].mean(axis=0)
    return samples


def accounting_diagnostics(candidates):
    rows = []
    for rate in (0.0, 0.03, -0.02):
        paths = simulate_gbm_paths(n_paths=1000, n_steps=252, mu=.05, seed=SEEDS["accounting"])
        dt, terminal = 1/252, paths[:,-1]
        for candidate in candidates:
            h = hedge_paths(paths, candidate["strategy"], r=rate,
                            epsilon=candidate["epsilon"] or 0, return_history=True)
            hist = h["history"]
            phi = hist["holding"]
            # Discounted stock gains independently reconstruct the hedge; no cash recurrence reused.
            discounted_gains = (phi[:,:-1]*(paths[:,1:]-paths[:,:-1]*np.exp(rate*dt))
                                *np.exp(rate*(1-h["times"][1:])))
            premium = 100*norm.cdf((rate+.02)/.2)-100*np.exp(-rate)*norm.cdf((rate-.02)/.2)
            gross_ref = premium*np.exp(rate)+discounted_gains.sum(axis=1)-np.maximum(terminal-100,0)
            q = np.diff(phi,axis=1,prepend=np.zeros((len(paths),1)))
            fees = .002*np.abs(q)*paths
            fv = (fees*np.exp(rate*(1-h["times"]))).sum(axis=1)
            pre_trade = hist["cash_net"][:,:-1]*np.exp(rate*dt)+phi[:,:-1]*paths[:,1:]
            rows.append({"r": rate, "policy_id": candidate["policy_id"], "paths": len(paths),
                         "gross_discounted_gain_residual": float(np.max(np.abs(gross_ref-h["pnl_gross"]))),
                         "net_discounted_gain_residual": float(np.max(np.abs(gross_ref-fv-h["pnl_net"]))),
                         "execution_identity_residual": float(np.max(np.abs(hist["portfolio_net"][:,1:]-pre_trade+fees[:,1:]))),
                         "gross_net_residual": float(np.max(np.abs(h["pnl_gross"]-h["pnl_net"]-fv))),
                         "terminal_target_absent": bool(np.isnan(hist["target_delta"][:,-1]).all()),
                         "terminal_stock_zero": bool((phi[:,-1]==0).all())})
    table = pd.DataFrame(rows)
    table.to_csv(TAB/"accounting_checks.csv", index=False)
    assert table.filter(regex="residual").to_numpy().max() < 1e-10


def validate_saved_metrics(candidates):
    table = pd.read_csv(ROOT/"results/03_cost_risk_optimisation/evaluation_summary.csv")
    arrays = np.load(ROOT/"results/03_cost_risk_optimisation/evaluation_path_results.npz")
    base = table.loc[table.kappa==.002].copy()
    ids = base.column_id.to_numpy(int)
    x, c = arrays["pnl_net"][:,ids], arrays["cost_fv"][:,ids]
    recompute = []
    for _, row in table.iterrows():
        j = int(row.column_id)
        independent = ref_stats(arrays["pnl_net"][:,j],arrays["cost_fv"][:,j])
        recompute.append({"policy_id":row.policy_id,"kappa":row.kappa,
                          **independent,"reported_es":row.es,"es_residual":independent["es95"]-row.es})
    pd.DataFrame(recompute).to_csv(TAB/"all_fee_metric_recomputation.csv",index=False)
    assert max(abs(r["es_residual"]) for r in recompute)<1e-10
    samples = bootstrap_metrics(x,c,1000,SEEDS["bootstrap"])
    np.savez_compressed(DIAG/"independent_baseline_bootstrap.npz",**samples)
    summaries = []
    for j,(_,row) in enumerate(base.iterrows()):
        stats = ref_stats(x[:,j],c[:,j])
        summaries.append({"policy_id":row.policy_id,"n_paths":len(x),**stats,
                          "mean_notional":row.mean_traded_notional,"mean_trades":row.mean_trades,
                          "mean_rebalances":row.mean_rebalances,"gross_rmse":row.rmse_gross})
        for metric in samples:
            low,high = np.quantile(samples[metric][:,j],[.025,.975])
            summaries[-1].update({metric+"_low":float(low),metric+"_high":float(high)})
    pd.DataFrame(summaries).to_csv(TAB/"baseline_metrics_with_intervals.csv",index=False)
    pairs = [(0,3),(1,3),(0,6),(1,6),(5,6),(0,1)]
    differences = []
    for first,second in pairs:
        for metric in samples:
            values = samples[metric][:,first]-samples[metric][:,second]
            low,high = np.quantile(values,[.025,.975])
            differences.append({"first":base.iloc[first].policy_id,"second":base.iloc[second].policy_id,
                                "metric":metric,"difference":summaries[first][metric]-summaries[second][metric],
                                "low":float(low),"high":float(high)})
    pd.DataFrame(differences).to_csv(TAB/"paired_comparisons.csv",index=False)
    # Family of six mean-P&L tests only; ES intervals remain explicitly pointwise.
    pvalues = np.array([ttest_1samp(x[:,a]-x[:,b],0).pvalue for a,b in pairs])
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1,np.maximum.accumulate(pvalues[order]*(len(pairs)-np.arange(len(pairs)))))
    pd.DataFrame([{"first":base.iloc[a].policy_id,"second":base.iloc[b].policy_id,
                   "raw_p_mean_pnl":pvalues[k],"holm_p_six_mean_pnl_tests":adjusted[k]}
                  for k,(a,b) in enumerate(pairs)]).to_csv(TAB/"multiple_comparison_check.csv",index=False)
    # Interaction: does volatility change the relative ES gap? Reuse paired scenario draws, not independent CIs.
    stress = pd.read_csv(ROOT/"results/04_volatility_stress_test/stress_summary.csv")
    draws = np.load(ROOT/"results/04_volatility_stress_test/paired_volatility_bootstrap.npz")["es"]
    interactions=[]
    for a,b in [(0,3),(1,3),(0,6),(1,6),(5,6)]:
        gap0 = stress.iloc[a].es-stress.iloc[b].es
        gap1 = stress.iloc[7+a].es-stress.iloc[7+b].es
        contrast=(draws[:,7+a]-draws[:,7+b])-(draws[:,a]-draws[:,b])
        low,high=np.quantile(contrast,[.025,.975])
        interactions.append({"first":base.iloc[a].policy_id,"second":base.iloc[b].policy_id,
                             "es_gap_20":gap0,"es_gap_25":gap1,"gap_change":gap1-gap0,
                             "low":float(low),"high":float(high)})
    pd.DataFrame(interactions).to_csv(TAB/"volatility_ranking_interactions.csv",index=False)
    return base, summaries


def fresh_convergence(candidates):
    print("Independent sample-size convergence: 50,000 fresh paths; frozen policies only.",flush=True)
    paths=simulate_gbm_paths(n_paths=50000,n_steps=252,seed=SEEDS["fresh_paths"])
    pnl,cost,gross=[],[],[]
    for candidate in candidates:
        r=hedge_paths(paths,candidate["strategy"],epsilon=candidate["epsilon"] or 0,return_history=False)
        pnl.append(r["pnl_net"]); cost.append(r["transaction_cost_fv"]); gross.append(r["pnl_gross"])
    x,c,g=np.column_stack(pnl),np.column_stack(cost),np.column_stack(gross)
    np.savez_compressed(DIAG/"fresh_50000_path_results.npz",pnl_net=x,cost=c,pnl_gross=g)
    rows=[]
    for m in (2000,5000,20000,50000):
        b=bootstrap_metrics(x[:m],c[:m],500,SEEDS["fresh_bootstrap"]+m)
        for j,candidate in enumerate(candidates):
            stats=ref_stats(x[:m,j],c[:m,j])
            low,high=np.quantile(b["es95"][:,j],[.025,.975])
            rows.append({"n_paths":m,"tail_observations":m//20,"policy_id":candidate["policy_id"],
                         **stats,"gross_rmse":float(np.sqrt(np.mean(g[:m,j]**2))),
                         "es_low":float(low),"es_high":float(high),"frozen_primary_policy":candidate["policy_id"]=="adaptive_0.05"})
    result=pd.DataFrame(rows)
    result.to_csv(TAB/"monte_carlo_path_count_convergence.csv",index=False)
    # Confirm the previously borderline fee/budget setting on NEW data, without selecting another policy.
    shifted=g[:,6]-c[:,6]/2
    b=bootstrap_metrics(shifted[:,None],(c[:,6]/2)[:,None],1000,94004)
    low,high=np.quantile(b["es95"][:,0],[.025,.975])
    dump(DIAG/"borderline_confirmation.json",{"policy_id":"adaptive_0.05","kappa":.001,"risk_budget":2,
          "n_paths":50000,"path_seed":SEEDS["fresh_paths"]+0,"bootstrap_seed":94004,
          **ref_stats(shifted,c[:,6]/2),"es_low":float(low),"es_high":float(high),
          "reselected_on_validation_data":False})
    fig,ax=plt.subplots(figsize=(8,4.5),layout="constrained")
    for policy,color in [("daily","#2459A6"),("weekly","#B65E19"),("adaptive_0.01","#168C83"),("adaptive_0.05","#A04574")]:
        d=result[result.policy_id==policy]
        ax.errorbar(d.n_paths,d.es95,yerr=np.maximum(0,np.vstack([d.es95-d.es_low,d.es_high-d.es95])),fmt="o-",capsize=3,color=color,label=policy)
    ax.axhline(3,color="gray",linestyle=":",label="Predefined primary limit")
    ax.set(xscale="log",xlabel="Paths (nested prefixes of a fresh validation sample)",ylabel="ES95 of net losses (currency)",title="Monte Carlo precision: policy fixed before this validation sample")
    ax.set_xticks([2000,5000,20000,50000],labels=["2,000","5,000","20,000","50,000"])
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(FIG/"path_count_convergence.png",dpi=180);plt.close(fig)


def reproduce_comparison(reproductions):
    rows=[]
    pairs=[("01_model_validation","pricing_validation.csv"),("01_model_validation","frictionless_convergence.csv"),
           ("02_baseline_hedging","summary.csv"),("03_cost_risk_optimisation","calibration_summary.csv"),
           ("03_cost_risk_optimisation","evaluation_summary.csv"),("04_volatility_stress_test","stress_summary.csv")]
    for folder,name in pairs:
        archived=ROOT/"results"/folder/name
        rerun=reproductions/folder/name
        if not rerun.exists():
            rows.append({"artifact":f"{folder}/{name}","status":"NOT VERIFIED","max_numeric_difference":None});continue
        a,b=pd.read_csv(archived),pd.read_csv(rerun)
        numeric=a.select_dtypes(include="number").columns
        difference=float(np.nanmax(np.abs(a[numeric].to_numpy()-b[numeric].to_numpy())))
        matches=a.shape==b.shape and difference<1e-10
        rows.append({"artifact":f"{folder}/{name}","status":"VERIFIED" if matches else "NOT VERIFIED","max_numeric_difference":difference})
    pd.DataFrame(rows).to_csv(TAB/"reproduction_comparison.csv",index=False)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduce", action="store_true",
                        help="Also execute pytest and all four experiments, comparing numeric tables.")
    args = parser.parse_args()
    for p in (OUT, TAB, FIG, DIAG):
        p.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT/"configs/study.json").read_text())
    frozen = json.loads((ROOT/"results/03_cost_risk_optimisation/frozen_selection.json").read_text())
    candidates = frozen["candidates"]
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "Independent numerical validation of fixed policies",
                "project_configuration": config, "seeds": SEEDS,
                "fresh_path_counts": [2000,5000,20000,50000],
                "bootstrap_repetitions": {"saved_baseline":1000,"fresh_prefixes":500,"borderline_confirmation":1000},
                "python": platform.python_version(),
                "versions": {x:importlib.metadata.version(x) for x in ["numpy","scipy","pandas","matplotlib","pytest"]},
                "source_sha256": snapshot(), "executions": []}
    if args.reproduce:
        with tempfile.TemporaryDirectory(prefix="hedging-reproduction-") as folder:
            reproductions = Path(folder)
            assert run_logged(manifest,"pytest",[sys.executable,"-m","pytest","-q"])
            for number,name in [(1,"model_validation"),(2,"baseline_hedging"),(3,"cost_risk_optimisation"),(4,"volatility_stress_test")]:
                study = f"{number:02d}_{name}"
                cmd = [sys.executable,f"experiments/{study}.py","--output-dir",reproductions/study]
                if number >= 3:
                    cmd += ["--figures-dir",reproductions/"figures"]
                if number == 4:
                    cmd += ["--selection-file",reproductions/"03_cost_risk_optimisation/frozen_selection.json"]
                assert run_logged(manifest,study,cmd), study
            comparisons = reproduce_comparison(reproductions)
            assert all(r["status"] == "VERIFIED" for r in comparisons), comparisons
            manifest["reproduction_comparisons"] = comparisons
    print("Independent accounting and metric checks", flush=True)
    accounting_diagnostics(candidates)
    validate_saved_metrics(candidates)
    fresh_convergence(candidates)
    manifest.update(completed_utc=datetime.now(timezone.utc).isoformat(),
                    accounting_status="VERIFIED", independent_metric_status="VERIFIED")
    dump(OUT/"validation_manifest.json",manifest)
    print("Numerical validation completed",flush=True)


if __name__ == "__main__":
    main()
