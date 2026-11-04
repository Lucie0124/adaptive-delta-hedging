"""Controlled funding, hedge-volatility and numerical-domain diagnostics."""
from pathlib import Path
import json
import sys
import warnings
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy.stats import norm
from src.hedging import hedge_paths, update_cash_account
from src.monte_carlo import simulate_gbm_paths
from src.black_scholes import bs_call_price, bs_call_delta, bs_call_gamma
from src.optimisation import apply_transaction_cost_rate
from run_validation import ref_stats, bootstrap_metrics, dump, TAB, DIAG


def main():
    TAB.mkdir(parents=True,exist_ok=True);DIAG.mkdir(parents=True,exist_ok=True)
    candidates=[("daily","daily",0.01),("weekly","weekly",0.01),
                ("adaptive_0.01","adaptive",.01),("adaptive_0.05","adaptive",.05)]
    # Under Q the expected terminal fee-free gain is zero for any predictable
    # bounded hedge; the funding-price mismatch alone sets expected gross P&L.
    paths=simulate_gbm_paths(mu=.04,sigma_true=.25,n_paths=20000,n_steps=64,seed=9876)
    expected=(bs_call_price(100,r=.04,sigma=.20)-bs_call_price(100,r=.04,sigma=.25))*np.exp(.04)
    rows=[]
    for label,strategy,epsilon in candidates:
        result=hedge_paths(paths,strategy,r=.04,sigma_model=.20,epsilon=epsilon,kappa=0,return_history=False)
        value=result["pnl_gross"]
        se=value.std(ddof=1)/np.sqrt(len(value))
        rows.append({"policy_id":label,"n_paths":20000,"n_steps":64,"seed":9876,
                     "expected_gross_pnl":expected,"mean_gross_pnl":value.mean(),"se":se,
                     "z_error":(value.mean()-expected)/se})
    pd.DataFrame(rows).to_csv(TAB/"risk_neutral_premium_invariant.csv",index=False)
    # Matched true-volatility paths: change only hedge-model volatility, hold
    # funding at the true-vol BS value. This is a diagnostic, not a candidate search.
    paths=simulate_gbm_paths(mu=.05,sigma_true=.25,n_paths=20000,n_steps=252,seed=202612)
    premium_shift=bs_call_price(100,sigma=.25)-bs_call_price(100,sigma=.20)
    data=[];pnl=[];cost=[]
    for label,strategy,epsilon in candidates:
        for model in (.20,.25):
            result=hedge_paths(paths,strategy,sigma_model=model,epsilon=epsilon,return_history=False)
            funding_shift=premium_shift if model==.20 else 0
            x=result["pnl_net"]+funding_shift
            pnl.append(x);cost.append(result["transaction_cost_fv"])
            data.append({"policy_id":label,"sigma_true":.25,"sigma_model":model,"premium":bs_call_price(100,sigma=.25),
                         **ref_stats(x,result["transaction_cost_fv"]),
                         "gross_rmse":float(np.sqrt(np.mean((result["pnl_gross"]+funding_shift)**2)))})
    samples=bootstrap_metrics(np.column_stack(pnl),np.column_stack(cost),500,94005)
    contrasts=[]
    for j,entry in enumerate(data):
        low,high=np.quantile(samples["es95"][:,j],[.025,.975])
        entry.update(es_low=float(low),es_high=float(high))
    for j,(label,_,_) in enumerate(candidates):
        d=samples["es95"][:,2*j]-samples["es95"][:,2*j+1]
        low,high=np.quantile(d,[.025,.975])
        contrasts.append({"policy_id":label,"metric":"ES wrong-minus-correct volatility; same fair premium",
                          "difference":data[2*j]["es95"]-data[2*j+1]["es95"],"low":float(low),"high":float(high)})
    pd.DataFrame(data).to_csv(TAB/"controlled_hedge_volatility_diagnostic.csv",index=False)
    pd.DataFrame(contrasts).to_csv(TAB/"controlled_hedge_volatility_contrasts.csv",index=False)
    # Bootstrap endpoint simulation error is distinct from uncertainty in population ES.
    t=pd.read_csv(ROOT/"results/03_cost_risk_optimisation/evaluation_summary.csv")
    draws=np.load(ROOT/"results/03_cost_risk_optimisation/evaluation_bootstrap.npz")["es"]
    rng=np.random.default_rng(94006);rows=[]
    for kappa in (.001,.002):
        j=int(t[(t.kappa==kappa)&(t.policy_id=="adaptive_0.05")].column_id.iloc[0])
        values=draws[:,j]
        endpoints=[np.quantile(values[rng.integers(len(values),size=len(values))],.975) for _ in range(3000)]
        low,high=np.quantile(endpoints,[.025,.975])
        rows.append({"kappa":kappa,"policy_id":"adaptive_0.05","published_upper_endpoint":np.quantile(values,.975),
                     "endpoint_mc_low":low,"endpoint_mc_high":high,"bootstrap_draws":500,"endpoint_resamples":3000})
    pd.DataFrame(rows).to_csv(TAB/"bootstrap_endpoint_monte_carlo_error.csv",index=False)
    # Record numerical-domain limitations outside the configured benchmark.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            edge={"bs_call_price_r_minus_1000":str(bs_call_price(100,r=-1000)),
                  "cash_r_1000":str(update_cash_account(-50,.5,.5,100,r=1000,dt=1)[0]),
                  "delta_tiny_positive_vol_time":str(bs_call_delta(100,sigma=1e-200,tau=1e-250)),
                  "gamma_tiny_positive_vol_time":str(bs_call_gamma(100,sigma=1e-200,tau=1e-250))}
    result=hedge_paths([[100,110]],"daily",return_history=False)
    wrong=apply_transaction_cost_rate(result,.002,.004)
    edge.update(correct_reference_pnl=float(result["pnl_net"][0]),wrong_reference_pnl=float(wrong["pnl_net"][0]),
                scope="latent API/finite-input edge limitations; not configured study errors")
    dump(DIAG/"numerical_edge_checks.json",edge)
    print(pd.DataFrame(data).to_string(index=False))


if __name__=="__main__":main()
