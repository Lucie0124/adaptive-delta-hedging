# Research outputs

All monetary outputs are currency units per short European call. Files here
come from the current implementation.
Experiment metadata contains configuration, seeds, dependency versions, source
hashes and path hashes. Rebuilding after a documentation change updates source
hashes without changing the numerical arrays.

| Directory | Contents | Reproduction command |
|---|---|---|
| `01_model_validation/` | Risk-neutral pricing, confidence interval, coupled frequency convergence, figures | `python experiments/01_model_validation.py` |
| `02_baseline_hedging/` | 2,000-path gross/net summaries, paired differences, per-path P&L and example ledger, figures | `python experiments/02_baseline_hedging.py` |
| `03_cost_risk_optimisation/` | Calibration/evaluation summaries, fixed decisions, path arrays, bootstrap draws | `python experiments/03_cost_risk_optimisation.py` |
| `04_volatility_stress_test/` | Frozen-policy volatility comparisons, premium diagnostic, paired bootstrap draws | `python experiments/04_volatility_stress_test.py` |
| `figures/` | Selection, cost sensitivity, stress distributions and ES figures | Experiments 3 and 4 |
| `validation/` | Independent accounting checks, recomputed risk measures, paired intervals, fresh sample convergence and controlled diagnostics | `python validation/run_validation.py` then `python validation/supplementary_checks.py` |

In the selection and stress outputs, **`column_id` in the corresponding summary
CSV maps to the columns of path-result and bootstrap matrices**. Preserve this
mapping when loading NPZ files. Calibration and evaluation use distinct path
seeds; policies within either sample share rows. Stress scenarios share shocks
across volatility values. Initial-comparison CSV rows repeat a path identifier
for each policy and fee case; they are not additional independent paths.

`frozen_selection.json` records calibration-only policy choices. The stress
script checks its configuration hash and baseline evaluation consistency.
`predeclared_configuration.json` records the input configuration for that run;
the filename does not establish immutable preregistration of historical runs.

The validation baseline intervals use 1,000 bootstrap repetitions, while the
main experiment uses 500. Distinct seeds cause slightly different endpoints.
`diagnostics/fresh_50000_path_results.npz` retains the independent larger sample;
the convergence CSV uses nested prefixes of those rows. Numerical-domain and
near-boundary diagnostics are retained alongside the raw bootstrap arrays.
`reproducibility.json` records the post-reorganisation test and exact numerical
comparison, and `tables/reproduction_comparison.csv` lists compared tables.

Raw path matrices and bootstrap draws are intentionally retained: they support
recomputation of published uncertainty and are not duplicate presentation files.
