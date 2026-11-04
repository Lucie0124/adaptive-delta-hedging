# Numerical validation

All results below use the current self-financing implementation, per sold call.
Historical coursework results are not reused in these calculations.
The complete parameter and seed definitions are in [methodology.md](methodology.md)
and [configs/study.json](../configs/study.json). The baseline comparison uses
2,000 physical paths; policy selection uses 20,000 calibration paths and
20,000 independent evaluation paths. A separate 50,000-path sample evaluates
the frozen policies without selecting a new threshold.

## Analytical pricing and frictionless convergence

The risk-neutral pricing check uses 2,000 exact terminal draws with seed
314159. Scenario simulations for hedging instead use the physical drift.

| Pricing diagnostic | Currency per call |
| --- | --- |
| Analytical Black-Scholes | 7.965567 |
| Risk-neutral Monte Carlo | 7.906879 |
| Standard error | 0.295376 |
| 95% interval | [7.327953, 8.485806] |
| Absolute estimation error | 0.058688 |

The analytical price lies within this run's 95% normal-CLT interval. A valid
95% interval need not contain the analytical price in every finite sample;
the regression test uses a wider statistical tolerance. This validates pricing
under the stated model, not a forecast of physical expected payoffs.

A coupled frictionless experiment uses 2,000 risk-neutral paths with seed
271828. Meshes are subsamples of the same 1,008-step Brownian paths, with
correct model volatility and zero costs.

| Intervals | Mean gross P&L | P&L SD | MAE | RMSE |
| --- | --- | --- | --- | --- |
| 12 | -0.0312 | 1.9446 | 1.4662 | 1.9443 |
| 48 | -0.0084 | 0.9829 | 0.7415 | 0.9827 |
| 252 | -0.0108 | 0.4440 | 0.3304 | 0.4440 |
| 1008 | 0.0019 | 0.2220 | 0.1657 | 0.2219 |

The paired mean-square-error reduction from 12 to 1,008 intervals is 3.731212,
with standard error 0.161478 and 95% interval [3.414721, 4.047702]. This supports
overall convergence; no strict sample-by-sample monotonicity is assumed.

![Frictionless replication error across coupled meshes.](../results/01_model_validation/frictionless_convergence.png)

## Financial accounting and numerical consistency

Deterministic ledger tests check cash interest, no-trade intervals, execution
conservation, initial and terminal fees, payoff settlement and the absence of
a maturity Delta trade. Independent discounted-gain checks cover all seven
policies at rates 0, 0.03 and -0.02, with 1,000 paths per case (seed 108).
The maximum absolute accounting residual is **1.05e-12** currency
units. Gross minus net P&L equals terminal-valued fees path by path; costs and
payoff are each charged once. A failed accounting check blocks interpretation
of the results.

Independent sorted-loss calculations reproduce ES for every fee-policy case
within floating-point precision. Numerical tables and saved path-result arrays
are compared when reproducing the four experiments. Current execution evidence
is in `results/validation/reproducibility.json` and
`results/validation/tables/reproduction_comparison.csv`.

An additional risk-neutral financial check uses 20,000 paths, 64 intervals,
seed 9876, rate 0.04, true volatility 0.25 and model volatility 0.20. Under
the risk-neutral drift, a predictable bounded stock hedge has zero expected
discounted trading gain. Expected gross terminal P&L therefore equals

$$[C_{BS}(0.20)-C_{BS}(0.25)]e^{0.04}=-1.990023,$$

irrespective of the chosen hedge policy. The observed means lie within about
1.62 standard errors of this value.

| Policy | Expected gross | Observed gross | SE | z error |
| --- | --- | --- | --- | --- |
| Daily | -1.9900 | -2.0012 | 0.0097 | -1.1509 |
| Weekly | -1.9900 | -2.0183 | 0.0175 | -1.6128 |
| Adaptive 0.01 | -1.9900 | -2.0015 | 0.0097 | -1.1805 |
| Adaptive 0.05 | -1.9900 | -1.9994 | 0.0102 | -0.9170 |

# Simulation results

## Initial 2,000-path hedging comparison

Daily, weekly and adaptive threshold 0.01 reuse the same physical paths
(seed 42) in both fee scenarios. The premium is the 20% Black-Scholes price.
Gross P&L is a fee-free shadow with identical trades; it is not obtained by
adding average fees to every net outcome.

| Fee case | Policy | Mean net P&L | SD | RMSE | Mean cost |
| --- | --- | --- | --- | --- | --- |
| no costs | Daily | -0.0003 | 0.4439 | 0.4438 | 0.0000 |
| no costs | Weekly | -0.0388 | 0.9731 | 0.9736 | 0.0000 |
| no costs | adaptive | 0.0011 | 0.4478 | 0.4476 | 0.0000 |
| with costs | Daily | -1.2478 | 0.5914 | 1.3808 | 1.2476 |
| with costs | Weekly | -0.7353 | 1.0231 | 1.2598 | 0.6965 |
| with costs | adaptive | -1.1826 | 0.5976 | 1.3250 | 1.1838 |

At zero costs, daily has slightly lower replication RMSE than adaptive 0.01,
while weekly has much higher dispersion. At a 0.2% half-spread, weekly saves
fees and has lower net MAE/RMSE than the two frequent policies in this sample.
That does not establish its tail-risk feasibility. The adaptive policy removes
about 40% of daily executed tickets but saves only about 5.1% of fees: proportional
costs depend on notional, and skipped small changes accumulate into larger trades.

## Independent 20,000-path baseline evaluation

At half-spread 0.002, the seven policies give the following evaluation results.
P&L SD is the sample standard deviation, not the standard error of its mean.
RMSE measures distance from zero P&L, including the mean funding shortfall.

| Policy | Mean net P&L | SD | Net RMSE | Gross RMSE | Mean cost | ES95 |
| --- | --- | --- | --- | --- | --- | --- |
| Daily | -1.2556 | 0.6038 | 1.3932 | 0.4424 | 1.2520 | 2.7653 |
| Weekly | -0.6918 | 0.9966 | 1.2131 | 0.9505 | 0.6955 | 3.1004 |
| Adaptive 0.005 | -1.2356 | 0.6072 | 1.3768 | 0.4437 | 1.2315 | 2.7512 |
| Adaptive 0.01 | -1.1937 | 0.6129 | 1.3418 | 0.4491 | 1.1884 | 2.7168 |
| Adaptive 0.02 | -1.0833 | 0.6268 | 1.2516 | 0.4764 | 1.0756 | 2.6269 |
| Adaptive 0.03 | -0.9731 | 0.6538 | 1.1723 | 0.5270 | 0.9645 | 2.5403 |
| Adaptive 0.05 | -0.8119 | 0.7544 | 1.1082 | 0.6669 | 0.7981 | 2.5372 |

The premium includes no cost loading, explaining much of the negative mean
net P&L. Fees vary across paths and correlate with replication outcomes, so
they also affect dispersion. Daily remains more accurate in gross RMSE than
the wider adaptive bands. Lower net RMSE is not equivalent to lower gross
replication error or to minimum expected cost subject to ES.

| Policy | Statistic | Estimate | 95% lower | 95% upper |
| --- | --- | --- | --- | --- |
| Daily | Mean net P&L | -1.2556 | -1.2638 | -1.2474 |
| Daily | Mean cost | 1.2520 | 1.2467 | 1.2575 |
| Daily | ES95 | 2.7653 | 2.7338 | 2.7968 |
| Weekly | Mean net P&L | -0.6918 | -0.7061 | -0.6769 |
| Weekly | Mean cost | 0.6955 | 0.6926 | 0.6984 |
| Weekly | ES95 | 3.1004 | 3.0385 | 3.1558 |
| Adaptive 0.01 | Mean net P&L | -1.1937 | -1.2021 | -1.1853 |
| Adaptive 0.01 | Mean cost | 1.1884 | 1.1830 | 1.1939 |
| Adaptive 0.01 | ES95 | 2.7168 | 2.6858 | 2.7492 |
| Adaptive 0.05 | Mean net P&L | -0.8119 | -0.8221 | -0.8014 |
| Adaptive 0.05 | Mean cost | 0.7981 | 0.7939 | 0.8025 |
| Adaptive 0.05 | ES95 | 2.5372 | 2.5044 | 2.5725 |

These intervals use 1,000 independent paired bootstrap resamples, seed 94001.
The main experiment separately stores 500-resample intervals (seed 732).
Their endpoints differ slightly because of bootstrap simulation error; they
estimate the same fixed-policy quantities.

| Policy | Notional | Interior rebalances | All tickets |
| --- | --- | --- | --- |
| Daily | 626.0204 | 249.3761 | 251.3707 |
| Weekly | 347.7383 | 49.8183 | 51.8180 |
| Adaptive 0.01 | 594.2005 | 148.5406 | 150.5406 |
| Adaptive 0.05 | 399.0564 | 38.2629 | 40.2629 |

Opening, interior and closing tickets are distinct. Zero share changes do not
count as executions; normal-CDF saturation near expiry can reduce executed
counts below eligible calendar dates. Adaptive 0.05 cuts roughly 84% of tickets
and 36.3% of fees relative to daily, rather than reducing both proportionately.

## Paired comparisons and statistical interpretation

All policies share stock paths. The following contrasts resample each path
jointly across policies and report first minus second.

| First minus second | ES95 difference | 95% lower | 95% upper |
| --- | --- | --- | --- |
| Daily minus Adaptive 0.01 | 0.0485 | 0.0440 | 0.0526 |
| Weekly minus Adaptive 0.01 | 0.3835 | 0.3194 | 0.4393 |
| Daily minus Adaptive 0.05 | 0.2281 | 0.1994 | 0.2559 |
| Weekly minus Adaptive 0.05 | 0.5632 | 0.5013 | 0.6213 |
| Adaptive 0.03 minus Adaptive 0.05 | 0.0031 | -0.0250 | 0.0281 |
| Daily minus Weekly | -0.3351 | -0.3916 | -0.2733 |

Daily minus adaptive 0.05 mean cost is 0.453928; the independent 1,000-resample
95% interval is [0.452344, 0.455614]. The main experiment's 500-resample interval
is [0.452279, 0.455614]. Weekly is cheaper than adaptive 0.05 but has higher ES.
Adaptive 0.03 and 0.05 have nearly equal ES: their interval contains zero.
The study therefore does not establish which of these two has lower population
ES, despite a clear cost difference.

Holm correction across the six specified mean-P&L contrasts retains their
differences at 5%; the ES intervals remain pointwise and are not converted
into simultaneous significance claims. Some stored tiny p-values underflow
to zero numerically; they do not mean an exact zero probability.

# Cost-constrained policy selection

## Calibration and independent evaluation

Selection minimises calibration mean terminal-valued fees among policies
whose point ES estimate meets the chosen budget. Calibration choices are
saved before independent evaluation paths are generated. Ties use lower ES,
then lexical policy ID. Evaluation and stress never reselect a policy.

At half-spread 0.002, the point-estimate calibration frontier contains adaptive
0.03, adaptive 0.05 and weekly. The selected threshold is **0.05**, with
calibration cost 0.796018 and ES 2.530101 under budget 3. It lies at the **upper
boundary of the tested threshold grid**, so it is only a finite-set optimum.

| Budget | Frozen choice | Evaluation cost | Evaluation ES95 | Headroom |
| --- | --- | --- | --- | --- |
| 1 | None | -- | -- | -- |
| 2 | None | -- | -- | -- |
| 3 | Adaptive 0.05 | 0.7981 | 2.5372 | 0.4628 |

The selected evaluation ES is 2.537173 with the main experiment's 95% interval
[2.506753, 2.569162], leaving point-estimate headroom 0.462827. No calibration
candidate meets budgets 1 or 2 at the 0.2% half-spread. This is infeasibility
within the seven tested policies, not proof that no admissible hedge could
meet those limits.

![Cost-risk trade-off in calibration with pointwise uncertainty.](../results/figures/cost_risk_frontier.png)

## Transaction-cost sensitivity

These policies depend on stock prices and model Delta, not cash balances or
fees. On a fixed trade sequence, fees scale exactly with the proportional
rate. This identity includes opening, interior and closing expenses and their
financing; it is tested against full direct ledger simulations.

| Half-spread | Frozen choice at budget 3 | Evaluation cost | Evaluation ES95 |
| --- | --- | --- | --- |
| 0.0000 | Daily | 0.0000 | 1.0071 |
| 0.0001 | Weekly | 0.0348 | 2.2337 |
| 0.0005 | Weekly | 0.1739 | 2.4134 |
| 0.0010 | Weekly | 0.3477 | 2.6403 |
| 0.0020 | Adaptive 0.05 | 0.7981 | 2.5372 |
| 0.0050 | None | -- | -- |

At zero fees all cost objectives tie, and the lower-ES tie break reports daily.
At small positive spreads weekly can meet budget 3 and is cheapest. At 0.005
no candidate is calibration-feasible. Thus adaptive hedging is not universally
optimal even within this study's model and candidate set.

## Sample-size convergence and a near-boundary case

A fresh independent sample uses seed 862202. The policies remain frozen, and
the samples below are nested prefixes, not independent replications at each
size. Bootstrap replicates number 500, with seed 94002 plus the path count.

| Paths | Daily ES | Weekly ES | Adaptive .01 ES | Adaptive .05 ES | .05 95% interval |
| --- | --- | --- | --- | --- | --- |
| 2,000 | 2.7053 | 3.1823 | 2.6527 | 2.5019 | [2.3927, 2.6276] |
| 5,000 | 2.6973 | 3.1373 | 2.6436 | 2.4968 | [2.4230, 2.5652] |
| 20,000 | 2.7335 | 3.1420 | 2.6820 | 2.4993 | [2.4675, 2.5336] |
| 50,000 | 2.7291 | 3.1438 | 2.6791 | 2.5257 | [2.5046, 2.5452] |

The 50,000-path adaptive 0.05 result supports baseline budget-3 feasibility
within the model. Nearby threshold rankings remain uncertain: the fresh
50,000-path ES for adaptive 0.03 is 2.516631, compared with 2.525674 for 0.05.
The new sample is not used to replace the calibration selection.

At half-spread 0.001 and budget 2, the original independent evaluation of frozen
adaptive 0.05 gives ES 1.974831 and interval [1.947923, 2.004507]. Point feasibility
holds, but the upper-endpoint diagnostic does not. On 50,000 fresh paths the
estimate is 1.969963 with interval [1.948600, 1.990360] (1,000 resamples,
seed 94004), strengthening evidence for this setting under the model. This
does not retroactively remove uncertainty from the original sample.

With only 500 bootstrap draws, extreme percentile endpoints have additional
simulation noise. A secondary resampling diagnostic (3,000 repetitions,
seed 94006) yields the following ranges for the reported upper endpoints;
these ranges concern bootstrap numerical precision, not new population-ES
confidence intervals.

| Half-spread | Reported upper endpoint | Endpoint simulation range |
| --- | --- | --- |
| 0.001 | 2.0045 | [2.0005, 2.0071] |
| 0.002 | 2.5692 | [2.5650, 2.5736] |

![Expected Shortfall precision for fixed policies on nested fresh samples.](../results/validation/figures/path_count_convergence.png)

# Sensitivity and robustness

## Frozen-policy volatility stress

True volatility increases from 20% to 25% and 30%, while the model Delta and
premium remain at 20%. Scenarios share Gaussian shocks, enabling paired
comparisons, and remain independent of calibration. All figures use the
original frozen policies; there is no stress-specific optimisation.

| True vol. | Policy | Mean net P&L | SD | Mean cost | ES95 |
| --- | --- | --- | --- | --- | --- |
| 0.20 | Daily | -1.2556 | 0.6038 | 1.2520 | 2.7653 |
| 0.20 | Weekly | -0.6918 | 0.9966 | 0.6955 | 3.1004 |
| 0.20 | Adaptive 0.01 | -1.1937 | 0.6129 | 1.1884 | 2.7168 |
| 0.20 | Adaptive 0.05 | -0.8119 | 0.7544 | 0.7981 | 2.5372 |
| 0.25 | Daily | -3.3663 | 1.4223 | 1.3640 | 6.5915 |
| 0.25 | Weekly | -2.7349 | 1.5936 | 0.7430 | 6.8760 |
| 0.25 | Adaptive 0.01 | -3.3123 | 1.4269 | 1.3075 | 6.5514 |
| 0.25 | Adaptive 0.05 | -2.9316 | 1.4115 | 0.9211 | 6.1792 |
| 0.30 | Daily | -5.4548 | 2.4515 | 1.4534 | 10.8036 |
| 0.30 | Weekly | -4.7675 | 2.5040 | 0.7815 | 11.0666 |
| 0.30 | Adaptive 0.01 | -5.4057 | 2.4517 | 1.4020 | 10.7632 |
| 0.30 | Adaptive 0.05 | -5.0318 | 2.3720 | 1.0291 | 10.3198 |

Adaptive 0.05 ES rises to 6.179234 at 25% true volatility, interval
[6.119002, 6.235605], and to 10.319762 at 30%, interval [10.233634, 10.403395].
It fails its original budget 3 by a large margin. Weekly also deteriorates;
fixed trading dates do not remove the model-volatility dependence of holdings.

![Frozen-policy Expected Shortfall under volatility misspecification.](../results/figures/expected_shortfall_comparison.png)

Paired changes in relative ES gaps are listed below. The common premium shift
cancels in pairwise ES gaps. The contrasts establish policy/model interactions
within these scenarios, but do not isolate a pure causal effect of trigger
timing, since holdings and trading rules also differ.

| First minus second | 20% ES gap | 25% ES gap | Gap change | Change 95% interval |
| --- | --- | --- | --- | --- |
| Daily minus Adaptive 0.01 | 0.0485 | 0.0402 | -0.0083 | [-0.0122, -0.0033] |
| Weekly minus Adaptive 0.01 | 0.3835 | 0.3246 | -0.0589 | [-0.1106, -0.0066] |
| Daily minus Adaptive 0.05 | 0.2281 | 0.4123 | 0.1842 | [0.1518, 0.2193] |
| Weekly minus Adaptive 0.05 | 0.5632 | 0.6968 | 0.1336 | [0.0740, 0.1908] |
| Adaptive 0.03 minus Adaptive 0.05 | 0.0031 | 0.1700 | 0.1669 | [0.1306, 0.2043] |

## Separating funding and hedge-volatility mismatch

At true volatility 25%, the 20% premium is 7.965567 versus a fair scenario
Black-Scholes premium of 9.947645. Adding the difference 1.982078 to initial
funding translates terminal P&L and ES at zero rate, leaving dispersion and
trading unchanged. For adaptive 0.05 this reduces ES from 6.179234 to 4.197157,
which still exceeds budget 3. Underpricing alone therefore does not account
for the full stress failure.

A controlled diagnostic uses identical 25% paths (seed 202612) and the same
fair premium, changing only the hedge-model volatility from 20% to 25%.
The policy forms and thresholds remain fixed.

| Policy | Hedge vol. | Fair-funded mean | Gross RMSE | Net ES95 |
| --- | --- | --- | --- | --- |
| Daily | 0.20 | -1.3842 | 1.0073 | 4.6095 |
| Daily | 0.25 | -1.2563 | 0.5548 | 3.0131 |
| Weekly | 0.20 | -0.7529 | 1.4498 | 4.8939 |
| Weekly | 0.25 | -0.6891 | 1.1898 | 3.6675 |
| Adaptive 0.01 | 0.20 | -1.3303 | 1.0087 | 4.5693 |
| Adaptive 0.01 | 0.25 | -1.1949 | 0.5632 | 2.9672 |
| Adaptive 0.05 | 0.20 | -0.9496 | 1.1050 | 4.1972 |
| Adaptive 0.05 | 0.25 | -0.8114 | 0.8350 | 2.9239 |

For adaptive 0.05, correct-volatility hedging yields ES 2.923907, with interval
[2.879727, 2.968480]. The paired wrong-minus-correct ES difference is 1.273250,
interval [1.223002, 1.320652] (500 resamples, seed 94005). This is a conditional
hedge-model diagnostic, not a new optimum or an assumption that future
volatility is observable. Fair funding plus correct model Delta answers a
different question from the principal fixed-premium stress.

# Discussion and limitations

The results support expected-cost selection under an explicit terminal ES
constraint within a specified synthetic model. They do not establish a
universal advantage for adaptive hedging. Weekly can be cheapest and feasible
at lower spreads; daily improves gross replication at high frequency; wider
adaptive bands can save fees while respecting the baseline budget.

The following qualifications are material to interpretation:

- **Finite candidate set:** threshold 0.05 is a grid endpoint. No continuous
  optimisation or general optimal-control result is established.
- **Model conditionality:** constant-volatility GBM, fixed proportional
  half-spreads, fractional shares and unrestricted symmetric financing omit
  jumps, stochastic volatility, dividends, market impact, margin and liquidity
  effects. There is no empirical market calibration or out-of-market test.
- **Stress failure:** baseline feasibility does not transfer to volatility
  misspecification. The selected fixed policy fails the original ES budget.
- **Statistical uncertainty:** ES uses only the worst 5% of paths. Bootstrap
  intervals are pointwise, not simultaneous grid certificates, and calibration
  intervals do not remove selection effects. The independent evaluation and
  fresh fixed-policy sample reduce tuning optimism but do not prove market
  performance. Nested sample sizes are correlated.
- **Design provenance:** code ordering verifies calibration/evaluation
  separation, but mutable configuration files and the absence of an immutable
  version history cannot independently establish preregistration of earlier
  grids and budgets. No claim of tuning is inferred from that limitation.
- **Numerical domain:** extreme finite rates or underflowing volatility/time
  combinations can produce infinity or NaN despite passing input finiteness
  checks. The cost-rescaling helper trusts the supplied reference spread.
  These limitations are recorded in the supplementary diagnostics; the
  configured benchmark is unaffected. The implementation is not certified
  for arbitrary extreme inputs or production use.

## Conclusion

For the tested baseline, adaptive threshold 0.05 is the lowest estimated-cost
calibration-feasible member of seven policies under budget 3. Independent
evaluation and a fresh 50,000-path calculation support feasibility under the
assumed model. Higher true volatility breaks the same risk limit. Cost rate,
premium convention, model volatility, candidate grid and uncertainty all
matter to the decision; statistical evidence within GBM does not establish
real-market performance.

The financial framework, relation to the literature and full references are
in [methodology.md](methodology.md). This study originated as academic
coursework at Zhejiang University and was subsequently reimplemented and
extended.
