# Methodology

## Introduction and research question

**How can a derivatives desk select a delta-hedging strategy that minimises
expected transaction costs while satisfying a predefined tail-risk constraint?**

This academic simulation study considers a hypothetical bank selling one
European call. It compares seven fixed policies within a constant-volatility
model. The economic motivation is derivatives risk management; this is not
a real bank engagement or a production trading system.

The finite-policy optimisation is

$$\min_{\pi\in\mathcal A}\mathbb E[TC_T^{\mathrm{fv}}(\pi)]
\quad\mathrm{subject\ to}\quad ES_{0.95}(-\Pi_T^{\mathrm{net}}(\pi))\le R_{\max}.$$

The set $\mathcal A$ contains daily, weekly and five adaptive policies. Exhaustive
comparison establishes an optimum within that set only. Selection uses
calibration data; evaluation and volatility stress retain the frozen decision.

## The banking problem

A client buys one European call with strike $K$ and maturity $T$. The bank
**sells** it, receives a premium $P_0$, and owes the client

$$H(S_T)=(S_T-K)^+=\max(S_T-K,0).$$

The option is a liability for the bank. To meet it, the bank buys a positive
stock position $\phi_t$ and holds cash $B_t$, which may be negative (borrowing).
The hedge portfolio, excluding the short option, is

$$V_t=B_t+\phi_tS_t.$$

Before maturity the whole desk book is $V_t-C(t,S_t)$. At maturity its realised
profit is $\Pi_T=V_T-H(S_T)$, with $V_T$ measured after stock liquidation and its
fee. The payoff, portfolio value and profit are different quantities. The
premium finances the hedge and must not be added again to terminal profit.

For example, take $S_0=K=100$, $T=1$, $r=0$, and volatility 20%. The
Black–Scholes premium is about 7.96557 and the initial Delta about 0.53983.
Without fees the bank buys 53.98278 of stock and borrows about 46.01722.
At $S_T=110$, the client receives 10. A hedge worth 9.70 after liquidation
produces bank P&L of $-0.30$; a hedge worth 10.20 produces $+0.20$.
These two hedge values are illustrative, not simulation results. At $S_T=90$
the payoff is zero, but the bank's trading and financing can still make a loss.

Continuous, costless Delta trading in the correctly specified Black–Scholes
model replicates the payoff. Discrete trading leaves the holding unchanged
while Delta moves. Transaction costs consume portfolio value each time stock
is traded, including opening and closing. Neither exposure disappears just
because the terminal payoff is known to be a call payoff.

## Stock scenarios: geometric Brownian motion

Under the physical probability measure $\mathbb P$,

$$dS_t=\mu S_t\,dt+\sigma_{\rm true}S_t\,dW_t^{\mathbb P}.$$

The drift $\mu$ is the instantaneous expected proportional return. Volatility
$\sigma_{\rm true}$ controls random return dispersion. Brownian motion has
independent increments $W_{t+h}-W_t\sim N(0,h)$ and continuous paths. A unit
Gaussian shock $Z\sim N(0,1)$ represents a standardised new increment.

Apply Itô's formula to $\log S_t$, using $(dW_t)^2=dt$:

$$d\log S_t=\left(\mu-\tfrac12\sigma_{\rm true}^2\right)dt
+\sigma_{\rm true}dW_t^{\mathbb P}.$$

Integrating between two observations and exponentiating gives

$$S_{t+h}=S_t\exp\!\left[
\left(\mu-\tfrac12\sigma_{\rm true}^2\right)h
+\sigma_{\rm true}\sqrt h\,Z\right].$$

This is exact sampling of GBM on the observation grid. It preserves positive
prices and avoids the time-discretisation bias and possible negative prices
of Euler's $S_{t+h}=S_t(1+\mu h+\sigma\sqrt h Z)$. The hedge still has
discretisation error because trading is discrete. Exact path sampling does
not remove that error.

GBM is a tractable baseline with known distributions and an analytical option
formula. It assumes continuous prices, constant volatility and drift, and
independent Gaussian log-return increments. It omits jumps, volatility
clustering, leverage effects and changing liquidity. Constant volatility is
an assumption, not evidence that model volatility equals future realised
volatility.

`simulate_gbm_paths` uses local `numpy.random.default_rng(seed)` and returns
`(n_paths, n_steps + 1)`, including $S_0$. Optional supplied shocks allow
explicit coupling. Generate one matrix per scenario and pass the same matrix
to all strategies. A trading decision reads only today's price, remaining
time and existing holding.

## Physical scenarios versus risk-neutral pricing

The physical measure $\mathbb P$ describes assumed real-world outcomes. The
baseline uses $\mu=0.05$ for hedging scenarios. The risk-neutral measure
$\mathbb Q$ is a valuation device: with no dividends, discounted stock is a
martingale, so its drift is $r$, and

$$C(0,S_0)=e^{-rT}\mathbb E^{\mathbb Q}[H(S_T)].$$

Using $\mu$ instead of $r$ in pricing would generally violate the replication
price. The bank's risk scenarios need not have risk-neutral probabilities.
Correct continuous replication holds pathwise in this model even if
$\mu\ne r$; that drift difference alone does not guarantee a positive hedge
profit. Discrete hedging may have a small signed bias under $\mathbb P$.

Keep two volatility inputs distinct:

| Input | Use |
|---|---|
| $\sigma_{\rm true}$ | Generates stock scenarios |
| $\sigma_{\rm model}$ | Sets the sold-call premium and every target Delta |

Both equal 0.20 in the baseline. The engine can accept different values;
volatility stress experiments are deferred. `risk_neutral_call_price_mc`
is a separate pricing check with drift $r$, discounted payoffs and an
independent seed. It does not supply the physical hedging paths.

## Black–Scholes valuation, Delta and Gamma

For remaining time $\tau=T-t>0$, $S,K>0$, $\sigma_{\rm model}>0$, define

$$d_1=\frac{\log(S/K)+(r+\tfrac12\sigma_{\rm model}^2)\tau}
{\sigma_{\rm model}\sqrt\tau},\qquad
d_2=d_1-\sigma_{\rm model}\sqrt\tau.$$

With $\Phi$ the standard normal CDF and $\varphi$ its density,

$$C(t,S)=S\Phi(d_1)-Ke^{-r\tau}\Phi(d_2),$$
$$\Delta(t,S)=\frac{\partial C}{\partial S}=\Phi(d_1),\qquad
\Gamma(t,S)=\frac{\partial^2C}{\partial S^2}
=\frac{\varphi(d_1)}{S\sigma_{\rm model}\sqrt\tau}.$$

Delta is the number of shares needed against **one short call**: a local stock
move $dS$ changes the call liability by approximately $\Delta dS$, offset by
the stock asset. A finite move also contributes about
$\tfrac12\Gamma(dS)^2$. High Gamma means Delta changes quickly, so a fixed
holding becomes stale faster between trades. For a short call, unhedged
convexity is a source of losses on sufficiently large moves, alongside the
time-decay and financing terms in the option price evolution.

Delta increases with spot. More volatility or more remaining time generally
pulls very out-of-the-money and very in-the-money Deltas away from their
near-expiry values of 0 and 1, respectively, but Delta is not globally
monotone in either volatility or time. For example, at $S=K,r=0$ it equals
$\Phi(\sigma\sqrt\tau/2)>1/2$ and rises with $\sigma\sqrt\tau$. Near maturity,
Gamma becomes concentrated around the strike; away from the strike it tends
to zero. Uniformly increasing Gamma at every spot is not a correct description.

### Limits and numerical contract

`black_scholes.py` broadcasts scalar and array inputs and returns a float for
scalar output. It rejects non-finite inputs, $S<0$, $K\le0$, $\sigma<0$ or
$\tau<0$. Zero stock is an explicit absorbing boundary, although GBM scenarios
start strictly positive.

* At $\tau=0$, price is payoff. Delta is 0 below strike, 1 above strike and
  **0.5 at the kink**, a convention rather than a unique derivative.
* Gamma at $\tau=0$ is returned as zero as a reporting convention. The payoff
  is not twice differentiable at strike; its distributional curvature is not
  the returned zero. Never use this value to approximate pre-expiry Gamma.
* At $\sigma=0$, price is $(S-Ke^{-r\tau})^+$. Delta is the corresponding
  step function, with 0.5 at equality. Gamma is reported as zero, with the
  same kink qualification.
* At $S=0$, price, Delta and Gamma are zero (right boundary convention).
* Positive but very short $\tau$ is evaluated as supplied. No artificial floor
  substitutes for maturity. Logarithms and divisions are restricted to the
  strictly positive, non-degenerate subset to avoid spurious warnings.

The hedger never asks for a new Delta at maturity. The optional history uses
`NaN` for terminal target Delta to make this absence explicit.

## Self-financing accounting

### Opening the hedge

Use the model premium $P_0=C(0,S_0;\sigma_{\rm model})$ and
$\phi_0=\Delta(0,S_0;\sigma_{\rm model})$. With proportional half-spread
$\kappa$ applied to absolute traded notional,

$$TC_0=\kappa|\phi_0|S_0,\qquad
B_0=P_0-\phi_0S_0-TC_0.$$

Therefore $V_0=P_0-TC_0$ and the initial whole-book mark is $-TC_0$.
The model premium includes no loading to pay for future hedging expenses.
Fractional shares, unlimited borrowing/lending at the same $r$, immediate
execution, and no dividends, stock-financing spread, margin or other cash
flows are assumed. Negative cash accrues interest as a borrowing balance.

### An interior observation and trade

At $t_i$, accrue cash first, holding the previous stock position fixed:

$$B_i^-=B_{i-1}e^{r(t_i-t_{i-1})},\qquad
V_i^-=B_i^-+\phi_{i-1}S_i.$$

Select $\phi_i$ using only available information and write
$q_i=\phi_i-\phi_{i-1}$. Buying $q_i>0$ consumes cash; selling $q_i<0$
raises cash. Either direction pays a positive fee:

$$TC_i=\kappa|q_i|S_i,$$
$$\boxed{B_i=B_{i-1}e^{r\Delta t}-q_iS_i-TC_i.}$$

Adding the post-trade stock position proves the execution identity:

$$V_i=B_i+\phi_iS_i
=B_i^-+\phi_{i-1}S_i-TC_i=V_i^--TC_i.$$

A trade at the current mid-price transfers value between cash and stock;
only its fee destroys total value. If there is no trade, cash only accrues
interest, and $TC_i=0$. Stock mark-to-market must **not** also be credited to
cash: the change $\phi_{i-1}(S_i-S_{i-1})$ is already in the stock's value.
The discrete gain identity is

$$V_i-V_{i-1}=B_{i-1}(e^{r\Delta t}-1)
+\phi_{i-1}(S_i-S_{i-1})-TC_i.$$

This is the correction to the historical engine.

In the frictionless continuous limit, $dV=rB\,dt+\phi\,dS$.
Itô's formula and the Black–Scholes PDE
$C_t+rSC_S+\tfrac12\sigma^2S^2C_{SS}-rC=0$
give $dC=r(C-S\Delta)dt+\Delta dS$ when true and model volatility agree.
Thus $V=C$, $\phi=\Delta$ is self-financing and has terminal value $H$.
The $\mu$ terms cancel in this replication argument. With positive spreads,
continuous rebalancing is not the frictionless limit to implement: its
turnover can become unbounded.

### Maturity: close stock once and pay once

Interior decisions are at $i=1,\ldots,N-1$. At $T=t_N$ accrue cash, execute
only the stock liquidation $q_N=-\phi_{N-1}$, and charge its spread:

$$TC_N=\kappa|\phi_{N-1}|S_N,$$
$$B_N=B_{N-1}e^{r\Delta t}+\phi_{N-1}S_N-TC_N,\qquad \phi_N=0,$$
$$\boxed{\Pi_T^{\rm net}=B_N-H(S_N)=V_T-H(S_T).}$$

The contract is cash-settled, including if the call is in the money. History
at $T$ records cash and portfolio value **after stock liquidation but before
option settlement**. Payoff and post-settlement cash (net P&L) are separate
outputs. There is no Delta-based maturity trade followed by another close.
The close fee is already deducted from $B_N$ and is not subtracted again.

### Gross P&L and the cost reconciliation

A shadow cash account follows the **same trades** with no fees. Its initial
cash is $P_0-\phi_0S_0$. Define

$$\Pi_T^{\rm gross}=B_N^{\rm gross}-H(S_N),\qquad
TC_T^{\rm fv}=\sum_{i=0}^{N}TC_i e^{r(T-t_i)}.$$

Cash-account differences accrue at $r$, so

$$\boxed{\Pi_T^{\rm net}=\Pi_T^{\rm gross}-TC_T^{\rm fv}.}$$

Both raw total fees $\sum TC_i$ and their terminal value are exposed. They
agree at the baseline $r=0$. Gross is a fee-free shadow with identical trade
decisions, not net P&L plus unaccrued fees when $r\ne0$.

### Manual check independent of a Delta rule

For an accounting illustration choose a premium of 8, $r=0$, $\kappa=0.002$,
prices $100\to110\to105$, and holdings $0.50\to0.75\to0$. The premium
and trades here are deliberately specified by hand, not Black–Scholes outputs.

| Event | Stock | Trade | Fee | Cash after trade | Shares after trade | Portfolio |
|---|---:|---:|---:|---:|---:|---:|
| Open | 100 | +0.50 | 0.1000 | -42.1000 | 0.50 | 7.9000 |
| Interior | 110 | +0.25 | 0.0550 | -69.6550 | 0.75 | 12.8450 |
| Liquidate | 105 | -0.75 | 0.1575 | 8.9375 | 0 | 8.9375 |

Before the interior trade the value is $-42.1+0.5(110)=12.9$; afterwards it is
$12.9-0.055=12.845$. At strike 100 the payoff is 5, giving net P&L 3.9375.
The fee-free cash ends at 9.25, so gross P&L is 4.25. The difference is
$0.1000+0.0550+0.1575=0.3125$, exactly the fees. A no-trade interval leaves
cash unchanged at zero rate, regardless of the stock movement.

## Trading policies, counts and metrics

All policies open at model Delta and close all shares at $T$. On the grid
$t_i=iT/N$, strictly before maturity:

* Daily: set holdings to target Delta at every interior observation.
* Weekly: do so when $i\bmod m=0$, with baseline $m=5$ trading steps.
  This is a grid-based approximation to weeks, not a holiday calendar.
* Adaptive: trade to target only if $|\Delta_i-\phi_{i-1}|>\varepsilon$.
  Compare against the last actual holding, not yesterday's target; equality
  does not trigger a trade.

Executed trades require a nonzero share change. No minimum trade-size
tolerance is imposed; an exactly unchanged floating-point Delta is no trade.
Report opening trades, interior rebalances and closing trades separately;
their sum is total executed trades. Daily has 251 eligible interior dates
at $N=252$, weekly 50. Executed counts can be lower if target is exactly
unchanged, including when the normal CDF saturates at 0 or 1. A zero holding
at maturity needs no executed closing trade. Traded notional includes all
three stages: $\sum_i|q_i|S_i$, distinct from trade count.

For $M$ path P&Ls $x_j$ report mean, sample standard deviation
($ddof=1$), MAE $M^{-1}\sum|x_j|$, and RMSE
$\sqrt{M^{-1}\sum x_j^2}$. Report each for gross and net. Negative net P&L
is a bank loss; low signed mean alone does not establish accuracy. Report
mean fees, terminal-valued fees, notional and interior/total counts. The
standard error of an estimated mean is sample standard deviation divided
by $\sqrt M$; it is distinct from the P&L standard deviation.

## Validation gate and scope

Analytical benchmarks, finite-difference Greeks, exact GBM moments,
risk-neutral Monte Carlo uncertainty, deterministic cash ledgers, terminal
settlement and gross/net reconciliation must pass before interpreting
results. A coupled zero-fee frequency experiment should show an overall
reduction of replication RMSE as the mesh shrinks, allowing sampling
uncertainty rather than demanding strict monotonicity in every sample.


## Empirical VaR and ES, including partial observations

For sorted losses $l_{(1)}\le\cdots\le l_{(n)}$ with equal weights,

$$\widehat{VaR}_{\alpha}=l_{(\lceil n\alpha\rceil)},\qquad
\widehat{ES}_{\alpha}=\frac{1}{1-\alpha}
\int_{\alpha}^{1}\widehat{VaR}_{u}\,du.$$

This uses the empirical inverse CDF, without linear interpolation between
VaR observations. To calculate ES, let $h=n(1-\alpha)$,
$m=\lfloor h\rfloor$ and $f=h-m$. Then

$$\widehat{ES}_{\alpha}=
\frac{\sum_{j=0}^{m-1}l_{(n-j)}+f\,l_{(n-m)}}{h}.$$

The fractional term is omitted if $f=0$. At 20,000 paths and 95% confidence,
the worst 1,000 observations each receive equal tail weight. For losses
$[1,2,3,4]$ and $\alpha=0.625$, $VaR=3$ and
$ES=(4+0.5\times3)/1.5=11/3$. At atoms, taking all observations $\ge VaR$
can include too much probability and differs from this definition. If every
outcome is profitable, VaR/ES can be negative; neither is floored at zero.

Implementation lives in `src/metrics.py`, with the same tail-mean routine
used by bootstrap resamples. Tests include known losses, ties, negative
losses, fractional tails, translation and positive scaling.

## Simulation design

The configuration is [configs/study.json](../configs/study.json). It records all grids, limits, random seeds and the supplementary stress. Defaults retain $S_0=K=100$, $T=1$, $r=0$,
$\mu=0.05$, $N=252$, and $\sigma_{\rm model}=0.20$.

| Choice | Convention |
|---|---|
| Policies | Daily, weekly every 5 grid steps; adaptive $\varepsilon\in\{0.005,0.01,0.02,0.03,0.05\}$ |
| Grid extension | Configurable; adaptive 0.01 always retained as historical benchmark |
| Cost cases | $\kappa\in\{0,0.0001,0.0005,0.001,0.002,0.005\}$ |
| Historical spread case | $\kappa=0.002$, a half-spread on traded notional |
| Risk limits | Fixed 1, 2 and 3 currency units per option; primary case 3 |
| Primary eligibility | Calibration point ES estimate $\le$ budget |
| Uncertainty diagnostic | Also report whether the upper endpoint of the pointwise two-sided 95% ES interval meets the budget |
| Cost tie break | Lower ES, then lexical policy ID; especially relevant when all fees are zero |
| Calibration | 20,000 physical paths; seed 202611 |
| Independent evaluation | 20,000 physical paths; seed 202612 |
| Bootstrap | 500 paired path resamples; seeds 731 calibration, 732 evaluation, 733 stress |
| True-volatility cases | 20%, original underestimated 25%, supplementary 30% |
| Principal premium | Model-volatility BS premium, fixed across true-volatility cases |

Costs are experimental spread scenarios, not calibrated universal market
quotes. Thresholds are candidate choices, not a theoretically optimal grid.
Budgets are hypothetical monetary limits selected without looking at final
evaluation results; they are not regulatory capital requirements.

The Pareto frontier retains a policy unless another has no higher mean cost
and no higher ES, with at least one strict improvement. Equal points both
remain. The frontier uses point estimates and can move with sampling noise.
At zero fees, all expected-cost objectives tie, so the cost optimum need
not be unique even though the reporting tie break selects one policy.

## Common paths, independent evaluation and uncertainty

Within a sample, each strategy and fee case sees identical stock prices.
Calibration and evaluation use independent seeds/shocks. Calibration
summaries and the frozen choices are saved **before evaluation paths are
generated**. Evaluation compares the predefined policies descriptively but
does not change any selected threshold, budget, feasibility rule or candidate
grid. Stress loads that frozen artifact; it never calls policy selection.

For these cash-independent policies, proportional fees scale linearly with
$\kappa$ on a fixed trade sequence, including their terminal value. The engine
runs once per policy at the reference rate; sensitivity uses this exact
identity. Tests compare the transformation with full ledgers at multiple
costs and positive/zero/negative rates. Stock decisions, turnover, counts
and gross P&L remain unchanged across fee cases.

Bootstrap resampling draws iid **path rows**, jointly across all candidate
and fee columns. Each replicate recomputes ES using the empirical definition.
Pointwise percentile bounds and bootstrap standard errors are saved.
Strategy contrasts subtract the two estimates within each paired replicate;
mean P&L contrasts additionally have paired normal-CLT intervals. Bootstrap
and path seeds are distinct and recorded.

The full 20,000-path sample has only 1,000 observations in the 5% tail. Its
size alone is no precision guarantee. Interpret interval width relative to
headroom, and distinguish point-estimate feasibility from the stricter upper
endpoint diagnostic. Bootstrap intervals are pointwise, not simultaneous
across a searched grid, and calibration intervals do not remove selection
bias. Frozen independent evaluation is the main protection against tuning
optimism. No evaluation winner is substituted after seeing results. If
headroom is comparable with uncertainty, the recommendation must remain
conditional and be assessed on a separately specified larger validation sample.

## Model risk and fixed-policy stress

Physical scenarios change $\sigma_{\rm true}$ from 20% to 25%, with a
specified supplementary 30%. Model Delta and sold-call premium remain at
20%. The same evaluation seed/dimensions give identical Gaussian shocks
across volatilities, enabling paired stress contrasts; these shocks remain
independent of calibration. The 20% stress baseline is checked against the
saved independent evaluation path hash and point metrics.

The main question is whether a **frozen baseline choice** still satisfies
its original budget. Reporting all policies under stress does not amount
to a stress-specific optimisation. No new stress optimum is selected.
Weekly dates are fixed but stock holdings still depend on model Delta, so
weekly has model risk too.

At 25% true volatility the fixed 20% premium also represents underpricing
relative to that scenario's Black–Scholes value. Thus P&L deterioration
combines premium/model risk and residual hedging risk; it cannot all be
attributed to trigger timing. A supplementary fair-premium diagnostic
keeps the same trades and model Delta, changing only initial funding:

$$\Pi_T^{\rm fair}=\Pi_T^{\rm fixed}
+[C_{BS}(\sigma_{\rm true})-C_{BS}(\sigma_{\rm model})]e^{rT}.$$

The deterministic funding shift changes mean P&L and translates ES, while
P&L standard deviation and execution costs stay unchanged. It does not
assume the desk knows future volatility, change principal results, or make
the mis-specified Delta correct.


## Relation to the literature and references

Leland motivates explicit proportional costs in discrete replication. The
present study uses unadjusted Black-Scholes pricing and Delta, rather than
Leland's effective-volatility adjustment. Sepp studies frequency, costs and
discrete hedging; his Sharpe-ratio objective differs from the terminal ES budget
used here. Whalley and Wilmott motivate no-transaction regions; the simple
adaptive rule here returns to Delta rather than trading to the edge of their
optimal band. Pochart and Bouchaud consider minimum *local* expected shortfall,
whereas this study constrains terminal ES across a finite candidate set.
Acerbi and Tasche motivate the quantile-integral ES definition at atoms.
These references do not prescribe the numerical defaults or prove the
optimality of the candidate policies.

- Leland, H. E. (1985). *Option Pricing and Replication with Transactions Costs*.
  Journal of Finance, 40(5), 1283-1301.
  [DOI](https://doi.org/10.1111/j.1540-6261.1985.tb02383.x).
- Sepp, A. (2013). *When You Hedge Discretely: Optimization of Sharpe Ratio for
  Delta-Hedging Strategy under Discrete Hedging and Transaction Costs*.
  Journal of Investment Strategies, 3(1), 19-59.
  [Author manuscript record](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1865998).
- Whalley, A. E., and Wilmott, P. (1999). *Optimal Hedging of Options with Small
  but Arbitrary Transaction Cost Structure*. Oxford Financial Research Centre,
  working paper 1999-MF-09.
  [Manuscript](https://users.ox.ac.uk/~ofrcinfo/file_links/mf_papers/1999mf09.pdf).
- Pochart, B., and Bouchaud, J.-P. (2004). *Option pricing and hedging with
  minimum local expected shortfall*. Quantitative Finance, 4(5), 607-618.
  [2003 preprint](https://arxiv.org/abs/cond-mat/0308570);
  [published article](https://doi.org/10.1080/14697680400023329).
- Acerbi, C., and Tasche, D. (2002). *On the coherence of Expected Shortfall*.
  Journal of Banking & Finance, 26, 1487-1503.
  [Primary preprint](https://arxiv.org/abs/cond-mat/0104295).

Machine-readable bibliographic entries are in [references.bib](references.bib).
