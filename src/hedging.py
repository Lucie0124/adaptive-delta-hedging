"""Self-financing stock/cash hedge for a bank short one cash-settled call.

Positive holdings are long shares. Premium is received once. Histories are
post-stock-trade marks, before option settlement; terminal holdings are zero.
There is no Delta target or rebalance at maturity, only mandatory liquidation.
"""

from typing import Literal, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .black_scholes import bs_call_price, bs_call_delta, call_payoff

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Strategy = Literal["daily", "weekly", "adaptive"]


class HedgingResult(TypedDict):
    strategy: Strategy
    stock_paths: FloatArray
    times: FloatArray
    premium: FloatArray
    terminal_payoff: FloatArray
    terminal_portfolio_net: FloatArray
    terminal_portfolio_gross: FloatArray
    pnl_net: FloatArray
    pnl_gross: FloatArray
    total_transaction_cost: FloatArray
    transaction_cost_fv: FloatArray
    total_traded_notional: FloatArray
    n_initial: IntArray
    n_rebalances: IntArray
    n_liquidation: IntArray
    n_trades: IntArray
    history: dict[str, NDArray] | None


def update_cash_account(
    cash: ArrayLike, old_holding: ArrayLike, new_holding: ArrayLike,
    stock_price: ArrayLike, *, r: float = 0.0, dt: float = 0.0,
    kappa: float = 0.002,
) -> tuple[FloatArray, FloatArray]:
    """Accrue cash, fund a stock trade, and charge its fee exactly once.

    Returns (post-trade cash, fee), broadcasting inputs. This function does
    not receive an old price: stock mark-to-market cannot enter cash here.
    Use dt=0 for opening, new_holding=0 for liquidation, and identical holdings
    for a no-trade interval. Negative cash accrues at the same rate as lending.
    """
    arrays = np.broadcast_arrays(*[np.asarray(x, dtype=float) for x in
                                   (cash, old_holding, new_holding, stock_price)])
    if any(np.any(~np.isfinite(x)) for x in arrays):
        raise ValueError("Cash, holdings and stock price must be finite.")
    if not np.all(np.isfinite([r, dt, kappa])) or dt < 0 or kappa < 0:
        raise ValueError("r must be finite; dt and kappa finite and nonnegative.")
    b, old, new, s = arrays
    if np.any(s <= 0):
        raise ValueError("Execution stock prices must be positive.")
    q = new - old
    fee = kappa * np.abs(q) * s
    return np.asarray(b*np.exp(r*dt)-q*s-fee), np.asarray(fee)


def rebalance_mask(
    strategy: Strategy, step: int, target_delta: ArrayLike, holding: ArrayLike,
    *, epsilon: float = 0.01, weekly_interval: int = 5,
) -> NDArray[np.bool_]:
    """Interior-date eligibility; the caller must exclude opening and maturity.

    Adaptive uses strict > epsilon relative to the current actual holding.
    A True mask with an unchanged target still executes no trade.
    """
    if strategy not in ("daily", "weekly", "adaptive"):
        raise ValueError("strategy must be 'daily', 'weekly' or 'adaptive'.")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative.")
    if (isinstance(weekly_interval, (bool, np.bool_))
            or not isinstance(weekly_interval, (int, np.integer)) or weekly_interval < 1):
        raise ValueError("weekly_interval must be a positive integer.")
    if (isinstance(step, (bool, np.bool_))
            or not isinstance(step, (int, np.integer)) or step < 1):
        raise ValueError("step must be a positive interior grid index.")
    target, actual = np.broadcast_arrays(np.asarray(target_delta, dtype=float),
                                         np.asarray(holding, dtype=float))
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(actual)):
        raise ValueError("target_delta and holding must be finite.")
    if strategy == "adaptive":
        return np.abs(target-actual) > epsilon
    eligible = strategy == "daily" or step % weekly_interval == 0
    return np.full(target.shape, eligible, dtype=bool)


def hedge_paths(
    stock_paths: ArrayLike, strategy: Strategy, *, strike: float = 100.0,
    maturity: float = 1.0, r: float = 0.0, sigma_model: float = 0.20,
    kappa: float = 0.002, epsilon: float = 0.01, weekly_interval: int = 5,
    return_history: bool = True,
) -> HedgingResult:
    """Evaluate a strategy on supplied paths without mutating or simulating them.

    Paths have shape (n_paths, n_steps+1) on a uniform grid. Initial premium
    and hedge use sigma_model. Fees include opening, interior trades and final
    liquidation. Gross is a parallel fee-free ledger with identical holdings.
    Set return_history=False to keep memory O(n_paths) beyond the shared paths;
    the same accounting is run in both modes. Counts are executed nonzero
    trades, with no minimum-size tolerance. Terminal history is before payoff;
    terminal_portfolio_net - terminal_payoff is post-settlement cash/P&L.
    """
    paths = np.asarray(stock_paths, dtype=float)
    if (paths.ndim != 2 or paths.shape[0] < 1 or paths.shape[1] < 2
            or not np.all(np.isfinite(paths)) or np.any(paths <= 0)):
        raise ValueError("stock_paths must be finite positive shape (n_paths, n_steps+1).")
    if not np.all(np.isfinite([strike, maturity, r, sigma_model, kappa])):
        raise ValueError("Hedging parameters must be finite.")
    if maturity <= 0 or strike <= 0 or sigma_model < 0 or kappa < 0:
        raise ValueError("Need maturity/strike > 0 and sigma_model/kappa >= 0.")
    # Validate policy parameters even if there are no interior dates.
    rebalance_mask(strategy, 1, 0.0, 0.0, epsilon=epsilon, weekly_interval=weekly_interval)
    n_paths, columns = paths.shape
    n_steps = columns-1
    dt = maturity / n_steps
    times = np.linspace(0, maturity, columns)
    premium = np.asarray(bs_call_price(paths[:, 0], strike, r, sigma_model, maturity))
    cash_net, cash_gross = premium.copy(), premium.copy()
    holding = np.zeros(n_paths)
    costs, cost_fv, notional = (np.zeros(n_paths) for _ in range(3))
    initial, rebalances, liquidation = (np.zeros(n_paths, dtype=np.int64) for _ in range(3))
    history = None
    if return_history:
        history = {name: np.empty_like(paths) for name in (
            "target_delta", "holding", "cash_net", "cash_gross", "portfolio_net",
            "portfolio_gross", "transaction_cost", "traded_notional",
        )}
        history["executed_trade"] = np.empty(paths.shape, dtype=bool)

    for i in range(columns):
        price = paths[:, i]
        if i == n_steps:
            target = np.full(n_paths, np.nan)  # no model Delta at maturity
            new_holding = np.zeros(n_paths)
        else:
            tau = maturity * (n_steps-i) / n_steps
            target = np.asarray(bs_call_delta(price, strike, r, sigma_model, tau))
            if i == 0:
                new_holding = target
            else:
                eligible = rebalance_mask(strategy, i, target, holding,
                                           epsilon=epsilon, weekly_interval=weekly_interval)
                new_holding = np.where(eligible, target, holding)
        q = new_holding-holding
        traded = q != 0.0
        step_dt = 0.0 if i == 0 else dt
        cash_net, fee = update_cash_account(cash_net, holding, new_holding, price,
                                            r=r, dt=step_dt, kappa=kappa)
        cash_gross, _ = update_cash_account(cash_gross, holding, new_holding, price,
                                           r=r, dt=step_dt, kappa=0.0)
        trade_notional = np.abs(q)*price
        costs += fee
        cost_fv += fee*np.exp(r*(maturity-times[i]))
        notional += trade_notional
        if i == 0:
            initial += traded
        elif i == n_steps:
            liquidation += traded
        else:
            rebalances += traded
        holding = new_holding
        if history is not None:
            for name, values in (
                ("target_delta", target), ("holding", holding),
                ("cash_net", cash_net), ("cash_gross", cash_gross),
                ("portfolio_net", cash_net+holding*price),
                ("portfolio_gross", cash_gross+holding*price),
                ("transaction_cost", fee), ("traded_notional", trade_notional),
                ("executed_trade", traded),
            ):
                history[name][:, i] = values
    payoff = np.asarray(call_payoff(paths[:, -1], strike))
    return HedgingResult(
        strategy=strategy, stock_paths=paths, times=times, premium=premium,
        terminal_payoff=payoff, terminal_portfolio_net=cash_net,
        terminal_portfolio_gross=cash_gross, pnl_net=cash_net-payoff,
        pnl_gross=cash_gross-payoff, total_transaction_cost=costs,
        transaction_cost_fv=cost_fv, total_traded_notional=notional,
        n_initial=initial, n_rebalances=rebalances, n_liquidation=liquidation,
        n_trades=initial+rebalances+liquidation, history=history,
    )
