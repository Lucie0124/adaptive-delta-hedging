import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import norm

from src.hedging import hedge_paths, update_cash_account, rebalance_mask
from src.monte_carlo import simulate_gbm_paths


def test_manual_cash_ledger_from_foundations():
    cash, fee0 = update_cash_account(8, 0, 0.5, 100)
    assert cash == pytest.approx(-42.1)
    assert fee0 == pytest.approx(0.1)
    before_trade = cash+0.5*110
    cash, fee1 = update_cash_account(cash, 0.5, 0.75, 110, dt=0.5)
    assert cash == pytest.approx(-69.655)
    assert cash+0.75*110 == pytest.approx(before_trade-fee1)
    cash, fee2 = update_cash_account(cash, 0.75, 0, 105, dt=0.5)
    assert cash == pytest.approx(8.9375)
    assert cash-5 == pytest.approx(3.9375)
    assert fee0+fee1+fee2 == pytest.approx(0.3125)


@pytest.mark.parametrize("cash", [-50.0, 20.0])
def test_no_trade_only_accrues_cash(cash):
    new_cash, fee = update_cash_account(cash, 0.5, 0.5, 150, r=0.07, dt=0.25)
    assert new_cash == pytest.approx(cash*np.exp(0.07*0.25))
    assert fee == 0


@pytest.mark.parametrize("old,new", [(0.3, 0.7), (0.7, 0.3), (0.7, 0.7), (0.7, 0.0)])
def test_execution_conserves_value_except_fee(old, new):
    cash, fee = update_cash_account(-20, old, new, 107, r=0.03, dt=0.2)
    assert cash+new*107 == pytest.approx(-20*np.exp(0.03*0.2)+old*107-fee)


def test_engine_deterministic_two_interval_ledger():
    paths = np.array([[100.0, 110.0, 105.0]])
    result = hedge_paths(paths, "daily", kappa=0.002)
    h = result["history"]
    # Independently construct the two model Deltas and BS premium.
    phi0 = norm.cdf(0.1)
    phi1 = norm.cdf((np.log(1.1)+0.02*0.5)/(0.2*np.sqrt(0.5)))
    premium = 100*(2*norm.cdf(0.1)-1)
    fee0, fee1, fee2 = 0.002*np.array([phi0*100, (phi1-phi0)*110, phi1*105])
    b0 = premium-phi0*100-fee0
    b1 = b0-(phi1-phi0)*110-fee1
    b2 = b1+phi1*105-fee2
    assert_allclose(h["cash_net"], [[b0, b1, b2]], atol=1e-12)
    assert_allclose(h["holding"], [[phi0, phi1, 0]], atol=1e-13)
    assert_allclose(h["transaction_cost"], [[fee0, fee1, fee2]], atol=1e-13)
    assert_allclose(result["pnl_net"], b2-5, atol=1e-12)
    assert_allclose(result["terminal_payoff"], 5)
    assert_array_equal(result["n_trades"], 3)


@pytest.mark.parametrize("strategy", ["daily", "weekly", "adaptive"])
@pytest.mark.parametrize("r", [0.0, 0.06, -0.02])
def test_full_accounting_identities_costs_and_settlement(strategy, r):
    paths = np.array([[100, 105, 98, 108, 103, 115, 112],
                      [100, 90, 95, 88, 91, 85, 80]], dtype=float)
    result = hedge_paths(paths, strategy, r=r, weekly_interval=2)
    h = result["history"]
    phi, cash = h["holding"], h["cash_net"]
    q = np.diff(phi, axis=1, prepend=np.zeros((2, 1)))
    notional = np.abs(q)*paths
    fees = 0.002*notional
    assert_allclose(h["traded_notional"], notional)
    assert_allclose(h["transaction_cost"], fees)
    assert_allclose(cash[:, 0]+phi[:, 0]*paths[:, 0], result["premium"]-fees[:, 0])
    before = cash[:, :-1]*np.exp(r/6)+phi[:, :-1]*paths[:, 1:]
    assert_allclose(h["portfolio_net"][:, 1:], before-fees[:, 1:], atol=1e-12)
    # This gain equation catches adding stock gains into cash twice.
    changes = (cash[:, :-1]*np.expm1(r/6)
               + phi[:, :-1]*np.diff(paths, axis=1)-fees[:, 1:])
    assert_allclose(np.diff(h["portfolio_net"], axis=1), changes, atol=1e-12)
    assert_allclose(result["total_transaction_cost"], fees.sum(axis=1))
    assert_allclose(result["transaction_cost_fv"],
                    (fees*np.exp(r*(1-result["times"]))).sum(axis=1))
    assert_allclose(result["pnl_gross"]-result["pnl_net"],
                    result["transaction_cost_fv"], atol=1e-12)
    assert_array_equal(phi[:, -1], 0)
    assert np.all(np.isnan(h["target_delta"][:, -1]))
    assert_allclose(result["terminal_payoff"], [12, 0])
    assert_allclose(result["pnl_net"], cash[:, -1]-[12, 0])
    assert_array_equal(result["n_trades"], h["executed_trade"].sum(axis=1))
    assert_array_equal(result["n_rebalances"], h["executed_trade"][:, 1:-1].sum(axis=1))
    # Fee-free shadow equals a separate zero-cost execution with same decisions.
    free = hedge_paths(paths, strategy, r=r, kappa=0, weekly_interval=2)
    assert_allclose(free["pnl_net"], result["pnl_gross"], atol=1e-12)
    assert_allclose(free["pnl_net"], free["pnl_gross"], atol=1e-12)
    assert_array_equal(free["total_transaction_cost"], 0)
    assert_array_equal(free["history"]["holding"], phi)


def test_weekly_no_trade_intervals_and_dates():
    paths = np.array([[100, 101, 98, 102, 105, 107, 106, 108]], dtype=float)
    result = hedge_paths(paths, "weekly", r=0.04, weekly_interval=5)
    h = result["history"]
    assert_array_equal(h["executed_trade"], [[True, False, False, False, False, True, False, True]])
    for i in (1, 2, 3, 4, 6):
        assert h["cash_net"][0, i] == pytest.approx(h["cash_net"][0, i-1]*np.exp(0.04/7))
    assert_array_equal(result["n_initial"], 1)
    assert_array_equal(result["n_rebalances"], 1)
    assert_array_equal(result["n_liquidation"], 1)


def test_adaptive_strict_boundary_and_last_actual_holding():
    # Binary-exact values distinguish strict > from >= at equality.
    assert_array_equal(rebalance_mask("adaptive", 1, [0.625, 0.626, 0.374], 0.5,
                                      epsilon=0.125), [False, True, True])
    paths = np.array([[100, 101, 102, 103, 105, 108, 112]], dtype=float)
    result = hedge_paths(paths, "adaptive", epsilon=0.1)
    h = result["history"]
    gap = np.abs(h["target_delta"][:, 1:-1]-h["holding"][:, :-2])
    assert_array_equal(h["executed_trade"][:, 1:-1], gap > 0.1)
    assert np.any(~h["executed_trade"][:, 1:-1])
    assert np.any(h["executed_trade"][:, 1:-1])


def test_no_maturity_delta_call_or_round_trip(monkeypatch):
    import src.hedging as module
    original = module.bs_call_delta
    evaluated_times = []

    def checked_delta(spot, strike, r, sigma, tau):
        assert tau > 0
        evaluated_times.append(tau)
        return original(spot, strike, r, sigma, tau)

    monkeypatch.setattr(module, "bs_call_delta", checked_delta)
    result = hedge_paths([[100, 140]], "daily")
    assert evaluated_times == [1.0]
    assert_array_equal(result["n_rebalances"], 0)
    phi0 = norm.cdf(0.1)
    assert result["history"]["transaction_cost"][0, -1] == pytest.approx(0.002*phi0*140)
    assert_array_equal(result["n_trades"], 2)


def test_no_lookahead_common_paths_readonly_inputs_and_history_modes():
    paths = simulate_gbm_paths(n_paths=10, n_steps=12)
    paths.setflags(write=False)
    results = [hedge_paths(paths, s) for s in ("daily", "weekly", "adaptive")]
    for result in results:
        assert result["stock_paths"] is paths
        assert_array_equal(result["stock_paths"], paths)
        compact = hedge_paths(paths, result["strategy"], return_history=False)
        assert compact["history"] is None
        assert_array_equal(compact["pnl_net"], result["pnl_net"])
        changed = paths.copy()
        changed[:, 8:] *= 1.3
        alternative = hedge_paths(changed, result["strategy"])
        for name in ("holding", "target_delta", "cash_net"):
            assert_array_equal(alternative["history"][name][:, :8], result["history"][name][:, :8])


def test_exactly_zero_trade_not_counted_and_zero_position_not_closed():
    result = hedge_paths([[100, 100, 100]], "daily", sigma_model=0)
    assert_array_equal(result["n_rebalances"], 0)  # ATM convention stays 0.5
    result = hedge_paths([[50, 50, 50]], "daily", sigma_model=0)
    assert_array_equal(result["n_trades"], 0)
    assert_array_equal(result["total_traded_notional"], 0)


def test_identical_repeat_execution():
    paths = simulate_gbm_paths(n_paths=12, n_steps=30, seed=91)
    first, second = (hedge_paths(paths, "adaptive") for _ in range(2))
    for name in ("pnl_net", "transaction_cost_fv", "total_traded_notional", "n_trades"):
        assert_array_equal(first[name], second[name])


@pytest.mark.parametrize("kwargs", [
    {"strategy": "unknown"}, {"maturity": 0}, {"strike": -1}, {"r": np.nan},
    {"sigma_model": -0.1}, {"kappa": -1}, {"epsilon": -1},
    {"weekly_interval": 0}, {"weekly_interval": 2.5},
])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        hedge_paths([[100, 110]], **({"strategy": "daily"} | kwargs))


@pytest.mark.parametrize("paths", [[], [100, 110], [[100]], [[100, -1]], [[100, np.nan]]])
def test_invalid_paths(paths):
    with pytest.raises(ValueError):
        hedge_paths(paths, "daily")
