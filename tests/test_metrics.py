import numpy as np
import pytest

from src.hedging import hedge_paths
from src.metrics import pnl_statistics, summarize_hedging, mean_confidence_interval


def test_hand_calculated_metrics():
    stats = pnl_statistics([-2, 0, 1, 3])
    assert stats["mean_pnl"] == pytest.approx(0.5)
    assert stats["std_pnl"] == pytest.approx(np.sqrt(13/3))
    assert stats["mae"] == pytest.approx(1.5)
    assert stats["rmse"] == pytest.approx(np.sqrt(3.5))
    assert stats["se_mean_pnl"] == pytest.approx(np.sqrt(13/3)/2)
    ci = mean_confidence_interval([-2, 0, 1, 3])
    assert ci["ci_low"] == pytest.approx(0.5-1.95996398454*np.sqrt(13/3)/2)
    assert ci["ci_high"] == pytest.approx(0.5+1.95996398454*np.sqrt(13/3)/2)


def test_cost_and_trade_aggregations_with_gross_net_reconciliation():
    result = hedge_paths([[100, 110], [100, 90]], "daily", kappa=0.002)
    summary = summarize_hedging(result)
    assert summary["mean_rebalances"] == 0
    assert summary["mean_initial_trades"] == 1
    assert summary["mean_liquidation_trades"] == 1
    assert summary["mean_trades"] == 2
    assert summary["mean_pnl_gross"]-summary["mean_pnl_net"] == pytest.approx(
        summary["mean_transaction_cost_fv"])
    assert summary["mean_total_transaction_cost"] == pytest.approx(
        0.002*summary["mean_traded_notional"])


@pytest.mark.parametrize("values", [[], [1], [1, np.nan], [[1, 2]]])
def test_reject_invalid_metric_inputs(values):
    with pytest.raises(ValueError):
        pnl_statistics(values)
