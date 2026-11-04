import numpy as np

from src.hedging import hedge_paths
from src.monte_carlo import simulate_gbm_paths


def test_frictionless_replication_improves_statistically_on_finer_grid():
    # Coupling keeps terminal prices identical. Test an overall coarse/fine
    # contrast with uncertainty; no monotonicity assertion for every sample.
    paths = simulate_gbm_paths(mu=0, n_steps=252, n_paths=10000, seed=2718)
    coarse = hedge_paths(paths[:, ::21], "daily", kappa=0, return_history=False)
    fine = hedge_paths(paths, "daily", kappa=0, return_history=False)
    improvement = coarse["pnl_gross"]**2-fine["pnl_gross"]**2
    se = improvement.std(ddof=1)/np.sqrt(improvement.size)
    assert improvement.mean() > 3*se
