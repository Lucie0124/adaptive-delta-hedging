"""End-to-end checks for the actual CLI, saved diagnostics and result schemas."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", ["01_model_validation.py", "02_baseline_hedging.py"])
def test_cli_writes_reproducible_inspectable_outputs(script, tmp_path):
    command = [sys.executable, str(ROOT / "experiments" / script), "--n-paths", "512",
               "--n-steps", "12", "--r", "0.03", "--output-dir", str(tmp_path)]
    if script.startswith("01"):
        command += ["--convergence-steps", "4", "16", "64"]
    run = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=60)
    assert run.returncode == 0, run.stdout+run.stderr
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["configuration"]["r"] == 0.03
    assert metadata["source_sha256"]
    checks = json.loads((tmp_path / "checks.json").read_text(encoding="utf-8"))
    assert len(list(tmp_path.glob("*.png"))) == 2
    if script.startswith("01"):
        assert checks["convergence_diagnostic_passed"]
        assert checks["pricing_error_below_five_standard_errors"]
        table = pd.read_csv(tmp_path / "pricing_validation.csv")
        assert table["n_paths"].iloc[0] == 512
        assert np.isfinite(table["standard_error"].iloc[0])
    else:
        table = pd.read_csv(tmp_path / "path_results.csv")
        assert len(table) == 6*512
        assert checks["gross_net_identity_passed"]
        assert_allclose(table["pnl_gross"]-table["pnl_net"],
                        table["transaction_cost_fv"], atol=1e-10)
