"""Small end-to-end run of the same independent-selection and stress workflow."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]


def test_calibration_freeze_evaluation_and_stress_cli(tmp_path):
    config = json.loads((ROOT/"configs/study.json").read_text())
    config.update(n_calibration=256, n_evaluation=256, n_steps=12, n_bootstrap=20,
                  epsilon_grid=[0.01, 0.03], cost_grid=[0, 0.002])
    path = tmp_path/"config.json"
    path.write_text(json.dumps(config))
    selection_output, stress_output, figures = tmp_path/"c", tmp_path/"d", tmp_path/"figures"
    run = subprocess.run([sys.executable, str(ROOT/"experiments/03_cost_risk_optimisation.py"),
                          "--config", str(path), "--output-dir", str(selection_output), "--figures-dir", str(figures)],
                         cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stdout+run.stderr
    frozen = selection_output/"frozen_selection.json"
    frozen_bytes = frozen.read_bytes()
    run = subprocess.run([sys.executable, str(ROOT/"experiments/04_volatility_stress_test.py"),
                          "--selection-file", str(frozen), "--output-dir", str(stress_output),
                          "--figures-dir", str(figures)], cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stdout+run.stderr
    assert frozen.read_bytes() == frozen_bytes
    assert len(list(figures.glob("*.png"))) == 5
    metadata = json.loads((selection_output/"metadata.json").read_text())
    assert metadata["calibration_paths_sha256"] != metadata["evaluation_paths_sha256"]
    assert metadata["selection_frozen_before_evaluation"]
    stress = pd.read_csv(stress_output/"stress_summary.csv")
    baseline = pd.read_csv(selection_output/"evaluation_summary.csv")
    baseline = baseline[baseline["kappa"] == 0.002]
    assert_allclose(stress.loc[stress["sigma_true"] == 0.2, "es"], baseline["es"], atol=1e-12)
    diagnostic = pd.read_csv(stress_output/"fair_premium_diagnostic.csv")
    assert_allclose(diagnostic["fair_premium_es"],
                    diagnostic["fixed_premium_es"]-diagnostic["terminal_premium_shift"], atol=1e-12)
    assert np.all(stress["sigma_model"] == 0.2)
    metadata = json.loads((stress_output/"metadata.json").read_text())
    assert not metadata["reoptimised_under_stress"]
