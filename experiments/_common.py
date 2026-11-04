"""Small shared CLI and provenance helpers for the two experiments."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parameter_parser(description: str, output_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    for name, default in (
        ("s0", 100.0), ("strike", 100.0), ("maturity", 1.0), ("r", 0.0),
        ("mu", 0.05), ("sigma-true", 0.20), ("sigma-model", 0.20),
        ("kappa", 0.002), ("epsilon", 0.01),
    ):
        parser.add_argument(f"--{name}", type=float, default=default)
    for name, default in (("n-steps", 252), ("n-paths", 2000), ("seed", 42),
                          ("weekly-interval", 5)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / output_name)
    return parser


def hedge_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {name: getattr(args, name) for name in
            ("strike", "maturity", "r", "sigma_model", "epsilon", "weekly_interval")}


def write_metadata(args: argparse.Namespace, extras: dict[str, Any]) -> None:
    """Save full configuration, actual command, versions and source hashes."""
    sources = sorted((ROOT / "src").glob("*.py")) + sorted((ROOT / "experiments").glob("*.py"))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=False)
    metadata = {
        "scope": "Synthetic European-call hedging benchmark",
        "configuration": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
        "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
        "python": platform.python_version(), "platform": platform.platform(),
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("numpy", "scipy", "pandas", "matplotlib", "pytest")},
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
        "settlement": "cash-settled call; stock liquidated once at maturity, closing spread paid",
        "counts": "executed nonzero trades; opening, interior and closing reported separately",
        "gross": "fee-free shadow using identical stock trades; gross-net = terminal-valued fees",
        **extras,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
