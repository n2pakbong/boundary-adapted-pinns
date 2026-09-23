"""Command-line runner for the maintained PDE examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .problems.committor.training import PeriodicCommittorConfig, run_periodic_committor
from .problems.mean_escape_time.training import OUBoundaryFactorConfig, run_ou_boundary_factor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a boundary-adapted PINN experiment.")
    parser.add_argument("config", type=Path, help="Path to a problem-specific JSON configuration")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory for run artifacts")
    parser.add_argument("--resume", action="store_true", help="Resume from <run-dir>/last.pt")
    args = parser.parse_args()
    raw_config = json.loads(args.config.read_text(encoding="utf-8"))
    if "method" in raw_config:
        run_periodic_committor(PeriodicCommittorConfig(**raw_config), args.run_dir, resume=args.resume)
    else:
        run_ou_boundary_factor(OUBoundaryFactorConfig(**raw_config), args.run_dir, resume=args.resume)


if __name__ == "__main__":
    main()
