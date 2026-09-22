"""Command-line entry point used by Colab and local smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from .training import OUBoundaryFactorConfig, run_ou_boundary_factor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a restartable MET-PINN experiment.")
    parser.add_argument("config", type=Path, help="Path to an experiment JSON configuration")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory for all run artifacts")
    parser.add_argument("--resume", action="store_true", help="Resume from <run-dir>/last.pt")
    args = parser.parse_args()
    config = OUBoundaryFactorConfig.from_file(args.config)
    run_ou_boundary_factor(config, args.run_dir, resume=args.resume)


if __name__ == "__main__":
    main()
