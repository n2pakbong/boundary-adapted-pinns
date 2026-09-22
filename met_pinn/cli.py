"""Command-line entry point used by Colab and local smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .training import (
    OUBoundaryFactorConfig,
    PeriodicCommittorConfig,
    run_ou_boundary_factor,
    run_periodic_committor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a restartable boundary-adapted PINN experiment.")
    parser.add_argument("config", type=Path, help="Path to an experiment JSON configuration")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory for all run artifacts")
    parser.add_argument("--resume", action="store_true", help="Resume from <run-dir>/last.pt")
    args = parser.parse_args()
    raw_config = json.loads(args.config.read_text(encoding="utf-8"))
    if "method" in raw_config:
        config = PeriodicCommittorConfig(**raw_config)
        run_periodic_committor(config, args.run_dir, resume=args.resume)
    else:
        config = OUBoundaryFactorConfig(**raw_config)
        run_ou_boundary_factor(config, args.run_dir, resume=args.resume)


if __name__ == "__main__":
    main()
