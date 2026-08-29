from __future__ import annotations

import argparse
import json

from .experiment import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ALNS on a synthetic capacitated VRP instance.")
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--customers", type=int, default=36)
    args = parser.parse_args()
    print(json.dumps(run_demo(args.iterations, args.seed, args.customers), indent=2))


if __name__ == "__main__":
    main()
