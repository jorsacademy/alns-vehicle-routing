from __future__ import annotations

import argparse

from .benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CVRP RL hyper-heuristic benchmark")
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--output", default="results/smoke")
    args = parser.parse_args()
    result = run_benchmark(args.config, args.output)
    print(
        {
            "fixed_operator": result["fixed_operator"],
            "records": len(result["records"]),
            "oracle_records": len(result["oracle"]),
        }
    )


if __name__ == "__main__":
    main()
