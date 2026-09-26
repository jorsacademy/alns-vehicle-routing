import json
from pathlib import Path

from rlhh_cvrp.benchmark import run_benchmark


def test_smoke_benchmark_has_paired_methods_and_oracle(tmp_path: Path) -> None:
    result = run_benchmark("configs/smoke.json", tmp_path)
    assert result["schema_version"] == 1
    assert result["policy_role"] == "select_low_level_search_operator_only"
    assert set(result["operator_names"]) == {
        "two_opt",
        "swap",
        "relocate",
        "ruin_recreate",
        "random_perturbation",
    }
    methods = {row["method"] for row in result["records"]}
    assert methods == {
        "fixed_best_validation",
        "random",
        "round_robin",
        "adaptive_weights",
        "q_learning",
    }
    assert result["oracle"]
    assert all(row["certified_optimal"] for row in result["oracle"])
    assert result["paired_differences"]
    assert all(row["feasibility_rate"] == 1.0 for row in result["records"])
    assert (tmp_path / "metrics.csv").stat().st_size > 0
    assert (tmp_path / "raw_results.jsonl").stat().st_size > 0
    reloaded = json.loads((tmp_path / "summary.json").read_text())
    assert reloaded["fixed_operator_selected_on"] == "validation"
