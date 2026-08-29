import json
import subprocess
import sys

from alns_vrp.data import CVRPInstance
from alns_vrp.local_search import two_opt
from alns_vrp.solution import Solution


def test_two_opt_does_not_worsen_route():
    instance = CVRPInstance(
        coordinates=((0.0, 0.0), (0.0, 2.0), (2.0, 0.0), (0.0, 1.0), (1.0, 0.0)),
        demands=(0, 1, 1, 1, 1),
        capacity=4,
        max_vehicles=1,
    )
    solution = Solution([[1, 2, 3, 4]])
    improved = two_opt(solution, instance)
    assert improved.total_distance(instance) <= solution.total_distance(instance) + 1e-12
    assert improved.is_feasible(instance)


def test_cli_outputs_valid_json():
    proc = subprocess.run(
        [sys.executable, "-m", "alns_vrp", "--iterations", "20", "--customers", "10", "--seed", "3"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["customers"] == 10
    assert payload["best_cost"] <= payload["initial_cost"] + 1e-9
    assert payload["route_loads"]
