# Time-Dependent Vehicle Routing with ALNS (Python)

A self-contained Python implementation of the Time-Dependent Vehicle Routing Problem (TDVRP) using Adaptive Large Neighborhood Search (ALNS).

The solver models departure-time-dependent travel speeds, vehicle capacity, customer service times, optional hard time windows, maximum route duration, and complete customer coverage.

## Repository contents

- `advanced_tdvrp.py` — ALNS solver and traffic model.
- `tests/test_tdvrp.py` — regression tests for feasibility invariants.
- `requirements.txt` — runtime dependencies.

## Key features

### Time-dependent traffic

Travel speed varies continuously through the day. Morning and evening congestion are represented with smooth Gaussian-shaped reductions, while travel time is integrated over the route so the speed can change during a trip.

### Feasibility checks

The solver checks:

- vehicle capacity,
- hard customer time windows,
- waiting for early arrivals,
- maximum route duration,
- missing customers,
- duplicate customer visits.

A candidate that drops or duplicates customers receives a large solution-level penalty. This prevents an incomplete route set from appearing artificially cheaper than a complete solution.

### ALNS operators

Implemented destroy operators:

- random removal,
- worst-cost removal,
- Shaw relatedness removal,
- time-oriented removal.

Implemented repair operators:

- greedy insertion,
- regret-2 insertion,
- regret-3 insertion.

Operator weights adapt during the search. Simulated annealing can accept selected non-improving moves to improve diversification.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Example

```python
from advanced_tdvrp import AdvancedTDVRP

solver = AdvancedTDVRP(
    num_customers=30,
    num_vehicles=4,
    vehicle_capacity=120,
    start_time=6.0,
    time_window_width=3.0,
    use_time_windows=True,
    seed=42,
)

solution = solver.alns_solve(
    max_iterations=3000,
    time_limit=120.0,
    segment_length=100,
)

solver.print_solution(solution)
solver.visualize_solution(save_path="advanced_tdvrp_solution.png")
```

Useful diagnostics:

```python
solution["all_feasible"]
solution["customers_served"]
solution["missing_customers"]
solution["duplicate_customers"]
solution["total_cost"]
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The regression tests cover early-arrival waiting semantics, missing-customer penalties, and customer-coverage behavior in a short ALNS run.

A local smoke test with 15 customers, 3 vehicles, capacity 120, time windows enabled, and seed 42 served all 15 customers with no missing or duplicate visits.

## Scope and limitations

This is a heuristic implementation; it does not provide an optimality guarantee. Solution quality and runtime depend on the generated instance, random seed, constraint tightness, and ALNS parameters.

The examples use synthetic customer coordinates, Euclidean distances, and a synthetic traffic profile. Operational deployment would require validated road-network travel times, real customer data, and application-specific constraints.

## References

- Malandraki, C. & Daskin, M. S. (1992), time-dependent vehicle routing and fleet scheduling.
- Ichoua, S., Gendreau, M. & Potvin, J.-Y. (2003), time-dependent vehicle dispatching.
- Pisinger, D. & Ropke, S. (2007), Adaptive Large Neighborhood Search for vehicle routing problems.

## License

No license file is included yet. Add a license before distributing or reusing the project under explicit open-source terms.
