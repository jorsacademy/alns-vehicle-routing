# Multi-Objective CVRP with NSGA-II in Python

Educational NSGA-II implementation for a multi-objective Capacitated Vehicle Routing Problem (CVRP).

## Objectives

All objectives are minimized:

1. `total_distance`
2. `max_route_duration`
3. `vehicles_used`
4. `priority_weighted_arrival`

The fourth objective is intentionally order-sensitive: high-priority customers are penalized more when they are served later. This avoids the common modeling mistake of summing customer priorities, which is constant when every customer is always served.

## Representation and feasibility

Each chromosome contains:

- a permutation of all customers;
- a target route count.

A deterministic dynamic-programming split decoder partitions the permutation into exactly that many capacity-feasible routes. This guarantees:

- every customer appears exactly once;
- no route exceeds vehicle capacity;
- the route count never exceeds the available fleet.

## NSGA-II

The implementation includes:

- order crossover;
- swap, reverse, and insertion mutation;
- fast non-dominated sorting;
- crowding-distance diversity preservation;
- binary tournament selection;
- seeded reproducibility.

This is a meta-heuristic. The returned set is an **approximate nondominated Pareto front**, not a proof of global Pareto optimality.

## Validation

The test suite includes:

- feasibility checks;
- pairwise nondominance checks;
- reproducibility;
- a check that the priority objective is not constant;
- a check that fleet-size trade-offs are actually present;
- exhaustive Pareto-oracle validation on a 6-customer instance.

For the 6-customer oracle instance, all feasible permutations and route counts are enumerated. The seeded NSGA-II run is required to recover exactly the same unique Pareto objective vectors.

Run:

```bash
python -m unittest discover -s tests -v
```

## Example

```python
from multiobjective_cvrp_nsga2 import (
    MultiObjectiveCVRPNSGA2,
    create_multiobjective_instance,
)

depot, customers, vehicles = create_multiobjective_instance(20, seed=42)

solver = MultiObjectiveCVRPNSGA2(
    customers,
    vehicles,
    depot,
    seed=42,
    population_size=100,
    generations=200,
)

front = solver.solve()

for solution in front:
    print(solution.objectives)
```

## Reference behavior

On the included 20-customer seed-42 instance, the approximate front contains feasible trade-offs using multiple fleet sizes. Results are deterministic for a fixed seed and parameter set, but they should not be described as globally Pareto-optimal without exhaustive or exact verification.

## Requirements

- Python 3.10+
- NumPy

## Limitations

This is an educational solver for small and medium synthetic CVRP experiments. It does not include time windows, heterogeneous capacities, pickup-and-delivery constraints, emissions models, external traffic data, advanced hybrid local search, or an exact multi-objective certificate.
