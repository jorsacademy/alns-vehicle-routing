# Traveling Salesman Optimization with Pyomo

A compact exact Traveling Salesman Problem (TSP) implementation built with Pyomo and the HiGHS MILP solver.

The project accepts either geographic coordinates or a custom pairwise cost matrix. For coordinate input it creates a Haversine great-circle distance matrix, formulates the TSP with binary arc variables, removes subtours with Miller-Tucker-Zemlin (MTZ) constraints, and returns an ordered tour.

## Features

- exact MILP optimization with Pyomo
- open-source HiGHS solver through `highspy`
- MTZ subtour-elimination constraints
- latitude/longitude input or custom distance matrices
- scikit-learn-style `fit()` API
- immutable `TSPSolution` result object
- dataframe `prescribe()` helper for visit order
- pytest coverage and GitHub Actions CI

## Installation

```bash
python -m pip install -e ".[dev]"
```

## Quick start

```python
import pandas as pd
from tsp_optimizer import TravelingSalesmanOptimizer

locations = pd.DataFrame(
    {
        "latitude": [41.0082, 41.0256, 41.0054, 41.0369],
        "longitude": [28.9784, 28.9741, 28.9768, 28.9957],
    },
    index=["Sultanahmet", "Taksim", "Grand Bazaar", "Dolmabahce"],
)

optimizer = TravelingSalesmanOptimizer().fit(
    locations,
    start="Sultanahmet",
)

print(optimizer.closed_tour_)
print(optimizer.total_distance_km_)
```

You can also solve directly from a custom matrix:

```python
optimizer.fit_distance_matrix(distance_matrix, start="A")
```

To append the optimal visit order to the original dataframe:

```python
ranked = optimizer.prescribe(locations, start="Sultanahmet")
```

## Model

For every directed arc `(i, j)`, the binary variable `x[i, j]` equals one when the tour travels from node `i` to node `j`.

The objective minimizes total travel cost. Every node has exactly one incoming and one outgoing selected arc. MTZ order variables on non-start nodes prevent disconnected subtours.

## Tests

```bash
pytest
```

GitHub Actions runs the test suite and example on Python 3.11 and 3.12.

## Notes

Haversine distances are straight-line great-circle distances. They are useful for optimization examples but are not road-network travel distances. For real routing, pass a road/travel-time matrix from a routing engine to `fit_distance_matrix()`.
