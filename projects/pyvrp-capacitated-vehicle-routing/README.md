# PyVRP Capacitated Vehicle Routing

A compact, reproducible Capacitated Vehicle Routing Problem (CVRP) example using [PyVRP](https://pyvrp.org/), a high-performance open-source vehicle-routing solver.

## Problem

A single depot must serve a set of delivery customers. Each customer has a positive demand, vehicles have identical capacity, and every customer must be visited exactly once. The objective is to minimise total routing distance while respecting vehicle capacity.

The included demonstration instance has six customers, two vehicles with capacity 15 each, and total demand 27. Distances are integer Manhattan distances so the example stays self-contained and deterministic.

## Why PyVRP

PyVRP provides a high-level `Model` interface for depots, clients, vehicle types, and routing edges while using a specialised metaheuristic solver underneath. This repository deliberately keeps the domain model separate from the solver call so validation and modelling logic can be tested independently.

## Install

```bash
python -m pip install -e '.[dev]'
```

The project pins `pyvrp==0.14.0` so CI exercises a known API version.

## Run

```bash
pyvrp-cvrp
```

The command prints feasibility, objective value, total distance, and number of routes used.

## Test

```bash
pytest
```

The test suite covers instance validation, capacity accounting, model construction, distance calculation, and an end-to-end PyVRP solve. Coverage must remain at or above 90%.

GitHub Actions runs the suite on Python 3.11, 3.12, 3.13, and 3.14.

## Structure

```text
src/pyvrp_cvrp/
├── __init__.py
├── cli.py
└── model.py

tests/
├── test_model.py
└── test_solver.py
```

## Modelling notes

PyVRP works with integer routing costs. Coordinates may be floating point, but distance and duration inputs should be scaled to appropriate integers before being supplied to the solver. This example therefore converts Manhattan distance to an integer explicitly.
