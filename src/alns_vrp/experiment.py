from __future__ import annotations

from dataclasses import asdict

from .alns import ALNSConfig, solve_alns
from .data import demo_instance


def run_demo(iterations: int = 3000, seed: int = 2026, customers: int = 36) -> dict:
    instance = demo_instance(seed=seed, n_customers=customers)
    result = solve_alns(instance, ALNSConfig(iterations=iterations), seed=seed)
    return {
        "seed": seed,
        "customers": instance.n_customers,
        "capacity": instance.capacity,
        "max_vehicles": instance.max_vehicles,
        "initial_cost": result.initial_cost,
        "best_cost": result.best_cost,
        "improvement_percent": result.improvement_percent,
        "routes": result.best_solution.routes,
        "route_loads": result.best_solution.route_loads(instance),
        "destroy_weights": result.destroy_weights,
        "repair_weights": result.repair_weights,
        "last_iteration": asdict(result.records[-1]),
    }
