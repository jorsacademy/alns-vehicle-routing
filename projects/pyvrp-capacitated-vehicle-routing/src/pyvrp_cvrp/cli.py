"""Command-line entry point for the PyVRP CVRP example."""

from .model import demo_instance, solve_instance


def main() -> None:
    instance = demo_instance()
    result = solve_instance(instance)

    print("PyVRP CVRP demo")
    print(f"Customers: {len(instance.customers)}")
    print(f"Total demand: {instance.total_demand}")
    print(f"Fleet capacity: {instance.fleet_capacity}")
    print(f"Feasible: {result.is_feasible()}")
    print(f"Objective: {result.cost():.0f}")
    print(f"Distance: {result.best.distance()}")
    print(f"Routes used: {result.best.num_routes()}")


if __name__ == "__main__":
    main()
