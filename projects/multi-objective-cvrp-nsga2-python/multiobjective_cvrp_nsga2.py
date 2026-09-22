from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Customer:
    id: int
    x: float
    y: float
    demand: int
    priority: int
    service_time: float


@dataclass(frozen=True)
class Vehicle:
    id: int
    capacity: int
    max_duration: float = 480.0


@dataclass
class Individual:
    permutation: Tuple[int, ...]
    route_count: int
    routes: List[List[int]] = field(default_factory=list)
    objectives: Dict[str, float] = field(default_factory=dict)
    rank: int = 0
    crowding_distance: float = 0.0

    def dominates(self, other: "Individual") -> bool:
        keys = self.objectives.keys()
        return (
            all(self.objectives[k] <= other.objectives[k] + 1e-12 for k in keys)
            and any(self.objectives[k] < other.objectives[k] - 1e-12 for k in keys)
        )


class MultiObjectiveCVRPNSGA2:
    """
    Educational NSGA-II solver for a symmetric multi-objective CVRP.

    Objectives (all minimized):
      1. total_distance
      2. max_route_duration
      3. vehicles_used
      4. priority_weighted_arrival

    The fourth objective is deliberately order-sensitive. Unlike summing customer
    priorities, it is not constant when every customer must be served.

    Representation:
      - each chromosome is a customer permutation
      - route_count is a separate gene
      - a deterministic split decoder partitions that permutation into exactly
        route_count capacity-feasible routes

    This is a heuristic. The returned front is an approximation to the true
    Pareto front, not a proof of global Pareto optimality.
    """

    def __init__(
        self,
        customers: Sequence[Customer],
        vehicles: Sequence[Vehicle],
        depot: Customer,
        *,
        seed: int = 42,
        population_size: int = 100,
        generations: int = 200,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.2,
    ):
        if not customers:
            raise ValueError("at least one customer is required")
        if not vehicles:
            raise ValueError("at least one vehicle is required")
        if population_size < 4:
            raise ValueError("population_size must be at least 4")
        if generations <= 0:
            raise ValueError("generations must be positive")

        ids = [c.id for c in customers]
        if sorted(ids) != list(range(1, len(customers) + 1)):
            raise ValueError("customer ids must be consecutive 1..n")
        if depot.id != 0:
            raise ValueError("depot id must be 0")
        if any(c.demand <= 0 for c in customers):
            raise ValueError("customer demands must be positive")
        if any(c.priority <= 0 for c in customers):
            raise ValueError("priorities must be positive")
        if any(c.service_time < 0 for c in customers):
            raise ValueError("service times must be nonnegative")
        if any(v.capacity <= 0 for v in vehicles):
            raise ValueError("vehicle capacities must be positive")

        capacities = {v.capacity for v in vehicles}
        if len(capacities) != 1:
            raise ValueError("this educational implementation assumes identical capacities")

        self.customers = list(customers)
        self.vehicles = list(vehicles)
        self.depot = depot
        self.n = len(customers)
        self.capacity = vehicles[0].capacity
        self.max_vehicles = len(vehicles)
        self.rng = random.Random(seed)

        if any(c.demand > self.capacity for c in customers):
            raise ValueError("a customer demand exceeds vehicle capacity")
        if sum(c.demand for c in customers) > self.capacity * self.max_vehicles:
            raise ValueError("total demand exceeds available fleet capacity")

        self.population_size = int(population_size)
        self.generations = int(generations)
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)

        coords = [(depot.x, depot.y)] + [(c.x, c.y) for c in customers]
        arr = np.asarray(coords, dtype=float)
        self.distance_matrix = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=2)

        self.pareto_front: List[Individual] = []
        self.generation_stats: List[dict] = []

    def _decode(self, permutation: Tuple[int, ...], route_count: int) -> List[List[int]]:
        """
        Split a permutation into exactly `route_count` capacity-feasible routes,
        minimizing total distance for that fixed permutation and route count.
        """
        if sorted(permutation) != list(range(1, self.n + 1)):
            raise ValueError("permutation must contain every customer exactly once")
        if not (1 <= route_count <= self.max_vehicles):
            raise ValueError("route_count outside available fleet")

        dp = [[math.inf] * (self.n + 1) for _ in range(route_count + 1)]
        pred = [[-1] * (self.n + 1) for _ in range(route_count + 1)]
        dp[0][0] = 0.0

        for r in range(1, route_count + 1):
            for i in range(self.n):
                if not math.isfinite(dp[r - 1][i]):
                    continue

                load = 0
                route_distance = 0.0
                prev = 0
                for j in range(i, self.n):
                    customer = permutation[j]
                    load += self.customers[customer - 1].demand
                    if load > self.capacity:
                        break

                    route_distance += self.distance_matrix[prev, customer]
                    prev = customer
                    closed = route_distance + self.distance_matrix[customer, 0]
                    candidate = dp[r - 1][i] + closed

                    if candidate < dp[r][j + 1] - 1e-12:
                        dp[r][j + 1] = candidate
                        pred[r][j + 1] = i

        if not math.isfinite(dp[route_count][self.n]):
            raise ValueError("permutation cannot be split into requested route count")

        segments = []
        r = route_count
        end = self.n
        while r > 0:
            start = pred[r][end]
            if start < 0:
                raise RuntimeError("invalid split predecessor")
            segments.append(permutation[start:end])
            end = start
            r -= 1
        segments.reverse()

        return [[0, *segment, 0] for segment in segments]

    def _evaluate(self, permutation: Tuple[int, ...], route_count: int) -> Individual:
        routes = self._decode(permutation, route_count)

        total_distance = 0.0
        max_route_duration = 0.0
        weighted_arrival = 0.0

        for route in routes:
            route_distance = 0.0
            elapsed = 0.0

            for a, b in zip(route, route[1:]):
                travel = float(self.distance_matrix[a, b])
                route_distance += travel
                elapsed += travel

                if b != 0:
                    customer = self.customers[b - 1]
                    weighted_arrival += customer.priority * elapsed
                    elapsed += customer.service_time

            total_distance += route_distance
            max_route_duration = max(max_route_duration, elapsed)

        objectives = {
            "total_distance": total_distance,
            "max_route_duration": max_route_duration,
            "vehicles_used": float(len(routes)),
            "priority_weighted_arrival": weighted_arrival,
        }
        return Individual(
            permutation=permutation,
            route_count=route_count,
            routes=routes,
            objectives=objectives,
        )

    def validate(self, individual: Individual) -> bool:
        seen = []
        if len(individual.routes) > self.max_vehicles:
            return False
        for route in individual.routes:
            if route[0] != 0 or route[-1] != 0:
                return False
            load = sum(self.customers[c - 1].demand for c in route[1:-1])
            if load > self.capacity:
                return False
            seen.extend(route[1:-1])
        return sorted(seen) == list(range(1, self.n + 1))

    def _random_individual(self) -> Individual:
        permutation = list(range(1, self.n + 1))
        self.rng.shuffle(permutation)
        permutation = tuple(permutation)
        route_counts = list(range(1, self.max_vehicles + 1))
        self.rng.shuffle(route_counts)
        for route_count in route_counts:
            try:
                return self._evaluate(permutation, route_count)
            except ValueError:
                continue
        raise RuntimeError("no feasible route count for generated permutation")

    def _order_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[int, ...]:
        p1 = parent1.permutation
        p2 = parent2.permutation
        if self.rng.random() >= self.crossover_rate:
            return p1

        a, b = sorted(self.rng.sample(range(self.n), 2))
        child = [None] * self.n
        child[a:b + 1] = p1[a:b + 1]

        remaining = [x for x in p2 if x not in child]
        cursor = 0
        for i in range(self.n):
            if child[i] is None:
                child[i] = remaining[cursor]
                cursor += 1
        return tuple(child)

    def _mutate(self, permutation: Tuple[int, ...]) -> Tuple[int, ...]:
        if self.rng.random() >= self.mutation_rate:
            return permutation

        seq = list(permutation)
        kind = self.rng.choice(("swap", "reverse", "insert"))

        if kind == "swap":
            i, j = self.rng.sample(range(self.n), 2)
            seq[i], seq[j] = seq[j], seq[i]
        elif kind == "reverse":
            i, j = sorted(self.rng.sample(range(self.n), 2))
            seq[i:j + 1] = reversed(seq[i:j + 1])
        else:
            i, j = self.rng.sample(range(self.n), 2)
            value = seq.pop(i)
            seq.insert(j, value)

        return tuple(seq)

    @staticmethod
    def fast_non_dominated_sort(population: List[Individual]) -> List[List[Individual]]:
        dominates_map = {id(p): [] for p in population}
        domination_count = {id(p): 0 for p in population}
        fronts: List[List[Individual]] = [[]]

        for p in population:
            for q in population:
                if p is q:
                    continue
                if p.dominates(q):
                    dominates_map[id(p)].append(q)
                elif q.dominates(p):
                    domination_count[id(p)] += 1

            if domination_count[id(p)] == 0:
                p.rank = 0
                fronts[0].append(p)

        i = 0
        while i < len(fronts) and fronts[i]:
            next_front: List[Individual] = []
            for p in fronts[i]:
                for q in dominates_map[id(p)]:
                    domination_count[id(q)] -= 1
                    if domination_count[id(q)] == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)

        return fronts

    @staticmethod
    def calculate_crowding_distance(front: List[Individual]) -> None:
        if not front:
            return
        for individual in front:
            individual.crowding_distance = 0.0
        if len(front) <= 2:
            for individual in front:
                individual.crowding_distance = math.inf
            return

        for objective in front[0].objectives:
            ordered = sorted(front, key=lambda x: x.objectives[objective])
            ordered[0].crowding_distance = math.inf
            ordered[-1].crowding_distance = math.inf

            low = ordered[0].objectives[objective]
            high = ordered[-1].objectives[objective]
            if math.isclose(low, high):
                continue

            for i in range(1, len(ordered) - 1):
                if math.isinf(ordered[i].crowding_distance):
                    continue
                ordered[i].crowding_distance += (
                    ordered[i + 1].objectives[objective]
                    - ordered[i - 1].objectives[objective]
                ) / (high - low)

    def _tournament(self, population: List[Individual]) -> Individual:
        a, b = self.rng.sample(population, 2)
        if a.rank != b.rank:
            return a if a.rank < b.rank else b
        if a.crowding_distance != b.crowding_distance:
            return a if a.crowding_distance > b.crowding_distance else b
        return a if self.rng.random() < 0.5 else b

    def solve(self, time_limit: float | None = None) -> List[Individual]:
        start = time.time()
        population = [self._random_individual() for _ in range(self.population_size)]

        fronts = self.fast_non_dominated_sort(population)
        for front in fronts:
            self.calculate_crowding_distance(front)

        self.generation_stats = []

        for generation in range(self.generations):
            if time_limit is not None and time.time() - start >= time_limit:
                break

            offspring = []
            while len(offspring) < self.population_size:
                p1 = self._tournament(population)
                p2 = self._tournament(population)
                child_perm = self._order_crossover(p1, p2)
                child_perm = self._mutate(child_perm)

                child_routes = self.rng.choice((p1.route_count, p2.route_count))
                if self.rng.random() < self.mutation_rate:
                    child_routes += self.rng.choice((-1, 1))
                child_routes = max(1, min(self.max_vehicles, child_routes))

                candidates = sorted(
                    range(1, self.max_vehicles + 1),
                    key=lambda k: (abs(k - child_routes), k),
                )
                child = None
                for k in candidates:
                    try:
                        child = self._evaluate(child_perm, k)
                        break
                    except ValueError:
                        continue
                if child is None or not self.validate(child):
                    raise RuntimeError("genetic operator produced infeasible individual")
                offspring.append(child)

            combined = population + offspring
            fronts = self.fast_non_dominated_sort(combined)
            for front in fronts:
                self.calculate_crowding_distance(front)

            next_population: List[Individual] = []
            for front in fronts:
                if len(next_population) + len(front) <= self.population_size:
                    next_population.extend(front)
                else:
                    ordered = sorted(front, key=lambda x: x.crowding_distance, reverse=True)
                    next_population.extend(
                        ordered[: self.population_size - len(next_population)]
                    )
                    break

            population = next_population

            final_front = self.fast_non_dominated_sort(population)[0]
            self.generation_stats.append(
                {
                    "generation": generation,
                    "pareto_size": len(final_front),
                    "best_distance": min(x.objectives["total_distance"] for x in final_front),
                    "best_max_duration": min(
                        x.objectives["max_route_duration"] for x in final_front
                    ),
                    "min_vehicles": min(x.objectives["vehicles_used"] for x in final_front),
                    "best_priority_arrival": min(
                        x.objectives["priority_weighted_arrival"] for x in final_front
                    ),
                }
            )

        front = self.fast_non_dominated_sort(population)[0]

        unique = {}
        for ind in front:
            key = (
                tuple(round(ind.objectives[k], 10) for k in ind.objectives),
                ind.permutation,
            )
            unique[key] = ind

        self.pareto_front = list(unique.values())
        return self.pareto_front


def create_multiobjective_instance(n_customers: int, seed: int = 42):
    rng = np.random.default_rng(seed)

    depot = Customer(
        id=0,
        x=50.0,
        y=50.0,
        demand=0,
        priority=0,
        service_time=0.0,
    )

    customers = []
    for i in range(1, n_customers + 1):
        customers.append(
            Customer(
                id=i,
                x=float(rng.uniform(0, 100)),
                y=float(rng.uniform(0, 100)),
                demand=int(rng.integers(5, 20)),
                priority=int(rng.choice([1, 2, 3, 4], p=[0.3, 0.3, 0.3, 0.1])),
                service_time=float(rng.uniform(5, 20)),
            )
        )

    vehicles = [Vehicle(id=i, capacity=100, max_duration=480.0) for i in range(5)]
    return depot, customers, vehicles


if __name__ == "__main__":
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

    print(f"Approximate Pareto front size: {len(front)}")
    for i, solution in enumerate(front[:10], 1):
        print(i, solution.objectives)
