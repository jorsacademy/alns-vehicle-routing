"""Time-Dependent Vehicle Routing Problem solved with ALNS.

The implementation is intentionally self-contained and focuses on correctness
invariants: every customer must be served exactly once, early arrivals may wait,
and infeasible routes are penalized rather than silently accepted.
"""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class TimeWindow:
    earliest: float
    latest: float

    def is_feasible(self, arrival_time: float) -> bool:
        """Early arrival is feasible because the vehicle can wait."""
        return arrival_time <= self.latest

    def wait_time(self, arrival_time: float) -> float:
        return max(0.0, self.earliest - arrival_time)


@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: int
    service_time: float = 15.0
    time_window: Optional[TimeWindow] = None
    priority: int = 1


@dataclass
class Vehicle:
    id: int
    capacity: int
    max_route_time: float = 480.0
    fixed_cost: float = 100.0
    variable_cost_per_km: float = 1.0
    variable_cost_per_min: float = 0.5
    route: List[int] = field(default_factory=lambda: [0, 0])
    current_load: int = 0


class TrafficModel:
    """Smooth daily traffic profile with morning/evening congestion."""

    def __init__(self, base_speed: float = 50.0):
        self.base_speed = base_speed
        self.morning_center = 8.0
        self.evening_center = 18.0
        self.rush_width = 1.5
        self.rush_reduction = 0.4

    def get_speed_factor(self, hour: float) -> float:
        hour %= 24.0
        morning = self.rush_reduction * math.exp(
            -((hour - self.morning_center) ** 2) / (2 * self.rush_width**2)
        )
        evening = self.rush_reduction * math.exp(
            -((hour - self.evening_center) ** 2) / (2 * self.rush_width**2)
        )
        night_bonus = 0.2 * (5.0 - hour) / 5.0 if 0 <= hour < 5 else 0.0
        return max(0.5, min(1.5, 1.0 - morning - evening + night_bonus))

    def calculate_travel_time(
        self, distance_km: float, departure_time_hours: float, step_hours: float = 0.05
    ) -> Tuple[float, float]:
        if distance_km <= 0:
            return 0.0, departure_time_hours

        remaining = distance_km
        now = departure_time_hours
        elapsed = 0.0
        while remaining > 1e-9:
            speed = self.base_speed * self.get_speed_factor(now)
            covered = min(remaining, speed * step_hours)
            dt = covered / speed
            remaining -= covered
            now += dt
            elapsed += dt
        return elapsed, now

    def get_speed_profile(self, start: float = 0.0, end: float = 24.0, points: int = 200):
        times = np.linspace(start, end, points)
        speeds = np.array([self.base_speed * self.get_speed_factor(t) for t in times])
        return times, speeds


class DestroyOperator(Enum):
    RANDOM = "random"
    WORST = "worst"
    SHAW = "shaw"
    TIME_ORIENTED = "time_oriented"


class RepairOperator(Enum):
    GREEDY = "greedy"
    REGRET_2 = "regret_2"
    REGRET_3 = "regret_3"


class OperatorManager:
    def __init__(self, reaction: float = 0.1):
        self.reaction = reaction
        self.destroy_weights = {op: 1.0 for op in DestroyOperator}
        self.repair_weights = {op: 1.0 for op in RepairOperator}
        self.destroy_scores = {op: 0.0 for op in DestroyOperator}
        self.repair_scores = {op: 0.0 for op in RepairOperator}
        self.destroy_uses = {op: 0 for op in DestroyOperator}
        self.repair_uses = {op: 0 for op in RepairOperator}

    @staticmethod
    def _pick(weights):
        ops = list(weights)
        vals = [max(1e-9, weights[o]) for o in ops]
        return random.choices(ops, weights=vals, k=1)[0]

    def select(self):
        return self._pick(self.destroy_weights), self._pick(self.repair_weights)

    def record(self, destroy, repair, result: str):
        score = {"new_best": 10.0, "better": 5.0, "accepted": 2.0, "rejected": 0.0}[result]
        self.destroy_scores[destroy] += score
        self.repair_scores[repair] += score
        self.destroy_uses[destroy] += 1
        self.repair_uses[repair] += 1

    def update(self):
        for op in DestroyOperator:
            if self.destroy_uses[op]:
                avg = self.destroy_scores[op] / self.destroy_uses[op]
                self.destroy_weights[op] = (1 - self.reaction) * self.destroy_weights[op] + self.reaction * avg
        for op in RepairOperator:
            if self.repair_uses[op]:
                avg = self.repair_scores[op] / self.repair_uses[op]
                self.repair_weights[op] = (1 - self.reaction) * self.repair_weights[op] + self.reaction * avg
        self.destroy_scores = {op: 0.0 for op in DestroyOperator}
        self.repair_scores = {op: 0.0 for op in RepairOperator}
        self.destroy_uses = {op: 0 for op in DestroyOperator}
        self.repair_uses = {op: 0 for op in RepairOperator}


class AdvancedTDVRP:
    MISSING_PENALTY = 10_000.0
    DUPLICATE_PENALTY = 10_000.0
    TW_PENALTY = 1_000.0
    CAPACITY_PENALTY = 100.0
    DURATION_PENALTY = 100.0

    def __init__(
        self,
        num_customers: int = 50,
        num_vehicles: int = 5,
        vehicle_capacity: int = 100,
        grid_size: int = 100,
        start_time: float = 6.0,
        time_window_width: float = 4.0,
        use_time_windows: bool = True,
        max_route_time: float = 480.0,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.grid_size = grid_size
        self.start_time = start_time
        self.time_window_width = time_window_width
        self.use_time_windows = use_time_windows
        self.max_route_time = max_route_time
        self.customers = self._generate_customers()
        self.distance_matrix = self._distance_matrix()
        self.traffic_model = TrafficModel()
        self.operator_manager = OperatorManager()
        self.best_solution: Optional[List[Vehicle]] = None
        self.best_cost = float("inf")
        self.best_costs: List[float] = []
        self.iteration_costs: List[float] = []

    def _generate_customers(self) -> List[Customer]:
        customers = [Customer(0, self.grid_size / 2, self.grid_size / 2, 0, 0.0, None)]
        for i in range(1, self.num_customers + 1):
            if self.use_time_windows:
                center = random.uniform(self.start_time + 1, self.start_time + 10)
                tw = TimeWindow(center - self.time_window_width / 2, center + self.time_window_width / 2)
            else:
                tw = None
            customers.append(
                Customer(
                    i,
                    random.uniform(10, self.grid_size - 10),
                    random.uniform(10, self.grid_size - 10),
                    random.randint(5, 30),
                    random.uniform(10, 20),
                    tw,
                    random.randint(1, 3),
                )
            )
        return customers

    def _distance_matrix(self) -> np.ndarray:
        n = len(self.customers)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = math.hypot(self.customers[i].x - self.customers[j].x, self.customers[i].y - self.customers[j].y)
                matrix[i, j] = matrix[j, i] = d
        return matrix

    def evaluate_route(self, route: List[int], departure_time: Optional[float] = None) -> Dict:
        departure = self.start_time if departure_time is None else departure_time
        now = departure
        distance = 0.0
        wait_hours = 0.0
        load = 0
        tw_violations = 0

        for a, b in zip(route, route[1:]):
            leg = self.distance_matrix[a, b]
            distance += leg
            _, arrival = self.traffic_model.calculate_travel_time(leg, now)
            if b == 0:
                now = arrival
                continue
            customer = self.customers[b]
            if self.use_time_windows and customer.time_window is not None:
                if arrival > customer.time_window.latest:
                    tw_violations += 1
                wait = customer.time_window.wait_time(arrival)
                wait_hours += wait
                now = max(arrival, customer.time_window.earliest)
            else:
                now = arrival
            now += customer.service_time / 60.0
            load += customer.demand

        total_time = (now - departure) * 60.0
        capacity_violation = max(0, load - self.vehicle_capacity)
        duration_violation = max(0.0, total_time - self.max_route_time)
        cost = distance + 0.5 * total_time
        cost += self.TW_PENALTY * tw_violations
        cost += self.CAPACITY_PENALTY * capacity_violation
        cost += self.DURATION_PENALTY * duration_violation
        return {
            "feasible": tw_violations == 0 and capacity_violation == 0 and duration_violation <= 1e-9,
            "total_distance": distance,
            "total_time": total_time,
            "total_wait_time": wait_hours * 60.0,
            "total_load": load,
            "time_window_violations": tw_violations,
            "capacity_violation": capacity_violation,
            "route_duration_violation": duration_violation,
            "cost": cost,
        }

    def _coverage(self, vehicles: List[Vehicle]) -> Dict:
        visits = [c for v in vehicles for c in v.route if c != 0]
        counts: Dict[int, int] = {}
        for c in visits:
            counts[c] = counts.get(c, 0) + 1
        expected = set(range(1, self.num_customers + 1))
        missing = sorted(expected - set(visits))
        duplicates = sorted(c for c, n in counts.items() if n > 1)
        return {"missing": missing, "duplicates": duplicates, "served": len(expected - set(missing))}

    def evaluate_solution(self, vehicles: List[Vehicle]) -> float:
        total = 0.0
        for vehicle in vehicles:
            if len(vehicle.route) > 2:
                r = self.evaluate_route(vehicle.route)
                total += vehicle.fixed_cost
                total += vehicle.variable_cost_per_km * r["total_distance"]
                total += vehicle.variable_cost_per_min * r["total_time"]
                total += self.TW_PENALTY * r["time_window_violations"]
                total += self.CAPACITY_PENALTY * r["capacity_violation"]
                total += self.DURATION_PENALTY * r["route_duration_violation"]
        coverage = self._coverage(vehicles)
        total += self.MISSING_PENALTY * len(coverage["missing"])
        total += self.DUPLICATE_PENALTY * len(coverage["duplicates"])
        return total

    def _vehicles(self) -> List[Vehicle]:
        return [Vehicle(i, self.vehicle_capacity, self.max_route_time) for i in range(self.num_vehicles)]

    def _best_insertions(self, vehicles: List[Vehicle], customer_id: int):
        options = []
        customer = self.customers[customer_id]
        for vehicle in vehicles:
            if vehicle.current_load + customer.demand > vehicle.capacity:
                continue
            old = self.evaluate_route(vehicle.route)["cost"]
            for pos in range(1, len(vehicle.route)):
                trial = vehicle.route[:pos] + [customer_id] + vehicle.route[pos:]
                result = self.evaluate_route(trial)
                if result["feasible"]:
                    options.append((result["cost"] - old, vehicle, pos))
        return sorted(options, key=lambda x: x[0])

    def parallel_insertion_construction(self) -> List[Vehicle]:
        vehicles = self._vehicles()
        unassigned = set(range(1, self.num_customers + 1))
        while unassigned:
            choice = None
            best_regret = -float("inf")
            for customer_id in unassigned:
                options = self._best_insertions(vehicles, customer_id)
                if not options:
                    continue
                regret = options[1][0] - options[0][0] if len(options) > 1 else 1e6 - options[0][0]
                if regret > best_regret:
                    best_regret = regret
                    choice = customer_id, options[0][1], options[0][2]
            if choice is None:
                break
            customer_id, vehicle, pos = choice
            vehicle.route.insert(pos, customer_id)
            vehicle.current_load += self.customers[customer_id].demand
            unassigned.remove(customer_id)
        return vehicles

    def _remove(self, vehicles: List[Vehicle], ids: Set[int]):
        result = copy.deepcopy(vehicles)
        removed = set()
        for v in result:
            kept = [0]
            for c in v.route[1:-1]:
                if c in ids:
                    removed.add(c)
                else:
                    kept.append(c)
            kept.append(0)
            v.route = kept
            v.current_load = sum(self.customers[c].demand for c in kept if c != 0)
        return result, removed

    def destroy(self, vehicles: List[Vehicle], operator: DestroyOperator, num_remove: int):
        customers = [c for v in vehicles for c in v.route if c != 0]
        if not customers:
            return copy.deepcopy(vehicles), set()
        num_remove = min(num_remove, len(customers))
        if operator == DestroyOperator.RANDOM:
            ids = set(random.sample(customers, num_remove))
        elif operator == DestroyOperator.WORST:
            scored = []
            for v in vehicles:
                for i in range(1, len(v.route) - 1):
                    a, c, b = v.route[i - 1], v.route[i], v.route[i + 1]
                    contribution = self.distance_matrix[a, c] + self.distance_matrix[c, b] - self.distance_matrix[a, b]
                    scored.append((contribution, c))
            ids = {c for _, c in sorted(scored, reverse=True)[:num_remove]}
        elif operator == DestroyOperator.TIME_ORIENTED:
            ids = set(sorted(customers, key=lambda c: (self.customers[c].time_window.latest if self.customers[c].time_window else 1e9))[:num_remove])
        else:
            seed = random.choice(customers)
            s = self.customers[seed]
            related = sorted(customers, key=lambda c: self.distance_matrix[seed, c] + 5 * abs(self.customers[c].demand - s.demand))
            ids = set(related[:num_remove])
        return self._remove(vehicles, ids)

    def repair(self, vehicles: List[Vehicle], unassigned: Set[int], operator: RepairOperator):
        result = copy.deepcopy(vehicles)
        remaining = set(unassigned)
        while remaining:
            choice = None
            score = float("inf") if operator == RepairOperator.GREEDY else -float("inf")
            for customer_id in remaining:
                options = self._best_insertions(result, customer_id)
                if not options:
                    continue
                if operator == RepairOperator.GREEDY:
                    if options[0][0] < score:
                        score = options[0][0]
                        choice = customer_id, options[0][1].id, options[0][2]
                else:
                    k = 2 if operator == RepairOperator.REGRET_2 else 3
                    available = options[:k]
                    regret = sum(o[0] for o in available[1:]) - (len(available) - 1) * available[0][0]
                    if len(available) < k:
                        regret += 1e5 * (k - len(available))
                    if regret > score:
                        score = regret
                        choice = customer_id, options[0][1].id, options[0][2]
            if choice is None:
                break
            customer_id, vehicle_id, pos = choice
            vehicle = result[vehicle_id]
            vehicle.route.insert(pos, customer_id)
            vehicle.current_load += self.customers[customer_id].demand
            remaining.remove(customer_id)
        return result

    def alns_solve(self, max_iterations: int = 3000, time_limit: float = 120.0, segment_length: int = 100) -> Dict:
        current = self.parallel_insertion_construction()
        current_cost = self.evaluate_solution(current)
        self.best_solution = copy.deepcopy(current)
        self.best_cost = current_cost
        self.best_costs = [self.best_cost]
        self.iteration_costs = [current_cost]
        temperature = max(1.0, 0.1 * current_cost)
        started = time.time()

        for iteration in range(max_iterations):
            if time.time() - started >= time_limit:
                break
            destroy_op, repair_op = self.operator_manager.select()
            served = sum(1 for v in current for c in v.route if c != 0)
            num_remove = max(1, int(0.2 * served))
            partial, removed = self.destroy(current, destroy_op, num_remove)
            candidate = self.repair(partial, removed, repair_op)
            candidate_cost = self.evaluate_solution(candidate)
            delta = candidate_cost - current_cost
            accepted = delta <= 0 or random.random() < math.exp(-delta / max(temperature, 1e-9))
            result = "rejected"
            if accepted:
                current, current_cost = candidate, candidate_cost
                result = "better" if delta < 0 else "accepted"
            if candidate_cost < self.best_cost:
                self.best_solution = copy.deepcopy(candidate)
                self.best_cost = candidate_cost
                result = "new_best"
            self.operator_manager.record(destroy_op, repair_op, result)
            if (iteration + 1) % segment_length == 0:
                self.operator_manager.update()
            temperature = max(0.01, temperature * 0.9995)
            self.best_costs.append(self.best_cost)
            self.iteration_costs.append(current_cost)

        return self._format_solution()

    def _format_solution(self) -> Dict:
        assert self.best_solution is not None
        vehicles_out = []
        total_distance = total_time = 0.0
        route_feasible = True
        for vehicle in self.best_solution:
            if len(vehicle.route) <= 2:
                continue
            r = self.evaluate_route(vehicle.route)
            route_feasible &= r["feasible"]
            total_distance += r["total_distance"]
            total_time += r["total_time"]
            vehicles_out.append({
                "vehicle_id": vehicle.id,
                "route": list(vehicle.route),
                "num_customers": len(vehicle.route) - 2,
                "load": r["total_load"],
                "capacity_utilization": r["total_load"] / vehicle.capacity,
                "distance": r["total_distance"],
                "time": r["total_time"],
                "wait_time": r["total_wait_time"],
                "tw_violations": r["time_window_violations"],
                "route_duration_violation": r["route_duration_violation"],
                "feasible": r["feasible"],
            })
        coverage = self._coverage(self.best_solution)
        return {
            "vehicles": vehicles_out,
            "num_routes": len(vehicles_out),
            "total_distance": total_distance,
            "total_time": total_time,
            "total_cost": self.best_cost,
            "avg_capacity_utilization": float(np.mean([v["capacity_utilization"] for v in vehicles_out])) if vehicles_out else 0.0,
            "customers_served": coverage["served"],
            "missing_customers": coverage["missing"],
            "duplicate_customers": coverage["duplicates"],
            "all_feasible": route_feasible and not coverage["missing"] and not coverage["duplicates"],
        }

    def print_solution(self, solution: Dict) -> None:
        for v in solution["vehicles"]:
            print(f"Vehicle {v['vehicle_id']}: {' -> '.join(map(str, v['route']))} | load={v['load']} | time={v['time']:.1f} min")
        print(f"Cost: {solution['total_cost']:.2f}")
        print(f"Customers served: {solution['customers_served']}/{self.num_customers}")
        print(f"Feasible: {solution['all_feasible']}")

    def visualize_solution(self, save_path: Optional[str] = None) -> None:
        if self.best_solution is None:
            raise RuntimeError("Call alns_solve() before visualize_solution().")
        fig, ax = plt.subplots(figsize=(10, 8))
        depot = self.customers[0]
        ax.scatter([depot.x], [depot.y], marker="s", s=180, label="Depot")
        for c in self.customers[1:]:
            ax.scatter([c.x], [c.y], s=50)
            ax.text(c.x + 1, c.y + 1, str(c.id), fontsize=8)
        for vehicle in self.best_solution:
            if len(vehicle.route) <= 2:
                continue
            xs = [self.customers[c].x for c in vehicle.route]
            ys = [self.customers[c].y for c in vehicle.route]
            ax.plot(xs, ys, marker="o", label=f"Vehicle {vehicle.id}")
        ax.set_title("Time-Dependent Vehicle Routes")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    solver = AdvancedTDVRP(num_customers=15, num_vehicles=3, vehicle_capacity=120, seed=42)
    solution = solver.alns_solve(max_iterations=1000, time_limit=30.0)
    solver.print_solution(solution)
