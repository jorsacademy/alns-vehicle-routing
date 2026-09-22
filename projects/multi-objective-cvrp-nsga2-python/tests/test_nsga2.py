import itertools
import unittest

from multiobjective_cvrp_nsga2 import (
    MultiObjectiveCVRPNSGA2,
    create_multiobjective_instance,
)


def objective_vector(ind):
    return tuple(round(ind.objectives[k], 8) for k in ind.objectives)


class NSGA2Tests(unittest.TestCase):
    def test_reference_front_is_feasible_and_nondominated(self):
        depot, customers, vehicles = create_multiobjective_instance(20, 42)
        solver = MultiObjectiveCVRPNSGA2(
            customers, vehicles, depot,
            seed=42, population_size=80, generations=60,
        )
        front = solver.solve(time_limit=20)

        self.assertTrue(front)
        self.assertTrue(all(solver.validate(ind) for ind in front))

        for i, a in enumerate(front):
            for j, b in enumerate(front):
                if i != j:
                    self.assertFalse(b.dominates(a))

    def test_vehicle_count_tradeoff_is_real(self):
        depot, customers, vehicles = create_multiobjective_instance(20, 42)
        solver = MultiObjectiveCVRPNSGA2(
            customers, vehicles, depot,
            seed=42, population_size=100, generations=80,
        )
        front = solver.solve(time_limit=20)
        counts = {int(ind.objectives["vehicles_used"]) for ind in front}
        self.assertGreaterEqual(len(counts), 2)

    def test_priority_objective_is_not_constant(self):
        depot, customers, vehicles = create_multiobjective_instance(8, 7)
        solver = MultiObjectiveCVRPNSGA2(
            customers, vehicles[:3], depot,
            seed=7, population_size=30, generations=10,
        )

        a = solver._evaluate(tuple(range(1, 9)), 2)
        b = solver._evaluate(tuple(reversed(range(1, 9))), 2)
        self.assertNotEqual(
            round(a.objectives["priority_weighted_arrival"], 8),
            round(b.objectives["priority_weighted_arrival"], 8),
        )

    def test_exhaustive_pareto_oracle_on_six_customers(self):
        depot, customers, vehicles = create_multiobjective_instance(6, 11)
        vehicles = vehicles[:3]
        solver = MultiObjectiveCVRPNSGA2(
            customers, vehicles, depot,
            seed=11, population_size=80, generations=150,
        )

        candidates = []
        for perm in itertools.permutations(range(1, 7)):
            for k in range(1, 4):
                try:
                    candidates.append(solver._evaluate(tuple(perm), k))
                except ValueError:
                    pass

        exact_front = solver.fast_non_dominated_sort(candidates)[0]
        exact_vectors = {objective_vector(ind) for ind in exact_front}

        approx_front = solver.solve(time_limit=20)
        approx_vectors = {objective_vector(ind) for ind in approx_front}

        self.assertTrue(approx_vectors)
        self.assertTrue(approx_vectors.issubset(exact_vectors))
        self.assertEqual(approx_vectors, exact_vectors)

    def test_reproducibility(self):
        depot, customers, vehicles = create_multiobjective_instance(10, 5)
        kwargs = dict(seed=99, population_size=40, generations=30)

        a_solver = MultiObjectiveCVRPNSGA2(customers, vehicles, depot, **kwargs)
        b_solver = MultiObjectiveCVRPNSGA2(customers, vehicles, depot, **kwargs)

        a = sorted(objective_vector(x) for x in a_solver.solve())
        b = sorted(objective_vector(x) for x in b_solver.solve())
        self.assertEqual(a, b)

    def test_total_capacity_infeasibility_rejected(self):
        depot, customers, vehicles = create_multiobjective_instance(8, 3)
        tiny_vehicles = [
            type(vehicles[0])(id=i, capacity=10, max_duration=480.0)
            for i in range(2)
        ]
        with self.assertRaises(ValueError):
            MultiObjectiveCVRPNSGA2(customers, tiny_vehicles, depot)


if __name__ == "__main__":
    unittest.main()
