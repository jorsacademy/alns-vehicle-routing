import random
import unittest

import numpy as np

from advanced_tdvrp import AdvancedTDVRP, TimeWindow


class TimeWindowTests(unittest.TestCase):
    def test_early_arrival_is_feasible(self):
        tw = TimeWindow(9.0, 11.0)
        self.assertTrue(tw.is_feasible(8.0))
        self.assertEqual(tw.wait_time(8.0), 1.0)
        self.assertFalse(tw.is_feasible(11.01))


class SolverInvariantTests(unittest.TestCase):
    def test_missing_customer_is_penalized(self):
        random.seed(7)
        np.random.seed(7)
        solver = AdvancedTDVRP(
            num_customers=6,
            num_vehicles=2,
            vehicle_capacity=100,
            use_time_windows=False,
            seed=7,
        )
        complete = solver.parallel_insertion_construction()
        complete_cost = solver.evaluate_solution(complete)

        incomplete = [type(v)(id=v.id, capacity=v.capacity) for v in complete]
        for src, dst in zip(complete, incomplete):
            dst.route = list(src.route)
            dst.current_load = src.current_load

        removed = None
        for vehicle in incomplete:
            customers = [c for c in vehicle.route if c != 0]
            if customers:
                removed = customers[0]
                vehicle.route.remove(removed)
                vehicle.current_load -= solver.customers[removed].demand
                break

        self.assertIsNotNone(removed)
        self.assertGreater(solver.evaluate_solution(incomplete), complete_cost + 5000)

    def test_smoke_solution_keeps_customer_coverage(self):
        solver = AdvancedTDVRP(
            num_customers=10,
            num_vehicles=3,
            vehicle_capacity=120,
            time_window_width=6.0,
            use_time_windows=True,
            seed=42,
        )
        solution = solver.alns_solve(max_iterations=100, time_limit=5.0)
        self.assertEqual(solution["customers_served"], 10)
        self.assertEqual(solution["missing_customers"], [])
        self.assertEqual(solution["duplicate_customers"], [])


if __name__ == "__main__":
    unittest.main()
