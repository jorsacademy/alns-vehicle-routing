from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import numpy as np
import pandas as pd
import pyomo.environ as pyo

EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True, slots=True)
class TSPSolution:
    """Result of a solved Traveling Salesman Problem."""

    tour: tuple[Hashable, ...]
    closed_tour: tuple[Hashable, ...]
    total_distance_km: float


class TravelingSalesmanOptimizer:
    """Exact TSP optimizer using a Pyomo MILP model and MTZ constraints."""

    def __init__(self, *, solver: str = "appsi_highs", tee: bool = False) -> None:
        self.solver_name = solver
        self.tee = tee
        self.model_: pyo.ConcreteModel | None = None
        self.solution_: TSPSolution | None = None
        self.distance_matrix_: pd.DataFrame | None = None
        self.solve_results_ = None

    @property
    def is_fitted(self) -> bool:
        return self.solution_ is not None

    @property
    def tour_(self) -> tuple[Hashable, ...]:
        self._require_fitted()
        assert self.solution_ is not None
        return self.solution_.tour

    @property
    def closed_tour_(self) -> tuple[Hashable, ...]:
        self._require_fitted()
        assert self.solution_ is not None
        return self.solution_.closed_tour

    @property
    def total_distance_km_(self) -> float:
        self._require_fitted()
        assert self.solution_ is not None
        return self.solution_.total_distance_km

    def fit(
        self,
        locations: pd.DataFrame,
        *,
        start: Hashable | None = None,
    ) -> "TravelingSalesmanOptimizer":
        """Solve a TSP from latitude/longitude coordinates."""
        self._validate_locations(locations)
        distances = haversine_distance_matrix(locations)
        return self.fit_distance_matrix(distances, start=start)

    def fit_distance_matrix(
        self,
        distances: pd.DataFrame,
        *,
        start: Hashable | None = None,
    ) -> "TravelingSalesmanOptimizer":
        """Solve a TSP from a square pairwise cost/distance matrix."""
        distances = self._normalize_distance_matrix(distances)
        nodes = list(distances.index)
        depot = nodes[0] if start is None else start

        if depot not in distances.index:
            raise ValueError(f"Unknown start node: {depot!r}")

        self.distance_matrix_ = distances.copy()
        self.solution_ = None
        self.model_ = None

        if len(nodes) == 1:
            self.solution_ = TSPSolution(
                tour=(depot,),
                closed_tour=(depot, depot),
                total_distance_km=0.0,
            )
            return self

        model = self._build_model(distances, depot)
        self._solve(model)
        self.model_ = model
        self.solution_ = self._extract_solution(model, depot)
        return self

    def prescribe(
        self,
        locations: pd.DataFrame,
        *,
        start: Hashable | None = None,
        sort: bool = True,
    ) -> pd.DataFrame:
        """Return a copy of ``locations`` with the optimal visit order appended."""
        self.fit(locations, start=start)
        order = {node: rank for rank, node in enumerate(self.tour_)}
        result = locations.copy()
        result["visit_order"] = result.index.map(order)
        if sort:
            result = result.sort_values("visit_order")
        return result

    def selected_arcs(self) -> list[tuple[Hashable, Hashable]]:
        """Return directed arcs selected by the optimal MILP solution."""
        self._require_fitted()
        if self.model_ is None:
            return []
        return [
            (i, j)
            for i, j in self.model_.ARCS
            if pyo.value(self.model_.x[i, j]) > 0.5
        ]

    def model_summary(self) -> dict[str, int]:
        """Return basic Pyomo model dimensions."""
        if self.model_ is None:
            return {"variables": 0, "constraints": 0, "objectives": 0}
        return {
            "variables": self.model_.nvariables(),
            "constraints": self.model_.nconstraints(),
            "objectives": self.model_.nobjectives(),
        }

    def _build_model(
        self,
        distances: pd.DataFrame,
        start: Hashable,
    ) -> pyo.ConcreteModel:
        nodes = list(distances.index)
        non_start = [node for node in nodes if node != start]
        arcs = [(i, j) for i in nodes for j in nodes if i != j]
        n = len(nodes)

        model = pyo.ConcreteModel(name="traveling_salesman_problem")
        model.NODES = pyo.Set(initialize=nodes, ordered=True)
        model.NON_START = pyo.Set(initialize=non_start, ordered=True)
        model.ARCS = pyo.Set(initialize=arcs, dimen=2)
        model.cost = pyo.Param(
            model.ARCS,
            initialize={(i, j): float(distances.at[i, j]) for i, j in arcs},
            within=pyo.NonNegativeReals,
        )
        model.x = pyo.Var(model.ARCS, domain=pyo.Binary)
        model.order = pyo.Var(
            model.NON_START,
            bounds=(1, n - 1),
            domain=pyo.NonNegativeReals,
        )

        model.objective = pyo.Objective(
            expr=sum(model.cost[i, j] * model.x[i, j] for i, j in model.ARCS),
            sense=pyo.minimize,
        )

        def one_departure_rule(m: pyo.ConcreteModel, i: Hashable):
            return sum(m.x[i, j] for j in m.NODES if j != i) == 1

        def one_arrival_rule(m: pyo.ConcreteModel, j: Hashable):
            return sum(m.x[i, j] for i in m.NODES if i != j) == 1

        def mtz_rule(m: pyo.ConcreteModel, i: Hashable, j: Hashable):
            if i == j:
                return pyo.Constraint.Skip
            return m.order[i] - m.order[j] + (n - 1) * m.x[i, j] <= n - 2

        model.one_departure = pyo.Constraint(model.NODES, rule=one_departure_rule)
        model.one_arrival = pyo.Constraint(model.NODES, rule=one_arrival_rule)
        model.subtour_elimination = pyo.Constraint(
            model.NON_START,
            model.NON_START,
            rule=mtz_rule,
        )
        return model

    def _solve(self, model: pyo.ConcreteModel) -> None:
        solver = pyo.SolverFactory(self.solver_name)
        if not solver.available(exception_flag=False):
            raise RuntimeError(
                f"Solver {self.solver_name!r} is unavailable. "
                "Install the solver dependency with `pip install highspy`."
            )

        results = solver.solve(model, tee=self.tee)
        self.solve_results_ = results
        if not pyo.check_optimal_termination(results):
            termination = getattr(results.solver, "termination_condition", "unknown")
            raise RuntimeError(f"Optimization did not terminate optimally: {termination}")

    def _extract_solution(
        self,
        model: pyo.ConcreteModel,
        start: Hashable,
    ) -> TSPSolution:
        selected = [
            (i, j)
            for i, j in model.ARCS
            if pyo.value(model.x[i, j]) > 0.5
        ]
        successor = {i: j for i, j in selected}
        route = [start]
        current = start
        expected_nodes = len(model.NODES)

        for _ in range(expected_nodes - 1):
            if current not in successor:
                raise RuntimeError("Invalid solution: route is disconnected")
            current = successor[current]
            if current == start:
                raise RuntimeError("Invalid solution: route closes before all nodes are visited")
            route.append(current)

        if successor.get(current) != start:
            raise RuntimeError("Invalid solution: route does not return to the start")

        closed_route = route + [start]
        assert self.distance_matrix_ is not None
        total_distance = sum(
            float(self.distance_matrix_.at[i, j])
            for i, j in zip(closed_route[:-1], closed_route[1:])
        )
        return TSPSolution(
            tour=tuple(route),
            closed_tour=tuple(closed_route),
            total_distance_km=float(total_distance),
        )

    @staticmethod
    def _validate_locations(locations: pd.DataFrame) -> None:
        if not isinstance(locations, pd.DataFrame):
            raise TypeError("locations must be a pandas DataFrame")
        if locations.empty:
            raise ValueError("locations cannot be empty")
        if locations.index.has_duplicates:
            raise ValueError("location index values must be unique")

        required = {"latitude", "longitude"}
        missing = required.difference(locations.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        lat = pd.to_numeric(locations["latitude"], errors="coerce")
        lon = pd.to_numeric(locations["longitude"], errors="coerce")
        if lat.isna().any() or lon.isna().any():
            raise ValueError("latitude and longitude must be numeric and non-null")
        if not lat.between(-90.0, 90.0).all():
            raise ValueError("latitude must be between -90 and 90 degrees")
        if not lon.between(-180.0, 180.0).all():
            raise ValueError("longitude must be between -180 and 180 degrees")

    @staticmethod
    def _normalize_distance_matrix(distances: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(distances, pd.DataFrame):
            raise TypeError("distances must be a pandas DataFrame")
        if distances.empty:
            raise ValueError("distance matrix cannot be empty")
        if distances.shape[0] != distances.shape[1]:
            raise ValueError("distance matrix must be square")
        if distances.index.has_duplicates or distances.columns.has_duplicates:
            raise ValueError("distance matrix labels must be unique")
        if set(distances.index) != set(distances.columns):
            raise ValueError("distance matrix rows and columns must contain the same nodes")

        normalized = distances.loc[distances.index, distances.index].astype(float)
        values = normalized.to_numpy()
        if not np.isfinite(values).all():
            raise ValueError("distance matrix must contain only finite values")
        if (values < 0).any():
            raise ValueError("distance matrix cannot contain negative values")
        return normalized

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Optimizer is not fitted. Call fit() first.")

    def __repr__(self) -> str:
        if not self.is_fitted:
            return f"TravelingSalesmanOptimizer(solver={self.solver_name!r}, fitted=False)"
        return (
            "TravelingSalesmanOptimizer("
            f"solver={self.solver_name!r}, nodes={len(self.tour_)}, "
            f"distance={self.total_distance_km_:.3f} km)"
        )


def haversine_distance_matrix(locations: pd.DataFrame) -> pd.DataFrame:
    """Return pairwise great-circle distances in kilometers."""
    TravelingSalesmanOptimizer._validate_locations(locations)
    coords = locations[["latitude", "longitude"]].astype(float)
    lat = np.radians(coords["latitude"].to_numpy())
    lon = np.radians(coords["longitude"].to_numpy())

    lat1, lat2 = lat[:, None], lat[None, :]
    dlat = lat2 - lat1
    dlon = lon[None, :] - lon[:, None]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    central_angle = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    distances = EARTH_RADIUS_KM * central_angle

    return pd.DataFrame(distances, index=locations.index.copy(), columns=locations.index.copy())
