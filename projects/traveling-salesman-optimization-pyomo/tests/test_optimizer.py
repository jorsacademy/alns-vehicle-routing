import numpy as np
import pandas as pd
import pytest

from tsp_optimizer import TravelingSalesmanOptimizer, haversine_distance_matrix


def test_haversine_matrix_is_symmetric_and_zero_diagonal():
    locations = pd.DataFrame(
        {"latitude": [41.0082, 41.0256], "longitude": [28.9784, 28.9741]},
        index=["Sultanahmet", "Taksim"],
    )
    distances = haversine_distance_matrix(locations)
    assert distances.shape == (2, 2)
    assert np.allclose(np.diag(distances), 0.0)
    assert distances.at["Sultanahmet", "Taksim"] == pytest.approx(
        distances.at["Taksim", "Sultanahmet"]
    )
    assert distances.at["Sultanahmet", "Taksim"] > 0


def test_exact_solver_finds_square_tour():
    nodes = ["A", "B", "C", "D"]
    distances = pd.DataFrame(
        [
            [0, 1, 10, 1],
            [1, 0, 1, 10],
            [10, 1, 0, 1],
            [1, 10, 1, 0],
        ],
        index=nodes,
        columns=nodes,
        dtype=float,
    )

    optimizer = TravelingSalesmanOptimizer().fit_distance_matrix(distances, start="A")

    assert optimizer.is_fitted
    assert optimizer.closed_tour_[0] == "A"
    assert optimizer.closed_tour_[-1] == "A"
    assert set(optimizer.tour_) == set(nodes)
    assert len(optimizer.selected_arcs()) == 4
    assert optimizer.total_distance_km_ == pytest.approx(4.0)


def test_prescribe_returns_visit_order():
    locations = pd.DataFrame(
        {
            "latitude": [41.0082, 41.0256, 41.0054, 41.0369],
            "longitude": [28.9784, 28.9741, 28.9768, 28.9957],
        },
        index=["Sultanahmet", "Taksim", "Grand Bazaar", "Dolmabahce"],
    )
    result = TravelingSalesmanOptimizer().prescribe(locations, start="Sultanahmet")
    assert result.index[0] == "Sultanahmet"
    assert result["visit_order"].tolist() == [0, 1, 2, 3]


def test_single_node_is_handled_without_solver():
    distances = pd.DataFrame([[0.0]], index=["A"], columns=["A"])
    optimizer = TravelingSalesmanOptimizer().fit_distance_matrix(distances)
    assert optimizer.tour_ == ("A",)
    assert optimizer.closed_tour_ == ("A", "A")
    assert optimizer.total_distance_km_ == 0.0


def test_invalid_coordinate_columns_raise():
    locations = pd.DataFrame({"lat": [41.0], "lon": [29.0]}, index=["A"])
    with pytest.raises(ValueError, match="Missing columns"):
        TravelingSalesmanOptimizer().fit(locations)


def test_negative_distance_rejected():
    distances = pd.DataFrame(
        [[0.0, -1.0], [1.0, 0.0]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    with pytest.raises(ValueError, match="negative"):
        TravelingSalesmanOptimizer().fit_distance_matrix(distances)
