import pandas as pd

from tsp_optimizer import TravelingSalesmanOptimizer

locations = pd.DataFrame(
    {
        "latitude": [41.0082, 41.0256, 41.0054, 41.0369, 41.0472],
        "longitude": [28.9784, 28.9741, 28.9768, 28.9957, 29.0267],
    },
    index=["Sultanahmet", "Taksim", "Grand Bazaar", "Dolmabahce", "Ortakoy"],
)

optimizer = TravelingSalesmanOptimizer().fit(locations, start="Sultanahmet")
print(optimizer)
print("Closed tour:", " -> ".join(optimizer.closed_tour_))
print(f"Total distance: {optimizer.total_distance_km_:.2f} km")
print(optimizer.prescribe(locations, start="Sultanahmet"))
