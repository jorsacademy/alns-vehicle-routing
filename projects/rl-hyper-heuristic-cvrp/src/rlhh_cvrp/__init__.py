"""RL hyper-heuristic benchmark for CVRP local-search operator selection."""

from .problem import CVRPInstance, SolutionAudit, audit_solution, generate_instance

__all__ = ["CVRPInstance", "SolutionAudit", "audit_solution", "generate_instance"]
__version__ = "0.1.0"
