"""MOKP: Multi-Objective Knapsack Problem.

Standard combinatorial multi-objective benchmark (Zitzler & Thiele 1999).

Decision space: subsets S ⊆ {0, ..., n-1} of items.
Constraint: total weight Σ_{i∈S} w_i ≤ W.
Objectives (maximized): F_k(S) = Σ_{i∈S} p_{i,k}  for k = 1, ..., m.
"""

import numpy as np
from ..problem_base import MultiObjectiveProblem


class MOKP(MultiObjectiveProblem):
    """Multi-Objective Knapsack Problem.

    Each item i has weight w_i and profit vector (p_{i,1}, ..., p_{i,m}).
    Select items to maximize total profit per objective without exceeding
    capacity W.
    """

    def __init__(
        self,
        num_items: int = 100,
        capacity: float | None = None,
        weights: np.ndarray | None = None,
        profits: np.ndarray | None = None,
        num_objectives: int = 2,
        capacity_ratio: float = 0.5,
        seed: int = 0,
    ):
        """Initialize MOKP instance.

        Args:
            num_items: Number of items (n).
            capacity: Knapsack capacity W. If None, computed as ratio of total weight.
            weights: Item weights, shape (n,). Generated if None.
            profits: Item profits per objective, shape (n, num_objectives). Generated if None.
            num_objectives: Number of objectives.
            capacity_ratio: W / Σ w_i, used when capacity is None.
            seed: Random seed for generating weights and profits.
        """
        super().__init__(
            num_components=num_items,
            solution_capacity=num_items,
            num_objectives=num_objectives,
        )
        self.num_items = num_items
        self.capacity_ratio = capacity_ratio

        rng = np.random.default_rng(seed)

        if weights is None:
            self.weights = rng.integers(1, 101, size=num_items).astype(float)
        else:
            self.weights = np.asarray(weights, dtype=float)

        if profits is None:
            self.profits = rng.integers(1, 101, size=(num_items, num_objectives)).astype(float)
        else:
            self.profits = np.asarray(profits, dtype=float)

        total_weight = float(np.sum(self.weights))
        if capacity is not None:
            self.capacity = float(capacity)
        else:
            self.capacity = capacity_ratio * total_weight

        # Precompute profit/weight ratio for repair (scalarized across objectives)
        self._pw_ratio = np.zeros(num_items)
        for i in range(num_items):
            self._pw_ratio[i] = float(np.mean(self.profits[i])) / max(self.weights[i], 1e-12)

    def evaluate(self, solution: set[int]) -> np.ndarray:
        """Evaluate a candidate solution.

        Returns:
            np.ndarray of shape (num_objectives,) with total profit per objective.
        """
        if not solution:
            return np.zeros(self.num_objectives)
        indices = list(solution)
        return np.sum(self.profits[indices], axis=0)

    def repair(self, solution: set[int]) -> set[int]:
        """Repair infeasible solutions by greedy removal.

        Removes items with lowest profit/weight ratio until total weight ≤ capacity.
        """
        if not solution:
            return solution

        current_weight = sum(self.weights[j] for j in solution)
        if current_weight <= self.capacity:
            return solution

        # Sort items in solution by ascending profit/weight ratio
        solution_list = list(solution)
        solution_list.sort(key=lambda j: self._pw_ratio[j])

        repaired = set(solution_list)
        for j in solution_list:
            if current_weight <= self.capacity:
                break
            repaired.remove(j)
            current_weight -= self.weights[j]

        return repaired

    @property
    def total_weight(self) -> float:
        return float(np.sum(self.weights))


def generate_mokp_instance(
    num_items: int = 100,
    capacity_ratio: float = 0.5,
    num_objectives: int = 2,
    seed: int = 0,
) -> MOKP:
    """Factory function for creating MOKP instances (Zitzler-Thiele scheme).

    Args:
        num_items: Number of items.
        capacity_ratio: W / Σ w_i, typically 0.5.
        num_objectives: Number of objectives (2 for standard bi-objective MOKP).
        seed: Random seed for reproducibility.

    Returns:
        MOKP instance with randomly generated weights and profits.
    """
    return MOKP(
        num_items=num_items,
        capacity_ratio=capacity_ratio,
        num_objectives=num_objectives,
        seed=seed,
    )
