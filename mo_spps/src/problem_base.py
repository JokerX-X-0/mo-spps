from abc import ABC, abstractmethod
import numpy as np


class MultiObjectiveProblem(ABC):
    """Abstract base class for multi-objective combinatorial optimization problems.

    Section 3, Section 25.2.
    All objectives are maximized. Minimization objectives should be negated.
    """

    def __init__(
        self,
        num_components: int,
        solution_capacity: int,
        num_objectives: int,
    ):
        self.num_components = num_components
        self.solution_capacity = solution_capacity
        self.num_objectives = num_objectives
        self.component_ids = list(range(num_components))
        self.ideal_point: np.ndarray | None = None
        self.nadir_point: np.ndarray | None = None
        self._all_objectives: list[np.ndarray] = []

    @abstractmethod
    def evaluate(self, solution: set[int]) -> np.ndarray:
        """Return objective vector F(S). All objectives are maximized.

        Args:
            solution: Set of component indices representing candidate S.

        Returns:
            np.ndarray of shape (num_objectives,) with maximized values.
        """
        ...

    def repair(self, solution: set[int]) -> set[int]:
        """Repair an infeasible solution. Default: no-op.

        Override in subclasses if problem-specific constraints need enforcement.
        """
        return solution

    def update_ideal_nadir(
        self, all_objectives: list[np.ndarray]
    ) -> None:
        """Update ideal and nadir point estimates from a collection of objective vectors."""
        if not all_objectives:
            return
        stacked = np.stack(all_objectives)
        self.ideal_point = np.max(stacked, axis=0)
        self.nadir_point = np.min(stacked, axis=0)
        self._all_objectives = list(all_objectives)

    def normalize_objectives(self, objectives: np.ndarray) -> np.ndarray:
        """Normalize objective vector using current ideal/nadir estimates.

        f_hat_k = (f_k - f_k^min) / (f_k^max - f_k^min + epsilon)
        Section 10.1.
        """
        if self.ideal_point is None or self.nadir_point is None:
            return objectives.copy()
        denom = self.ideal_point - self.nadir_point + 1e-12
        return (objectives - self.nadir_point) / denom
