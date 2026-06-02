from dataclasses import dataclass, field
import numpy as np


@dataclass
class Agent:
    """Agent state at iteration t. Section 7, Section 25.2.

    A_i^t = {S_i^t, F_i^t, G_i^t, pi_i^t, w_i, r_i^t, c_i^t, d_i^t, h_i^t}
    """

    solution: set[int]
    objectives: np.ndarray
    component_preference: np.ndarray
    objective_preference: np.ndarray
    budget: float = 0.0
    pareto_rank: int = 1
    crowding_distance: float = 0.0
    decision_diversity: float = 0.0
    no_archive_contribution_steps: int = 0
    rank_score: float = 0.0
    crowding_score: float = 0.0
    id: int = field(default_factory=lambda: _next_id())

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Agent):
            return False
        return self.id == other.id


_id_counter: int = 0


def _next_id() -> int:
    global _id_counter
    _id_counter += 1
    return _id_counter


def reset_id_counter(start: int = 0) -> None:
    global _id_counter
    _id_counter = start


class AgentFactory:
    """Factory methods for creating Agent instances."""

    @staticmethod
    def create_agent(
        solution: set[int],
        objectives: np.ndarray,
        comp_pref: np.ndarray,
        obj_pref: np.ndarray,
        budget: float = 0.0,
    ) -> Agent:
        return Agent(
            solution=solution,
            objectives=objectives,
            component_preference=comp_pref,
            objective_preference=obj_pref,
            budget=budget,
        )

    @staticmethod
    def create_with_random_preferences(
        num_components: int,
        num_objectives: int,
        solution: set[int],
        objectives: np.ndarray,
        rng: np.random.Generator,
    ) -> Agent:
        comp_pref = rng.dirichlet(np.ones(num_components))
        obj_pref = rng.dirichlet(np.ones(num_objectives))
        return Agent(
            solution=solution,
            objectives=objectives,
            component_preference=comp_pref,
            objective_preference=obj_pref,
        )
