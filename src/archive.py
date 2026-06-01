from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from . import pareto
from .operators import jaccard_similarity, jaccard_distance

if TYPE_CHECKING:
    from .agent import Agent


class ParetoArchive:
    """External Pareto archive for storing non-dominated solutions.

    Section 4, Section 14, Section 25.2.

    A^{t+1} = ND(A^t ∪ Pop^t ∪ C^t)
    """

    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self.solutions: list[set[int]] = []
        self.objectives: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.solutions)

    def update(self, candidates: list[Agent]) -> int:
        """Merge candidates into archive.

        Steps:
        1. Add all candidate solutions that are not dominated by current archive.
        2. Remove any archive solutions dominated by candidates.
        3. Remove duplicates.
        4. Prune if exceeding max_size.

        Returns: number of candidates added to the archive.
        """
        if not candidates:
            return 0

        added_count = 0
        candidate_objs: list[np.ndarray] = []
        candidate_sols: list[set[int]] = []

        for agent in candidates:
            obj = agent.objectives
            sol = agent.solution

            # Check if candidate is dominated by any archive member
            dominated_by_archive = False
            for arch_obj in self.objectives:
                if pareto.dominates(arch_obj, obj):
                    dominated_by_archive = True
                    break

            if not dominated_by_archive:
                candidate_objs.append(obj.copy())
                candidate_sols.append(sol)
                added_count += 1

        if added_count == 0:
            return 0

        # Merge
        self.solutions.extend(candidate_sols)
        self.objectives.extend(candidate_objs)

        # Remove dominated
        self._remove_dominated()

        # Remove duplicates (after merge, only need to check new ones vs all)
        self.remove_duplicates()

        # Prune
        if len(self.solutions) > self.max_size:
            self.prune_by_crowding()

        return added_count

    def _remove_dominated(self) -> None:
        """Remove any solution that is dominated by another in the archive."""
        n = len(self.solutions)
        dominated = np.zeros(n, dtype=bool)

        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if pareto.dominates(self.objectives[i], self.objectives[j]):
                    dominated[j] = True

        if dominated.any():
            self.solutions = [
                s for i, s in enumerate(self.solutions) if not dominated[i]
            ]
            self.objectives = [
                o for i, o in enumerate(self.objectives) if not dominated[i]
            ]

    def remove_duplicates(self) -> None:
        """Remove solutions with identical component sets or identical
        objective vectors (within floating-point tolerance).

        Section 14.3.
        """
        n = len(self.solutions)
        to_remove = set()

        for i in range(n):
            if i in to_remove:
                continue
            for j in range(i + 1, n):
                if j in to_remove:
                    continue
                same_components = self.solutions[i] == self.solutions[j]
                obj_close = bool(
                    np.allclose(
                        self.objectives[i], self.objectives[j], rtol=1e-10
                    )
                )
                if same_components or obj_close:
                    to_remove.add(j)

        if to_remove:
            self.solutions = [
                s for i, s in enumerate(self.solutions) if i not in to_remove
            ]
            self.objectives = [
                o for i, o in enumerate(self.objectives) if i not in to_remove
            ]

    def prune_by_crowding(self) -> None:
        """Remove solutions with lowest crowding distance until |A| <= max_size.

        Section 14.1. Phase 1: objective-space crowding distance only.
        """
        while len(self.solutions) > self.max_size:
            objs = np.stack(self.objectives)
            cd = pareto.crowding_distance(objs)
            # Remove the one with smallest crowding distance
            min_idx = int(np.argmin(cd))
            self.solutions.pop(min_idx)
            self.objectives.pop(min_idx)

    def prune_by_hybrid_score(
        self,
        omega_o: float = 0.7,
        omega_d: float = 0.3,
    ) -> None:
        """Prune using hybrid objective-decision diversity score.

        Section 14.2:
            Score(S) = omega_o * CD_obj(S) + omega_d * D_archive(S)

        Deferred to Phase 4.
        """
        while len(self.solutions) > self.max_size:
            objs = np.stack(self.objectives)
            cd_obj = pareto.normalize_crowding_score(
                pareto.crowding_distance(objs)
            )
            d_archive = self._compute_archive_diversity_scores()
            scores = omega_o * cd_obj + omega_d * d_archive
            min_idx = int(np.argmin(scores))
            self.solutions.pop(min_idx)
            self.objectives.pop(min_idx)

    def _compute_archive_diversity_scores(self) -> np.ndarray:
        """Compute decision-space diversity of each solution within the archive.

        D_archive(S) = 1 - mean_{S' != S} sim(S, S')
        Section 14.2.
        """
        n = len(self.solutions)
        scores = np.zeros(n)
        for i in range(n):
            total_sim = 0.0
            for j in range(n):
                if i != j:
                    total_sim += jaccard_similarity(
                        self.solutions[i], self.solutions[j]
                    )
            scores[i] = 1.0 - total_sim / max(n - 1, 1)
        return scores

    def select_elite_by_direction(
        self, direction: np.ndarray
    ) -> tuple[set[int], np.ndarray] | None:
        """Select archive solution with highest cosine similarity to direction.

        elite = argmax_{S in A} cos(F_hat(S), w_new)
        Section 17.4. Used in Phase 3.
        """
        if not self.solutions:
            return None

        best_idx = 0
        best_cos = -np.inf

        for i, obj in enumerate(self.objectives):
            obj_norm = np.linalg.norm(obj)
            dir_norm = np.linalg.norm(direction)
            if obj_norm < 1e-12 or dir_norm < 1e-12:
                cos_sim = 0.0
            else:
                cos_sim = float(np.dot(obj, direction) / (obj_norm * dir_norm))
            if cos_sim > best_cos:
                best_cos = cos_sim
                best_idx = i

        return self.solutions[best_idx], self.objectives[best_idx].copy()

    def to_agent_list(self) -> list[Agent]:
        """Convert archive contents to lightweight Agent objects."""
        from .agent import Agent

        agents = []
        for sol, obj in zip(self.solutions, self.objectives):
            agents.append(
                Agent(
                    solution=sol,
                    objectives=obj.copy(),
                    component_preference=np.ones(1),
                    objective_preference=np.ones(1),
                )
            )
        return agents

    def get_objectives_array(self) -> np.ndarray:
        """Return objectives as (N, m) array."""
        if not self.objectives:
            return np.empty((0, 0))
        return np.stack(self.objectives)
