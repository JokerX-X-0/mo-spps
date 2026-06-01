from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .agent import Agent


class SharedPool:
    """Shared component pool that regulates component reuse in the active population.

    Section 5, Section 8, Section 11, Section 25.2.

    Phase 1: soft-pressure mode only (Section 8.2-8.3, simplified with U=0, kappa=0).
    Pool state is derived from current population occupancy (stateless).
    """

    def __init__(
        self,
        capacities: dict[int, int],
        base_weights: dict[int, float] | None = None,
        mode: str = "soft_pressure",
        epsilon: float = 0.01,
        tau: float = 1.0,
    ):
        self.capacities = capacities
        self.base_weights = base_weights or {
            j: 1.0 for j in capacities
        }
        self.mode = mode
        self.epsilon = epsilon
        self.tau = tau

        # Mutable state used only in hard-cap mode (deferred to Phase 4)
        self._remaining: dict[int, int] = dict(capacities)

    @property
    def num_components(self) -> int:
        return len(self.capacities)

    def compute_occupancy(
        self, population: list[Agent]
    ) -> dict[int, int]:
        """Compute u_j: how many agents in the population include component j.

        Section 11.3: u_j^t = sum_i I(j in S_i^t)
        """
        occupancy: dict[int, int] = {j: 0 for j in self.capacities}
        for agent in population:
            for j in agent.solution:
                if j in occupancy:
                    occupancy[j] += 1
        return occupancy

    def compute_available(
        self, population: list[Agent]
    ) -> dict[int, float]:
        """Compute soft-pressure availability: q_tilde_j = max(Q_j - u_j, 0).

        Section 11.3.
        """
        occupancy = self.compute_occupancy(population)
        return {
            j: max(self.capacities[j] - occupancy[j], 0)
            for j in self.capacities
        }

    def sample(
        self,
        preference: np.ndarray,
        population: list[Agent],
        size: int,
        rng: np.random.Generator,
    ) -> list[int]:
        """Sample components without replacement using soft-pressure probabilities.

        Simplified formula (Section 8.3, Phase 1: U=0, kappa=0):

            p_j ∝ (epsilon + q_tilde_j / Q_j)^tau * rho_j * pi_{i,j}

        Sampling is done without replacement. After each draw, the selected
        component's probability is set to 0 for the next draw.

        Args:
            preference: Agent's component preference vector pi_i, shape (M,).
            population: Current active population for occupancy stats.
            size: Number of components to sample (shop_size).
            rng: Random number generator.

        Returns:
            List of sampled component indices.
        """
        if self.mode == "soft_pressure":
            return self._sample_soft_pressure(preference, population, size, rng)
        elif self.mode == "hard_cap":
            return self._sample_hard_cap(preference, size, rng)
        else:
            raise ValueError(f"Unknown pool mode: {self.mode}")

    def _sample_soft_pressure(
        self,
        preference: np.ndarray,
        population: list[Agent],
        size: int,
        rng: np.random.Generator,
    ) -> list[int]:
        available = self.compute_available(population)
        M = self.num_components
        components = list(self.capacities.keys())

        result: list[int] = []
        probs = np.zeros(M)

        for _ in range(size):
            for idx, j in enumerate(components):
                if j in result:
                    probs[idx] = 0.0
                    continue
                q_tilde = available[j]
                Q_j = self.capacities[j]
                rho_j = self.base_weights.get(j, 1.0)
                pi_ij = preference[idx]
                probs[idx] = (
                    (self.epsilon + q_tilde / max(Q_j, 1)) ** self.tau
                    * rho_j
                    * pi_ij
                )

            total = probs.sum()
            if total <= 0:
                # Fallback: uniform over unselected components
                candidates = [
                    c for c in components if c not in result
                ]
                if not candidates:
                    break
                chosen = int(rng.choice(candidates))
            else:
                probs /= total
                chosen_idx = int(rng.choice(M, p=probs))
                chosen = components[chosen_idx]

            result.append(chosen)

        return result

    def _sample_hard_cap(
        self,
        preference: np.ndarray,
        size: int,
        rng: np.random.Generator,
    ) -> list[int]:
        """Hard-cap sampling. Section 8.1.

        p_j ∝ q_j * rho_j * pi_{i,j}
        If q_j == 0, probability is 0.
        """
        M = self.num_components
        components = list(self.capacities.keys())

        result: list[int] = []
        probs = np.zeros(M)

        for _ in range(size):
            for idx, j in enumerate(components):
                if j in result or self._remaining.get(j, 0) <= 0:
                    probs[idx] = 0.0
                    continue
                rho_j = self.base_weights.get(j, 1.0)
                pi_ij = preference[idx]
                probs[idx] = self._remaining[j] * rho_j * pi_ij

            total = probs.sum()
            if total <= 0:
                candidates = [
                    c for c in components
                    if c not in result and self._remaining.get(c, 0) > 0
                ]
                if not candidates:
                    break
                chosen = int(rng.choice(candidates))
            else:
                probs /= total
                chosen_idx = int(rng.choice(M, p=probs))
                chosen = components[chosen_idx]

            result.append(chosen)
            if chosen in self._remaining:
                self._remaining[chosen] -= 1

        return result

    def update_transition(
        self,
        old_solution: set[int],
        new_solution: set[int],
    ) -> None:
        """Update pool state when an agent changes its solution.

        Section 11.2-11.3.
        Soft-pressure: no-op (occupancy derived from population).
        Hard-cap: adjust remaining capacities.
        """
        if self.mode == "hard_cap":
            added = new_solution - old_solution
            removed = old_solution - new_solution
            for j in added:
                if j in self._remaining:
                    self._remaining[j] -= 1
            for j in removed:
                if j in self._remaining:
                    self._remaining[j] += 1

    def validate(self, population: list[Agent]) -> None:
        """Assert pool invariants. Section 11.4.

        Raises AssertionError if any invariant is violated.
        """
        # Agent solutions contain no duplicate components (enforced by set)
        for agent in population:
            assert isinstance(
                agent.solution, set
            ), "Agent solution must be a set"

        # Occupancy consistency
        occupancy = self.compute_occupancy(population)
        if self.mode == "hard_cap":
            for j in self.capacities:
                assert (
                    0 <= self._remaining[j] <= self.capacities[j]
                ), f"Hard-cap remaining capacity out of bounds for component {j}"
                assert (
                    self._remaining[j] + occupancy[j] == self.capacities[j]
                ), f"Hard-cap invariant violated for component {j}"

        # Soft-pressure: occupancy should match population
        if self.mode == "soft_pressure":
            for j in self.capacities:
                expected = sum(1 for a in population if j in a.solution)
                assert (
                    occupancy[j] == expected
                ), f"Occupancy mismatch for component {j}"

    def reset_hard_cap(self) -> None:
        """Reset hard-cap remaining capacities to full."""
        self._remaining = dict(self.capacities)
