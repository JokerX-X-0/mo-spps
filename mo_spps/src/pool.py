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
        kappa: float = 0.0,
    ):
        self.capacities = capacities
        self.base_weights = base_weights or {
            j: 1.0 for j in capacities
        }
        self.mode = mode
        self.epsilon = epsilon
        self.tau = tau
        self.kappa = kappa

        # Mutable state used only in hard-cap mode
        self._remaining: dict[int, int] = dict(capacities)

        # Component utility scores for utility-guided sampling (Phase 4)
        self._component_utilities: dict[int, float] = {}

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

    def compute_pressure_factors(
        self, population: list[Agent]
    ) -> dict[int, float]:
        """Compute continuous soft-pressure factors.

        φ_j = 1 / (1 + u_j / Q_j)

        Smooth decay without hard clipping at Q_j.
        """
        occupancy = self.compute_occupancy(population)
        return {
            j: 1.0 / (1.0 + occupancy[j] / max(self.capacities[j], 1))
            for j in self.capacities
        }

    def sample(
        self,
        preference: np.ndarray,
        population: list[Agent],
        size: int,
        rng: np.random.Generator,
    ) -> list[int]:
        """Sample components without replacement.

        Mode "soft_pressure": clipped formula (epsilon + max(Q_j-u_j,0)/Q_j)^tau
        Mode "continuous": smooth formula (epsilon + 1/(1+u_j/Q_j))^tau
        Both include utility boost (1 + kappa * U_j) when kappa > 0.

        Sampling without replacement — selected components are excluded from
        subsequent draws.
        """
        if self.mode == "soft_pressure":
            return self._sample_soft_pressure(preference, population, size, rng)
        elif self.mode == "continuous":
            return self._sample_continuous(preference, population, size, rng)
        elif self.mode == "hard_cap":
            return self._sample_hard_cap(preference, size, rng)
        elif self.mode == "none":
            return self._sample_none(preference, size, rng)
        else:
            raise ValueError(f"Unknown pool mode: {self.mode}")

    def _sample_none(
        self,
        preference: np.ndarray,
        size: int,
        rng: np.random.Generator,
    ) -> list[int]:
        """Uniform sampling without pool pressure.

        p_j ∝ rho_j * pi_{i,j} — no occupancy factor.
        """
        M = self.num_components
        components = list(self.capacities.keys())

        result: list[int] = []
        probs = np.zeros(M)

        for _ in range(size):
            for idx, j in enumerate(components):
                if j in result:
                    probs[idx] = 0.0
                    continue
                probs[idx] = self.base_weights.get(j, 1.0) * preference[idx]

            total = probs.sum()
            if total <= 0:
                candidates = [c for c in components if c not in result]
                if not candidates:
                    break
                chosen = int(rng.choice(candidates))
            else:
                probs /= total
                chosen_idx = int(rng.choice(M, p=probs))
                chosen = components[chosen_idx]

            result.append(chosen)

        return result

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
                U_j = self._component_utilities.get(j, 0.0)
                pressure_factor = q_tilde / max(Q_j, 1)
                utility_boost = 1.0 + self.kappa * U_j
                probs[idx] = (
                    (self.epsilon + pressure_factor) ** self.tau
                    * rho_j * pi_ij * utility_boost
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

    def _sample_continuous(
        self,
        preference: np.ndarray,
        population: list[Agent],
        size: int,
        rng: np.random.Generator,
    ) -> list[int]:
        """Continuous soft-pressure sampling.

        φ_j = (ε + 1/(1 + u_j/Q_j))^τ
        p_j ∝ φ_j · ρ_j · π_{i,j} · (1 + κ · U_j)

        No hard clipping — pressure decays smoothly as usage increases.
        """
        phi = self.compute_pressure_factors(population)
        M = self.num_components
        components = list(self.capacities.keys())

        result: list[int] = []
        probs = np.zeros(M)

        for _ in range(size):
            for idx, j in enumerate(components):
                if j in result:
                    probs[idx] = 0.0
                    continue
                rho_j = self.base_weights.get(j, 1.0)
                pi_ij = preference[idx]
                U_j = self._component_utilities.get(j, 0.0)
                utility_boost = 1.0 + self.kappa * U_j
                probs[idx] = (
                    (self.epsilon + phi[j]) ** self.tau
                    * rho_j * pi_ij * utility_boost
                )

            total = probs.sum()
            if total <= 0:
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

    def update_capacities(self, new_capacities: dict[int, int]) -> None:
        """Update dynamic reference capacities for adaptive Q (Section 3.4).

        Q_j changes affect the pressure denominator in continuous/soft_pressure
        modes. Higher Q_j means weaker penalty for the same occupancy.
        """
        self.capacities = dict(new_capacities)

    def set_component_utilities(self, utilities: dict[int, float]) -> None:
        """Update component utility scores U_j for utility-guided sampling.

        Section 8.2. U_j measures historical contribution to the archive.
        Set to empty dict or all-zeros to disable utility guidance.
        """
        self._component_utilities = dict(utilities)

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

        # Soft-pressure / none: occupancy should match population
        if self.mode in ("soft_pressure", "none"):
            for j in self.capacities:
                expected = sum(1 for a in population if j in a.solution)
                assert (
                    occupancy[j] == expected
                ), f"Occupancy mismatch for component {j}"

    def reset_hard_cap(self) -> None:
        """Reset hard-cap remaining capacities to full."""
        self._remaining = dict(self.capacities)
