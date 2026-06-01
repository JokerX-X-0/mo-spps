"""Tests for SharedPool: sampling, occupancy, invariants.

Section 34.1.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.pool import SharedPool
from src.agent import Agent


def make_agent(solution: set[int], objectives=None) -> Agent:
    if objectives is None:
        objectives = np.array([1.0, 1.0])
    return Agent(
        solution=solution,
        objectives=objectives,
        component_preference=np.ones(5) / 5,
        objective_preference=np.array([0.5, 0.5]),
    )


class TestSharedPoolOccupancy:
    def test_empty_population(self):
        capacities = {j: 5 for j in range(10)}
        pool = SharedPool(capacities)
        occupancy = pool.compute_occupancy([])
        assert all(v == 0 for v in occupancy.values())

    def test_occupancy_tracks_population(self):
        capacities = {j: 5 for j in range(10)}
        pool = SharedPool(capacities)
        agents = [
            make_agent({0, 1, 2}),
            make_agent({0, 3, 5}),
        ]
        occupancy = pool.compute_occupancy(agents)
        assert occupancy[0] == 2
        assert occupancy[1] == 1
        assert occupancy[2] == 1
        assert occupancy[3] == 1
        assert occupancy[4] == 0
        assert occupancy[5] == 1

    def test_available_decreases_with_occupancy(self):
        capacities = {j: 5 for j in range(10)}
        pool = SharedPool(capacities)
        agents = [make_agent({0, 0}), make_agent({0})]  # duplicates in set ignored
        # Actually, sets can't have duplicates, so just test normal behavior
        agents2 = [make_agent({0}), make_agent({0})]
        available = pool.compute_available(agents2)
        assert available[0] == 3  # Q=5, u=2
        assert available[9] == 5  # Q=5, u=0


class TestSoftPressureSampling:
    def test_samples_correct_size(self):
        capacities = {j: 10 for j in range(20)}
        pool = SharedPool(capacities, mode="soft_pressure")
        rng = np.random.default_rng(42)
        preference = np.ones(20) / 20
        result = pool.sample(preference, [], 5, rng)
        assert len(result) == 5
        # No duplicates
        assert len(set(result)) == 5
        # All valid component ids
        assert all(0 <= j < 20 for j in result)

    def test_high_occupancy_reduces_probability(self):
        """Components heavily occupied in the population are sampled less often."""
        capacities = {j: 5 for j in range(10)}
        pool = SharedPool(capacities, mode="soft_pressure", tau=2.0)
        rng = np.random.default_rng(42)

        # Make component 0 very popular
        agents = [make_agent({0}) for _ in range(5)]
        # Component 0 is at its capacity limit => q_tilde_0 = 0
        # Sampling probability for component 0 should be very low

        preference = np.ones(10) / 10
        counts = {j: 0 for j in range(10)}

        for _ in range(1000):
            result = pool.sample(preference, agents, 3, rng)
            for j in result:
                counts[j] += 1

        # Component 0 should be sampled less than average
        avg_count = sum(counts.values()) / 10
        assert counts[0] < avg_count, (
            f"Component 0 count {counts[0]} should be below average {avg_count:.1f}"
        )

    def test_samples_without_replacement(self):
        capacities = {j: 10 for j in range(30)}
        pool = SharedPool(capacities)
        rng = np.random.default_rng(42)
        preference = np.ones(30) / 30
        result = pool.sample(preference, [], 10, rng)
        assert len(result) == len(set(result))


class TestPoolValidation:
    def test_validate_soft_pressure(self):
        capacities = {j: 5 for j in range(5)}
        pool = SharedPool(capacities, mode="soft_pressure")
        agents = [make_agent({0, 1})]
        pool.validate(agents)  # Should not raise

    def test_validate_agent_solution_is_set(self):
        capacities = {j: 3 for j in range(5)}
        pool = SharedPool(capacities)
        agents = [make_agent({0, 1})]
        pool.validate(agents)  # Sets are fine


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
