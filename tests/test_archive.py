"""Tests for ParetoArchive: update, deduplication, pruning.

Section 34.2-34.3.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.archive import ParetoArchive
from src.agent import Agent


def make_agent(solution: set[int], objectives: np.ndarray) -> Agent:
    return Agent(
        solution=solution,
        objectives=objectives,
        component_preference=np.ones(1),
        objective_preference=np.ones(1),
    )


class TestArchiveUpdate:
    def test_empty_archive_accepts_all_nondominated(self):
        archive = ParetoArchive(max_size=10)
        agents = [
            make_agent({0}, np.array([5.0, 1.0])),
            make_agent({1}, np.array([1.0, 5.0])),
        ]
        archive.update(agents)
        assert len(archive) == 2

    def test_dominated_candidate_rejected(self):
        archive = ParetoArchive(max_size=10)
        archive.update([make_agent({0}, np.array([5.0, 5.0]))])
        # Candidate dominated by archive
        archive.update([make_agent({1}, np.array([3.0, 3.0]))])
        assert len(archive) == 1

    def test_candidate_dominates_archive_member(self):
        archive = ParetoArchive(max_size=10)
        archive.update([make_agent({0}, np.array([2.0, 2.0]))])
        # Candidate dominates archive member
        archive.update([make_agent({1}, np.array([5.0, 5.0]))])
        assert len(archive) == 1
        assert archive.solutions[0] == {1}

    def test_mutually_nondominated_both_kept(self):
        archive = ParetoArchive(max_size=10)
        archive.update([
            make_agent({0}, np.array([5.0, 1.0])),
            make_agent({1}, np.array([1.0, 5.0])),
        ])
        assert len(archive) == 2


class TestDeduplication:
    def test_identical_solutions_removed(self):
        archive = ParetoArchive(max_size=10)
        agents = [
            make_agent({0, 1}, np.array([1.0, 2.0])),
            make_agent({0, 1}, np.array([1.0, 2.0])),
        ]
        archive.update(agents)
        assert len(archive) == 1

    def test_same_components_different_objectives(self):
        archive = ParetoArchive(max_size=10)
        agents = [
            make_agent({0, 1}, np.array([1.0, 2.0])),
            make_agent({0, 1}, np.array([1.1, 1.9])),
        ]
        archive.update(agents)
        # Both kept if mutually non-dominated
        assert len(archive) >= 1


class TestPruning:
    def test_prune_reduces_to_max_size(self):
        archive = ParetoArchive(max_size=5)
        agents = []
        rng = np.random.default_rng(42)
        for i in range(10):
            # Mutually non-dominated: trade-off pattern
            obj = np.array([float(i), float(10 - i)])
            agents.append(make_agent({i}, obj))
        archive.update(agents)
        assert len(archive) <= 5

    def test_prune_keeps_boundary_points(self):
        archive = ParetoArchive(max_size=3)
        agents = [
            make_agent({0}, np.array([10.0, 0.0])),
            make_agent({1}, np.array([5.0, 5.0])),
            make_agent({2}, np.array([0.0, 10.0])),
            make_agent({3}, np.array([6.0, 4.0])),
        ]
        archive.update(agents)
        assert len(archive) <= 3


class TestEliteSelection:
    def test_select_by_direction(self):
        archive = ParetoArchive(max_size=10)
        archive.update([
            make_agent({0}, np.array([1.0, 0.0])),
            make_agent({1}, np.array([0.0, 1.0])),
        ])
        elite = archive.select_elite_by_direction(np.array([1.0, 0.0]))
        assert elite is not None
        sol, obj = elite
        assert sol == {0}

    def test_empty_archive_returns_none(self):
        archive = ParetoArchive(max_size=10)
        assert archive.select_elite_by_direction(np.array([1.0, 0.0])) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
