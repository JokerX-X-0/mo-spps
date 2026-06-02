"""Tests for metrics: hypervolume, entropy, Jaccard distance, concentration.

Section 34.3.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.metrics import (
    compute_hypervolume,
    compute_component_entropy,
    compute_average_jaccard_distance,
    compute_reuse_concentration,
    compute_pool_occupancy,
    MetricsTracker,
)
from src.agent import Agent


class TestHypervolume:
    def test_2d_hypervolume(self):
        objs = np.array([[4.0, 0.0], [0.0, 4.0], [2.0, 2.0]])
        ref = np.array([-1.0, -1.0])
        hv = compute_hypervolume(objs, ref)
        assert hv > 0

    def test_empty_set(self):
        objs = np.empty((0, 2))
        hv = compute_hypervolume(objs, np.array([0.0, 0.0]))
        assert hv == 0.0


class TestComponentEntropy:
    def test_uniform_distribution(self):
        occupancy = {0: 5, 1: 5, 2: 5, 3: 5}
        h, h_norm = compute_component_entropy(occupancy, 4)
        # Near maximum entropy
        assert h_norm > 0.9

    def test_concentrated_distribution(self):
        occupancy = {0: 19, 1: 1, 2: 0, 3: 0}
        h, h_norm = compute_component_entropy(occupancy, 4)
        assert h_norm < 0.5

    def test_zero_total(self):
        occupancy = {0: 0, 1: 0}
        h, h_norm = compute_component_entropy(occupancy, 2)
        assert h == 0.0


class TestJaccardDistance:
    def test_identical_sets(self):
        solutions = [{0, 1, 2}, {0, 1, 2}]
        d = compute_average_jaccard_distance(solutions)
        assert d == 0.0

    def test_disjoint_sets(self):
        solutions = [{0, 1}, {2, 3}]
        d = compute_average_jaccard_distance(solutions)
        assert d == 1.0

    def test_partial_overlap(self):
        solutions = [{0, 1, 2}, {0, 1, 3}]
        d = compute_average_jaccard_distance(solutions)
        assert 0.0 < d < 1.0

    def test_single_solution(self):
        d = compute_average_jaccard_distance([{0, 1}])
        assert d == 0.0


class TestReuseConcentration:
    def test_uniform_usage(self):
        occupancy = {0: 5, 1: 5, 2: 5, 3: 5}
        gini = compute_reuse_concentration(occupancy)
        assert gini < 0.1

    def test_concentrated_usage(self):
        occupancy = {0: 20, 1: 0, 2: 0, 3: 0}
        gini = compute_reuse_concentration(occupancy)
        assert gini > 0.5


class TestMetricsTracker:
    def test_record_and_retrieve(self):
        tracker = MetricsTracker()
        tracker.record(
            iteration=1,
            fe_count=100,
            archive_size=15,
            hypervolume=42.0,
            avg_jaccard_distance=0.5,
            component_entropy=2.0,
            component_entropy_norm=0.8,
            reuse_concentration=0.3,
        )
        history = tracker.get_history()
        assert history["iteration"] == [1]
        assert history["hypervolume"] == [42.0]
        assert len(history["pool_occupancy"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
