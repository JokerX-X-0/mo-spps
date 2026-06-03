"""Tests for MOKP problem instance."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.problems.mo_kp import MOKP, generate_mokp_instance


class TestMOKP:
    def test_evaluate_returns_correct_shape(self):
        problem = MOKP(num_items=50, seed=42)
        obj = problem.evaluate({0, 1, 2, 5, 10})
        assert obj.shape == (2,)
        assert np.all(obj >= 0)

    def test_empty_solution(self):
        problem = MOKP(num_items=50, seed=42)
        obj = problem.evaluate(set())
        assert obj.shape == (2,)
        assert obj[0] == 0.0
        assert obj[1] == 0.0

    def test_weight_constraint_repair(self):
        problem = MOKP(num_items=20, capacity_ratio=0.1, seed=42)
        # Select all items — should exceed capacity
        all_items = set(range(20))
        repaired = problem.repair(all_items)
        total_weight = sum(problem.weights[j] for j in repaired)
        assert total_weight <= problem.capacity
        assert len(repaired) < 20

    def test_feasible_solution_unchanged(self):
        problem = MOKP(num_items=100, capacity_ratio=0.5, seed=42)
        # Pick just a few items — well within capacity
        small = {0, 1, 2}
        repaired = problem.repair(small)
        assert repaired == small

    def test_deterministic_seed(self):
        p1 = MOKP(num_items=50, seed=123)
        p2 = MOKP(num_items=50, seed=123)
        assert np.array_equal(p1.weights, p2.weights)
        assert np.array_equal(p1.profits, p2.profits)
        assert p1.capacity == p2.capacity

    def test_factory_function(self):
        problem = generate_mokp_instance(
            num_items=100, capacity_ratio=0.5, seed=999
        )
        assert problem.num_items == 100
        assert problem.num_components == 100
        assert problem.solution_capacity == 100
        assert problem.num_objectives == 2
        assert problem.capacity > 0
        assert len(problem.weights) == 100
        assert problem.profits.shape == (100, 2)

    def test_3_objective_instance(self):
        problem = MOKP(num_items=50, num_objectives=3, seed=42)
        obj = problem.evaluate({0, 1, 2})
        assert obj.shape == (3,)
        assert problem.num_objectives == 3

    def test_all_items_removed_when_capacity_tiny(self):
        problem = MOKP(num_items=20, capacity_ratio=0.001, seed=42)
        all_items = set(range(20))
        repaired = problem.repair(all_items)
        assert repaired == set()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
